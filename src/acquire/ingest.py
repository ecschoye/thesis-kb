"""Main ingestion: local PDFs + Zotero + S2 enrichment."""
import os
import argparse
import hashlib
from pathlib import Path
from src.utils import load_config, load_json, save_json
from src.acquire.parse_filenames import scan_pdf_directory
from src.acquire.zotero import parse_zotero_export, match_pdf_to_zotero
from src.acquire.enrich import batch_enrich


def make_paper_id(pdf_path):
    """Generate a stable, filesystem-safe paper ID."""
    stem = Path(pdf_path).stem
    if len(stem) > 80:
        h = hashlib.md5(stem.encode()).hexdigest()[:8]
        stem = stem[:72] + "_" + h
    safe = stem.replace(" ", "_")
    for ch in ["/", ":", "*", "?", "<", ">", "|"]:
        safe = safe.replace(ch, "_")
    return safe


def run_ingest(config_path="config.yaml", zotero_path=None, enrich=True, paper_filter=None):
    """Ingest PDFs and enrich metadata.

    Args:
        paper_filter: Optional substring to match against paper IDs or filenames.
            When set, only matching papers are added/enriched. Note: the manifest
            is rebuilt from scanned PDFs, so papers whose PDFs are no longer in
            pdf_dir will be dropped regardless of this filter.
    """
    cfg = load_config(config_path)
    pdf_dir = cfg["paths"]["pdf_dir"]
    corpus_dir = cfg["paths"]["corpus_dir"]
    os.makedirs(corpus_dir, exist_ok=True)

    print(f"[1/4] Scanning {pdf_dir}...")
    pdf_entries = scan_pdf_directory(pdf_dir)
    print(f"  Found {len(pdf_entries)} PDFs")
    if not pdf_entries:
        print("No PDFs found.")
        return

    if zotero_path and os.path.exists(zotero_path):
        print("[2/4] Loading Zotero export...")
        zotero_entries = parse_zotero_export(zotero_path)
        print(f"  Loaded {len(zotero_entries)} entries")
        matched = 0
        for entry in pdf_entries:
            match = match_pdf_to_zotero(entry, zotero_entries)
            if match:
                entry["arxiv_id"] = match.get("arxiv_id")
                entry["zotero_id"] = match.get("zotero_id")
                if match.get("year") and not entry.get("year"):
                    entry["year"] = match["year"]
                if match.get("title"):
                    entry["title_zotero"] = match["title"]
                matched += 1
        print(f"  Matched {matched}/{len(pdf_entries)}")
    else:
        print("[2/4] No Zotero export, skipping...")

    # Build full manifest from scanned PDFs
    full_manifest = []
    for entry in pdf_entries:
        pid = make_paper_id(entry["pdf_path"])
        full_manifest.append({
            "paper_id": pid,
            "local_pdf": entry["pdf_path"],
            "title": entry.get("title_zotero") or entry.get("title", ""),
            "authors": entry.get("authors", []),
            "authors_str": entry.get("authors_str", ""),
            "year": entry.get("year"),
            "arxiv_id": entry.get("arxiv_id"),
            "zotero_id": entry.get("zotero_id"),
            "doi": None, "abstract": "", "source": "local",
        })

    # Merge metadata from existing manifest to skip already-enriched papers
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    if os.path.exists(manifest_path):
        old_manifest = load_json(manifest_path)
        old_by_id = {p["paper_id"]: p for p in old_manifest}
        merge_keys = ("title", "authors", "authors_str", "year", "arxiv_id",
                      "doi", "abstract", "citation_count",
                      "influential_citation_count", "publication_types")
        merged = 0
        for paper in full_manifest:
            old = old_by_id.get(paper["paper_id"])
            if old:
                for k in merge_keys:
                    old_val = old.get(k)
                    new_val = paper.get(k)
                    # Merge if old has a value and new is missing/empty
                    if old_val is not None and old_val != "" and old_val != [] and (
                        new_val is None or new_val == "" or new_val == []
                    ):
                        paper[k] = old_val
                merged += 1
        print(f"  Merged metadata from existing manifest for {merged} papers")

    # Filter to specific paper(s) if requested
    if paper_filter:
        filt = paper_filter.lower()
        matched_papers = [
            p for p in full_manifest
            if filt in p["paper_id"].lower() or filt in p.get("local_pdf", "").lower()
        ]
        if not matched_papers:
            print(f"  No papers matching '{paper_filter}'")
            return
        print(f"  Filtered to {len(matched_papers)} paper(s) matching '{paper_filter}'")
        # Only enrich the matched papers
        if enrich:
            print(f"[3/4] Enriching {len(matched_papers)} papers via S2...")
            matched_papers = batch_enrich(matched_papers, delay=1.0)
        else:
            print("[3/4] S2 enrichment disabled")
        # Replace matched entries in full manifest
        matched_by_id = {p["paper_id"]: p for p in matched_papers}
        manifest = [matched_by_id.get(p["paper_id"], p) for p in full_manifest]
    else:
        if enrich:
            print(f"[3/4] Enriching {len(full_manifest)} papers via S2...")
            full_manifest = batch_enrich(full_manifest, delay=1.0)
        else:
            print("[3/4] S2 enrichment disabled")
        manifest = full_manifest

    manifest_path = os.path.join(corpus_dir, "manifest.json")
    save_json(manifest, manifest_path)
    n_ax = sum(1 for p in manifest if p.get("arxiv_id"))
    n_doi = sum(1 for p in manifest if p.get("doi"))
    n_abs = sum(1 for p in manifest if p.get("abstract"))
    print(f"[4/4] Saved {manifest_path}")
    print(f"  Total: {len(manifest)}, ArXiv: {n_ax}, DOI: {n_doi}, Abstract: {n_abs}")


def run_re_enrich(config_path="config.yaml"):
    """Re-enrich only papers with missing metadata in an existing manifest."""
    cfg = load_config(config_path)
    corpus_dir = cfg["paths"]["corpus_dir"]
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("No manifest.json found. Run ingest first.")
        return

    manifest = load_json(manifest_path)
    missing = [
        p for p in manifest
        if not p.get("year") or not p.get("authors") or not p.get("abstract")
    ]
    print(f"[re-enrich] {len(missing)}/{len(manifest)} papers have missing metadata")
    if not missing:
        print("  All papers already have year, authors, and abstract.")
        return

    manifest = batch_enrich(manifest, delay=1.0)

    save_json(manifest, manifest_path)
    n_year = sum(1 for p in manifest if p.get("year"))
    n_auth = sum(1 for p in manifest if p.get("authors"))
    n_abs = sum(1 for p in manifest if p.get("abstract"))
    print(f"  After re-enrich: year={n_year}, authors={n_auth}, abstract={n_abs} / {len(manifest)}")


def main():
    ap = argparse.ArgumentParser(description="Ingest local PDFs")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-z", "--zotero", help="Zotero pipe-delimited export")
    ap.add_argument("--no-enrich", action="store_true",
                    help="Skip Semantic Scholar enrichment")
    ap.add_argument("--re-enrich", action="store_true",
                    help="Re-enrich only papers with missing metadata in existing manifest")
    ap.add_argument("--paper", type=str, default=None,
                    help="Only process paper(s) matching this substring (matches ID or filename)")
    args = ap.parse_args()
    if args.re_enrich:
        run_re_enrich(args.config)
    else:
        run_ingest(args.config, args.zotero, enrich=not args.no_enrich, paper_filter=args.paper)


if __name__ == "__main__":
    main()
