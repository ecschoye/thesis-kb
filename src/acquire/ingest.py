"""Main ingestion: local PDFs + Zotero + S2 enrichment."""
import os, argparse, hashlib
from pathlib import Path
from src.utils import load_config, save_json
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


def run_ingest(config_path="config.yaml", zotero_path=None, enrich=True):
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

    manifest = []
    for entry in pdf_entries:
        pid = make_paper_id(entry["pdf_path"])
        manifest.append({
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

    if enrich:
        print(f"[3/4] Enriching {len(manifest)} papers via S2...")
        manifest = batch_enrich(manifest, delay=1.0)
    else:
        print("[3/4] S2 enrichment disabled")

    manifest_path = os.path.join(corpus_dir, "manifest.json")
    save_json(manifest, manifest_path)
    n_ax = sum(1 for p in manifest if p.get("arxiv_id"))
    n_doi = sum(1 for p in manifest if p.get("doi"))
    n_abs = sum(1 for p in manifest if p.get("abstract"))
    print(f"[4/4] Saved {manifest_path}")
    print(f"  Total: {len(manifest)}, ArXiv: {n_ax}, DOI: {n_doi}, Abstract: {n_abs}")


def main():
    ap = argparse.ArgumentParser(description="Ingest local PDFs")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-z", "--zotero", help="Zotero pipe-delimited export")
    ap.add_argument("--no-enrich", action="store_true",
                    help="Skip Semantic Scholar enrichment")
    args = ap.parse_args()
    run_ingest(args.config, args.zotero, enrich=not args.no_enrich)


if __name__ == "__main__":
    main()
