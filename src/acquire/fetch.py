"""Fetch missing papers from ArXiv/S2 based on Zotero export."""
import os
import time
import re
import argparse
import hashlib
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from src.utils import load_config, load_json, save_json
from src.acquire.zotero import parse_zotero_export, _norm
from src.acquire.enrich import enrich_via_s2


def download_pdf(url, dest_path, timeout=30):
    """Download PDF and validate header."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            with open(dest_path, "wb") as f:
                f.write(resp.read())
        with open(dest_path, "rb") as f:
            hdr = f.read(5)
        if not hdr.startswith(b"%PDF"):
            os.remove(dest_path)
            return False
        return True
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def make_safe_filename(title, arxiv_id=None, max_len=80):
    """Create a filesystem-safe filename from title."""
    if arxiv_id:
        safe = arxiv_id.replace("/", "_").replace(".", "_")
        return safe
    safe = re.sub(r"[^a-zA-Z0-9\s\-]", "", title)
    safe = re.sub(r"\s+", "_", safe).strip("_")
    if len(safe) > max_len:
        h = hashlib.md5(title.encode()).hexdigest()[:8]
        safe = safe[:max_len - 9] + "_" + h
    return safe


def find_already_have(manifest, zotero_entries, threshold=0.80):
    """Find which Zotero entries we already have locally."""
    manifest_norms = []
    for p in manifest:
        t = _norm(p.get("title", ""))
        manifest_norms.append(t)
    # Also check by arxiv_id
    manifest_arxiv = set(
        p.get("arxiv_id") for p in manifest if p.get("arxiv_id")
    )

    have, missing = [], []
    for ze in zotero_entries:
        # Check arxiv_id match
        if ze.get("arxiv_id") and ze["arxiv_id"] in manifest_arxiv:
            have.append(ze)
            continue
        # Check title similarity
        matched = False
        for mn in manifest_norms:
            if SequenceMatcher(None, ze["title_norm"], mn).ratio() > threshold:
                matched = True
                break
        if matched:
            have.append(ze)
        else:
            missing.append(ze)
    return have, missing


def dedup_entries(entries, threshold=0.85):
    """Remove duplicate Zotero entries by title similarity."""
    seen_norms = []
    unique = []
    for e in entries:
        tn = e.get("title_norm", "")
        is_dup = any(
            SequenceMatcher(None, tn, s).ratio() > threshold
            for s in seen_norms
        )
        if not is_dup:
            seen_norms.append(tn)
            unique.append(e)
    return unique


def fetch_paper(entry, pdf_dir, delay=1.0):
    """Try to download a paper PDF. Returns (path, source) or (None, None)."""
    arxiv_id = entry.get("arxiv_id")
    title = entry.get("title", "")
    fname = make_safe_filename(title, arxiv_id)
    dest = os.path.join(pdf_dir, f"{fname}.pdf")

    if os.path.exists(dest):
        return dest, "cached"

    # Try ArXiv first
    if arxiv_id:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if download_pdf(url, dest):
            time.sleep(delay)
            return dest, "arxiv"
        time.sleep(delay)

    # Try Semantic Scholar for open access PDF
    s2 = enrich_via_s2(title)
    if s2:
        # S2 doesn not return PDF URLs directly in our current code,
        # but we can try arxiv_id from S2 if we did not have one
        s2_arxiv = s2.get("arxiv_id")
        if s2_arxiv and s2_arxiv != arxiv_id:
            url = f"https://arxiv.org/pdf/{s2_arxiv}.pdf"
            if download_pdf(url, dest):
                entry["arxiv_id"] = s2_arxiv  # update entry
                time.sleep(delay)
                return dest, "s2_arxiv"
        # Try DOI redirect via doi.org
        doi = s2.get("doi")
        if doi:
            entry["doi"] = doi
    time.sleep(delay)
    return None, None


def run_fetch(config_path="config.yaml", zotero_path=None, dry_run=False):
    """Fetch papers from Zotero list that are missing locally."""
    cfg = load_config(config_path)
    pdf_dir = cfg["paths"]["pdf_dir"]
    corpus_dir = cfg["paths"]["corpus_dir"]
    os.makedirs(pdf_dir, exist_ok=True)

    if not zotero_path:
        print("No Zotero export provided. Use -z flag.")
        return

    # Load Zotero entries
    print(f"[fetch] Loading Zotero export: {zotero_path}")
    zot_entries = parse_zotero_export(zotero_path)
    print(f"  Raw entries: {len(zot_entries)}")

    # Dedup
    zot_entries = dedup_entries(zot_entries)
    print(f"  After dedup: {len(zot_entries)}")

    # Load existing manifest
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    manifest = load_json(manifest_path) if os.path.exists(manifest_path) else []
    print(f"  Existing manifest: {len(manifest)} papers")

    # Diff
    have, missing = find_already_have(manifest, zot_entries)
    n_with_arxiv = sum(1 for e in missing if e.get("arxiv_id"))
    print(f"  Already have: {len(have)}")
    print(f"  Missing: {len(missing)} ({n_with_arxiv} with ArXiv ID)")

    if dry_run:
        print("\n[DRY RUN] Would fetch:")
        for e in missing[:20]:
            src = "arxiv" if e.get("arxiv_id") else "s2_lookup"
            title = e["title"][:70]
            print(f"  [{src:10s}] {title}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return

    # Fetch missing papers
    print(f"\n[fetch] Downloading {len(missing)} papers...")
    fetched, failed = 0, 0
    new_entries = []
    for i, entry in enumerate(missing):
        path, source = fetch_paper(entry, pdf_dir, delay=1.5)
        if path:
            paper_id = Path(path).stem
            new_entry = {
                "paper_id": paper_id,
                "local_pdf": path,
                "title": entry.get("title", ""),
                "authors": [],
                "year": entry.get("year"),
                "arxiv_id": entry.get("arxiv_id"),
                "zotero_id": entry.get("zotero_id"),
                "doi": entry.get("doi"),
                "abstract": "",
                "source": source,
            }
            new_entries.append(new_entry)
            fetched += 1
            title = entry["title"][:50]
            print(f"  [{i+1}/{len(missing)}] OK ({source}) {title}")
        else:
            failed += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(missing)}] {fetched} ok, {failed} failed")

    # Merge into manifest
    manifest.extend(new_entries)
    save_json(manifest, manifest_path)
    print(f"\nDone: {fetched} fetched, {failed} failed")
    print(f"Manifest now has {len(manifest)} papers")


def main():
    ap = argparse.ArgumentParser(description="Fetch missing papers from Zotero list")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-z", "--zotero", required=True, help="Zotero export file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be fetched without downloading")
    args = ap.parse_args()
    run_fetch(args.config, args.zotero, args.dry_run)


if __name__ == "__main__":
    main()
