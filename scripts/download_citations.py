#!/usr/bin/env python3
"""Download PDFs for relevant citations using Semantic Scholar + arXiv."""
import argparse
import csv
import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,externalIds,openAccessPdf,isOpenAccess"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"

# S2 API key (optional, raises rate limit from 1/sec to 10/sec)
S2_API_KEY = os.environ.get("S2_API_KEY", "")


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_existing_index(pdf_dir):
    """Build a set of normalized titles from existing PDF filenames."""
    titles = set()
    for f in Path(pdf_dir).iterdir():
        if f.suffix.lower() != ".pdf":
            continue
        stem = f.stem
        stem = re.sub(r"^.+?\s+-\s+\d{4}\s+-\s+", "", stem)
        stem = re.sub(r"^\d{4}_\d{4,5}$", "", stem)
        stem = stem.replace("_", " ")
        norm = normalize(stem)
        if len(norm) > 10:
            titles.add(norm)
    return titles


def is_already_downloaded(title, existing_titles, threshold=0.85):
    norm = normalize(title)
    if not norm or len(norm) < 10:
        return False
    if norm in existing_titles:
        return True
    for et in existing_titles:
        if SequenceMatcher(None, norm, et).ratio() >= threshold:
            return True
    return False


def _s2_headers():
    h = {"User-Agent": "ThesisKB/1.0"}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h


def _s2_request(url, params, max_retries=4):
    """Make an S2 API request with retry + backoff."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15, headers=_s2_headers())
            if resp.status_code == 429:
                wait = min(3 * (attempt + 1), 15)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    return None


def search_s2(title):
    """Search S2 for a paper. Returns {url, s2_title, arxiv_id, doi} or None."""
    data = _s2_request(f"{S2_API}/paper/search",
                       {"query": title[:200], "limit": 3, "fields": S2_FIELDS})
    if not data:
        return None
    results = data.get("data", [])
    if not results:
        return None

    best, best_sim = None, 0
    for r in results:
        sim = SequenceMatcher(None, title.lower(), (r.get("title") or "").lower()).ratio()
        if sim > best_sim:
            best_sim, best = sim, r
    if best_sim < 0.70:
        return None

    ext_ids = best.get("externalIds") or {}
    pdf_info = best.get("openAccessPdf")
    pdf_url = pdf_info["url"] if pdf_info and pdf_info.get("url") else None
    arxiv_id = ext_ids.get("ArXiv")

    # If no OA PDF but has arXiv ID, use arXiv directly
    if not pdf_url and arxiv_id:
        pdf_url = ARXIV_PDF.format(arxiv_id=arxiv_id)

    if not pdf_url:
        return None

    return {
        "url": pdf_url,
        "s2_title": best.get("title", ""),
        "arxiv_id": arxiv_id,
        "doi": ext_ids.get("DOI"),
    }


def _extract_arxiv_id_from_venue(venue):
    """Try to extract arXiv ID from venue text like 'arXiv preprint arXiv:2103.12345'."""
    m = re.search(r"(\d{4}\.\d{4,5})", venue)
    return m.group(1) if m else None


def download_pdf(url, dest_path, timeout=30):
    try:
        resp = requests.get(url, timeout=timeout, stream=True,
                            headers={"User-Agent": "ThesisKB/1.0"},
                            allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "html" in ct and "pdf" not in ct:
            return False
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        with open(dest_path, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            os.remove(dest_path)
            return False
        return True
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def make_filename(row, s2_result):
    arxiv_id = s2_result.get("arxiv_id") if s2_result else None
    if arxiv_id:
        return arxiv_id.replace(".", "_") + ".pdf"

    authors = row.get("authors", "")
    year = row.get("year", "")
    title = s2_result.get("s2_title", "") if s2_result else row.get("title", "")

    first_author = ""
    if authors:
        parts = re.split(r",\s*(?=[A-Z])", authors)
        if parts:
            words = parts[0].strip().split()
            if words:
                first_author = words[-1].rstrip(",.")
    if not first_author:
        first_author = "Unknown"

    safe_title = re.sub(r"[^\w\s-]", "", title)
    safe_title = re.sub(r"\s+", " ", safe_title).strip()[:80]

    if year:
        return f"{first_author} - {year} - {safe_title}.pdf"
    return f"{first_author} - {safe_title}.pdf"


def search_s2_for_arxiv(title):
    """Search S2 just to get the arXiv ID for a paper."""
    data = _s2_request(f"{S2_API}/paper/search",
                       {"query": title[:200], "limit": 3, "fields": "title,externalIds"})
    if not data:
        return None
    results = data.get("data", [])
    if not results:
        return None

    best, best_sim = None, 0
    for r in results:
        sim = SequenceMatcher(None, title.lower(), (r.get("title") or "").lower()).ratio()
        if sim > best_sim:
            best_sim, best = sim, r
    if best_sim < 0.70 or not best:
        return None

    ext_ids = best.get("externalIds") or {}
    return ext_ids.get("ArXiv")


def retry_failed(args):
    """Retry previously failed downloads using arXiv as fallback."""
    log_path = "download_log.json"
    if not os.path.exists(log_path):
        print("No download_log.json found")
        return

    with open(log_path) as f:
        log = json.load(f)

    failed = [e for e in log if e["status"] in ("download_failed", "not_found")]
    print(f"Found {len(failed)} failed/not-found to retry")

    pdf_dir = Path(args.output_dir)
    stats = {"downloaded": 0, "arxiv_fallback": 0, "still_failed": 0}
    updated = []

    for entry in tqdm(failed, desc="Retrying"):
        title = entry["title"]
        orig_url = entry.get("url", "")

        # Strategy 1: Retry original URL (only for download_failed, not not_found)
        if orig_url:
            filename_guess = re.sub(r"[^\w\s-]", "", title)[:80].strip() + ".pdf"
            dest = pdf_dir / filename_guess
            if not dest.exists():
                if download_pdf(orig_url, str(dest), timeout=45):
                    stats["downloaded"] += 1
                    entry["status"] = "downloaded"
                    entry["filename"] = filename_guess
                    updated.append(entry)
                    continue

        # Strategy 2: Query S2 for arXiv ID, download from arXiv
        arxiv_id = search_s2_for_arxiv(title)
        time.sleep(args.delay)

        if arxiv_id:
            arxiv_url = ARXIV_PDF.format(arxiv_id=arxiv_id)
            filename = arxiv_id.replace(".", "_") + ".pdf"
            dest = pdf_dir / filename
            if dest.exists():
                stats["downloaded"] += 1
                entry["status"] = "already_exists"
                entry["filename"] = filename
                updated.append(entry)
                continue
            if download_pdf(arxiv_url, str(dest)):
                stats["arxiv_fallback"] += 1
                entry["status"] = "downloaded"
                entry["filename"] = filename
                updated.append(entry)
                continue

        stats["still_failed"] += 1
        updated.append(entry)

    # Update log: replace failed/not_found entries with updated ones
    failed_titles = {e["title"] for e in failed}
    new_log = [e for e in log if e["status"] not in ("download_failed", "not_found") or e["title"] not in failed_titles]
    new_log.extend(updated)
    with open(log_path, "w") as f:
        json.dump(new_log, f, indent=2)

    print("\nRetry results:")
    print(f"  Downloaded (original URL): {stats['downloaded']}")
    print(f"  Downloaded (arXiv):        {stats['arxiv_fallback']}")
    print(f"  Still failed:              {stats['still_failed']}")


def main():
    ap = argparse.ArgumentParser(description="Download PDFs for cited papers")
    ap.add_argument("--input", default="citations_relevant.csv")
    ap.add_argument("--output-dir", default=os.path.expanduser("~/Documents/thesis-papers"))
    ap.add_argument("--delay", type=float, default=1.0,
                    help="Delay between S2 API calls")
    ap.add_argument("--max-downloads", type=int, default=0,
                    help="Max papers to attempt (0 = unlimited)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-citations", type=int, default=2)
    ap.add_argument("--resume", action="store_true",
                    help="Skip entries already in download_log.json")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Retry previously failed downloads")
    args = ap.parse_args()

    if args.retry_failed:
        retry_failed(args)
        return

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if int(r.get("occurrence_count", 0)) >= args.min_citations]
    print(f"Loaded {len(rows)} citations (>={args.min_citations}x cited)")

    pdf_dir = Path(args.output_dir)
    existing = build_existing_index(pdf_dir)
    print(f"Found {len(existing)} existing papers in {pdf_dir}")

    # Load previous log for resume — only skip successes, not failures
    already_tried = set()
    if args.resume and os.path.exists("download_log.json"):
        with open("download_log.json") as f:
            prev_log = json.load(f)
        already_tried = {e["title"] for e in prev_log
                         if e["status"] in ("downloaded", "already_exists")}
        print(f"Resuming: skipping {len(already_tried)} previously downloaded")

    to_download = []
    skipped = 0
    for r in rows:
        if r["title"] in already_tried:
            skipped += 1
        elif is_already_downloaded(r["title"], existing):
            skipped += 1
        else:
            to_download.append(r)
    print(f"Skipping {skipped} already done, {len(to_download)} to search")

    if args.max_downloads > 0:
        to_download = to_download[:args.max_downloads]

    stats = {"found": 0, "downloaded": 0, "not_found": 0, "no_oa": 0, "failed": 0}
    log = []

    for row in tqdm(to_download, desc="Downloading"):
        title = row["title"]

        # First: check if venue has arXiv ID — skip S2 entirely
        arxiv_id = _extract_arxiv_id_from_venue(row.get("venue", ""))
        result = None

        if arxiv_id:
            url = ARXIV_PDF.format(arxiv_id=arxiv_id)
            result = {"url": url, "s2_title": title, "arxiv_id": arxiv_id, "doi": None}
        else:
            result = search_s2(title)
            time.sleep(args.delay)

        if not result:
            stats["not_found"] += 1
            log.append({"title": title, "status": "not_found"})
            continue

        stats["found"] += 1
        filename = make_filename(row, result)
        dest = pdf_dir / filename

        if args.dry_run:
            stats["downloaded"] += 1
            log.append({"title": title, "status": "found",
                         "url": result["url"], "filename": filename})
            continue

        if dest.exists():
            stats["downloaded"] += 1
            log.append({"title": title, "status": "already_exists", "filename": filename})
            continue

        if download_pdf(result["url"], str(dest)):
            stats["downloaded"] += 1
            existing.add(normalize(title))
            log.append({"title": title, "status": "downloaded", "filename": filename})
        else:
            stats["failed"] += 1
            log.append({"title": title, "status": "download_failed", "url": result["url"]})

    print("\nResults:")
    print(f"  Found:       {stats['found']}")
    print(f"  Downloaded:  {stats['downloaded']}")
    print(f"  Not found:   {stats['not_found']}")
    print(f"  Failed:      {stats['failed']}")

    # Append to existing log if resuming
    log_path = "download_log.json"
    if args.resume and os.path.exists(log_path):
        with open(log_path) as f:
            prev = json.load(f)
        log = prev + log
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log:         {log_path}")


if __name__ == "__main__":
    main()
