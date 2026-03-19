#!/usr/bin/env python3
"""Re-enrich papers with broken metadata (year IS NULL) from Semantic Scholar."""
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquire.enrich import enrich_via_arxiv_id, enrich_via_s2

DB_PATH = Path(__file__).resolve().parent.parent / "kb" / "nuggets.db"


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    broken = db.execute(
        "SELECT paper_id, title, authors, year, arxiv_id, doi, abstract "
        "FROM papers WHERE year IS NULL"
    ).fetchall()

    print(f"Found {len(broken)} papers with broken metadata\n")

    fixed = 0
    failed = []

    for i, paper in enumerate(broken):
        pid = paper["paper_id"]
        arxiv_id = paper["arxiv_id"]

        if arxiv_id:
            print(f"[{i+1}/{len(broken)}] {pid} (arXiv:{arxiv_id}) ... ", end="", flush=True)
            result = enrich_via_arxiv_id(arxiv_id)
        else:
            # Build a search title from paper_id or existing title
            title = paper["title"]
            if not title or len(title) < 10:
                title = pid.replace("_", " ")
            # Clean up common filename patterns
            for pattern in [" - ", "  "]:
                title = title.replace(pattern, " ")
            # Remove trailing hash suffixes like _883deb6a
            import re
            title = re.sub(r"\s*[a-f0-9]{8}$", "", title)
            print(f"[{i+1}/{len(broken)}] {pid} (title: '{title[:60]}') ... ", end="", flush=True)
            result = enrich_via_s2(title)

        if result and result.get("year"):
            authors_json = json.dumps(result["authors"]) if result.get("authors") else "[]"
            s2_title = result.get("s2_title", "")
            db.execute(
                """UPDATE papers SET
                    title = CASE WHEN title IS NULL OR length(title) < 20 THEN ? ELSE title END,
                    authors = CASE WHEN authors IS NULL OR authors = '[]' THEN ? ELSE authors END,
                    year = ?,
                    doi = CASE WHEN doi IS NULL THEN ? ELSE doi END,
                    abstract = CASE WHEN abstract IS NULL OR abstract = '' THEN ? ELSE abstract END,
                    citation_count = ?,
                    influential_citation_count = ?,
                    paper_type = ?
                WHERE paper_id = ?""",
                (
                    s2_title,
                    authors_json,
                    result["year"],
                    result.get("doi"),
                    result.get("abstract", ""),
                    result.get("citation_count", 0),
                    result.get("influential_citation_count", 0),
                    ",".join(result.get("publication_types", [])),
                    pid,
                ),
            )
            print(f"OK → '{s2_title[:60]}' ({result['year']})")
            fixed += 1
        else:
            print("FAILED")
            failed.append(pid)

        time.sleep(1.0)

    db.commit()
    db.close()

    print(f"\n{'='*60}")
    print(f"Fixed: {fixed}/{len(broken)}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for pid in failed:
            print(f"  - {pid}")
    else:
        print("All papers fixed!")


if __name__ == "__main__":
    main()
