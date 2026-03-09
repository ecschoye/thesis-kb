#!/usr/bin/env python3
"""List completed papers (those with nuggets extracted) with metadata."""

import os
import json

NUGGETS_DIR = "corpus/nuggets"
MANIFEST = "corpus/manifest.json"


def get_done_ids(nuggets_dir=NUGGETS_DIR):
    """Return set of paper IDs that have nuggets extracted."""
    done = set()
    for f in os.listdir(nuggets_dir):
        if f.endswith(".json"):
            done.add(f.replace(".json", ""))
    return done


def load_manifest(manifest_path=MANIFEST):
    """Load paper metadata from manifest.json, keyed by paper_id."""
    papers = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest:
            pid = entry.get("paper_id", "")
            if pid:
                papers[pid] = entry
    return papers


def main():
    done_ids = get_done_ids()
    papers = load_manifest()

    results = []
    for pid in sorted(done_ids):
        info = papers.get(pid, {})
        arxiv_id = info.get("arxiv_id", "")
        title = info.get("title", "Unknown")
        year = info.get("year", "?")
        author_str = info.get("authors_str", "")
        display_id = arxiv_id if arxiv_id else pid[:40]
        results.append((display_id, title, author_str, year))

    results.sort(key=lambda x: x[0])

    print(f"{'ID':<45} {'Title':<80} {'Authors':<40} {'Year'}")
    print("-" * 170)
    for display_id, title, author_str, year in results:
        print(f"{display_id:<45} {title[:78]:<80} {str(author_str)[:38]:<40} {year}")

    print(f"\nTotal: {len(results)} papers done out of {len(papers)} total")


if __name__ == "__main__":
    main()
