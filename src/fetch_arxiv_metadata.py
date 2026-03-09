#!/usr/bin/env python3
"""Fetch author/title/year metadata from arXiv API and update manifest.json."""

import json
import time
import urllib.request
import xml.etree.ElementTree as ET

MANIFEST = "corpus/manifest.json"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
BATCH_SIZE = 100  # arXiv recommends <=200
DELAY = 3  # seconds between requests (arXiv rate limit)


def fetch_batch(arxiv_ids):
    """Fetch metadata for a batch of arXiv IDs."""
    id_list = ",".join(arxiv_ids)
    url = f"http://export.arxiv.org/api/query?id_list={id_list}&max_results={len(arxiv_ids)}"
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-kb-metadata/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return data


def parse_entries(xml_data):
    """Parse arXiv API XML response into metadata dicts."""
    root = ET.fromstring(xml_data)
    results = {}
    for entry in root.findall(f"{ATOM_NS}entry"):
        # Extract arXiv ID from the <id> URL
        id_url = entry.findtext(f"{ATOM_NS}id", "")
        # e.g. http://arxiv.org/abs/1804.01306v2
        arxiv_id = id_url.split("/abs/")[-1].split("v")[0] if "/abs/" in id_url else ""
        if not arxiv_id:
            continue

        title = entry.findtext(f"{ATOM_NS}title", "").replace("\n", " ").strip()
        # Clean up extra whitespace
        title = " ".join(title.split())

        authors = []
        for author_el in entry.findall(f"{ATOM_NS}author"):
            name = author_el.findtext(f"{ATOM_NS}name", "").strip()
            if name:
                authors.append(name)

        published = entry.findtext(f"{ATOM_NS}published", "")
        year = int(published[:4]) if len(published) >= 4 else None

        summary = entry.findtext(f"{ATOM_NS}summary", "").strip()
        summary = " ".join(summary.split())

        results[arxiv_id] = {
            "title": title,
            "authors": authors,
            "authors_str": ", ".join(authors),
            "year": year,
            "abstract": summary,
        }
    return results


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    # Collect all arXiv IDs that need metadata
    to_fetch = []
    for entry in manifest:
        aid = entry.get("arxiv_id", "")
        if aid:
            to_fetch.append(aid)

    print(f"Fetching metadata for {len(to_fetch)} papers from arXiv API...")

    # Fetch in batches
    all_metadata = {}
    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} papers)...")

        try:
            xml_data = fetch_batch(batch)
            parsed = parse_entries(xml_data)
            all_metadata.update(parsed)
            print(f"    Got {len(parsed)} results")
        except Exception as e:
            print(f"    Error: {e}")

        if i + BATCH_SIZE < len(to_fetch):
            time.sleep(DELAY)

    # Update manifest entries
    updated = 0
    for entry in manifest:
        aid = entry.get("arxiv_id", "")
        if aid in all_metadata:
            meta = all_metadata[aid]
            entry["title"] = meta["title"]
            entry["authors"] = meta["authors"]
            entry["authors_str"] = meta["authors_str"]
            entry["year"] = meta["year"]
            entry["abstract"] = meta["abstract"]
            updated += 1

    print(f"\nUpdated {updated}/{len(manifest)} entries in manifest")

    # Write back
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved to {MANIFEST}")


if __name__ == "__main__":
    main()
