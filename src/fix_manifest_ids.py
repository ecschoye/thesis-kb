#!/usr/bin/env python3
"""Look up missing arxiv_id and doi via Semantic Scholar API."""
import json
import urllib.request
import urllib.parse
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

with open("corpus/manifest.json") as f:
    manifest = json.load(f)

# Only entries missing arxiv_id or doi
to_fix = [e for e in manifest if not e.get("arxiv_id") or not e.get("doi")]
# Skip ones that already have both
to_fix = [e for e in to_fix if not (e.get("arxiv_id") and e.get("doi"))]
print(f"Looking up {len(to_fix)} entries...")

fixed_arxiv = 0
fixed_doi = 0
errors = 0
matched = 0

def query_s2(title, retries=3):
    q = urllib.parse.quote(title[:200])
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=1&fields=externalIds,title,year"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return None
        except Exception:
            time.sleep(3)
    return None

for i, entry in enumerate(to_fix):
    title = entry.get("title", "")
    if not title:
        continue

    data = query_s2(title)
    if data and data.get("data") and len(data["data"]) > 0:
        result = data["data"][0]
        ext = result.get("externalIds", {})
        result_title = result.get("title", "").lower().strip()
        our_title = title.lower().strip().rstrip(".")

        if result_title[:40] == our_title[:40] or our_title[:40] in result_title:
            matched += 1
            arxiv_id = ext.get("ArXiv", "")
            doi = ext.get("DOI", "")
            if arxiv_id and not entry.get("arxiv_id"):
                entry["arxiv_id"] = arxiv_id
                fixed_arxiv += 1
            if doi and not entry.get("doi"):
                entry["doi"] = doi
                fixed_doi += 1
    else:
        errors += 1

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(to_fix)} (matched: {matched}, +{fixed_arxiv} arxiv, +{fixed_doi} doi, {errors} err)")

    # Save every 50 entries
    if (i + 1) % 50 == 0:
        with open("corpus/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    time.sleep(3.5)  # ~100 req / 6 min = safe under S2 limit

# Final save
with open("corpus/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

has_arxiv = sum(1 for e in manifest if e.get("arxiv_id"))
has_doi = sum(1 for e in manifest if e.get("doi"))
still_no = sum(1 for e in manifest if not e.get("arxiv_id") and not e.get("doi"))
print(f"\nDone: matched {matched}, +{fixed_arxiv} arxiv, +{fixed_doi} doi, {errors} errors")
print(f"Now: {has_arxiv} with arxiv_id, {has_doi} with doi, {still_no} missing both (of {len(manifest)})")
