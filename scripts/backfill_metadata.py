"""Backfill manifest metadata by recovering arXiv IDs from paper_ids
and extracting titles from PDF text, then re-enriching via S2."""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.acquire.enrich import enrich_via_s2

MANIFEST = "corpus/manifest.json"
TEXT_DIR = "corpus/texts"
ARXIV_RE = re.compile(r"^(\d{4})_(\d{4,5})$")


def extract_title_from_text(paper_id):
    """Try to pull a title from the first page of extracted text."""
    path = os.path.join(TEXT_DIR, f"{paper_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        pages = data.get("pages", [])
        if not pages:
            return None
        text = pages[0].get("text", "")
        # Take non-empty lines from the first page, skip very short ones
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Skip common preamble lines
        skip = {"abstract", "arxiv", "proceedings", "conference", "journal",
                "please do not remove this page", "preprint"}
        candidates = []
        for line in lines[:15]:
            low = line.lower()
            if len(line) < 10:
                continue
            if any(s in low for s in skip):
                continue
            if low.startswith("http"):
                continue
            # Title lines are typically medium-length, not too long
            if 10 <= len(line) <= 200:
                candidates.append(line)
            if len(candidates) >= 3:
                break
        # The title is usually the longest of the first few candidate lines
        if candidates:
            return max(candidates, key=len)
    except Exception:
        pass
    return None


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    # Step 1: Recover arXiv IDs from paper_ids
    recovered_arxiv = 0
    for paper in manifest:
        if paper.get("arxiv_id"):
            continue
        m = ARXIV_RE.match(paper["paper_id"])
        if m:
            paper["arxiv_id"] = f"{m.group(1)}.{m.group(2)}"
            recovered_arxiv += 1
    print(f"[1/3] Recovered {recovered_arxiv} arXiv IDs from paper_ids")

    # Step 2: Extract titles from PDF text for named papers without titles
    extracted_titles = 0
    for paper in manifest:
        title = paper.get("title", "")
        if title and title != paper["paper_id"].replace("_", " ") and len(title) > 15:
            continue
        extracted = extract_title_from_text(paper["paper_id"])
        if extracted:
            paper["title"] = extracted
            extracted_titles += 1
    print(f"[2/3] Extracted {extracted_titles} titles from PDF text")

    # Step 3: Re-enrich papers missing metadata
    to_enrich = [
        p for p in manifest
        if not p.get("year") or not p.get("authors") or not p.get("abstract")
    ]
    print(f"[3/3] Enriching {len(to_enrich)} papers via S2...")

    enriched = 0
    failed = 0
    for i, paper in enumerate(to_enrich):
        title = paper.get("title", "")
        arxiv_id = paper.get("arxiv_id")
        result = enrich_via_s2(title, arxiv_id=arxiv_id)
        if result:
            if result.get("s2_title") and (not title or len(title) < 20):
                paper["title"] = result["s2_title"]
            if not paper.get("arxiv_id"):
                paper["arxiv_id"] = result.get("arxiv_id")
            if not paper.get("doi"):
                paper["doi"] = result.get("doi")
            if not paper.get("abstract"):
                paper["abstract"] = result.get("abstract", "")
            if not paper.get("authors") or not paper["authors"]:
                paper["authors"] = result.get("authors", [])
                paper["authors_str"] = ", ".join(result.get("authors", []))
            if not paper.get("year"):
                paper["year"] = result.get("year")
            paper["citation_count"] = result.get("citation_count", 0)
            paper["influential_citation_count"] = result.get("influential_citation_count", 0)
            paper["publication_types"] = result.get("publication_types", [])
            enriched += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(to_enrich)} done ({enriched} enriched, {failed} failed)")
            # Save progress periodically
            with open(MANIFEST, "w") as f:
                json.dump(manifest, f, indent=2)

        # S2 rate limit: 100 req/5min without key, ~1/s is safe
        time.sleep(0.5)

    # Final save
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    # Stats
    n_title = sum(1 for p in manifest if p.get("title") and p["title"] != p["paper_id"].replace("_", " "))
    n_abs = sum(1 for p in manifest if (p.get("abstract") or "").strip())
    n_auth = sum(1 for p in manifest if p.get("authors"))
    n_year = sum(1 for p in manifest if p.get("year"))
    print(f"\nDone: enriched {enriched}, failed {failed}")
    print(f"Manifest coverage: title={n_title}, abstract={n_abs}, authors={n_auth}, year={n_year} / {len(manifest)}")


if __name__ == "__main__":
    main()
