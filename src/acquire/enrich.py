"""Enrich paper metadata via Semantic Scholar API."""
import time
from difflib import SequenceMatcher
import requests

MIN_TITLE_SIMILARITY = 0.75
S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,authors,externalIds,year"


def _title_similarity(a, b):
    """Normalized title similarity (case-insensitive)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _extract_result(p):
    """Extract enrichment data from an S2 paper result."""
    ext = p.get("externalIds") or {}
    return {
        "s2_title": p.get("title", ""),
        "abstract": p.get("abstract", ""),
        "doi": ext.get("DOI"),
        "arxiv_id": ext.get("ArXiv"),
        "authors": [a.get("name", "") for a in (p.get("authors") or [])],
        "year": p.get("year"),
    }


def enrich_via_arxiv_id(arxiv_id, max_retries=2):
    """Look up a paper directly by its arXiv ID via S2."""
    url = f"{S2_API}/paper/ArXiv:{arxiv_id}"
    params = {"fields": S2_FIELDS}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return _extract_result(resp.json())
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"    S2 arXiv lookup failed for {arxiv_id}: {e}")
            return None
    return None


def enrich_via_s2(title, arxiv_id=None, max_retries=2):
    """Enrich via S2 title search, falling back to arXiv ID direct lookup."""
    # Try direct arXiv ID lookup first (most reliable)
    if arxiv_id:
        result = enrich_via_arxiv_id(arxiv_id, max_retries)
        if result:
            return result

    # Fall back to title search
    if not title or len(title.strip()) < 10:
        return None

    base = f"{S2_API}/paper/search"
    params = {
        "query": title[:200], "limit": 3,
        "fields": S2_FIELDS,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(base, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            results = resp.json().get("data", [])
            if not results:
                print(f"    S2 title search: no results for '{title[:60]}'")
                return None
            # Pick the best-matching result above the similarity threshold
            p = None
            for candidate in results:
                sim = _title_similarity(title, candidate.get("title", ""))
                if sim >= MIN_TITLE_SIMILARITY:
                    p = candidate
                    break
            if p is None:
                best = results[0]
                best_sim = _title_similarity(title, best.get("title", ""))
                print(f"    S2 title search: below threshold (best sim={best_sim:.2f}) for '{title[:60]}'")
                return None
            return _extract_result(p)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"    S2 title search failed for '{title[:60]}': {e}")
            return None
    return None


def batch_enrich(papers, delay=1.0):
    """Enrich papers missing metadata via S2."""
    enriched = 0
    skipped = 0
    for i, paper in enumerate(papers):
        # Skip only if we already have ALL key metadata
        has_year = paper.get("year") is not None
        has_authors = bool(paper.get("authors"))
        has_abstract = bool(paper.get("abstract"))
        if has_year and has_authors and has_abstract:
            skipped += 1
            continue

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
            enriched += 1
        if (i + 1) % 10 == 0:
            print(f"  Enriched {i+1}/{len(papers)} ({enriched} new, {skipped} already complete)...")
        time.sleep(delay)
    print(f"  Enriched {enriched}/{len(papers)} papers via S2 ({skipped} already complete)")
    return papers
