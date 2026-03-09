"""Enrich paper metadata via Semantic Scholar API."""
import time
from difflib import SequenceMatcher
import requests

MIN_TITLE_SIMILARITY = 0.75


def _title_similarity(a, b):
    """Normalized title similarity (case-insensitive)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def enrich_via_s2(title, max_retries=2):
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title[:200], "limit": 3,
        "fields": "title,abstract,authors,externalIds,year",
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
                return None
            # Pick the best-matching result above the similarity threshold
            p = None
            for candidate in results:
                sim = _title_similarity(title, candidate.get("title", ""))
                if sim >= MIN_TITLE_SIMILARITY:
                    p = candidate
                    break
            if p is None:
                return None
            ext = p.get("externalIds") or {}
            return {
                "s2_title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "doi": ext.get("DOI"),
                "arxiv_id": ext.get("ArXiv"),
                "authors": [a.get("name", "")
                            for a in (p.get("authors") or [])],
                "year": p.get("year"),
            }
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    return None


def batch_enrich(papers, delay=1.0):
    enriched = 0
    for i, paper in enumerate(papers):
        if paper.get("abstract") and paper.get("doi"):
            continue
        title = paper.get("title", "")
        if len(title) < 10:
            continue
        result = enrich_via_s2(title)
        if result:
            if not paper.get("arxiv_id"):
                paper["arxiv_id"] = result.get("arxiv_id")
            if not paper.get("doi"):
                paper["doi"] = result.get("doi")
            if not paper.get("abstract"):
                paper["abstract"] = result.get("abstract", "")
            if not paper.get("authors") or not paper["authors"]:
                paper["authors"] = result.get("authors", [])
            if not paper.get("year"):
                paper["year"] = result.get("year")
            enriched += 1
        if (i + 1) % 10 == 0:
            print(f"  Enriched {i+1}/{len(papers)}...")
        time.sleep(delay)
    print(f"  Enriched {enriched}/{len(papers)} papers via S2")
    return papers
