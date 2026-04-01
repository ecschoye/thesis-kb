"""Enrich paper metadata via Semantic Scholar API."""
import os
import time
from difflib import SequenceMatcher
import requests

MIN_TITLE_SIMILARITY = 0.75
S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,authors,externalIds,year,citationCount,influentialCitationCount,publicationTypes"

def _s2_headers():
    """Return headers with API key if available."""
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


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
        "citation_count": p.get("citationCount", 0) or 0,
        "influential_citation_count": p.get("influentialCitationCount", 0) or 0,
        "publication_types": p.get("publicationTypes") or [],
    }


def enrich_via_arxiv_id(arxiv_id, max_retries=2):
    """Look up a paper directly by its arXiv ID via S2."""
    url = f"{S2_API}/paper/ArXiv:{arxiv_id}"
    params = {"fields": S2_FIELDS}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=_s2_headers(), timeout=15)
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


def enrich_via_doi(doi, max_retries=2):
    """Look up a paper directly by its DOI via S2."""
    url = f"{S2_API}/paper/DOI:{doi}"
    params = {"fields": S2_FIELDS}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=_s2_headers(), timeout=15)
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
            print(f"    S2 DOI lookup failed for {doi}: {e}")
            return None
    return None


def enrich_via_s2(title, arxiv_id=None, doi=None, max_retries=2):
    """Enrich via S2 direct ID lookup (arXiv, DOI), falling back to title search."""
    # Try direct arXiv ID lookup first (most reliable)
    if arxiv_id:
        result = enrich_via_arxiv_id(arxiv_id, max_retries)
        if result:
            return result

    # Try DOI lookup
    if doi:
        result = enrich_via_doi(doi, max_retries)
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
            resp = requests.get(base, params=params, headers=_s2_headers(), timeout=15)
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
    authority_enriched = 0
    for i, paper in enumerate(papers):
        has_year = paper.get("year") is not None
        authors = paper.get("authors") or []
        has_authors = bool(authors) and not (
            len(authors) == 1 and " " not in authors[0]
        )
        has_abstract = bool(paper.get("abstract"))
        has_authority = "citation_count" in paper

        if has_year and has_authors and has_abstract and has_authority:
            skipped += 1
            continue

        title = paper.get("title", "")
        arxiv_id = paper.get("arxiv_id")
        doi = paper.get("doi")
        result = enrich_via_s2(title, arxiv_id=arxiv_id, doi=doi)
        if result:
            if result.get("s2_title") and (not title or len(title) < 20):
                paper["title"] = result["s2_title"]
            if not paper.get("arxiv_id"):
                paper["arxiv_id"] = result.get("arxiv_id")
            if not paper.get("doi"):
                paper["doi"] = result.get("doi")
            if not paper.get("abstract"):
                paper["abstract"] = result.get("abstract", "")
            existing = paper.get("authors") or []
            s2_authors = result.get("authors", [])
            # Override if missing, single-word (likely parsed from title), or S2 has more detail
            if not existing or (len(existing) == 1 and " " not in existing[0] and s2_authors):
                paper["authors"] = s2_authors
                paper["authors_str"] = ", ".join(s2_authors)
            if not paper.get("year"):
                paper["year"] = result.get("year")
            # Authority metadata (always update — may be newly available)
            paper["citation_count"] = result.get("citation_count", 0)
            paper["influential_citation_count"] = result.get("influential_citation_count", 0)
            paper["publication_types"] = result.get("publication_types", [])
            if has_year and has_authors and has_abstract:
                authority_enriched += 1
            else:
                enriched += 1
        if (i + 1) % 10 == 0:
            print(f"  Enriched {i+1}/{len(papers)} ({enriched} new, {authority_enriched} authority-only, {skipped} already complete)...")
        time.sleep(delay)
    print(f"  Enriched {enriched}/{len(papers)} papers via S2 ({authority_enriched} authority-only, {skipped} already complete)")
    return papers
