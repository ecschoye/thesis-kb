"""API clients for arXiv, Semantic Scholar, and OpenAlex."""

import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "title,abstract,authors,externalIds,year,"
    "citationCount,influentialCitationCount,publicationTypes,openAccessPdf"
)

OPENALEX_API = "https://api.openalex.org/works"


@dataclass
class CandidatePaper:
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    arxiv_id: str | None = None
    doi: str | None = None
    s2_id: str | None = None
    source: str = ""
    citation_count: int = 0
    url: str = ""
    pdf_url: str | None = None


# ── Rate limiter ──────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self):
        self._last: dict[str, float] = {}

    def wait(self, source: str, min_delay: float):
        now = time.monotonic()
        last = self._last.get(source, 0)
        gap = min_delay - (now - last)
        if gap > 0:
            time.sleep(gap)
        self._last[source] = time.monotonic()


_limiter = RateLimiter()


# ── arXiv ─────────────────────────────────────────────────────────────

def _arxiv_search(query, max_results=50, date_from=None, max_retries=3):
    """Search arXiv Atom API with retry. Returns list[CandidatePaper]."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(max_retries):
        _limiter.wait("arxiv", 5.0)  # arXiv needs generous delays
        try:
            resp = requests.get(
                "http://export.arxiv.org/api/query",
                params=params,
                headers={"User-Agent": "thesis-kb-discover/1.0"},
                timeout=45,
            )
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    [arXiv] rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return _parse_arxiv_xml(resp.content, date_from, source="arxiv")
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            print(f"  [arXiv] query timed out: {query[:40]}")
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            print(f"  [arXiv] query failed: {e}")
            return []
    return []


def _parse_arxiv_xml(xml_data, date_from=None, source="arxiv"):
    """Parse arXiv Atom XML into CandidatePaper list."""
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        id_url = entry.findtext(f"{ATOM_NS}id", "")
        arxiv_id = id_url.split("/abs/")[-1].split("v")[0] if "/abs/" in id_url else ""
        if not arxiv_id:
            continue

        published = entry.findtext(f"{ATOM_NS}published", "")
        year = int(published[:4]) if len(published) >= 4 else None

        # Date filter
        if date_from and published:
            try:
                pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if pub_date.date() < date_from:
                    continue
            except Exception:
                pass

        title = " ".join(entry.findtext(f"{ATOM_NS}title", "").split())
        abstract = " ".join(entry.findtext(f"{ATOM_NS}summary", "").split())
        authors = [
            a.findtext(f"{ATOM_NS}name", "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        authors = [a for a in authors if a]

        # PDF link
        pdf_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append(CandidatePaper(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            arxiv_id=arxiv_id,
            source=source,
            citation_count=0,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=pdf_url,
        ))
    return papers


# ── Semantic Scholar ──────────────────────────────────────────────────

def _s2_headers():
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def _s2_request(url, params=None, max_retries=3):
    """Make an S2 API request with retry + rate limiting."""
    delay = 1.0 if not os.environ.get("S2_API_KEY") else 0.15
    for attempt in range(max_retries):
        _limiter.wait("s2", delay)
        try:
            resp = requests.get(url, params=params, headers=_s2_headers(), timeout=20)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  [S2] rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"  [S2] request failed: {e}")
            return None
    return None


def _s2_paper_to_candidate(p, source="s2_search"):
    """Convert S2 paper dict to CandidatePaper."""
    ext = p.get("externalIds") or {}
    oa = p.get("openAccessPdf") or {}
    return CandidatePaper(
        title=p.get("title", ""),
        authors=[a.get("name", "") for a in (p.get("authors") or [])],
        year=p.get("year"),
        abstract=p.get("abstract") or "",
        arxiv_id=ext.get("ArXiv"),
        doi=ext.get("DOI"),
        s2_id=p.get("paperId"),
        source=source,
        citation_count=p.get("citationCount", 0) or 0,
        url=f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
        pdf_url=oa.get("url"),
    )


def s2_keyword_search(query, max_results=50):
    """Search S2 by keywords. Returns list[CandidatePaper]."""
    data = _s2_request(
        f"{S2_API}/paper/search",
        params={"query": query[:200], "limit": min(max_results, 100), "fields": S2_FIELDS},
    )
    if not data:
        return []
    return [
        _s2_paper_to_candidate(p)
        for p in data.get("data", [])
        if p.get("title")
    ]


def s2_citation_expand(arxiv_id, direction="citations", limit=100):
    """Get citing or referenced papers for an arXiv ID via S2.

    direction: "citations" (papers that cite this) or "references" (papers this cites).
    """
    paper_id = f"ArXiv:{arxiv_id}"
    data = _s2_request(
        f"{S2_API}/paper/{paper_id}/{direction}",
        params={"fields": S2_FIELDS, "limit": limit},
    )
    if not data:
        return []
    results = []
    for item in data.get("data", []):
        p = item.get("citingPaper" if direction == "citations" else "citedPaper")
        if p and p.get("title"):
            results.append(_s2_paper_to_candidate(p, source=f"s2_{direction}"))
    return results


# ── OpenAlex ──────────────────────────────────────────────────────────

def _reconstruct_abstract(inverted_index):
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def openalex_search(query, max_results=50, date_from=None):
    """Search OpenAlex. Returns list[CandidatePaper]."""
    params = {
        "search": query,
        "per_page": min(max_results, 50),
        "mailto": "thesis-kb@example.com",
    }
    if date_from:
        params["filter"] = f"from_publication_date:{date_from.isoformat()}"

    _limiter.wait("openalex", 0.15)
    try:
        resp = requests.get(OPENALEX_API, params=params, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [OpenAlex] query failed: {e}")
        return []

    papers = []
    for work in resp.json().get("results", []):
        doi_raw = work.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

        # Extract arXiv ID from locations
        arxiv_id = None
        for loc in work.get("locations", []):
            landing = loc.get("landing_page_url") or ""
            if "arxiv.org/abs/" in landing:
                arxiv_id = landing.split("/abs/")[-1].split("v")[0]
                break

        # PDF URL
        pdf_url = None
        oa = work.get("open_access") or {}
        pdf_url = oa.get("oa_url")

        papers.append(CandidatePaper(
            title=work.get("display_name", ""),
            authors=[
                a.get("author", {}).get("display_name", "")
                for a in (work.get("authorships") or [])
            ],
            year=work.get("publication_year"),
            abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
            arxiv_id=arxiv_id,
            doi=doi,
            source="openalex",
            citation_count=work.get("cited_by_count", 0),
            url=work.get("id", ""),
            pdf_url=pdf_url,
        ))
    return papers


# ── Unified search interface ─────────────────────────────────────────

def search_all_sources(
    queries,
    sources=("arxiv", "s2"),
    seed_arxiv_ids=None,
    max_per_query=50,
    citation_expand_n=20,
    date_from=None,
):
    """Run all queries across selected sources. Returns list[CandidatePaper]."""
    all_papers = []

    for i, query in enumerate(queries):
        print(f"  Query {i+1}/{len(queries)}: {query[:60]}...")

        if "arxiv" in sources:
            results = _arxiv_search(query, max_results=max_per_query, date_from=date_from)
            print(f"    arXiv: {len(results)} results")
            all_papers.extend(results)

        if "s2" in sources:
            results = s2_keyword_search(query, max_results=max_per_query)
            print(f"    S2: {len(results)} results")
            all_papers.extend(results)

        if "openalex" in sources:
            results = openalex_search(query, max_results=max_per_query, date_from=date_from)
            print(f"    OpenAlex: {len(results)} results")
            all_papers.extend(results)

    # Citation graph expansion via S2
    if "s2" in sources and seed_arxiv_ids:
        seeds = seed_arxiv_ids[:citation_expand_n]
        print(f"\n  Citation expansion: {len(seeds)} seed papers...")
        for j, aid in enumerate(seeds):
            citing = s2_citation_expand(aid, direction="citations", limit=50)
            print(f"    Seed {j+1}/{len(seeds)} ({aid}): {len(citing)} citing papers")
            all_papers.extend(citing)

    print(f"\n  Total raw candidates: {len(all_papers)}")
    return all_papers
