"""Shared state and helpers for thesis-kb tools.

Re-exports singleton accessors from the main mcp_server module so tool
modules don't import mcp_server directly (avoids circular imports).
The main server sets these after startup via ``init()``.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI

# ---------------------------------------------------------------------------
# Singletons — set by init() from mcp_server.py at startup
# ---------------------------------------------------------------------------

_db: sqlite3.Connection | None = None
_collection: chromadb.Collection | None = None
_embed_client: OpenAI | None = None
_cfg: dict[str, Any] = {}
_project_root: Path = Path()

DB_PATH: Path = Path()
CHROMA_PATH: Path = Path()
COLLECTION_NAME: str = "thesis_nuggets"
EMBED_BASE_URL: str = ""
EMBED_MODEL: str = ""
EMBED_INSTRUCTION: str = ""


def init(
    cfg: dict,
    project_root: Path,
    db_path: Path,
    chroma_path: Path,
    collection_name: str,
    embed_base_url: str,
    embed_model: str,
    embed_instruction: str,
):
    """Called once from mcp_server.py to set shared state."""
    global _cfg, _project_root, DB_PATH, CHROMA_PATH, COLLECTION_NAME
    global EMBED_BASE_URL, EMBED_MODEL, EMBED_INSTRUCTION
    _cfg = cfg
    _project_root = project_root
    DB_PATH = db_path
    CHROMA_PATH = chroma_path
    COLLECTION_NAME = collection_name
    EMBED_BASE_URL = embed_base_url
    EMBED_MODEL = embed_model
    EMBED_INSTRUCTION = embed_instruction


# ---------------------------------------------------------------------------
# Lazy singleton accessors
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(str(DB_PATH))
        _db.row_factory = sqlite3.Row
    return _db


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key="none")
    return _embed_client


# ---------------------------------------------------------------------------
# Query / enrichment helpers
# ---------------------------------------------------------------------------

def sanitize_fts5_query(query: str) -> str:
    """Escape FTS5 special syntax so user input is treated as plain terms."""
    query = re.sub(r'["\']', " ", query)
    tokens = query.split()
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


def embed_query(text: str) -> list[float]:
    """Embed a query string via the configured embedding backend."""
    client = get_embed_client()
    formatted = f"Instruct: {EMBED_INSTRUCTION}\nQuery: {text}" if EMBED_INSTRUCTION else text
    resp = client.embeddings.create(model=EMBED_MODEL, input=[formatted])
    return resp.data[0].embedding


def get_paper(paper_id: str) -> dict | None:
    """Look up paper metadata from SQLite."""
    db = get_db()
    row = db.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return dict(row) if row else None


def get_nugget_qa(nugget_id: str) -> dict | None:
    """Get question/answer for a nugget from SQLite."""
    db = get_db()
    row = db.execute(
        "SELECT question, answer, type, section, paper_id FROM nuggets WHERE nugget_id = ?",
        (nugget_id,),
    ).fetchone()
    return dict(row) if row else None


def enrich_nugget(nugget_id: str, meta: dict, doc: str, distance: float) -> dict:
    """Build enriched result dict for a nugget."""
    paper_id = meta.get("paper_id", "")
    paper = get_paper(paper_id)
    return {
        "nugget_id": nugget_id,
        "distance": round(distance, 4),
        "type": meta.get("type", ""),
        "confidence": meta.get("confidence", ""),
        "section": meta.get("section", ""),
        "question": doc.split(" A: ")[0].split("Q: ")[-1] if "Q: " in doc else doc,
        "answer": doc.split(" A: ")[-1] if " A: " in doc else "",
        "document": doc,
        "paper_id": paper_id,
        "paper_title": paper.get("title", "") if paper else "",
        "paper_year": paper.get("year") if paper else None,
        "paper_authors": paper.get("authors", "") if paper else "",
        "arxiv_id": paper.get("arxiv_id", "") if paper else "",
    }


# ---------------------------------------------------------------------------
# Thesis file access
# ---------------------------------------------------------------------------

def get_thesis_root() -> Path | None:
    """Return thesis root directory from config, or None."""
    thesis_cfg = _cfg.get("thesis", {})
    root = thesis_cfg.get("root")
    if root:
        return Path(root).expanduser()
    return None


def read_bib_file() -> str | None:
    """Read the thesis .bib file."""
    thesis_cfg = _cfg.get("thesis", {})
    root = get_thesis_root()
    if not root:
        return None
    bib_rel = thesis_cfg.get("bib_file", "bibtex/bibliography.bib")
    bib_path = root / bib_rel
    if bib_path.exists():
        return bib_path.read_text(encoding="utf-8")
    return None


def read_glossary_file() -> str | None:
    """Read the thesis glossary .tex file."""
    thesis_cfg = _cfg.get("thesis", {})
    root = get_thesis_root()
    if not root:
        return None
    gls_rel = thesis_cfg.get("glossary_file", "bibtex/glossary.tex")
    gls_path = root / gls_rel
    if gls_path.exists():
        return gls_path.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# Cite key resolution (ported from cite_prefetch.py)
# ---------------------------------------------------------------------------

def parse_author_from_key(key: str) -> str | None:
    """Extract author last name from cite key like 'AliAkbarpour_2025:slug'."""
    base = key.split(":")[0] if ":" in key else key
    parts = base.split("_")
    return parts[0] if parts else None


def parse_year_from_key(key: str) -> int | None:
    """Extract year from cite key like 'AliAkbarpour_2025:slug'."""
    base = key.split(":")[0] if ":" in key else key
    for part in base.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def parse_slug_from_key(key: str) -> str:
    """Extract slug from cite key like 'Author_2025:Emerging-Trends-DVS' -> 'emerging trends dvs'."""
    if ":" not in key:
        return ""
    slug = key.split(":", 1)[1]
    return slug.lower().replace("-", " ").replace("_", " ")


def slug_title_score(slug: str, title: str) -> float:
    """Score how well a cite key slug matches a paper title (0-1)."""
    if not slug or not title:
        return 0.0
    slug_words = set(slug.split())
    stop = {"", "a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "from", "by"}
    title_words = set(re.split(r"\W+", title.lower())) - stop
    if not slug_words:
        return 0.0
    overlap = len(slug_words & title_words)
    recall = overlap / len(slug_words)
    precision = overlap / len(title_words) if title_words else 0
    return recall * 0.7 + precision * 0.3


def resolve_cite_key(key: str) -> dict | None:
    """Resolve a cite key to a paper dict. Returns best match or None."""
    author = parse_author_from_key(key)
    if not author:
        return None
    year = parse_year_from_key(key)
    slug = parse_slug_from_key(key)

    db = get_db()
    # Try author + year first
    q = "SELECT * FROM papers WHERE authors LIKE ?"
    params: list = [f"%{author}%"]
    if year:
        q += " AND year = ?"
        params.append(year)
    q += " LIMIT 25"
    rows = db.execute(q, params).fetchall()
    papers = [dict(r) for r in rows]

    # Relax year if poor match
    best = max((slug_title_score(slug, p.get("title", "")) for p in papers), default=0) if slug and papers else 0
    if (best < 0.8 or not papers) and year:
        rows2 = db.execute(
            "SELECT * FROM papers WHERE authors LIKE ? LIMIT 25", (f"%{author}%",)
        ).fetchall()
        seen = {p["paper_id"] for p in papers}
        papers.extend(dict(r) for r in rows2 if dict(r)["paper_id"] not in seen)

    # Title fallback
    if slug and (not papers or best < 0.8):
        words = [w for w in slug.split() if len(w) > 2][:3]
        if words:
            conds = " AND ".join("title LIKE ?" for _ in words)
            parms = [f"%{w}%" for w in words]
            rows3 = db.execute(f"SELECT * FROM papers WHERE {conds} LIMIT 10", parms).fetchall()
            seen = {p["paper_id"] for p in papers}
            papers.extend(dict(r) for r in rows3 if dict(r)["paper_id"] not in seen)

    if not papers:
        return None

    if slug and len(papers) > 1:
        papers.sort(key=lambda p: slug_title_score(slug, p.get("title", "")), reverse=True)

    return papers[0]
