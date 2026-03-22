"""MCP server for the thesis knowledge base.

Exposes ChromaDB vector search + SQLite FTS5 keyword search as tools
for Claude Code. Persistent process — ChromaDB and Ollama init happens
once at startup, then queries are instant.
"""
import os
import re
import json
import sqlite3
from pathlib import Path

import chromadb
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config & initialisation
# ---------------------------------------------------------------------------

def _find_config():
    """Resolve config path: env var > config-ollama.yaml > config.yaml."""
    if env := os.environ.get("THESIS_KB_CONFIG"):
        return env
    base = Path(__file__).resolve().parent.parent
    for name in ("config-ollama.yaml", "config.yaml"):
        p = base / name
        if p.exists():
            return str(p)
    raise FileNotFoundError("No config file found")


def _load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


CFG_PATH = _find_config()
CFG = _load_config(CFG_PATH)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Resolve paths relative to project root
KB_DIR = PROJECT_ROOT / CFG["paths"]["kb_dir"]
DB_PATH = KB_DIR / CFG.get("store", {}).get("sqlite", {}).get("db_name", "nuggets.db")
CHROMA_PATH = KB_DIR / "chromadb"
COLLECTION_NAME = CFG.get("store", {}).get("chromadb", {}).get("collection_name", "thesis_nuggets")

# Embedding config
EMB_CFG = CFG.get("embed", {})
EMB_BACKEND = EMB_CFG.get("backend", "ollama")
EMB_EMBEDDING = EMB_CFG.get("embedding", {})
EMBED_INSTRUCTION = EMB_EMBEDDING.get("instruction", "")

if EMB_BACKEND == "ollama":
    ollama_cfg = EMB_CFG.get("ollama", {})
    EMBED_BASE_URL = ollama_cfg.get("base_url", "http://localhost:11434/v1")
    EMBED_MODEL = ollama_cfg.get("model", "qwen3-embedding:8b")
else:
    vllm_cfg = EMB_CFG.get("vllm", {})
    port = int(os.environ.get("VLLM_PORT", vllm_cfg.get("port", 8000)))
    EMBED_BASE_URL = f"http://localhost:{port}/v1"
    EMBED_MODEL = vllm_cfg.get("model", "Qwen/Qwen3-Embedding-8B")

# ---------------------------------------------------------------------------
# Shared state (initialized once, reused across tool calls)
# ---------------------------------------------------------------------------

_db: sqlite3.Connection | None = None
_collection: chromadb.Collection | None = None
_embed_client: OpenAI | None = None


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(str(DB_PATH))
        _db.row_factory = sqlite3.Row
    return _db


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key="none")
    return _embed_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_fts5_query(query: str) -> str:
    """Escape FTS5 special syntax so user input is treated as plain terms."""
    query = re.sub(r'["\']', " ", query)
    tokens = query.split()
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


def _embed_query(text: str) -> list[float]:
    """Embed a query string via the configured embedding backend."""
    client = _get_embed_client()
    formatted = f"Instruct: {EMBED_INSTRUCTION}\nQuery: {text}" if EMBED_INSTRUCTION else text
    resp = client.embeddings.create(model=EMBED_MODEL, input=[formatted])
    return resp.data[0].embedding


def _get_paper(paper_id: str) -> dict | None:
    """Look up paper metadata from SQLite."""
    db = _get_db()
    row = db.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return dict(row) if row else None


def _enrich_nugget(nugget_id: str, meta: dict, doc: str, distance: float) -> dict:
    """Build enriched result dict for a nugget."""
    paper_id = meta.get("paper_id", "")
    paper = _get_paper(paper_id)
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


def _get_nugget_qa(nugget_id: str) -> dict | None:
    """Get question/answer for a nugget from SQLite."""
    db = _get_db()
    row = db.execute(
        "SELECT question, answer, type, section, paper_id FROM nuggets WHERE nugget_id = ?",
        (nugget_id,)
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("thesis-kb", instructions="""
Thesis knowledge base with ~150k QA nuggets from 1500+ academic papers on
event cameras, RGB-Event fusion, spiking neural networks, and object detection.

Use semantic_search for natural language queries (best quality).
Use bm25_search for keyword-based lookups (no embedding needed).
Use find_papers / get_paper_nuggets for paper-specific lookups.

Paper IDs use underscores (2401_17151); convert to arXiv dots (2401.17151) for citations.
Nugget types: method, result, claim, limitation, comparison, background.
""")


@mcp.tool()
def semantic_search(
    query: str,
    n: int = 20,
    types: list[str] | None = None,
    section: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[dict]:
    """Vector search over the thesis KB using ChromaDB + Ollama embeddings.

    Args:
        query: Natural language search query.
        n: Max results to return (default 20).
        types: Filter by nugget types (e.g. ["method", "result"]).
        section: Filter by paper section (e.g. "abstract", "methods").
        year_min: Minimum paper year (inclusive).
        year_max: Maximum paper year (inclusive).
    """
    collection = _get_collection()

    # Build ChromaDB where clause
    where_clauses = []
    if types and len(types) == 1:
        where_clauses.append({"type": types[0]})
    elif types and len(types) > 1:
        where_clauses.append({"type": {"$in": types}})
    if section:
        where_clauses.append({"section": section})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    query_embedding = _embed_query(query)
    fetch_n = n * 3 if (year_min or year_max) else n

    kwargs = {"query_embeddings": [query_embedding], "n_results": fetch_n}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    enriched = []
    for i in range(len(results["ids"][0])):
        nugget_id = results["ids"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        doc = results["documents"][0][i]

        paper_id = meta.get("paper_id", "")
        paper = _get_paper(paper_id)
        year = paper.get("year") if paper else None

        if year_min and (not year or year < year_min):
            continue
        if year_max and (not year or year > year_max):
            continue

        # Get Q/A from SQLite for cleaner output
        nugget_qa = _get_nugget_qa(nugget_id)

        enriched.append({
            "nugget_id": nugget_id,
            "distance": round(distance, 4),
            "type": meta.get("type", ""),
            "confidence": meta.get("confidence", ""),
            "section": meta.get("section", ""),
            "question": nugget_qa["question"] if nugget_qa else "",
            "answer": nugget_qa["answer"] if nugget_qa else "",
            "paper_id": paper_id,
            "paper_title": paper.get("title", "") if paper else "",
            "paper_year": year,
            "paper_authors": paper.get("authors", "") if paper else "",
            "arxiv_id": paper.get("arxiv_id", "") if paper else "",
        })
        if len(enriched) >= n:
            break

    return enriched


@mcp.tool()
def multi_search(
    queries: list[str],
    n: int = 30,
    types: list[str] | None = None,
) -> list[dict]:
    """Run multiple semantic queries and return deduplicated results, sorted by distance.

    Args:
        queries: List of search queries.
        n: Max total results to return (default 30).
        types: Filter by nugget types.
    """
    seen = set()
    all_results = []
    per_query = max(5, -(-n // len(queries)) + 3)
    for q in queries:
        results = semantic_search(q, n=per_query, types=types)
        for r in results:
            if r["nugget_id"] not in seen:
                seen.add(r["nugget_id"])
                all_results.append(r)
    all_results.sort(key=lambda x: x["distance"])
    return all_results[:n]


@mcp.tool()
def bm25_search(
    query: str,
    n: int = 20,
    types: list[str] | None = None,
) -> list[dict]:
    """Keyword search over the thesis KB using SQLite FTS5 (BM25 ranking).

    Args:
        query: Search terms (keywords).
        n: Max results (default 20).
        types: Filter by nugget types.
    """
    db = _get_db()
    sanitized = _sanitize_fts5_query(query)

    try:
        rows = db.execute(
            """SELECT nugget_id, rank
               FROM nuggets_fts
               WHERE nuggets_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (sanitized, n * 3 if types else n),
        ).fetchall()
    except Exception:
        return []

    results = []
    for nugget_id, rank in rows:
        nugget = _get_nugget_qa(nugget_id)
        if not nugget:
            continue
        if types and nugget["type"] not in types:
            continue

        paper = _get_paper(nugget["paper_id"])
        results.append({
            "nugget_id": nugget_id,
            "bm25_score": round(-rank, 4),
            "type": nugget["type"],
            "section": nugget["section"],
            "question": nugget["question"],
            "answer": nugget["answer"],
            "paper_id": nugget["paper_id"],
            "paper_title": paper.get("title", "") if paper else "",
            "paper_year": paper.get("year") if paper else None,
            "paper_authors": paper.get("authors", "") if paper else "",
            "arxiv_id": paper.get("arxiv_id", "") if paper else "",
        })
        if len(results) >= n:
            break

    return results


@mcp.tool()
def find_papers(
    author: str | None = None,
    title: str | None = None,
    year: int | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search for papers by author name, title substring, or year.

    Args:
        author: Author name substring (case-insensitive).
        title: Title substring (case-insensitive).
        year: Exact publication year.
        limit: Max results (default 10).
    """
    db = _get_db()
    conditions = []
    params: list = []
    if author:
        conditions.append("authors LIKE ?")
        params.append(f"%{author}%")
    if title:
        conditions.append("title LIKE ?")
        params.append(f"%{title}%")
    if year:
        conditions.append("year = ?")
        params.append(year)
    if not conditions:
        return []
    where = " AND ".join(conditions)
    rows = db.execute(
        f"SELECT * FROM papers WHERE {where} LIMIT ?",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def get_paper_nuggets(
    paper_id: str,
    types: list[str] | None = None,
    limit: int = 50,
) -> dict:
    """Get all nuggets for a specific paper.

    Args:
        paper_id: Paper ID (underscore format, e.g. "2401_17151").
        types: Filter by nugget types.
        limit: Max nuggets (default 50).
    """
    db = _get_db()
    paper = _get_paper(paper_id)

    query = "SELECT * FROM nuggets WHERE paper_id = ?"
    params: list = [paper_id]
    if types:
        placeholders = ",".join("?" * len(types))
        query += f" AND type IN ({placeholders})"
        params.extend(types)
    query += " LIMIT ?"
    params.append(limit)

    rows = db.execute(query, params).fetchall()
    nuggets = [dict(r) for r in rows]

    return {
        "paper": paper,
        "nugget_count": len(nuggets),
        "nuggets": nuggets,
    }


@mcp.tool()
def get_paper_info(paper_id: str) -> dict | None:
    """Get metadata for a single paper.

    Args:
        paper_id: Paper ID (underscore format, e.g. "2401_17151").
    """
    paper = _get_paper(paper_id)
    if not paper:
        return None
    # Add nugget count
    db = _get_db()
    count = db.execute(
        "SELECT COUNT(*) FROM nuggets WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0]
    paper["nugget_count"] = count
    return paper


@mcp.tool()
def kb_stats() -> dict:
    """Get knowledge base statistics: paper/nugget counts, type distribution, year distribution."""
    db = _get_db()
    n_papers = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_nuggets = db.execute("SELECT COUNT(*) FROM nuggets").fetchone()[0]
    types = db.execute(
        "SELECT type, COUNT(*) FROM nuggets GROUP BY type"
    ).fetchall()
    years = db.execute(
        "SELECT year, COUNT(*) FROM papers WHERE year IS NOT NULL GROUP BY year ORDER BY year"
    ).fetchall()
    sections = db.execute(
        "SELECT section, COUNT(*) FROM nuggets GROUP BY section ORDER BY COUNT(*) DESC"
    ).fetchall()
    return {
        "total_papers": n_papers,
        "total_nuggets": n_nuggets,
        "nuggets_by_type": {row[0]: row[1] for row in types},
        "papers_by_year": {row[0]: row[1] for row in years},
        "nuggets_by_section": {row[0]: row[1] for row in sections},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
