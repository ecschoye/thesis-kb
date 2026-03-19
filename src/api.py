"""FastAPI backend for thesis-kb web chat interface."""
import os, json, asyncio, re, time, math
import httpx
from collections import defaultdict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

from src.utils import load_config
from src.query import ThesisKB
from src.embed.embedder import make_embed_client, format_nugget_text
from src.rerank import rerank_nuggets
from src.log import get_logger

log = get_logger("api", "api.log")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_config_path = os.environ.get("THESIS_KB_CONFIG", "config.yaml")
_cfg = None
_kb: ThesisKB | None = None
_embed_client = None
_embed_model = None
_embed_instruction = ""
_collection = None
_executor = ThreadPoolExecutor(max_workers=4)
_bib_lookup: dict[str, str] = {}  # normalised key → bibtex cite key
_feedback_db = None  # separate SQLite connection for feedback

# Mode-specific retrieval strategies
MODE_ROUTING = {
    "background": {
        "n_retrieve": 25,
        "allowed_types": {"background", "method"},
        "preferred_sections": {"abstract", "introduction", "background", "related work"},
        "authority_boost": 1.5,  # stronger citation count boost
        "review_boost": 0.3,    # extra boost for review/survey papers
        "max_per_paper": 2,
    },
    "draft": {
        "n_retrieve": 25,
        "preferred_sections": None,  # all sections
        "authority_boost": 1.0,
        "review_boost": 0.0,
        "max_per_paper": 3,
    },
    "check": {
        "n_retrieve": 30,
        "preferred_sections": {"results", "experiments", "methods", "discussion"},
        "authority_boost": 1.2,
        "review_boost": 0.0,
        "max_per_paper": 4,  # more per paper for verification
    },
    "review": {
        "n_retrieve": 30,
        "preferred_sections": {"results", "experiments", "methods", "discussion"},
        "authority_boost": 1.2,
        "review_boost": 0.0,
        "max_per_paper": 4,
    },
    "compare": {
        "n_retrieve": 25,
        "preferred_sections": {"results", "experiments", "discussion", "abstract"},
        "authority_boost": 1.0,
        "review_boost": 0.0,
        "max_per_paper": 3,
    },
    "gaps": {
        "n_retrieve": 30,
        "preferred_sections": {"discussion", "conclusion", "limitations", "future work"},
        "authority_boost": 0.8,  # newer, less-cited papers may have gaps
        "review_boost": 0.2,
        "max_per_paper": 2,
    },
    "outline": {
        "n_retrieve": 30,
        "preferred_sections": None,  # cast wide net
        "authority_boost": 1.0,
        "review_boost": 0.1,
        "max_per_paper": 2,
    },
}

COMMANDS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "commands"
WEB_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
WEB_MODE_FILES = {"background": "background_web.md"}
MODE_FILES = {
    "survey": "survey.md",
    "gaps": "gaps.md",
    "compare": "compare.md",
    "check": "check.md",
    "cite": "cite.md",
    "review": "review.md",
    "draft": "draft.md",
    "outline": "outline.md",
    "background": "background.md",
}


def _parse_bib_file(bib_path: str) -> dict[str, str]:
    """Parse a .bib file and return lookup dicts mapping DOI/arXiv/title → bibtex key."""
    lookup: dict[str, str] = {}
    try:
        text = Path(bib_path).expanduser().read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return lookup

    # Split into entries
    entries = re.findall(r"@\w+\{([^,]+),([^@]*)", text, re.DOTALL)
    for cite_key, body in entries:
        cite_key = cite_key.strip()

        # Extract DOI
        doi_m = re.search(r"doi\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        if doi_m:
            doi = doi_m.group(1).strip().lower()
            lookup[f"doi:{doi}"] = cite_key

        # Extract arXiv ID from eprint, note, or url fields
        for field in ("eprint", "note", "url"):
            field_m = re.search(rf"{field}\s*=\s*\{{([^}}]+)\}}", body, re.IGNORECASE)
            if field_m:
                arxiv_m = re.search(r"(\d{4}\.\d{4,5})", field_m.group(1))
                if arxiv_m:
                    lookup[f"arxiv:{arxiv_m.group(1)}"] = cite_key
                    break

        # Extract title for fuzzy matching
        title_m = re.search(r"title\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        if title_m:
            title = re.sub(r"\s+", " ", title_m.group(1).strip().lower())
            lookup[f"title:{title}"] = cite_key

    return lookup


def _resolve_bibtex_key(arxiv_id: str, doi: str, title: str) -> str | None:
    """Look up a real bibtex key by arXiv ID, DOI, or normalised title."""
    if arxiv_id:
        key = _bib_lookup.get(f"arxiv:{arxiv_id}")
        if key:
            return key
    if doi:
        key = _bib_lookup.get(f"doi:{doi.lower()}")
        if key:
            return key
    if title:
        normalised = re.sub(r"\s+", " ", title.strip().lower())
        key = _bib_lookup.get(f"title:{normalised}")
        if key:
            return key
    return None


def _extract_cite_keys(text: str) -> list[str]:
    r"""Extract individual \cite{} keys from text, splitting comma-separated keys."""
    raw = re.findall(r"\\cite\{([^}]+)\}", text)
    keys = []
    for group in raw:
        for k in group.split(","):
            k = k.strip()
            if k:
                keys.append(k)
    return list(dict.fromkeys(keys))  # deduplicate, preserve order


def _resolve_cite_to_paper_id(key: str) -> str | None:
    """Resolve a cite key to a paper_id using the bib lookup and SQLite."""
    # Check if this key is a value in _bib_lookup (i.e., it's a real bib key)
    for lookup_key, bib_key in _bib_lookup.items():
        if bib_key == key:
            # Extract the identifier to search papers table
            if lookup_key.startswith("arxiv:"):
                arxiv_id = lookup_key[6:]
                row = _kb.db.execute(
                    "SELECT paper_id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
                ).fetchone()
                if row:
                    return row[0]
            elif lookup_key.startswith("doi:"):
                doi = lookup_key[4:]
                row = _kb.db.execute(
                    "SELECT paper_id FROM papers WHERE LOWER(doi) = ?", (doi,)
                ).fetchone()
                if row:
                    return row[0]
            elif lookup_key.startswith("title:"):
                title = lookup_key[6:]
                row = _kb.db.execute(
                    "SELECT paper_id FROM papers WHERE LOWER(title) = ?", (title,)
                ).fetchone()
                if row:
                    return row[0]

    # Fallback: parse author + year from key like "AuthorLast_YEAR:slug"
    base = key.split(":")[0] if ":" in key else key
    parts = base.split("_")
    author = parts[0] if parts else None
    year = None
    for p in parts:
        if p.isdigit() and len(p) == 4:
            year = int(p)
            break
    if author:
        query = "SELECT paper_id, title FROM papers WHERE authors LIKE ?"
        params: list = [f"%{author}%"]
        if year:
            query += " AND year = ?"
            params.append(year)
        query += " LIMIT 10"
        rows = _kb.db.execute(query, params).fetchall()
        if rows:
            # If there's a slug, score by title match
            if ":" in key:
                slug = key.split(":", 1)[1].lower().replace("-", " ").replace("_", " ")
                slug_words = set(slug.split())
                best_pid, best_score = None, 0
                for r in rows:
                    title_words = set(re.split(r"\W+", (r[1] or "").lower())) - {"", "a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "from", "by"}
                    if slug_words and title_words:
                        overlap = len(slug_words & title_words)
                        score = overlap / len(slug_words)
                        if score > best_score:
                            best_score = score
                            best_pid = r[0]
                if best_pid and best_score > 0.3:
                    return best_pid
            return rows[0][0]
    return None


def _init_feedback_db(kb_dir: str):
    """Initialize feedback SQLite database."""
    global _feedback_db
    import sqlite3
    fb_path = os.path.join(kb_dir, "feedback.db")
    _feedback_db = sqlite3.connect(fb_path, check_same_thread=False)
    _feedback_db.row_factory = sqlite3.Row
    _feedback_db.execute("""
        CREATE TABLE IF NOT EXISTS nugget_feedback (
            nugget_id TEXT,
            paper_id TEXT,
            rating INTEGER DEFAULT 0,
            query TEXT,
            mode TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (nugget_id, query)
        )
    """)
    _feedback_db.execute("""
        CREATE TABLE IF NOT EXISTS paper_feedback (
            paper_id TEXT PRIMARY KEY,
            rating INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    _feedback_db.commit()
    log.info("Feedback DB initialized at %s", fb_path)


def _init(config_path: str):
    global _cfg, _kb, _embed_client, _embed_model, _embed_instruction, _collection, _bib_lookup
    log.info("Initializing with config=%s", config_path)
    _cfg = load_config(config_path)
    _kb = ThesisKB(config_path)
    _embed_client, _embed_model = make_embed_client(_cfg)
    emb_cfg = _cfg.get("embed", {}).get("embedding", {})
    # Use query_instruction at query time if available
    _embed_instruction = emb_cfg.get("query_instruction", emb_cfg.get("instruction", ""))
    _collection = _kb.collection
    bib_path = _cfg.get("paths", {}).get("bib_file", "")
    _bib_lookup = _parse_bib_file(bib_path) if bib_path else {}
    _init_feedback_db(_cfg["paths"]["kb_dir"])
    stats = _kb.stats()
    log.info("KB loaded: %d papers, %d nuggets, %d bib keys",
             stats["total_papers"], stats["total_nuggets"], len(_bib_lookup))


def _shutdown():
    global _kb
    if _kb:
        _kb.close()
        _kb = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init(_config_path)
    yield
    _shutdown()


app = FastAPI(title="Thesis KB API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    n: int = 10

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    mode: str = "survey"
    n_variants: int = 6
    n_retrieve: int = 20
    n_context: int = 16
    model: str = "minimax/minimax-m2.5"
    latex_mode: bool = False
    year_min: int | None = None
    year_max: int | None = None
    excluded_nuggets: list[str] = []
    excluded_papers: list[str] = []
    type_filter: list[str] = []
    pinned_papers: list[str] = []
    max_per_paper: int = 3
    rerank: bool = True
    rerank_top_n: int = 60
    rerank_weight: float = 0.6

class RetrieveRequest(BaseModel):
    query: str
    mode: str = "background"
    n_context: int = 16
    n_variants: int = 6
    n_retrieve: int = 20
    model: str = "minimax/minimax-m2.5"
    year_min: int | None = None
    year_max: int | None = None
    type_filter: list[str] = []
    max_per_paper: int = 3
    rerank: bool = True
    rerank_top_n: int = 60
    rerank_weight: float = 0.6

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        stats = _kb.stats()
        ecfg = _cfg.get("embed", {})
        backend = ecfg.get("backend", "vllm")
        if backend == "ollama":
            emb_model = ecfg.get("ollama", {}).get("model", "")
        else:
            emb_model = ecfg.get("vllm", {}).get("model", "")
        ncfg = _cfg.get("nuggets", {})
        llm_backend = ncfg.get("backend", "vllm")
        if llm_backend == "ollama":
            llm_model = ncfg.get("ollama", {}).get("model", "")
        else:
            llm_model = ncfg.get("vllm", {}).get("model", "")
        return {
            "status": "ok",
            "nuggets": stats["total_nuggets"],
            "papers": stats["total_papers"],
            "embed_backend": backend,
            "embed_model": emb_model,
            "llm_model": llm_model,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "nuggets": 0,
            "papers": 0,
            "embed_backend": "",
            "embed_model": "",
            "llm_model": "",
        }


@app.get("/stats")
async def stats_endpoint():
    if not _kb:
        raise HTTPException(503, "KB not initialized")
    try:
        return _kb.stats()
    except Exception as e:
        raise HTTPException(500, str(e))


class FeedbackRequest(BaseModel):
    nugget_id: str
    paper_id: str
    rating: int  # +1 (helpful) or -1 (irrelevant)
    query: str = ""
    mode: str = ""


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Record user feedback on a retrieved nugget."""
    if not _feedback_db:
        raise HTTPException(503, "Feedback DB not initialized")
    _feedback_db.execute(
        """INSERT OR REPLACE INTO nugget_feedback (nugget_id, paper_id, rating, query, mode)
           VALUES (?, ?, ?, ?, ?)""",
        (req.nugget_id, req.paper_id, req.rating, req.query, req.mode),
    )
    # Aggregate paper-level rating
    _feedback_db.execute(
        """INSERT INTO paper_feedback (paper_id, rating, updated_at)
           VALUES (?, ?, datetime('now'))
           ON CONFLICT(paper_id) DO UPDATE SET
             rating = (SELECT SUM(rating) FROM nugget_feedback WHERE paper_id = ?),
             updated_at = datetime('now')""",
        (req.paper_id, req.rating, req.paper_id),
    )
    _feedback_db.commit()
    return {"status": "ok"}


@app.get("/api/feedback/papers")
async def get_paper_feedback():
    """Get aggregated paper feedback for boosting."""
    if not _feedback_db:
        return {}
    rows = _feedback_db.execute(
        "SELECT paper_id, rating FROM paper_feedback WHERE rating != 0"
    ).fetchall()
    return {r["paper_id"]: r["rating"] for r in rows}


@app.get("/api/feedback/nuggets")
async def get_nugget_feedback():
    """Get nugget-level feedback."""
    if not _feedback_db:
        return {}
    rows = _feedback_db.execute(
        "SELECT nugget_id, SUM(rating) as total FROM nugget_feedback GROUP BY nugget_id HAVING total != 0"
    ).fetchall()
    return {r["nugget_id"]: r["total"] for r in rows}


@app.get("/api/chunk/{paper_id}/{chunk_id}")
async def get_chunk(paper_id: str, chunk_id: int):
    """Load a parent chunk's full text from disk."""
    if not _kb:
        raise HTTPException(503, "KB not initialized")
    text = _kb.load_chunk(paper_id, chunk_id)
    if text is None:
        raise HTTPException(404, "Chunk not found")
    return {"paper_id": paper_id, "chunk_id": chunk_id, "text": text}


@app.get("/api/papers/search")
async def paper_search(q: str = "", limit: int = 10):
    """Search papers by title, author, or year. Each token must match at least one field."""
    if not _kb or not _kb.db:
        raise HTTPException(503, "KB not initialized")
    q = q.strip()
    if not q:
        return []
    tokens = q.split()
    conditions = []
    params = []
    for token in tokens:
        conditions.append(
            "(title LIKE ? OR authors LIKE ? OR CAST(year AS TEXT) LIKE ?)"
        )
        params.extend([f"%{token}%", f"%{token}%", f"%{token}%"])
    where = " AND ".join(conditions)
    rows = _kb.db.execute(
        f"SELECT paper_id, title, authors, year, arxiv_id FROM papers WHERE {where} LIMIT ?",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not _kb:
        raise HTTPException(503, "KB not initialized")
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            _executor, lambda: _kb.query(req.query, n_results=req.n)
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


async def _run_retrieval(
    query: str,
    mode: str,
    n_variants: int,
    n_retrieve: int,
    n_context: int,
    model: str,
    type_filter: list[str],
    max_per_paper: int,
    year_min: int | None,
    year_max: int | None,
    rerank: bool,
    rerank_top_n: int,
    rerank_weight: float,
    excluded_nuggets: list[str] | None = None,
    excluded_papers: list[str] | None = None,
    pinned_papers: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Run the full retrieval pipeline and return (enriched_nuggets, variants).

    Shared by /retrieve and /chat endpoints.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY not set")

    t0 = time.time()
    loop = asyncio.get_event_loop()
    last_msg = query

    expand_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
    )

    def _expand():
        # Mode-specific expansion prompts
        if mode == "draft":
            system_content = (
                f"You are a search query expander for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a topic to draft a thesis paragraph about, generate {n_variants} search queries. "
                f"Focus on finding concrete evidence for writing:\n"
                f"- 3 queries targeting \"method\" nuggets (architectures, algorithms, how things work)\n"
                f"- 2 queries targeting \"result\" nuggets (quantitative performance, benchmarks)\n"
                f"- 1 query targeting \"background\" nuggets (definitions, context)\n"
                f'{"- Fill extra slots with method or result types" if n_variants > 6 else ""}\n\n'
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                f"No explanation."
            )
        elif mode == "outline":
            system_content = (
                f"You are a search query expander for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a thesis section to outline, generate {n_variants} broad search queries "
                f"spanning the full scope of the section. Cast a wide net:\n"
                f"- 1 query targeting \"background\" nuggets (foundational concepts)\n"
                f"- 2 queries targeting \"method\" nuggets (different approaches/techniques)\n"
                f"- 1 query targeting \"result\" nuggets (key findings)\n"
                f"- 1 query targeting \"comparison\" nuggets (trade-offs, alternatives)\n"
                f"- 1 query targeting \"limitation\" nuggets (open problems, gaps)\n"
                f'{"- Fill extra slots with any type" if n_variants > 6 else ""}\n\n'
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                f"No explanation."
            )
        elif mode in ("review", "check"):
            system_content = (
                f"You are a claim extractor for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a piece of thesis text, extract the {n_variants} most important "
                f"factual claims and convert each into a search query to verify it against "
                f"the knowledge base. Focus on:\n"
                f"- Technical claims about how sensors/methods work (target_type: \"method\")\n"
                f"- Quantitative claims with numbers or performance figures (target_type: \"result\")\n"
                f"- Comparative claims (X is better/worse than Y) (target_type: \"comparison\")\n"
                f"- Claims about limitations or challenges (target_type: \"limitation\")\n"
                f"- Definitional or background claims (target_type: \"background\")\n\n"
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                f"Each query should be a direct search for evidence supporting or contradicting the claim.\n"
                f"No explanation."
            )
        elif mode == "background":
            system_content = (
                f"You are a search query expander for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a topic for a neutral background paragraph, generate {n_variants} search queries "
                f"that find FACTUAL, DEFINITIONAL content only. Focus on:\n"
                f"- How the technology/concept works (mechanisms, principles, architecture)\n"
                f"- Measurable properties (specifications, parameters, characteristics)\n"
                f"- Definitions and foundational concepts\n\n"
                f"DO NOT generate queries about:\n"
                f"- Comparisons between technologies\n"
                f"- Limitations, weaknesses, or drawbacks\n"
                f"- Advantages or motivations for alternative approaches\n\n"
                f"Use this distribution:\n"
                f"- {max(n_variants // 2, 1)} queries targeting \"background\" nuggets (definitions, context, foundational concepts)\n"
                f"- {n_variants - max(n_variants // 2, 1)} queries targeting \"method\" nuggets (mechanisms, how things work)\n\n"
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "background"}}, ...]\n'
                f"No explanation."
            )
        else:
            system_content = (
                f"You are a search query expander for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a user question, generate {n_variants} search queries. "
                f"Each query MUST target a specific nugget type. Use this distribution:\n"
                f"- 2 queries targeting \"method\" nuggets (how things work, architectures, algorithms)\n"
                f"- 1 query targeting \"result\" nuggets (quantitative performance, benchmarks, metrics)\n"
                f"- 1 query targeting \"comparison\" nuggets (X vs Y, trade-offs, advantages/disadvantages)\n"
                f"- 1 query targeting \"limitation\" nuggets (weaknesses, open problems, failure cases)\n"
                f"- 1 query targeting \"background\" nuggets (definitions, context, foundational concepts)\n"
                f'{"- Fill extra slots with any type" if n_variants > 6 else ""}\n\n'
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                f"No explanation."
            )

        resp = expand_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": last_msg},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()
        return json.loads(text)

    try:
        raw = await loop.run_in_executor(_executor, _expand)
        if not isinstance(raw, list):
            raw = [last_msg]
        if raw and isinstance(raw[0], dict):
            variants = [item.get("query", str(item)) for item in raw]
            variant_types = [item.get("target_type") for item in raw]
        else:
            variants = [str(item) for item in raw]
            variant_types = [None] * len(variants)
    except Exception:
        if mode == "background":
            variants = [
                last_msg,
                f"definition and principles of {last_msg}",
                f"how {last_msg} works mechanism",
                f"properties and characteristics of {last_msg}",
            ]
            variant_types = ["background", "background", "method", "method"]
        else:
            variants = [
                last_msg,
                f"methods and architectures for {last_msg}",
                f"experimental results and benchmarks for {last_msg}",
                f"limitations and challenges of {last_msg}",
            ]
            variant_types = [None, "method", "result", "limitation"]

    log.debug("Expanded to %d variants: %s", len(variants), variants)

    # --- Step 2: Embed each variant ---
    def _embed_one(text):
        formatted = format_nugget_text(
            {"question": text, "answer": ""}, _embed_instruction
        )
        kwargs = {"model": _embed_model, "input": [formatted]}
        emb_cfg = _cfg.get("embed", {}).get("embedding", {})
        dims = emb_cfg.get("dimensions")
        if dims:
            kwargs["dimensions"] = dims
        resp = _embed_client.embeddings.create(**kwargs)
        return resp.data[0].embedding

    embed_tasks = [
        loop.run_in_executor(_executor, _embed_one, v) for v in variants
    ]
    embeddings = await asyncio.gather(*embed_tasks)

    # --- Step 3: Multi-vector ChromaDB retrieval + BM25 ---
    routing = MODE_ROUTING.get(mode, {})
    effective_n_retrieve = routing.get("n_retrieve", n_retrieve)

    def _retrieve(vec, target_type=None):
        kwargs = {"query_embeddings": [vec], "n_results": effective_n_retrieve}
        if type_filter:
            if len(type_filter) == 1:
                kwargs["where"] = {"type": type_filter[0]}
            else:
                kwargs["where"] = {"type": {"$in": type_filter}}
        elif target_type:
            kwargs["where"] = {"type": target_type}
        return _collection.query(**kwargs)

    retrieve_tasks = [
        loop.run_in_executor(_executor, _retrieve, emb, vtype)
        for emb, vtype in zip(embeddings, variant_types)
    ]

    # BM25 retrieval (parallel with vector retrieval)
    def _bm25_search():
        bm25_results = []
        for v in variants:
            bm25_results.extend(_kb.bm25_search(v, n_results=effective_n_retrieve))
        return bm25_results

    bm25_task = loop.run_in_executor(_executor, _bm25_search)
    all_results = await asyncio.gather(*retrieve_tasks)
    bm25_raw = await bm25_task

    # --- Step 4: Reciprocal Rank Fusion ---
    rrf_scores: dict[str, float] = {}
    overlap_count: dict[str, int] = {}
    matched_queries: dict[str, list[int]] = {}
    nugget_data: dict[str, dict] = {}

    for qi, result_set in enumerate(all_results):
        ids = result_set["ids"][0]
        metas = result_set["metadatas"][0]
        docs = result_set["documents"][0]
        dists = result_set["distances"][0]
        for rank, (nid, meta, doc, dist) in enumerate(
            zip(ids, metas, docs, dists)
        ):
            rrf_scores[nid] = rrf_scores.get(nid, 0) + 1 / (rank + 30)
            overlap_count[nid] = overlap_count.get(nid, 0) + 1
            matched_queries.setdefault(nid, []).append(qi)
            if nid not in nugget_data:
                nugget_data[nid] = {
                    "nugget_id": nid,
                    "type": meta.get("type", ""),
                    "confidence": meta.get("confidence", ""),
                    "section": meta.get("section", ""),
                    "document": doc,
                    "paper_id": meta.get("paper_id", ""),
                    "distance": round(dist, 4),
                    "thesis_relevance": meta.get("thesis_relevance", 3),
                }

    # Merge BM25 results into RRF
    bm25_seen = {}
    for nid, score in bm25_raw:
        if nid not in bm25_seen or score > bm25_seen[nid]:
            bm25_seen[nid] = score
    bm25_ranked = sorted(bm25_seen.keys(), key=lambda x: bm25_seen[x], reverse=True)
    for rank, nid in enumerate(bm25_ranked[:effective_n_retrieve]):
        rrf_scores[nid] = rrf_scores.get(nid, 0) + 1 / (rank + 30)
        overlap_count[nid] = overlap_count.get(nid, 0) + 1
        matched_queries.setdefault(nid, []).append(len(variants))
        if nid not in nugget_data:
            row = _kb.db.execute(
                "SELECT * FROM nuggets WHERE nugget_id = ?", (nid,)
            ).fetchone()
            if row:
                r = dict(row)
                nugget_data[nid] = {
                    "nugget_id": nid,
                    "type": r.get("type", ""),
                    "confidence": r.get("confidence", ""),
                    "section": r.get("section", ""),
                    "document": f"Q: {r.get('question', '')}\nA: {r.get('answer', '')}",
                    "paper_id": r.get("paper_id", ""),
                    "distance": 0.0,
                    "thesis_relevance": r.get("thesis_relevance", 3),
                    "source_chunk": r.get("source_chunk"),
                }
    log.debug("BM25 contributed %d unique nuggets (%d new)",
              len(bm25_ranked), len([n for n in bm25_ranked if n not in nugget_data]))

    # Thesis relevance boosting
    for nid in rrf_scores:
        relevance = nugget_data[nid].get("thesis_relevance", 3)
        try:
            relevance = int(relevance)
        except (ValueError, TypeError):
            relevance = 3
        rrf_scores[nid] *= 1.0 + (relevance - 3) * 0.2

    # Section-aware boosting (mode-specific)
    preferred_sections = routing.get("preferred_sections")
    if preferred_sections:
        for nid in rrf_scores:
            section = nugget_data[nid].get("section", "").lower()
            if any(ps in section for ps in preferred_sections):
                rrf_scores[nid] *= 1.15

    # Feedback boosting (from user's prior ratings)
    if _feedback_db:
        try:
            paper_fb = {r[0]: r[1] for r in _feedback_db.execute(
                "SELECT paper_id, rating FROM paper_feedback WHERE rating != 0"
            ).fetchall()}
            nugget_fb = {r[0]: r[1] for r in _feedback_db.execute(
                "SELECT nugget_id, SUM(rating) FROM nugget_feedback GROUP BY nugget_id HAVING SUM(rating) != 0"
            ).fetchall()}
            for nid in rrf_scores:
                nfb = nugget_fb.get(nid, 0)
                if nfb > 0:
                    rrf_scores[nid] *= 1.3
                elif nfb < 0:
                    rrf_scores[nid] *= 0.5
                pid = nugget_data[nid]["paper_id"]
                pfb = paper_fb.get(pid, 0)
                if pfb > 0:
                    rrf_scores[nid] *= 1.0 + min(pfb * 0.05, 0.3)
                elif pfb < 0:
                    rrf_scores[nid] *= max(1.0 + pfb * 0.1, 0.3)
        except Exception as e:
            log.warning("Feedback boosting failed: %s", e)

    # Depth-of-coverage: boost papers where many nuggets matched
    paper_hits: dict[str, set[str]] = defaultdict(set)
    for nid in rrf_scores:
        paper_hits[nugget_data[nid]["paper_id"]].add(nid)

    paper_depth: dict[str, float] = {}
    for pid, nids in paper_hits.items():
        total = _kb.paper_nugget_count(pid) or 1
        raw_hits = len(nids)
        hit_fraction = min(raw_hits / total, 1.0)
        abs_boost = min(math.log2(max(raw_hits, 1)) / 5, 0.3)
        paper_depth[pid] = 1.0 + hit_fraction * 0.3 + abs_boost

    for nid in rrf_scores:
        pid = nugget_data[nid]["paper_id"]
        rrf_scores[nid] *= paper_depth.get(pid, 1.0)

    # Paper authority boost (also penalizes papers with missing metadata)
    auth_scale = routing.get("authority_boost", 1.0)
    review_boost_val = routing.get("review_boost", 0.0)
    _authority_cache: dict[str, float] = {}
    for nid in rrf_scores:
        pid = nugget_data[nid]["paper_id"]
        if pid not in _authority_cache:
            paper = _kb._get_paper(pid)
            if paper:
                # Penalize papers with broken metadata (no year, placeholder title)
                has_year = paper.get("year") is not None
                title = paper.get("title", "")
                has_real_title = bool(title) and not re.match(r"^\d{4}[\s_]\d{4,5}$", title)
                if not has_year or not has_real_title:
                    _authority_cache[pid] = 0.3  # heavy penalty
                    continue
                cc = paper.get("citation_count") or 0
                icc = paper.get("influential_citation_count") or 0
                ptype = (paper.get("paper_type") or "").lower()
                is_review = "review" in ptype or "survey" in ptype
                authority = 1.0 + auth_scale * 0.1 * min(math.log10(max(cc, 1)), 4)
                if icc > 10:
                    authority += 0.1
                if is_review:
                    authority += review_boost_val
                _authority_cache[pid] = authority
            else:
                _authority_cache[pid] = 1.0
        rrf_scores[nid] *= _authority_cache[pid]

    # Cross-encoder reranking
    if rerank:
        pre_rerank = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        rrf_scores = await loop.run_in_executor(
            _executor,
            lambda: rerank_nuggets(
                query=last_msg,
                nugget_ids=pre_rerank,
                nugget_data=nugget_data,
                rrf_scores=rrf_scores,
                top_n=rerank_top_n,
                blend_weight=rerank_weight,
            ),
        )
        log.info("Reranked top %d candidates (weight=%.1f)", rerank_top_n, rerank_weight)

    # Sort by RRF score
    ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Exclude rejected nuggets and papers
    _excluded_nuggets = set(excluded_nuggets or [])
    _excluded_papers = set(excluded_papers or [])
    if _excluded_nuggets or _excluded_papers:
        ranked = [
            nid for nid in ranked
            if nid not in _excluded_nuggets
            and nugget_data[nid]["paper_id"] not in _excluded_papers
        ]

    # Mode-specific type restrictions
    allowed_types = routing.get("allowed_types")
    if allowed_types:
        ranked = [nid for nid in ranked if nugget_data[nid]["type"] in allowed_types]

    # Type-balanced + paper-diverse selection
    by_type = defaultdict(list)
    for nid in ranked:
        ntype = nugget_data[nid]["type"]
        by_type[ntype].append(nid)

    selected = []
    paper_counts: dict[str, int] = defaultdict(int)
    max_pp = routing.get("max_per_paper", max_per_paper)

    def _can_add(nid):
        return paper_counts[nugget_data[nid]["paper_id"]] < max_pp

    sorted_types = sorted(by_type.keys(), key=lambda t: rrf_scores[by_type[t][0]], reverse=True)
    for t in sorted_types:
        added = 0
        for nid in by_type[t]:
            if added >= 2 or len(selected) >= n_context:
                break
            if nid not in selected and _can_add(nid):
                selected.append(nid)
                paper_counts[nugget_data[nid]["paper_id"]] += 1
                added += 1
        if len(selected) >= n_context:
            break

    for nid in ranked:
        if len(selected) >= n_context:
            break
        if nid not in selected and _can_add(nid):
            selected.append(nid)
            paper_counts[nugget_data[nid]["paper_id"]] += 1

    for nid in ranked:
        if len(selected) >= n_context:
            break
        if nid not in selected:
            selected.append(nid)
            paper_counts[nugget_data[nid]["paper_id"]] += 1

    top_ids = selected

    # Inject pinned paper nuggets
    pinned_nids = []
    if pinned_papers:
        for pid in pinned_papers:
            rows = _kb.get_paper_nuggets(pid)
            if not rows:
                continue
            rows.sort(key=lambda r: r.get("thesis_relevance", 3), reverse=True)
            type_counts_pin = defaultdict(int)
            paper_nuggets = []
            for r in rows:
                ntype = r.get("type", "")
                if type_counts_pin[ntype] >= 4:
                    continue
                type_counts_pin[ntype] += 1
                paper_nuggets.append(r)
                if len(paper_nuggets) >= 15:
                    break
            for r in paper_nuggets:
                nid = r["nugget_id"]
                if nid in nugget_data or nid in pinned_nids:
                    continue
                nugget_data[nid] = {
                    "nugget_id": nid,
                    "type": r.get("type", ""),
                    "confidence": r.get("confidence", ""),
                    "section": r.get("section", ""),
                    "document": f"Q: {r.get('question', '')}\nA: {r.get('answer', '')}",
                    "paper_id": pid,
                    "distance": 0.0,
                    "thesis_relevance": r.get("thesis_relevance", 3),
                    "pinned": True,
                }
                rrf_scores[nid] = 1.0
                overlap_count[nid] = 0
                matched_queries[nid] = []
                pinned_nids.append(nid)
        if len(pinned_nids) > 20:
            pinned_nids.sort(key=lambda n: nugget_data[n].get("thesis_relevance", 3), reverse=True)
            pinned_nids = pinned_nids[:20]
        top_ids = [nid for nid in top_ids if nid not in pinned_nids]
        max_rrf = max(4, n_context - len(pinned_nids))
        top_ids = pinned_nids + top_ids[:max_rrf]
        log.info("Pinned %d nuggets from %d papers", len(pinned_nids), len(pinned_papers))

    log.info("Retrieved %d unique nuggets, selected %d after RRF+balancing",
             len(nugget_data), len(top_ids))

    # Enrich with paper metadata from SQLite
    bibtex_key_counts: dict[str, int] = {}
    paper_id_to_bibtex_key: dict[str, str] = {}  # cache: same paper → same key

    def _make_bibtex_key(authors: str, year, arxiv_id: str = "", title: str = "", paper_id: str = "") -> str:
        # Return cached key if same paper_id already processed
        if paper_id and paper_id in paper_id_to_bibtex_key:
            return paper_id_to_bibtex_key[paper_id]
        # Extract surname
        surname = ""
        if authors and authors not in ("[]", ""):
            first = authors.split(",")[0].strip()
            parts = first.split()
            if parts:
                surname = parts[-1]
                surname = "".join(c for c in surname if c.isalpha())
        if not surname and arxiv_id:
            # Fallback: use arXiv ID as key base
            key = f"arXiv_{arxiv_id.replace('.', '_')}"
            if paper_id:
                paper_id_to_bibtex_key[paper_id] = key
            return key
        if not surname:
            surname = "Unknown"
        yr = str(year) if year else "XXXX"
        # Build slug from title
        slug = ""
        if title and title != title.replace(" ", "_"):  # not just a paper_id
            words = re.split(r"\W+", title.lower())
            stop = {"a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "from", "by", "is", "are", "was", "were"}
            slug_words = [w for w in words if w and w not in stop][:4]
            if slug_words:
                slug = "-".join(slug_words)
        base = f"{surname}_{yr}"
        if slug:
            base = f"{base}:{slug}"
        count = bibtex_key_counts.get(base, 0)
        bibtex_key_counts[base] = count + 1
        key = base if count == 0 else f"{base}-{chr(ord('a') + count - 1)}"
        if paper_id:
            paper_id_to_bibtex_key[paper_id] = key
        return key

    def _enrich(nid):
        nd = nugget_data[nid]
        paper = _kb._get_paper(nd["paper_id"])
        nd["paper_title"] = paper.get("title", "") if paper else ""
        nd["paper_year"] = paper.get("year") if paper else None
        nd["paper_authors"] = paper.get("authors", "") if paper else ""
        nd["arxiv_id"] = paper.get("arxiv_id", "") if paper else ""
        nd["doi"] = paper.get("doi", "") if paper else ""
        real_key = _resolve_bibtex_key(nd["arxiv_id"], nd["doi"], nd["paper_title"])
        nd["bibtex_key"] = real_key if real_key else _make_bibtex_key(
            nd["paper_authors"], nd["paper_year"], nd["arxiv_id"], nd["paper_title"],
            paper_id=nd["paper_id"]
        )
        nd["bib_status"] = "real" if real_key else "generated"
        nd["rrf_score"] = round(rrf_scores[nid], 4)
        nd["overlap_count"] = overlap_count[nid]
        nd["matched_queries"] = matched_queries.get(nid, [])
        if "source_chunk" not in nd:
            row = _kb.db.execute(
                "SELECT source_chunk FROM nuggets WHERE nugget_id = ?", (nid,)
            ).fetchone()
            nd["source_chunk"] = row[0] if row else None
        return nd

    top_nuggets = [_enrich(nid) for nid in top_ids]

    # Year filtering (post-enrichment since we need paper_year)
    if year_min or year_max:
        top_nuggets = [
            n for n in top_nuggets
            if n["paper_year"] is not None
            and (year_min is None or n["paper_year"] >= year_min)
            and (year_max is None or n["paper_year"] <= year_max)
        ]

    log.info("Retrieval complete: %d nuggets, %.1fs", len(top_nuggets), time.time() - t0)
    return top_nuggets, variants


@app.post("/retrieve")
async def retrieve_endpoint(req: RetrieveRequest):
    """Run the full retrieval pipeline and return ranked nuggets as JSON.

    Same pipeline as /chat (query expansion, RRF, reranking, diversity caps)
    but returns sources directly without LLM generation.
    """
    if not _kb:
        raise HTTPException(503, "KB not initialized")
    try:
        top_nuggets, variants = await _run_retrieval(
            query=req.query,
            mode=req.mode,
            n_variants=req.n_variants,
            n_retrieve=req.n_retrieve,
            n_context=req.n_context,
            model=req.model,
            type_filter=req.type_filter,
            max_per_paper=req.max_per_paper,
            year_min=req.year_min,
            year_max=req.year_max,
            rerank=req.rerank,
            rerank_top_n=req.rerank_top_n,
            rerank_weight=req.rerank_weight,
        )
        results = []
        for n in top_nuggets:
            results.append({
                "nugget_id": n["nugget_id"],
                "paper_id": n["paper_id"],
                "paper_title": n["paper_title"],
                "paper_year": n["paper_year"],
                "paper_authors": n["paper_authors"],
                "arxiv_id": n["arxiv_id"],
                "type": n["type"],
                "section": n["section"],
                "document": n["document"],
                "rrf_score": float(n["rrf_score"]),
                "overlap_count": int(n["overlap_count"]),
                "matched_queries": [int(q) for q in n["matched_queries"]],
                "n_variants": len(variants),
                "confidence": n["confidence"],
                "thesis_relevance": int(n.get("thesis_relevance", 3)),
                "bibtex_key": n["bibtex_key"],
                "bib_status": n.get("bib_status", "generated"),
                "pinned": bool(n.get("pinned", False)),
                "distance": float(n.get("distance", 0.0)),
            })
        return {"results": results, "variants": variants}
    except HTTPException:
        raise
    except Exception as e:
        log.error("Retrieve error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not _kb:
        raise HTTPException(503, "KB not initialized")

    try:
        last_msg = req.messages[-1].content if req.messages else ""
        log.info("Chat request: mode=%s model=%s query=%s",
                 req.mode, req.model, last_msg[:100])

        # Auto-extract \cite{} keys and pin cited papers
        pinned_from_cites = []
        if req.mode in ("background", "draft", "review", "check"):
            cite_keys = _extract_cite_keys(last_msg)
            if cite_keys:
                for ck in cite_keys:
                    pid = _resolve_cite_to_paper_id(ck)
                    if pid:
                        pinned_from_cites.append(pid)
                        log.info("Cite key '%s' resolved to paper_id=%s", ck, pid)
                    else:
                        log.warning("Cite key '%s' could not be resolved", ck)
                if pinned_from_cites:
                    # Merge with explicit pinned_papers, dedup
                    existing = set(req.pinned_papers)
                    for pid in pinned_from_cites:
                        if pid not in existing:
                            req.pinned_papers.append(pid)
                            existing.add(pid)
                    log.info("Auto-pinned %d papers from cite keys", len(pinned_from_cites))

        top_nuggets, variants = await _run_retrieval(
            query=last_msg,
            mode=req.mode,
            n_variants=req.n_variants,
            n_retrieve=req.n_retrieve,
            n_context=req.n_context,
            model=req.model,
            type_filter=req.type_filter,
            max_per_paper=req.max_per_paper,
            year_min=req.year_min,
            year_max=req.year_max,
            rerank=req.rerank,
            rerank_top_n=req.rerank_top_n,
            rerank_weight=req.rerank_weight,
            excluded_nuggets=req.excluded_nuggets,
            excluded_papers=req.excluded_papers,
            pinned_papers=req.pinned_papers,
        )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(500, "OPENROUTER_API_KEY not set")

        t0 = time.time()
        loop = asyncio.get_event_loop()
        # --- Step 5: Build context + call OpenRouter ---
        web_file = WEB_MODE_FILES.get(req.mode, "")
        web_path = WEB_PROMPTS_DIR / web_file if web_file else None
        if web_path and web_path.exists():
            mode_prompt = web_path.read_text()
            log.info("Using web prompt: %s", web_file)
        else:
            mode_file = MODE_FILES.get(req.mode, "survey.md")
            mode_path = COMMANDS_DIR / mode_file
            if mode_path.exists():
                mode_prompt = mode_path.read_text()
            else:
                mode_prompt = (COMMANDS_DIR / "survey.md").read_text()

        sources_xml = "<sources>\n"
        for n in top_nuggets:
            pinned_attr = ' pinned="true"' if n.get("pinned") else ""
            bib_status = n.get("bib_status", "generated")
            sources_xml += (
                f'<source bibtex_key="{n["bibtex_key"]}" bib_status="{bib_status}" '
                f'paper="{n["paper_title"]}" year="{n["paper_year"]}" type="{n["type"]}" '
                f'overlap="{n["overlap_count"]}/{len(variants)}"{pinned_attr}>\n'
                f'{n["document"]}\n'
                f"</source>\n"
            )
        sources_xml += "</sources>\n\n"

        system_prompt = sources_xml + mode_prompt

        if req.latex_mode or req.mode == "background":
            key_table = "\n".join(
                f"  \\cite{{{n['bibtex_key']}}} → {n['paper_title']} ({n['paper_year']})"
                for n in top_nuggets
            )
            system_prompt += (
                "\n\n## LaTeX Citation Mode\n"
                "Use \\cite{bibtex_key} for ALL citations instead of (arXiv:XXXX, YEAR).\n"
                "Use the bibtex_key attribute from each <source> tag.\n"
                "Citation key mapping:\n" + key_table + "\n"
            )

        chat_messages = [{"role": "system", "content": system_prompt}]
        for m in req.messages:
            chat_messages.append({"role": m.role, "content": m.content})

        or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        )

        # --- Step 6: Stream response ---
        log.info("Streaming response: %d context nuggets, %.1fs retrieval",
                 len(top_nuggets), time.time() - t0)

        async def _stream():
            max_retries = 2
            accumulated = ""

            for attempt in range(max_retries):
                def _call_openrouter(msgs=chat_messages, tokens=4000):
                    return or_client.chat.completions.create(
                        model=req.model,
                        messages=msgs,
                        stream=True,
                        max_tokens=tokens,
                        temperature=0.3,
                    )

                # On retry, append partial content so LLM continues from where it stopped
                if attempt > 0 and accumulated:
                    log.info("Stream retry %d, continuing from %d chars", attempt, len(accumulated))
                    retry_messages = chat_messages + [
                        {"role": "assistant", "content": accumulated}
                    ]
                    remaining_tokens = max(1000, 4000 - len(accumulated) // 3)
                    stream = await loop.run_in_executor(
                        _executor, lambda: _call_openrouter(retry_messages, remaining_tokens)
                    )
                else:
                    stream = await loop.run_in_executor(_executor, _call_openrouter)

                try:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            delta = chunk.choices[0].delta.content
                            accumulated += delta
                            yield f"data: {json.dumps({'type': 'delta', 'content': delta})}\n\n"
                    break  # success — exit retry loop
                except Exception as e:
                    log.warning("Stream error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                    if attempt == max_retries - 1:
                        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                        return  # stop streaming — don't send sources after error
                    # otherwise retry

            # Send sources
            sources_payload = []
            for n in top_nuggets:
                sources_payload.append({
                    "nugget_id": n["nugget_id"],
                    "paper_id": n["paper_id"],
                    "paper_title": n["paper_title"],
                    "paper_year": n["paper_year"],
                    "paper_authors": n["paper_authors"],
                    "arxiv_id": n["arxiv_id"],
                    "type": n["type"],
                    "section": n["section"],
                    "document": n["document"],
                    "rrf_score": n["rrf_score"],
                    "overlap_count": n["overlap_count"],
                    "matched_queries": n["matched_queries"],
                    "n_variants": len(variants),
                    "confidence": n["confidence"],
                    "thesis_relevance": n.get("thesis_relevance", 3),
                    "bibtex_key": n["bibtex_key"],
                    "bib_status": n.get("bib_status", "generated"),
                    "pinned": n.get("pinned", False),
                    "source_chunk": n.get("source_chunk"),
                })
            yield f"data: {json.dumps({'type': 'sources', 'nuggets': sources_payload, 'variants': variants})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        log.error("Chat error: %s", e, exc_info=True)
        err_str = str(e).lower()
        if "connection" in err_str or "refused" in err_str:
            raise HTTPException(
                503,
                "Embedding server offline. Start Ollama with: ollama serve",
            )
        raise HTTPException(500, str(e))


def set_config(path: str):
    """Allow startup script to override config path before lifespan runs."""
    global _config_path
    _config_path = path


if __name__ == "__main__":
    import argparse, uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-p", "--port", type=int, default=8001)
    args = ap.parse_args()
    os.environ["THESIS_KB_CONFIG"] = args.config
    _config_path = args.config
    uvicorn.run("src.api:app", host="0.0.0.0", port=args.port, reload=False)
