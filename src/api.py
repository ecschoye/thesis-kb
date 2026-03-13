"""FastAPI backend for thesis-kb web chat interface."""
import os, json, asyncio, re, time, math
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
    model: str = "google/gemini-3-flash-preview"
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


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not _kb:
        raise HTTPException(503, "KB not initialized")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY not set")

    try:
        t0 = time.time()
        loop = asyncio.get_event_loop()

        # --- Step 1: Query expansion via OpenRouter ---
        last_msg = req.messages[-1].content if req.messages else ""
        log.info("Chat request: mode=%s model=%s query=%s",
                 req.mode, req.model, last_msg[:100])
        expand_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        def _expand():
            # Mode-specific expansion prompts
            if req.mode == "draft":
                system_content = (
                    f"You are a search query expander for an academic knowledge base about "
                    f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                    f"Given a topic to draft a thesis paragraph about, generate {req.n_variants} search queries. "
                    f"Focus on finding concrete evidence for writing:\n"
                    f"- 3 queries targeting \"method\" nuggets (architectures, algorithms, how things work)\n"
                    f"- 2 queries targeting \"result\" nuggets (quantitative performance, benchmarks)\n"
                    f"- 1 query targeting \"background\" nuggets (definitions, context)\n"
                    f'{"- Fill extra slots with method or result types" if req.n_variants > 6 else ""}\n\n'
                    f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                    f"No explanation."
                )
            elif req.mode == "outline":
                system_content = (
                    f"You are a search query expander for an academic knowledge base about "
                    f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                    f"Given a thesis section to outline, generate {req.n_variants} broad search queries "
                    f"spanning the full scope of the section. Cast a wide net:\n"
                    f"- 1 query targeting \"background\" nuggets (foundational concepts)\n"
                    f"- 2 queries targeting \"method\" nuggets (different approaches/techniques)\n"
                    f"- 1 query targeting \"result\" nuggets (key findings)\n"
                    f"- 1 query targeting \"comparison\" nuggets (trade-offs, alternatives)\n"
                    f"- 1 query targeting \"limitation\" nuggets (open problems, gaps)\n"
                    f'{"- Fill extra slots with any type" if req.n_variants > 6 else ""}\n\n'
                    f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                    f"No explanation."
                )
            elif req.mode in ("review", "check"):
                system_content = (
                    f"You are a claim extractor for an academic knowledge base about "
                    f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                    f"Given a piece of thesis text, extract the {req.n_variants} most important "
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
            elif req.mode == "background":
                system_content = (
                    f"You are a search query expander for an academic knowledge base about "
                    f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                    f"Given a topic for a neutral background paragraph, generate {req.n_variants} search queries "
                    f"that find FACTUAL, DEFINITIONAL content only. Focus on:\n"
                    f"- How the technology/concept works (mechanisms, principles, architecture)\n"
                    f"- Measurable properties (specifications, parameters, characteristics)\n"
                    f"- Definitions and foundational concepts\n\n"
                    f"DO NOT generate queries about:\n"
                    f"- Comparisons between technologies\n"
                    f"- Limitations, weaknesses, or drawbacks\n"
                    f"- Advantages or motivations for alternative approaches\n\n"
                    f"Use this distribution:\n"
                    f"- {max(req.n_variants // 2, 1)} queries targeting \"background\" nuggets (definitions, context, foundational concepts)\n"
                    f"- {req.n_variants - max(req.n_variants // 2, 1)} queries targeting \"method\" nuggets (mechanisms, how things work)\n\n"
                    f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "background"}}, ...]\n'
                    f"No explanation."
                )
            else:
                system_content = (
                    f"You are a search query expander for an academic knowledge base about "
                    f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                    f"Given a user question, generate {req.n_variants} search queries. "
                    f"Each query MUST target a specific nugget type. Use this distribution:\n"
                    f"- 2 queries targeting \"method\" nuggets (how things work, architectures, algorithms)\n"
                    f"- 1 query targeting \"result\" nuggets (quantitative performance, benchmarks, metrics)\n"
                    f"- 1 query targeting \"comparison\" nuggets (X vs Y, trade-offs, advantages/disadvantages)\n"
                    f"- 1 query targeting \"limitation\" nuggets (weaknesses, open problems, failure cases)\n"
                    f"- 1 query targeting \"background\" nuggets (definitions, context, foundational concepts)\n"
                    f'{"- Fill extra slots with any type" if req.n_variants > 6 else ""}\n\n'
                    f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                    f"No explanation."
                )

            resp = expand_client.chat.completions.create(
                model=req.model,
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
            # Parse structured format [{query, target_type}, ...] or plain strings
            if raw and isinstance(raw[0], dict):
                variants = [item.get("query", str(item)) for item in raw]
                variant_types = [item.get("target_type") for item in raw]
            else:
                variants = [str(item) for item in raw]
                variant_types = [None] * len(variants)
        except Exception:
            # Deterministic fallback with type diversity
            if req.mode == "background":
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
        routing = MODE_ROUTING.get(req.mode, {})
        effective_n_retrieve = routing.get("n_retrieve", req.n_retrieve)

        def _retrieve(vec, target_type=None):
            kwargs = {"query_embeddings": [vec], "n_results": effective_n_retrieve}
            # Global type filter overrides per-variant type targeting
            if req.type_filter:
                if len(req.type_filter) == 1:
                    kwargs["where"] = {"type": req.type_filter[0]}
                else:
                    kwargs["where"] = {"type": {"$in": req.type_filter}}
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

        # --- Step 4a: Merge BM25 results into RRF ---
        # Deduplicate and rank BM25 results
        bm25_seen = {}
        for nid, score in bm25_raw:
            if nid not in bm25_seen or score > bm25_seen[nid]:
                bm25_seen[nid] = score
        bm25_ranked = sorted(bm25_seen.keys(), key=lambda x: bm25_seen[x], reverse=True)
        for rank, nid in enumerate(bm25_ranked[:effective_n_retrieve]):
            rrf_scores[nid] = rrf_scores.get(nid, 0) + 1 / (rank + 30)
            overlap_count[nid] = overlap_count.get(nid, 0) + 1
            matched_queries.setdefault(nid, []).append(len(variants))  # BM25 = extra "query"
            if nid not in nugget_data:
                # Fetch nugget from SQLite since BM25 may find nuggets not in vector results
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
                    rrf_scores[nid] *= 1.15  # 15% boost for preferred sections

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
                    # Nugget-level feedback
                    nfb = nugget_fb.get(nid, 0)
                    if nfb > 0:
                        rrf_scores[nid] *= 1.3  # 30% boost for liked nuggets
                    elif nfb < 0:
                        rrf_scores[nid] *= 0.5  # 50% penalty for disliked
                    # Paper-level feedback
                    pid = nugget_data[nid]["paper_id"]
                    pfb = paper_fb.get(pid, 0)
                    if pfb > 0:
                        rrf_scores[nid] *= 1.0 + min(pfb * 0.05, 0.3)
                    elif pfb < 0:
                        rrf_scores[nid] *= max(1.0 + pfb * 0.1, 0.3)
            except Exception as e:
                log.warning("Feedback boosting failed: %s", e)

        # Depth-of-coverage: boost papers where many nuggets matched (deep coverage)
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

        # Paper authority boost (citation count + paper type, mode-aware)
        auth_scale = routing.get("authority_boost", 1.0)
        review_boost = routing.get("review_boost", 0.0)
        _authority_cache: dict[str, float] = {}
        for nid in rrf_scores:
            pid = nugget_data[nid]["paper_id"]
            if pid not in _authority_cache:
                paper = _kb._get_paper(pid)
                if paper:
                    cc = paper.get("citation_count") or 0
                    icc = paper.get("influential_citation_count") or 0
                    ptype = (paper.get("paper_type") or "").lower()
                    is_review = "review" in ptype or "survey" in ptype
                    authority = 1.0 + auth_scale * 0.1 * min(math.log10(max(cc, 1)), 4)
                    if icc > 10:
                        authority += 0.1
                    if is_review:
                        authority += review_boost
                    _authority_cache[pid] = authority
                else:
                    _authority_cache[pid] = 1.0
            rrf_scores[nid] *= _authority_cache[pid]

        # --- Step 4b: Cross-encoder reranking ---
        if req.rerank:
            pre_rerank = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            rrf_scores = await loop.run_in_executor(
                _executor,
                lambda: rerank_nuggets(
                    query=last_msg,
                    nugget_ids=pre_rerank,
                    nugget_data=nugget_data,
                    rrf_scores=rrf_scores,
                    top_n=req.rerank_top_n,
                    blend_weight=req.rerank_weight,
                ),
            )
            log.info("Reranked top %d candidates (weight=%.1f)", req.rerank_top_n, req.rerank_weight)

        # Sort by RRF score
        ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Exclude rejected nuggets and papers from feedback loop
        if req.excluded_nuggets or req.excluded_papers:
            excluded_nids = set(req.excluded_nuggets)
            excluded_pids = set(req.excluded_papers)
            ranked = [
                nid for nid in ranked
                if nid not in excluded_nids
                and nugget_data[nid]["paper_id"] not in excluded_pids
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
        max_pp = routing.get("max_per_paper", req.max_per_paper)

        def _can_add(nid):
            return paper_counts[nugget_data[nid]["paper_id"]] < max_pp

        # First pass: pick up to 2 from each type, respecting per-paper cap
        sorted_types = sorted(by_type.keys(), key=lambda t: rrf_scores[by_type[t][0]], reverse=True)
        for t in sorted_types:
            added = 0
            for nid in by_type[t]:
                if added >= 2 or len(selected) >= req.n_context:
                    break
                if nid not in selected and _can_add(nid):
                    selected.append(nid)
                    paper_counts[nugget_data[nid]["paper_id"]] += 1
                    added += 1
            if len(selected) >= req.n_context:
                break

        # Second pass: fill remaining slots by RRF score, respecting per-paper cap
        for nid in ranked:
            if len(selected) >= req.n_context:
                break
            if nid not in selected and _can_add(nid):
                selected.append(nid)
                paper_counts[nugget_data[nid]["paper_id"]] += 1

        # Third pass: if still under n_context, relax paper cap
        for nid in ranked:
            if len(selected) >= req.n_context:
                break
            if nid not in selected:
                selected.append(nid)
                paper_counts[nugget_data[nid]["paper_id"]] += 1

        top_ids = selected

        # --- Inject pinned paper nuggets ---
        pinned_nids = []
        if req.pinned_papers:
            for pid in req.pinned_papers:
                rows = _kb.get_paper_nuggets(pid)
                if not rows:
                    continue
                # Sort by thesis_relevance desc
                rows.sort(key=lambda r: r.get("thesis_relevance", 3), reverse=True)
                # Type diversity: max 4 per type, max 15 per paper
                type_counts = defaultdict(int)
                paper_nuggets = []
                for r in rows:
                    ntype = r.get("type", "")
                    if type_counts[ntype] >= 4:
                        continue
                    type_counts[ntype] += 1
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
            # Hard cap: max 20 pinned nuggets
            if len(pinned_nids) > 20:
                pinned_nids.sort(key=lambda n: nugget_data[n].get("thesis_relevance", 3), reverse=True)
                pinned_nids = pinned_nids[:20]
            # Remove any pinned IDs already in top_ids, then prepend
            top_ids = [nid for nid in top_ids if nid not in pinned_nids]
            # Ensure RRF slots get at least 4
            max_rrf = max(4, req.n_context - len(pinned_nids))
            top_ids = pinned_nids + top_ids[:max_rrf]
            log.info("Pinned %d nuggets from %d papers", len(pinned_nids), len(req.pinned_papers))

        log.info("Retrieved %d unique nuggets, selected %d after RRF+balancing",
                 len(nugget_data), len(top_ids))

        # Enrich with paper metadata from SQLite
        bibtex_key_counts: dict[str, int] = {}

        def _make_bibtex_key(authors: str, year) -> str:
            """Generate bibtex key from first author surname + year."""
            surname = "unknown"
            if authors:
                first = authors.split(",")[0].strip()
                parts = first.split()
                if parts:
                    surname = parts[-1].lower()
                    surname = "".join(c for c in surname if c.isalpha())
            yr = str(year) if year else "0000"
            base = f"{surname}{yr}"
            count = bibtex_key_counts.get(base, 0)
            bibtex_key_counts[base] = count + 1
            if count == 0:
                return base
            return base + chr(ord("a") + count - 1)

        def _enrich(nid):
            nd = nugget_data[nid]
            paper = _kb._get_paper(nd["paper_id"])
            nd["paper_title"] = paper.get("title", "") if paper else ""
            nd["paper_year"] = paper.get("year") if paper else None
            nd["paper_authors"] = paper.get("authors", "") if paper else ""
            nd["arxiv_id"] = paper.get("arxiv_id", "") if paper else ""
            nd["doi"] = paper.get("doi", "") if paper else ""
            real_key = _resolve_bibtex_key(nd["arxiv_id"], nd["doi"], nd["paper_title"])
            nd["bibtex_key"] = real_key if real_key else _make_bibtex_key(nd["paper_authors"], nd["paper_year"])
            nd["rrf_score"] = round(rrf_scores[nid], 4)
            nd["overlap_count"] = overlap_count[nid]
            nd["matched_queries"] = matched_queries.get(nid, [])
            # Fetch source_chunk from SQLite if not already present
            if "source_chunk" not in nd:
                row = _kb.db.execute(
                    "SELECT source_chunk FROM nuggets WHERE nugget_id = ?", (nid,)
                ).fetchone()
                nd["source_chunk"] = row[0] if row else None
            return nd

        top_nuggets = [_enrich(nid) for nid in top_ids]

        # Year filtering (post-enrichment since we need paper_year)
        if req.year_min or req.year_max:
            top_nuggets = [
                n for n in top_nuggets
                if n["paper_year"] is not None
                and (req.year_min is None or n["paper_year"] >= req.year_min)
                and (req.year_max is None or n["paper_year"] <= req.year_max)
            ]

        # --- Step 5: Build context + call OpenRouter ---
        mode_file = MODE_FILES.get(req.mode, "survey.md")
        mode_path = COMMANDS_DIR / mode_file
        if mode_path.exists():
            mode_prompt = mode_path.read_text()
        else:
            mode_prompt = (COMMANDS_DIR / "survey.md").read_text()

        sources_xml = "<sources>\n"
        for n in top_nuggets:
            pinned_attr = ' pinned="true"' if n.get("pinned") else ""
            sources_xml += (
                f'<source id="{n["nugget_id"]}" paper="{n["paper_title"]}" '
                f'year="{n["paper_year"]}" type="{n["type"]}" overlap="{n["overlap_count"]}/{len(variants)}" '
                f'score="{n["rrf_score"]:.3f}" bibtex_key="{n["bibtex_key"]}"{pinned_attr}>\n'
                f'{n["document"]}\n'
                f"</source>\n"
            )
        sources_xml += "</sources>\n\n"

        system_prompt = sources_xml + mode_prompt

        if req.latex_mode:
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
        )

        # --- Step 6: Stream response ---
        log.info("Streaming response: %d context nuggets, %.1fs retrieval",
                 len(top_nuggets), time.time() - t0)

        async def _stream():
            def _call_openrouter():
                return or_client.chat.completions.create(
                    model=req.model,
                    messages=chat_messages,
                    stream=True,
                    max_tokens=4000,
                    temperature=0.3,
                )

            stream = await loop.run_in_executor(_executor, _call_openrouter)

            try:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'type': 'delta', 'content': delta})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

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
