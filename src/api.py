"""FastAPI backend for thesis-kb web chat interface."""
import os, json, asyncio
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
}


def _init(config_path: str):
    global _cfg, _kb, _embed_client, _embed_model, _embed_instruction, _collection
    _cfg = load_config(config_path)
    _kb = ThesisKB(config_path)
    _embed_client, _embed_model = make_embed_client(_cfg)
    emb_cfg = _cfg.get("embed", {}).get("embedding", {})
    _embed_instruction = emb_cfg.get("instruction", "")
    _collection = _kb.collection


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
    type_filter: list[str] = []

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
        loop = asyncio.get_event_loop()

        # --- Step 1: Query expansion via OpenRouter ---
        last_msg = req.messages[-1].content if req.messages else ""
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
            variants = [
                last_msg,
                f"methods and architectures for {last_msg}",
                f"experimental results and benchmarks for {last_msg}",
                f"limitations and challenges of {last_msg}",
            ]
            variant_types = [None, "method", "result", "limitation"]

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

        # --- Step 3: Multi-vector ChromaDB retrieval ---
        def _retrieve(vec, target_type=None):
            kwargs = {"query_embeddings": [vec], "n_results": req.n_retrieve}
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
        all_results = await asyncio.gather(*retrieve_tasks)

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

        # Thesis relevance boosting
        for nid in rrf_scores:
            relevance = nugget_data[nid].get("thesis_relevance", 3)
            try:
                relevance = int(relevance)
            except (ValueError, TypeError):
                relevance = 3
            rrf_scores[nid] *= 1.0 + (relevance - 3) * 0.2

        # Sort by RRF score
        ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Exclude rejected nuggets from feedback loop
        if req.excluded_nuggets:
            excluded = set(req.excluded_nuggets)
            ranked = [nid for nid in ranked if nid not in excluded]

        # Type-balanced selection: guarantee diversity across nugget types
        by_type = defaultdict(list)
        for nid in ranked:
            ntype = nugget_data[nid]["type"]
            by_type[ntype].append(nid)

        selected = []
        # First pass: pick up to 2 from each type (guarantees diversity)
        # Sort types by their top nugget's RRF score so most-relevant types get priority
        sorted_types = sorted(by_type.keys(), key=lambda t: rrf_scores[by_type[t][0]], reverse=True)
        for t in sorted_types:
            for nid in by_type[t][:2]:
                if nid not in selected:
                    selected.append(nid)
                if len(selected) >= req.n_context:
                    break
            if len(selected) >= req.n_context:
                break

        # Second pass: fill remaining slots by pure RRF score
        for nid in ranked:
            if len(selected) >= req.n_context:
                break
            if nid not in selected:
                selected.append(nid)

        top_ids = selected

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
            nd["bibtex_key"] = _make_bibtex_key(nd["paper_authors"], nd["paper_year"])
            nd["rrf_score"] = round(rrf_scores[nid], 4)
            nd["overlap_count"] = overlap_count[nid]
            nd["matched_queries"] = matched_queries.get(nid, [])
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
            sources_xml += (
                f'<source id="{n["nugget_id"]}" paper="{n["paper_title"]}" '
                f'year="{n["paper_year"]}" type="{n["type"]}" overlap="{n["overlap_count"]}/{len(variants)}" '
                f'score="{n["rrf_score"]:.3f}" bibtex_key="{n["bibtex_key"]}">\n'
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
                })
            yield f"data: {json.dumps({'type': 'sources', 'nuggets': sources_payload, 'variants': variants})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        # Check if embed server is unreachable
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
