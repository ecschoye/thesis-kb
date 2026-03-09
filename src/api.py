"""FastAPI backend for thesis-kb web chat interface."""
import os, json, asyncio
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
    n_variants: int = 4
    n_retrieve: int = 12
    n_context: int = 8
    model: str = "google/gemini-flash-1.5"

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
            resp = expand_client.chat.completions.create(
                model=req.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a search query expander for an academic knowledge base about "
                            f"event-based vision, spiking neural networks, and autonomous driving. "
                            f"Given a user question, generate {req.n_variants} distinct search queries "
                            f"that together cover different angles of what the user is asking. "
                            f"Output ONLY a JSON array of strings, no explanation."
                        ),
                    },
                    {"role": "user", "content": last_msg},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                text = "\n".join(lines).strip()
            return json.loads(text)

        try:
            variants = await loop.run_in_executor(_executor, _expand)
            if not isinstance(variants, list):
                variants = [last_msg]
        except Exception:
            variants = [last_msg]

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
        def _retrieve(vec):
            return _collection.query(
                query_embeddings=[vec], n_results=req.n_retrieve
            )

        retrieve_tasks = [
            loop.run_in_executor(_executor, _retrieve, emb) for emb in embeddings
        ]
        all_results = await asyncio.gather(*retrieve_tasks)

        # --- Step 4: Reciprocal Rank Fusion ---
        rrf_scores: dict[str, float] = {}
        overlap_count: dict[str, int] = {}
        nugget_data: dict[str, dict] = {}

        for result_set in all_results:
            ids = result_set["ids"][0]
            metas = result_set["metadatas"][0]
            docs = result_set["documents"][0]
            dists = result_set["distances"][0]
            for rank, (nid, meta, doc, dist) in enumerate(
                zip(ids, metas, docs, dists)
            ):
                rrf_scores[nid] = rrf_scores.get(nid, 0) + 1 / (rank + 60)
                overlap_count[nid] = overlap_count.get(nid, 0) + 1
                if nid not in nugget_data:
                    nugget_data[nid] = {
                        "nugget_id": nid,
                        "type": meta.get("type", ""),
                        "confidence": meta.get("confidence", ""),
                        "section": meta.get("section", ""),
                        "document": doc,
                        "paper_id": meta.get("paper_id", ""),
                        "distance": round(dist, 4),
                    }

        # Sort by RRF score
        ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = ranked[: req.n_context]

        # Enrich with paper metadata from SQLite
        def _enrich(nid):
            nd = nugget_data[nid]
            paper = _kb._get_paper(nd["paper_id"])
            nd["paper_title"] = paper.get("title", "") if paper else ""
            nd["paper_year"] = paper.get("year") if paper else None
            nd["paper_authors"] = paper.get("authors", "") if paper else ""
            nd["arxiv_id"] = paper.get("arxiv_id", "") if paper else ""
            nd["rrf_score"] = round(rrf_scores[nid], 4)
            nd["overlap_count"] = overlap_count[nid]
            return nd

        top_nuggets = [_enrich(nid) for nid in top_ids]

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
                f'year="{n["paper_year"]}" overlap="{n["overlap_count"]}/{len(variants)}" '
                f'score="{n["rrf_score"]:.3f}">\n'
                f'{n["document"]}\n'
                f"</source>\n"
            )
        sources_xml += "</sources>\n\n"

        system_prompt = sources_xml + mode_prompt

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
                    "paper_title": n["paper_title"],
                    "paper_year": n["paper_year"],
                    "paper_authors": n["paper_authors"],
                    "arxiv_id": n["arxiv_id"],
                    "type": n["type"],
                    "section": n["section"],
                    "document": n["document"],
                    "rrf_score": n["rrf_score"],
                    "overlap_count": n["overlap_count"],
                    "n_variants": len(variants),
                })
            yield f"data: {json.dumps({'type': 'sources', 'nuggets': sources_payload})}\n\n"
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
