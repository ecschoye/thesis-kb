# Web UI Status

## What was built

| File | Description |
|------|-------------|
| `src/api.py` | FastAPI backend with `/health`, `/query`, `/chat` (SSE streaming) endpoints |
| `static/index.html` | Single-file React 18 + Tailwind chat UI (CDN only, no build step) |
| `scripts/start_api.sh` | Start API on IDUN (HPC), accepts `--config` and `--port` flags |
| `scripts/start_local.sh` | Start API locally on Mac with Ollama |
| `README.md` | Updated with Web UI section (architecture, usage, env vars) |

## Chat Pipeline

```
User question
    │
    ▼
Step 1: Query Expansion (OpenRouter)
    │  Calls a cheap/fast model (default: google/gemini-flash-1.5) to generate
    │  N distinct search query variants from the user's question.
    │  Falls back to [original_message] if parsing fails.
    ▼
Step 2: Embed Variants
    │  Each variant string is embedded via the configured embed backend:
    │    - Ollama (local Mac): qwen3-embedding:8b at localhost:11434
    │    - vLLM (IDUN): Qwen/Qwen3-Embedding-8B at localhost:8000
    │  Uses format_nugget_text({"question": text, "answer": ""}, instruction)
    │  Runs concurrently via ThreadPoolExecutor + asyncio.gather
    ▼
Step 3: Multi-Vector ChromaDB Retrieval
    │  Each embedding vector queries ChromaDB directly:
    │    collection.query(query_embeddings=[vec], n_results=n_retrieve)
    │  Produces N ranked result lists (one per variant).
    ▼
Step 4: Reciprocal Rank Fusion (RRF)
    │  Merges all result lists:
    │    rrf_score[nugget_id] += 1 / (rank + 60)
    │  Sorts by rrf_score descending, takes top n_context nuggets.
    │  Tracks overlap_count (how many variant lists each nugget appeared in).
    ▼
Step 5: Build Context + OpenRouter Call
    │  Loads system prompt from .claude/commands/{mode}.md
    │  Prepends <sources> XML block with top nuggets (including rrf_score, overlap)
    │  Calls OpenRouter with stream=True using the user's chosen model.
    ▼
Step 6: SSE Stream
    data: {"type": "delta", "content": "..."}      (text chunks)
    data: {"type": "sources", "nuggets": [...]}     (source cards)
    data: {"type": "done"}                          (end of stream)
```

## How to Start

### On IDUN (HPC)

```bash
# 1. Set OPENROUTER_API_KEY (or put it in ~/.openrouter_key)
export OPENROUTER_API_KEY=sk-or-...

# 2. Start the API (uses config.yaml → vLLM for embeddings)
bash scripts/start_api.sh --config config.yaml --port 8001

# 3. From Mac, open SSH tunnel
ssh -N -L 8001:localhost:8001 ecschoye@idun-login1.hpc.ntnu.no

# 4. Open in browser
# static/index.html?api=http://localhost:8001
```

Note: vLLM embedding server must be running on IDUN (port 8000) for query embedding to work.

### Locally on Mac

```bash
# 1. Start Ollama and pull the embedding model
ollama serve
ollama pull qwen3-embedding:8b

# 2. Set OPENROUTER_API_KEY (or put it in ~/.openrouter_key)
export OPENROUTER_API_KEY=sk-or-...

# 3. Start the API (uses config-ollama.yaml → Ollama for embeddings)
bash scripts/start_local.sh

# 4. Open in browser
# static/index.html  (defaults to http://localhost:8001)
```

Requires KB files (kb/chromadb/, kb/nuggets.db) to be present locally.

## Configuration

| Scenario | Config file | Embed backend | Embed model |
|----------|-------------|---------------|-------------|
| IDUN (HPC) | `config.yaml` | vLLM (port 8000) | `Qwen/Qwen3-Embedding-8B` |
| Mac (local) | `config-ollama.yaml` | Ollama (port 11434) | `qwen3-embedding:8b` |

- `OPENROUTER_API_KEY` — Required for query expansion and answer generation. Set in env or store at `~/.openrouter_key`.
- `THESIS_KB_CONFIG` — Optional env var to override config path (scripts set this automatically).

## Pip Packages Confirmed

| Package | Version |
|---------|---------|
| fastapi | 0.135.1 |
| uvicorn | 0.41.0 |
| openai | 2.26.0 |
| sse-starlette | 3.3.2 |

## ChromaDB Confirmed

- Collection `thesis_nuggets`: **150,728 nuggets**
- `collection.query(query_embeddings=[...])` — supported and verified

## Known Issues / Next Steps

1. **`ThesisKB.stats()` exists** — Verified at `src/query.py:150-169`. Returns `total_nuggets`, `total_papers`, `nuggets_by_type`, `papers_by_year`, `nuggets_by_section`. The `/health` endpoint calls this correctly.

2. **`ThesisKB._get_paper()` exists** — Verified at `src/query.py:141-148`. Private method that looks up paper metadata from SQLite by paper_id. Called in `api.py` to enrich RRF results with paper titles/years/authors.

3. **Embed format for query variants** — Uses `format_nugget_text({"question": text, "answer": ""}, instruction)` which produces `Instruct: {instruction}\nQuery: Q: {text} A: `. The empty answer field means the format includes a trailing `A: ` which is slightly wasteful but functionally correct — the embedding model will still produce a reasonable vector. This matches how `ThesisKB._embed_query()` works (query.py:37-44), so query-time behavior is consistent.

4. **Not yet tested end-to-end** — The API imports verify clean, but a live test requires:
   - Ollama running with `qwen3-embedding:8b` (for local) or vLLM embedding server (for IDUN)
   - `OPENROUTER_API_KEY` set
   - Open `static/index.html` in browser and send a query

5. **UI loads React/Tailwind/marked.js from CDN** — Requires internet access in the browser. Works fine for local dev; if running on an air-gapped network, these would need to be vendored.

6. **SQLite thread safety** — `ThesisKB` opens a single SQLite connection at startup. FastAPI runs in an async event loop with a ThreadPoolExecutor. The `_get_paper()` calls happen in the main thread (not in the executor), so concurrent `/chat` requests could hit SQLite from multiple coroutines. In practice this is fine for single-user usage but could be an issue under load. Fix: use `check_same_thread=False` or a connection pool.
