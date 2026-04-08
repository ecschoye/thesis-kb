# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

RAG-powered knowledge base for an MSc thesis on RGB-Event fusion and spiking neural networks. 1,564 papers processed into ~153k QA nuggets, searchable via vector (ChromaDB) + full-text (SQLite FTS5) retrieval.

## Commands

### Install
```bash
# HPC (IDUN)
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Local (Mac)
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
brew install tesseract  # for OCR fallback
```

### Running the Web UI
```bash
# Local (requires Ollama running separately with `ollama serve`)
bash scripts/start_local.sh

# HPC
bash scripts/start_api.sh --config config.yaml --port 8001
# From Mac: ssh -N -L 8001:localhost:8001 ecschoye@idun-login1.hpc.ntnu.no
# Open static/index.html?api=http://localhost:8001
```

### Pipeline Stages (each is a Python module)
```bash
source venv/bin/activate
python -m src.acquire -c config.yaml         # PDF ingest + Semantic Scholar metadata
python -m src.extract -c config.yaml         # PDF → text (PyMuPDF + OCR fallback)
python -m src.chunk -c config.yaml           # text → token chunks
python -m src.nuggets -c config.yaml         # chunks → QA nuggets (raw extraction only)
python -m src.nuggets.unified -c config.yaml # single-pass: extract + quality + augment (preferred)
python -m src.nuggets.unified -c config.yaml --reprocess  # quality + augment on existing nuggets
python -m src.embed -c config.yaml           # nuggets → vector embeddings
python -m src.store -c config.yaml           # embeddings → ChromaDB + SQLite
```

### SLURM (HPC)
```bash
sbatch slurm/full_pipeline.slurm   # GPU — full: extract → chunk → unified → embed → store
sbatch slurm/extract.slurm         # CPU — text extraction
sbatch slurm/chunk.slurm           # CPU — chunking
sbatch slurm/nuggets.slurm         # GPU — raw nugget extraction (Qwen3.5-27B)
sbatch slurm/unified.slurm         # GPU — unified pipeline (Qwen3.5-27B)
sbatch slurm/embed.slurm           # GPU — embedding (Qwen3-Embedding-8B)
sbatch slurm/store.slurm           # CPU — build KB
```

GPU scripts support **dynamic multi-GPU scaling** — change `--gres=gpu:N` to allocate more GPUs; vLLM tensor parallelism and concurrency scale automatically.

### Testing & Linting
No test suite or linting config exists. Ruff has been used ad-hoc (`.ruff_cache/` present) but no rules are configured.

### Mac ↔ IDUN Transfers
- Mac: `~/thesis-kb/`
- IDUN: `ecschoye@idun.hpc.ntnu.no:/cluster/work/ecschoye/thesis-kb/`
- Use `rsync -avz --progress` to sync subdirectories between the two.

## Adding a New Paper to the KB

1. **Place the PDF**: `cp /path/to/paper.pdf /cluster/work/ecschoye/thesis-papers/`
   - Naming: arXiv ID with underscores (`2401_17151.pdf`). Non-arXiv papers use any descriptive filename.

2. **Register in manifest**: `python -m src.acquire -c config.yaml`
   - Scans `pdf_dir`, enriches metadata via Semantic Scholar, updates `corpus/manifest.json`.

3. **Run the pipeline** — each stage skips already-processed papers:
   ```bash
   python -m src.extract -c config.yaml    # skips if corpus/texts/{id}.json exists
   python -m src.chunk -c config.yaml      # skips if corpus/chunks/{id}.json exists
   python -m src.nuggets.unified -c config.yaml  # skips if corpus/nuggets_unified/{id}.json exists
   python -m src.embed -c config.yaml      # full rebuild — re-embeds all nuggets
   python -m src.store -c config.yaml      # full rebuild — recreates ChromaDB + SQLite
   ```
   **On HPC**: `sbatch slurm/full_pipeline.slurm`

4. **To reprocess a paper**, delete its output file and rerun:
   ```bash
   rm corpus/texts/2401_17151.json
   rm corpus/chunks/2401_17151.json
   rm corpus/nuggets/2401_17151.json        # or corpus/nuggets_unified/
   ```

## Architecture

**Two-phase system**: batch pipeline builds the KB, then a query server answers questions from it.

### Batch Pipeline
```
Acquire → Extract → Chunk → Nuggets → Embed → Store
  PDFs    PyMuPDF   tokens   LLM QA    vectors  ChromaDB+SQLite
```

The **unified pipeline** (`src.nuggets.unified`) combines nugget extraction, quality scoring, and augmentation into a single per-paper pass, replacing the legacy three-stage approach (`extract` → `quality_main` → `augment_main`).

### Query Server (src/api.py)
```
User query → Expand (LLM generates search variants) → Embed → Retrieve (vector + BM25)
  → RRF fusion → Cross-encoder rerank (flashrank) → LLM streaming answer via SSE
```

**Mode-based retrieval** — different slash commands (background, draft, check, review, compare, gaps, outline) each configure retrieval parameters (n_retrieve, preferred_sections, authority_boost, max_per_paper).

### Key Modules
- `src/api.py` — FastAPI backend, SSE streaming, mode-based retrieval routing
- `src/query.py` — ThesisKB class: ChromaDB + SQLite + FTS5 + embedding client
- `src/rerank.py` — Cross-encoder reranking (ms-marco-MiniLM-L-12-v2 via flashrank)
- `src/nuggets/extract.py` — LLM nugget extraction with domain-specific system prompt
- `src/nuggets/unified.py` — Single-pass extract + quality + augment (preferred over separate stages)
- `src/embed/embedder.py` — Pipelined instruction-aware embedding (Qwen3-Embedding-8B)
- `src/store/kb.py` — ChromaDB + SQLite KB builder with quality score integration
- `src/code_extract/` — Code-aware KB: indexes codebase alongside papers
- `src/utils.py` — Config loading, LLM client factory (`make_llm_client`)
- `static/index.html` — Complete React 18 frontend (single file, no build step)

Each pipeline module has `__main__.py` for CLI invocation via `python -m src.<module>`.

### Frontend
Single-file React 18 app (~1,700 lines). Uses Tailwind CDN with `dark:` classes, Babel for in-browser JSX, marked + DOMPurify for markdown. No build step.

### LLM Backends
Four deployment configs, all using OpenAI-compatible API:
- `config.yaml` — HPC: vLLM (Qwen3.5-27B for LLM, Qwen3-Embedding-8B for embed)
- `config-ollama.yaml` — Local: Ollama (qwen3.5:9b + qwen3-embedding:8b)
- `config-openrouter.yaml` — API: OpenRouter for LLM, Ollama for embed
- `config-openrouter-free.yaml` — API: OpenRouter free tier

## Conventions

- **Paper IDs** use underscores (`2401_17151`); arXiv IDs use dots (`2401.17151`)
- **Nugget types**: method, result, claim, limitation, comparison, background
- **Config**: YAML files, `THESIS_KB_CONFIG` env var overrides default path
- **Env**: `OPENROUTER_API_KEY` required for API mode (loaded from `.env`); `VLLM_PORT` auto-set by SLURM scripts
- **Logging**: `src/log.py` → `logs/` with rotating file handlers
- **Corpus outputs**: `corpus/nuggets/` (legacy), `corpus/nuggets_unified/` (unified pipeline)

## Claude Integration

### MCP Server (`src/mcp_server.py`)
Persistent process exposing KB search via ChromaDB vector search + SQLite FTS5. Configured in `.mcp.json`. Uses Ollama (`qwen3-embedding:8b`) for query embeddings locally.

**Tools:**
| Tool | Purpose |
|------|---------|
| `semantic_search(query, n, types, section, year_min, year_max)` | Vector search via ChromaDB |
| `multi_search(queries, n, types)` | Multiple queries, deduplicated |
| `bm25_search(query, n, types)` | FTS5 keyword search |
| `find_papers(author, title, year)` | Paper metadata lookup |
| `get_paper_nuggets(paper_id, types, limit)` | All nuggets for a paper |
| `get_paper_info(paper_id)` | Single paper metadata |
| `kb_stats()` | Counts, type/year distributions |

### Slash Commands (.claude/commands/)
11 thesis-writing commands that use MCP tools for retrieval: `/survey`, `/draft`, `/outline`, `/cite`, `/check`, `/compare`, `/gaps`, `/background`, `/review`, `/stats`, `/intro`

### Hooks
`cite_prefetch.py` runs on every prompt submission — extracts `\cite{Key}` references, queries SQLite for matching papers, returns nuggets as context. Must use `hookSpecificOutput` wrapper format.

## Writing Style

- Never use em dashes. Prefer commas first, then separate sentences. Parentheses or colons are acceptable but use sparingly.

## Sibling Projects

This KB serves two sibling repositories. All three share state through the Obsidian vault at ~/vault/.

- **Implementation**: ~/SMCM-MCFNet (the model codebase)
- **Thesis**: ~/TDT4900-master-thesis (the LaTeX document)

## Project Brain

The shared knowledge base is the Obsidian vault at ~/vault/, structured as a neural graph.
- Read ~/vault/brainstem/index.md to orient at session start
- Read the relevant status file in ~/vault/prefrontal/status/
- Read the last 3 entries from the relevant devlog in ~/vault/hippocampus/devlog/

When you learn something about a concept: update ~/vault/cortex/concepts/
When a significant decision is made: create a file in ~/vault/cortex/decisions/
Always use [[wikilinks]] between files.

## Session Protocol

**MANDATORY: Before responding to the user's first message, execute `/orient`.** This reads current state from the vault and briefs the user. Do not skip this. Do not jump straight to the user's request. The vault is the shared brain across three interconnected repos, and it must stay current.

If the user's first message is a greeting or open-ended ("hey", "what's up", "let's work"), run `/orient` and brief them. If their first message is a specific task, still run `/orient` silently, mention the briefing briefly, then address their request.

### Cross-project updates

When KB work produces insights relevant to the other projects, update the shared devlog:
- **New papers or findings relevant to implementation**: prepend to ~/vault/hippocampus/devlog/shared.md
- **New papers or findings relevant to thesis writing**: prepend to ~/vault/hippocampus/devlog/shared.md
- **Concept updates** (e.g., ingested a paper that changes understanding of a technique): update ~/vault/cortex/concepts/

### On session end via /wrap-up:
- If session produced cross-project insights, prepend to ~/vault/hippocampus/devlog/shared.md
- Update concept/decision files in ~/vault/cortex/ as needed
