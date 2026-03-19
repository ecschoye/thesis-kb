# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

RAG-powered knowledge base for an MSc thesis on RGB-Event fusion and spiking neural networks. 844 papers processed into ~150k QA nuggets, searchable via vector (ChromaDB) + full-text (SQLite FTS5) retrieval.

## Commands

### Running the Web UI
```bash
# Local (requires Ollama running separately with `ollama serve`)
bash scripts/start_local.sh

# HPC
bash scripts/start_api.sh --config config.yaml --port 8001
```

### Pipeline Stages (each is a Python module)
```bash
source venv/bin/activate
python -m src.acquire -c config.yaml
python -m src.extract -c config.yaml
python -m src.chunk -c config.yaml
python -m src.nuggets -c config.yaml
python -m src.nuggets.quality_main -c config.yaml
python -m src.nuggets.augment_main -c config.yaml
python -m src.embed -c config.yaml
python -m src.store -c config.yaml
```

### HPC (SLURM)
```bash
sbatch slurm/reprocess_pipeline.slurm   # full: quality → augment → embed → store
sbatch slurm/nugget_extract.slurm       # individual stages also available
```

### Mac ↔ IDUN Transfers
- Mac: `~/thesis-kb/`
- IDUN: `ecschoye@idun.hpc.ntnu.no:/cluster/work/ecschoye/thesis-kb/`
- Use `rsync -avz --progress` to sync subdirectories between the two.

### Install
```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

## Architecture

**Two-phase system**: batch pipeline builds the KB, then a query server answers questions from it.

### Batch Pipeline
```
Acquire → Extract → Chunk → Nuggets → [Quality] → [Augment] → Embed → Store
  PDFs    PyMuPDF   tokens   LLM QA     scoring     improve    vectors  ChromaDB+SQLite
```
Quality and Augment stages exist but have never been run in production.

### Query Server (src/api.py)
```
User query → Expand (LLM generates search variants) → Embed → Retrieve (vector + BM25)
  → RRF fusion → Cross-encoder rerank (flashrank) → LLM streaming answer via SSE
```

The API has **mode-based retrieval** — different slash commands (background, draft, check, review, compare, gaps, outline) each configure different retrieval parameters (n_retrieve, preferred_sections, authority_boost, max_per_paper).

### Key Modules
- `src/api.py` — FastAPI backend, SSE streaming, mode-based retrieval routing
- `src/query.py` — ThesisKB class: ChromaDB + SQLite + FTS5 + embedding client
- `src/rerank.py` — Cross-encoder reranking (ms-marco-MiniLM-L-12-v2 via flashrank)
- `src/nuggets/extract.py` — LLM nugget extraction with domain-specific system prompt
- `src/embed/embedder.py` — Instruction-aware embedding (Qwen3-Embedding-8B)
- `src/store/kb.py` — ChromaDB + SQLite KB builder
- `src/utils.py` — Config loading, LLM client factory (`make_llm_client`)
- `static/index.html` — Complete React 18 frontend (single file, no build step)

### Frontend
Single-file React 18 app (~1600 lines). Uses Tailwind CDN with `dark:` classes, Babel for in-browser JSX, marked + DOMPurify for markdown. No build step — just open in browser.

### LLM Backends
Three deployment configs, all using OpenAI-compatible API:
- `config.yaml` — HPC: vLLM (Qwen3.5-27B for LLM, Qwen3-Embedding-8B for embed)
- `config-ollama.yaml` — Local: Ollama (qwen3.5:9b + qwen3-embedding:8b)
- `config-openrouter.yaml` — API: OpenRouter for LLM, Ollama for embed

## Conventions

- **Paper IDs** use underscores (`2401_17151`); arXiv IDs use dots (`2401.17151`)
- **Nugget types**: method, result, claim, limitation, comparison, background
- **Config**: YAML files, `THESIS_KB_CONFIG` env var overrides default path
- **Env**: `OPENROUTER_API_KEY` required for API mode (loaded from `.env`)
- **Logging**: `src/log.py` → `logs/` with rotating file handlers

## Claude Integration

### MCP Server (thesis-papers-rag)
40 tools for the knowledge base — see the global CLAUDE.md tool reference. Key tools: `find_citations`, `search_papers`, `get_paper_summary`, `section_gap_analysis`, `review_section`.

### Slash Commands (.claude/commands/)
8 thesis-writing commands: `/survey`, `/draft`, `/outline`, `/cite`, `/check`, `/compare`, `/gaps`, `/background`, `/review`, `/stats`

### Hooks
`cite_prefetch.py` runs on every prompt submission — extracts `\cite{Key}` references, queries SQLite for matching papers, returns nuggets as context. Must use `hookSpecificOutput` wrapper format.
