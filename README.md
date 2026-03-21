# thesis-kb

RAG knowledge base for an MSc thesis on event cameras, RGB-Event fusion, spiking neural networks, and object detection. Processes academic papers into searchable QA nuggets backed by vector embeddings (ChromaDB) and full-text search (SQLite FTS5).

**1,564 papers · 153k nuggets · 6 nugget types · 11 retrieval modes**

## Architecture

Two-phase system: a batch pipeline builds the KB offline, then a query server answers questions from it in real time.

### Batch Pipeline

```
Acquire → Extract → Chunk → Nuggets → Embed → Store
  PDFs    PyMuPDF   tokens   LLM QA    vectors  ChromaDB + SQLite
```

The **unified pipeline** (`src.nuggets.unified`) combines nugget extraction, quality scoring, and augmentation into a single per-paper pass, replacing the legacy three-stage approach.

| Stage | Module | Description |
|-------|--------|-------------|
| **Acquire** | `src.acquire` | Ingest PDFs, enrich metadata via Semantic Scholar |
| **Extract** | `src.extract` | Text extraction with multi-column detection and OCR fallback |
| **Chunk** | `src.chunk` | Token chunking with sentence-boundary flex and block protection |
| **Nuggets** | `src.nuggets` | LLM nugget extraction (Qwen3.5-27B) |
| **Unified** | `src.nuggets.unified` | Single-pass extract + quality + augment (replaces separate stages) |
| **Embed** | `src.embed` | Pipelined instruction-aware embedding with metadata tags (Qwen3-Embedding-8B) |
| **Store** | `src.store` | ChromaDB vector index + SQLite metadata + FTS5 full-text search |

### Query Server

```
User query → Expand (LLM) → Embed → Retrieve (vector + BM25)
  → RRF fusion → Cross-encoder rerank (flashrank) → LLM streaming answer
```

The API has **mode-based retrieval** — different commands (`/background`, `/draft`, `/check`, `/review`, `/compare`, `/gaps`, `/outline`) each configure retrieval parameters (n_retrieve, preferred_sections, authority_boost, max_per_paper).

### Key Modules

| File | Purpose |
|------|---------|
| `src/api.py` | FastAPI backend, SSE streaming, mode-based retrieval routing |
| `src/query.py` | ThesisKB class: ChromaDB + SQLite + FTS5 + embedding client |
| `src/rerank.py` | Cross-encoder reranking (ms-marco-MiniLM-L-12-v2 via flashrank) |
| `src/nuggets/extract.py` | LLM nugget extraction with domain-specific system prompt |
| `src/nuggets/unified.py` | Unified extract + quality + augment pipeline |
| `src/embed/embedder.py` | Pipelined instruction-aware embedding with asymmetric metadata |
| `src/store/kb.py` | ChromaDB + SQLite KB builder with quality score integration |
| `static/index.html` | Single-file React 18 frontend (~1,700 lines, no build step) |

## Setup

### HPC (IDUN)

```bash
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Local (Mac)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew install tesseract  # for OCR fallback
```

## Usage

### Running stages individually

```bash
python -m src.acquire -c config.yaml
python -m src.extract -c config.yaml
python -m src.chunk -c config.yaml
python -m src.nuggets -c config.yaml              # raw extraction only
python -m src.nuggets.unified -c config.yaml       # unified: extract + quality + augment
python -m src.nuggets.unified -c config.yaml --reprocess  # quality + augment on existing nuggets
python -m src.embed -c config.yaml
python -m src.store -c config.yaml
```

### SLURM (HPC)

One script per stage, fully independent. All GPU scripts support **dynamic multi-GPU scaling** — change `--gres=gpu:N` to allocate more GPUs and vLLM parameters (tensor parallelism, max sequences) scale automatically.

```bash
sbatch slurm/extract.slurm        # CPU  — text extraction
sbatch slurm/chunk.slurm          # CPU  — chunking
sbatch slurm/nuggets.slurm        # GPU  — raw nugget extraction (Qwen3.5-27B)
sbatch slurm/unified.slurm        # GPU  — unified pipeline (Qwen3.5-27B)
sbatch slurm/embed.slurm          # GPU  — embedding (Qwen3-Embedding-8B)
sbatch slurm/store.slurm          # CPU  — build KB
```

Full pipeline (all stages, handles model swaps):

```bash
sbatch slurm/full_pipeline.slurm   # GPU — extract → chunk → unified → embed → store
```

**Multi-GPU example** (2× throughput for nugget extraction):
```bash
# Edit the SBATCH header or override:
sbatch --gres=gpu:2 slurm/unified.slurm
```

| Script | Partition | Time | GPU | Memory |
|--------|-----------|------|-----|--------|
| `extract.slurm` | CPUQ | 1h | — | 16GB |
| `chunk.slurm` | CPUQ | 30m | — | 8GB |
| `nuggets.slurm` | GPUQ | 20h | 1–N × 80GB | 64GB |
| `unified.slurm` | GPUQ | 18h | 1–N × 80GB | 64GB |
| `embed.slurm` | GPUQ | 4h | 1–N × 80GB | 32GB |
| `store.slurm` | CPUQ | 30m | — | 16GB |
| `full_pipeline.slurm` | GPUQ | 24h | 1–N × 80GB | 64GB |

### GPU Utilization

The pipeline is tuned for maximum GPU throughput:

- **vLLM prefix caching** — nugget extraction reuses the same system prompt; prefix caching avoids re-computing KV cache for it on every request
- **Tensor parallelism** — auto-detected from SLURM GPU allocation; splits model across GPUs for higher throughput
- **Scaled concurrency** — `max-num-seqs` scales as 64 × GPU count; Python-side `max_workers` set to 64 to saturate the GPU queue
- **Pipelined embedding** — multiple batches submitted concurrently (configurable `embed_workers`); batch N+1 queued while GPU processes batch N
- **Large embedding batches** — batch size 256 for Qwen3-Embedding-8B (small model, GPU can handle it)

## Web UI

Chat interface for querying the knowledge base with multi-query expansion and RAG.

**On IDUN (HPC):**
```bash
bash scripts/start_api.sh --config config.yaml --port 8001
# From Mac:
ssh -N -L 8001:localhost:8001 ecschoye@idun-login1.hpc.ntnu.no
# Open static/index.html?api=http://localhost:8001
```

**Locally (requires Ollama + KB files):**
```bash
ollama serve  # in another terminal
ollama pull qwen3-embedding:8b
bash scripts/start_local.sh
# Open static/index.html
```

## Configuration

Three config files for different deployment targets:

| File | Backend | Use case |
|------|---------|----------|
| `config.yaml` | vLLM | HPC (IDUN) |
| `config-ollama.yaml` | Ollama | Local (Mac) |
| `config-openrouter.yaml` | OpenRouter | API-based |

Key config sections:

- **paths** — Input/output directories for each pipeline stage
- **extract** — `ocr_fallback`, `column_detection`
- **chunk** — `token_size`, `overlap`, `flex_pct`, `protect_blocks`
- **nuggets** — LLM model, extraction/quality/augmentation parameters, `max_workers`
- **embed** — Embedding model, backend selection, `batch_size`, `embed_workers`, instruction text
- **store** — ChromaDB collection and SQLite settings
- **retrieval** — RRF constant, reranker weights, per-mode overrides

## Directory Structure

```
corpus/
  manifest.json          # Paper metadata (1,564 entries)
  texts/                 # Extracted text per paper
  chunks/                # Token-chunked documents
  nuggets/               # Raw QA nuggets per paper
  nuggets_unified/       # Unified pipeline output (quality-filtered + augmented)
kb/
  embeddings.npy         # Embedding matrix (153k × 4096)
  nuggets_with_embeddings.json
  chromadb/              # ChromaDB vector index
  nuggets.db             # SQLite metadata + FTS5
  feedback.db            # User feedback ratings
slurm/                   # SLURM job scripts (7 scripts)
static/
  index.html             # React 18 frontend (single file, no build step)
.claude/
  commands/              # 11 thesis-writing slash commands
  hooks/                 # cite_prefetch.py — auto-context for \cite{} references
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes (for query server) | Set in env or `~/.openrouter_key` |
| `THESIS_KB_CONFIG` | No | Override config path (default: `config.yaml`) |
| `VLLM_PORT` | No (auto-set by SLURM) | vLLM server port; Python reads this before config fallback |
| `HF_HOME` | No (HPC only) | Hugging Face cache directory |
