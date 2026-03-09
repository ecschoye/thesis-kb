# thesis-kb

Knowledge base pipeline for a master's thesis on event cameras, RGB-Event fusion, spiking neural networks, and object detection. Processes academic papers into searchable QA nuggets backed by vector embeddings.

## Pipeline

```
Acquire → Extract → Chunk → Nuggets → Quality → Augment → Embed → Store → Query
```

| Stage | Module | Description |
|-------|--------|-------------|
| **Acquire** | `src.acquire` | Ingest PDFs from local dir or Zotero export, enrich metadata via Semantic Scholar |
| **Extract** | `src.extract` | Extract text from PDFs using PyMuPDF with section detection |
| **Chunk** | `src.chunk` | Token-based chunking (400 tokens, 50 overlap) with section awareness |
| **Nuggets** | `src.nuggets` | LLM-based QA nugget extraction using Qwen3.5-27B via vLLM |
| **Quality** | `src.nuggets.quality_main` | Score nuggets on 6 quality dimensions |
| **Augment** | `src.nuggets.augment_main` | Improve weak nuggets and gap-fill sparse chunks |
| **Embed** | `src.embed` | Embed nuggets using Qwen3-Embedding-8B via vLLM |
| **Store** | `src.store` | Build ChromaDB vector index + SQLite metadata DB |
| **Query** | `src.query` | Semantic search with multi-query expansion |

## Setup

```bash
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Each stage runs as a Python module:

```bash
python -m src.acquire -c config.yaml
python -m src.extract -c config.yaml
python -m src.chunk -c config.yaml
python -m src.nuggets -c config.yaml          # extraction
python -m src.nuggets.quality_main -c config.yaml
python -m src.nuggets.augment_main -c config.yaml
python -m src.embed -c config.yaml
python -m src.store -c config.yaml
```

### SLURM (HPC)

Full reprocess pipeline (quality + augment + embed + store):
```bash
sbatch slurm/reprocess_pipeline.slurm
```

Individual stages:
```bash
sbatch slurm/nugget_extract.slurm
sbatch slurm/nugget_quality.slurm
sbatch slurm/nugget_augment.slurm
sbatch slurm/embed_nuggets.slurm
```

## Configuration

All settings are in `config.yaml`. Key sections:

- **paths**: Input/output directories for each stage
- **nuggets**: LLM model, extraction/quality/augmentation parameters
- **embed**: Embedding model and backend (`vllm` for HPC, `ollama` for local)
- **store**: ChromaDB collection and SQLite settings

## Directory Structure

```
corpus/
  texts/          # Extracted text from PDFs
  chunks/         # Token-chunked documents
  nuggets/        # Base QA nuggets per paper
  nuggets_quality/# Quality scores per paper
  nuggets_augmented/ # Improved + gap-filled nuggets
kb/
  embeddings.npy  # Embedding matrix
  nuggets_with_embeddings.json
  chroma/         # ChromaDB vector index
  nuggets.db      # SQLite metadata
```
