#!/bin/bash
#SBATCH --job-name=kb-rebuild
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-16:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/rebuild_%j.out
#SBATCH --error=logs/rebuild_%j.err

# Full pipeline rebuild after audit improvements.
#
# Usage:
#   sbatch slurm/full_rebuild.slurm           # full run (CPU + GPU)
#   bash slurm/full_rebuild.slurm --cpu-only  # CPU steps only (run on Mac)
#   sbatch slurm/full_rebuild.slurm --gpu-only # GPU steps only (after CPU done)

set -euo pipefail

# ── Parse flags ──────────────────────────────────────────────────────

RUN_CPU=true
RUN_GPU=true

for arg in "$@"; do
    case "$arg" in
        --cpu-only)  RUN_GPU=false ;;
        --gpu-only)  RUN_CPU=false ;;
        --help|-h)
            echo "Usage: full_rebuild.slurm [--cpu-only|--gpu-only]"
            echo ""
            echo "  (no flag)    Run full pipeline (CPU + GPU)"
            echo "  --cpu-only   Run CPU steps only: re-enrich, extract, chunk"
            echo "               Safe to run on Mac before submitting GPU job"
            echo "  --gpu-only   Run GPU steps only: nuggets, quality, augment, embed, store"
            echo "               Use after --cpu-only has completed"
            exit 0
            ;;
        *) echo "Unknown flag: $arg (use --help)"; exit 1 ;;
    esac
done

# ── Environment setup ────────────────────────────────────────────────

# Only load HPC modules if running on cluster (not Mac)
if command -v module &> /dev/null; then
    module purge
    module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 Tesseract/5.3.4-GCCcore-13.3.0
    source .venv/bin/activate
    export HF_HOME=/cluster/work/ecschoye/huggingface
    export HF_HUB_ENABLE_HF_TRANSFER=0
else
    # Mac / local — assume venv is already active or activate it
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/.."
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
fi

export PYTHONUNBUFFERED=1
mkdir -p logs

VLLM_PID=""

cleanup() {
    echo "=== Cleanup ==="
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_vllm() {
    echo "Waiting for vLLM server..."
    for i in $(seq 1 60); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM process died during startup"
            exit 1
        fi
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM ready after $((i * 10)) seconds"
            return 0
        fi
        if [ "$i" -eq 60 ]; then
            echo "ERROR: vLLM failed to start within 10 minutes"
            exit 1
        fi
        sleep 10
    done
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID $VLLM_PID)..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
        echo "Waiting for GPU memory release..."
        sleep 10
    fi
}

# ── Phase 0: CPU-only steps ──────────────────────────────────────────

if [ "$RUN_CPU" = true ]; then
    echo ""
    echo "============================================"
    echo "  Phase 0: Metadata + Extract + Chunk (CPU)"
    echo "============================================"

    echo ""
    echo "=== Step 0a: Re-enrich paper metadata via Semantic Scholar ==="
    python -m src.acquire.ingest --re-enrich
    echo "Re-enrich complete."

    echo ""
    echo "=== Step 0b: Clear old extraction/chunk/nugget outputs ==="
    rm -rf corpus/texts corpus/chunks corpus/nuggets corpus/nuggets_quality corpus/nuggets_augmented
    echo "Cleared: texts, chunks, nuggets, quality, augmented"

    echo ""
    echo "=== Step 0c: Re-extract text from PDFs (improved section detection) ==="
    python -m src.extract

    echo ""
    echo "=== Step 0d: Re-chunk texts (800 tokens, 100 overlap) ==="
    python -m src.chunk

    if [ "$RUN_GPU" = false ]; then
        echo ""
        echo "============================================"
        echo "  CPU steps complete!"
        echo "============================================"
        echo ""
        echo "Next: submit GPU job with --gpu-only"
        echo "  sbatch slurm/full_rebuild.slurm --gpu-only"
        exit 0
    fi
fi

# ── Phase 1: Nugget extraction + quality + augment (Qwen3.5-27B) ────

if [ "$RUN_GPU" = true ]; then

    # Verify CPU outputs exist
    if [ ! -d "corpus/chunks" ] || [ -z "$(ls corpus/chunks/ 2>/dev/null)" ]; then
        echo "ERROR: corpus/chunks/ is empty. Run --cpu-only first."
        exit 1
    fi

    echo ""
    echo "============================================"
    echo "  Phase 1: Nugget Extraction (Qwen3.5-27B)"
    echo "============================================"

    vllm serve Qwen/Qwen3.5-27B \
        --port 8000 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --dtype bfloat16 \
        --enable-chunked-prefill \
        --max-num-seqs 128 \
        --trust-remote-code &
    VLLM_PID=$!
    wait_for_vllm

    echo ""
    echo "=== Step 1: Extract nuggets ==="
    python -m src.nuggets

    echo ""
    echo "=== Step 2: Quality scoring ==="
    python -m src.nuggets.quality_main

    echo ""
    echo "=== Step 3: Augmentation ==="
    python -m src.nuggets.augment_main

    stop_vllm

    # ── Phase 2: Embedding + Store (Qwen3-Embedding-8B) ─────────────

    echo ""
    echo "============================================"
    echo "  Phase 2: Embed + Build KB (Qwen3-Embed-8B)"
    echo "============================================"

    vllm serve Qwen/Qwen3-Embedding-8B \
        --convert embed \
        --port 8000 \
        --max-model-len 1024 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 &
    VLLM_PID=$!
    wait_for_vllm

    echo ""
    echo "=== Step 4: Embed nuggets ==="
    python -m src.embed

    echo ""
    echo "=== Step 5: Build KB (ChromaDB + SQLite) ==="
    python -m src.store

    stop_vllm
fi

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
