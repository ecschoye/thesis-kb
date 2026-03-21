#!/bin/bash
#SBATCH --job-name=kb-full
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

# Full pipeline: extract → chunk → unified nuggets → embed → store.
# Single job, handles model swaps between GPU phases.

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 Tesseract/5.3.4-GCCcore-13.3.0
source .venv/bin/activate

export HF_HOME=/cluster/work/ecschoye/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
mkdir -p logs

VLLM_PID=""

cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_vllm() {
    echo "Waiting for vLLM..."
    for i in $(seq 1 60); do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM died"; exit 1; }
        curl -s http://localhost:8000/health >/dev/null 2>&1 && { echo "Ready ($((i*10))s)"; return 0; }
        [ "$i" -eq 60 ] && { echo "ERROR: vLLM timeout"; exit 1; }
        sleep 10
    done
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
        echo "GPU memory release..."
        sleep 10
    fi
}

# ── Phase 1: CPU stages ──────────────────────────────────────────────

echo ""
echo "=========================================="
echo "  Phase 1: Extract + Chunk (CPU)"
echo "=========================================="

echo "=== Extract text from PDFs ==="
python -m src.extract -c config.yaml

echo "=== Chunk texts ==="
python -m src.chunk -c config.yaml

# ── Phase 2: Unified nuggets (Qwen3.5-27B) ───────────────────────────

echo ""
echo "=========================================="
echo "  Phase 2: Unified Nuggets (Qwen3.5-27B)"
echo "=========================================="

vllm serve Qwen/Qwen3.5-27B \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --enable-chunked-prefill \
    --max-num-seqs 64 \
    --trust-remote-code &
VLLM_PID=$!
wait_for_vllm

echo "=== Unified nugget pipeline ==="
python -m src.nuggets.unified -c config.yaml

stop_vllm

# ── Phase 3: Embed + Store (Qwen3-Embedding-8B) ──────────────────────

echo ""
echo "=========================================="
echo "  Phase 3: Embed + Store (Qwen3-Embed-8B)"
echo "=========================================="

vllm serve Qwen/Qwen3-Embedding-8B \
    --task embed \
    --port 8000 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 &
VLLM_PID=$!
wait_for_vllm

echo "=== Embed nuggets ==="
python -m src.embed -c config.yaml

echo "=== Build KB ==="
python -m src.store -c config.yaml

stop_vllm

echo ""
echo "=========================================="
echo "  Pipeline complete"
echo "=========================================="
