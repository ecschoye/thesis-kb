#!/bin/bash
#SBATCH --job-name=kb-improve
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/improve_%j.out
#SBATCH --error=logs/improve_%j.err

# Full retrieval improvement pipeline:
#   1. Quality scoring (Qwen3.5-27B)
#   2. Augmentation   (Qwen3.5-27B)
#   3. Re-embed        (Qwen3-Embedding-8B, asymmetric doc_instruction)
#   4. Rebuild KB      (ChromaDB + SQLite + FTS5, CPU)
#
# Usage:
#   sbatch slurm/quality_augment_embed.slurm

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source .venv/bin/activate

export HF_HOME=/cluster/work/ecschoye/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
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

# ── Phase 1: Quality + Augment (Qwen3.5-27B) ────────────────────────

echo ""
echo "============================================"
echo "  Phase 1: Quality + Augment (Qwen3.5-27B)"
echo "============================================"

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

echo ""
echo "=== Step 1: Quality scoring (~144k nuggets) ==="
python -m src.nuggets.quality_main
echo "Quality scoring complete."

echo ""
echo "=== Step 2: Augmentation (improve weak + gap-fill) ==="
python -m src.nuggets.augment_main
echo "Augmentation complete."

stop_vllm

# ── Phase 2: Embed + Build KB (Qwen3-Embedding-8B) ──────────────────

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
echo "=== Step 3: Embed nuggets (asymmetric doc_instruction) ==="
python -m src.embed

stop_vllm

echo ""
echo "=== Step 4: Build KB (ChromaDB + SQLite + FTS5) ==="
python -m src.store

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
