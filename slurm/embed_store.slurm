#!/bin/bash
#SBATCH --job-name=kb-embed-store
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/embed_store_%j.out
#SBATCH --error=logs/embed_store_%j.err

# Re-embed nuggets (merging augmented versions) and rebuild KB.
#
# Prerequisites:
#   - corpus/nuggets/           (original nuggets)
#   - corpus/nuggets_augmented/ (augmented nuggets — merged automatically)
#   - config.yaml paths.augmented_dir must point to corpus/nuggets_augmented
#
# Usage:
#   sbatch slurm/embed_store.slurm

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

# ── Verify augmented nuggets exist ────────────────────────────────────

AUG_DIR="corpus/nuggets_augmented"
if [ ! -d "$AUG_DIR" ] || [ -z "$(ls "$AUG_DIR"/*.json 2>/dev/null | head -1)" ]; then
    echo "ERROR: $AUG_DIR is empty or missing. Run augmentation first."
    exit 1
fi
AUG_COUNT=$(ls "$AUG_DIR"/*.json 2>/dev/null | wc -l)
echo "Found $AUG_COUNT augmented nugget files in $AUG_DIR"

# ── Start vLLM embedding server ──────────────────────────────────────

echo ""
echo "============================================"
echo "  Embed + Store (Qwen3-Embedding-8B)"
echo "============================================"

vllm serve Qwen/Qwen3-Embedding-8B \
    --convert embed \
    --port 8000 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 &
VLLM_PID=$!

echo "Waiting for vLLM server..."
for i in $(seq 1 60); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died during startup"
        exit 1
    fi
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready after $((i * 10)) seconds"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: vLLM failed to start within 10 minutes"
        exit 1
    fi
    sleep 10
done

# ── Step 1: Embed nuggets (with augmented merge) ─────────────────────

echo ""
echo "=== Step 1: Embed nuggets (merging augmented) ==="
python -m src.embed

# ── Stop vLLM (embedding no longer needed) ───────────────────────────

echo "Stopping vLLM..."
kill "$VLLM_PID"
wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""

# ── Step 2: Build KB (CPU only) ──────────────────────────────────────

echo ""
echo "=== Step 2: Build KB (ChromaDB + SQLite + FTS5) ==="
python -m src.store

echo ""
echo "============================================"
echo "  Embed + Store complete!"
echo "============================================"
