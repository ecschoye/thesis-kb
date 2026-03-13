#!/bin/bash
#SBATCH --job-name=kb-qa
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/quality_augment_%j.out
#SBATCH --error=logs/quality_augment_%j.err

# Combined quality scoring + augmentation job.
# Shares a single vLLM instance (Qwen3.5-27B) for both steps.
#
# Usage:
#   sbatch slurm/quality_augment.slurm
#
# After completion, run:
#   sbatch slurm/embed_nuggets.slurm    # re-embed with new doc_instruction
#   bash scripts/rebuild_kb.sh           # rebuild ChromaDB + SQLite + FTS5

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

echo "=== Step 1: Start vLLM (Qwen3.5-27B) ==="
vllm serve Qwen/Qwen3.5-27B \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --enable-chunked-prefill \
    --max-num-seqs 64 \
    --trust-remote-code &
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

echo ""
echo "=== Step 2: Quality scoring (~144k nuggets) ==="
python -m src.nuggets.quality_main
echo "Quality scoring complete."

echo ""
echo "=== Step 3: Augmentation (improve weak + gap-fill) ==="
python -m src.nuggets.augment_main
echo "Augmentation complete."

echo ""
echo "============================================"
echo "  Quality + Augmentation complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. sbatch slurm/embed_nuggets.slurm  (re-embed with asymmetric instructions)"
echo "  2. bash scripts/rebuild_kb.sh          (rebuild ChromaDB + SQLite + FTS5)"
