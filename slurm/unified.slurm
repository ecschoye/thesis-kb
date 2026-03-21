#!/bin/bash
#SBATCH --job-name=kb-unified
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/unified_%j.out
#SBATCH --error=logs/unified_%j.err

# Unified nugget pipeline: extract + quality + augment in one pass.
# Requires: chunks in corpus/chunks/
#
# Multi-GPU: change --gres=gpu:2 (or more) to enable tensor parallelism.
# vLLM parameters scale automatically based on allocated GPU count.
#
# Usage:
#   sbatch slurm/unified.slurm              # full: extract + quality + augment
#   sbatch slurm/unified.slurm --reprocess  # quality + augment on existing nuggets

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source .venv/bin/activate

export HF_HOME=/cluster/work/ecschoye/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
mkdir -p logs

# Dynamic port to avoid collisions when multiple jobs land on the same node
export VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))

# Detect GPU count and scale vLLM parameters
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
GPU_COUNT=${GPU_COUNT:-1}
MAX_NUM_SEQS=$((64 * GPU_COUNT))
echo "GPUs: $GPU_COUNT | max-num-seqs: $MAX_NUM_SEQS | port: $VLLM_PORT"

# Pass through any flags (e.g. --reprocess)
EXTRA_FLAGS="${@}"

VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Starting vLLM (Qwen3.5-27B) ==="
vllm serve Qwen/Qwen3.5-27B \
    --port $VLLM_PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-seqs $MAX_NUM_SEQS \
    --tensor-parallel-size $GPU_COUNT \
    --trust-remote-code &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 60); do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM died"; exit 1; }
    curl -s http://localhost:$VLLM_PORT/health >/dev/null 2>&1 && { echo "Ready ($((i*10))s)"; break; }
    [ "$i" -eq 60 ] && { echo "ERROR: vLLM timeout"; exit 1; }
    sleep 10
done

echo "=== Unified nugget pipeline ${EXTRA_FLAGS} ==="
python -m src.nuggets.unified -c config.yaml ${EXTRA_FLAGS}

echo "=== Done ==="
