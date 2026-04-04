#!/bin/bash
#SBATCH --job-name=kb-test-unified
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_unified_%j.out
#SBATCH --error=logs/test_unified_%j.err

# Test run: 3 papers through unified pipeline using Qwen3.5-9B on V100.
#
# Usage:
#   sbatch slurm/test_unified.slurm
#   sbatch slurm/test_unified.slurm --limit 5
#   sbatch slurm/test_unified.slurm --paper-ids 1305_2982,1409_4842

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source .venv/bin/activate

export HF_HOME=/cluster/work/ecschoye/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TIKTOKEN_CACHE_DIR=/cluster/work/ecschoye/thesis-kb/.cache/tiktoken
export PYTHONUNBUFFERED=1
mkdir -p logs

export VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))

# Smaller model for testing — fits on V100 (32GB)
MODEL="Qwen/Qwen3.5-9B"
MAX_NUM_SEQS=16

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model: $MODEL | max-num-seqs: $MAX_NUM_SEQS | port: $VLLM_PORT"

# Default: 3 small papers if no flags given
EXTRA_FLAGS="${@:---limit 3 --paper-ids 1305_2982,1409_4842,1409_1556}"

VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Starting vLLM ($MODEL) ==="
vllm serve $MODEL \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-seqs $MAX_NUM_SEQS \
    --trust-remote-code &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 60); do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM died"; exit 1; }
    curl -s http://localhost:$VLLM_PORT/health >/dev/null 2>&1 && { echo "Ready ($((i*10))s)"; break; }
    [ "$i" -eq 60 ] && { echo "ERROR: vLLM timeout"; exit 1; }
    sleep 10
done

echo "=== Test unified pipeline ${EXTRA_FLAGS} ==="
export MAX_NUM_SEQS
python -m src.nuggets.unified -c config.yaml ${EXTRA_FLAGS}

echo "=== Done ==="
