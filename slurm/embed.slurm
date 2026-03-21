#!/bin/bash
#SBATCH --job-name=kb-embed
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

# Embed nuggets using Qwen3-Embedding-8B via vLLM.
# Requires: nuggets in corpus/nuggets/ or corpus/nuggets_unified/

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
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Starting vLLM (Qwen3-Embedding-8B) ==="
vllm serve Qwen/Qwen3-Embedding-8B \
    --task embed \
    --port 8000 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 60); do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM died"; exit 1; }
    curl -s http://localhost:8000/health >/dev/null 2>&1 && { echo "Ready ($((i*10))s)"; break; }
    [ "$i" -eq 60 ] && { echo "ERROR: vLLM timeout"; exit 1; }
    sleep 10
done

echo "=== Embed nuggets ==="
python -m src.embed -c config.yaml

echo "=== Done ==="
