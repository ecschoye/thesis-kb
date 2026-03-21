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

# ── Phase 1: Unified extract+quality+augment (Qwen3.5-27B) ───────────

echo "=== Phase 1: Start vLLM (Qwen3.5-27B) ==="
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

echo "=== Running unified nugget pipeline ==="
python -m src.nuggets.unified -c config.yaml

# ── Offload LLM ──────────────────────────────────────────────────────

echo "=== Offloading Qwen3.5-27B ==="
kill "$VLLM_PID"
wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""
echo "Waiting for GPU memory release..."
sleep 10

# ── Phase 2: Embedding + Store (Qwen3-Embedding-8B) ──────────────────

echo "=== Phase 2: Start vLLM (Qwen3-Embedding-8B) ==="
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

echo "=== Embed nuggets ==="
python -m src.embed -c config.yaml

echo "=== Build KB ==="
python -m src.store -c config.yaml

echo "=== Unified pipeline complete ==="
