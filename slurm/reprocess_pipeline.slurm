#!/bin/bash
#SBATCH --job-name=kb-reprocess
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/reprocess_%j.out
#SBATCH --error=logs/reprocess_%j.err

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

# ── Phase 1: Quality + Augmentation (Qwen3.5-27B) ──────────────────────

echo "=== Phase 1: Start vLLM (Qwen3.5-27B) for quality + augmentation ==="
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

echo "=== Step 1: Run nugget quality checks ==="
python -m src.nuggets.quality_main

echo "=== Step 2: Run nugget augmentation ==="
python -m src.nuggets.augment_main

# ── Offload Qwen3.5-27B ────────────────────────────────────────────────

echo "=== Offloading Qwen3.5-27B ==="
kill "$VLLM_PID"
wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""
echo "Qwen3.5-27B stopped, waiting for GPU memory release..."
sleep 10

# ── Phase 2: Embedding + Store (Qwen3-Embedding-8B) ────────────────────

echo "=== Phase 2: Start vLLM (Qwen3-Embedding-8B) for embedding ==="
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

echo "=== Step 3: Embed nuggets ==="
python -m src.embed

echo "=== Step 4: Build KB ==="
python -m src.store

echo "=== Reprocessing pipeline complete ==="
