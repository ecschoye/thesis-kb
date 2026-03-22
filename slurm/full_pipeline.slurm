#!/bin/bash
#SBATCH --job-name=kb-full
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

# Full pipeline: extract → chunk → unified nuggets → embed → store.
# Single job, handles model swaps between GPU phases.
#
# Multi-instance scaling (recommended for >=4 GPUs):
#   With N GPUs, launches N/2 independent TP=2 vLLM instances.
#   Each instance gets its own GPU pair and port.
#   Python workers round-robin requests across all instances.
#   Near-linear throughput scaling vs single-instance tensor parallelism.
#
# Examples:
#   --gres=gpu:2  → 1 instance  (TP=2, baseline)
#   --gres=gpu:4  → 2 instances (TP=2 each, ~2x throughput)
#   --gres=gpu:6  → 3 instances (TP=2 each, ~3x throughput)

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 tesseract/5.5.0-GCCcore-13.3.0
source .venv/bin/activate

export HF_HOME=/cluster/work/ecschoye/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
mkdir -p logs

# ── GPU topology ────────────────────────────────────────────────────
# Split GPUs into TP=2 pairs for multi-instance serving
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES not set"; exit 1
fi
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_IDS[@]}
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "ERROR: Need at least 2 GPUs, got $GPU_COUNT"; exit 1
fi
TP_SIZE=2
NUM_INSTANCES=$((GPU_COUNT / TP_SIZE))
if [ $((GPU_COUNT % TP_SIZE)) -ne 0 ]; then
    echo "WARNING: $GPU_COUNT GPUs not evenly divisible by TP=$TP_SIZE, using $((NUM_INSTANCES * TP_SIZE))/$GPU_COUNT GPUs"
fi

SEQS_PER_INSTANCE=128
export MAX_NUM_SEQS=$((SEQS_PER_INSTANCE * NUM_INSTANCES))

# Build port list and CUDA_VISIBLE_DEVICES per instance
# Use high port range (10000-19999) to avoid collisions with common services
BASE_PORT=$((10000 + SLURM_JOB_ID % 10000))
VLLM_PORTS=""
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i * 10))
    if [ -z "$VLLM_PORTS" ]; then
        VLLM_PORTS="$PORT"
    else
        VLLM_PORTS="$VLLM_PORTS,$PORT"
    fi
done
export VLLM_PORTS
export VLLM_PORT=$BASE_PORT  # backwards compat for single-instance code paths

echo "GPUs: $GPU_COUNT | instances: $NUM_INSTANCES (TP=$TP_SIZE each)"
echo "Ports: $VLLM_PORTS | max-num-seqs/instance: $SEQS_PER_INSTANCE | total workers: $MAX_NUM_SEQS"

VLLM_PIDS=()

cleanup() {
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"; wait "$pid" 2>/dev/null || true
        fi
    done
    VLLM_PIDS=()
}
trap cleanup EXIT

wait_for_vllm() {
    local port=$1 pid=$2
    echo "Waiting for vLLM on port $port..."
    for i in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || { echo "ERROR: vLLM (port $port) died"; exit 1; }
        curl -s "http://localhost:$port/health" >/dev/null 2>&1 && { echo "Ready port $port ($((i*10))s)"; return 0; }
        [ "$i" -eq 60 ] && { echo "ERROR: vLLM timeout (port $port)"; exit 1; }
        sleep 10
    done
}

stop_vllm() {
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"; wait "$pid" 2>/dev/null || true
        fi
    done
    VLLM_PIDS=()
    echo "GPU memory release..."
    sleep 10
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
echo "  $NUM_INSTANCES instance(s), TP=$TP_SIZE each"
echo "=========================================="

# Launch instances sequentially (wait for each to be ready before next,
# so model loading doesn't compete for GPU memory)
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i * 10))
    GPU_START=$((i * TP_SIZE))
    INSTANCE_GPUS=$(echo "${GPU_IDS[@]:$GPU_START:$TP_SIZE}" | tr ' ' ',')

    echo "Launching vLLM instance $i on GPUs $INSTANCE_GPUS, port $PORT"
    # Subshell with export so vLLM worker subprocesses inherit the GPU mask
    (
        export CUDA_VISIBLE_DEVICES=$INSTANCE_GPUS
        exec vllm serve Qwen/Qwen3.5-27B \
            --port $PORT \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.92 \
            --dtype bfloat16 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --max-num-seqs $SEQS_PER_INSTANCE \
            --tensor-parallel-size $TP_SIZE \
            --trust-remote-code
    ) &
    VLLM_PIDS+=($!)
    wait_for_vllm $PORT ${VLLM_PIDS[$i]}
done

echo "=== Unified nugget pipeline ==="
python -m src.nuggets.unified -c config.yaml

stop_vllm

# ── Phase 3: Embed + Store (Qwen3-Embedding-8B) ──────────────────────

echo ""
echo "=========================================="
echo "  Phase 3: Embed + Store (Qwen3-Embed-8B)"
echo "  $NUM_INSTANCES instance(s), TP=$TP_SIZE each"
echo "=========================================="

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i * 10))
    GPU_START=$((i * TP_SIZE))
    INSTANCE_GPUS=$(echo "${GPU_IDS[@]:$GPU_START:$TP_SIZE}" | tr ' ' ',')

    (
        export CUDA_VISIBLE_DEVICES=$INSTANCE_GPUS
        exec vllm serve Qwen/Qwen3-Embedding-8B \
            --task embed \
            --port $PORT \
            --max-model-len 1024 \
            --gpu-memory-utilization 0.92 \
            --dtype bfloat16 \
            --tensor-parallel-size $TP_SIZE
    ) &
    VLLM_PIDS+=($!)
    wait_for_vllm $PORT ${VLLM_PIDS[$i]}
done

echo "=== Embed nuggets ==="
python -m src.embed -c config.yaml

echo "=== Build KB ==="
python -m src.store -c config.yaml

stop_vllm

echo ""
echo "=========================================="
echo "  Pipeline complete"
echo "=========================================="
