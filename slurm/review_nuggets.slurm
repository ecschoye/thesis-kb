#!/bin/bash
#SBATCH --job-name=kb-review
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu80g"
#SBATCH --time=0-08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/review_%j.out
#SBATCH --error=logs/review_%j.err

# Review existing unified nuggets: score, improve weak, remove bad, gap-fill.
# Unlike --regenerate (which re-extracts from scratch), --review reads the
# current unified nuggets and refines them in-place.
#
# Usage:
#   sbatch slurm/review_nuggets.slurm                          # review ALL papers
#   sbatch slurm/review_nuggets.slurm --oldest 500             # review 500 oldest
#   sbatch slurm/review_nuggets.slurm --paper-ids 2401_17151   # review specific papers
#
# Extra args are passed to `python -m src.nuggets.unified --review`.
# After review, re-embeds and rebuilds the KB.
#
# Multi-instance scaling (same as full_pipeline.slurm):
#   --gres=gpu:2  → 1 instance  (TP=2, baseline)
#   --gres=gpu:4  → 2 instances (TP=2 each, ~2x throughput)

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source .venv/bin/activate

export HF_HOME=/cluster/work/ecschoye/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR=/cluster/work/ecschoye/thesis-kb/.cache/tiktoken
export PYTHONUNBUFFERED=1
mkdir -p logs

EXTRA_ARGS="${@}"
echo "Extra args: ${EXTRA_ARGS:-none}"

# ── GPU topology ────────────────────────────────────────────────────
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
export VLLM_PORT=$BASE_PORT

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
    for i in $(seq 1 200); do
        kill -0 "$pid" 2>/dev/null || { echo "ERROR: vLLM (port $port) died"; exit 1; }
        curl -s "http://localhost:$port/health" >/dev/null 2>&1 && { echo "Ready port $port ($((i*3))s)"; return 0; }
        [ "$i" -eq 200 ] && { echo "ERROR: vLLM timeout (port $port)"; exit 1; }
        sleep 3
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
    sleep 3
}

# ── Phase 1: Review nuggets (Qwen3.5-27B) ────────────────────────────

echo ""
echo "=========================================="
echo "  Phase 1: Review Nuggets (Qwen3.5-27B)"
echo "  $NUM_INSTANCES instance(s), TP=$TP_SIZE each"
echo "=========================================="

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i * 10))
    GPU_START=$((i * TP_SIZE))
    INSTANCE_GPUS=$(echo "${GPU_IDS[@]:$GPU_START:$TP_SIZE}" | tr ' ' ',')

    echo "Launching vLLM instance $i on GPUs $INSTANCE_GPUS, port $PORT"
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

echo "=== Review unified nuggets ==="
python -m src.nuggets.unified -c config.yaml --review $EXTRA_ARGS

stop_vllm

# ── Phase 2: Embed + Store (Qwen3-Embedding-8B) ──────────────────────

echo ""
echo "=========================================="
echo "  Phase 2: Embed + Store (Qwen3-Embed-8B)"
echo "  $NUM_INSTANCES instance(s), TP=$TP_SIZE each"
echo "=========================================="

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i * 10))
    GPU_START=$((i * TP_SIZE))
    INSTANCE_GPUS=$(echo "${GPU_IDS[@]:$GPU_START:$TP_SIZE}" | tr ' ' ',')

    echo "Launching embed instance $i on GPUs $INSTANCE_GPUS, port $PORT"
    (
        export CUDA_VISIBLE_DEVICES=$INSTANCE_GPUS
        exec vllm serve Qwen/Qwen3-Embedding-8B \
            --convert embed \
            --port $PORT \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.92 \
            --dtype bfloat16 \
            --tensor-parallel-size $TP_SIZE
    ) &
    VLLM_PIDS+=($!)
    wait_for_vllm $PORT ${VLLM_PIDS[$i]}
done

echo "=== Embed paper nuggets ==="
python -m src.embed -c config.yaml

echo "=== Embed code nuggets + merge + build KB ==="
python -m src.code_extract.embed_and_merge -c config.yaml

stop_vllm

echo ""
echo "=========================================="
echo "  Review pipeline complete"
echo "=========================================="
