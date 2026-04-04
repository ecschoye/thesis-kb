#!/bin/bash
#SBATCH --job-name=kb-full
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu80g"
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ecschoye@stud.ntnu.no

# GPU pipeline: unified nuggets → embed → store.
# Run CPU stages first (slurm/cpu_stages.slurm) or inline with --cpu flag.
#
# Usage:
#   # CPU stages first, then GPU:
#   CPU_JOB=$(sbatch --parsable slurm/cpu_stages.slurm)
#   sbatch --dependency=afterok:$CPU_JOB --time=1-00:00:00 slurm/full_pipeline.slurm
#
#   # Or all-in-one (includes CPU stages):
#   sbatch --time=1-00:00:00 slurm/full_pipeline.slurm --cpu
#
#   # With extra args for unified pipeline:
#   sbatch slurm/full_pipeline.slurm --regenerate --oldest 500
#   sbatch slurm/full_pipeline.slurm --regenerate
#   sbatch slurm/full_pipeline.slurm --regenerate --paper-ids 2401_17151,2301_00456
#
# Extra args (except --cpu) are passed to `python -m src.nuggets.unified`.
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
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR=/cluster/work/ecschoye/thesis-kb/.cache/tiktoken
export PYTHONUNBUFFERED=1
mkdir -p logs

# Parse --cpu flag (run CPU stages inline) and pass rest to unified pipeline
RUN_CPU=false
EXTRA_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--cpu" ]; then
        RUN_CPU=true
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done
EXTRA_ARGS="${EXTRA_ARGS# }"  # trim leading space
echo "Extra args: ${EXTRA_ARGS:-none} (cpu=$RUN_CPU)"

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

SEQS_PER_INSTANCE=96
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
WATCHDOG_PID=""

# Per-instance GPU assignments (indexed by instance number)
declare -a INSTANCE_GPU_MAP
declare -a INSTANCE_PORT_MAP

cleanup() {
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null && wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
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
        kill -0 "$pid" 2>/dev/null || { echo "ERROR: vLLM (port $port) died during startup"; return 1; }
        curl -s "http://localhost:$port/health" >/dev/null 2>&1 && { echo "Ready port $port ($((i*3))s)"; return 0; }
        [ "$i" -eq 200 ] && { echo "ERROR: vLLM timeout (port $port)"; return 1; }
        sleep 3
    done
}

# Launch a single vLLM instance (used for initial start and restarts)
# Args: instance_index model [extra_args...]
launch_vllm_instance() {
    local idx=$1 model=$2
    shift 2
    local port=${INSTANCE_PORT_MAP[$idx]}
    local gpus=${INSTANCE_GPU_MAP[$idx]}
    echo "Launching vLLM instance $idx on GPUs $gpus, port $port (model=$model)"
    (
        export CUDA_VISIBLE_DEVICES=$gpus
        exec vllm serve "$model" --port $port "$@"
    ) &
    VLLM_PIDS[$idx]=$!
    wait_for_vllm $port ${VLLM_PIDS[$idx]}
}

# Background watchdog: checks health every 30s, restarts dead instances
# Args: model [extra_vllm_args...]
# Runs in a subshell — snapshots INSTANCE_PORT_MAP/INSTANCE_GPU_MAP at fork time
# and launches replacement vLLM processes directly.
start_watchdog() {
    local model=$1
    shift
    local extra_args=("$@")
    # Snapshot maps for the subshell (arrays don't survive naively)
    local _ports="${INSTANCE_PORT_MAP[*]}"
    local _gpus="${INSTANCE_GPU_MAP[*]}"
    local _tp=$TP_SIZE
    local _n=$NUM_INSTANCES
    local _seqs=$SEQS_PER_INSTANCE
    (
        read -ra PORTS <<< "$_ports"
        read -ra GPUS <<< "$_gpus"
        sleep 60  # initial grace period
        while true; do
            for idx in $(seq 0 $((_n - 1))); do
                port=${PORTS[$idx]}
                gpus=${GPUS[$idx]}
                if ! curl -s --max-time 5 "http://localhost:$port/health" >/dev/null 2>&1; then
                    echo "WATCHDOG: instance $idx (port $port) is down, restarting..."
                    # Kill anything still bound to that port
                    lsof -ti ":$port" 2>/dev/null | xargs -r kill 2>/dev/null || true
                    sleep 5  # let GPU memory free
                    echo "WATCHDOG: launching vLLM on GPUs $gpus, port $port"
                    CUDA_VISIBLE_DEVICES=$gpus vllm serve "$model" --port $port "${extra_args[@]}" &
                    new_pid=$!
                    # Wait for it to become healthy
                    for attempt in $(seq 1 200); do
                        kill -0 "$new_pid" 2>/dev/null || { echo "WATCHDOG: restart died for instance $idx"; break; }
                        curl -s "http://localhost:$port/health" >/dev/null 2>&1 && {
                            echo "WATCHDOG: instance $idx back up ($((attempt*3))s)"
                            break
                        }
                        sleep 3
                    done
                fi
            done
            sleep 30
        done
    ) &
    WATCHDOG_PID=$!
    echo "Watchdog started (pid=$WATCHDOG_PID, interval=30s)"
}

stop_watchdog() {
    if [ -n "$WATCHDOG_PID" ]; then
        kill "$WATCHDOG_PID" 2>/dev/null; wait "$WATCHDOG_PID" 2>/dev/null || true
        WATCHDOG_PID=""
        echo "Watchdog stopped"
    fi
}

stop_vllm() {
    stop_watchdog
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"; wait "$pid" 2>/dev/null || true
        fi
    done
    VLLM_PIDS=()
    echo "GPU memory release..."
    sleep 3
}

# ── Phase 1 (optional): CPU stages ───────────────────────────────────

if [ "$RUN_CPU" = true ]; then
    echo ""
    echo "=========================================="
    echo "  Phase 1: Extract + Chunk (CPU)"
    echo "=========================================="

    echo "=== Extract text from PDFs ==="
    python -m src.extract -c config.yaml

    echo "=== Chunk texts ==="
    python -m src.chunk -c config.yaml

    echo "=== Extract code chunks (AST) ==="
    python -m src.code_extract -c config.yaml
else
    echo "Skipping CPU stages (run slurm/cpu_stages.slurm first, or pass --cpu)"
fi

# ── Phase 2: Unified nuggets (Qwen3.5-27B) ───────────────────────────

echo ""
echo "=========================================="
echo "  Phase 2: Unified Nuggets (Qwen3.5-27B)"
echo "  $NUM_INSTANCES instance(s), TP=$TP_SIZE each"
echo "=========================================="

# Build per-instance GPU and port maps
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    INSTANCE_PORT_MAP[$i]=$((BASE_PORT + i * 10))
    GPU_START=$((i * TP_SIZE))
    INSTANCE_GPU_MAP[$i]=$(echo "${GPU_IDS[@]:$GPU_START:$TP_SIZE}" | tr ' ' ',')
done

NUGGET_VLLM_ARGS=(
    --max-model-len 16384
    --gpu-memory-utilization 0.92
    --dtype bfloat16
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-num-seqs $SEQS_PER_INSTANCE
    --tensor-parallel-size $TP_SIZE
    --trust-remote-code
)

# Launch instances sequentially to avoid NCCL port collisions
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    launch_vllm_instance "$i" "Qwen/Qwen3.5-27B" "${NUGGET_VLLM_ARGS[@]}"
done

# Start watchdog to auto-restart crashed instances
start_watchdog "Qwen/Qwen3.5-27B" "${NUGGET_VLLM_ARGS[@]}"

echo "=== Unified nugget pipeline ==="
python -m src.nuggets.unified -c config.yaml $EXTRA_ARGS

# If not already reviewing/regenerating, also review existing nuggets
if [[ ! "$EXTRA_ARGS" =~ --review ]] && [[ ! "$EXTRA_ARGS" =~ --regenerate ]]; then
    echo "=== Review existing nuggets ==="
    python -m src.nuggets.unified -c config.yaml --review
fi

echo "=== Code nugget extraction ==="
python -m src.code_extract.nuggets -c config.yaml

stop_vllm

# ── Phase 3: Embed + Store (Qwen3-Embedding-8B) ──────────────────────

echo ""
echo "=========================================="
echo "  Phase 3: Embed + Store (Qwen3-Embed-8B)"
echo "  $NUM_INSTANCES instance(s), TP=$TP_SIZE each"
echo "=========================================="

EMBED_VLLM_ARGS=(
    --convert embed
    --max-model-len 2048
    --gpu-memory-utilization 0.92
    --dtype bfloat16
    --tensor-parallel-size $TP_SIZE
)

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    launch_vllm_instance "$i" "Qwen/Qwen3-Embedding-8B" "${EMBED_VLLM_ARGS[@]}"
done

start_watchdog "Qwen/Qwen3-Embedding-8B" "${EMBED_VLLM_ARGS[@]}"

echo "=== Embed paper nuggets ==="
python -m src.embed -c config.yaml

echo "=== Embed code nuggets + merge + build KB ==="
python -m src.code_extract.embed_and_merge -c config.yaml

stop_vllm

echo ""
echo "=========================================="
echo "  Pipeline complete"
echo "=========================================="
