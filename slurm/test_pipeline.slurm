#!/bin/bash
#SBATCH --job-name=kb-test
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_pipeline_%j.out
#SBATCH --error=logs/test_pipeline_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ecschoye@stud.ntnu.no

# Test full pipeline on a few papers using Qwen3.5-9B on a single V100.
# Runs: extract → chunk → unified nuggets → embed → store
#
# Usage:
#   sbatch slurm/test_pipeline.slurm                                   # 3 default papers
#   sbatch slurm/test_pipeline.slurm --paper-ids 1305_2982,1409_4842   # specific papers
#   sbatch slurm/test_pipeline.slurm --limit 5                         # first 5 unprocessed

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

export VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))

# Test model — fits on V100 (32GB)
LLM_MODEL="Qwen/Qwen3.5-9B"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
MAX_NUM_SEQS=16
export MAX_NUM_SEQS

# Default: 3 small papers
EXTRA_ARGS="${@:---limit 3 --paper-ids 1305_2982,1409_4842,1409_1556}"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "LLM: $LLM_MODEL | Embed: $EMBED_MODEL"
echo "Port: $VLLM_PORT | max-num-seqs: $MAX_NUM_SEQS"
echo "Args: $EXTRA_ARGS"

VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_vllm() {
    echo "Waiting for vLLM..."
    for i in $(seq 1 90); do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM died"; exit 1; }
        curl -s http://localhost:$VLLM_PORT/health >/dev/null 2>&1 && { echo "Ready ($((i*5))s)"; return 0; }
        [ "$i" -eq 90 ] && { echo "ERROR: vLLM timeout"; exit 1; }
        sleep 5
    done
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    sleep 3
}

# ── Phase 1: CPU stages ────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Phase 1: Extract + Chunk (CPU)"
echo "=========================================="

python -m src.extract -c config.yaml
python -m src.chunk -c config.yaml

# ── Phase 2: Unified nuggets ────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Phase 2: Unified Nuggets ($LLM_MODEL)"
echo "=========================================="

vllm serve $LLM_MODEL \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-seqs $MAX_NUM_SEQS \
    --trust-remote-code &
VLLM_PID=$!
wait_for_vllm

python -m src.nuggets.unified -c config.yaml $EXTRA_ARGS

stop_vllm

# ── Phase 3: Embed + Store ──────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Phase 3: Embed + Store ($EMBED_MODEL)"
echo "=========================================="

vllm serve $EMBED_MODEL \
    --convert embed \
    --port $VLLM_PORT \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 &
VLLM_PID=$!
wait_for_vllm

python -m src.embed -c config.yaml
python -m src.store -c config.yaml

stop_vllm

# ── Validation ──────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Validation"
echo "=========================================="

FAIL=0

# Check unified nuggets were produced
UNIFIED_COUNT=$(find corpus/nuggets_unified -name "*.json" -newer logs/test_pipeline_${SLURM_JOB_ID}.out 2>/dev/null | wc -l)
echo "Unified nugget files produced: $UNIFIED_COUNT"

# Check for scoring failures
SCORING_FAILED=$(grep -rl '"scoring_failed": true' corpus/nuggets_unified/ 2>/dev/null | wc -l)
if [ "$SCORING_FAILED" -gt 0 ]; then
    echo "WARNING: $SCORING_FAILED papers had scoring failures (quality LLM errors)"
fi

# Check for papers with 0 nuggets but removed entries (data loss pattern)
ZERO_NUGGETS=$(python -c "
import json, glob
bad = []
for f in glob.glob('corpus/nuggets_unified/*.json'):
    d = json.load(open(f))
    if d.get('num_nuggets', 0) == 0 and d.get('num_removed', 0) > 0:
        bad.append(f.split('/')[-1])
print(len(bad))
for b in bad[:5]: print(f'  {b}')
" 2>/dev/null)
echo "Papers with 0 kept + removals (data loss): $ZERO_NUGGETS"

# Check embeddings
if [ -f kb/embeddings.npy ]; then
    EMB_SHAPE=$(python -c "import numpy as np; e=np.load('kb/embeddings.npy'); print(f'{e.shape[0]} nuggets, {e.shape[1]}d')" 2>/dev/null)
    echo "Embeddings: $EMB_SHAPE"
else
    echo "ERROR: kb/embeddings.npy not found"
    FAIL=1
fi

# Check SQLite
if [ -f kb/nuggets.db ]; then
    DB_COUNT=$(python -c "import sqlite3; c=sqlite3.connect('kb/nuggets.db'); print(c.execute('SELECT COUNT(*) FROM nuggets').fetchone()[0])" 2>/dev/null)
    echo "SQLite nuggets: $DB_COUNT"
else
    echo "ERROR: kb/nuggets.db not found"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "=== Test pipeline PASSED — safe to run full pipeline ==="
else
    echo ""
    echo "=== Test pipeline FAILED — check errors above ==="
    exit 1
fi
