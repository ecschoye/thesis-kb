#!/bin/bash
# Nightly nugget review via free OpenRouter models.
# Reviews the N oldest papers in corpus/nuggets_unified/.
# Intended to run from crontab on the login node (no GPU needed).
#
# With free models (1000 req/day, 20 req/min), budget is roughly:
# ~30 req/paper (quality batches + improve + gap-fill) → ~30 papers/night
#
# Usage:
#   ./scripts/nightly_review.sh          # review 30 oldest (default)
#   ./scripts/nightly_review.sh 50       # review 50 oldest

set -euo pipefail
cd /cluster/work/ecschoye/thesis-kb

CONFIG="config-openrouter-free.yaml"
LOG="logs/nightly_review_$(date +%Y%m%d_%H%M%S).log"

mkdir -p logs

echo "=== Nightly review @ $(date) ===" | tee "$LOG"

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source .venv/bin/activate

python -m src.nuggets.unified -c "$CONFIG" --review >> "$LOG" 2>&1
EXIT_CODE=$?

# Clean up old logs (keep last 30 days)
find logs/ -name "nightly_review_*.log" -mtime +30 -delete 2>/dev/null || true

echo "=== Done (exit $EXIT_CODE) @ $(date) ===" | tee -a "$LOG"
exit $EXIT_CODE
