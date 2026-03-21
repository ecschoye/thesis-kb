#!/bin/bash
#SBATCH --job-name=kb-store
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/store_%j.out
#SBATCH --error=logs/store_%j.err

# Build ChromaDB + SQLite knowledge base from embedded nuggets.
# CPU-only. No GPU needed.
# Requires: kb/nuggets_with_embeddings.json + kb/embeddings.npy

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

export PYTHONUNBUFFERED=1
mkdir -p logs

echo "=== Build KB ==="
python -m src.store -c config.yaml

echo "=== Done ==="
