#!/bin/bash
#SBATCH --job-name=kb-chunk
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=0-00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/chunk_%j.out
#SBATCH --error=logs/chunk_%j.err

# Chunk extracted texts into token windows.
# CPU-only. No GPU needed.

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

export PYTHONUNBUFFERED=1
mkdir -p logs

echo "=== Chunk texts ==="
python -m src.chunk -c config.yaml

echo "=== Done ==="
