#!/bin/bash
#SBATCH --job-name=kb-extract
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# Extract text from PDFs (column detection + OCR fallback).
# CPU-only. No GPU needed.

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 Tesseract/5.3.4-GCCcore-13.3.0
source .venv/bin/activate

export PYTHONUNBUFFERED=1
mkdir -p logs

echo "=== Extract text from PDFs ==="
python -m src.extract -c config.yaml

echo "=== Done ==="
