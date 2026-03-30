#!/bin/bash
#SBATCH --job-name=kb-cpu
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=0-01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/cpu_%j.out
#SBATCH --error=logs/cpu_%j.err

# CPU-only stages: extract → chunk → code extract.
# No GPU needed. Run before the GPU pipeline:
#
#   CPU_JOB=$(sbatch --parsable slurm/cpu_stages.slurm)
#   sbatch --dependency=afterok:$CPU_JOB --time=1-00:00:00 slurm/full_pipeline.slurm

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 tesseract/5.5.0-GCCcore-13.3.0
source .venv/bin/activate

export PYTHONUNBUFFERED=1
mkdir -p logs

echo "=========================================="
echo "  CPU Stages: Backfill + Extract + Chunk"
echo "=========================================="

echo "=== Backfill manifest metadata via S2 ==="
python scripts/backfill_metadata.py

echo "=== Extract text from PDFs ==="
python -m src.extract -c config.yaml

echo "=== Chunk texts ==="
python -m src.chunk -c config.yaml

echo "=== Extract code chunks (AST) ==="
python -m src.code_extract -c config.yaml

echo "=========================================="
echo "  CPU stages complete"
echo "=========================================="
