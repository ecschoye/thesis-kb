#!/bin/bash
set -euo pipefail

echo "Submitting thesis-kb pipeline..."

JOB1=$(sbatch --parsable slurm/nugget_extract.slurm)
echo "Job 1 (extract+chunk+nuggets): $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/embed_nuggets.slurm)
echo "Job 2 (embed+store, depends on $JOB1): $JOB2"

echo ""
echo "Monitor: squeue -u $USER"
echo "Logs:    tail -f logs/nuggets_${JOB1}.out"
echo "         tail -f logs/embed_${JOB2}.out"
