#!/bin/bash
#SBATCH --job-name=kb-nuggets-ollama
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/nuggets_%j.out
#SBATCH --error=logs/nuggets_%j.err

set -euo pipefail
module load Python/3.13.5-GCCcore-14.3.0
module load ollama/0.11.10-GCCcore-14.3.0-CUDA-12.9.1
source .venv/bin/activate

mkdir -p logs

# --- Ollama setup ---
# Use node-local storage for model weights (faster, avoids home quota)
export OLLAMA_MODELS="${TMPDIR:-/tmp}/ollama_models_${SLURM_JOB_ID}"
export OLLAMA_HOST="127.0.0.1:11434"
mkdir -p "$OLLAMA_MODELS"

echo "=== Starting Ollama server ==="
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama server..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready after ${i} seconds"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Ollama failed to start"
        kill $OLLAMA_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Pull the model ---
MODEL="qwen3:27b"
echo "=== Pulling model: $MODEL ==="
ollama pull "$MODEL"
echo "Model pulled successfully."

echo "=== Step 1: Extract text from PDFs ==="
python -m src.extract

echo "=== Step 2: Chunk extracted texts ==="
python -m src.chunk

echo "=== Step 3: Extract nuggets via Ollama ==="
python -m src.nuggets --config config-ollama.yaml

echo "=== Cleanup ==="
kill $OLLAMA_PID 2>/dev/null || true
wait $OLLAMA_PID 2>/dev/null || true
rm -rf "$OLLAMA_MODELS"
echo "Job complete."
