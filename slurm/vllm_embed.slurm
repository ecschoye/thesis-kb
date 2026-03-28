#!/bin/bash
#SBATCH --job-name=vllm-embed
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="gpu40g"
#SBATCH --time=0-08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/vllm_embed_%j.log
#SBATCH --error=logs/vllm_embed_%j.log

# Long-running vLLM embedding server for the MCP server.
# The MCP server (.mcp.json) connects to localhost:8000, so Claude Code
# must run on the SAME node as this job.
#
# Usage:
#   1. Submit:  sbatch slurm/vllm_embed.slurm
#   2. Find the node:  squeue -u $USER -n vllm-embed -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"
#   3. SSH into it:  ssh <nodename>   (e.g. ssh idun-04-01)
#   4. Run Claude Code from /cluster/work/ecschoye/thesis-kb on that node.
#   5. semantic_search / MCP tools will hit localhost:8000 automatically.
#
# To stop the server: scancel <jobid>

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source .venv/bin/activate

export HF_HOME=/cluster/work/ecschoye/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "=== vLLM Embedding Server ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Port: 8000"
echo ""
echo "To use: ssh $(hostname) from a login node, then run Claude Code."
echo ""

# Run in foreground — SLURM keeps the job alive as long as this process runs.
exec python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-8B \
    --port 8000 \
    --convert embed \
    --max-model-len 2048 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.3
