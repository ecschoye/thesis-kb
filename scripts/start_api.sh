#!/bin/bash
# Start the thesis-kb API server
# Usage: bash scripts/start_api.sh [--config config-ollama.yaml] [--port 8001]

set -euo pipefail
cd /cluster/work/ecschoye/thesis-kb

CONFIG="config.yaml"
PORT="8001"

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

source .venv/bin/activate

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  if [ -f ~/.openrouter_key ]; then
    export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
  else
    echo "ERROR: OPENROUTER_API_KEY not set and ~/.openrouter_key not found"
    exit 1
  fi
fi

echo "Starting API on port $PORT with config $CONFIG"
echo "Open tunnel from Mac: ssh -N -L ${PORT}:localhost:${PORT} ecschoye@idun-login1.hpc.ntnu.no"
echo "Then open: static/index.html?api=http://localhost:${PORT}"

export THESIS_KB_CONFIG="$CONFIG"
uvicorn src.api:app --host 0.0.0.0 --port "$PORT"
