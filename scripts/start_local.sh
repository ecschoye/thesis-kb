#!/bin/bash
# Run API locally on Mac (requires Ollama running with the embedding model)
# Usage: bash scripts/start_local.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "ERROR: Ollama not running. Start with: ollama serve"
  exit 1
fi

source .venv/bin/activate

export OPENROUTER_API_KEY=$(cat ~/.openrouter_key 2>/dev/null || echo "${OPENROUTER_API_KEY:-}")

if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set"
  exit 1
fi

export THESIS_KB_CONFIG="config-ollama.yaml"

echo "Starting API locally on port 8001 (config: config-ollama.yaml)"
echo "Open: static/index.html"
uvicorn src.api:app --host 127.0.0.1 --port 8001 --reload
