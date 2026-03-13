#!/bin/bash
# Rebuild ChromaDB + SQLite + FTS5 from existing embeddings (CPU only).
# Run this after embed_nuggets.slurm completes, or locally after any
# nugget/embedding changes.
#
# Usage:
#   bash scripts/rebuild_kb.sh                     # uses config.yaml (HPC)
#   bash scripts/rebuild_kb.sh config-ollama.yaml   # uses local config

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config.yaml}"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "============================================"
echo "  Rebuild KB (CPU only)"
echo "  Config: $CONFIG"
echo "============================================"

echo ""
echo "=== Step 1: Build ChromaDB + SQLite ==="
python -m src.store -c "$CONFIG"

echo ""
echo "=== Step 2: Verify FTS5 index ==="
python -c "
from src.query import ThesisKB
kb = ThesisKB('$CONFIG')
# _ensure_fts5 runs in __init__, verify it works
results = kb.bm25_search('spiking neural network', n_results=3)
print(f'FTS5 OK: {len(results)} results for test query')
kb.close()
"

echo ""
echo "============================================"
echo "  KB rebuild complete!"
echo "============================================"
echo ""
echo "Stats:"
python -c "
from src.query import ThesisKB
kb = ThesisKB('$CONFIG')
s = kb.stats()
print(f'  Papers:  {s[\"total_papers\"]}')
print(f'  Nuggets: {s[\"total_nuggets\"]}')
for t, c in sorted(s['nuggets_by_type'].items()):
    print(f'    {t:15s} {c}')
kb.close()
"
