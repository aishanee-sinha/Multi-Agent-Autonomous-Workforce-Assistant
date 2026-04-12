#!/bin/bash
# setup_chromadb.sh — Install and run ChromaDB on EC2 alongside vLLM
# ====================================================================
# ChromaDB runs as a persistent server on port 8001.
# vLLM continues to run on port 8000.
#
# Usage:
#   bash rlhf/setup_chromadb.sh          # foreground (for testing)
#   bash rlhf/setup_chromadb.sh --bg     # background (for production)
#
# Data is persisted at /data/chromadb — survives restarts.

set -euo pipefail

CHROMADB_PORT="${CHROMADB_PORT:-8001}"
CHROMADB_DATA="${CHROMADB_DATA:-/data/chromadb}"
CHROMADB_HOST="${CHROMADB_HOST:-0.0.0.0}"

echo "=== ChromaDB Setup ==="
echo "  Port: ${CHROMADB_PORT}"
echo "  Data: ${CHROMADB_DATA}"

# 1. Install ChromaDB if not present
if ! python3 -c "import chromadb" 2>/dev/null; then
    echo "Installing ChromaDB..."
    pip install chromadb>=0.4.0
else
    echo "ChromaDB already installed."
fi

# 2. Create data directory
mkdir -p "${CHROMADB_DATA}"

# 3. Start ChromaDB server
if [[ "${1:-}" == "--bg" ]]; then
    echo "Starting ChromaDB in background..."
    nohup chroma run \
        --host "${CHROMADB_HOST}" \
        --port "${CHROMADB_PORT}" \
        --path "${CHROMADB_DATA}" \
        > /var/log/chromadb.log 2>&1 &
    echo "ChromaDB PID: $!"
    echo "Logs: /var/log/chromadb.log"
else
    echo "Starting ChromaDB in foreground..."
    chroma run \
        --host "${CHROMADB_HOST}" \
        --port "${CHROMADB_PORT}" \
        --path "${CHROMADB_DATA}"
fi
