#!/bin/bash
# LifeoraAI — start Ollama + API server in one command
# Usage: ./start.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/../.venv/bin/activate"

echo "🌿 Starting LifeoraAI..."

# ── 1. Activate venv ──────────────────────────────────────────────────────────
if [ ! -f "$VENV" ]; then
  echo "❌  Virtual environment not found at $VENV"
  echo "    Run: python3 -m venv ../.venv && source ../.venv/bin/activate && pip install -e '.[dev]'"
  exit 1
fi
source "$VENV"

# ── 2. Start Ollama if not already running ────────────────────────────────────
if curl -s http://localhost:11434 > /dev/null 2>&1; then
  echo "✓  Ollama already running"
else
  echo "→  Starting Ollama..."
  ollama serve > /tmp/ollama.log 2>&1 &
  OLLAMA_PID=$!
  echo "   Ollama PID: $OLLAMA_PID"

  # Wait up to 10 seconds for Ollama to be ready
  for i in $(seq 1 10); do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
      echo "✓  Ollama ready"
      break
    fi
    sleep 1
  done
fi

# ── 3. Start FastAPI server ───────────────────────────────────────────────────
echo "→  Starting API server at http://localhost:8000"
echo "   UI  → http://localhost:8000"
echo "   Docs → http://localhost:8000/docs"
echo ""
cd "$PROJECT_DIR"
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# ── Cleanup on exit ───────────────────────────────────────────────────────────
if [ -n "$OLLAMA_PID" ]; then
  echo "Stopping Ollama (PID $OLLAMA_PID)..."
  kill "$OLLAMA_PID" 2>/dev/null || true
fi
