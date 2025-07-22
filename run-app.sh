#!/usr/bin/env bash
set -e

echo "⚙️  Activando entorno virtual…"
source venv/bin/activate

echo "🚀  Levantando FastAPI en http://127.0.0.1:8000 …"
python3 -m uvicorn src.main.pythonscripts.pythonscripts1.app:app --reload --port 8000


