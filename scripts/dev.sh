#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "Creating Python virtualenv..."
  python3 -m venv .venv
fi

echo "Installing Python dependencies..."
.venv/bin/pip install -q -r requirements.txt

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

exec .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
