#!/usr/bin/env bash
set -euo pipefail

RUNNER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNNER_ROOT"
PYTHON="${PYTHON_BIN:-python3}"
"$PYTHON" -c 'import sys; sys.exit("Python 3.12 or later is required. Set PYTHON_BIN to a compatible interpreter.") if sys.version_info < (3, 12) else None'
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -c 'import sys; sys.exit("Existing .venv needs Python 3.12 or later.") if sys.version_info < (3, 12) else None'
.venv/bin/python -m pip install -r requirements.txt
if [[ ! -e .env ]]; then
  (umask 077; cp .env.example .env)
fi
bash scripts/run.sh --help > /dev/null
printf '\nRunner ready. Existing .env configuration was preserved.\n'
printf 'Set BACKEND_URL and API_KEY in .env, then paste the dashboard benchmark command.\n'
printf 'Run on the inference machine. Start Ollama or vLLM there separately.\n'
printf 'Optional temporary vLLM startup requires vLLM installed in this .venv.\n'
