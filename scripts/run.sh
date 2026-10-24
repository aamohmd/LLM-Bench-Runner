#!/usr/bin/env bash
set -euo pipefail

RUNNER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$RUNNER_ROOT/.venv/bin/python" ]]; then
  printf 'Runner is not installed. Run: bash "%s/scripts/setup.sh"\n' "$RUNNER_ROOT" >&2
  exit 1
fi
cd "$RUNNER_ROOT"
exec .venv/bin/python scripts/run_remote_benchmark.py "$@"
