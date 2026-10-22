#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$BENCHMARK_ROOT/dist"
BUNDLE_NAME="universal-benchmark-runner"
BUNDLE_DIR="$DIST_DIR/$BUNDLE_NAME"
ZIP_FILE="$DIST_DIR/$BUNDLE_NAME.zip"

rm -rf "$BUNDLE_DIR" "$ZIP_FILE"
mkdir -p "$BUNDLE_DIR/scripts"

cp -R "$BENCHMARK_ROOT/src" "$BUNDLE_DIR/src"
cp -R "$BENCHMARK_ROOT/benchmark_scoring" "$BUNDLE_DIR/benchmark_scoring"
cp "$SCRIPT_DIR/run_remote_benchmark.py" "$BUNDLE_DIR/scripts/"
cp "$BENCHMARK_ROOT/requirements.txt" "$BUNDLE_DIR/requirements.txt"
cp "$BENCHMARK_ROOT/README.md" "$BENCHMARK_ROOT/LICENSE" "$BUNDLE_DIR/"
if REVISION="$(git -C "$BENCHMARK_ROOT" rev-parse --short HEAD 2>/dev/null)"; then
  if [[ -n "$(git -C "$BENCHMARK_ROOT" status --porcelain --untracked-files=no)" ]]; then
    REVISION="${REVISION}-dirty"
  fi
else
  REVISION="unreleased"
fi
printf '%s\n' "$REVISION" > "$BUNDLE_DIR/REVISION"
(
  cd "$DIST_DIR"
  zip -qr "$BUNDLE_NAME.zip" "$BUNDLE_NAME" -x '*/__pycache__/*' '*/.DS_Store' '*.pyc'
)
rm -rf "$BUNDLE_DIR"

echo "Bundle created: $ZIP_FILE"
