#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for script in 01_native_claude.sh 02_native_codex.sh 03_native_opencode.sh 04_bare.sh; do
  echo "=== running $script ==="
  bash "$HERE/$script"
done
