#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
TAG="EXAMPLE-bare-$(date +%Y%m%d-%H%M%S)"
python3 op_manager.py run \
  --op "$OP" --harness bare --agent bare \
  --model "$MODEL_KEY" \
  --candidate-evaluations-max 1 --final-reprofile-count 1 \
  --sandbox-url "$SANDBOX_URL" \
  --tag "$TAG"
