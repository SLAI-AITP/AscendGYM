#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
TAG="EXAMPLE-native-claude-$(date +%Y%m%d-%H%M%S)"
python3 op_manager.py run \
  --op "$OP" --harness native --agent claude \
  --model "$MODEL_KEY" \
  --candidate-evaluations-max 1 --final-reprofile-count 1 \
  --tag "$TAG"
