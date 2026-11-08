#!/usr/bin/env bash
# Shared environment for the add_rms_norm example.
# Export these before running any examples/*.sh script.
: "${ASCEND_GYM_WORKSPACE:?export ASCEND_GYM_WORKSPACE=/path/to/final}"
: "${ASCEND_GYM_DATA_ROOT:?export ASCEND_GYM_DATA_ROOT=/path/to/ascend_gym_data}"
: "${SANDBOX_URL:?export SANDBOX_URL=http://<sandbox-host>:<port>}"

# Optional: codex template home (must contain models.json).
# If unset, bridge uses ~/.codex.
# export CODEX_HOME=/path/to/codex-template

export MODEL_KEY=DeepSeek-V4-Flash-BF16
export OP=add_rms_norm
cd "$ASCEND_GYM_WORKSPACE/bridge"
