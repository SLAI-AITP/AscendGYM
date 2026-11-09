#!/usr/bin/env bash
# Source this file from the workspace root (or by its absolute path).
_ASCEND_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASCEND_GYM_DATA_ROOT="$_ASCEND_DATA_DIR"
export ASCEND_GYM_WORKSPACE="$(cd "$_ASCEND_DATA_DIR/.." && pwd)"
unset _ASCEND_DATA_DIR
