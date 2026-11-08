#!/bin/bash

# AscendC Performance Tools Backend 启动脚本
# 用法: bash start.sh [--op-path <路径>] [--op-path-version <路径>] [--port <端口>] [--native-tool] [--min-improvement-epsilon <float>]
#   --op-path                   待优化的原始算子代码目录（可选）
#   --op-path-version           优化后的算子代码输出目录（可选）
#   --port                      服务监听端口，默认 7077（可选）
#   --native-tool               启用原生工具调用格式解析（Qwen3.6 等模型的 <function=...> 格式）（可选）
#   --min-improvement-epsilon   接受为新 best snapshot 所需的相对提升幅度（可选，默认 0.02）。
#                               必须与 Client orchestrator.min_improvement_epsilon 对齐

echo "=========================================="
echo "  AscendC Performance Tools Backend"
echo "=========================================="
echo ""

# 解析参数
OP_PATH=""
OP_PATH_VERSION=""
PORT=7077
NATIVE_TOOL=0
MIN_IMPROVEMENT_EPSILON=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --op-path)
            OP_PATH="$2"
            shift 2
            ;;
        --op-path-version)
            OP_PATH_VERSION="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --native-tool)
            NATIVE_TOOL=1
            shift
            ;;
        --min-improvement-epsilon)
            MIN_IMPROVEMENT_EPSILON="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash start.sh [--op-path <路径>] [--op-path-version <路径>] [--port <端口>] [--native-tool] [--min-improvement-epsilon <float>]"
            exit 1
            ;;
    esac
done

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate kg_backend

echo "服务地址: http://0.0.0.0:${PORT}"
echo "健康检查: http://0.0.0.0:${PORT}/health"
[ -n "$OP_PATH" ]         && echo "op-path:         ${OP_PATH}"
[ -n "$OP_PATH_VERSION" ] && echo "op-path-version: ${OP_PATH_VERSION}"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 构建启动参数
APP_ARGS="--port ${PORT}"
[ -n "$OP_PATH" ]         && APP_ARGS="${APP_ARGS} --op-path ${OP_PATH}"
[ -n "$OP_PATH_VERSION" ] && APP_ARGS="${APP_ARGS} --op-path-version ${OP_PATH_VERSION}"
[ "$NATIVE_TOOL" = "1" ]  && APP_ARGS="${APP_ARGS} --native-tool"
[ -n "$MIN_IMPROVEMENT_EPSILON" ] && APP_ARGS="${APP_ARGS} --min-improvement-epsilon ${MIN_IMPROVEMENT_EPSILON}"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python app.py ${APP_ARGS}
