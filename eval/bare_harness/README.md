# bare_harness

## 1. 定位

`bare` harness 评测裸模型本身的算子优化能力。模型只能使用 Backend 提供的工具池（读取文件、回填代码、提交编译/精度/profiling、结束任务），不依赖 coding agent CLI，也不使用 Pi。

实现基于 k-lens-v2：

- `client_v2/bare_client/`：单会话客户端；
- `client_v2/config.yaml` + `config_loader.py`：客户端配置；
- `Backend/`：k-lens-v2 Backend，包含 `finish_task` 工具。

## 2. 前置条件

- Python 3.10+，依赖见根目录 `requirements.txt`。
- 可访问的远端沙盒。
- 一个 OpenAI 兼容的模型端点。

不需要 Node 或 claude/codex/opencode CLI。

## 3. 模型配置

模型登记在 `bridge/configs/op_catalog.json → models`，运行时用 `--model <registry-key>` 选择。

k-lens-v2 支持的 provider：

```text
vllm / openrouter / openai / glm-api / glm-client
```

catalog 中的 `local` 会在 bare 路径映射为 `vllm`。本地 vLLM 示例：

```json
"my-local-model": {
  "provider": "local",
  "name": "DeepSeek-V4-Flash-BF16",
  "openai_base": "http://127.0.0.1:8008/v1",
  "wire_api": "chat",
  "context": 1000000,
  "api_key": "any-value",
  "tool_format": "json"
}
```

## 4. 单算子运行

```bash
cd $ASCEND_GYM_WORKSPACE/bridge

python3 op_manager.py run \
  --op swi_glu \
  --harness bare \
  --agent bare \
  --model <MODEL_KEY> \
  --candidate-evaluations-max 8 \
  --final-reprofile-count 3 \
  --sandbox-url http://<sandbox-host>:<port> \
  --tag BARE-$(date +%Y%m%d-%H%M%S)
```

## 5. 批量运行

```bash
python3 start_all_runs.py \
  --harness bare --agent bare \
  --ops all \
  --model <MODEL_KEY> \
  --candidate-evaluations-max 8 --final-reprofile-count 3 \
  --parallel 2 \
  --sandbox-url http://<sandbox-host>:<port> \
  --tag BARE-BATCH-$(date +%Y%m%d-%H%M%S)
```

## 6. 预算与评测协议

- `--candidate-evaluations-max`：候选评测硬上限。
- `--final-reprofile-count`：最终复测次数，不占候选预算。
- `--candidate-solutions-target`：可选软目标，默认等于候选上限。
- profiling 协议固定为 8 总轮 = 3 预热 + 5 测量，统计方式为 mean；与 bridge 的通用 `--profiling-runs` 口径不同。

## 7. 产物

输出目录：

```text
$ASCEND_GYM_WORKSPACE/runs/<TAG>/<op>-<agent>-<TAG>/
```

主要文件：

| 文件 | 说明 |
|---|---|
| `result.json` | `klens-bare-client-result-v1`，包含 success、speedup、baseline/best、终止原因 |
| `evaluations.jsonl` | 每次候选评测 |
| `trajectory.jsonl` | 模型与工具交互轨迹 |
| `usage.json` | token 使用 |
| `_client/` | bare_client 原始产物 |

## 8. 注意事项

- bare 使用 k-lens-v2 Backend（含 `finish_task`），不要指向其他版本的 Backend。
- bare 的 profiling 协议固定为 8 总轮 / 3 预热 / mean；与 native 的聚合口径不同，不应直接比较绝对 `duration_ms`。
- 模型偶尔只返回文本而不发起工具调用时，可通过 catalog 模型条目的 `protocol_correction_max` 调整纠偏次数。

相关文档：

- [README：安装与环境](../../README.md#installation)
- [README：快速开始](../../README.md#quick-start)
- [docs/BRIDGE_USAGE.md](../../docs/BRIDGE_USAGE.md)
- [docs/BRIDGE_CONFIG.md](../../docs/BRIDGE_CONFIG.md)
