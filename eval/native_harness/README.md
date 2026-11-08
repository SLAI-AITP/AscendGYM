# native_harness

## 1. 定位

`native` harness 让真实的 coding agent（Claude Code / Codex / OpenCode）在原生工具环境中自主完成算子优化。框架只负责：

- 提供算子源码、测试与评测入口；
- 通过 `sandbox_evaluator.py` 提交编译、精度与 profiling；
- 记录候选、最终复测与 token 使用。

agent 的循环、工具调用与修改策略由对应 CLI 自行决定。

## 2. 目录

```text
native_harness/
├── native_bench/      # runner、config、evaluation、client 网关
└── core/              # contract、metrics、profiling
```

`native_bench/client.py` 是 runner 用子进程拉起的评测网关；`native_bench/prompt.txt` 是任务提示模板。

## 3. 前置条件

- Python 3.10+，依赖见根目录 `requirements.txt`。
- 可访问的远端沙盒。运行前设置：

```bash
export SANDBOX_URL=http://<sandbox-host>:<port>
```

- 已安装所选 agent CLI：
  - `claude`：Claude Code；
  - `codex`：Codex CLI，并准备可用的模板 `CODEX_HOME`（至少包含 `models.json`；未设置时 bridge 读取 `~/.codex`）；
  - `opencode`：OpenCode，并准备全局 opencode 配置。

## 4. 模型配置

模型登记在 `bridge/configs/op_catalog.json → models`，运行时用 `--model <registry-key>` 选择。

- `claude`：bridge 为每次 run 生成 `claude-settings.json`，使用模型条目中的 `anthropic_base` 与凭据。若上游没有 `/v1/messages`，bridge 自动启动 `bridge/tools/proxy.py --protocol anthropic`。
- `codex`：bridge 为每次 run 生成独立 `CODEX_HOME`，使用模型条目中的 `openai_base`；codex 0.162+ 固定使用 `wire_api = "responses"`。若上游没有 `/v1/responses`，bridge 自动启动 `bridge/tools/proxy.py --protocol responses`。
- `opencode`：bridge 以全局 opencode 配置为底，生成 per-run 配置。

模型端点与密钥的写法见 [docs/BRIDGE_CONFIG.md](../../docs/BRIDGE_CONFIG.md)。

### 4.1 codex 的 `CODEX_HOME`

codex 这里有**两层** `CODEX_HOME`，不要混淆：

| 层 | 谁设置 | 位置/用途 |
|---|---|---|
| **模板 `CODEX_HOME`** | 用户 | bridge 读取其中的 `config.toml` 与 `models.json`，作为生成 per-run 配置的模板 |
| **per-run `CODEX_HOME`** | bridge 自动生成 | `$ASCEND_GYM_WORKSPACE/runs/<TAG>/.per-run/<run_id>/codex-home/`（无 `--tag` 时为 `runs/.per-run/...`），每次 run 独立，只注入给本次 codex 子进程 |

bridge 需要 per-run `CODEX_HOME`，原因是：

- codex 0.162+ 只接受 `wire_api = "responses"`，且需要完整的 `models.json` 条目；
- 每次 run 的模型、上游地址和代理端口都可能不同；
- per-run 配置可以固定本次模型/provider，不修改用户的全局 codex 配置，也避免并行 run 互相覆盖；
- bridge 只把 per-run `CODEX_HOME` 传给 codex 子进程，不会改变其他进程的 `CODEX_HOME`。

用户只需要设置模板：

```bash
export CODEX_HOME=/path/to/codex-template
```

不设置时，bridge 使用默认值：

```text
~/.codex
```

模板至少需要：

- `models.json`：至少包含一个可用的模型条目；推荐包含与 `--model` 同名的 `slug`；
- `config.toml`：可选。缺失或为空时，bridge 会生成一份最小配置，但仍需要 `models.json`。

最小 `config.toml` 示例（`<MODEL_NAME>` 替换为模型名）：

```toml
model = "<MODEL_NAME>"
model_provider = "local"
model_context_window = 1000000
```

生成 per-run 配置时，bridge 会：

- 改写 `model`、`model_provider`、`model_context_window`、`model_catalog_json`；
- 删除模板里的 `[model_providers.*]` 段，避免把其他 provider 的密钥复制进 run 产物；
- 追加本次 provider 的 `base_url`、`wire_api`、token 与 `stream_idle_timeout_ms`；
- 从模板 `models.json` 复制匹配的条目；找不到时用已有条目做字段模板，连模板条目都没有则报错。

不同情况下的行为：

| 情况 | 结果 |
|---|---|
| 不设置 `CODEX_HOME`，`~/.codex` 有可用 `models.json` | 正常；`~/.codex` 作为模板，仍生成 per-run home |
| 不设置 `CODEX_HOME`，`~/.codex` 没有 `models.json` | codex 启动前报错：`codex 模型目录里没有可用模板 ...` |
| 设置到空目录/错误目录 | 同上报错；不会静默改用全局 provider |
| 不传 `--model` | bridge 不生成 per-run home；codex 直接使用 `$CODEX_HOME`（未设置则 `~/.codex`）的全局配置 |
| 手动在 bridge 外运行 codex | 由用户自己设置 `CODEX_HOME`；默认 `~/.codex`，没有 per-run 隔离 |

因此：

- 本包的 native codex run 必须传 `--model <registry-key>`；
- 不要把 `CODEX_HOME` 指向 `runs/<TAG>/.per-run/<run_id>/codex-home/`（无 tag 时去掉 `<TAG>`），这是 bridge 生成的目录；
- 如果不想污染 `~/.codex`，可以准备一个独立模板目录，启动前 `export CODEX_HOME=<template-dir>`。

## 5. 单算子运行

```bash
cd $ASCEND_GYM_WORKSPACE/bridge

python3 op_manager.py run \
  --op swi_glu \
  --harness native \
  --agent claude \
  --model <MODEL_KEY> \
  --candidate-evaluations-max 8 \
  --final-reprofile-count 3 \
  --tag NATIVE-$(date +%Y%m%d-%H%M%S)
```

将 `--agent claude` 替换为 `--agent codex` 或 `--agent opencode` 可使用其他 CLI。

## 6. 批量运行

```bash
python3 start_all_runs.py \
  --harness native --agent claude \
  --ops all \
  --model <MODEL_KEY> \
  --candidate-evaluations-max 8 --final-reprofile-count 3 \
  --parallel 2 \
  --tag NATIVE-BATCH-$(date +%Y%m%d-%H%M%S)
```

## 7. 预算与停止条件

- `--candidate-evaluations-max`：候选评测硬上限。
- `--final-reprofile-count`：最终复测次数，不占候选预算。
- `--candidate-solutions-target`：可选软目标，要求不同方案数不少于该值。
- native 没有 token、墙钟或模型轮数上限；停止由候选数、agent 完成、人工停止或致命错误决定。

## 8. 产物

输出目录：

```text
$ASCEND_GYM_WORKSPACE/runs/<TAG>/<op>-<agent>-<TAG>/
```

主要文件：

| 文件 | 说明 |
|---|---|
| `result.json` | `native-bench-result-v1`，包含 success、终止原因、baseline/best、最终复测 |
| `metrics.json` | 指标，含 `raw_speedup`、`STS_time` 等 |
| `solutions.json` | 候选方案与评测记录 |
| `evaluations/` | 每次评测的输入与结果 |
| `agent.stdout.jsonl` / `agent.stderr.txt` | agent CLI 输出 |

## 9. 完成时累计 token

使用 bridge 的 `run --harness native --model <MODEL_KEY>` 时，模型通路按每次响应的服务端 usage 计数。候选评估完成时记录 `cumulative_tokens` 和 `completion_usage_checkpoint`，不把异步收到的 CLI 小计或会话终态总量当作候选完成量。

候选完成之前的模型请求缺少 usage、完成边界处仍有模型请求未结束、计数暂时无法恢复时，相关计数和 STS_token 保持 unavailable。单独启动 native_bench 且未配置本次运行的计数器时，同样不估算 token。运行时使用 bridge 的 `--runs-dir` 将产物放到发布目录外。

候选计数和整段会话总量分别判定可用性。服务端显式返回有效 `total_tokens` 时，即使 input/output 拆分缺失，也可使用该精确总量；缺失的拆分保持为空。候选完成后的请求缺少 usage，不影响此前有完整证据的候选坐标和 STS。临时查询失败可由后续同一运行账本按原完成时刻恢复，恢复结果写入已有结果文件并记录重核事件；不会把边界后完成的请求算入较早候选。

## 10. 注意事项

- native 与 bare 的 profiling 聚合口径不同，不应直接比较绝对 `duration_ms`。跨 harness 比较请固定算子、模型与候选预算，并使用各自的 speedup 指标。
- `SANDBOX_URL` 通过环境变量传入；仅使用 `--sandbox-url` 不会改变 native evaluator 的沙盒地址。
- codex 的 `models.json` 必须是可用模板；缺失时 bridge 无法生成 per-run `CODEX_HOME`。
- 不建议在同一个输出目录重复运行；`--tag` 会创建新的批次目录。

相关文档：

- [USER_GUIDE.md](../../USER_GUIDE.md)
- [QUICK_START.md](../../QUICK_START.md)
- [docs/BRIDGE_USAGE.md](../../docs/BRIDGE_USAGE.md)
- [docs/BRIDGE_CONFIG.md](../../docs/BRIDGE_CONFIG.md)
