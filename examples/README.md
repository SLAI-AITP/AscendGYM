# add_rms_norm 示例

编译执行环境可选择[官方镜像](../docs/SANDBOX_IMAGE.md)或[兼容自建服务](../docs/SELF_HOSTED_ENVIRONMENT.md)。连接地址为服务基础 URL，`op_manager.py run` 的 `--sandbox-url` 优先于 `SANDBOX_URL`，均未设置时默认 `http://127.0.0.1:9000`。源码和冻结数据通过 HTTP 上传；客户端无需服务源码。请求与返回格式见[服务协议](../docs/EXECUTION_BACKEND.md)。

本示例使用以下环境变量：

- 模型：`DeepSeek-V4-Flash-BF16`
- 沙盒：由 `SANDBOX_URL` 指定
- 算子：`add_rms_norm`
- 预算：`candidate-evaluations-max=1`，`final-reprofile-count=1`
- 执行方式：串行，`parallel=1`

## 环境

```bash
export ASCEND_GYM_WORKSPACE=/path/to/workspace
export ASCEND_GYM_DATA_ROOT=/path/to/ascend_gym_data
export SANDBOX_URL=http://<sandbox-host>:<port>

# codex 可选：模板 CODEX_HOME（至少包含 models.json）
# 不设置时 bridge 使用 ~/.codex
# export CODEX_HOME=/path/to/codex-template
```

数据根中需要包含：

```text
$ASCEND_GYM_DATA_ROOT/
├── ops_src/
└── packages/
```

## 前置条件

- `claude`、`codex`、`opencode` 已安装；若使用 codex，模板 `CODEX_HOME` 需可用（至少包含 `models.json`），详见 [native_harness/README.md §4.1](../eval/native_harness/README.md#41-codex-的-codex_home)。
- 模型条目 `DeepSeek-V4-Flash-BF16` 的 `anthropic_base` / `openai_base` 使用占位主机 `<MODEL_HOST>`；运行前替换为实际的 OpenAI-compatible `/v1` 地址。
- bridge 启动 claude/codex 前会自动探测 `/v1/messages` 和 `/v1/responses`：
  - 原生支持则直连；
  - 返回 404/405 或探测失败时，自动启动 `bridge/tools/proxy.py` 做协议转换，无需在 catalog 里配置开关。

## 四次启动命令

| 顺序 | harness / agent | 脚本 |
|---|---|---|
| 1 | native / claude | `examples/01_native_claude.sh` |
| 2 | native / codex | `examples/02_native_codex.sh` |
| 3 | native / opencode | `examples/03_native_opencode.sh` |
| 4 | bare | `examples/04_bare.sh` |

逐个执行：

```bash
bash examples/01_native_claude.sh
bash examples/02_native_codex.sh
bash examples/03_native_opencode.sh
bash examples/04_bare.sh
```

或串行执行全部：

```bash
bash examples/run_all_serial.sh
```

## 结果

每次运行输出到：

```text
$ASCEND_GYM_WORKSPACE/runs/<TAG>/add_rms_norm-<agent>-<TAG>/
```

主要读取：

- `native`：`result.json`、`metrics.json → raw_speedup`、`solutions.json`
- `bare`：`result.json`、`evaluations.jsonl`、`events.jsonl`、`usage.json`
