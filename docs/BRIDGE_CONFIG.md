# 配置说明

编译执行环境可选择[官方镜像](../docs/SANDBOX_IMAGE.md)或[兼容自建服务](../docs/SELF_HOSTED_ENVIRONMENT.md)。连接地址为服务基础 URL，`op_manager.py run` 的 `--sandbox-url` 优先于 `SANDBOX_URL`，均未设置时默认 `http://127.0.0.1:9000`。源码和冻结数据通过 HTTP 上传；客户端无需服务源码。请求与返回格式见[服务协议](../docs/EXECUTION_BACKEND.md)。

本目录保存 bridge 的共享配置。字段级参考见 [CATALOG_REFERENCE.md](CATALOG_REFERENCE.md)。

```text
configs/
├── op_catalog.json          # 共享总配置
└── op_catalog.resolved.json # scan/build 生成，供检查
```

## 1. 环境变量

配置中的路径支持环境变量展开：

```bash
export ASCEND_GYM_WORKSPACE=/path/to/workspace
export ASCEND_GYM_DATA_ROOT=/path/to/ascend_gym_data
```

- `${ASCEND_GYM_WORKSPACE}`：代码包根目录。
- `${ASCEND_GYM_DATA_ROOT}`：数据根目录，包含 `ops_src/` 与 `packages/`。

`outputs.runs_dir` 使用 CWD 相对路径 `../runs`；从 `bridge/` 执行时产物位于包根 `runs/`。

## 2. op_catalog.json

### 2.1 顶层字段

| 字段 | 说明 |
|---|---|
| `schema_version` / `name` | 配置版本与名称 |
| `source_root` | 算子源码根 |
| `search_roots` | 源码仓库内的候选搜索路径 |
| `package` | 输入包裁剪规则 |
| `ops` | 参与评测的算子列表 |
| `outputs` | 输入包、产物、native 代码根 |
| `models` | 模型注册表 |
| `bare` | bare harness 的专属配置 |
| `agents` / `backend` | native agent 与评测器配置 |
| `budget_defaults` / `metrics_defaults` | 预算与指标默认值 |
| `infra_retry` | 沙盒/设备故障重试策略 |

### 2.2 source_root 与 search_roots

`source_root` 指向 `ops_src`。`search_roots` 中的 `path` 相对于 `source_root`：

```json
"search_roots": [
  {"id": "test_kernels:ops_nn", "path": "ops_nn/test_kernels/kernels",
   "ops_repo": "nn", "bucket": "test_kernels"}
]
```

`ops[].prefer` 指定优先搜索的 `id`。

### 2.3 ops

```json
"ops": [
  {"name": "swi_glu", "prefer": ["vllm_ascend"]},
  {"name": "rms_norm"}
]
```

`name` 必须在某个 `search_roots` 路径下存在，并包含 `op_host/`、`op_kernel/` 与测试脚本。

### 2.4 outputs

```json
"outputs": {
  "inputs_dir": "${ASCEND_GYM_DATA_ROOT}/packages",
  "runs_dir": "../runs",
  "native_root": "${ASCEND_GYM_WORKSPACE}/eval/native_harness"
}
```

- `inputs_dir`：算子输入包目录，由 `op_manager.py build` 生成。
- `runs_dir`：动态产物目录，CWD 相对。
- `native_root`：native harness 代码根。

## 3. models

模型以 key 注册，运行时通过 `--model <key>` 选择。

```json
"models": {
  "my-local-model": {
    "provider": "local",
    "name": "DeepSeek-V4-Flash-BF16",
    "openai_base": "http://127.0.0.1:8008/v1",
    "anthropic_base": "http://127.0.0.1:8008",
    "wire_api": "chat",
    "context": 1000000,
    "max_output": 32768,
    "api_key": "any-value",
    "tool_format": "json"
  }
}
```

| 字段 | 说明 |
|---|---|
| `provider` | `local`、`bigmodel`、`gptplus5`、`openai`、`openrouter` |
| `openai_base` | OpenAI 兼容端点；bare/native codex/opencode 使用 |
| `anthropic_base` | native claude 使用；缺省时由 `openai_base` 推导 |
| `wire_api` | `chat` 或 `responses`；bare 使用 `chat` |
| `tool_format` | `json` 或 `xml` |
| `context` / `max_output` | 上下文与输出上限 |
| `api_key` | 本地占位或私有配置中的密钥，不随包提交 |
| `api_key_env` | 密钥环境变量名；优先于 api_key，变量缺失或为空即报错 |

native claude/codex 启动前会自动探测上游协议：

- claude 探测 `POST <anthropic_base>/v1/messages`（1-token 最小请求并检查 `message` 结构）；
- codex 探测 `POST <openai_base>/v1/responses`（1-token 最小请求并检查 `response` 结构）；
- 原生协议受支持时通过本地透传通路读取服务端 usage；不受支持时由 `bridge/tools/proxy.py` 转换协议并计数；
- 不需要在模型条目里配置代理开关。

### 3.1 codex 的 `CODEX_HOME`

codex 使用两层 `CODEX_HOME`：

- **模板 `CODEX_HOME`**：由用户设置，bridge 读取其中的 `config.toml` 和 `models.json`；未设置时使用 `~/.codex`。
- **per-run `CODEX_HOME`**：bridge 在 `$ASCEND_GYM_WORKSPACE/runs/<TAG>/.per-run/<run_id>/codex-home/`（无 `--tag` 时为 `runs/.per-run/...`）下自动生成，并在启动 codex 时注入。它固定本次 run 的模型、provider、`base_url`、`wire_api` 和代理地址，不修改用户全局配置，也避免并行 run 相互覆盖。

只有传了 `--model` 才会生成 per-run `CODEX_HOME`。不传 `--model` 时，codex 直接使用模板 `CODEX_HOME`（未设置则 `~/.codex`）中的全局模型配置。

模板要求：

- `models.json` 至少包含一个可用条目；推荐放一行与 `--model` 同名的 `slug`；
- `config.toml` 可选；缺失时 bridge 会生成最小配置，但仍需要 `models.json`。

启动前设置模板目录：

```bash
export CODEX_HOME=/path/to/codex-template
```

如果模板目录不存在或没有可用 `models.json`，bridge 会在启动 codex 前报错：

```text
codex 模型目录里没有可用模板 ...
```

不会静默回退到其他 provider。完整说明见 [native_harness/README.md §4.1](../eval/native_harness/README.md#41-codex-的-codex_home)。

云模型运行前将 `REPLACE_WITH_YOUR_API_KEY` 替换为真实密钥。密钥不应提交或分发。

## 4. bare

```json
"bare": {
  "client_root": "${ASCEND_GYM_WORKSPACE}/eval/bare_harness/client_v2",
  "backend_root": "${ASCEND_GYM_WORKSPACE}/eval/bare_harness/Backend",
  "python": "python3",
  "default_model": "my-local-model",
  "profiling": {"total_runs": 8, "warmup_runs": 3, "statistic": "mean"},
  "backend": {
    "compile_timeout_ms": 1800000,
    "run_timeout_ms": 3600000,
    "start_timeout_seconds": 90
  },
  "client": {"backend_timeout_seconds": 3600}
}
```

- `backend_root` 必须是 k-lens-v2 Backend，包含 `finish_task`。
- profiling 协议固定为 8 总轮 / 3 预热 / mean。

## 5. agents 与 backend

```json
"agents": {
  "claude": {"kind": "claude", "executable": "claude",
             "timeout_seconds": 1200, "retry_attempts": 2,
             "retry_backoff_seconds": 30},
  "codex": {"kind": "command", "timeout_seconds": 1200},
  "opencode": {"kind": "command", "timeout_seconds": 1800}
},
"backend": {
  "kind": "command",
  "evidence_kind": "real",
  "command": ["{python}", "{config_dir}/sandbox_evaluator.py"],
  "timeout_seconds": 5400
}
```

- native agent 的 CLI 与模型由 `--agent`、`--model` 和 per-run 配置决定。
- `{python}` 与 `{config_dir}` 由 bridge 在启动 native evaluator 时替换。

## 6. package、预算与重试

- `package`：保留目录、禁止文件、最大文件数/体积、可编辑前缀、测试目录名。
- `budget_defaults`：`candidate_evaluations_max`、`final_reprofile_count`、`wall_clock_seconds_max` 等默认值。
- `metrics_defaults`：native 的指标检查点。
- `infra_retry`：沙盒/设备故障的重试次数、退避与提示文本。

## 7. 配置检查

```bash
cd $ASCEND_GYM_WORKSPACE/bridge
python3 op_manager.py build --all
python3 op_manager.py validate --op <OP> --harness native --model <KEY>
python3 op_manager.py run --op <OP> --harness bare --agent bare \
  --model <KEY> --candidate-evaluations-max 1 --final-reprofile-count 1 \
  --sandbox-url $SANDBOX_URL --tag CONFIG-CHECK --dry-run
```

## 8. 相关文档

- [BRIDGE_USAGE.md](BRIDGE_USAGE.md)
- [CATALOG_REFERENCE.md](CATALOG_REFERENCE.md)
- [README：模型配置](../README.md#models)
- [../eval/native_harness/README.md](../eval/native_harness/README.md)
- [../eval/bare_harness/README.md](../eval/bare_harness/README.md)
