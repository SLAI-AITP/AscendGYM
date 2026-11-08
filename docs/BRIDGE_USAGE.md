# ascend_gym_bridge 使用手册

本文说明 bridge 的完整用法。各 harness 的算法、配置与注意事项见对应 README：

- [native_harness](../eval/native_harness/README.md)
- [bare_harness](../eval/bare_harness/README.md)

## 1. 范围

bridge 是评测统一入口，负责：

- 解析算子与输入包；
- 生成 run 配置；
- 启动 harness；
- 批量调度、监控、停止与重测；
- 将 harness 产物归一化为统一输出目录。

支持列表：`native`、`bare`。

## 2. 目录

```text
bridge/
├── op_manager.py          # 单算子入口
├── start_all_runs.py      # 批量启动 / 状态 / 停止
├── sandbox_evaluator.py   # native 评测器
├── claude_agent_wrapper.py
├── bare_bench/            # bare 适配
├── configs/               # op_catalog 配置
└── tools/                 # 复测、协议代理、监控、任务队列
```

## 3. 前置条件

```bash
export ASCEND_GYM_WORKSPACE=/path/to/workspace
export ASCEND_GYM_DATA_ROOT=/path/to/ascend_gym_data
export SANDBOX_URL=http://<sandbox-host>:<port>
```

安装依赖：

```bash
pip install -r $ASCEND_GYM_WORKSPACE/requirements.txt
```

native 需要对应 agent CLI。详见各 harness README。

## 4. 命令

所有命令在 `bridge/` 目录下执行。

### 4.1 解析与构建

```bash
python3 op_manager.py scan
python3 op_manager.py build --all
```

- `scan`：解析 `ops_src`，写出 `configs/op_catalog.resolved.json`。
- `build`：生成 `outputs.inputs_dir` 下的算子输入包。

### 4.2 校验

```bash
python3 op_manager.py validate --op swi_glu --harness native --model <MODEL_KEY>
```

`bare` 没有离线 preflight，使用 `run --dry-run` 检查配置。

### 4.3 单算子运行

```bash
python3 op_manager.py run \
  --op <OP> \
  --harness <native|bare> \
  --agent <AGENT> \
  --model <MODEL_KEY> \
  --candidate-evaluations-max <B> \
  --final-reprofile-count <N> \
  --sandbox-url $SANDBOX_URL \
  --tag <TAG>
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--candidate-evaluations-max` | 候选评测硬上限 |
| `--final-reprofile-count` | 最终复测次数，不占候选预算 |
| `--candidate-solutions-target` | native 软目标（可选） |
| `--wall-clock-seconds` | 搜索墙钟；仅 bare 使用，`0` 表示不限 |
| `--dry-run` | 打印命令与环境，不实际执行 |

### 4.4 参数优先级

结构化 CLI 参数覆盖 `configs/op_catalog.json` 中的默认值；bare 的 profiling 协议仍以 catalog 为准。

## 5. 批量运行

```bash
python3 start_all_runs.py \
  --harness <native|bare> \
  --agent <AGENT> \
  --ops all \
  --model <MODEL_KEY> \
  --candidate-evaluations-max <B> \
  --final-reprofile-count <N> \
  --parallel <N> \
  --tag <TAG>
```

`--parallel` 控制本机同时运行的 run 数量，不控制远端沙盒 worker 数。

查看状态与停止：

```bash
python3 start_all_runs.py --harness <H> --agent <A> --ops all --tag <TAG> --status
python3 start_all_runs.py --harness <H> --agent <A> --ops all --tag <TAG> --stop
```

### 5.1 final verification 固定卡失败

如果批次已结束，但某个 run 的 `result.json.success=false`，且 `final_verification` 的失败在固定同一张 NPU 上反复出现，而同一最佳版本在另一张 NPU 上能通过，不要直接重跑整个候选预算。先用 `reprofile_best.py` 对最佳版本发起复测；复测通过后它会调用 `apply_final_recheck.py` 写回：

```bash
# 1) 检查输入并预览；native 需要 SANDBOX_URL，bare 可省略
python3 tools/reprofile_best.py \
  --run-dir "$ASCEND_GYM_WORKSPACE/runs/<TAG>/<RUN_ID>" \
  --count 5 --npu node-0:npu0 --sandbox-url "$SANDBOX_URL" --dry-run

# 2) 复测 + 写回（native/bare 通用）
python3 tools/reprofile_best.py \
  --run-dir "$ASCEND_GYM_WORKSPACE/runs/<TAG>/<RUN_ID>" \
  --count 5 --npu node-0:npu0 --sandbox-url "$SANDBOX_URL" --apply
```

`--count` 不传时沿用原 run 的 `final_verification.requested`。如果只想复测、不写回，去掉 `--apply`，再按需执行：

```bash
python3 tools/apply_final_recheck.py \
  --run-dir "$ASCEND_GYM_WORKSPACE/runs/<TAG>/<RUN_ID>" \
  --npu node-0:npu0 --yes
```

写回会保留原失败到 `recheck_history`、替换 `final_verification`、更新 `events.jsonl` 并备份原 `result.json`。复测次数由 `recheck.requested` 决定，不硬编码 5 次。完整参数、写入内容和安全边界见 [BRIDGE_TOOLS.md §2](BRIDGE_TOOLS.md#2-reprofile_bestpy) 与 [§3](BRIDGE_TOOLS.md#3-apply_final_recheckpy)。

## 6. 监控与队列

工具位于 `tools/`，说明见 [BRIDGE_TOOLS.md](BRIDGE_TOOLS.md)。

```bash
python3 tools/monitor/native_monitor.py --tag <TAG> --single
python3 tools/task_dispatcher.py list --queue <QUEUE>
```

## 7. 产物

```text
$ASCEND_GYM_WORKSPACE/runs/<TAG>/<op>-<agent>-<TAG>/
```

`outputs.runs_dir` 为 CWD 相对路径 `../runs`；在 `bridge/` 下执行时产物位于包根 `runs/`。

常见文件：

```text
result.json
evaluations.jsonl
events.jsonl
usage.json
manifest.json
agent.stdout.txt / agent.stderr.txt
_client/                # bare 原始产物
```

各 harness 的结果字段见对应 README。

## 8. 配置

完整配置说明见 [BRIDGE_CONFIG.md](BRIDGE_CONFIG.md)，字段参考见 [CATALOG_REFERENCE.md](CATALOG_REFERENCE.md)。

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `operator not resolved` | 检查 `source_root` / `search_roots` 与实际源码 |
| 输入包缺失 | 执行 `python3 op_manager.py build --all` |
| 沙盒连接失败 | 检查 `SANDBOX_URL`、`/health`、`/status` |
| 模型 401 | 检查 `models.<key>.api_key` 与端点 |
| 不支持 harness | 使用 `native` 或 `bare` |

## 10. 文档索引

- [USER_GUIDE.md](../USER_GUIDE.md)
- [QUICK_START.md](../QUICK_START.md)
- [BRIDGE_CONFIG.md](BRIDGE_CONFIG.md)
- [BRIDGE_TOOLS.md](BRIDGE_TOOLS.md)
- [eval/native_harness/README.md](../eval/native_harness/README.md)
- [eval/bare_harness/README.md](../eval/bare_harness/README.md)
