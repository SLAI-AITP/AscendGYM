# 工具

本目录包含 bridge 的辅助工具。所有命令在 `bridge/` 目录下执行。

```text
tools/
├── proxy.py                    # Anthropic/Responses -> Chat 协议代理
├── reprofile_best.py           # native/bare 最终复测发起器
├── apply_final_recheck.py      # 手工复测写回 / final verification 改判
├── task_dispatcher.py          # 任务队列与依赖调度
└── monitor/native_monitor.py   # 批次进度监控
```

## 1. proxy.py

`proxy.py` 是 run 级协议代理，只监听 `127.0.0.1` 的随机端口，由 `op_manager.py` 自动拉起和关闭。它包含两种协议转换模式，以及保持上游协议的 `passthrough` 模式。三种模式均可为本次 native run 记录服务端 usage：

```bash
# Claude Code: /v1/messages -> 上游 /v1/chat/completions
python3 tools/proxy.py --protocol anthropic \
  --upstream http://127.0.0.1:8008/v1 --model <MODEL> --port <PORT>

# Codex: /v1/responses -> 上游 /v1/chat/completions
python3 tools/proxy.py --protocol responses \
  --upstream http://127.0.0.1:8008/v1 --name <MODEL> --port <PORT>
```

`op_manager.py` 在 native claude/codex run 前会自动探测上游：

- 发送 1-token 最小请求到 `/v1/messages` 或 `/v1/responses`，并检查 `message` / `response` 结构；
- 返回 404/405、连接失败/超时、5xx 或 2xx 但结构不匹配时，启动对应代理；
- 400/401/403/422 视为接口存在，使用透传计数通路，由真实请求暴露认证或参数问题；
- 不需要在 `op_catalog.json` 中配置代理开关。

原生协议可用时，claude/codex 使用 `passthrough`；opencode 同样通过该模式读取 usage。Chat Completions 流式请求会要求上游返回 usage。计数通路和观察凭据由 `op_manager.py` 按 run 自动配置，不写入模型日志或摘要。

候选评估完成时，可信网关按本机单调时钟查询该时刻的累计 token。请求收据保存在已有 `usage.json` 中；模型请求未完成、usage 缺失或计数通路失效时，相关计数保持 unavailable，不根据终态总量回填。`--dry-run` 不启动这些服务。实际运行使用 `--runs-dir` 将产物放到发布目录以外。

生命周期：

- 代理是当前 `op_manager.py` 的子进程，启动时带 `--parent-pid`；
- 正常结束由 `atexit` 和 `stop_model_proxy()` 终止；
- 父进程被 `SIGKILL` 时，代理的 parent watchdog 会在约 2 秒内退出；
- 只绑定 `127.0.0.1` 随机端口，不占用全局端口，不影响同机其他用户。

## 2. reprofile_best.py

为已结束的 native/bare run 发起最终复测。它从 `result.json.best_verified` 找到最佳版本：

- native：读取 `snapshots/<solution_id>/files`，按 `config.resolved.json` 的 `evaluator.command` 连续执行复测；
- bare：启动本 run 的 Backend，先 `restore_op_version` 恢复最佳版本，再连续执行 `ascend_code_profiling`。

复测结果写入 `<run-dir>/_final_recheck_<timestamp>/result.json`，格式与 `apply_final_recheck.py` 的输入一致。工具不会重新运行 agent，也不消耗候选预算。

典型场景：批次已结束、`result.json.success=false`，`final_verification` 在固定同一张 NPU 上连续失败，需要换一张卡对同一最佳版本重新取证。

命令：

```bash
cd "$ASCEND_GYM_WORKSPACE/bridge"

# 只做输入检查和计划预览，不启动 evaluator/Backend
python3 tools/reprofile_best.py \
  --run-dir "$ASCEND_GYM_WORKSPACE/runs/<TAG>/<RUN_ID>" \
  --count 5 --npu node-0:npu0 --sandbox-url "$SANDBOX_URL" --dry-run

# native：复测并写回
python3 tools/reprofile_best.py \
  --run-dir "$ASCEND_GYM_WORKSPACE/runs/<TAG>/<RUN_ID>" \
  --count 5 --npu node-0:npu0 --sandbox-url "$SANDBOX_URL" --apply

# bare：配置里已有 sandbox 时可不传 --sandbox-url
python3 tools/reprofile_best.py \
  --run-dir "$ASCEND_GYM_WORKSPACE/runs/<TAG>/<RUN_ID>" \
  --count 5 --npu node-0:npu0 --apply
```

| 参数 | 说明 |
|---|---|
| `--count` | 复测次数；默认取原 run 的 `final_verification.requested`，其次 `stop.final_reprofile_count` |
| `--npu` | 目标 NPU（`node-0:npu0`）；通过 `SANDBOX_PREFERRED_NODE_ID` / `SANDBOX_PREFERRED_NPU_ID` 传给 evaluator/Backend |
| `--sandbox-url` | native 未设置 `SANDBOX_URL` 时必填；bare 用于覆盖 run 配置里的 sandbox 地址 |
| `--ops-repo` / `--data-dir` | native 覆盖 `package.json` 中的仓库名/测试数据目录 |
| `--version` | bare 复测版本；默认取 `best_verified.version` |
| `--config` | run 配置；native 默认 `<run-dir>/config.resolved.json`，bare 默认 `<run-dir>.json` |
| `--recheck-dir` | 输出目录；默认 `<run-dir>/_final_recheck_<timestamp>` |
| `--apply` | 复测 `valid=true` 后调用 `apply_final_recheck.py --yes` |
| `--force-apply` | 允许对已经 `success=true` 的 run 再写回（传给 apply 的 `--force`） |
| `--no-timing-refresh` | 写回时不刷新 `TIMING.md`（传给 apply） |
| `--dry-run` | 只做 preflight（快照/暂存目录/package/Backend）并打印计划 |

行为与安全边界：

- 只有 `len(evaluations) == requested` 且全部 `status=passed`、`correctness_passed=true`，复测才记为 `valid=true`；
- 任一次失败立即停止后续复测；`valid=false` 时不调用 `apply_final_recheck.py`，进程退出码为 2；
- native 会校验 evaluator 回显的 `profiling_protocol`，与 run 配置不一致时记为 `profiling_protocol_mismatch`；
- bare 的 Backend 绑定 `127.0.0.1` 随机端口，复测结束（含异常）后关闭；
- 如果原 `result.json` 已是 `success=true`，`--apply` 默认不写入；需要追加复测记录时再加 `--force-apply`；
- 不修改原 `result.json`；写回由 `apply_final_recheck.py` 完成，并保留备份；
- 不修改 golden、测试、源码、评测器或同批次其他 run。

## 3. apply_final_recheck.py

用于把一个已经完成、且 `valid=true`、所有 evaluation 都通过的 `_final_recheck_*` 复测结果写回 run 的 `result.json`。典型场景：

- 批次已经结束；
- `result.json.success=false`；
- `final_verification` 存在失败，且失败在固定的同一张 NPU 上重复出现，不限定必须在 4/5；
- 同一最佳版本在另一张 NPU 上完成复测，所有 requested evaluation 都通过。复测次数不硬编码为 5，按 `recheck.requested`；常见配置是 5/5。

本工具按 `result.json` 的 `best_verified` / `final_verification` 字段工作，不绑定具体 harness。复测可由本目录的 `reprofile_best.py` 生成（推荐），也可由外部复测流程生成同格式的 `_final_recheck_*`。

前置条件：

- run 目录下存在 `result.json`；
- 存在已经完成的 `_final_recheck_*` 目录，其中 `result.json.valid=true`，且所有 evaluation 都是 `passed`；
- 原 run 有 `best_verified`，且 `candidate_evaluations > 0`；baseline-only 或无候选的 run 会被拒绝。

命令：

```bash
cd "$ASCEND_GYM_WORKSPACE/bridge"

# 预览：自动选择最新 _final_recheck_*，并尝试从 profiling 日志识别 NPU
python3 tools/apply_final_recheck.py \
  --run-dir "$ASCEND_GYM_WORKSPACE/runs/<TAG>/<RUN_ID>" \
  --dry-run

# 确认后写入
python3 tools/apply_final_recheck.py \
  --run-dir "$ASCEND_GYM_WORKSPACE/runs/<TAG>/<RUN_ID>" \
  --npu node-0:npu0 --yes
```

完整流程可以直接使用上一节的 `reprofile_best.py ... --apply`；如果复测时没有传 `--apply`，再手动执行这里的写回命令。

| 参数 | 说明 |
|---|---|
| `--recheck-dir` | 不传时使用最新的 `_final_recheck_*` |
| `--npu` | 不传时从 recheck 的 `assigned_server` / profiling 日志提取 |
| `--version` | 默认取 `result.json.best_verified.version` |
| `--force` | 允许对已经 `success=true` 的 run 再追加一次复测记录 |
| `--no-timing-refresh` | 不尝试刷新批次 `TIMING.md` |
| `--dry-run` / `--yes` | 前者只预览；后者确认写入 |

写入内容：

- 校验不通过时退出码 2，**不写任何文件**；
- 原 `success`、`operational_status`、`termination_reason`、`raw_termination_reason` 和原 `final_verification` 写入 `recheck_history`；
- 替换 `final_verification`，标注 `source=manual_recheck`、`target_npu`、`recheck_dir`、`timestamp`；复测证据带版本时（bare）同时标注 `version`；
- 设置 `success=true`，并更新 `events.jsonl` 最后一个 `run_finished.success=true`；
- 备份原文件为 `result.json.bak-before-recheck-<timestamp>`；
- 批次存在 `TIMING.md` 且 `tools/timing_report.py` 存在时尝试刷新；当前包没有内置该报告生成器时会跳过。

安全边界：

- 只修改指定的一个 run，不改同批次其他算子；
- 不重新执行复测，也不消费候选预算；
- 不改 golden、测试、源码或评测器；
- 本工具只负责写回已有复测证据，不启动复测；`_final_recheck_*` 可由 `reprofile_best.py` 或外部复测流程生成。复测次数由 `recheck.requested` 决定，工具不强制 5 次。

## 4. task_dispatcher.py

用于按队列提交评测任务，或等待某个批次结束后自动执行下一条命令。

```bash
# 等待 TAG 结束后执行重测命令
python3 tools/task_dispatcher.py submit \
  --queue main --label rerun \
  --after-tag <TAG> --after-agent <AGENT> \
  --cmd "python3 start_all_runs.py ..."

# 从 JSON 一次提交多条
python3 tools/task_dispatcher.py submit --queue main --spec jobs.json

# 后台常驻 / 只跑一轮
python3 tools/task_dispatcher.py run --queue main --daemon
python3 tools/task_dispatcher.py run --queue main --once

# 查看 / 停止 daemon
python3 tools/task_dispatcher.py list --queue main
python3 tools/task_dispatcher.py stop --queue main
```

状态目录默认为 `bridge/.dispatcher`；作业日志位于对应队列目录下。

## 5. monitor/native_monitor.py

按 `--tag` 展示批次进度。纯本地读取，无模型调用。

```bash
# 单次输出
python3 tools/monitor/native_monitor.py --tag <TAG> --single

# 循环监控，可同时指定多个 tag
python3 tools/monitor/native_monitor.py --tag <TAG_A> --tag <TAG_B> \
  --loop --interval 120 --sandbox-url $SANDBOX_URL

# 自定义输出文件
python3 tools/monitor/native_monitor.py --tag <TAG> --single \
  --md /path/to/progress.md --jsonl /path/to/progress.jsonl
```

默认输出到 `tools/monitor/native_progress_<tag>.md` 与同名 `.jsonl`。

## 6. 相关文档

- [BRIDGE_USAGE.md](BRIDGE_USAGE.md)
- [BRIDGE_CONFIG.md](BRIDGE_CONFIG.md)
- [../QUICK_START.md](../QUICK_START.md)
