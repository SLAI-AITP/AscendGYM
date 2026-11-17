# 故障排查

排查时保留原始日志、输入快照和配置。部署、协议和测试要求见[镜像说明](SANDBOX_IMAGE.md)、[服务协议](EXECUTION_BACKEND.md)和[测试指南](OPERATOR_TEST_GUIDE.md)。

## 1. 先确定失败发生在哪里

先查看 run 目录下的 `result.json`，确定问题发生在任务打包、模型启动、服务连接、编译、测试还是 profiling 阶段。早期失败可能没有结果文件，此时检查启动终端和已有日志。

`--dry-run` 只检查启动配置，`/ping` 只检查连通性。无模型联调可在仓库根目录执行以下命令，并将服务地址替换为实际值：

```bash
export ASCEND_GYM_WORKSPACE="$PWD"
export ASCEND_GYM_DATA_ROOT="$ASCEND_GYM_WORKSPACE/datasets"
export SANDBOX_URL='http://<sandbox-host>:9000'
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py smoke --op add_rms_norm --cases all --profiling-runs 5
```

该命令检查 add_rms_norm 的编译、冻结测试与 profiling；默认另有 3 轮预热，以实际日志为准。记录命令输出的 smoke 目录。

## 2. 去哪里找日志

native run 通常位于 `runs/<TAG>/<RUN_ID>/`：

| 文件或目录 | 排查用途 |
|---|---|
| `result.json` | 运行状态、终止原因与最终复测 |
| `evaluations.jsonl`、`solutions.json` | 定位需要检查的评测记录 |
| `evaluations/<EID>/` | 单次结果、编译与测试日志、profiling 归档 |
| `agent.stdout.jsonl`、`agent.stderr.txt` | agent 启动与执行问题 |
| `usage.json`、`metrics.json` | token 证据和指标不可用原因 |

使用现有 HTTP evaluator 时，单次目录中的 `compile_*.txt`、`run_*.txt` 和 `stderr.txt` 分别用于查看编译、测试及 evaluator 错误。自定义服务的细分日志以实际适配器为准。

## 3. 常见问题

| 现象 | 检查与处理 |
|---|---|
| 找不到任务或测试 | 核对数据根和算子目录，用 `scan` 检查解析结果；源目录应直接包含 `*_test.py` |
| 输入包仍是旧内容 | 普通 `build` 和 `smoke` 不会刷新已有包。先备份包内修改，再使用 `build --op <OP_NAME> --force` 重建 |
| 服务连接失败或地址不符 | 核对基础 URL、网络和端口映射。`run --sandbox-url` 优先于 `SANDBOX_URL`；编译服务健康检查使用 `/ping` |
| 模型认证或启动失败 | 检查模型端点、认证、协议和 agent 配置；Codex 模板准备见 [native 指南](../eval/native_harness/README.md) |
| 编译失败或扩展无法导入 | 查看完整编译日志，核对工具链、构建资源、扩展产物和 Python/PyTorch 版本兼容性 |
| 找不到 golden 或测试数据 | 核对数据路径和实际上传内容。远端服务不能直接读取客户端本地路径 |
| 精度不通过 | 先验证原始基线，再核对候选、用例和容差；不修改 golden 或放宽断言来规避失败 |
| 测试轮数或 profiling 异常 | 检查测试是否支持 `--runs`、是否完成全部轮次，以及 msprof 输出和返回的采集文件 |
| 指标不可用或与加速比不一致 | 保留缺失原因，不补零或估算 token；STS 窗口与 `raw_speedup` 的区别见[结果说明](RESULTS_AND_SUBMISSION.md#3-指标口径) |
| 最终复测失败 | 核对被复测版本、逐次结果与执行环境，保留失败记录；复测方法见[工具说明](BRIDGE_TOOLS.md) |

## 4. 设备异常与超时

若原始基线也无法运行，先检查设备、驱动和环境；若基线正常而特定候选反复触发异常，优先检查候选实现。超时后先确认远端执行状态，再决定是否重试，避免重复提交。

区分单次编译或运行超时、评测重试预算与候选次数上限。直连单 worker 时，实际设备由部署配置确定。

## 5. 怎么才算完成联调

检查实际编译结果、全部测试轮次、错误日志、profiling 归档及有效耗时。`smoke` 的汇总状态是定位入口，不替代上述证据。联调通过一个任务，不代表所有任务、自建环境或其他硬件组合均已通过。


## 6. 反馈问题时附什么

提供仓库与数据版本、任务、harness、脱敏后的命令、环境、失败阶段、相关评测 ID 和必要日志，并说明原始基线是否通过、问题能否重复。

分享前移除认证信息和无关私有内容；材料范围见[结果与提交说明](RESULTS_AND_SUBMISSION.md)。

## 7. Bare 的开放状态

bare 完整 harness 当前暂不开放，现阶段使用公开的 native 路径。缺少 bare client 不是编译服务故障，后续获取方式由维护者公布。
