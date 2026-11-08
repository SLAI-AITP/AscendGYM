# AscendGYM 评测指标

## 正式任务与测试用例

正式任务由 `datasets/registry.json` 和 `bridge/configs/op_catalog.json` 共同定义，共 17 项。两种 harness 均使用任务随包提供的全部冻结用例，catalog 的 `data_cases` 统一为 `all`。冻结输入、golden、数值容差和测试脚本不得由优化 agent 修改。

## STS_token

设正确基线耗时为 `L0`，每次候选评估完成时的累计 token 为 `B`。累计量包含此前失败、无改进尝试及其他模型请求的消耗；仅通过正确性检查且快于此前最好结果的候选产生改进事件。同一完成坐标保留最低耗时，起始加速比为 `S0=1`。

```text
S_k = L0 / L_best(B_k)
Δ_k = log2(S_k) − log2(S_{k−1})
base_score = 100 × log2(S_final)
delay_penalty = 50 × Σ[(B_k / B_max) × Δ_k]
STS_token = base_score − delay_penalty
```

只计入 `0 ≤ B_k ≤ B_max` 的改进。零坐标的改进不扣延迟分；无改进时得分为 0。预算之外的候选不影响该预算内的 STS。

`B_max` 取 `metrics.token_checkpoints` 的最后一个值。其余检查点仅用于输出展示曲线，改变内部采样点不改变 STS。默认 catalog 的 `B_max` 为 128000 token；这是计分窗口，不是 native 的停止条件。比较不同运行时必须固定同一预算。

例如预算为 100，在 90 token 完成唯一一次 2 倍加速时，基础分为 100，延迟扣分为 45，STS 为 55；在 20 token 完成则为 90。

通过 bridge 的 `run --harness native --model <MODEL_KEY>` 启动时，每次运行配置独立的模型通路计数器。支持原生协议的端点保持请求协议透传，需要转换协议的端点复用现有代理；计数读取转换前的服务端 usage，不使用文本分词或终态总 token 回填。

可信评测网关在候选评估完成时记录单调时钟坐标，再向本次运行的计数器查询该时刻已完成模型请求的累计 input/output token。此前失败和无改进候选消耗、已完成的压缩及其他模型请求均包含在累计量中；缓存 token 按服务端字段语义纳入 input，reasoning 已含在 output 时不重复相加。边界之后完成的请求不回填到较早候选。

`evaluations.jsonl` 和 `solutions.json` 的 `cumulative_tokens`、`completion_usage_checkpoint` 保存各次完成计数及证据边界；已有 `usage.json` 保存请求计数收据，可用已有 usage replay 入口离线复核。运行记录写到发布目录外，不随源码发布。

缺少正确基线、任一候选的可靠完成计数，或候选完成之前的模型请求缺少 usage 时，不估算、不插值，`STS_token` 返回 `unavailable`。候选完成时仍有模型请求进行中时也保持不可用，因为无法得知该请求在此刻已经实际消耗的精确 token。缺少计数器配置的独立 native 入口仍不能以 CLI 会话终态总量代替逐候选坐标。

候选计数和整段会话总量分别判定可用性。服务端显式返回有效 `total_tokens` 时，即使 input/output 拆分缺失，也可使用该精确总量；缺失的拆分保持为空。候选完成后的请求缺少 usage，不影响此前有完整证据的候选坐标和 STS。临时查询失败可由后续同一运行账本按原完成时刻恢复，恢复结果写入已有结果文件并记录重核事件；不会把边界后完成的请求算入较早候选。整段会话总量仍单独标记可用性；该总量缺失时不宣称整段运行证据完整。

结果字段包括 `STS`、`base_score`、`delay_penalty`、`budget_max`、`improvements`、`curve`、`S_at_Bmax`、`VLR`、`S_equiv` 和公式标识 `linear-budget-delay-v1`。`S_equiv=2^(STS/100)`，`VLR=1−1/S_at_Bmax`。实现位于 `core/metrics.py` 的 `sts()`。

## 时间诊断与其他指标

`STS_time` 使用同一延迟扣分公式，横轴换为候选评估完成时的 `solution_elapsed_sec`，计分窗口取 `metrics.time_checkpoints_sec` 的最后一个值。它用于时间效率诊断，不能替代 token 指标。运行未正常完成时仅标记为部分观察结果。

`raw_speedup` 为正确基线耗时除以搜索阶段最低正确耗时，可能包含 STS 预算窗口之外的候选。最终复测不作为搜索改进事件计分。

配置有效的单次调用 workload recipe 后，单任务 PAOS 使用 `sqrt(PKAI × PTEI)`。recipe 必须明确每步调用次数、每步 token 数、其他步骤耗时及单次调用延迟语义；缺失时返回 `unavailable`。不从批次聚合耗时推断单次调用，也不虚构硬件计数。

单次运行不自动具备正式榜单资格。跨任务聚合需要固定任务权重、用例、预算和排除策略；Reliability 需要独立重复运行，重复 profiling 不等同于独立运行。
