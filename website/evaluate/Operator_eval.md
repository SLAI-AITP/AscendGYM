# 算子生成能力评测方案

本文只定义基于算子真实轨迹下，可计算的评测指标，输出维度分别为：**准确率、STS、PAOS**。缺失字段统一输出 `null`，不得用 0 或估算值替代。

## 1. 最小数据采集

### 1.1 轨迹字段

| 采集项 | 来源 | 用途 | 中文译名 |
| --- | --- | --- | --- |
| `run_id`、`success`、`operational_status`、`termination_reason` | `result.json` | 任务标识和运行状态 | 运行标识、成功状态、运行状态、终止原因 |
| `baseline_duration_ms` | `result.json`；回退到 `candidates.jsonl` 中 `evaluation_kind=baseline` 的 `duration_ms` | 性能基线 `L0` | 基线耗时（毫秒） |
| `status`、`correctness_passed`、`evaluation_kind` | `candidates.jsonl` | 正确性门槛 | 候选状态、正确性通过、评测类型 |
| `duration_ms` | `candidates.jsonl` | 候选 kernel latency | 候选算子耗时（毫秒） |
| `cumulative_tokens` | `candidates.jsonl` 或 `solutions.jsonl` | STS token 横轴 | 累计 token 数 |
| `solution_elapsed_sec` | `candidates.jsonl` 或 `solutions.jsonl` | STS 时间横轴 | 解生成累计耗时（秒） |
| `token_usage.total_tokens`、`llm_rounds`、`tool_calls` | `result.json` | 成本统计 | 总 token 数、LLM 轮数、工具调用数 |
| `profiling.active_time_us`、`aicore_time_us`、`aiv_time_us`、`core_count`、`vec_ratio`、`cube_ratio`、`mte2_ratio` | 候选 `profiling` | PAOS 资源指标；缺失则为 `null` | 活跃时间、AI Core 时间、AI Vector 时间、核数及硬件利用率比例 |

### 1.2 PAOS 最小 recipe

PAOS 使用简洁版，只需要一个 workload recipe。默认规则如下：

```json
{
  "runtime_scenario": "decode",
  "batch_size": 1,
  "chunk_len": null,
  "T_other_ms": 0,
  "ops": [
    {
      "task_id": "<operator-run-name>",
      "calls_per_step": "<n_i>"
    }
  ]
}
```

- `G` 是运行时场景量，不是固定的模型常量：decode 场景 `G = batch_size`，prefill 场景 `G = batch_size × chunk_len`。
- `runtime_scenario`、`batch_size` 和 prefill 的 `chunk_len` 必须由 benchmark 方按实际评测场景指定；不能用 Agent token 或候选累计 token 代替。
- `calls_per_step=n_i` 优先来自真实 workload recipe；缺少时默认使用 `1`，并在 PAOS 输出的 `input_source` 中标明，可由 CLI 覆盖。
- `T_other_ms` 表示未优化部分的 step 时间；缺失时使用 `0`，并标记为默认假设，不能把该结果解读为完整 workload 结论。

### 1.3 PAOS 输入变量规范

| 变量 | 含义 | 单位/类型 | 规范输入 | 推荐来源与默认值 |
| --- | --- | --- | --- | --- |
| `G` | 运行时场景下每 Step 实际处理的 token 数，用于把 step 时间换算为 token 吞吐 | token/step；正数 | decode：`G = batch_size`；prefill：`G = batch_size × chunk_len` | 优先 benchmark 显式 `G` 或由场景字段计算；可用 `--paos-g` |
| `runtime_scenario` | 当前 benchmark 的运行时场景 | `decode` 或 `prefill` | 必须由 benchmark 方指定，不能从 Agent token usage 推断 | recipe 的 `runtime_scenario`/`scenario` 或 `--paos-scenario` |
| `batch_size` | 一个 workload step 同时处理的样本数 | sample；正数，通常为整数 | 应填写当前场景实际 batch，不要填写候选数或 profiling 样本数 | recipe 或 `--paos-batch-size` |
| `chunk_len` | prefill 场景每个样本在当前 chunk 中处理的 token 数 | token/sample；prefill 时为正数 | decode 场景不需要；prefill 必须指定 | recipe 或 `--paos-chunk-len` |
| `calls_per_step` | 当前算子在一个 workload step 中被调用的次数，即公式中的 `n_i` | 次/step；非负数，通常为整数 | 应填写真实调用次数；同一算子多次调用应填写总次数，不要填写候选数或 profiling 样本数 | 优先 `recipe.ops[].calls_per_step`/`n_i`，可用 `--paos-calls-per-step` 覆盖，默认 `1` |
| `T_other_ms` | step 中除当前优化算子外的未优化耗时，包括其他算子和已知固定开销 | ms/step；非负数 | 必须与 `T_step`、`duration_ms` 使用相同的 step 和测量口径；未知时填 `0` 但必须保留默认来源标记 | 优先 `recipe.T_other_ms`，可用 `--paos-t-other-ms` 覆盖，默认 `0` |
| `core_count` | profiling 时参与执行或用于 active time 归一化的 AI Core 数量 | 个；正整数 | 应填写 profiling 实际设备和采集口径对应的核数；不能填写 Block Num、Mix Block Num 或设备总核数的猜测值 | 优先 profiling/硬件元数据；当前简洁版主公式不使用，缺失保持 `null` |

输入注意事项：

- `runtime_scenario`、`batch_size`、`chunk_len`、`calls_per_step`、`T_other_ms` 必须对应同一个 workload 场景，不能跨 batch、shape 或数据类型拼接。
- benchmark 方必须明确提供场景；没有场景或有效 `G` 时，脚本输出 `missing_runtime_G`，不使用静默默认值。
- `T_other_ms` 不能重复包含被优化算子的 `n_i × t_i`，否则会重复计算 step 时间。
- `core_count` 不改变 `PAOS_base`、`PTEI` 或 `PKAI`；它只服务于 active time、`U_i` 等 profiling 资源指标的归一化。
- 手工输入优先级高于 recipe；脚本会在 `paos.input_source` 中记录 `recipe`、`cli` 或 `default`。不确定时宁可使用默认值并标注，不要填入未经测量的估算值。
- `t_i^0` 和 `t_i^1` 分别取 baseline 与最佳通过候选的 `duration_ms`，不从代码推断。

## 2. 记录筛选规则

### 2.1 有效 baseline

baseline 必须同时满足：

```text
evaluation_kind == "baseline"
status == "passed"
correctness_passed == true
duration_ms > 0
```

优先使用 `result.baseline_duration_ms`，但只有它对应的 baseline 正确性有效时才打开性能门槛。没有有效 baseline 时，STS 和 PAOS 均为 `unavailable`。

### 2.2 有效候选

候选 `j` 必须满足：

```text
evaluation_kind == "candidate"
status == "passed"
correctness_passed == true
duration_ms > 0
```

失败候选计入候选总数和准确率分母，但不进入性能曲线。通过候选中最小的 `duration_ms` 为 `L_best`。

## 3. 准确率指标

### 3.1 Run 级指标

| 指标 | 公式 | 中文译名 |
| --- | --- | --- |
| `TaskPass` | 存在有效 baseline 或有效候选时为 `1`，否则为 `0` | 任务通过 |
| `CandidatePassRate` | `N_passed / N_completed`；分母为 0 时为 `null` | 候选通过率 |
| `CorrectnessGate` | 有效 baseline 且 `N_passed > 0` 时为 `1`，否则为 `0` | 正确性门槛 |
| `RunCompleted` | `operational_status == "completed"` 时为 `1`，否则为 `0` | 运行完成 |
| `FailureReason` | 原样保留 `termination_reason` | 失败原因 |

只有 `CorrectnessGate=1` 的 run 才进入 STS 和 PAOS 性能汇总。

### 3.2 Suite 级指标

令 `N` 为 run 总数：

$$
TaskPassRate = \frac{\sum_i TaskPass_i}{N},\qquad
GateRate = \frac{\sum_i CorrectnessGate_i}{N}
$$

## 4. STS 指标（metrics_master.md 2.1）

令 `L0` 为 baseline latency，`L_best(B)` 为 token 或时间预算 `B` 内所有有效候选的 best-so-far latency：

$$
S_i(B)=\frac{L_0}{L_{\mathrm{best}}(B)},\qquad
\mathrm{VLR}_i(B)=1-\frac{1}{S_i(B)}
                  =1-\frac{L_{\mathrm{best}}(B)}{L_0}
$$

### 4.1 最小 STS 输出

| 指标 | 计算公式 | 数据来源 | 中文译名 |
| --- | --- | --- | --- |
| `Speedup` | $S_i=\frac{L_0}{L_{\mathrm{best}}}$ | baseline/candidate `duration_ms` | 加速比 |
| `VLR` | $\mathrm{VLR}_i=1-\frac{1}{S_i}=1-\frac{L_{\mathrm{best}}}{L_0}$ | `Speedup` | 延迟降低率 |
| `TokensTo1%` / `TokensTo5%` / `TokensTo10%` | $\min\{T_j\mid \mathrm{VLR}_j\ge\Delta\}$，其中 $\Delta\in\{0.01,0.05,0.10\}$ | `cumulative_tokens`、`duration_ms` | 达到目标收益所需 token 数 |
| `TimeTo1%` / `TimeTo5%` / `TimeTo10%` | $\min\{W_j\mid \mathrm{VLR}_j\ge\Delta\}$，其中 $\Delta\in\{0.01,0.05,0.10\}$ | `solution_elapsed_sec`、`duration_ms` | 达到目标收益所需时间 |
| `AULS_token` | 对数 token 轴上的 best-so-far 面积 | `cumulative_tokens` | token 轴加速曲线面积 |
| `AULS_time` | 对数时间轴上的 best-so-far 面积 | `solution_elapsed_sec` | 时间轴加速曲线面积 |
| `STS_token` | $100\left(0.5\log_2 S(B_{\max})+0.5\,\mathrm{AULS}_{\mathrm{token}}\right)$ | token 曲线 | token 效率综合分 |
| `STS_time` | $100\left(0.5\log_2 S(B_{\max})+0.5\,\mathrm{AULS}_{\mathrm{time}}\right)$ | 时间曲线 | 时间效率综合分 |
| `S_equiv` | $2^{\mathrm{STS}_{\mathrm{token}}/100}$ | `STS_token` | 等效加速比 |

默认 `Δ ∈ {0.01, 0.05, 0.10}`，对应输出字段 `TokensTo1%`、`TokensTo5%`、`TokensTo10%` 及三档 `TimeTo`。未达标时字段为 `null`。可通过 `--targets` 覆盖默认阈值，阈值以小数传入。

### 4.2 AULS 公式

按横轴递增排列有效点，加入 baseline 起点，令：

$$
g_k=\log_2\left(S_i(B_k)\right),\qquad
\gamma_k=\frac{\ln(B_k/B_{k-1})}{\ln(B_K/B_0)}
$$

$$
AULS_i=\sum_{k=1}^{K}\gamma_k\frac{g_{k-1}+g_k}{2}
$$

其中横轴分别取 `cumulative_tokens` 和 `solution_elapsed_sec`。横轴缺失、非正数或有效点不足两个时，AULS 为 `null`。

### 4.3 Suite 级 STS 汇总

只对通过 `CorrectnessGate` 且对应指标非空的 run 汇总：

$$
\mathrm{EBGS}=\exp\left(\frac{1}{M}\sum_{i=1}^{M}\ln\left(S_i(B_{\max})\right)\right)
$$

$$
\mathrm{EIR}_{\Delta}=\frac{1}{M}\sum_{i=1}^{M}\mathbf{1}\left[\mathrm{VLR}_i(B_{\max})\ge\Delta\right]
$$

| 指标 | 计算公式 | 中文译名 |
| --- | --- | --- |
| `EBGS` | `exp(mean(log(S_i(Bmax))))` | 专家基线几何平均加速比 |
| `EIR_Δ` | `mean(1[VLR_i(Bmax) >= Δ])` | 达到目标收益的任务比例 |
| `mean(STS_token)` | 通过门槛 run 的 `STS_token` 算术平均 | 平均 token 效率综合分 |
| `mean(STS_time)` | 通过门槛 run 的 `STS_time` 算术平均 | 平均时间效率综合分 |
| `median_TokensTo1%` / `median_TokensTo5%` / `median_TokensTo10%` | 对应达标 run 的 `TokensTo` 中位数 | 各档达标 token 数中位数 |
| `median_TimeTo1%` / `median_TimeTo5%` / `median_TimeTo10%` | 对应达标 run 的 `TimeTo` 中位数 | 各档达标时间中位数 |

## 5. PAOS 简洁版指标（metrics_master.md 2.2）

PAOS 在 baseline 和最佳候选均通过、且 benchmark 提供有效





运行时场景量 `G` 时计算。脚本优先读取 recipe；`T_other_ms` 和 `calls_per_step` 仍可使用默认值，但 `G` 必须按 decode/prefill 场景由 benchmark 指定或计算，并在 `input_source` 中标明来源。

### 5.1 单算子 step 合成

对 recipe 中每个算子 `i`：

$$
t_i^0=\text{baseline duration},\qquad
 t_i^1=\text{best passed duration},\qquad
S_i^{lat}=\frac{t_i^0}{t_i^1}
$$

令 `n_i=calls_per_step`，`T_other` 为 recipe 中的未优化 step 时间：

$$
T_{step}^0=T_{other}+\sum_i n_i t_i^0,
\qquad
T_{step}^1=T_{other}+\sum_i n_i t_i^1
$$

$$
PTEI=eTPSGain=\frac{T_{step}^0}{T_{step}^1},
\qquad eTPS^r=\frac{G}{T_{step}^r}
$$

`eTPS` 的单位为 token/ms；如需 token/s，乘以 `1000`。`G` 是当前 decode/prefill 运行时场景量。

### 5.2 PKAI 与 PAOS

以 baseline token-time 为权重：

$$
w_i=\frac{n_i t_i^0}{T_{step}^0},\qquad
PKAI=\exp\left(\sum_i w_i\ln(S_i^{lat})\right)
$$

PAOS 简洁版：

$$
PAOS_{base}=\sqrt{PKAI\times PTEI}
$$

| 指标 | 计算公式 | 中文译名 |
| --- | --- | --- |
| `S_i^lat` | `t_i^0 / t_i^1` | 算子延迟加速比 |
| `T_step^0` | `T_other + Σ(n_i × t_i^0)` | 优化前 step 总耗时 |
| `T_step^1` | `T_other + Σ(n_i × t_i^1)` | 优化后 step 总耗时 |
| `eTPS^0` | `G / T_step^0` | 优化前估算 token 吞吐 |
| `eTPS^1` | `G / T_step^1` | 优化后估算 token 吞吐 |
| `PTEI` | `T_step^0 / T_step^1` | token 效率提升指数 |
| `PKAI` | `exp(Σ(w_i × ln(S_i^lat)))` | 加权算子加速指数 |
| `PAOS_base` | `sqrt(PKAI × PTEI)` | PAOS 简洁版综合分 |

对于单算子 recipe 且 `T_other_ms=0`，有：

$$
PKAI=S_i^{lat},\qquad PTEI=S_i^{lat},\qquad PAOS_{base}=S_i^{lat}
$$

### 5.3 PAOS 资源指标

脚本会按候选的 `evaluation_id` 自动读取 `profiling/<evaluation_id>/profile_output/**/op_summary_*.csv`。当前文件可提供 `Task Duration(us)`、`aicore_time(us)`、`aiv_time(us)`、`aiv_vec_ratio`、`aic_mte2_ratio` 和 `cube_utilization(%)`。多条 profiling 样本使用中位数聚合。若字段或 CSV 不存在，则对应值为 `null`。

可读取并输出的资源指标：

| 指标 | 公式 | 中文译名 |
| --- | --- | --- |
| `profiling.baseline.ActiveTimeUs` / `profiling.agent.ActiveTimeUs` | CSV 的 `aicore_time(us)`，无值时回退 `aiv_time(us)` | baseline/优化后 AI 活跃时间（微秒） |
| `profiling.S_i_aicore` | `baseline.ActiveTimeUs / agent.ActiveTimeUs` | AI Core 活跃时间加速比 |
| `profiling.U_i` | `agent.ActiveTimeUs / (CSV Task Duration(us) × 1000)` | 算子 AI 活跃利用率 |
| `profiling.U_step` | `Σ(n_i × agent.ActiveTimeUs_i) / T_step_agent` | step 级 AI 活跃利用率 |
| `profiling.VecRatio/CubeRatio/MTE2Ratio` | 读取 `aiv_vec_ratio`、`cube_utilization(%)`、`aic_mte2_ratio`/`aiv_mte2_ratio` | Vector/Cube/MTE2 利用率比例 |

当前轨迹的 `candidates.jsonl` 中硬件字段为 `null`，但 profiling CSV 可以提供上述资源指标；`core_count`、运行时场景和 `G` 当前仍无法从文件可靠获得，必须由 benchmark recipe/CLI 指定；`n_i`、`T_other_ms` 仍可按 recipe/CLI/默认值策略处理。

### 5.4 PAOS 缺失规则

| 情况 | PAOS 输出 | 中文译名 |
| --- | --- | --- |
| 缺场景或 `G` | `status=unavailable`，原因 `missing_runtime_G` | benchmark 未提供运行时场景量 |
| 缺 `calls_per_step` | 使用 `calls_per_step=1`，并标注来源 | 缺少每 step 调用次数 |
| 缺有效 baseline 或通过候选 | `status=unavailable`，原因 `correctness_gate_failed` | 未通过正确性门槛 |
| 缺 `T_other_ms` | 使用 `T_other_ms=0`，并标注来源 | 缺少未优化 step 耗时，结果为默认假设 |
| baseline 和候选完整 | 输出 `T_step^0/T_step^1`、`eTPS^0/eTPS^1`、`S_i^lat`、`PTEI`、`PKAI`、`PAOS_base` | 可计算 PAOS 简洁版 |

## 6. 推荐输出结构

脚本输出分为 suite 级 `accuracy`、`cost`、`sts`、`paos`、`catalog` 和 run 级 `runs[]`。所有不可计算项使用 `null` 或 `status=unavailable`，不使用 0 代替。每个 run 和 suite 都按大类组织，不把指标平铺混在一起。

输出同时面向两类消费者：

- **数值消费者**（看板、报表）：直接读取 `accuracy` / `cost` / `sts` / `paos` / `durations`，不需要自己再算聚合。
- **展示消费者**（前端 UI）：读取 `catalog` 拿到指标的中文名、单位、方向与友好定义，避免前端自行维护一套可能漂移的文案。

### 6.1 cost 的计费明细

`cost` 同时给出「消耗」与「可计费」两类信息，后者是 API 价格换算的输入：

| 字段 | 含义 |
| --- | --- |
| `InputTokens` | **未命中缓存**的输入 token（`uncached_input_tokens`，缺失回退 `input_tokens`） |
| `CachedInputTokens` | 命中缓存的输入 token（`cache_read_tokens`） |
| `CacheWriteTokens` | 写入缓存的 token（`cache_write_tokens`） |
| `OutputTokens` | 输出 token（`output_tokens`） |
| `UsageAvailable` | 原始 `token_usage.available`，用于区分“真的是 0”与“未采集” |

suite 级对应 `TotalInputTokens` / `TotalCachedInputTokens` / `TotalCacheWriteTokens` / `TotalOutputTokens`。

计费在下游完成，脚本**不预设任何价格**：

```text
cost = InputTokens/1e6 · input_price
     + CachedInputTokens/1e6 · cached_input_price
     + CacheWriteTokens/1e6 · cache_write_price
     + OutputTokens/1e6 · output_price
```

价格来源为使用者导入的价格表（前端 `src/lib/pricing.js` 或构建期 `scripts/pricing.json`）。
未配置价格的模型，`api_cost_usd` 保持 `null` 并给出 `api_cost_usd_reason`，不得用 0 或估算值替代。

### 6.2 catalog 的展示契约

`catalog` 携带展示层所需的全部元数据，前端据此渲染标签与释义：

```json
{
  "catalog": {
    "version": "1.0",
    "categories": [{ "key": "sts", "label": "STS · token 效率轴", "subtitle": "Speedup–Token Surface" }],
    "metrics": {
      "mean_STS_token": { "label": "STS Token", "short": "STS", "unit": "", "direction": "up", "category": "sts",
                          "definition": "主榜分：100·[0.5·log₂S(B_max) + 0.5·AULS_token]。" }
    }
  }
}
```

`direction` 取 `up` / `down` / `context`；`context` 表示仅作诊断、不参与排序。

JSON 结构如下：

```json
{
  "root": "operator",
  "n_runs": 3,
  "accuracy": {
    "TaskPassRate": 0.667,
    "GateRate": 0.333,
    "RunCompletedRate": 0.333
  },
  "cost": {
    "TotalTokens": 3716205.0,
    "median_TotalTokens": 942649.0,
    "TotalWallClockSec": 6876.33,
    "median_WallClockSec": 1659.63,
    "mean_LLMRounds": 19.0,
    "median_LLMRounds": 12.0,
    "TotalToolCalls": 55.0,
    "median_ToolCalls": 12.0,
    "TotalInputTokens": 3160765.0,
    "TotalCachedInputTokens": 0.0,
    "TotalCacheWriteTokens": 0.0,
    "TotalOutputTokens": 79129.0
  },
  "sts": {
    "EBGS": 1.013,
    "mean_Speedup": 1.013,
    "median_Speedup": 1.013,
    "max_Speedup": 1.013,
    "mean_VLR": 0.013,
    "mean_AULS_token": 0.0114,
    "mean_AULS_time": 0.009,
    "mean_STS_token": 1.514,
    "mean_STS_time": 1.396,
    "EIR_1%": 1.0,
    "EIR_5%": 0.0,
    "EIR_10%": 0.0,
    "median_TokensTo1%": 1021280.0,
    "median_TimeTo1%": 2451.44,
    "median_TokensTo5%": null,
    "median_TimeTo5%": null,
    "median_TokensTo10%": null,
    "median_TimeTo10%": null
  },
  "paos": {
    "status": "computed",
    "computed_runs": 1,
    "runtime_scenario": "decode",
    "G": 1,
    "batch_size": 1,
    "chunk_len": null,
    "T_other_ms": 0,
    "calls_per_step": [1],
    "input_source": {
      "recipe": "default",
      "G": "computed",
      "runtime_scenario": "cli",
      "batch_size": "cli",
      "chunk_len": "missing",
      "T_other_ms": "default",
      "calls_per_step": "default"
    },
    "mean_PAOS_base": 1.0132,
    "mean_PTEI": 1.0132,
    "mean_PKAI": 1.0132,
    "mean_eTPS_base": 120.18,
    "mean_eTPS_agent": 121.76,
    "mean_S_i_aicore": 0.9312,
    "mean_U_i": 0.00058,
    "mean_U_step": 0.4727,
    "mean_VecRatio": 0.118,
    "mean_CubeRatio": 0.0,
    "mean_MTE2Ratio": 0.0
  },
  "catalog": {
    "version": "1.0",
    "categories": [
      { "key": "accuracy", "label": "正确性", "subtitle": "Correctness" },
      { "key": "cost", "label": "成本", "subtitle": "Cost & effort" },
      { "key": "sts", "label": "STS · token 效率轴", "subtitle": "Speedup–Token Surface" },
      { "key": "paos", "label": "PAOS · 价值轴", "subtitle": "Profile Agent Optimization Score" },
      { "key": "durations", "label": "延迟", "subtitle": "Latency" }
    ],
    "metrics": {
      "PAOS_base": {
        "label": "PAOS",
        "short": "PAOS",
        "unit": "×",
        "direction": "up",
        "category": "paos",
        "definition": "价值轴主分 √(PKAI × PTEI)：既奖励算得快，也奖励优化在关键路径上。"
      }
    }
  },
  "runs": [
    {
      "task_id": "fused_gdn_gating-agent-20260918-185348",
      "status": "passed",
      "accuracy": {
        "TaskPass": 1,
        "CandidatePassRate": 1.0,
        "CorrectnessGate": 1,
        "RunCompleted": 1,
        "FailureReason": "agent_finished",
        "CandidatesTotal": 12,
        "CandidatesPassed": 12,
        "BaselineEvidence": "candidate_trace"
      },
      "cost": {
        "TotalTokens": 2297245.0,
        "TotalWallClockSec": 4243.54,
        "SearchWallClockSec": 4013.97,
        "LLMRounds": 33.0,
        "ToolCalls": 32.0
      },
      "durations": {
        "BaselineMs": 0.00832108,
        "BestMs": 0.00821284,
        "Speedup": 1.0132,
        "BaselineEvaluationId": "E0000",
        "BestEvaluationId": "E0007",
        "BaselineEvidence": "candidate_trace"
      },
      "sts": {
        "status": "computed",
        "Speedup": 1.0132,
        "VLR": 0.013,
        "AULS_token": 0.0114,
        "AULS_time": 0.009,
        "STS_token": 1.514,
        "STS_time": 1.396,
        "S_equiv": 1.0106,
        "TokensTo1%": 1021280.0,
        "TimeTo1%": 2451.44,
        "TokensTo5%": null,
        "TimeTo5%": null,
        "TokensTo10%": null,
        "TimeTo10%": null,
        "VerificationPassRate": 1.0,
        "VerificationMedianMs": 0.0084,
        "VerificationCV": 0.0122,
        "SpeedupConfirmed": 0.9851,
        "VLRConfirmed": -0.0151
      },
      "paos": {
        "status": "computed",
        "G": 1.0,
        "runtime_scenario": "decode",
        "batch_size": 1.0,
        "chunk_len": null,
        "T_other_ms": 0.0,
        "calls_per_step": [1.0],
        "T_step_base_ms": 0.0083,
        "T_step_agent_ms": 0.0082,
        "eTPS_base": 120.18,
        "eTPS_agent": 121.76,
        "S_i_lat": 1.0132,
        "PTEI": 1.0132,
        "PKAI": 1.0132,
        "PAOS_base": 1.0132,
        "profiling": {
          "S_i_aicore": 0.9312,
          "U_i": 0.00058,
          "U_step": 0.4727,
          "VecRatio": 0.118,
          "CubeRatio": 0.0,
          "MTE2Ratio": 0.0
        }
      },
      "anytime": {
        "BaselineMs": 0.00832108,
        "token": [{ "Tokens": 1021280.0, "BestMs": 0.00821284, "Speedup": 1.0132 }],
        "time": [{ "Seconds": 2451.44, "BestMs": 0.00821284, "Speedup": 1.0132 }]
      }
    }
  ]
}
```

下面逐块说明新增字段的用途：

| 块 | 字段 | 用途 |
| --- | --- | --- |
| suite `cost` | `TotalTokens`、`TotalWallClockSec`、`mean/median_LLMRounds`、`TotalToolCalls`、`median_ToolCalls` | 成本维度：token 量、wall-clock、轮数、工具调用，供 UI 直接展示而无需自行求和 |
| suite `sts` | `mean/median/max_Speedup` | 加速比聚合，供 UI 展示 `S_equiv` 与延迟摘要 |
| suite `paos` | `runtime_scenario`、`G`、`batch_size`、`chunk_len`、`T_other_ms`、`calls_per_step`、`input_source` | 场景可信度：标注 PAOS 是在哪个运行时场景下算出来的，以及哪些参数来自 recipe / CLI / 默认假设 |
| run `durations` | `BaselineMs`、`BestMs`、`Speedup`、`BaselineEvaluationId`、`BestEvaluationId` | 原始延迟与评测证据 ID，让延迟类展示不再需要反推 |
| run `anytime` | `token[]`、`time[]`（含 `Tokens`/`Seconds`/`BestMs`/`Speedup`） | 真实的 best-so-far 轨迹点，供 anytime 曲线直接绘制，不需要插值 |
| `catalog` | `categories`、`metrics{label,short,unit,direction,category,definition}` | 指标展示契约：中文名、单位、排序方向与友好定义，前后端共用一份文案 |

suite `paos` 中的场景字段取自第一条 PAOS 可计算的 run；当所有 run 都不可计算时这些字段为 `null`。`catalog` 与数值无关，仅描述指标语义，可被前端直接消费。

`sts` 包含三档 `EIR`、三档 `median_TokensTo` 和 `median_TimeTo`，档位由 `--targets` 决定；改变阈值后 `EIR_*` / `TokensTo*` / `TimeTo*` / `catalog` 之外的键名会同步变化。所有汇总分母只统计对应指标非 `null` 的 run；单 run 只能解释为点估计，不能据此声明可靠性。

### 6.1 运行命令

在项目根目录执行：

```bash
python evaluate/evaluate_operator_runs.py --root operator --output operator/eval_summary.json
```

**默认行为（按身份拆分）**：不给 `--no-split` 时，脚本会把 run 按 **agent × 驱动模型**拆成多个文件，
文件名由 run 里读到的身份自动推导（`slug(agent)-slug(model).json`，如 `opencode-qwen3-5-122b.json`），
写入看板数据集目录 `src/data/eval_datasets_summary/`：

```bash
# 采集 + 拆分 + 自动命名 + 落到看板目录
python evaluate/evaluate_operator_runs.py --root operator --paos-scenario decode --paos-batch-size 1

# 只列出将写入的文件名
python evaluate/evaluate_operator_runs.py --root operator --dry-run

# 写到别处
python evaluate/evaluate_operator_runs.py --root operator --out-dir operator/export
```

拆分**不丢 run**：身份解析不出的 run 仍会写入（文件名退化为 `{agent}.json` 或 `unidentified.json`）。
要回到「单个合并 summary」（`visualize_operator_metrics.py` 的输入），用 `--no-split`；
省略 `--output` 时直接打印 JSON；使用自定义目标时传入小数阈值：

```bash
python evaluate/evaluate_operator_runs.py --root operator --targets 0.01 0.05 0.10
```

手工指定 decode 场景 PAOS 输入：

```bash
python evaluate/evaluate_operator_runs.py --root operator --paos-scenario decode --paos-batch-size 1 --paos-t-other-ms 0 --paos-calls-per-step 1
```

手工指定 prefill 场景：

```bash
python evaluate/evaluate_operator_runs.py --root operator --paos-scenario prefill --paos-batch-size 1 --paos-chunk-len 512
```

导出与 `evaluate_operator_runs.py` run 级输出维度对齐的 CSV。CSV 按一个 run 一行，覆盖 `accuracy`、`cost`、完整 `sts`、`paos` 和 profiling 字段；列表或对象字段以 JSON 字符串保存，缺失值保持为空：

```bash
python evaluate/visualize_operator_metrics.py --input operator/eval_summary.json --output-dir operator/export
```
