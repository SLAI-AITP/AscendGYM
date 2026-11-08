# Operator 算子评测体系设计

> 本文是AscendGYM的评测方案，该评测集主要回答："在最终峰值性能和token效率上，Agent 能在多大程度上改进主流工作负载下的AscendC的内核效率？"

---

## 1. 设计目标

这套评测体系需要回答2个工程问题：

| 子问题 | 体系 | 回答 |
|---|---|---|
| Agent在多少Token成本下，能拿到多少**算子性能收益**？是否可靠、是否通用、需要多长时间？ | **STS**（Speedup–Token Surface） | Agent的**效率与能力** |
| 生成的算子能否带**模型token吞吐**提升？是否逼近**硬件理论性能上限**？ | **PAOS**（Profile Agent Optimization Score） | 优化的**价值与质量** |


因此，评测系统需要输入以下数据：

- 算子生成准确率
- 算子生成任务的Token消耗数量
- 算子生成任务总时长
- 算子利用率（被调用的次数）
- 算子性能收益

---

## 2. 数据源：必须来自SandBox的实际评测数据（本project归档至operator文档）

所有评测逻辑都必须从operator目录中读取原始产物，不能“靠估算”填充缺失数据。

### 2.1 必需输入

| 文件 | 用途 | 关键字段 |
| --- | --- | --- |
| operator/<run_id>/result.json | 任务运行总览 | run_id, baseline_duration_ms, best_duration_ms, speedup, token_usage, llm_rounds |
| operator/<run_id>/candidates.jsonl | 候选实现明细 | status, duration_ms, correctness_passed, cumulative_tokens, solution_elapsed_sec |
| operator/<run_id>/metrics.json | 任务级聚合摘要 | STS_token, STS_time, official_ranking_eligible, run_validity |
| operator/<run_id>/input_manifest.json | 输入快照和任务来源 | op_name, source_hash, files |
| operator/<run_id>/profiling/E000x/response.json | 功能验证结果 | status, passed/failed verification |

### 2.2 可选输入

| 文件 | 用途 |
| --- | --- |
| operator/<run_id>/recipe.json | PAOS 计算的工作负载描述 |
| operator/recipes/*.json | 统一 recipe 配置 |
| operator/<run_id>/profiling/*/report.md | profiling 诊断、硬件分析 |

### 2.3 数据来源约束

- result.json 负责任务状态和总览信息；
- candidates.jsonl 负责 candidate 层的细粒度性能数据；
- recipe.json 负责模型级工作负载价值，不可缺失时隐式填 1；
- `G` 是运行时场景量，必须由 benchmark 方指定：decode 场景 `G=batch_size`，prefill 场景 `G=batch_size×chunk_len`。缺少场景或场景参数时输出 `missing_runtime_G`，不静默补默认值。

---

## 3. 核心评测维度

评测体系分为 4 层：

### 3.1 正确性维度

- TaskPassRate：至少一个候选通过功能测试的任务占比
- SolutionPassRate：通过功能测试的候选解占候选总数比例
- CorrectnessGate：任务级硬门槛，只有存在通过解才允许进入性能榜

### 3.2 过程效率维度（STS）

- Speedup：已验证最佳实现相对 baseline 的性能放大倍数
- VLR：Validated Latency Reduction，延迟降低率
- AULS_token：在 token 预算下的 anytime 面积
- AULS_time：在 wall-clock 预算下的 anytime 面积
- TokensTo1% / TimeTo1%、TokensTo5% / TimeTo5%、TokensTo10% / TimeTo10%：首次达到 1%、5%、10% 收益所需 token 或时间
- STS_token / STS_time：综合主分数
- EBGS：跨任务几何平均加速比
- EIR_1% / EIR_5% / EIR_10%：分别至少达到 1%、5%、10% 收益的任务比例

### 3.3 工作负载价值维度（PAOS）

- PTEI：模型 step 吞吐增益
- PKAI：token-time 加权有效算子加速
- PAOS_base：工作负载价值基础综合分

PAOS 在正确性门槛通过时启用。recipe 优先提供真实 workload 参数，缺失参数由默认值或 CLI 手工参数补齐，并在输出中显式标注来源。

### 3.4 可靠性维度

- 多 run 达标率
- 中位数/分布/置信区间
- 单 run 只能作为点估计，不得声明稳定性结论

---

## 4. 指标清单与数据来源、公式

下面给出“每一个指标的数据源 + 计算公式”列表，方便开发者直接对照代码和原始文件。

| 指标 | 维度 | 数据来源 | 计算公式 / 语义 |
| --- | --- | --- | --- |
| TaskPassRate | 正确性 | candidates.jsonl + result.json | 任务中是否至少存在一个 status=passed 或 correctness_passed=true。任务级占比。 |
| SolutionPassRate | 正确性 | candidates.jsonl | 通过的候选数 / 完成的候选总数。 |
| CorrectnessGate | 正确性 | candidates.jsonl | Gate_i = 1 if ∃ passed candidate else 0。未通过则不能进入性能榜。 |
| Speedup | 性能 | result.json + candidates.jsonl | S_i = L_0 / L_best，其中 L_0 为 baseline_duration_ms，L_best 为已通过最优 candidate 的 duration_ms。 |
| VLR | 性能 | result.json + candidates.jsonl | VLR_i = 1 - 1 / S_i。表示延迟降低比例。 |
| AULS_token | STS | candidates.jsonl | 在 token 预算轴上，对 best-so-far 曲线做 log 空间归一化面积积分。 |
| AULS_time | STS | candidates.jsonl | 在 wall-clock 轴上，对 best-so-far 曲线做 log 空间归一化面积积分。 |
| TokensTo1% / TokensTo5% / TokensTo10% | STS | candidates.jsonl | 最早满足 VLR ≥ 1% / 5% / 10% 的 cumulative_tokens。若未达成，则记为 null/NA。 |
| TimeTo1% / TimeTo5% / TimeTo10% | STS | candidates.jsonl | 最早满足 VLR ≥ 1% / 5% / 10% 的 solution_elapsed_sec。若未达成，则记为 null/NA。 |
| EBGS | STS 汇总 | runs[].speedup_final | $\mathrm{EBGS}=\exp\left(\frac{1}{N}\sum_i\ln S_i\right)$，用于跨任务几何平均收益。 |
| EIR_1% / EIR_5% / EIR_10% | STS 汇总 | runs[].sts.VLR | $\mathrm{EIR}_{\Delta}=\frac{1}{N}\sum_i\mathbf{1}[\mathrm{VLR}_i\ge\Delta]$，其中 $\Delta\in\{0.01,0.05,0.10\}$。 |
| STS_token | STS | candidates.jsonl | $\mathrm{STS}_{\mathrm{token}}=100(0.5\,\mathrm{EndGain}+0.5\,\mathrm{AULS}_{\mathrm{token}})$。 |
| STS_time | STS | candidates.jsonl | $\mathrm{STS}_{\mathrm{time}}=100(0.5\,\mathrm{EndGain}+0.5\,\mathrm{AULS}_{\mathrm{time}})$。 |
| PTEI | PAOS | recipe.json | PTEI = T_step^0 / T_step^1，表示 step 吞吐增益。 |
| PKAI | PAOS | recipe.json + candidate speedup | $\mathrm{PKAI}=\exp\left(\sum_i w_i\ln S_i\right)$，对权重较大的算子进行加权。 |
| PAOS_base | PAOS | recipe.json + PTEI + PKAI | $\mathrm{PAOS}_{\mathrm{base}}=\sqrt{\mathrm{PKAI}\times\mathrm{PTEI}}$。 |
| Reliability@Target | 可靠性 | 多 run 的结果 | 在 K 次独立 run 中，达到给定目标的比例。当前 operator 多为单 run，故仅可作为 partial。 |

### 4.1 正确性门槛公式

任务级硬门槛：

$$
\mathrm{Gate}_i =
\begin{cases}
1, & \text{if any candidate passes},\\
0, & \text{otherwise}.
\end{cases}
$$

工程实现规则：

- 只要某个 candidate 的 status == "passed" 或 correctness_passed == true，就允许进入性能统计。
- 若所有 candidate 都失败，则任务状态为 failed 或 blocked，不能计入 speedup。

### 4.2 最终加速与 VLR 公式

其中，$L_0$ 为 `baseline_duration_ms`，$L_{\mathrm{best}}$ 为已通过候选中的最小 `duration_ms`。

Speedup：

$$
S_i = \frac{L_0}{L_{\mathrm{best}}}
$$

VLR：

$$
\mathrm{VLR}_i = 1 - \frac{1}{S_i}
                   = 1 - \frac{L_{\mathrm{best}}}{L_0}
$$

这两个指标都来自结果层，而不是从代码中推断。

### 4.3 AULS 公式

AULS 的核心思想：收益越早出现、且后续有持续保留，分值越高。

定义 best-so-far 曲线：

$$
L_{\mathrm{best}}(b)
  = \min\left\{L_j \mid j \text{ passed},\; \mathrm{axis}_j \le b\right\}
$$

再定义：

$$
S(b) = \frac{L_0}{L_{\mathrm{best}}(b)}
$$

归一化收益函数：

$$
q(S) = \frac{\log_2(S)}{1 + \log_2(S)}
$$

最后采用 trapezoid-like area integration：

$$
\mathrm{AULS}
  = \sum_{k=1}^{K}
    (x_k - x_{k-1})
    \frac{q(S_{k-1}) + q(S_k)}{2}
$$

这里的 x_k 是对数预算空间的归一化坐标。

### 4.4 STS 公式

STS 结合“最终收益”和“过程效率”两部分：

$$
\mathrm{EndGain} = q\left(S_{B_{\max}}\right)
$$

$$
\mathrm{STS}
  = 100\left(0.5\,\mathrm{EndGain} + 0.5\,\mathrm{AULS}\right)
$$

分别在 token 和 time 轴上计算：

- STS_token：token 预算轴
- STS_time：wall-clock 轴

### 4.5 PAOS 公式

脚本优先使用 recipe 中的真实场景字段；`G` 按 decode/prefill 规则计算，或由 benchmark 通过 `--paos-g` 显式指定。`T_other_ms` 和 `calls_per_step` 仍支持默认值与命令行覆盖，但 `G` 不使用静默默认值。

$$
T_{\mathrm{step}} = T_{\mathrm{other}} + \sum_i n_i t_i
$$

$$
\mathrm{PTEI} = \frac{T_{\mathrm{step}}^0}{T_{\mathrm{step}}^1}
$$

$$
\mathrm{PKAI} = \exp\left(\sum_i w_i\ln S_i\right)
$$

$$
\mathrm{PAOS}_{\mathrm{base}}
  = \sqrt{\mathrm{PKAI}\times\mathrm{PTEI}}
$$

输出必须包含 `input_source`，区分 `recipe`、`cli` 和 `default`。默认值不是外部估算，但使用默认值时应在结果中标注，不能将其解读为完整 workload recipe。

#### PAOS 输入变量规范

| 变量 | 含义 | 单位/合法范围 | 输入要求 |
| --- | --- | --- | --- |
| `G` | 当前运行时场景每个 workload step 对应的总 token 数，用于计算 `eTPS=G/T_step` | token/step；`>0` | decode：`G=batch_size`；prefill：`G=batch_size×chunk_len`；也可由 `--paos-g` 显式指定 |
| `runtime_scenario` | benchmark 当前运行时场景 | `decode` 或 `prefill` | 必须由 benchmark 方指定，可由 recipe 或 `--paos-scenario` 提供 |
| `batch_size` | 每个 workload step 同时处理的样本数 | sample；`>0`，通常为整数 | decode/prefill 均需与实际场景一致，可由 `--paos-batch-size` 提供 |
| `chunk_len` | prefill 当前 chunk 每个样本处理的 token 数 | token/sample；prefill 时 `>0` | 仅 prefill 使用，可由 recipe 或 `--paos-chunk-len` 提供 |
| `calls_per_step` | 每 step 调用被优化算子的次数 `n_i` | 次/step；`>=0`，通常为整数 | 必须是 workload 调用次数，不是候选数、profiling 次数或 LLM 调用次数；缺失时默认 `1`，可由 `--paos-calls-per-step` 覆盖 |
| `T_other_ms` | 不属于当前优化算子的其余 step 耗时 | ms/step；`>=0` | 不得包含 `n_i × t_i`；缺失时默认 `0`，可由 `--paos-t-other-ms` 覆盖 |
| `core_count` | profiling active time 归一化所使用的 AI Core 数量 | 个；正整数 | 只能来自实际 profiling/硬件元数据；不要用 Block Num、Mix Block Num 或猜测的设备总核数代替；当前不参与 PAOS 简洁版主公式 |

输入优先级为：显式 `G`/真实 recipe 场景字段 > CLI 手工参数 > 默认值。`runtime_scenario`、`batch_size`、`chunk_len`、`calls_per_step`、`T_other_ms` 必须使用同一 workload 的 shape、batch、数据类型和 step 定义；没有有效场景量时不计算 PAOS。脚本输出的 `input_source` 是审计依据。

---

## 5. 自动化脚本要求

### 5.1 评测脚本入口

建议入口脚本：

```bash
python evaluate/evaluate_operator_runs.py --root operator --paos-scenario decode --paos-batch-size 1
```

### 5.2 脚本职责

脚本必须完成以下 7 项：

1. 扫描 operator 目录中的所有 run 目录；
2. 识别 result.json 和 candidates.jsonl；
3. 解析正确性状态；
4. 计算 speedup、VLR、AULS、STS；
5. 汇总 suite 级指标；
6. 读取 recipe 或默认/手工输入，计算 PAOS 简洁版并输出各字段及输入来源；
7. 按 (agent × 驱动模型) 拆分结果，并根据身份自动推导文件名写入数据集目录
   （`slug(agent)-slug(model).json`），使产物可直接被看板读取，无需手工改名。

### 5.3 必须支持的输出结构

```json
{
  "root": "operator",
  "n_runs": 3,
  "accuracy": {
    "TaskPassRate": 0.667,
    "GateRate": 0.333
  },
  "sts": {
    "EBGS": 1.006,
    "EIR_1%": 1.0,
    "EIR_5%": 0.0,
    "EIR_10%": 0.0,
    "mean_STS_token": 0.67,
    "mean_STS_time": 0.62,
    "median_TokensTo1%": 12000,
    "median_TimeTo1%": 38.5,
    "median_TokensTo5%": null,
    "median_TimeTo5%": null,
    "median_TokensTo10%": null,
    "median_TimeTo10%": null
  },
  "paos": {
    "status": "computed",
    "runtime_scenario": "decode",
    "G": 1,
    "batch_size": 1,
    "T_other_ms": 0,
    "calls_per_step": [1],
    "T_step_base_ms": 10.0,
    "T_step_agent_ms": 9.87,
    "eTPS_base": 0.1,
    "eTPS_agent": 0.1013,
    "S_i_lat": 1.013,
    "PTEI": 1.013,
    "PKAI": 1.013,
    "PAOS_base": 1.013,
    "input_source": {"recipe": "default", "G": "default", "T_other_ms": "default", "calls_per_step": "default"}
  },
  "runs": [
    {
      "task_id": "fused_gdn_gating-...",
      "status": "passed",
      "accuracy": {"CorrectnessGate": 1},
      "sts": {
        "Speedup": 1.013,
        "VLR": 0.013,
        "AULS_token": 0.0083,
        "AULS_time": 0.0064,
        "STS_token": 1.34,
        "STS_time": 1.25,
        "TokensTo1%": 12000,
        "TimeTo1%": 38.5,
        "TokensTo5%": null,
        "TimeTo5%": null,
        "TokensTo10%": null,
        "TimeTo10%": null
      },
      "paos": {
        "status": "computed",
        "runtime_scenario": "decode",
        "G": 1,
        "batch_size": 1,
        "T_other_ms": 0,
        "calls_per_step": [1],
        "T_step_base_ms": 10.0,
        "T_step_agent_ms": 9.87,
        "eTPS_base": 0.1,
        "eTPS_agent": 0.1013,
        "S_i_lat": 1.013,
        "PTEI": 1.013,
        "PKAI": 1.013,
        "PAOS_base": 1.013
      }
    }
  ]
}
```

---

## 6. 运行判定规则与输出结构

脚本输出固定分为 suite 级 `accuracy`、`sts`、`paos` 和逐任务 `runs[]`，指标不平铺。run 级 `sts` 包含 `Speedup`、`VLR`、`AULS_token`、`AULS_time`、`STS_token`、`STS_time`，以及三档 `TokensTo1%/5%/10%`、`TimeTo1%/5%/10%`。suite 级 `sts` 额外包含 `EBGS`、三档 `EIR_1%/5%/10%` 和三档 `median_TokensTo`、`median_TimeTo`。

缺失或未达标统一输出 `null`，PAOS 通过 `status=unavailable` 和 `reason` 说明原因。汇总时仅使用对应指标非 `null` 的 run；失败任务不从 `runs[]` 删除。

推荐命令：

```bash
# 默认：按 agent × 驱动模型自动拆分并命名，写入 src/data/eval_datasets_summary/
python evaluate/evaluate_operator_runs.py --root operator --paos-scenario decode --paos-batch-size 1

# 合并为单个 summary（可视化脚本的输入）
python evaluate/evaluate_operator_runs.py --root operator --no-split --output operator/eval_summary.json
python evaluate/evaluate_operator_runs.py --root operator --no-split --targets 0.01 0.05 0.10
python evaluate/visualize_operator_metrics.py --input operator/eval_summary.json --output-dir operator/export
```

拆分文件名由 run 身份推导（`slug(agent)-slug(model).json`），不需要显式指定输出文件名；
`--dry-run` 可先查看将要写入的文件清单，`--out-dir` 可改写入位置。
身份解析不出的 run 不会被丢弃，文件名退化为 `{agent}.json` 或 `unidentified.json`。

PAOS 手工输入参数：

```bash
python evaluate/evaluate_operator_runs.py --root operator --paos-scenario decode --paos-batch-size 1 --paos-t-other-ms 0 --paos-calls-per-step 1
```

### 6.1 任务判定

- status == "passed" 或 correctness_passed == true：有效候选；
- 若无有效通过候选：任务进入 failed 或 blocked 状态；
- 不通过的 candidate 不能用作 speedup 计算；
- 结果要保留失败记录，避免“看起来很成功”的误判。

### 6.2 单 run 与多 run

- 单 run：只能计算点估计，不能宣称稳定性；
- 多 run（>=3）：可启动 Reliability 计算；
- 当前 operator 样例主要为单 run，因此要明确标注为 partial，不可误报为可靠结论。

### 6.3 排名规则

推荐的排序逻辑：

1. CorrectnessGate
2. STS 主分数
3. 可靠性下界
4. 成本（token/time）

但排序逻辑不应写死进脚本，脚本主职责是产出原始指标，不做“高明”但不透明的排序决策。

---

## 7. 工程落地顺序

### P0：基础可用

- 读取 result.json 和 candidates.jsonl；
- 计算 correctness gate、speedup、VLR、AULS、STS；
- 输出 run 级和 suite 级 JSON。

### P1：稳定性增强

- 多 run 聚合；
- 统计达标 run 数量；
- 计算 reliability。

### P2：PAOS 扩展

- 引入 recipe.json；
- 提供 G、n_i、T_other；
- 正确性门槛通过即可输出 PAOS 简洁版；使用默认参数时必须结合 `input_source` 解读，不能视作完整 workload 结论。

### P3：profiling 深化

- 已从 `profiling/<evaluation_id>/profile_output/**/op_summary_*.csv` 读取 active time、Vector/Cube/MTE2 比例；多样本使用中位数；
- 输出 `profiling.S_i_aicore`、`profiling.U_i`、`profiling.U_step` 及 baseline/agent active time；
- `core_count` 若文件中没有可靠字段则保持缺失；运行时场景和 `G` 必须由 recipe 或 CLI 指定，不使用默认值；`n_i`、`T_other_ms` 仍可使用 recipe、CLI 或默认值并记录来源；
- 在可用时继续扩展 roofline PPUI。

---

## 8. CSV 导出

评估结果通过 `visualize_operator_metrics.py` 导出为单一 CSV 文件，不再生成图表、图片或额外的可视化汇总 JSON：

```bash
python evaluate/visualize_operator_metrics.py \
  --input operator/eval_summary.json \
  --output-dir operator/export
```

输出文件为 `operator/export/operator_metrics.csv`，包含任务状态、CorrectnessGate、Speedup、VLR、AULS、STS 以及 1%、5%、10% 三档 TokensTo/TimeTo 指标。缺失值保留为空，不用 0 替代。

---

## 9. 最终结论

这套评测体系的核心设计原则是：

- 正确性永远先于性能；
- 用真实 operator 数据来计算，不做隐式补值；
- 既看单任务收益，也看过程效率；
- 对 PAOS 采用“有 recipe 才计算”的严格门槛；
- 缺失数据时明确降级，而不是输出假装完整的结果。

这使得它既保留了 metrics_master_optimized.md 的科学性，又具备工程落地性，能直接支撑开发者评审、性能对比和后续 benchmark 迭代

Token消耗
时间戳
性能提升
STS考察Token效率（100W Token得到多少性能提升，核算ROI）
PAOS 考察 Coding 与Agent的组合，计算算子性能优化的上限与性能优化空间

优化空间：基于硬件计算算子理论计算性能上限
