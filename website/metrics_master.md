# Ascend 内核优化 Benchmark 指标总纲

> 本文档是评测集指标体系的**总纲**：记录两套指标体系（STS / PAOS）的选择原因、选定指标、计算公式与递推关系、所需数据及其就绪度。
> 依据：`孟博gpt对话_中文.txt`（STS 体系）、`基于Profiling的Ascend内核优化Benchmark指标设计.md`（PAOS 体系）、`EvalSet/leaderboard/seven_dimensions.md`（七维度业务面）、agent 实际采集数据（result.json / solutions.jsonl / op_summary.csv）实证。
> 配套：`EvalSet/leaderboard/eval_schema.md`（榜单列定义）、`docs/eval_design.md`（埋点设计）。

---

## 0. 一句话定位

Benchmark 要回答：**"在最终峰值性能和 token 效率方面，Agent 能在多大程度上改进主流工作负载下的 Ascend C 内核？"**

这句话天然拆成两个子问题，分别由两套体系回答：

| 子问题 | 体系 | 回答 |
|---|---|---|
| Agent 在 token 预算下能拿到多少**已验证加速**？多早、多广、多可靠？ | **STS**（Speedup–Token Surface） | Agent 的**效率与能力** |
| 加速是否转化为**模型 token 吞吐**提升？是否逼近**硬件峰值**？ | **PAOS**（Profile Agent Optimization Score） | 优化的**价值与质量** |

**七维度**（正确性/优化率/token/轮数/时间/利用率/可靠性）是这两套体系面向读者的**业务投影**，不是独立计算层。

---

## 1. 为什么是这两套体系（选择原因）

### 1.1 为什么需要 STS（token 效率轴）

内核优化是一个**有预算约束的 anytime 问题**：给定初始正确的 Ascend C 实现，Agent 消耗 t 个 token 能拿到多少已验证提升。仅报"最终加速比"或"每 token 加速比"都不够：

- **最终加速比**忽略了 Agent 多早找到优化——同样 1.2× 加速，1 万 token 找到 vs 400 万 token 找到，价值天差地别。
- **每 token 加速比**会让"快速肤浅优化"和"慢速深度优化"不可区分。
- 大加速（24h 后 10×）会被有限阈值集合（1.05/1.2/1.5/2.0×）**饱和**截断。

STS 的解法：**连续 `log₂ S`（无界、不饱和）+ 对数 token 轴下的 anytime 曲线面积（AULS）**，并按 `λ=0.5` 混合"峰值性能"与"尽早改进"。这是 Agent 能力的主排名轴。

### 1.2 为什么需要 PAOS（profiling 价值轴）

STS 只看"内核变快了多少倍"，但有两个盲区：

- **加速可能不在关键路径上**：一个占 step 时间 0.1% 的内核拿到 5× 加速，远不如占 20% 的内核拿到 5% 加速。STS 的单任务 `S_i` 不含 workload 权重。
- **加速可能以资源为代价**：变快但吃满内存、或离硬件峰值仍很远，价值打折。

PAOS 的解法：用 **model cluster profile recipe**（算子调用图谱 + tokens/step）把单算子 profiling 合成**模型级 token 吞吐**（eTPS），并分解为 `eTPSGain = 利用率增益 × AI Core token cost 增益`，再合层 `√(PKAI·PTEI)`。这把"内核加速"翻译成"模型真实收益 + 硬件逼近度"。

### 1.3 两套的耦合点

- **共享原子量**：`t_i`（kernel_time）、`status`（正确性）、`S_i = t_base/t_agent`（kernel 加速比）。
- **分叉点**：STS 把 `S_i` 喂进 anytime 曲线（横轴 = token）；PAOS 把 `t_i` 喂进 model step 合成（横轴 = AI Core time / utilization）。
- **正交**：一个 Agent 可以 STS 高（很快找到加速）但 PAOS 低（优化的是非关键算子）；反之亦然。两者**都保留、联合发布**。

---

## 2. 选定指标清单

### 2.1 STS 体系（token 效率轴）

| 指标 | 含义 | 角色 |
|---|---|---|
| `S_i(B)` | 已验证加速比 `L⁰/L^{best}(B)` | 原子量 |
| `VLR` | 延迟降低率 `1−1/S_i` | 可读量 |
| `EBGS@B` | 专家基线几何平均加速比 `exp(Σw·ln S_i)` | 跨任务峰值 |
| `EIR_Δ@B` | 改进广度（Δ∈{1,3,5,10}%）`Σw·1[VLR≥Δ]` | 广度 |
| `AULS_i` | 对数 token 轴下加速比曲线面积 | anytime 效率 |
| `STS_token` | 主榜分 `100·Σw·[0.5·log₂S(Bmax)+0.5·AULS]` | **主排名** |
| `STS_time` | 同上，横轴换 wall-clock | 基础设施效率诊断 |
| `Tokens-to-Δ` | 首达 Δ% 的累计 token | token 效率 |
| `S_equiv` | `2^(STS/100)` | 可解释等效加速比 |

### 2.2 PAOS 体系（profiling 价值轴）

| 指标 | 含义 | 角色 |
|---|---|---|
| `S_i^lat` | kernel latency speedup（= STS 的 `S_i`） | 原子量 |
| `S_i^aicore` | AI Core active time speedup `a_base/a_agent` | 资源加速 |
| `U_i` / `U_step` | AI Core 利用率 `a_i/t_i` / step 级 | 利用率 |
| `eTPS` / `eTPSGain` | 估算 tokens/s 及其增益 `T_step^base/T_step^agent` | **token 吞吐** |
| `AICoreTokenCostGain` | 每 token AI Core cost 降低 `A_token^base/A_token^agent` | 分解量 |
| `PKAI` | token-time 加权 kernel speedup `exp(Σw̃·ln S_i^lat,eff)` | kernel 加速能力 |
| `PTEI` | = `eTPSGain` | token 效率 |
| `PPUI` | peak utilization index `1+RHC` | 峰值逼近 |
| `PAOS` | 综合分 `√(PKAI·PTEI)`（简洁版）/ `PKAI^0.4·PTEI^0.4·PPUI^0.2`（完整版） | **主排名** |

### 2.3 七维度 → 体系映射

| 七维度 | 来自 STS | 来自 PAOS |
|---|---|---|
| ① 正确性 | FuncPassRate（硬门槛，单独发布） | — |
| ② 优化率 | `S_i` / EBGS / VLR / STS | `S_i^lat` / PKAI |
| ③ token 量 | cumulative_tokens / AULS / Tokens-to-Δ | — |
| ④ 交互轮数 | llm_rounds / wasted_round_rate | — |
| ⑤ 优化时间 | wall_clock / STS_time | — |
| ⑥ 利用率 | — | `U_i` / Vec·Cube Ratio / PPUI |
| ⑦ 可靠性 | Reliability@Δ（K≥3） | — |

---

## 3. 计算公式与递推关系

### 3.1 原子层（agent 采集 + profiling）

```
单解 j（一条 solution）：
  status_{i,j}   ∈ {passed, compile_failed, test_failed, test_mismatch, ...}
  t_{i,j}        = duration_ms                      # kernel_time（passed 才有效）
  T_{i,j}        = cumulative_tokens                # 生成该解累计 token
  W_{i,j}        = solution_elapsed_sec             # 生成该解累计 wall-clock
  a_{i,j}        = ai_core_time 或 aiv_time（见 §5） # AI Core active time

单任务 i：
  L_i^0          = baseline_duration_ms
  L_i^{best}(t)  = min(L_i^0, min{ t_{i,j} : T_{i,j}≤t ∧ status=passed })   # 不过 correctness 回退 baseline
  S_i(t)         = L_i^0 / L_i^{best}(t)            # 已验证加速比（≥1）
```

### 3.2 STS 递推（token 效率轴）

```
VLR_i(t)        = 1 − 1/S_i(t)                     # 延迟降低率
EBGS@B          = exp( Σ_i w_i · ln S_i(B) )        # 跨任务几何均值（排除 abnormal/saturated）

AULS_i          = Σ_k γ_k · ( log₂S_i(B_{k-1}) + log₂S_i(B_k) ) / 2      # 对数 token 检查点梯形
                  γ_k = ln(B_k/B_{k-1}) / ln(B_max/B_min)
                  （检查点 B_0<...<B_K=B_max 对数间隔）

STS_token       = 100 · Σ_i w_i · [ 0.5·log₂S_i(B_max) + 0.5·AULS_i ]    # λ=0.5
STS_time        = 同上，T_{i,j} 换 W_{i,j}
S_equiv         = 2^(STS/100)                       # 等效加速比

EIR_Δ@B         = Σ_i w_i · 1[ VLR_i^LCB(B) ≥ Δ ]   # 改进广度（LCB=置信下界，单 run 用点估计）
Tokens-to-Δ     = min{ t : VLR_i(t) ≥ Δ }           # 首达 token
```

**递推链**：`status+t_{i,j}+T_{i,j}` → `L_i^{best}(t)` → `S_i(t)` → `{EBGS, VLR, AULS, STS, EIR, Tokens-to-Δ}`。

### 3.3 PAOS 递推（profiling 价值轴）

需先有 **recipe**：每模型簇每场景的 `{G(tokens/step), ops:[{kernel, shape_id, calls_per_step:n_i}]}`。

```
# kernel 级
S_i^lat          = t_i^base / t_i^agent
S_i^aicore       = a_i^base / a_i^agent
U_i              = a_i^norm / t_i      （a_i^norm = a_i^sum / core_count，若 ai_core_time 已是单核则不除）
ΔU_i             = U_i^agent − U_i^base

# token 映射（kernel → 每 token 贡献）
τ_i              = n_i · t_i / G       （token time per kernel）
Δτ_i             = n_i/G · (t_i^base − t_i^agent)
α_i              = n_i · a_i / G       （AI Core token cost per kernel）

# step 级（合成模型 token step）
T_step           = Σ_i n_i·t_i + T_other      （未优化算子 speedup=1；无 overhead 数据时 T_other=0，标注 profile-only）
T_token          = T_step / G
eTPS             = G / T_step = 1/T_token
eTPSGain         = T_step^base / T_step^agent
A_token          = Σ_i n_i·a_i / G
AICoreTokenCostGain = A_token^base / A_token^agent
U_step           = A_token / T_token
eTPSGain         = (U_step^agent/U_step^base) · (A_token^base/A_token^agent)   # 分解恒等式

# 加权聚合
w_i              = n_i·t_i^base / T_step^base   （token-time 权重）
w̃_i              = w_i / Σ_j w_j                （评测 kernel 内归一化）
S_i^lat,eff      = S_i^lat if pass else 1.0
PKAI             = exp( Σ_i w̃_i · ln S_i^lat,eff )
PTEI             = eTPSGain

# 峰值（需 utilization 或 roofline）
RHC_U            = clip( (U_step^agent − U_step^base) / (1 − U_step^base), −1, 1 )
PPUI             = 1 + RHC_U        （或 roofline 版 1+RHC_roofline）

# 综合
PAOS             = √(PKAI · PTEI)                                  # 简洁版（推荐，不需 PPUI）
PAOS_full        = PKAI^0.4 · PTEI^0.4 · PPUI^0.2                  # 完整版
PAOS_final       = PAOS · FuncPassRate^λ   （λ=2，非正式榜 penalty）
PAOS_overall     = exp( Σ_c α_c · ln PAOS_c )                      # 跨模型簇几何均值
```

**递推链**：`t_i+a_i(可选)+n_i+G+status` → `{S_i^lat, U_i, eTPS}` → `{PKAI, PTEI, PPUI}` → `PAOS`。

---

## 4. 所需数据清单

| 数据 | 用途 | 来源 |
|---|---|---|
| `t_i` kernel_time | STS+PAOS 原子量 | `eval_result.duration_ms` / op_summary `Task Duration(us)` |
| `status` 正确性 | 硬门槛 | `eval_result.status` |
| `T_{i,j}` cumulative_tokens | STS anytime 横轴 | `solutions.jsonl.cumulative_tokens` |
| `W_{i,j}` wall-clock | STS_time 横轴 | `solution_elapsed_sec` |
| `a_i` AI Core active time | PAOS 利用率/分解 | op_summary `aicore_time(us)`（cube）/ `aiv_time(us)`（vector）|
| `Vec/Cube/MTE2 Ratio` | 利用率诊断 | op_summary `aiv_vec_ratio` / `aic_mac_ratio` / `aic_mte2_ratio` |
| `core_count` | `a_i` 归一化 | op_summary `Block Num` |
| `n_i` calls/step | PAOS step 合成 | **recipe**（需构造）|
| `G` tokens/step | PAOS step 合成 | **recipe**（需人工指定场景）|
| `FLOPs` / memory bytes | roofline（PPUI 完整版） | msprof `ArithmeticUtilization.csv` / `Memory.csv`（**当前未采集**）|

---

## 5. 数据就绪度（实证：当前 agent 数据能算什么）

> 实证来源：Swiglu run 的 `result.json` / `solutions.jsonl` / `op_summary_*.csv`（43 列真实表头）/ `models_configs/`。

### 5.1 STS 体系：**全部就绪** ✅

`status` + `duration_ms` + `cumulative_tokens` + `solution_elapsed_sec` 全部已在 `solutions.jsonl`。STS_token / AULS / EBGS / EIR / Tokens-to-Δ 可立即计算。

### 5.2 PAOS 体系：**简洁版可算，完整版受限**

| PAOS 指标 | 就绪度 | 说明 |
|---|---|---|
| `S_i^lat` / PKAI | 🟡 缺 recipe 的 `n_i` | `t_i` 有；token-time 权重 `w_i=n_i·t_i^base/T_step` 需 `n_i` |
| eTPS / PTEI | 🟡 缺 recipe 的 `n_i`+`G` | step 合成必需 |
| `S_i^aicore` / `U_i` / `U_step` / AICoreTokenCostGain | 🟡 `a_i` 部分有 | 见 §5.3 |
| PPUI（utilization 版） | 🟡 依赖 `a_i` | 同上 |
| PPUI（roofline 版）/ RHC_roofline | 🔴 缺 FLOPs+bytes | msprof 细分表未采集 |
| **PAOS 简洁版 `√(PKAI·PTEI)`** | 🟡 补 recipe 即可算 | 不需 PPUI |
| PAOS 完整版（含 PPUI/roofline） | 🔴 需重新 profiling | 见 §5.4 |

### 5.3 `a_i`（AI Core active time）的实证情况

op_summary 真实表头（43 列）含：`aicore_time(us)`（AI Cube Core）、`aiv_time(us)`（AI Vector Core）、`aiv_vec_time(us)`、`aic_mac_time(us)` 等。

- **Swiglu 是纯 vector 算子**：`aicore_time=0.0`（64/64 行全 0），主时间是 `aiv_time`。`select_metric_columns` 过滤掉全零列 → `result.json.hardware_metrics` 只剩 `{Task Duration, AI Vector Time, Vec Ratio}` 3 个 key。
- **结论**：PAOS 的 `a_i` 不能统一用 `aicore_time`——**vector 算子必须用 `aiv_time`**，cube 算子用 `aicore_time`。这是 Ascend 双核（Cube+Vector）架构的固有要求。当前 `hardware_metrics` 已含 `AI Vector Time`，对 Swiglu 可算 `U_i = aiv_vec_time/t_i`（近似）和 Vec Ratio gain。

### 5.4 FLOPs / memory bytes：**当前完全缺失** 🔴

- op_summary.csv（43 列）**无任何 bytes/FLOPs 列**（只有 time + ratio + cube_utilization）。
- msprof 的 `Memory.csv`（HBM read/write bytes）、`ArithmeticUtilization.csv`（cube/vec FLOPs）**当前 profiling 流程未产出**——`find` 全树零命中。
- `extract_msprof_fields.py` 能解析这 10 张细分表（字段映射准确），但（a）数据不存在、（b）它不在主流水线（`ascendc_code_profiling_service.py`）上，只被独立 service 调用。
- **解锁路径**：profiling 命令追加 `--analysis_type` 等参数重跑，产出 Memory.csv / ArithmeticUtilization.csv，再激活 extract_msprof_fields 接入主流水线。在此之前，roofline 类指标（RHC_roofline、Token Roofline Efficiency）算不了。

---

## 6. 缺失的 recipe 数据：能否从 model_config / profiling 派生？（实证）

recipe 需要两个量：`tokens_per_step (G)` 和 `calls_per_step (n_i)`。

### 6.1 `tokens_per_step (G)`：**不能自动派生，必须人工指定** ❌

- `models_configs/*.json`（deepseek_v4_pro / kimi_k3 / qwen3.6_27b）通篇是**静态结构字段**（hidden_size / num_hidden_layers / moe_intermediate_size 等），**无 batch_size / chunk_len / seq_len / scenario**。
- G 是**运行时场景量**：decode 场景 `G = batch_size`；prefill 场景 `G = batch_size × chunk_len`。须由 benchmark 方按场景指定。
- golden case 的 token 维（SwiGlu 的 `1×{1,6,32,512,1024,2048}×6144`，第二维 = token 数）是**算子测试用例的枚举**，选哪个对应哪个场景是人工决策，不是派生。

### 6.2 `calls_per_step (n_i)`：**结构部分可半自动派生，运行时部分不可派生** 🟡

以 SwiGlu @ DeepSeek-V4-Pro 为例（transformers 源码 `modeling_deepseek_v4.py` 实证）：

- 单个 decoder layer 内 SwiGLU 调用点：① 路由专家 `_apply_gate`（L1023-1030，每个命中专家 1 次）+ ② 共享专家 `DeepseekV4MLP.forward`（L986-989，固定 1 次）。
- **结构骨架（可派生）**：`n_SwiGLU/token/layer ≈ num_experts_per_tok(6) + 1 = 7`；整模型 `≈ 7 × num_hidden_layers(61) = 427`。`num_experts_per_tok`、`num_hidden_layers` 都在 config。
- **运行时精确值（不可派生）**：精确次数依赖 router 决策（每 step 实际命中哪些专家、batch 内 token 分布到 384 专家的情况），需 profiling 或负载假设。
- **可派生 vs 不可派生的分界**：层内固定次数算子（如 rmsnorm：`input_layernorm` + `post_attention_layernorm` + 最终 `norm` = `2×61+1=123`）能纯静态推出；**路由相关算子（SwiGLU/MoE）只能给上界/期望值**。

### 6.3 recipe 现状：**EvalSet 完全无此概念** ❌

- 全项目 grep `calls_per_step|tokens_per_step|model_cluster|step_profile|recipe` → 零命中。
- `eval_meta.json` 固定 4 字段 `{op_name, golden_root, default_model, default_shape}`（README 契约），无 recipe 扩展。
- **recipe 是需新建的层**：建议新增 `EvalSet/recipes/<model_cluster>_<scenario>.json`，不破坏 eval_meta 契约。

### 6.4 派生可行性总表

| 量 | 可派生？ | 来源 |
|---|---|---|
| `G` (tokens/step) | ❌ 必须人工指定 | 场景（decode=batch / prefill=batch×chunk） |
| `n_i` 结构骨架 | 🟡 半自动 | config `num_hidden_layers`/`num_experts_per_tok` + transformers 调用点统计 |
| `n_i` 运行时精确 | ❌ 需 profiling/假设 | router 决策 + token 分布 |
| shape 对齐（kernel↔模型） | ✅ 可派生 | config 维度（如 `6144=2×moe_intermediate_size`）+ golden case shape |

---

## 7. 落地优先级

1. **P0（数据全就绪，立即实现）**：STS 体系全套——`Eval/sts.py`，读 `solutions.jsonl` 算 STS_token / AULS / EBGS / EIR / Tokens-to-Δ。这是当前能完整跑通的主排名轴。
2. **P1（补 recipe 即可）**：PAOS 简洁版 `√(PKAI·PTEI)`——新建 `EvalSet/recipes/`，人工填 `G`、半自动派生 `n_i` 结构骨架；`a_i` 调 `GroupStats.active_time_us()`（坑2已修，自动按 cube/vector 选列）。
3. **P2（需重新 profiling）**：PAOS 完整版 PPUI/roofline——profiling 命令加 `--analysis_type` 产出 Memory.csv / ArithmeticUtilization.csv，激活 `extract_msprof_fields.py` 接入主流水线。
4. **P2（需多次 run）**：Reliability@Δ（K≥3 独立 run）。

---

## 8. 已知数据质量坑（实证发现）

1. **【✅ 已修 2026-08-06】`profile_op_summary.py` 列名不匹配**：候选列 csv 列名 `Block Dim`/`Mix Block Dim` 与真实表头 `Block Num`/`Mix Block Num` 不一致 → `core_count` 取不到。已把 CANDIDATE_METRIC_COLUMNS 的 csv 列名改为 `Block Num`/`Mix Block Num`（friendly label `Block Dim`/`Mix Block Dim` 保留不变，下游 key 不变）。实测：Swiglu `Block Num` 样本值=48 现已能进 groups。
2. **【✅ 已修 2026-08-06】vector 算子 `a_i` 用错列**：PAOS 若统一用 `aicore_time`，Swiglu 这类纯 vector 算子恒为 0（实测 64/64 行 `aicore_time=0.0`）。已在 `GroupStats` 新增 `active_time_label()`/`active_time_us()`：cube 优先 `AI Core Time (us)`，vector 回退 `AI Vector Time (us)`。实测 Swiglu 返回 `'AI Vector Time (us)'`、值 13.375us（非零）。**PAOS 计算层应调 `group.active_time_us()` 取 `a_i`，不要硬编码列名。**
3. **【⏳ 观望，暂不动】hw_metrics 白名单过滤**：主流水线 `_serialize_hw_metrics` 的 `HW_CORE_LABELS`（8 个 label，`ascendc_code_profiling_service.py:590`）只回传白名单字段，叠加 `select_metric_columns` 砍全零列。对纯 vector 算子（Swiglu）合理；但 cube/mixed 算子可能丢有用字段（如 Scalar Ratio）。**暂观望**：等真正有 cube 算子入库、且发现字段缺失影响 PAOS 计算时，再决定是否放宽白名单到 CANDIDATE_METRIC_COLUMNS 全集（18 个）。

---

## 9. 数据缺口建设待办（获取路径详记）

> 三类数据缺口挡住 PAOS 的不同部分。STS 全套不受影响（数据全就绪）。PAOS 简洁版 `√(PKAI·PTEI)` + utilization 版 PPUI 只需缺口 B+C；完整版 roofline 才需缺口 A。

### 缺口 A：FLOPs + memory bytes（解锁 PAOS roofline 指标）

- **缺什么**：`aic_cube_fops`/`aiv_vec_fops`（FLOPs）、`read_main_memory_datas`/`write_main_memory_datas`（HBM bytes）。用于 roofline 上限计算（`RHC_roofline`、Token Roofline Efficiency）。
- **为何缺**：当前 profiling 只产 4 个 csv（op_summary/op_statistic/api_statistic/task_time），均无 bytes/FLOPs 列（op_summary 43 列已确认）。这些量在 msprof 的 `Memory.csv`/`ArithmeticUtilization.csv` 里，但当前 profiling 流程未产出（全树零命中）。根因：`ascendc_code_profiling_service.py:291` 的 profiling 命令调 sandbox `main.py`，只传 `--profile --profiling-runs`，未传控制细分分析的参数。
- **获取路径（按推荐度）**：
  1. **【推荐】sandbox profiling 加 `--analysis_type` 重跑**：改 `ascendc_code_profiling_service.py:291` 的 `profiling_cmd` 加 `--analysis_type ...`，并确认 sandbox `main.py`（`PROFILING_MAIN_PY_PATH=./main.py`）透传给底层 msprof。**前置不确定点：sandbox 的 main.py 是否接受 `--analysis_type`，需先核实 sandbox 代码**。
  2. **激活 `extract_msprof_fields.py`**：它已能解析 10 张细分表（字段映射已验证准确），接入主流水线 `_extract_and_build_report`，产出后让 result.json 多一组 `msprof_detail_metrics`。
  3. **手动 msprof 兜底**：脱离 agent 自动流程，编译运行后手动 `msprof --application ... --analysis_type=ai-core`，适合先验证指标可行性。
- **影响范围**：只挡 PAOS roofline 版 PPUI。**PAOS 简洁版不依赖它**，可用 utilization 版 PPUI（`1+RHC_U`）顶上。

### 缺口 B：recipe 的 `tokens_per_step (G)`（解锁 PAOS step 合成）

- **缺什么**：每个 model cluster × scenario 的 `G`（一个推理 step 处理/生成的 token 数）。
- **为何缺**：`models_configs/*.json` 通篇静态结构（hidden_size/num_layers），无 batch_size/chunk_len/scenario。G 是运行时场景量。
- **获取路径**：**人工指定，不可自动派生**。decode `G=batch_size`；prefill `G=batch_size×chunk_len`。落地新建 `EvalSet/recipes/<model_cluster>_<scenario>.json`，字段 `{model_cluster, scenario, tokens_per_step, ops:[...]}`。SwiGlu golden case 已枚举 token 维（1/6/32/512/1024/2048），选哪个 G 对应哪个 scenario。

### 缺口 C：recipe 的 `calls_per_step (n_i)`（解锁 PAOS step 合成）

- **缺什么**：每算子在一个 step 里的调用次数 `n_i`。EvalSet 完全无 recipe 概念（全项目 grep 零命中）。
- **获取路径（半自动派生）**：
  - **结构骨架可派生**：`num_hidden_layers`(config) × 算子层内调用次数(transformers 代码)。SwiGlu 已查清：单 token 单层 ≈ `num_experts_per_tok(6) + 1(共享专家) = 7`，整模型 `≈ 7 × num_hidden_layers(61) = 427`。
  - **运行时精确值不可派生**：路由相关算子（SwiGlu/MoE）精确次数依赖 router 决策，需 profiling 真实负载或给假设。层内固定算子（如 rmsnorm `2×61+1=123`）可纯静态推出。
  - 落地：recipe 给期望值/上界（`n_i = 7 × num_layers`）即可，PAOS 的 token-time 权重 `w_i = n_i·t_i^base/T_step` 用期望值足够。

### 缺口建设优先级

| 缺口 | 工作量 | 解锁 | 前置 |
|---|---|---|---|
| B + C（recipe 文件） | ~1h（含派生 n_i） | PAOS 简洁版主轴 | 无 |
| A（细分表） | 改 sandbox `gen_prof_code_application.py:25` 的 msprof 命令加 `--analysis-type`（已核实） | PAOS roofline | 改的是 sandbox 仓库（非本项目），影响面大，**已搁置** |

---

*总纲依据实证：op_summary.csv 43 列真实表头、Swiglu hardware_metrics 实际 key、models_configs 结构字段、transformers V4 调用点行号、extract_msprof_fields.py 10 表映射。详见会话调查记录。*

---

## 10. `calls_per_step (n_i)` 推导方法论（缺口 C 补齐方案）

> 目标：对每个算子，推出"一个推理 step 内它被调用的次数 `n_i`"，喂给 PAOS 的 step 合成（`T_step = Σ n_i·t_i`）。
> 不写统计脚本，只给可复用的方法论 + SwiGlu 实例验证。

### 10.1 口径定义（必须先钉死，否则 T_step 会错）

PAOS 设计文档 §6.5 的 `τ_i = n_i·t_i/G` 定义了三者关系，由此钉死口径：

- **`t_i`** = profiled **单次算子调用**的耗时（op_summary 里 Task Duration）。这次调用可能处理 1 个 token，也可能处理 4096 个 token——t_i 已含其处理的全部 token，**不再除以 token 数**。
- **`n_i`** = 一个 step 内该算子被 **launch（调用）的次数**（profiler 视角：op_summary 里该 op_type 出现的行数）。
- **`G`** = 一个 step 处理/生成的 token 数。

**关键推论**：`n_i` 是"算子调用次数"，**不是"每 token 算子次数 × token 数"**。这两者在共享专家这类"一次调用处理全部 token"的算子上最容易混淆——共享专家一层只 launch 1 次（不管多少 token），`n_i=1`，不是 `n_i=token数`。

### 10.2 推导四步法（适用任意算子）

1. **读 config 拿层结构**：`num_hidden_layers`（决定纵向放大倍数）、算子相关字段（如 MoE 的 `num_experts_per_tok`/`n_shared_experts`）。
2. **读 transformers 源码数"一层内的调用点"**：在 `modeling_<model>.py` 的 decoder layer forward 里，grep 该算子的数学语义（如 SwiGlu = `act(gate)*up` / `chunk(2,...)`），统计**一层 forward 内它执行几次**。区分两种调用模式：
   - **固定调用**（与 token 数无关）：如 RMSNorm 每 layer 固定 2-3 次、共享专家每层 1 次。`n_i = 每层级数 × num_layers`。
   - **token/路由相关调用**（随负载变）：如 MoE 路由专家，`n_i = 该层实际命中专家数`。
3. **判断是否全层都有该算子**：查 `mlp_layer_types`/`first_k_dense_replace` 等，确认 61 层是否都用同种 FFN（V4 全 MoE，无 dense 层）。若有 dense/MoE 混合，按层类型分段计。
4. **纵向聚合**：`n_i = 每层级数 × num_layers`（固定调用）；路由调用按场景给期望值（见 10.3）。

### 10.3 decode vs prefill：路由算子 n_i 的场景差异（关键）

路由相关算子（MoE 路由专家）的 n_i **随场景变**，这是 n_i 推导里唯一的"非静态"部分：

| 场景 | G | 路由专家每层命中数（= launch 次数） | 说明 |
|---|---|---|---|
| **decode b1** | 1 | = `num_experts_per_tok`（6） | 1 个 token 命中 6 个专家，6 次 launch |
| **decode b32** | 32 | ≤ 384，期望 = min(384, 32×6 去重) | 32 token 各命中 6 专家，去重后被激活的专家数 |
| **prefill b1 chunk4096** | 4096 | ≤ 384，期望接近 384（但受 top_k=6 约束） | 4096 token 散开，绝大多数专家被至少 1 token 选中 |

- **共享专家**不受场景影响：每层永远 launch 1 次（一次吃全部 token）。
- **精确命中数需真实负载 profiling**（取决于 router 决策 + token 分布）。**期望值对 PAOS 加权足够**——PAOS 用 n_i 算 token-time 权重 `w_i = n_i·t_i^base/T_step`，权重的小偏差不影响排名结论。
- **取值建议**：recipe 里 decode 用确定值（`num_experts_per_tok + n_shared`）；prefill 给"期望命中数"并标注口径，或直接用 decode 的 n_i 近似（保守下界）。

### 10.4 实例验证：SwiGlu @ DeepSeek-V4-PRO

**第1步 config**：`num_hidden_layers=61`、`num_experts_per_tok=6`、`n_shared_experts=1`、`mlp_layer_types=("hash_moe","moe")`（全 61 层 MoE）。

**第2步 数调用点**（`modeling_deepseek_v4.py` 实证）：
- `DeepseekV4Experts._apply_gate`（L1027-1030）：`gate,up=chunk(2); act(gate)*up` —— 每个**命中**的路由专家执行 1 次。
- `DeepseekV4MLP.forward`（L989）：`act(gate)*up` —— 共享专家，一层固定 1 次（处理全部 token）。
- 每 MoE block = 路由专家集合 + 1 共享专家（L1098 `routed + shared_experts(residual)`）。

**第3步 全层确认**：61 层全是 MoE（前若干 hash_moe + 其余 moe，top_k 均 6），无 dense FFN。

**第4步 聚合**：
- **decode b1**：每层命中 6 路由 + 1 共享 = 7 → `n_i = 7 × 61 = 427`（确定值）。
- **prefill b1 chunk4096**：每层命中专家数期望 ≈ 接近 384（受 top_k=6 约束的实际上界）+ 1 共享 → `n_i` 期望 ≈ `(命中专家数+1) × 61`，范围 `[7×61, 385×61] = [427, 23485]`，需负载假设取期望。

**结论**：SwiGlu decode n_i=427 写进 recipe（确定值）；prefill 标注口径取期望或用 decode 近似。完整 recipe 见 `EvalSet/recipes/DeepSeek-V4.json`。

### 10.5 方法论适用范围与边界

- ✅ **适用**：所有 transformer 算子（RMSNorm/Attention/RoPE/SwiGlu/MoE 路由）。固定调用算子（RMSNorm/RoPE）能纯静态推出精确值；路由算子给期望值。
- ⚠️ **caveat**：n_i 是"期望/上界"，不是运行时精确值。对 PAOS 排名足够；若需精确模型级 eTPS，需真实 serving profiling（属缺口 A 同级的"重新 profiling"范畴）。
- 🔄 **可复用**：换算子/换模型时，重走四步法（config 层数 → 源码数调用点 → 全层确认 → 聚合），不需写统计脚本——源码数调用点靠 grep + 精读单层 forward 即可。
