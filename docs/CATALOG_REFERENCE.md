# 算子优化清单配置（op catalog）

用于把"一份算子名清单 + 算子源码根目录"解析成 harness 可直接运行的输入：
自动定位 `op_host/`、`op_kernel/`、冻结测试脚本、金标数据，并生成规范的输入包、
任务描述、运行配置和启动命令。

<!-- toc -->
## 目录

- [1. 文件构成](#1-文件构成)
- [2. 清单配置怎么描述算子](#2-清单配置怎么描述算子)
- [3. 常用命令](#3-常用命令)
- [4. 输入包规则（由 harness 约束）](#4-输入包规则由-harness-约束)
- [5. 评测桥接的两个配套能力](#5-评测桥接的两个配套能力)
- [6. 当前清单（17 个算子）](#6-当前清单17-个算子)
- [7. 冒烟测试（smoke）](#7-冒烟测试smoke)
  - [本次发现并修复的问题](#本次发现并修复的问题)

<!-- /toc -->

## 1. 文件构成

| 文件 | 作用 |
|---|---|
| `configs/op_catalog.json` | **清单配置**：算子名列表、搜索根目录、包裁剪规则、预算默认值、agent 定义（根目录 `op_catalog.json` 为兼容软链） |
| `op_catalog.resolved.json` | 解析结果：每个算子最终选中的源码、测试、金标数据、告警（自动生成） |
| `op_manager.py` | 命令行工具：`scan` / `build` / `validate` / `config` / `run` |
| `<inputs_dir>/<op>_agent/` | 生成的 harness 输入包（只有 UTF-8 文本，可直接被 harness 读取） |
| `<inputs_dir>/<op>.package.json` | 该输入包的元数据：可编辑文件、测试路径、金标数据位置等 |
| `<inputs_dir>/<op>_spec.md` | 自动生成的算子任务描述 |

## 2. 清单配置怎么描述算子

`op_catalog.json` 的核心是 `search_roots` + `ops`：

```jsonc
{
  "source_root": "${ASCEND_GYM_DATA_ROOT}/ops_src/sandbox/client/example/cli/ascend_c",
  "search_roots": [
    {"id": "test_kernels:ops_nn",           "path": "ops_nn/test_kernels/kernels",        "ops_repo": "nn",        "bucket": "test_kernels"},
    {"id": "test_kernels:ops_transformer",  "path": "ops_transformer/test_kernels/kernels","ops_repo": "transformer","bucket": "test_kernels"},
    {"id": "vllm_ascend",                   "path": "vllm_ascend/kernels",                "ops_repo": "vllm_ascend", "bucket": "vllm"}
    // ...
  ],
  "ops": [
    {"name": "swi_glu",           "prefer": ["test_kernels:ops_nn", "stable:ops_nn"]},
    {"name": "mla_prolog",        "prefer": ["test_kernels:ops_transformer"], "data_cases": "all"},
    {"name": "fused_gdn_gating",  "prefer": ["vllm_ascend"]}
  ]
}
```

- `prefer`：候选目录的优先级；分辨率不足时自动回退到其余搜索根。
- `data_cases`：金标用例入沙盒的策略。缺省 `all`；`auto` 表示按用例体积从小到大
  选到字节预算（默认 12MB）为止——用于 `interleave_rope`、`rope_with_sin_cos_cache`、
  `mla_prolog` 这类金标总量几十 MB 的算子；也可写 `"0,4,9"` 指定用例。
- 单个算子目录的组成（`op_host/`、`op_kernel/`、`<op>_test.py`、`test_data/`）由工具自动识别，
  不需要在清单里逐项填写。

## 3. 常用命令

```bash
cd ${ASCEND_GYM_WORKSPACE}/bridge

# 1) 解析清单，打印每个算子的源码/测试/金标与告警
python3 op_manager.py scan

# 2) 生成输入包 + 任务描述（已存在的包默认跳过，--force 重建）
python3 op_manager.py build --all
python3 op_manager.py build --op swi_glu --force

# 3) 用 harness 的真实预检校验每个包（文件数、体积、UTF-8、可编辑/测试路径）
python3 op_manager.py validate --op all

# 4) 生成某个算子的运行配置（可选）
python3 op_manager.py config --op swi_glu --agent claude --max-model-turns 60

# 5) 直接启动（自动带齐 SANDBOX_* 环境变量）
python3 op_manager.py run --op swi_glu --agent claude \
    --run-id swi_glu-claude-mini-001 \
    --candidate-evaluations-max 1 --max-model-turns 60 --final-reprofile-count 1 \
    --profiling-runs 1

# 只查看将执行的命令与环境变量
python3 op_manager.py run --op swi_glu --agent claude --dry-run
```

> 说明：`op_manager.py run` 是单算子执行器，`--op` 必填单值、不支持 `--op all`；
> 一次跑全部/多个算子用 `start_all_runs.py`（见 [BRIDGE_USAGE.md](BRIDGE_USAGE.md) §5，它内部逐个调用 `op_manager.py run`）。
> `--run-id` 可省略：默认 `<op>-<agent>-<时间戳>`，也可用 `--tag` 指定批次标签
> （`--run-id` 优先级最高）。只有 `smoke` / `validate` 支持 `--op all`。

`--agent` 支持 `claude` / `codex` / `opencode`；三者都由 native harness 直接拉起各自 CLI，opencode 仅用于 `--harness native` 并使用其原生工具。

## 4. 输入包规则（由 harness 约束）

- 只收录 `op_host/`、`op_kernel/`、`op_graph/`、`framework/`、`CMakeLists.txt`、`README.md`；
  自动剔除 `*.pt`、`*.bin`、`*.so`、`*.log`、`__pycache__` 等非文本内容。
- 全包必须是 UTF-8 文本、不超过 256 个文件 / 8MB（harness 硬性限制）。
- 可编辑范围固定为 `op_host/**` 与 `op_kernel/**`；冻结测试位于 `op_speed_test/`，受 harness 写保护。
- 金标数据（`.pt`）**不会**进入输入包，由评测桥接在每次评测时按需注入沙盒。

## 5. 评测桥接的两个配套能力

`sandbox_evaluator.py` 为此新增：

| 环境变量 | 作用 |
|---|---|
| `SANDBOX_TEST_DATA_DIR` | 宿主机 `test_data/` 路径，桥接把它放到沙盒 `<op>/test_data/`，冻结测试即可按原有相对路径找到金标 |
| `SANDBOX_TEST_DATA_CASES` | `all`（默认）/ `auto`（按体积预算选小用例）/ `0,4,9`（显式用例） |
| `SANDBOX_TEST_DATA_MAX_BYTES` | `auto` 模式的字节预算，默认 12MB |
| `SANDBOX_EXPERIMENTAL` | 算子位于 `experimental/` 时置 1 |
| `SANDBOX_SUCCESS_PATTERN` / `SANDBOX_FAILURE_PATTERN` | 正确性判定的正则覆盖；默认已识别 `All tests PASSED`、`ALL CHECKS PASSED`、pytest `N passed`、`N/M checks passed` 与 `REGRESSION DETECTED`、`[FAIL]`、`N failed` 等标志 |

当只注入部分用例时，桥接会同步裁剪 `cases_manifest.json`，避免测试脚本引用缺失用例。

## 6. 当前清单（17 个算子）

| 算子 | 仓库 | 测试 | 金标 |
|---|---|---|---|
| fused_gdn_gating | vllm_ascend | pytest（CPU 金标，无需数据） | - |
| moe_token_unpermute | ops_transformer | golden-compare | 10 例 / 1.7MB |
| moe_fused_topk | ops_transformer | golden-compare | 10 例 / 0.2MB |
| interleave_rope | ops_transformer | golden-compare | 10 例 / 48.7MB（正式评测使用全部用例） |
| qkv_rms_norm_rope_cache | ops_transformer | golden-compare | 10 例 / 6.4MB |
| clipped_swiglu | ops_nn | golden-compare | 10 例 / 1.5MB |
| hc_post | vllm_ascend | pytest（CPU 金标） | - |
| swi_glu | ops_nn | golden-compare | 10 例 / 1.5MB |
| recurrent_gated_delta_rule | ops_transformer | golden-compare | 10 例 / 5.1MB |
| hc_pre_sinkhorn | vllm_ascend | pytest（CPU 金标） | - |
| moe_gating_top_k | ops_transformer | golden-compare | 10 例 / 0.2MB |
| rope_with_sin_cos_cache | ops_transformer | golden-compare | 10 例 / 53.7MB（正式评测使用全部用例） |
| add_rms_norm | ops_nn | golden-compare | 10 例 / 3.9MB |
| rms_norm | ops_nn | golden-compare | 10 例 / 2.0MB |
| lightning_indexer | ops_transformer | golden-compare | 10 例 / 1.2MB |
| sparse_flash_attention | ops_transformer | golden-compare | 10 例 / 5.2MB |
| mla_prolog | ops_transformer | golden-compare | 10 例 / 97.2MB（正式评测使用全部用例） |

说明：

- 4 个算子（`recurrent_gated_delta_rule`、`moe_gating_top_k`、`lightning_indexer`、
  `sparse_flash_attention`）在 `ops_transformer` 与 `vllm_ascend` 都有实现，清单默认选
  `ops_transformer`（自带金标测试），`vllm_ascend` 作为备选记录在解析结果里。
- 3 个 vllm_ascend 独有算子（`fused_gdn_gating`、`hc_post`、`hc_pre_sinkhorn`）的测试自带
  CPU 金标，不需要外部数据；其成功标志为 pytest 汇总行，桥接已适配。
- `vllm_ascend/kernels/lightning_indexer` 与 `sparse_flash_attention` 目录里没有测试脚本，
  因此这两个算子走 `ops_transformer` 的实现。

## 7. 冒烟测试（smoke）

`smoke` 子命令在真实沙盒上跑一次"编译 → 冻结测试 → msprof → 指标解析"，不调用模型、不消耗 token：

```bash
# 单个算子
python3 op_manager.py smoke --op swi_glu

# 批量（并发数按沙盒可用 worker 数给：先看 http://127.0.0.1:9000/status 的 is_available）
python3 op_manager.py smoke --parallel 4 --op add_rms_norm --op rms_norm --op mla_prolog

# 全部 17 个
python3 op_manager.py smoke --parallel 4 --op all
```

每个算子产出 `<runs_dir>/smoke-<时间戳>/<算子>/{smoke.json,artifacts/}`，
整批产出 `summary.json`；退出码非 0 表示有算子未通过。

### 本次发现并修复的问题

首轮冒烟中 3 个 `vllm_ascend` 独有算子全部 `test_failed`，根因是沙盒里 vllm_ascend 的
执行目录是仓库根、脚本被放在 `csrc/main_exec_code.py`，而桥接原先固定用 `./main_exec_code.py`，
msprof 找不到脚本报 `Script params are invalid`。修复方式：桥接改为解析脚本的绝对路径
（找不到时自动在目录树中搜索），并补充 cwd / msprof 版本等诊断信息与"profiling 失败时直接
跑一遍测试"的兜底。修复后 3 个算子全部通过。
