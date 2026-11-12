# CLAUDE.md

> AscendGYM 接入说明：通过 bridge 启动时，Backend 使用公开的 `bridge/bare_bench/sandbox_runner/main.py` 经 HTTP 调用编译服务，地址由 `SANDBOX_URL` 或启动参数确定。无需本机安装私有服务源码或 NPU Conda 环境。下文涉及本地 Conda、`SANDBOX_PATH` 和 `_test.sh` 的默认配置属于独立运行 Backend 的旧路径，不是 AscendGYM 推荐入口。当前服务契约见 [EXECUTION_BACKEND.md](../../../docs/EXECUTION_BACKEND.md)。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AscendC Tools Backend（`_renew` 性能评测变体）— 面向华为昇腾 NPU 算子优化的 Flask REST API 服务，作为 `ascendc_agent` 客户端的工具执行后端。

本仓库聚焦**算子性能优化（profile / fix / backfill）**，不再做 1D 算子生成、容器内 debug、operator_cleanup 等"全流程编排"工作。通过 AscendGYM bridge 启动时，编译/profiling 经公开 runner 访问配置的 HTTP 服务，不需要本机 NPU Conda 环境或 SSH 进入服务容器。

**与 `ascendc_tools_backend_zzc`（无 `_renew` 后缀）的关系**：两者已经分叉到不同方向，不建议直接同步代码。

## 启动服务

```bash
conda activate kg_backend
cd Backend
bash start.sh --port 7077 \
              --op-path /path/to/original_op_dir \
              --op-path-version /path/to/version_output_dir \
              [--native-tool]
# 或直接：python app.py --port 7077 ...
```

- `--op-path` / `--op-path-version` 覆盖 `config/settings.py` 的 `RUS_OP_PATH` / `RUS_OP_PATH_VERSION`，并在启动时注入到所有 service 实例（见 `app.py` 末尾的 service-instance 注入段）。
- `--native-tool` 或环境变量 `NATIVE_TOOL=1` 切到 Qwen3 `<function=…>` XML 工具调用解析（默认走 OpenAI JSON 格式）。
- 健康检查：`GET /health` → 返回 `tool_list`。
- 工具入口：`POST /api/`，请求体为 LLM 的 tool-call JSON（OpenAI 格式）；启用 native-tool 时为 Qwen `<tool_call>` XML，由 `parse_native_tool_call`（`app.py:64` 起）转回 OpenAI JSON 再交给 `check_tool_format`。

## 工具列表

`app.py` 第 49 行 `tool_list` 注册的 11 个工具，由 `tool_entrance`（`app.py:922`）统一分发：

| 工具名 | 服务文件 | 说明 |
|--------|---------|------|
| `code_backfill` | `code_backfill_service.py` → `code_backfill_2d_service.py` | 把 LLM 产出的代码写到 `<RUS_OP_PATH_VERSION>/<op_name>/<op_snake>/{op_host,op_kernel}/...` |
| `ascend_code_fix` | `ascendc_code_fix_rus_service.py` | 函数级 / 文件级代码替换（纯字符串解析，不调 LLM） |
| `list_board_files` | `list_board_files_service.py` | 列出某算子目录下 `op_host` / `op_kernel` 板块的文件 |
| `view_all_file_content` | `view_all_file_content_service.py` | 读取单文件全文（带版本切换） |
| `get_op_version` | `get_op_version_service.py` | 回读历史版本快照 (`<root>/op_backup/<op>_version{N}/`) |
| `ascend_code_profiling` | `ascendc_code_profiling_service.py` | 调用配置的 profiling runner 完成编译、测试和 profiling，再解析日志与采样结果 |
| `solve_gemm_tiling_optimization` | `solve_gemm_tiling_optimization_service.py`（→ `services/gemm/` MIP solver） | GEMM tiling 优化求解（独立子模块，详见 `services/gemm/CLAUDE.md`） |
| `search_from_windrag` / `search_from_windrag_error` | `search_from_windrag_service.py` | WindVector + rerank 的 RAG 检索（同一 handler 处理两种入口） |
| `load_cann_skills` | `load_cann_skills_service.py` | 加载 `knowledge_base/GTSOPS-CANN-skills/` 下的 CANN Markdown 技能 |
| `skill_summary` | `app.py:skill_summary` 直接实现 | 客户端在 global_best 刷新时调用的轻量摘要工具 |

辅助模块（未注册为工具，被其他 service 复用）：

- `services/profile_op_summary.py` — 把 msprof JSON 渲染成评测报告 Markdown，由 `ascendc_code_profiling_service` / `generate_profiling_report_service` 调用。
- `services/extract_msprof_fields.py` — 从 msprof CSV 抽字段。
- `services/generate_profiling_report_service.py` — 单独的"重新生成历史 profiling 报告"工具，目前未在 `tool_list` 中暴露，但被单测引用。
- `services/sandbox_compile_test_service.py` — 独立编译/测试封装；当前 AscendGYM profiling 命令由 `ascendc_code_profiling_service` 按 `PROFILING_CONFIG` 构造。

## 架构说明

### 请求流程

```
POST /api/  →  parse_native_tool_call (if NATIVE_TOOL=1)
            →  check_tool_format()  →  tool_entrance() 路由分发
            →  各 service.process()/execute()/...
            →  service 内部本地 subprocess（沙盒）或 HTTP 调用（RAG）
```

**不再有 SSH / Paramiko / `docker exec` / 容器内执行**。先前版本依赖的 `utils/container_manager.py` 已经下线（如需保留的 helper 都已内联到对应 service）。

### 关键层

- **`config/settings.py`** — 配置的单一入口，全部字段都能通过环境变量覆盖：
  - `RUS_OP_PATH`：待优化的原始算子根目录。
  - `RUS_OP_PATH_VERSION`：算子版本化输出根目录（优先级高于 `RUS_OP_PATH`；service 会先在 version dir 找代码，找不到再 fallback 到原 path）。
  - `SANDBOX_PATH`：SandboxFusion 中 `ascend_c` runner 样例目录（`<op_name>_test.sh` 所在地）。
  - `WINDVECTOR_CONFIG`：WindVector + rerank + rag_qa 三个 URL，供 `search_from_windrag_service` 使用。
  - `CODE_BACKFILL_CONFIG`：`supported_boards / supported_dtypes / supported_socs` 白名单（`code_backfill_2d_service` 用来校验入参）。
  - `SANDBOX_CONFIG`：`sandbox_path` / `timeout`（默认 3600s）/ `conda_env`（默认 `sandbox-npu-runtime`）。
  - `PROFILING_CONFIG`：`ascend_c_path` / `main_py_path` / `host` / `port` / `compile_timeout`（默认 1800000ms）/ `run_timeout`（默认 3600000ms）/ `profiling_runs`（默认 8）。
- **算子目录布局** — 现行布局**不再带 `_custom` 后缀**：`<RUS_OP_PATH_VERSION 或 RUS_OP_PATH>/<op_name>/<to_snake_case_compact(op_name)>/op_host/...`。带 `_custom` 的算子名本身合法（只是不再作为隐式标记）；`utils/backup.py::_find_custom_dir` 的语义已改为"包含 `op_host` 子目录的子目录"，与命名解耦。
- **`utils/naming.py`** — `to_snake_case_compact(...)` 是 service 拼路径用的工具函数；如 `SparseAttnSharedkv` → `sparse_attn_sharedkv`。当前 profiling 入口由 `PROFILING_CONFIG.main_py_path` 指定，不按这个名字寻找 `_test.sh`。
- **`utils/backup.py`** — 版本快照工具，`get_op_version` 用它列举/恢复 `op_backup/<op_name>_version{N}/` 下的历史代码。

### `ascend_code_profiling` 的执行机制

`ascendc_code_profiling_service` 调用流程：

1. `_resolve_op_root(op_name)` —— 优先从 `RUS_OP_PATH_VERSION` 取，否则回退到 `RUS_OP_PATH`，构造路径 `<root>/<op_name>/<op_snake>`。
2. 查找待执行测试脚本：优先 `<op_dir>/<op_snake>_test.py`；找不到时取目录里第一个 `.py` 文件作为 fallback。
3. 按 `PROFILING_CONFIG` 中的 runner 目录、脚本和 host/port 构造命令。AscendGYM bridge 将其指向公开 `sandbox_runner/main.py`，由 runner 经 HTTP 提交编译与 profiling；不执行算子目录里的 `_test.sh`。
4. 校验沙盒响应：若 `status == 'SandboxError'` 且 `compile_result` 为空 → 直接 `compile_failed` 并把 sandbox `message` 抽出来；否则按 AscendC / Triton 分支分别校验 `compile_result.return_code`、`run_result.return_code`、`profiling_result`。
5. 调 `profile_op_summary.profile_op_summary(..., op_name=...)` 渲染 Markdown 报告（标题始终带算子名，表格永远渲染 `Op Type / Input Shapes / Output Shapes` 三列）。
6. 返回四种状态：`success`、`compile_failed`、`no_run_result`、`parse_error`（解析失败时在 `message` 末尾附沙盒输出 tail 便于定位）。

### Native-tool 协议

- `app.py` 顶部 `_NATIVE_TOOL_MODE = bool(int(os.getenv('NATIVE_TOOL', '0')))`；命令行 `--native-tool` 会显式把它打开。
- 启用时 `tool_entrance` 看到 `data` 是 `str` 就先丢给 `parse_native_tool_call`（同文件 `app.py:64`）把 `<function=name><parameter=k>v</parameter></function>` 转成 OpenAI JSON，再交给 `check_tool_format`。
- 客户端侧也有一份等价的解析（`Client/agent.py::_parse_native_tool_call_inner`），两端要同步演进。

## 版本管理

仓库初始化了本地 git（无远程），用 commit 当快照：

```bash
git log --oneline
git checkout <commit_hash> -- .         # 回退
git add -A && git commit -m "描述"      # 新快照
```

## 数据目录

- `op_desc/` — 算子描述 JSON，被 `solve_gemm_tiling_optimization` / 部分 RAG 入口用。
- `knowledge_base/GTSOPS-CANN-skills/` — `load_cann_skills` 加载的 CANN 技能文档。
- `NPUKernelBench-45C2/` — 算子评测基准集。
- `module_test/` — 模块级单测（pytest 脚本，独立运行；典型如 `profiling_validation_unit_test.py` 覆盖 `ascend_code_profiling` 的状态机分支）。
