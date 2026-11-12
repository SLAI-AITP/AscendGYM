# Backend — tool execution service

> AscendGYM 接入说明：通过 bridge 启动时，Backend 使用公开的 `bridge/bare_bench/sandbox_runner/main.py` 经 HTTP 调用编译服务，地址由 `SANDBOX_URL` 或启动参数确定。无需本机安装私有服务源码或 NPU Conda 环境。下文涉及本地 Conda、`SANDBOX_PATH` 和 `_test.sh` 的默认配置属于独立运行 Backend 的旧路径，不是 AscendGYM 推荐入口。当前服务契约见 [EXECUTION_BACKEND.md](../../../docs/EXECUTION_BACKEND.md)。

The Backend is a single-process Flask service that executes everything
the Client's LLM cannot do itself: read operator source, edit it,
compile and profile via SandboxFusion, persist version snapshots, and
do RAG / GEMM-tiling lookups.

It has no LLM of its own and no concept of the search strategy
running on the Client. It just exposes a small set of named tools at
`POST /api/` and answers each call with a JSON dict.

For the top-level overview, see [`../README.md`](../README.md).
For the authoritative service map and quirks, see
[`./CLAUDE.md`](./CLAUDE.md).

---

## Variant note

This is the `_renew` performance-evaluation variant. It diverges in
non-trivial ways from `ascendc_tools_backend_zzc` —
cross-syncing code between variants is explicitly discouraged.

The big differences in this variant:

- **AscendGYM execution path.** The Backend invokes the public bridge runner,
  which uploads sources and tests to an HTTP compilation service. NPU execution
  happens at that service; local NPU Conda activation is not required for this path.
- **No `operator_cleanup`, no `code_framework_generate`, no
  `pytorch_golden_generate`.** This variant is performance-oriented,
  not full-pipeline.
- **No `_custom` suffix on operator directories.** Directory discovery
  keys on "contains an `op_host` subdir", not on naming.

---

## Module layout

```
Backend/
├── README.md                  ← this file
├── CLAUDE.md                  ← authoritative service map (read this)
├── app.py                     ← Flask app + tool_entrance dispatcher
├── start.sh                   ← shell wrapper around `python app.py`
├── requirements.txt
│
├── config/
│   └── settings.py            ← single config module (env-var driven)
│
├── services/                  ← one file per tool, plus shared helpers
│   ├── ascendc_code_profiling_service.py    ← compile + run + msprof
│   ├── ascendc_code_fix_rus_service.py      ← function/file level replace
│   ├── code_backfill_service.py             ← thin wrapper around _2d
│   ├── code_backfill_2d_service.py          ← writes op_host/op_kernel
│   ├── list_board_files_service.py
│   ├── view_all_file_content_service.py
│   ├── get_op_version_service.py            ← version snapshot recovery
│   ├── sandbox_compile_test_service.py      ← SandboxFusion wrapper
│   ├── generate_profiling_report_service.py ← re-render past reports
│   ├── profile_op_summary.py                ← msprof JSON → markdown
│   ├── extract_msprof_fields.py             ← CSV field extraction
│   ├── search_from_windrag_service.py       ← RAG + rerank
│   ├── load_cann_skills_service.py
│   ├── solve_gemm_tiling_optimization_service.py
│   ├── gemm/                                ← MIP GEMM tiling solver
│   └── kernel_meta/                         ← cached kernel metadata
│
├── utils/
│   ├── backup.py              ← op_backup/version{N}/ + best snapshot gating
│   ├── naming.py              ← to_snake_case_compact
│   ├── exceptions.py
│   ├── extract_code.py
│   ├── error2func_new.py
│   ├── json_utils.py
│   ├── sandbox_output_parser.py
│   ├── validators.py
│   └── vllm_client.py
│
├── module_test/               ← runnable scenario tests (no pytest)
│   ├── profiling_validation_unit_test.py    ← profiling state machine
│   ├── sandbox_compile_test.py
│   ├── sandbox_profile_test.py
│   ├── code_backfill_test.py
│   ├── code_compile_test.py
│   ├── code_debug_test.py
│   ├── code_generate_test.py
│   └── pytorch_golden_generate_test.py
│
└── knowledge_base/            ← CANN skill markdown loaded by load_cann_skills
```

---

## Running

### Requirements

```bash
conda activate kg_backend           # repo convention
pip install -r requirements.txt     # Flask 3.0.0, flask-cors 4.0.0, paramiko 3.4.0
```

`paramiko` is a historical dependency from the SSH-based pipeline;
not used in the active code paths of this variant but still imported
in legacy modules. It's safe to keep installed.

### Start

```bash
cd Backend
bash start.sh --port 7077 \
              --op-path /path/to/original_op_dir \
              --op-path-version /path/to/version_output_dir \
              [--native-tool] \
              [--min-improvement-epsilon 0.02]
```

Equivalent to:

```bash
python app.py --port 7077 \
              --op-path ... \
              --op-path-version ... \
              [--native-tool] \
              [--min-improvement-epsilon 0.02]
```

CLI flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--port` | `7077` | Flask listening port. |
| `--op-path` | `RUS_OP_PATH` env / settings.py default | Original (read-only) operator source root. |
| `--op-path-version` | `RUS_OP_PATH_VERSION` env / settings.py default | Versioned output root. All writes (backfill, fix, snapshots) go here. |
| `--native-tool` | off | Parse Qwen-style `<function=…>` XML tool calls in addition to OpenAI JSON. |
| `--min-improvement-epsilon` | `0.02` | Relative improvement required for `update_best_snapshot` to accept a new best. **MUST match Client's `orchestrator.min_improvement_epsilon`.** |

CLI flags overwrite the corresponding settings on the running service
instances directly (see `app.py` near the bottom). They do not mutate
the module-level dicts in `config/settings.py`.

Health check:

```bash
curl http://<host>:7077/health
# → {"status": "healthy", "available_tools": [...13 names...], ...}
```

---

## Tool list

`app.py:49` is the canonical registration. 13 tools dispatched via
`tool_entrance` at `POST /api/`:

| Tool name | Service file | Purpose |
|-----------|--------------|---------|
| `code_backfill` | `code_backfill_service.py` → `code_backfill_2d_service.py` | Write LLM code into `op_host` / `op_kernel`. |
| `ascend_code_fix` | `ascendc_code_fix_rus_service.py` | Function- or file-level replacement (pure string parsing, no LLM). |
| `list_board_files` | `list_board_files_service.py` | List files under an operator's board. |
| `view_all_file_content` | `view_all_file_content_service.py` | Read one source file. |
| `get_op_version` | `get_op_version_service.py` | Recover from `op_backup/`: `current` / `best` / `latest` / `<op>_version{N}`. |
| `ascend_code_profiling` | `ascendc_code_profiling_service.py` | Compile + run + msprof through SandboxFusion. The single most important tool. |
| `finish_task` | `finish_task_service.py` | Validate and acknowledge an explicit bare-model termination request; the client owns loop shutdown. |
| `solve_gemm_tiling_optimization` | `solve_gemm_tiling_optimization_service.py` (→ `services/gemm/`) | MIP-based GEMM tile sizing. Details: `services/gemm/CLAUDE.md`. |
| `search_from_windrag` / `search_from_windrag_error` | `search_from_windrag_service.py` | WindVector + rerank RAG retrieval. |
| `load_cann_skills` | `load_cann_skills_service.py` | Load CANN skill markdown. |
| `skill_summary` | inline in `app.py` | Lightweight summary called on Client `global_best` refresh. |

Helper modules not registered as tools but reused by services:

- `services/profile_op_summary.py` — render msprof JSON into the
  markdown profiling report embedded in the response.
- `services/extract_msprof_fields.py` — CSV field extraction.
- `services/sandbox_compile_test_service.py` — generic SandboxFusion
  wrapper used internally by `ascend_code_profiling`.
- `services/generate_profiling_report_service.py` — re-render a past
  report. Not in `tool_list`; module-test only.

---

## Request flow

```
POST /api/   { "name": "<tool>", "parameters": {...} }
       │       (or Qwen XML if NATIVE_TOOL=1)
       ▼
   parse_native_tool_call  (only if NATIVE_TOOL=1 + body is str)
       │
       ▼
   check_tool_format → (tool_name, tool_para, ground_truth?)
       │
       ▼
   tool_entrance dispatches to one of the 11 named handlers
       │
       ▼
   <tool_handler>(tool_para)
       │
       ▼
   <service>.process / .execute / ...
       │
       │   (e.g. ascend_code_profiling →
       │     public bridge runner → HTTP /run_code)
       ▼
   jsonify({ "success": bool, "tool_name": str, "tool_call_result": {...} })
```

For the AscendGYM entry point, the local subprocess is the bridge HTTP runner.
Compilation, testing and profiling happen at the configured execution service;
the Backend does not directly enter the service container.

---

## Operator directory layout

```
<RUS_OP_PATH_VERSION>/
├── <op_name>/
│   └── <to_snake_case_compact(op_name)>/
│       ├── op_host/...
│       └── op_kernel/...
└── op_backup/
    ├── <op_name>_version1/
    ├── <op_name>_version2/
    ├── ...
    └── .<op_name>_best.txt        ← best version pointer, JSON
```

- Reads prefer `RUS_OP_PATH_VERSION`; fall back to `RUS_OP_PATH` for
  the original baseline if version dir is absent.
- A successful profiling triggers `snapshot_to_op_backup` →
  `update_best_snapshot`. Each profiling becomes one immutable
  `<op>_versionN` directory.
- `.<op>_best.txt` is a single-line JSON: `{"version": N,
  "exec_time_us": float}`. It is updated only when the new run beats
  the previous best by ≥ `min_improvement_epsilon` (default 2%) —
  this is the gate that keeps `get_op_version("best")` consistent
  with the Client's notion of "real improvement".

The previous variant's `_custom` directory suffix is no longer used.
Operator names that contain `_custom` remain valid; the discovery
rule keys on "contains an `op_host` subdir", not on naming.

---

## Configuration

`config/settings.py` is the only config module. Every field is
overridable via environment variable. Sections that matter:

```python
RUS_OP_PATH              # env: RUS_OP_PATH
RUS_OP_PATH_VERSION      # env: RUS_OP_PATH_VERSION
SANDBOX_PATH             # env: SANDBOX_PATH

WINDVECTOR_CONFIG = { ... }   # env: WINDVECTOR_URL / RERANK_URL / RAG_QA_URL / ...

CODE_BACKFILL_CONFIG = {
    "2d_work_dir": RUS_OP_PATH,
    "supported_boards": ["op_host", "op_kernel"],
    "supported_dtypes": [...],
    "supported_socs":   [...],
}

SANDBOX_CONFIG = {
    "sandbox_path": SANDBOX_PATH,
    "timeout":      3600,             # env: SANDBOX_TIMEOUT
    "conda_env":    "sandbox-npu-runtime",
}

PROFILING_CONFIG = {
    "ascend_c_path":            SANDBOX_PATH,
    "main_py_path":             "./main.py",
    "host":                     "127.0.0.1",
    "port":                     "9000",
    "compile_timeout":          1800000,   # ms
    "run_timeout":              3600000,   # ms
    "profiling_runs":           8,
    "warmup_runs":              3,
    "profiling_statistic":      "mean",
    "min_improvement_epsilon":  0.02,      # MUST match Client orchestrator
}
```

`profiling_runs` is the total number of rounds. All 8 rounds are correctness
checked; the first 3 are warmup and excluded from the mean, leaving 5 measured
rounds. Configure with `PROFILING_RUNS=8` and `PROFILING_WARMUP_RUNS=3` before
starting Backend. Profiling responses echo the resolved protocol.

### Cross-process invariant

`PROFILING_CONFIG['min_improvement_epsilon']` here MUST match
`Client/config.yaml: orchestrator.min_improvement_epsilon`. If they
drift:

- The Backend may write `op_backup/.<op>_best.txt` to a version the
  Client would have rejected as "noise-level improvement".
- The LLM, on calling `get_op_version("best")`, gets that
  noise-version code as the starting point — corrupting its
  reasoning chain.

Both default to `0.02`. Changing one without the other silently
breaks the search.

---

## Tests

`module_test/` contains runnable scenario scripts — no pytest, no
fixtures, no real backend. Each file asserts with `sys.exit(1)` on
failure.

| File | What it covers |
|------|----------------|
| `profiling_validation_unit_test.py` | The profiling state machine: SandboxError handling, compile_failed branches, parse_error tail, test script search, MD report rendering, `min_improvement_epsilon` gate (5 scenarios). |
| `sandbox_compile_test.py` | SandboxFusion wrapper happy path. |
| `sandbox_profile_test.py` | Profiling response shape. |
| `code_backfill_test.py` | Backfill into op_host / op_kernel. |
| `code_compile_test.py`, `code_debug_test.py`, `code_generate_test.py`, `pytorch_golden_generate_test.py` | Legacy module tests; may target services not active in this variant. |

Run individually:

```bash
python module_test/profiling_validation_unit_test.py
```

Or sweep all of them:

```bash
for f in module_test/*.py; do printf "%-50s " "$f"; \
  python "$f" >/tmp/_out.log 2>&1 && echo OK || echo FAIL; done
```

(Some tests need a live backend or a populated `RUS_OP_PATH` and
will fail in isolation — `profiling_validation_unit_test.py` is the
one that's fully self-contained.)

---

## Conventions & gotchas

- **`Backend/CLAUDE.md` is the source of truth** for service-by-
  service behaviour and the SandboxFusion contract. This README
  gives orientation; CLAUDE.md gives detail.
- **No global lock on disk writes.** `op_backup/` snapshots are
  per-operator, so concurrent runs on different operators are safe.
  Two simultaneous runs on the SAME operator will race for
  `<op>_versionN` numbering and `.<op>_best.txt`.
- **Single-file Flask service.** `app.py` holds the dispatcher, all
  handler shims, and the `argparse` glue. Service instances are
  module-level singletons defined at the top.
- **Tool name = dispatch key = JSON `function.name`.** Adding a
  tool requires: a new service file, a new `if` branch in
  `tool_entrance`, and a matching `Client/tools/<name>/tool_call.json`
  in the Client repo. See `Client/tools/README.md`.
- **HTTP timeouts are large by design.** `SANDBOX_CONFIG['timeout']`
  is 1 hour; profiling routinely needs 10–30 min on large operators.
  Don't add aggressive retries — surface the slowness instead.
- **Native-tool parsing is opt-in.** Default is OpenAI JSON. Setting
  `NATIVE_TOOL=1` or passing `--native-tool` enables the
  `<function=…><parameter=…>…</parameter></function>` XML path. The
  Client must be running in the matching mode.
- **Local git repo only** in this variant — use commits as snapshots.

---

## See also

- [`CLAUDE.md`](./CLAUDE.md) — authoritative service map, profiling
  state machine, native-tool protocol.
- [`services/gemm/CLAUDE.md`](./services/gemm/CLAUDE.md) — MIP GEMM
  tiling solver internals.
- Top-level [`../README.md`](../README.md) — system overview.
- [AscendGYM bare harness](../README.md) — client configuration and supported entry point.
