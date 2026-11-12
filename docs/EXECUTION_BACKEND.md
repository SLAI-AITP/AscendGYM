# 编译执行服务协议

本协议描述现有 bridge 的 HTTP 接口。官方镜像与兼容的自建服务使用相同调用方式。环境搭建见[自建环境说明](SELF_HOSTED_ENVIRONMENT.md)。

## 1. 工具与服务的边界

```text
coding agent 调用评测工具
  → harness/Backend 定位候选源码和只读测试
  → bridge 读取并上传文件内容
  → 服务编译算子、运行冻结测试、采集 profiling
  → 返回日志和 profiling 文件
  → bridge 生成精度结论、耗时和诊断
```

native 的工具包括 `evaluate_baseline()` 和 `evaluate_candidate(candidate_snapshot_id)`；harness 将快照解析为本地 `source_dir` 后交给 evaluator。bare 通过 Backend 的 `ascend_code_profiling` 工具调用公开 runner。工具名和参数由 harness 定义，不是远端 HTTP 服务的接口名。

远端服务不能假设能读取客户端的绝对路径。现有 bridge 上传的是相对路径与文件内容；服务应在任务工作目录中还原文件，并将任务间的构建和运行产物隔离。

## 2. HTTP 接口

`SANDBOX_URL` 是服务基础地址，不包含 `/run_code`。`GET /ping` 用于健康检查；`POST /run_code` 接收 JSON，完成一次同步编译执行并返回 JSON。

| 请求字段 | 含义 |
|---|---|
| `language` | 当前正式算子任务使用 `ascend_c` |
| `ops_repo` | 构建后端，如 `nn`、`transformer`、`vllm_ascend`；服务需支持所选任务对应的后端 |
| `files` | `{相对路径: base64文件内容}`，包含源码、测试及适用任务的 golden |
| `code` | 执行驱动代码；当前 bridge 用它启动测试和 profiling，并生成结果归档 |
| `compile_timeout` / `run_timeout` | 编译与执行超时，单位为秒 |
| `fetch_files` | 需取回的文件名，当前包含 `build_output.log`、`profile_output.zip` |
| `lint` / `experimental` | 当前 bridge 可能传入的构建选项 |

示意请求，省略了实际代码和 base64 数据：

```json
{
  "language": "ascend_c",
  "ops_repo": "nn",
  "files": {
    "add_rms_norm/op_host/example.cpp": "<base64>",
    "main_exec_code.py": "<base64>"
  },
  "code": "<测试与 profiling 驱动代码>",
  "compile_timeout": 1800,
  "run_timeout": 3600,
  "fetch_files": ["build_output.log", "profile_output.zip"],
  "lint": false,
  "experimental": false
}
```

实际请求包含打包后的任务源码、冻结测试，以及按任务配置选取的冻结数据；正式评测应使用全部用例。服务必须保证测试能找到数据；与后端构建所需的路径布局不同时，可在工作目录中进行适配。构建后应在配置好的运行环境内执行 `code`，而不是只编译后直接返回成功。

## 3. 返回结果

```json
{
  "status": "Success",
  "message": "",
  "compile_result": {
    "status": "Finished", "return_code": 0,
    "stdout": "<编译标准输出>", "stderr": ""
  },
  "run_result": {
    "status": "Finished", "return_code": 0,
    "stdout": "<冻结测试和 profiling 输出>", "stderr": ""
  },
  "files": {
    "build_output.log": "<base64>",
    "profile_output.zip": "<base64>"
  }
}
```

失败时应返回真实的失败状态、退出码和诊断；编译失败时不应继续执行测试。超时和服务错误也必须明确返回，不能用成功标志代替实际执行。HTTP 成功或 `BUILD SUCCESSFUL` 仅说明部分链路完成，不能作为正确性与 profiling 验收结果。

冻结测试必须保持原有断言、容差和 golden，支持 `--runs N` 并重复完整用例集。bridge 识别测试输出中的成功/失败标志；具体格式见 [OPERATOR_TEST_GUIDE.md](OPERATOR_TEST_GUIDE.md)。服务应保留测试 stdout/stderr，避免隐藏失败。

## 4. Profiling 数据

当前 bridge 使用 msprof 的 ZIP 归档解析耗时。为兼容 native 与 bare，归档应保留非空 `op_summary*.csv`、原列名和每轮记录顺序；不建议把多份不同采集的数据放进同一归档。

- `Task Duration(us)`：任务时长指标，单位微秒，不等于整次请求的墙钟耗时。native 按算子标识分组，计算各组平均时长后相加，再转换为毫秒。
- `Op Name`、`OP Type` 或 `Kernel Name` 等标识列，用于区分 kernel。
- `Task Start Time(us)`：native 用于排序；`Task Type` 存在时，其值需能识别 AI kernel。bare 还使用 `OP Type`、`Input Shapes`、`Output Shapes`、`Input Data Types` 等列生成分组和报告，建议保留 msprof 原始汇总表。

自建服务采用其他 profiler 时，需将结果转换为现有解析器接受的格式，或修改相应适配器；仅返回一个任意格式的耗时数值不能直接接入当前 HTTP runner。

bare 当前固定为 8 轮总测试、3 轮预热、5 轮测量、mean。其解析器按记录行序划分轮次，要求总行数能被总轮数整除，并依赖各轮记录数量与顺序一致。native 按算子分组，仅在某组样本数可被总轮数整除时丢弃该组预热样本，否则保留该组全部样本；需结合实际报告确认预热处理。native 的 `--profiling-runs` 表示测量轮数，`--warmup-runs` 表示额外预热轮数，具体以本次配置为准。比较结果时应固定测试集、采样协议、硬件和软件环境，不能直接比较不同环境的绝对耗时。

## 5. 自定义接口与调度

实现上述协议的服务可直接通过 `SANDBOX_URL` 接入。现有 HTTP runner 不自动处理任意接口或认证协议，存在差异时应提供适配器。

native 可通过 catalog 的 `backend.command` 替换 evaluator。命令从 stdin 接收评测请求，从 stdout 返回 Evaluation JSON；必须符合 `eval/native_harness/core/contract.py` 的字段约束，并在有固定 profiling 协议的请求中回显 `raw.profiling_protocol`。这不是只替换 URL。

bare 当前使用 `bridge/bare_bench/sandbox_runner/main.py`。非兼容 HTTP 服务需要适配该调用路径，当前没有通用本地执行器配置开关。

bridge 可能发送 `required_npu`、`required_npu_types`、`preferred_node_id` 和 `preferred_npu_id`。这些属于调度约束，不是单 worker 的基本编译协议。直连单 worker 时应由部署方固定设备；不能认为额外字段一定得到执行。需要这些约束时，使用明确支持它们的调度服务。
