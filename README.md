<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="website/public/assets/logo/mark-dark.svg">
    <img src="website/public/assets/logo/mark-primary.svg" width="104" height="104" alt="AscendGYM logo">
  </picture>
</p>

<h1 align="center">AscendGYM</h1>

<p align="center">
  <strong>A Benchmark for Kernel Optimization on Ascend</strong><br>
  真实训推场景下的昇腾算子优化评测
</p>

<p align="center">
  <a href="#sandbox-setup">环境准备</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="website/public/AscendGYM_TechReport.pdf">技术报告</a> ·
  <a href="#documentation">文档</a> ·
  <a href="#license">许可</a>
</p>

---

AscendGYM 用于评测模型和 Harness 优化昇腾算子的能力，开源评测任务数据和评测工程，提供任务准备、正确性测试、性能测量和结果分析的统一入口。编译执行环境可使用在 Quay 公开发布、可直接拉取的官方 AscendForge 镜像，也可使用符合[服务协议](docs/EXECUTION_BACKEND.md)的自建服务；使用开源评测集和工程不要求获得官方镜像。

| 任务与数据 | 评测方式 | 执行环境 |
|:---:|:---:|:---:|
| **Batch 1 （17 Tasks）** · Batch 2/3 后续更新 | **Native / Bare** · 启动入口 | [官方镜像](docs/SANDBOX_IMAGE.md) / [自建服务](docs/SELF_HOSTED_ENVIRONMENT.md) |

## 项目主页

项目主页提供交互式评测榜单，可按模型、Harness 和算子查看结果，对比加速比、token 消耗及估算 API 成本，并查阅 STS 等指标说明和技术报告。

**访问地址：[AscendGYM 项目主页](https://slai-aitp.github.io/AscendGYM/)**。

**使用导航：** [环境准备](#sandbox-setup) · [客户端依赖](#installation) · [上游依赖](#upstream-dependencies) · [数据与任务](#data) · [模型配置](#models) · [结果查看](#results) · [常见问题](#faq)

## 仓库组成

```text
AscendGYM/
├── bridge/             # 评测入口、配置、批次调度和工具
├── datasets/           # 正式任务、源码基线、冻结测试和 golden
├── environment/        # 执行环境参考版本、上游版本与测试依赖清单
├── eval/
│   ├── native_harness/ # 原生 coding agent 自主循环
│   └── bare_harness/   # 裸模型与工具池
├── docs/               # 跨模块详细文档
├── examples/           # 启动示例
├── website/            # 项目主页、榜单与技术报告
├── requirements.txt
├── LICENSE
└── README.md
```

`environment/` 提供[参考软件栈](environment/reference-stack.json)、[上游仓库版本](environment/public-dependencies.json)和[测试侧 Python 依赖](environment/requirements-tests.txt)，用于查阅编译执行环境的配置。评测客户端的 Python 依赖见根目录 [requirements.txt](requirements.txt)，镜像部署步骤见下方[环境准备](#sandbox-setup)。

| Harness | 适用方式 | 详细说明 |
|---|---|---|
| native | Claude Code、Codex、OpenCode 等原生 agent | [native_harness](eval/native_harness/README.md) |
| bare | 模型端点与工具池 | [bare_harness](eval/bare_harness/README.md) |

正式评测使用任务随包提供的全部冻结用例；数据目录另保留 3 个算子的 golden。

<a id="quick-start"></a>
## 快速开始

以下流程使用 bare harness 跑一个任务。需要 Python 3.10+、可访问的模型端点，以及可访问的编译执行服务。

**操作顺序：安装客户端 → 准备镜像服务 → 配置模型与任务 → 运行评测。** 客户端和编译执行服务可以位于不同机器。

### 1. 获取仓库并安装客户端依赖

```bash
git clone https://github.com/SLAI-AITP/AscendGYM.git
cd AscendGYM

export ASCEND_GYM_WORKSPACE="$PWD"
export ASCEND_GYM_DATA_ROOT="$ASCEND_GYM_WORKSPACE/datasets"

python3 -m pip install -r "$ASCEND_GYM_WORKSPACE/requirements.txt"
```

已有仓库时，在仓库根目录设置上述变量并安装依赖即可。运行时也可使用 `source datasets/activate.sh` 设置工作区和随包数据根。

<a id="sandbox-setup"></a>
### 2. 准备编译执行环境

在配有 NPU 和兼容驱动的 **Linux arm64 执行主机**上拉取官方镜像，无需申请权限或登录 Quay：

```bash
export SANDBOX_IMAGE='quay.io/slai-aitp/ascendgym:v1.0.0-a3-cann9.0.1'
docker pull "$SANDBOX_IMAGE"
```

**拉取后，按[镜像指南中的启动命令](docs/SANDBOX_IMAGE.md#2-启动服务)启动服务**，配置 NPU 设备、驱动目录和端口映射。完整说明见 [SANDBOX_IMAGE.md](docs/SANDBOX_IMAGE.md)。已有服务时可直接进行下面的连接检查；也可按[自建环境指南](docs/SELF_HOSTED_ENVIRONMENT.md)准备兼容服务。

回到**评测客户端**，设置实际服务地址并检查编译、冻结测试及 profiling（不调用模型）：

```bash
export SANDBOX_URL='http://<sandbox-host>:9000'
curl --fail "$SANDBOX_URL/ping"
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py smoke --op add_rms_norm --cases all --profiling-runs 5
```

将 `<sandbox-host>` 替换为执行主机地址，端口与部署配置保持一致；客户端和服务在同一台机器时可使用 `127.0.0.1`。`/ping` 用于检查连通性，smoke 用于检查实际编译执行，结果说明见[安装与环境](#installation)。

### 3. 准备模型与任务包

在 `bridge/configs/op_catalog.json` 的 `models` 中配置模型端点和认证方式，记下条目的 key。密钥使用环境变量传入，见[模型配置](#models)。

```bash
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py build --all
```

该命令从 `datasets/ops_src/` 生成评测用的 `datasets/packages/`。`packages/` 可以在安装后生成，不要求预先随仓库上传。

### 4. 运行一个 bare 任务

将下面的 `<MODEL_KEY>` 替换为 catalog 中的模型条目 key：

```bash
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py run \
  --op swi_glu \
  --harness bare \
  --agent bare \
  --model '<MODEL_KEY>' \
  --candidate-evaluations-max 1 \
  --final-reprofile-count 1 \
  --sandbox-url "$SANDBOX_URL" \
  --tag BARE-DEMO-$(date +%Y%m%d-%H%M%S)
```

首次可加 `--dry-run` 离线检查命令与环境。该检查不探测模型协议、不启动代理、不提交评测，因此不能验证端点认证或 NPU 执行。

### 5. 查看结果

运行产物位于：

```text
$ASCEND_GYM_WORKSPACE/runs/<TAG>/<op>-<agent>-<TAG>/
```

bare 的主要结果为 `result.json`；native 的加速比见 `metrics.json` 中的 `raw_speedup`，候选记录见 `solutions.json`。更多操作见[监控、停止与结果](#results)。

<a id="installation"></a>
## 安装与环境

### 评测客户端

- bare：安装根目录 `requirements.txt`，不需要 Node 或 coding agent CLI。
- native：在客户端依赖基础上安装所选 agent 的 CLI。Claude Code 使用 `claude`，Codex 使用 `codex`，OpenCode 使用 `opencode`。
- Codex 需要模板 `CODEX_HOME`；OpenCode 需要可用的全局配置。具体要求见 [native harness 文档](eval/native_harness/README.md)。

### 编译执行服务

评测客户端通过 HTTP 提交算子源码和冻结测试，由服务完成编译、正确性测试及 profiling。客户端与服务可在不同机器运行；仅运行客户端时无需安装 CANN 或 NPU 驱动。

- **官方镜像**：AscendForge 不随本仓库提供源码。公开镜像为 `quay.io/slai-aitp/ascendgym:v1.0.0-a3-cann9.0.1`，可直接拉取；拉取、启动与接入方法见 [SANDBOX_IMAGE.md](docs/SANDBOX_IMAGE.md)。
- **自建服务**：自行准备编译工具链和算子绑定，提供兼容的执行接口。能力要求与参考环境见 [SELF_HOSTED_ENVIRONMENT.md](docs/SELF_HOSTED_ENVIRONMENT.md)。

两种方式均通过 `SANDBOX_URL` 连接。`op_manager.py run` 的 `--sandbox-url` 优先于环境变量；均未设置时默认 `http://127.0.0.1:9000`。不同协议的服务需要适配器，不能只替换 URL。

镜像拉取、服务启动入口、连接检查和 smoke 命令见[环境准备](#sandbox-setup)。

`--cases all` 指定全部冻结用例。未覆盖 `SANDBOX_WARMUP_RUNS` 时，命令运行 3 轮预热加 5 轮测量，以日志中的实际协议为准。`/ping` 只检查服务可达；`smoke` 汇总编译、测试输出和 profiling 解析结果。正式验收仍需核对全部测试轮次和错误日志，不能只凭一条通过标志或 profiler 的退出码认定全部测试成功。单个任务的结果不代表所有任务已通过。

<a id="upstream-dependencies"></a>
## 自建环境参考依赖

[公开依赖版本清单](environment/public-dependencies.json)记录用于参考构建环境的四个上游仓库及固定 commit。任务源码基线和 golden 已随 `datasets/` 提供；连接已有服务的客户端不需要下载完整构建仓库。

自建服务应按所选后端准备构建资源、匹配的 CANN/PyTorch 环境和测试调用所需的 Python 绑定。固定 commit 是复现信息，不是完整安装程序。详见[自建环境说明](docs/SELF_HOSTED_ENVIRONMENT.md)。仓库不依赖 `AscendForge/` 目录或 Git 子模块来安装评测客户端。

<a id="data"></a>
## 数据准备与任务注册

### 使用随包数据

正式任务索引为 [datasets/registry.json](datasets/registry.json)，数据来源与目录说明见 [datasets/README.md](datasets/README.md)。默认数据根为 `$ASCEND_GYM_WORKSPACE/datasets`；使用外部数据时可另设 `ASCEND_GYM_DATA_ROOT`。

```text
$ASCEND_GYM_DATA_ROOT/
├── ops_src/       # 算子源码基线、测试和冻结数据
└── packages/      # 由 build 命令生成的任务包
```

`ops_src` 的结构应与 catalog 的 `search_roots` 对应，例如：

```text
ops_src/
├── ops_nn/{test_kernels,stable,experimental}/kernels/<op>/
├── ops_transformer/{test_kernels,stable,experimental}/kernels/<op>/
└── vllm_ascend/kernels/<op>/
```

数据源中的算子目录应包含 `op_host/`、`op_kernel/`，以及直接放在该目录下的 `<op>_test.py`（或其他 `*_test.py`）。`build` 会把选中的测试复制到生成包的 `agent/op_speed_test/`；仅在原始源码的子目录放测试，不会被自动发现。冻结测试的接口要求见 [OPERATOR_TEST_GUIDE.md](docs/OPERATOR_TEST_GUIDE.md)。使用已有数据或预构建包时，可以将对应目录链接到数据根；确认链接目标在运行机器上可访问。

`build` 生成的任务包包含 `agent/`、`spec.md` 和 `package.json`，并生成 `bridge/configs/op_catalog.resolved.json`。golden 通过 `package.json` 中的 `data_dir` 指向原数据目录，提交评测时再由 bridge 上传；`build` 不会默认把 golden 复制到生成包内。

### 注册与校验任务

编辑 `bridge/configs/op_catalog.json`。算子名必须能在 `search_roots` 下找到，`prefer` 指定搜索优先级。例如：

```json
"ops": [
  {"name": "swi_glu", "prefer": ["test_kernels:ops_nn"], "data_cases": "all"},
  {"name": "rms_norm", "data_cases": "all"}
]
```

正式评测保持 `data_cases: "all"`。自定义调试子集的结果不应直接与完整用例结果比较。

```json
"source_root": "${ASCEND_GYM_DATA_ROOT}/ops_src",
"outputs": {
  "inputs_dir": "${ASCEND_GYM_DATA_ROOT}/packages",
  "runs_dir": "../runs",
  "native_root": "${ASCEND_GYM_WORKSPACE}/eval/native_harness"
}
```

构建并执行 native 的离线校验：

```bash
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py build --all
python3 op_manager.py validate --op swi_glu --harness native
```

bare 可用[快速开始](#quick-start)的运行命令加 `--dry-run` 检查配置。

<a id="models"></a>
## 模型配置

模型统一登记在 `bridge/configs/op_catalog.json` 的 `models` 中；运行参数 `--model` 接受条目的 key。

### 本地模型端点

```json
"my-local-model": {
  "provider": "local",
  "name": "DeepSeek-V4-Flash-BF16",
  "openai_base": "http://127.0.0.1:8008/v1",
  "anthropic_base": "http://127.0.0.1:8008",
  "wire_api": "chat",
  "context": 1000000,
  "max_output": 32768,
  "api_key": "any-value",
  "tool_format": "json"
}
```

上述地址、模型名、上下文长度和输出上限需要与实际服务匹配；`any-value` 仅适用于不校验密钥的本地服务。

| Harness / agent | 使用字段 |
|---|---|
| bare | `openai_base` |
| native / Claude Code | `anthropic_base` |
| native / Codex | `openai_base`、`wire_api` |
| native / OpenCode | `openai_base`，通过本次运行的配置传入 |

`wire_api` 支持 `chat` 或 `responses`；`tool_format` 支持 `json` 或 `xml`。

### 密钥配置

需要认证的模型条目配置环境变量名称：

```json
"api_key_env": "YOUR_API_KEY"
```

运行前设置对应环境变量。`api_key_env` 优先于 `api_key`，变量缺失或为空时会报错。bare 将解析后的值传给客户端 `API_KEY`；native 使用各 agent 的私有运行配置，协议代理从子进程环境接收密钥。

日志、`model.json` 和 bare 的 `manifest.json` 使用脱敏值。实际运行配置可能包含密钥，文件权限为 `0600`，`.per-run` 目录为 `0700`；不要将整个 `runs/` 目录作为可分享结果。

<a id="results"></a>
## 监控、停止与结果

所有 bridge 命令均在 `$ASCEND_GYM_WORKSPACE/bridge` 下执行。默认 `runs_dir` 为相对路径 `../runs`。

### 监控与停止

```bash
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 tools/monitor/native_monitor.py --tag '<TAG>' --single
python3 start_all_runs.py --harness bare --agent bare --ops all --tag '<TAG>' --status
```

前台运行可用 `Ctrl-C` 停止。批次与队列分别使用：

```bash
python3 start_all_runs.py --harness bare --agent bare --ops all --tag '<TAG>' --stop
python3 tools/task_dispatcher.py stop --queue '<QUEUE>'
```

### 读取结果

bare 的 `result.json` 示例：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["ASCEND_GYM_WORKSPACE"]) / "runs/<TAG>/<RUN_ID>/result.json"
d = json.loads(p.read_text())
print({k: d.get(k) for k in ("success", "speedup", "baseline_duration_ms",
                             "best_duration_ms", "termination_reason",
                             "candidate_evaluations")})
PY
```

native 的加速比读取 `metrics.json → raw_speedup`，候选记录读取 `solutions.json`。跨 harness 比较时固定算子、测试用例、模型、预算和测量条件，不直接比较不同条件下的绝对 `duration_ms`。

STS 的定义、计分窗口和 token 缺失时的处理见[指标说明](eval/native_harness/core/references/metrics_master.md)。

### 最佳版本复测

若最终验证在固定单卡上重复失败，而同一版本在其他卡上通过，可用 `bridge/tools/reprofile_best.py` 重新评测最佳版本。`--apply` 会串行完成复测与结果写回；次数默认沿用原 run，也可用 `--count` 指定。操作前阅读[工具说明](docs/BRIDGE_TOOLS.md)。

<a id="faq"></a>
## 常见问题

### bridge 是什么？

`bridge/` 是统一评测适配层，负责启动 harness、调度批次、监控、停止、复测和结果归一化。两个 harness 的实现位于 `eval/`。

### 两个路径变量有什么区别？

- `ASCEND_GYM_WORKSPACE` 是仓库根目录，用于定位 `bridge/`、`eval/` 和运行产物。
- `ASCEND_GYM_DATA_ROOT` 是数据根目录，包含 `ops_src/` 和 `packages/`；默认使用随包的 `datasets/`。

### native 的 Claude Code / Codex 需要手动启动协议代理吗？

bridge 会在实际运行前探测上游协议：Claude Code 使用 `/v1/messages`，Codex 使用 `/v1/responses`。支持原生协议时通过本地透传通路读取 usage；需要转换时由 `bridge/tools/proxy.py` 转换协议并计数。

本地代理绑定 `127.0.0.1` 随机端口，随 run 结束而关闭。`--dry-run` 不执行该探测。详见 [BRIDGE_TOOLS.md](docs/BRIDGE_TOOLS.md) 与 [BRIDGE_CONFIG.md](docs/BRIDGE_CONFIG.md)。

### Codex 的 CODEX_HOME 如何设置？

用户可设置模板目录：

```bash
export CODEX_HOME=/path/to/codex-template
```

模板至少包含 `models.json`，推荐其中含与 `--model` 同名的 `slug`；`config.toml` 可选。不设置时读取 `~/.codex`。

传入 `--model` 后，bridge 为本次运行生成独立目录 `runs/<TAG>/.per-run/<run_id>/codex-home/`，无 tag 时为 `runs/.per-run/...`，只注入本次 Codex 子进程。不要将用户模板变量指向这个临时目录。模板缺失或为空会报错。详见 [native harness 文档](eval/native_harness/README.md)。

<a id="documentation"></a>
## 详细文档

| 文档 | 内容 |
|---|---|
| [官方镜像](docs/SANDBOX_IMAGE.md) | 镜像平台访问、拉取、启动与接入 |
| [自建环境](docs/SELF_HOSTED_ENVIRONMENT.md) | 编译执行能力、参考依赖与验收 |
| [服务协议](docs/EXECUTION_BACKEND.md) | 工具调用、文件传输和返回结果 |
| [环境参考文件](environment/) | 参考软件栈、上游仓库版本与测试侧依赖清单 |
| [datasets](datasets/README.md) | 任务索引、冻结数据与来源 |
| [native harness](eval/native_harness/README.md) | 原生 agent 的配置与运行 |
| [bare harness](eval/bare_harness/README.md) | 裸模型与工具池 |
| [EVAL_OVERVIEW](docs/EVAL_OVERVIEW.md) | 评测栈概览 |
| [BRIDGE_USAGE](docs/BRIDGE_USAGE.md) | bridge 完整用法 |
| [BRIDGE_CONFIG](docs/BRIDGE_CONFIG.md) | 模型与运行配置 |
| [BRIDGE_TOOLS](docs/BRIDGE_TOOLS.md) | 复测、代理、监控和队列 |
| [CATALOG_REFERENCE](docs/CATALOG_REFERENCE.md) | catalog 字段参考 |
| [OPERATOR_TEST_GUIDE](docs/OPERATOR_TEST_GUIDE.md) | 冻结测试脚本接口 |
| [DATA_LAYOUT](docs/DATA_LAYOUT.md) | 自定义数据根布局 |
| [examples](examples/README.md) | 多种 agent 的启动示例 |
| [主页开发与部署](website/README.md) | 主页本地预览、静态构建与 GitHub Pages 部署 |

<a id="license"></a>
## 许可

本项目自行编写且未另行声明许可的评测框架、配置、测试脚本和文档采用 [Apache License 2.0](LICENSE)。该许可仅覆盖项目有权授权的原创内容；第三方源码、移植或派生内容、图片及其他材料继续适用其原有许可和版权声明。

算子快照和评测数据的来源及适用范围见 [datasets/README.md](datasets/README.md)，官方镜像及其中第三方组件以镜像平台提供的版本说明和组件附带的许可声明为准。开源评测工程的许可不替代镜像或公开依赖各自的许可。
