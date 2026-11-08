<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="website/public/assets/logo/mark-dark.svg">
    <img src="website/public/assets/logo/mark-primary.svg" width="104" height="104" alt="AscendGYM logo">
  </picture>
</p>

<h1 align="center">AscendGYM</h1>

<p align="center">
  <strong>A Benchmark for Kernel Optimization on Ascend</strong><br>
  真实训练与推理工作负载上的算子优化评测
</p>

<p align="center">
  <a href="#quick-start">快速开始</a> ·
  <a href="website/public/AscendGYM_TechReport.pdf">技术报告</a> ·
  <a href="website/">项目主页源码</a> ·
  <a href="#documentation">文档</a> ·
  <a href="#license">许可</a>
</p>

---

AscendGYM 用于评测模型和 agent harness 优化昇腾算子的能力，提供从任务准备、沙盒编译与正确性测试，到性能测量和结果分析的统一入口。

| 任务与数据 | 评测方式 | 执行环境 |
|:---:|:---:|:---:|
| **17 个正式任务** · 全部冻结用例 | **Native / Bare** · 统一启动入口 | **AscendForge** · 昇腾沙盒 |

> **上传进度：** 仓库正在分模块上传，完整评测需等待 `AscendForge/`、`bridge/`、`datasets/` 和 `eval/` 全部合入。主页源码已纳入，在线部署将在准备完成后启用。

**使用导航：** [安装与环境](#installation) · [上游依赖](#upstream-dependencies) · [数据与任务](#data) · [模型配置](#models) · [结果查看](#results) · [常见问题](#faq)

## 仓库组成

```text
AscendGYM/
├── AscendForge/        # 沙盒服务和构建适配
├── bridge/             # 评测入口、配置、批次调度和工具
├── datasets/           # 正式任务、源码基线、冻结测试和 golden
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

| Harness | 适用方式 | 详细说明 |
|---|---|---|
| native | Claude Code、Codex、OpenCode 等原生 agent | [native_harness](eval/native_harness/README.md) |
| bare | 模型端点与工具池 | [bare_harness](eval/bare_harness/README.md) |

正式评测使用任务随包提供的全部冻结用例；数据目录另保留 3 个算子的 golden。

<a id="quick-start"></a>
## 快速开始

以下流程使用 bare harness 跑一个任务。需要 Python 3.10+、可访问的模型端点，以及已部署并可用的 AscendForge 沙盒。首次部署沙盒请先完成[安装与环境](#installation)和[上游构建依赖](#upstream-dependencies)。

### 1. 获取仓库并设置路径

```bash
git clone https://github.com/SLAI-AITP/AscendGYM.git
cd AscendGYM

export ASCEND_GYM_WORKSPACE="$PWD"
export ASCEND_GYM_DATA_ROOT="$ASCEND_GYM_WORKSPACE/datasets"
export SANDBOX_URL='http://<sandbox-host>:<port>'

python3 -m pip install -r "$ASCEND_GYM_WORKSPACE/requirements.txt"
```

将 `SANDBOX_URL` 替换为实际沙盒地址。已有仓库时，在仓库根目录设置上述变量即可。运行时也可使用 `source datasets/activate.sh` 设置工作区和随包数据根。

### 2. 准备模型与任务包

在 `bridge/configs/op_catalog.json` 的 `models` 中配置模型端点和认证方式，记下条目的 key。密钥使用环境变量传入，见[模型配置](#models)。

```bash
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py build --all
```

该命令从 `datasets/ops_src/` 生成评测用的 `datasets/packages/`。`packages/` 可以在安装后生成，不要求预先随仓库上传。

### 3. 运行一个 bare 任务

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

### 4. 查看结果

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

### 沙盒服务

算子的编译、正确性测试和 profiling 由 AscendForge 完成，需要兼容的昇腾硬件、驱动、CANN 和相关运行环境。部署方法见 [AscendForge](AscendForge/README.md) 和[部署说明](AscendForge/README_OPEN_SOURCE.md)。

上游构建仓库按下面的固定 commit 获取。客户端与沙盒可部署在不同机器上，客户端通过 `SANDBOX_URL` 访问服务。

<a id="upstream-dependencies"></a>
## 上游构建依赖

AscendGYM 根目录是唯一 Git 根，`AscendForge/` 作为普通目录随主仓发布。四个完整上游仓库单独下载到 `AscendForge/runtime/ascend_c/`，不随主仓复制，也不在主仓登记 `.gitmodules` 或 gitlink。版本同时记录在 [PUBLIC_DEPENDENCIES.json](AscendForge/PUBLIC_DEPENDENCIES.json)。

### 复现基线（固定 commit）

| 仓库 | 固定 commit | 对应 tag |
|---|---|---|
| ops-nn | `d93dc7dfef407a65a2c4139b9a46deefe3d02751` | 无 |
| ops-transformer | `250ea09c089246f6927ca5f0f799b84ceb047672` | 无 |
| ops-math | `a522538bb2cfa5f836ee99d488092ea2fadebd85` | 无 |
| vllm-ascend | `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db` | `v0.23.0rc1` |

三个 CANN 仓库的上述快照声明版本为 `9.1.0`，没有对应 tag。按 commit 检出，不要用其他版本的 tag 替代。

<details>
<summary><strong>获取和校验上游依赖：展开完整命令</strong></summary>

在四个目标目录尚不存在时执行：

```bash
ASCENDGYM_ASCEND_C="$ASCEND_GYM_WORKSPACE/AscendForge/runtime/ascend_c"
mkdir -p "$ASCENDGYM_ASCEND_C"

git clone https://gitcode.com/cann/ops-math.git "$ASCENDGYM_ASCEND_C/ops-math"
git -C "$ASCENDGYM_ASCEND_C/ops-math" checkout --detach a522538bb2cfa5f836ee99d488092ea2fadebd85

git clone https://gitcode.com/cann/ops-nn.git "$ASCENDGYM_ASCEND_C/ops-nn"
git -C "$ASCENDGYM_ASCEND_C/ops-nn" checkout --detach d93dc7dfef407a65a2c4139b9a46deefe3d02751

git clone https://gitcode.com/cann/ops-transformer.git "$ASCENDGYM_ASCEND_C/ops-transformer"
git -C "$ASCENDGYM_ASCEND_C/ops-transformer" checkout --detach 250ea09c089246f6927ca5f0f799b84ceb047672
git -C "$ASCENDGYM_ASCEND_C/ops-transformer" submodule update --init --recursive

git clone https://github.com/vllm-project/vllm-ascend.git "$ASCENDGYM_ASCEND_C/vllm-ascend"
git -C "$ASCENDGYM_ASCEND_C/vllm-ascend" checkout --detach f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db
git -C "$ASCENDGYM_ASCEND_C/vllm-ascend" submodule update --init --recursive

bash "$ASCEND_GYM_WORKSPACE/AscendForge/scripts/apply_public_patches.sh"
```

这里的 `submodule update` 只用于上游仓库自己的嵌套依赖，包括 ops-transformer 的 catlass `c8948c340c4f30fb5964ccdd6e3a0bc7615fe202` 和 vllm-ascend 的 catlass `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。主仓不使用 submodule 初始化流程。

补丁脚本会应用 `ops-transformer-host-protoc.patch` 等当前适用补丁。检出后可核对版本：

```bash
for repo in ops-nn ops-transformer ops-math vllm-ascend; do
    printf '%s %s\n' "$repo" \
        "$(git -C "$ASCENDGYM_ASCEND_C/$repo" rev-parse HEAD)"
done
```

输出应与表中的四个 commit 一致。gtsrus 源码已内置，评测任务和 golden 保存在 `datasets/`，不写入这些上游仓库。

</details>

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

算子目录通常包括 `op_host/`、`op_kernel/` 和 `<op>_test.py` 或 `op_speed_test/*.py`。冻结测试的接口要求见 [OPERATOR_TEST_GUIDE.md](docs/OPERATOR_TEST_GUIDE.md)。使用已有数据或预构建包时，可以将对应目录链接到数据根；确认链接目标在运行机器上可访问。

生成后的任务包包含 `agent/`、`spec.md`、`package.json`，以及适用任务的 `test_data/`。`build` 同时生成 `bridge/configs/op_catalog.resolved.json`。

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
| [AscendForge](AscendForge/README.md) | 沙盒部署与运行环境 |
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
| [项目主页](website/README.md) | 主页本地预览、静态构建与 GitHub Pages 部署 |

<a id="license"></a>
## 许可

本项目自行编写且未另行声明许可的评测框架、配置、测试脚本和文档采用 [Apache License 2.0](LICENSE)。该许可仅覆盖项目有权授权的原创内容；第三方源码、移植或派生内容、图片及其他材料继续适用其原有许可和版权声明。

算子快照和评测数据的来源及适用范围见 [datasets/README.md](datasets/README.md)，服务及内置组件的许可边界见 [AscendForge/THIRD_PARTY_NOTICES.md](AscendForge/THIRD_PARTY_NOTICES.md)。公开依赖的许可不由项目 Apache-2.0 声明替代。

内置 gtsrus 采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)，不属于项目 Apache-2.0 授权范围；其中第三方材料继续适用原有许可。包含该组件的使用和分发须遵守适用的署名、非商业等条件。
