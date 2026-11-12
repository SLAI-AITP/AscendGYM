# 自建编译执行环境

开源评测任务与工程不要求使用官方镜像。自建环境需完成工具链配置和服务适配，提供编译、冻结测试执行及 profiling 能力，再通过[服务协议](EXECUTION_BACKEND.md)接入 bridge。依赖清单用于准备参考环境，不能替代执行服务本身。

## 1. 环境职责

| 位置 | 需要准备的能力 |
|---|---|
| 评测客户端 | Python 3.10+、根 requirements.txt；运行模型评测时另需模型端点，native 按所选 agent 安装 CLI |
| 编译执行主机 | 分配的 NPU、兼容驱动和 CANN、C++ 编译器/CMake 等构建工具、可用的 profiling 工具 |
| 测试运行环境 | 匹配的 torch/torch_npu、numpy，以及任务需要的算子 Python 绑定 |
| 执行适配层 | 接收源码和只读测试、编译、执行、采集 profiling、返回日志和数据 |

这几部分可以在不同机器上，连接已有服务的客户端不需要 NPU。CANN、驱动和系统构建工具不由 `pip install -r` 安装。

## 2. Python 与构建依赖

根 [requirements.txt](../requirements.txt)用于评测主机，不是 NPU 编译环境的完整依赖清单。测试的公共 Python 依赖说明见 [environment/requirements-tests.txt](../environment/requirements-tests.txt)。torch 与 torch_npu 需按目标架构、Python ABI、驱动和 CANN 的兼容组合安装；不要将任意两个版本混用。

[reference-stack.json](../environment/reference-stack.json)记录已有验证记录的参考组合；它不是所有任务或平台的最低版本要求，也不是自建环境已通过验收的声明。[公开仓库版本表](../environment/public-dependencies.json)保留四个构建仓库的 commit；按需获取构建资源，包含所需嵌套依赖和适用的构建修正。任务源码快照、冻结测试和 golden 位于 `datasets/`，不要通过重新生成 golden 来适配编译失败或精度不一致。

## 3. 算子调用约定

当前 17 个正式任务中，nn/transformer 的 14 个测试使用 `import gtsrusops as ops`，调用 `ops.npu_<operator>(...)`；另外 3 个 vLLM 类任务使用 `vllm_ascend` 注册的算子。自建编译流程必须产生这些测试可调用的扩展，或提供保持相同调用语义的适配层。

`gtsrusops` 在这里是编译扩展模块名，不能当作已经发布的通用 pip 包。函数参数顺序、输出张量、dtype、shape 和返回结构以各任务冻结测试为准。仅编译得到一个共享库而未提供可调用的 Python 绑定，无法完成精度验收。扩展必须在匹配的 Python/PyTorch ABI 下构建。安装 vllm_ascend 的 Python 源码也不代表目标 NPU 算子已完成编译和注册。

无需采用官方服务内部的构建方法，但必须执行提交的候选实现，保留冻结测试与采样协议，并返回 bridge 可以解析的结果。

## 4. 接入与验收

实现兼容的 HTTP 服务后：

```bash
export ASCEND_GYM_WORKSPACE=/path/to/AscendGYM
export ASCEND_GYM_DATA_ROOT="$ASCEND_GYM_WORKSPACE/datasets"
export SANDBOX_URL="http://<build-service-host>:<port>"
curl --fail "$SANDBOX_URL/ping"
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py smoke --op add_rms_norm --cases all --profiling-runs 5
```

该联调不需要模型端点。检查编译日志、每轮冻结测试结果、profiling 归档和解析出的正数耗时；现有 wrapper 中 profiler 退出码不一定等于测试子进程退出码，应结合完整测试输出和错误日志判断。先验证每个拟支持后端的代表任务，再对拟公布支持的任务逐一执行基线验收；单个任务通过不能代表全部任务可用。

保存硬件型号、驱动、CANN、Python、torch/torch_npu、公开依赖 commit、编译参数与采样协议，随结果记录环境。现有官方运行包的验证记录不能替代自建环境的独立验收。
