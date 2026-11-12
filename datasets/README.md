# AscendGYM 独立评测集

本目录与 AscendForge 服务代码、四个单独获取的上游构建仓库分开存放。
ops_src/ 只包含当前评测引用的算子基线快照和必须保留的 golden 任务，不是公开算子仓库的完整副本。

~~~text
datasets/
├── registry.json
├── tasks/
│   ├── batch1/<task_id>/{README.md,task.json}
│   └── preserved-golden/<task_id>/{README.md,task.json}
├── ops_src/<backend>/<bucket>/kernels/<op>/  # 基线、冻结测试、golden
├── SOURCE_MANIFEST.json
└── packages/                               # 本地生成，Git 忽略
~~~

- batch1 对应 bridge/configs/op_catalog.json 当前的 17 个正式任务，未增删其列表。
- preserved-golden 保存另外 3 个已带 golden 的算子，不自动加入正式评测。
- 从原两套 example 目录核对的 488 个张量路径，完整对应到本目录的 384 个张量文件；104 个是两套目录间的重复副本。
- golden、冻结测试和 manifest 不交给优化 agent 修改；允许修改的源码前缀记录在任务描述中。
- 任务基线与 golden 随本目录提供。服务端按需准备构建依赖；公开版本清单见 [environment/public-dependencies.json](../environment/public-dependencies.json)，接入方式见[自建环境说明](../docs/SELF_HOSTED_ENVIRONMENT.md)。客户端无需检出服务源码或完整构建仓库。
- SOURCE_MANIFEST.json 固定当前快照哈希；许可证和来源声明保存在 LICENSES/。

在工作区根目录选择随包数据：

~~~bash
source datasets/activate.sh
cd bridge
python3 op_manager.py scan
python3 op_manager.py build --all
~~~

也可设置 ASCEND_GYM_DATA_ROOT 指向用户自己的独立数据目录，search_roots 的结构保持不变。
单任务回归建议使用公开 bridge 的无模型验收入口：

```bash
cd "$ASCEND_GYM_WORKSPACE/bridge"
python3 op_manager.py smoke --op add_rms_norm --cases all --profiling-runs 5
```

先设置 `SANDBOX_URL`，见[服务协议](../docs/EXECUTION_BACKEND.md)。该命令编译源码，执行全部冻结用例并采集 profiling，不重新生成 golden。`datasets/run_task.py` 是需要预装 `sandbox-client` 的可选客户端入口；根 requirements 不提供该命令，没有该客户端时使用上述 bridge 入口。

## 冻结测试脚本的 `--runs` 契约

bare（k-lens-v2）的 profiling 命令是 `msprof --application="python3 <test>.py --runs N"`，
脚本必须自己解析 `--runs` 并把整套用例重复 N 次。bare 的行数整除检查只是数据结构检查，不能证明测试已经执行完整轮次。native 同样传递 `--runs`，但其聚合器不强制同一整除关系；两条路径都需要核对测试的实际执行轮次。

当前正式任务的冻结测试均提供 `--runs` 入口，未指定时默认执行一轮。新增测试时应遵循同一约定，保持原有用例、断言、容差与 golden 不变。

`SOURCE_MANIFEST.json` 记录当前文件哈希和任务文件统计。更新测试脚本或算子文档后应同步维护该清单；算子实现、冻结张量和 golden 的变更须单独审核。

新增或替换算子测试脚本的完整要求见
[docs/OPERATOR_TEST_GUIDE.md](../docs/OPERATOR_TEST_GUIDE.md)。

## 来源与许可

本目录中项目自行编写且未另行声明许可的测试脚本、任务定义及文档采用 [Apache License 2.0](../LICENSE)。第三方源码、移植测试及其派生内容继续遵循原有许可；保留文件中的版权和许可声明。

- `ops_src/ops_nn/`：算子基线及对应上游材料适用 [ops-nn 原许可](LICENSES/ops-nn/LICENSE) 与相应第三方通知。
- `ops_src/ops_transformer/`：算子基线及对应上游材料适用 [ops-transformer 原许可](LICENSES/ops-transformer/LICENSE) 与相应第三方通知。
- `ops_src/vllm_ascend/`：保留 [vllm-ascend 原许可](LICENSES/vllm-ascend/LICENSE) 及已有第三方声明。

冻结输入张量和 golden 是正确性检查所需的评测材料，不是模型优化的历史成绩。其生成入口为各算子目录中的采集或测试脚本；仅对项目有权授权的自行生成内容适用 Apache-2.0。来源于第三方或基于其受许可材料派生的内容仍受原条款约束，不因转换为张量而重新许可。固定版本和文件哈希由 SOURCE_MANIFEST.json 记录，不构成第三方授权证明。
