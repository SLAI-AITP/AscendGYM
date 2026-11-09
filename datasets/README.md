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
- 上游构建依赖按根目录 README 和 AscendForge/PUBLIC_DEPENDENCIES.json 的固定版本单独取得；本仓库不登记 .gitmodules 或 gitlink。不会向这些上游目录写入本评测集的 golden 或测试脚本。
- SOURCE_MANIFEST.json 固定当前快照哈希；许可证和来源声明保存在 LICENSES/。

在工作区根目录选择随包数据：

~~~bash
source datasets/activate.sh
cd bridge
python3 op_manager.py scan
python3 op_manager.py build --all
~~~

也可设置 ASCEND_GYM_DATA_ROOT 指向用户自己的独立数据目录，search_roots 的结构保持不变。
单任务直接通过已经运行的统一代理回归（不调用模型）：

~~~bash
python3 datasets/run_task.py add_rms_norm --host 127.0.0.1 --port 9000
~~~

先加 --dry-run 仅查看命令。该命令要求环境已安装 sandbox-client，且服务及其上游依赖已经准备好。
它只运行冻结测试，不重新生成 golden。

旧 AscendForge/example 和 sandbox/client/example 中的四个大批量上游副本已移除，
旧路径不再作为任务入口。不要重新运行全库复制工具来恢复这些副本。

## 冻结测试脚本的 `--runs` 契约（2026-11-04）

bare（k-lens-v2）的 profiling 命令是 `msprof --application="python3 <test>.py --runs N"`，
脚本必须自己解析 `--runs` 并把整套用例重复 N 次；否则 Backend 的
`total_rows % total_loops == 0` 检查不过，baseline 阶段直接失败（native 不受影响，
所以缺 `--runs` 的包只在 bare 下暴露问题）。

2026-11-04 之前，本快照里的 17 个冻结测试都只跑一遍。现已全部补齐 `--runs`：

- 16 个脚本直接用改造后版本整文件替换，`diff` 只有一个 hunk（新增 `_run_suite` + argparse），
  其余字节完全一致；
- `add_rms_norm` 是特例：源仓库那份 `*_custom_test.py` 属于更早一代脚本（结构不同、
  带硬编码绝对路径、`DISABLE_L2_CACHE` 等），不能整文件覆盖，只在现有
  `add_rms_norm_test.py` 末尾追加了同样的 `_run_suite` 收尾块。

`SOURCE_MANIFEST.json` 记录当前文件哈希和任务文件统计。更新测试脚本或算子文档后应同步维护该清单；算子实现、冻结张量和 golden 的变更须单独审核。

新增或替换算子测试脚本的完整要求见
[docs/OPERATOR_TEST_GUIDE.md](../docs/OPERATOR_TEST_GUIDE.md)。

## 来源与许可

本目录中项目自行编写且未另行声明许可的测试脚本、任务定义及文档采用 [Apache License 2.0](../AscendForge/LICENSE)。第三方源码、移植测试及其派生内容继续遵循原有许可；保留文件中的版权和许可声明。

- `ops_src/ops_nn/`：算子基线及对应上游材料适用 [ops-nn 原许可](LICENSES/ops-nn/LICENSE) 与相应第三方通知。
- `ops_src/ops_transformer/`：算子基线及对应上游材料适用 [ops-transformer 原许可](LICENSES/ops-transformer/LICENSE) 与相应第三方通知。
- `ops_src/vllm_ascend/`：保留 [vllm-ascend 原许可](LICENSES/vllm-ascend/LICENSE) 及已有第三方声明。

冻结输入张量和 golden 是正确性检查所需的评测材料，不是模型优化的历史成绩。其生成入口为各算子目录中的采集或测试脚本；仅对项目有权授权的自行生成内容适用 Apache-2.0。来源于第三方或基于其受许可材料派生的内容仍受原条款约束，不因转换为张量而重新许可。固定版本和文件哈希由 SOURCE_MANIFEST.json 记录，不构成第三方授权证明。

