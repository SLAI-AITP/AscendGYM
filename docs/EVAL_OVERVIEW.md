# 评测栈

本目录包含两个 harness 的实现本体。评测的启动、调度与结果归一化由上一级的 `bridge/` 负责。

| harness | 目录 | 说明 |
|---|---|---|
| native | [native_harness/README.md](../eval/native_harness/README.md) | 原生 coding agent 自主循环 |
| bare | [bare_harness/README.md](../eval/bare_harness/README.md) | 裸模型 + 工具池 |

通用约定：

- 代码路径使用 `${ASCEND_GYM_WORKSPACE}/eval/...`。
- 数据路径使用 `${ASCEND_GYM_DATA_ROOT}/...`。
- 运行产物由 bridge 写入 `${ASCEND_GYM_WORKSPACE}/runs/`（CWD 相对路径 `../runs`，从 `bridge/` 执行）。

安装、数据准备与模型配置见 [README](../README.md#installation)；首次运行见 [README：快速开始](../README.md#quick-start)。
