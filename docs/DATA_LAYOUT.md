# 数据根布局

默认使用仓库随包的 `datasets/`（见[数据集说明](../datasets/README.md)）。如需使用自己的外部数据目录，运行前设置：

```bash
export ASCEND_GYM_DATA_ROOT=/path/to/ascend_gym_data
```

期望布局：

```text
${ASCEND_GYM_DATA_ROOT}/
├── ops_src/       # 算子源码根（catalog.source_root）
└── packages/      # 算子包（catalog.outputs.inputs_dir）
```

运行产物不放在这里：`outputs.runs_dir` 使用 CWD 相对路径 `../runs`，
从 `bridge/` 执行时落到包根 `runs/`（即 `${ASCEND_GYM_WORKSPACE}/runs/`）。
