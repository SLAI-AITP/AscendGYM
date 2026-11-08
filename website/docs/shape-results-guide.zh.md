# Shape 数据填写模板

这次只填写一个 CSV，不需要之前的多层 JSON、协议块或候选信息。

## 使用方法

1. 打开 `shape-results.csv`。已预填当前 21 个配置、353 个“配置×算子”的 entry_id、operator 和 run_id，与当前图表对应。
2. **每行填写一个 shape / case。** 同一算子有多少个 case，就复制该行多少次，保留前三列不变，逐行填写 case_id、shape、dtype。
3. 有逐 shape 性能数据时，再填两列耗时和 correctness；若暂时只采集 shape，耗时留空，correctness 保持 not_measured。
4. 只采集部分算子也可以：提交已填写的行即可，其他占位行可以删除。保持 CSV UTF-8 编码，不要合并单元格或改动表头。

也提供只有表头的 `shape-results.blank.csv`。`shape-results.example.csv` 只展示格式，里面所有尺寸、耗时与标识均为演示，不能作为真实结果提交。

## 九列分别填什么

| 列名 | 填写方式 |
| --- | --- |
| entry_id | 已预填，保持不变；它标识 Harness×Model 配置，不能换成页面展示名。 |
| operator | 已预填，保持不变。 |
| run_id | 已预填，保持不变；对应当前图表使用的那个 run。若实际采集其他 run，请填写真实 run_id 并说明。 |
| case_id | 环境中的 case ID；同一个配置、算子、run 内不能重复。若环境没有 ID，可依 manifest 顺序固定编号 case_001、case_002 等。 |
| shape | 具体输入尺寸。单输入例如 x=[8,4096]；多输入例如 x=[8,4096]; weight=[4096]。所有输入都列出；标量使用 []。case 若由额外参数区分，可追加 axis=-1; epsilon=1e-5 等。 |
| dtype | 单输入例如 float16；多输入例如 x=bfloat16; weight=float16。使用环境实际 dtype。 |
| t_base_us | 可选，当前 case 的基线耗时，纯数字，单位 µs。 |
| t_best_us | 可选，当前 case 在选定候选内核上的耗时，纯数字，单位 µs。 |
| correctness | pass / fail / error / not_measured。没有测正确性时保留 not_measured，不能填 pass。 |

例：

```text
shape: x=[8,4096]; weight=[4096]
dtype: x=float16; weight=float16
```

CSV 中尺寸包含逗号，使用 Excel / Numbers 正常保存 CSV 即可自动加引号。若用脚本导出，请用 Python csv.DictWriter，不要手动用逗号拼接字符串。

## 注意

- 耗时未知就留空，不填 0，不写“100us”等带单位文本；ms 转 µs 乘 1000。
- 同一个 run 的不同 shape 应使用同一个选定候选内核，不要逐 shape 换成不同候选的最好值。
- 失败、报错和未测 case 也保留对应行，以便展示完整 shape 分布。
- 无需手填加速比：仅在 correctness=pass 且两项耗时有效时，页面计算 t_base_us / t_best_us。
- 每个算子的 case 数来自环境的完整 case 清单，不是全站统计中的 951。

完成后把填写好的 CSV 发回即可接入现在的行内展开列表。当前 Imports 页面尚不直接支持此 CSV；无需将其转换为旧版 summary JSON。
