# Perf × Cost 分析与 API 等价成本估算

核验日期：2026-10-26。价格单位为 USD / 100 万 token。

## A/B 是什么关系

A 的纵轴为 STS，横轴为累计 token 或估算 API 成本；越靠左上越好，Pareto 前沿表示无法同时提高 STS 并降低成本的配置。STS 自身已包含 token 延迟惩罚，因此它不是单独的最终 GM 加速比。

B 保留 STS 纵轴，将横轴替换为 PAOS（无量纲的场景相关 profiling 收益指标）。它是独立诊断视角，不是另一种 API 成本，也不是 A 的下一阶段。旧指标设计采用 PAOS = sqrt(PKAI × PTEI)，结合 recipe 加权算子收益与估算 step 级收益；PAOS=1 是基准参考。技术报告指出，默认单算子 recipe 下这些量会退化为内核加速比，不能据此声称真实工作负载吞吐提升。当前 21 个配置的 PAOS 均未导出，所以 B 没有可用散点。

## 官方单价

| 模型 | 输入 | 缓存输入 | 输出 | 来源 |
| --- | ---: | ---: | ---: | --- |
| Claude-Opus-5 | 5 | 0.5 | 25 | [官方页面](https://platform.claude.com/docs/en/about-claude/pricing) |
| GPT-5.6 Terra | 2 | 0.2 | 12 | [官方页面](https://developers.openai.com/api/docs/models/gpt-5.6-terra) |
| GLM-5.3 | 1.4 | 0.26 | 4.4 | [官方页面](https://docs.z.ai/guides/overview/pricing) |
| Qwen3.5-122B-A10B | 0.115 | 不支持/未公布 | 0.917 | [官方页面](https://www.alibabacloud.com/help/en/model-studio/qwen3-5-122b-a10b) |
| Qwen3.6-35B-A3B | 0.248 | 不支持/未公布 | 1.485 | [官方页面](https://www.alibabacloud.com/help/en/model-studio/qwen3-6-35b-a3b) |
| DeepSeek-V4-Flash | 0.3 | 0.006 | 1.2 | [官方页面](https://api-docs.deepseek.com/quick_start/pricing) |
| GPT-5.6 Luna | 0.2 | 0.02 | 1.2 | [官方页面](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |

- **Claude-Opus-5**：Standard API; 5-minute cache writes. 1-hour writes cost $10/1M and are not used in the scenario.
- **GPT-5.6 Terra**：Standard API, input <=272K per request; longer prompts are billed at 2x input and 1.5x output.
- **GLM-5.3**：Standard USD API pricing; cached-input storage is temporarily free.
- **Qwen3.5-122B-A10B**：China (Beijing), input <=128K; 128K–256K input/output rates are $0.287/$2.294. Context caching is unsupported.
- **Qwen3.6-35B-A3B**：China (Beijing). Context caching is unsupported. Singapore input/output rates are $0.375/$2.25.
- **DeepSeek-V4-Flash**：Peak rates. Original V4-Flash has retired; the legacy API name now serves V4.1-Flash. This is a successor-rate proxy, not the historical BF16 model bill. Off-peak rates are half.
- **GPT-5.6 Luna**：Standard API, input <=272K per request; longer prompts are billed at 2x input and 1.5x output.

## 估算方法

C = T / 1e6 × (0.9 × P_input + 0.1 × P_output)。T 使用当前导出的累计 token 总量，含失败尝试，不截断为 STS 的 50M 预算。所有配置统一假设 90% 输入、10% 输出、无缓存；这不是从日志恢复的实际用量比例。

CSV 另外提供 5% / 20% 输出占比和 80% 输入缓存命中场景；它们是敏感性情景，不是成本上下界或置信区间。Qwen 官方不支持缓存，因此缓存场景留空。DeepSeek 统一采用当前 Flash 后继版本的 peak 单价；off-peak 为一半，且原 V4-Flash/BF16 的价格不可据此确认为相同。

Claude 5 分钟缓存写入 $6.25/1M，1 小时 $10/1M；OpenAI 缓存写入按输入单价 1.25 倍。主情景未假设任何缓存读写。Qwen 使用北京区域，Qwen3.5 输入≤128K；OpenAI 输入≤272K。长上下文附加费、工具、税、订阅、加速卡与环境费用均未计入。

当前 UI bundle 把计费分类留空。上游 normalized_runs 虽有分项字段，但含估算和不同计数口径，例如部分 Claude 记录的 cached_input 大于 input；在完成分类口径核对前，不将它们简单相加或称为真实账单。T 本身保留补测源数据中的估算属性。没有把预估成本写回原始实验指标或改变 STS 排名。

## 配置估算（费用显示两位小数，CSV 保留计算精度）

| 配置 | Cohort | 累计 token（M） | API 等价估算（USD） |
| --- | --- | ---: | ---: |
| Claude Code × Claude-Opus-5 | harness | 1,389.784 | 9,728.49 |
| Claude Code × GPT-5.6 Terra | harness | 490.527 | 1,471.58 |
| Claude Code × GLM-5.3 | harness | 582.736 | 990.65 |
| Claude Code × Qwen3.5-122B-A10B | harness | 232.752 | 45.43 |
| Claude Code × Qwen3.6-35B-A3B | harness | 324.953 | 120.79 |
| Claude Code × DeepSeek-V4-Flash | harness | 820.864 | 320.14 |
| Codex × GPT-5.6 Terra | harness | 2,387.283 | 7,161.85 |
| Codex × GLM-5.3 | harness | 1,658.805 | 2,819.97 |
| Codex × Qwen3.5-122B-A10B | harness | 926.983 | 180.95 |
| Codex × Qwen3.6-35B-A3B | harness | 447.902 | 166.49 |
| Codex × DeepSeek-V4-Flash | harness | 1,060.282 | 413.51 |
| OpenCode × Claude-Opus-5 | harness | 247.159 | 1,730.11 |
| OpenCode × GPT-5.6 Terra | harness | 51.899 | 155.70 |
| OpenCode × GLM-5.3 | harness | 598.64 | 1,017.69 |
| OpenCode × Qwen3.5-122B-A10B | harness | 439.008 | 85.69 |
| OpenCode × Qwen3.6-35B-A3B | harness | 406.414 | 151.06 |
| OpenCode × DeepSeek-V4-Flash | harness | 219.337 | 85.54 |
| Default Cold × GPT-5.6 Luna | default | 6,488.778 | 1,946.63 |
| Default Warm × GPT-5.6 Luna | default | 970.82 | 291.25 |
| Default No Lessons × GPT-5.6 Luna | default | 1,881.229 | 564.37 |
| Default Code Only × GPT-5.6 Luna | default | 1,469.735 | 440.92 |

只可比较相同 cohort 和覆盖范围。Default 模式有重复运行，不应将其累计费用与单次 Harness 数据当成严格受控对照。

数据：src/data/supplement_bundle.json；价格：src/data/Token_price.json；公式：src/lib/pricing.js。运行 node scripts/export-api-cost-estimates.mjs 可重建 CSV 和本文。
