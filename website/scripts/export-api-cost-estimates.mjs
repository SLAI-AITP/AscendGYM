import fs from 'node:fs';
import { applyPricing, estimateApiCost, priceRowFor } from '../src/lib/pricing.js';
import { configurationDisplayName, modelDisplayName } from '../src/lib/displayNames.js';
const root = new URL('../', import.meta.url);
const table = JSON.parse(fs.readFileSync(new URL('src/data/Token_price.json', root)));
const bundle = JSON.parse(fs.readFileSync(new URL('src/data/supplement_bundle.json', root)));
const entries = applyPricing(bundle.entries, table);
const columns = ['entry_id', 'configuration', 'cohort', 'tokens', 'sts', 'estimate_usd_10pct_output_no_cache', 'scenario_usd_5pct_output_no_cache', 'scenario_usd_20pct_output_no_cache', 'scenario_usd_10pct_output_80pct_input_cache', 'rate_basis', 'official_source', 'checked'];
const quote = x => `"${String(x ?? '').replaceAll('"', '""')}"`;
const csv = [columns.join(',')];
for (const e of entries) {
  const row = priceRowFor(table, e.engine_model.id), tokens = e.overall.tokens_total_k * 1000;
  const scenario = (output, cached = 0) => estimateApiCost(tokens, row, { enabled: true, output_fraction: output, cached_input_fraction: cached });
  csv.push([e.id, configurationDisplayName(e), e.cohort, tokens, e.overall.sts_token, e.overall.api_cost_usd, scenario(.05), scenario(.2), scenario(.1, .8), row.rate_basis, row.source_url, row.updated].map(quote).join(','));
}
fs.mkdirSync(new URL('docs/', root), { recursive: true });
fs.writeFileSync(new URL('docs/api-cost-estimates-2026-10-26.csv', root), '\uFEFF' + csv.join('\n') + '\n');
const models = [...new Set(entries.map(e => e.engine_model.id))].map(id => ({ id, ...priceRowFor(table, id) }));
const rates = models.map(r => `| ${modelDisplayName(r.id)} | ${r.input} | ${r.cached_input ?? '不支持/未公布'} | ${r.output} | [官方页面](${r.source_url}) |`).join('\n');
const usd = value => new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
const millions = value => new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 }).format(value);
const results = entries.map(e => `| ${configurationDisplayName(e)} | ${e.cohort} | ${millions(e.overall.tokens_total_k / 1000)} | ${usd(e.overall.api_cost_usd)} |`).join('\n');
fs.writeFileSync(new URL('docs/perf-cost-analysis-2026-10-26.md', root), `# Perf × Cost 分析与 API 等价成本估算\n\n核验日期：2026-10-26。价格单位为 USD / 100 万 token。\n\n## A/B 是什么关系\n\nA 的纵轴为 STS，横轴为累计 token 或估算 API 成本；越靠左上越好，Pareto 前沿表示无法同时提高 STS 并降低成本的配置。STS 自身已包含 token 延迟惩罚，因此它不是单独的最终 GM 加速比。\n\nB 保留 STS 纵轴，将横轴替换为 PAOS（无量纲的场景相关 profiling 收益指标）。它是独立诊断视角，不是另一种 API 成本，也不是 A 的下一阶段。旧指标设计采用 PAOS = sqrt(PKAI × PTEI)，结合 recipe 加权算子收益与估算 step 级收益；PAOS=1 是基准参考。技术报告指出，默认单算子 recipe 下这些量会退化为内核加速比，不能据此声称真实工作负载吞吐提升。当前 21 个配置的 PAOS 均未导出，所以 B 没有可用散点。\n\n## 官方单价\n\n| 模型 | 输入 | 缓存输入 | 输出 | 来源 |\n| --- | ---: | ---: | ---: | --- |\n${rates}\n\n${models.map(r => '- **' + modelDisplayName(r.id) + '**：' + r.note).join('\n')}\n\n## 估算方法\n\nC = T / 1e6 × (0.9 × P_input + 0.1 × P_output)。T 使用当前导出的累计 token 总量，含失败尝试，不截断为 STS 的 50M 预算。所有配置统一假设 90% 输入、10% 输出、无缓存；这不是从日志恢复的实际用量比例。\n\nCSV 另外提供 5% / 20% 输出占比和 80% 输入缓存命中场景；它们是敏感性情景，不是成本上下界或置信区间。Qwen 官方不支持缓存，因此缓存场景留空。DeepSeek 统一采用当前 Flash 后继版本的 peak 单价；off-peak 为一半，且原 V4-Flash/BF16 的价格不可据此确认为相同。\n\nClaude 5 分钟缓存写入 $6.25/1M，1 小时 $10/1M；OpenAI 缓存写入按输入单价 1.25 倍。主情景未假设任何缓存读写。Qwen 使用北京区域，Qwen3.5 输入≤128K；OpenAI 输入≤272K。长上下文附加费、工具、税、订阅、加速卡与环境费用均未计入。\n\n当前 UI bundle 把计费分类留空。上游 normalized_runs 虽有分项字段，但含估算和不同计数口径，例如部分 Claude 记录的 cached_input 大于 input；在完成分类口径核对前，不将它们简单相加或称为真实账单。T 本身保留补测源数据中的估算属性。没有把预估成本写回原始实验指标或改变 STS 排名。\n\n## 配置估算（费用显示两位小数，CSV 保留计算精度）\n\n| 配置 | Cohort | 累计 token（M） | API 等价估算（USD） |\n| --- | --- | ---: | ---: |\n${results}\n\n只可比较相同 cohort 和覆盖范围。Default 模式有重复运行，不应将其累计费用与单次 Harness 数据当成严格受控对照。\n\n数据：src/data/supplement_bundle.json；价格：src/data/Token_price.json；公式：src/lib/pricing.js。运行 node scripts/export-api-cost-estimates.mjs 可重建 CSV 和本文。\n`);
console.log(`Exported ${entries.length} cost estimates and ${models.length} official price records.`);
