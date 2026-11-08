// ============================================================================
// Bundle snapshot generator.
//
// Reads the project's OWN dataset folder (`src/data/eval_datasets_summary/`) and
// writes `src/data/operator_bundle.json` — a checked-in snapshot of the same
// board the browser assembles at runtime.
//
// The browser is the primary consumer and does NOT need this file (see
// src/services/datasets.js). The snapshot exists for three things that cannot
// read a folder of ES modules:
//   • scripts/mock-api.mjs  — the `GET /api/bundle` dev stand-in
//   • scripts/audit-field-mapping.mjs — the field-mapping audit
//   • diffing — a committed artifact makes dataset changes reviewable
//
// Parsing goes through `src/lib/datasets.js`, the same module the browser uses,
// so this file can never describe a board the UI would not render.
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { applyPricing } from '../src/lib/pricing.js';
import { datasetsFromMap, DATASET_DIR, isDatasetFile } from '../src/lib/datasets.js';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(scriptDir, '..');
const datasetDir = path.join(workspaceRoot, ...DATASET_DIR.split('/'));
const outputPath = path.join(workspaceRoot, 'src', 'data', 'operator_bundle.json');

// Build-time price table. This is the SAME file the browser reads
// (src/data/Token_price.json), so the snapshot and the live UI cannot disagree
// on cost. Rows left null mean "no verifiable official price", and every cost
// derived from them stays null rather than being guessed.
const pricingPath = path.join(workspaceRoot, 'src', 'data', 'Token_price.json');

/** Read every dataset file in the folder, keyed by path (as the glob would). */
function readDatasetFolder() {
  if (!fs.existsSync(datasetDir)) {
    throw new Error(`数据集目录不存在：${path.relative(workspaceRoot, datasetDir)}`);
  }
  const map = {};
  for (const name of fs.readdirSync(datasetDir).sort()) {
    if (!isDatasetFile(name)) continue;
    const full = path.join(datasetDir, name);
    try {
      map[name] = JSON.parse(fs.readFileSync(full, 'utf8'));
    } catch (error) {
      // Surface the filename: a JSON typo in one file must not look like the
      // whole folder failed to parse.
      throw new Error(`${name} 不是合法 JSON：${error.message}`);
    }
  }
  return map;
}

function main() {
  const { bundle, entries, files, problems } = datasetsFromMap(readDatasetFolder());

  if (!entries.length) {
    throw new Error(`${DATASET_DIR} 中没有可用的数据集（至少需要一个含 runs 数组的 summary）`);
  }

  // Build-time pricing from src/data/Token_price.json. A model whose row is
  // still null simply has no cost — the snapshot records that gap instead of
  // filling it in.
  let pricingTable = null;
  let pricedCount = 0;
  if (fs.existsSync(pricingPath)) {
    const rawPricing = JSON.parse(fs.readFileSync(pricingPath, 'utf8'));
    pricingTable = rawPricing.models ? rawPricing : { ...rawPricing, models: {} };
    applyPricing(entries, pricingTable).forEach((priced, index) => {
      entries[index].overall = priced.overall;
      if (priced.overall.api_cost_usd != null) pricedCount += 1;
    });
  }

  fs.writeFileSync(outputPath, `${JSON.stringify(bundle, null, 2)}\n`, 'utf8');

  const meta = bundle.meta;
  console.log(`Operator bundle written to ${path.relative(workspaceRoot, outputPath)}`);
  console.log(`  entries=${meta.entries_evaluated} agents=${meta.agents.length} operators=${meta.operators.length} runs=${meta.runs_total} tokens=${Math.round(meta.tokens_total_k)}k`);
  console.log(`  pricing=${pricingTable
    ? `已为 ${pricedCount}/${entries.length} 个 entry 计算 api_cost_usd`
    : '未找到 src/data/Token_price.json（api_cost_usd = null）'}`);
  console.log(`  ${DATASET_DIR}：读取 ${files.length} 个文件`);
  files.forEach((file, index) => {
    const entry = entries[index];
    console.log(`   - ${file} → ${entry ? `${entry.agent.id} × ${entry.engine_model.id}` : '（未生成 entry）'}`);
  });
  problems.forEach(problem => console.log(`   ! ${problem}`));
}

main();
