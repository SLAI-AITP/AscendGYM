import assert from 'node:assert/strict';
import fs from 'node:fs';
import { computeApiCost, estimateApiCost, applyPricing, priceRowFor } from '../src/lib/pricing.js';
const table = JSON.parse(fs.readFileSync(new URL('../src/data/Token_price.json', import.meta.url)));
const bundle = JSON.parse(fs.readFileSync(new URL('../src/data/supplement_bundle.json', import.meta.url)));
const terra = priceRowFor(table, 'gpt-5.6-terra');
assert.equal(estimateApiCost(1e6, terra, table.estimation), 3);
assert.equal(estimateApiCost(1e6, priceRowFor(table, 'claude-opus-5'), table.estimation), 7);
assert.equal(estimateApiCost(0, terra, table.estimation), 0);
assert.equal(estimateApiCost(-1, terra, table.estimation), null);
assert.equal(estimateApiCost(1e6, terra, { enabled: true, output_fraction: 2 }), null);
assert.equal(estimateApiCost(1e6, priceRowFor(table, 'Qwen3.5-122B'), { enabled: true, output_fraction: .1, cached_input_fraction: .8 }), null);
assert.equal(computeApiCost({ input: 1e6, cached_input: null, cache_write: 0, output: 1e6 }, terra), null);
assert.equal(computeApiCost({ input: 1e6, cached_input: 1e6, cache_write: 0, output: 1e6 }, terra), 14.2);
assert.equal(computeApiCost({ input: 1e6, cached_input: 1e6, cache_write: 0, output: 1e6 }, { input: 2, output: 12 }), null);
const priced = applyPricing(bundle.entries, table);
assert.equal(new Set(priced.map(e => e.engine_model.id)).size, 7);
for (const entry of priced) {
  assert.equal(entry.overall.api_cost_kind, 'scenario_estimate', entry.id);
  assert.ok(Number.isFinite(entry.overall.api_cost_usd) && entry.overall.api_cost_usd > 0, entry.id);
  assert.equal(entry.overall.sts_token, bundle.entries.find(e => e.id === entry.id).overall.sts_token);
  assert.ok(entry.overall.api_cost_price.source_url.startsWith('https://'));
}
assert.equal(priceRowFor(table, 'DeepSeek-V4-Flash-BF16').rate_basis, 'successor_proxy');
assert.equal(priceRowFor(table, 'unknown-model'), null);
console.log('PASS: 7 source-linked model rates, 21 explicit estimates, complete billing checks, invalid inputs, unchanged STS and successor provenance.');
