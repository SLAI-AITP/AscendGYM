// ============================================================================
// Model API pricing.
//
// The dashboard never invents a price. The table lives in a shipped file
// (`src/data/Token_price.json`) rather than in a visitor's browser, so every
// reader of a given commit sees the same cost figures. A model whose row has no
// official price stays `null` and the UI shows "–".
//
// Table schema (`ascendgym-pricing-v1`):
//   {
//     "currency": "USD",
//     "unit": "per_1M_tokens",
//     "models": {
//       "<engine_model.id>": {
//         "input":      <USD per 1M uncached input tokens>,
//         "cached_input": <USD per 1M cached input tokens>,   // optional
//         "cache_write":  <USD per 1M cache-write tokens>,    // optional
//         "output":     <USD per 1M output tokens>,
//         "vendor": "<optional label>",
//         "source": "<where the price came from>",
//         "updated": "YYYY-MM-DD"
//       }
//     }
//   }
//
// This module only READS a table: lookup, cost arithmetic, and the per-entry
// fill-in used by the benchmark store. There is no writer, because there is no
// import path — changing a price means editing the JSON file.
// ============================================================================

/** Cost formula inputs, in token units (not thousands). */
export const TOKEN_FIELDS = ['input', 'cached_input', 'cache_write', 'output'];

/** One model row is usable when at least one price is a real number. */
export function isRowPriced(row) {
  if (!row) return false;
  return TOKEN_FIELDS.some(field => Number.isFinite(row[field]));
}

/** Case/space-tolerant lookup: exact → case-insensitive → unique vendor match. */
export function priceRowFor(table, modelId) {
  const models = table?.models;
  if (!models || modelId == null) return null;
  const key = String(modelId);
  if (models[key] && isRowPriced(models[key])) return models[key];
  const lowered = key.toLowerCase();
  const direct = Object.entries(models)
    .find(([id, row]) => id.toLowerCase() === lowered && isRowPriced(row));
  return direct ? direct[1] : null;
}

/** `{input, cached_input, cache_write, output}` token counts → USD, or null. */
export function computeApiCost(usage, priceRow) {
  if (!priceRow || !usage) return null;
  // Never present a partial sum as the price of a complete run. All components
  // must be explicit, and every nonzero component needs a known rate.
  if (!TOKEN_FIELDS.every(field => Number.isFinite(usage[field]) && usage[field] >= 0
    && (usage[field] === 0 || (Number.isFinite(priceRow[field]) && priceRow[field] >= 0)))) return null;
  const perMillion = 1e6;
  let total = 0;
  let priced = false;
  // Components are disjoint billing categories. Missing prices were rejected
  // above unless the corresponding component explicitly has zero tokens.
  const parts = [
    [usage.input, priceRow.input],
    [usage.cached_input, priceRow.cached_input],
    [usage.cache_write, priceRow.cache_write],
    [usage.output, priceRow.output]
  ];
  for (const [tokens, price] of parts) {
    if (!Number.isFinite(tokens) || tokens <= 0 || !Number.isFinite(price)) continue;
    total += (tokens / perMillion) * price;
    priced = true;
  }
  return priced || TOKEN_FIELDS.every(field => usage[field] === 0) ? total : null;
}

/** Common mix scenario for incomplete billing exports; never a measured bill. */
export function estimateApiCost(totalTokens, priceRow, assumption) {
  if (!assumption?.enabled || !Number.isFinite(totalTokens) || totalTokens < 0
    || !Number.isFinite(priceRow?.input) || priceRow.input < 0
    || !Number.isFinite(priceRow?.output) || priceRow.output < 0) return null;
  const fraction = assumption.output_fraction;
  const cachedFraction = assumption.cached_input_fraction ?? 0;
  if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1
    || !Number.isFinite(cachedFraction) || cachedFraction < 0 || cachedFraction > 1
    || (cachedFraction > 0 && !Number.isFinite(priceRow.cached_input))) return null;
  const inputRate = cachedFraction === 0 ? priceRow.input
    : (1 - cachedFraction) * priceRow.input + cachedFraction * priceRow.cached_input;
  return totalTokens / 1e6 * ((1 - fraction) * inputRate + fraction * priceRow.output);
}

/** Token usage in the shape `computeApiCost` expects, from a cost block. */
export function usageFromCost(cost) {
  if (!cost) return null;
  return {
    input: cost.input_tokens ?? null,
    cached_input: cost.cached_input_tokens ?? null,
    cache_write: cost.cache_write_tokens ?? null,
    output: cost.output_tokens ?? null
  };
}

/**
 * Apply a price table to entries in place-ish fashion: returns new entry
 * objects with `overall.api_cost_usd` (+ reason) filled in. Entries whose
 * model has no price row keep `null` and record why.
 */
export function applyPricing(entries, table) {
  return (entries ?? []).map(entry => {
    const row = priceRowFor(table, entry.engine_model?.id);
    if (!row) {
      return {
        ...entry,
        overall: {
          ...entry.overall,
          api_cost_usd: null,
          api_cost_usd_reason: 'pricing_row_missing',
          api_cost_kind: null,
          api_cost_assumption: null,
          api_cost_price: null
        }
      };
    }
    const usage = entry.overall?.cost?.token_usage ?? usageFromCost(entry.overall?.cost);
    const repriced = computeApiCost(usage, row);
    const estimate = repriced == null ? estimateApiCost(entry.overall?.tokens_total_k == null ? null : entry.overall.tokens_total_k * 1000, row, table.estimation) : null;
    const total = repriced ?? estimate;
    const kind = repriced != null ? 'repriced_usage' : estimate != null ? 'scenario_estimate' : null;
    return {
      ...entry,
      overall: {
        ...entry.overall,
        api_cost_usd: total,
        api_cost_usd_reason: total == null ? 'token_usage_incomplete' : kind,
        api_cost_kind: kind,
        api_cost_assumption: kind === 'scenario_estimate' ? { ...table.estimation } : null,
        api_cost_price: {
          input: row.input,
          cached_input: row.cached_input,
          cache_write: row.cache_write,
          output: row.output,
          source: row.source,
          source_url: row.source_url,
          source_model: row.source_model,
          rate_basis: row.rate_basis,
          note: row.note,
          updated: row.updated,
          currency: table.currency ?? 'USD'
        },
        cost: {
          ...entry.overall?.cost,
          token_usage: usage,
          api_cost_usd: total
        }
      }
    };
  });
}
