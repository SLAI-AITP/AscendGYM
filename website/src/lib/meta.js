// Static metric metadata, flag vocabulary, rubric columns and the metric
// catalog bridge — the single source of truth for display labels.

import { VENDORS, VENDOR_LIST } from './vendors';
import { PRECISION } from './precision';

// Metric keys mirror the evaluator's own field names wherever a value maps
// 1:1 onto it (`TaskPass`, `GateRate`, `mean_VLR`, `EBGS`, `BestMs`), so the
// dashboard and evaluate_operator_runs.py cannot drift apart.
export const metricMeta = {
  fast_1: { label: 'fast₁ @50M', short: 'FAST₁', unit: '%', direction: 'up', kind: 'measurement' },
  sts_token: { label: 'STS Token', short: 'STS', unit: '', direction: 'up', kind: 'measurement' },
  budget_gm: { label: 'GM @50M', short: 'S_EQUIV', unit: '×', direction: 'up', kind: 'ratioDetail' },
  ebgs: { label: 'GM Speedup', short: 'EBGS', unit: '×', direction: 'up', kind: 'ratioDetail' },
  speedup_mean: { label: 'Mean Speedup', short: 'S', unit: '×', direction: 'up', kind: 'ratioDetail' },
  vlr_mean: { label: 'VLR', short: 'VLR', unit: '%', direction: 'up', kind: 'measurement' },
  task_pass_rate: { label: 'Task Pass Rate', short: 'TASK %', unit: '%', direction: 'up', kind: 'measurement' },
  gate_rate: { label: 'Correctness Gate', short: 'GATE %', unit: '%', direction: 'up', kind: 'measurement' },
  candidate_pass_rate: { label: 'Candidate Pass Rate', short: 'CAND %', unit: '%', direction: 'up', kind: 'measurement' },
  compile_pass: { label: 'Compilation Pass Rate', short: 'COMPILE %', unit: '%', direction: 'up', kind: 'measurement' },
  best_duration_us: { label: 'Best Latency', short: 'BEST µs', unit: 'µs', direction: 'down', kind: 'measurement' },
  baseline_duration_ms: { label: 'Baseline Latency', short: 'L0', unit: 'ms', direction: 'down', kind: 'measurement' },
  tokens_total_k: { label: 'Cumulative Tokens', short: 'TOKENS', unit: 'k', direction: 'down', kind: 'count' },
  api_cost_usd: { label: 'Total API Cost', short: 'COST', unit: 'USD', direction: 'down', kind: 'measurement' },
  rounds_median: { label: 'LLM Rounds (median)', short: 'ROUNDS', unit: '', direction: 'down', kind: 'count' },
  wall_time_median_s: { label: 'Wall Time (median)', short: 'WALL', unit: 's', direction: 'down', kind: 'measurement' },
  tool_calls_total: { label: 'Tool Calls', short: 'TOOLS', unit: '', direction: 'down', kind: 'count' },
  profiling_runs_median: { label: 'Profiling Runs', short: 'PROFILE', unit: '', direction: 'down', kind: 'count' },
  tool_call_acc: { label: 'Tool-Call Accuracy', short: 'TOOL %', unit: '%', direction: 'up', kind: 'measurement' },
  // Value-axis score is nested on the entry, hence the dotted key.
  'paos.paos': { label: 'PAOS', short: 'PAOS', unit: '×', direction: 'up', kind: 'ratioDetail' },
  'paos.pkai': { label: 'PKAI', short: 'PKAI', unit: '×', direction: 'up', kind: 'ratioDetail' },
  'paos.ptei': { label: 'PTEI', short: 'PTEI', unit: '×', direction: 'up', kind: 'ratioDetail' }
};

/**
 * Stamp `decimals` from the shared precision policy.
 *
 * Each metric declares its *kind*; the decimal count is derived, never hand
 * typed. That is what stops `best_duration_us` from rendering one decimal in
 * the table while another readout of the same value renders two.
 */
for (const meta of Object.values(metricMeta)) {
  meta.decimals = PRECISION[meta.kind] ?? PRECISION.measurement;
}

/** Leaderboard "Re-rank" options, ordered performance → reliability → cost. */
export const rankMetricOptions = [
  'sts_token',
  'budget_gm',
  'ebgs',
  'vlr_mean',
  'task_pass_rate',
  'tokens_total_k',
  'rounds_median',
  'wall_time_median_s'
];

// Process heatmap: the three customer-facing effort dimensions (§2.1 #operators).
// `unit` is only spelled out for bare-integer metrics — the token / wall-time
// formatters already embed their own magnitude, so a cell reads "283K" and
// "2918.5s" rather than "283K k tokens".
export const processMeta = {
  tokens_k: { label: 'Cumulative Tokens', unit: 'k tokens' },
  rounds: { label: 'LLM Rounds (mean)', unit: 'rounds' },
  wall_time_s: { label: 'Wall Time (mean)', unit: 's' }
};

// C7 — collectible utilization channels (op_summary_*.csv direct reads).
// l2_bw / hbm_bw / overlap stay null (not_collected) and are NOT rendered.
export const utilizationChannels = [
  { key: 'vec_ratio', label: 'Vec Ratio', source: 'aiv_vec_ratio' },
  { key: 'mac_ratio', label: 'MAC Ratio', source: 'aic_mac_ratio' },
  { key: 'scalar_ratio', label: 'Scalar Ratio', source: 'aic_scalar_ratio' },
  { key: 'mte2_ratio', label: 'MTE2 Ratio', source: 'aic_mte2_ratio' }
];

// Vendor marks/colours now live in `lib/vendors.js` (the icon registry).
// This map is kept as a label→filename compatibility shim for callers that
// still look up an asset by the vendor display label.
export const vendorAssetMap = Object.fromEntries(
  VENDOR_LIST.filter(vendor => vendor.asset).map(vendor => [vendor.label, vendor.asset])
);

/** Vendors that ship a brand mark, plus their accent colour and monogram. */
export const vendorOptions = VENDOR_LIST.map(({ id, label, asset, glyph, color }) => ({ id, label, asset, glyph, color }));

export { VENDORS };

/** §4.4 / §7.2 — flag vocabulary: icon, severity and the rule text shown on hover. */
export const flagMeta = {
  fake_optimization: { icon: '🟥', level: 'red', label: 'Unconfirmed optimization', rule: 'Representative speedup conflicts with per-shape evidence; inspect the underlying measurements.' },
  shape_regression: { icon: '🟥', level: 'red', label: 'Shape regression', rule: 'At least one shape has latency speedup below 0.9' },
  unconverged: { icon: '🟧', level: 'amber', label: 'Not converged', rule: 'The cycle budget was exhausted before convergence.' },
  high_waste: { icon: '🟧', level: 'amber', label: 'High wasted-round fraction', rule: 'More than 50% of rounds failed compilation or correctness checks' },
  plateau: { icon: '🟦', level: 'blue', label: 'Plateau', rule: 'S(T_max) − S(T_mid) < ε; late consumption produced no material improvement' },
  baseline_abnormal: { icon: '⬜', level: 'gray', label: 'Invalid baseline', rule: 'Invalid baseline; excluded from cross-operator aggregation' }
};

// §5.2 — v2 rubric columns (directions updated; context columns render neutral).
export const rubricColumns = [
  { key: 'candidate_pass_rate', label: 'Candidate Pass Rate', definition: 'Correct candidates divided by evaluated candidates; operator-balanced aggregation.', context: false },
  { key: 'fast_1', label: 'fast₁ @50M', definition: 'Fraction of tasks with correctness-validated speedup strictly greater than 1 within 50M tokens; available trajectories only.', context: false },
  { key: 'sts_token', label: 'STS Token', definition: '100 log₂ S_n − 50 Σ (B_k/B_max) Δg_k; equally weighted operators, 50M tokens per task.', context: false },
  { key: 'task_pass_rate', label: 'Task Pass Rate', definition: 'Attempted tasks with a valid baseline AND at least one correct search candidate, divided by all attempted tasks.', context: false },
  { key: 'vlr_mean', label: 'VLR', definition: 'Mean latency reduction: 100 × (1 − 1/S), using the 50M-token incumbent.', context: false },
  { key: 'ebgs', label: 'GM Speedup', definition: 'Geometric mean speedup over terminal selected trajectories, following the supplementary figure tables.', context: false },
  { key: 'tokens_total_k', label: 'Cumulative Tokens', definition: 'Cumulative tokens consumed by selected logical runs, including failed attempts.', context: false },
  { key: 'rounds_median', label: 'LLM Rounds (median)', definition: 'Median LLM interaction rounds over selected runs.', context: false },
  { key: 'wall_time_median_s', label: 'Wall Time (median)', definition: 'Median elapsed wall-clock time over selected runs.', context: false },
  { key: 'profiling_runs_median', label: 'Profiling Runs', definition: 'Median profiling run count; unavailable when not exported.', context: false },
  { key: 'vec_ratio_best', label: 'Vec Ratio', definition: 'Mean AI vector activity ratio; diagnostic context only.', context: true },
  { key: 'u_best', label: 'AI Core Utilization', definition: 'Mean active-resource time ratio; diagnostic context only.', context: true }
];

// §9 glossary moved out of the dashboard: the term list is documentation and
// now lives in README.md §5 "术语表". Only the compact flag vocabulary and the
// agent short marks stay here, because they are rendered, not read.

// Agent short marks used by the core field and compact labels.
export const agentShort = {
  'world-model': 'WM',
  'k-search': 'KS',
  'claude-code': 'CC',
  'openhands': 'OH'
};

export function agentInitials(agentId) {
  if (agentShort[agentId]) return agentShort[agentId];
  return String(agentId).split(/[-_]/).map(part => part[0]?.toUpperCase() ?? '').join('').slice(0, 2);
}

// ---------------------------------------------------------------------------
// Metric catalog bridge.
//
// The evaluator owns metric *semantics* and now ships a `catalog` block with
// friendly labels/definitions. When a bundle carries it we overlay those on
// the built-in display labels, so the dashboard cannot drift from the backend
// contract. The static table above stays as an offline fallback.
// ---------------------------------------------------------------------------
export const metricCatalog = {
  categories: [],
  definitions: {},
  version: null,
  loaded: false
};

/**
 * Place a metric the static table has never seen into a precision kind.
 *
 * The evaluator's catalog can introduce keys at any time, and defaulting them
 * all to `measurement` would render `Rounds` as `42.00` — false precision on a
 * count. The unit is the only signal available, and it is enough: a ratio
 * carries `×`, anything else with a unit is a measurement, and a bare number
 * from the evaluator is a count (rounds / calls / runs).
 */
function kindForCatalogEntry(unit) {
  if (unit === '×') return 'ratioDetail';
  return unit ? 'measurement' : 'count';
}

export function applyMetricCatalog(catalog) {
  if (!catalog?.metrics) return;
  metricCatalog.categories = catalog.categories ?? [];
  metricCatalog.version = catalog.version ?? null;
  metricCatalog.loaded = true;
  Object.entries(catalog.metrics).forEach(([key, entry]) => {
    const existing = metricMeta[key] ?? {};
    const kind = existing.kind ?? kindForCatalogEntry(entry.unit ?? existing.unit);
    metricMeta[key] = {
      ...existing,
      label: entry.label ?? existing.label ?? key,
      short: entry.short ?? existing.short,
      unit: entry.unit ?? existing.unit ?? '',
      direction: entry.direction ?? existing.direction ?? 'up',
      kind,
      decimals: PRECISION[kind] ?? PRECISION.measurement,
      category: entry.category
    };
    metricCatalog.definitions[key] = {
      label: entry.label ?? existing.label ?? key,
      short: entry.short ?? '',
      unit: entry.unit ?? '',
      direction: entry.direction ?? 'up',
      category: entry.category ?? 'other',
      definition: entry.definition ?? ''
    };
  });
}

/** Catalog rows grouped by category, in the evaluator's declared order. */
export function catalogByCategory() {
  const order = metricCatalog.categories.length
    ? metricCatalog.categories
    : [...new Set(Object.values(metricCatalog.definitions).map(item => item.category))]
        .map(key => ({ key, label: key, subtitle: '' }));
  return order
    .map(category => ({
      ...category,
      rows: Object.entries(metricCatalog.definitions)
        .filter(([, item]) => item.category === category.key)
        .map(([key, item]) => ({ key, ...item }))
    }))
    .filter(group => group.rows.length);
}
