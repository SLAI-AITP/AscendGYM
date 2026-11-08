// ============================================================================
// STS / PAOS metric engine — v2.0 requirements §4 (shared by the frontend,
// the static-bundle generator and the anchor-verification script, so that
// displayed numbers, generated mock data and the Appendix-A verification all
// run through exactly the same formulas).
//
// Symbols (per §4): solution j; L0 baseline representative latency;
// t_j representative latency of solution j; T_j cumulative tokens; W_j wall s.
// The frontend only *consumes* pre-computed fields from the backend bundle;
// these functions exist so the static fallback bundle and the anchor tests
// are provably consistent with the published contract.
// ============================================================================
//
// Imports `precision.js` (a leaf, no dependencies) rather than `format.js`:
// this module is loaded by `scripts/verify-anchors.mjs` with plain Node, and
// `format.js` pulls in `meta.js` -> `vendors.js` through extensionless imports
// that bare Node cannot resolve. The extension below is REQUIRED for the same
// reason — bare Node does not do Vite's extension guessing.
import { PRECISION } from './precision.js';

export const EPS_DEFAULT = 0.01; // metrics.epsilon_used fallback (1% significance)

/** Default VLR tiers, mirroring the evaluator's DEFAULT_TARGETS. */
export const DEFAULT_TARGETS = [1, 5, 10];

const log2 = value => Math.log(value) / Math.LN2;

// ------------------------------- §4.1 atoms --------------------------------

export function geomean(values) {
  const clean = values.filter(Number.isFinite);
  if (!clean.length) return null;
  return Math.exp(clean.reduce((sum, v) => sum + Math.log(v), 0) / clean.length);
}

/**
 * Best-so-far validated speedup S(t) = L0 / L^best(t), where only
 * `status == "passed"` solutions participate and the result never drops
 * below 1 (a validated kernel can never be "worse than baseline" on paper).
 * `points` must be sorted ascending on the axis and include the baseline
 * origin {t: 0, s: 1}; each later point is one *passed* solution.
 * `axis` = "tokens" | "time".
 */
export function sCurve(points, axis = 'tokens') {
  const key = axis === 'time' ? 'wall_sec' : 'cum_tokens_k';
  const sorted = [...points].sort((a, b) => a[key] - b[key]);
  let best = 1;
  return sorted.map(point => {
    best = Math.max(best, point.s ?? 1);
    return { ...point, s_best: best };
  });
}

export function sAtBmax(points, axis = 'tokens') {
  const curve = sCurve(points, axis);
  return curve.length ? curve[curve.length - 1].s_best : 1;
}

export function sAtMidWall(points) {
  // S(T_mid): speedup at the wall-clock midpoint of the run (plateau rule).
  const withWall = points.filter(p => Number.isFinite(p.wall_sec));
  if (withWall.length < 2) return sAtBmax(points);
  const midWall = Math.max(...withWall.map(p => p.wall_sec)) / 2;
  const curve = sCurve(points, 'tokens');
  let value = 1;
  for (const point of curve) {
    if ((point.wall_sec ?? 0) <= midWall) value = point.s_best;
    else break;
  }
  return value;
}

/**
 * AULS (§4.2): area under the log-speedup curve on a logarithmic token
 * (or wall) axis. Checkpoints = K=20 log-spaced points between B_min and
 * B_max plus every solution token, merged. γ_k = ln(B_k/B_{k-1})/ln(B_max/B_min).
 * A constant S = 2× yields AULS = 1 by construction.
 * B_min is the first *passed* solution (the origin at t=0 is excluded because
 * ln(0) is undefined).
 */
export function auls(points, axis = 'tokens', k = 20) {
  const key = axis === 'time' ? 'wall_sec' : 'cum_tokens_k';
  const curve = sCurve(points, axis);
  const passed = curve.filter(p => p[key] > 0 && p.s_best > 1);
  if (passed.length < 1) return 0;
  const bMin = passed[0][key];
  const bMax = curve[curve.length - 1][key];
  if (!(bMax > bMin)) return 0;

  const checkpoints = new Set();
  for (let i = 0; i < k; i += 1) checkpoints.add(bMin * Math.pow(bMax / bMin, i / (k - 1)));
  passed.forEach(p => checkpoints.add(p[key]));
  const grid = [...checkpoints].sort((a, b) => a - b);

  const sAt = t => {
    let value = 1;
    for (const point of curve) {
      if (point[key] <= t) value = point.s_best;
      else break;
    }
    return value;
  };

  let area = 0;
  for (let i = 1; i < grid.length; i += 1) {
    const gamma = Math.log(grid[i] / grid[i - 1]) / Math.log(bMax / bMin);
    area += gamma * (log2(sAt(grid[i - 1])) + log2(sAt(grid[i]))) / 2;
  }
  return area;
}

/** STS (token efficiency score, 0-100 scale): 100·[0.5·log2 S(B_max) + 0.5·AULS]. */
export function stsFromParts(sBmax, aulsValue, axis = 'tokens') {
  return 100 * (0.5 * log2(Math.max(sBmax, 1)) + 0.5 * aulsValue);
}

export function sEquiv(stsToken) { return Math.pow(2, stsToken / 100); }

/** VLR: validated latent reduction = 1 − 1/S, as percent. */
export function vlr(s) { return (1 - 1 / Math.max(s, 1)) * 100; }

/**
 * Tokens-to-Δ (§4.2): first cumulative token count where VLR(T_j) ≥ Δ.
 * Returns {value, reason} — reason ∈ null | "never_reached".
 */
export function tokensToDelta(points, deltaPct, axis = 'tokens') {
  const key = axis === 'time' ? 'wall_sec' : 'cum_tokens_k';
  for (const point of sCurve(points, axis)) {
    if (point[key] <= 0) continue;
    if (vlr(point.s_best) >= deltaPct) return { value: point[key] * (axis === 'tokens' ? 1000 : 1), reason: null };
  }
  return { value: null, reason: 'never_reached' };
}

// ------------------------------- §4.3 PAOS ---------------------------------

/**
 * Run-level PAOS: geometric mean of per-shape S_lat = t_base / t_best over
 * shapes whose op_type matches the target operator.
 */
export function paosFromShapes(durationByShape) {
  const ratios = (durationByShape || [])
    .filter(row => Number.isFinite(row.s_lat) && row.s_lat > 0)
    .map(row => row.s_lat);
  if (!ratios.length) return null;
  return geomean(ratios);
}

export function worstBestShapes(durationByShape) {
  const rows = (durationByShape || []).filter(row => Number.isFinite(row.s_lat));
  if (!rows.length) return { worst: null, best: null };
  const worst = rows.reduce((a, b) => (b.s_lat < a.s_lat ? b : a));
  const best = rows.reduce((a, b) => (b.s_lat > a.s_lat ? b : a));
  return { worst: toShapePoint(worst), best: toShapePoint(best) };
}

function toShapePoint(row) {
  return { shape: row.shape, dtype: row.dtype, s_lat: round(row.s_lat, 3) };
}

/**
 * eTPS (absolute throughput, representative shape × recipe):
 * T_step = n_i · t_i (profile-only, T_other = 0 — annotated caliber),
 * eTPS = G / T_step.  eTPSGain on the representative shape; the bundle also
 * reports the geometric-mean caliber (== PAOS) as `etps_gain`.
 */
export function etps(recipeRow, tSeconds, g = 1) {
  if (!recipeRow || !Number.isFinite(tSeconds) || tSeconds <= 0) return null;
  const tStep = recipeRow.n_i * tSeconds;
  return g / tStep;
}

/** PPUI (utilization version): 1 + clip((U_best − U_base)/(1 − U_base), −1, 1). */
export function ppui(uBase, uBest) {
  if (!Number.isFinite(uBase) || !Number.isFinite(uBest) || uBase >= 1) return null;
  const delta = (uBest - uBase) / (1 - uBase);
  return 1 + Math.min(1, Math.max(-1, delta));
}

/** Non-official final PAOS with correctness penalty: PAOS · CorrectnessGate² (λ=2). */
export function paosFinal(paosValue, funcPassRatePercent) {
  if (!Number.isFinite(paosValue)) return null;
  return paosValue * Math.pow((funcPassRatePercent ?? 0) / 100, 2);
}

// ------------------------------- §4.4 flags --------------------------------

/**
 * Automatic health flags. `input` = operator-level raw signals:
 * { sBmax, paos, shapes, terminationReason, wastedRoundRate, epsilon, sAtMidWall }
 * Returns the ordered flag list (🟥 > 🟧 > 🟦).
 */
export function computeFlags(input) {
  const flags = [];
  const epsilon = input.epsilon ?? EPS_DEFAULT;
  const sBmax = input.sBmax ?? 1;
  const paos = input.paos;
  if (sBmax > 1 + epsilon && Number.isFinite(paos) && paos < 1) flags.push('fake_optimization');
  if ((input.shapes || []).some(row => Number.isFinite(row.s_lat) && row.s_lat < 0.9)) flags.push('shape_regression');
  if (input.terminationReason === 'cycle_limit_reached') flags.push('unconverged');
  if (Number.isFinite(input.wastedRoundRate) && input.wastedRoundRate > 0.5) flags.push('high_waste');
  if (Number.isFinite(input.sAtMidWall) && sBmax - input.sAtMidWall < epsilon) flags.push('plateau');
  return flags;
}

export const FLAG_ORDER = ['fake_optimization', 'shape_regression', 'unconverged', 'high_waste', 'plateau', 'baseline_abnormal'];

export function sortFlags(flags) {
  return FLAG_ORDER.filter(name => (flags || []).includes(name));
}

// --------------------------- cross-operator rollup --------------------------

/**
 * Entry-level overall across the operators that are *included* in
 * cross-operator aggregation (baseline_quality === "normal" only, per the
 * v2.0 red lines). `weights` is optional; defaults to uniform.
 * `targets` are the VLR tiers in percent (from the evaluator's `meta.targets`).
 */
export function aggregateOverall(operators, weights, targets = DEFAULT_TARGETS) {
  const included = operators.filter(op => (op.baseline_quality ?? 'normal') === 'normal');
  if (!included.length) return null;
  const weightOf = op => (weights?.[op.operator] ?? 1) / included.reduce((sum, o) => sum + (weights?.[o.operator] ?? 1), 0);

  const weighted = (fn) => included.reduce((sum, op) => sum + fn(op) * weightOf(op), 0);
  const median = values => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };

  const sTerms = included.map(op => 0.5 * log2(Math.max(op.s_bmax, 1)) + 0.5 * (op.auls ?? 0));
  const stsToken = 100 * included.reduce((sum, op, i) => sum + sTerms[i] * weightOf(op), 0);
  const sTimeTerms = included.map(op => 0.5 * log2(Math.max(op.s_bmax_time, 1)) + 0.5 * (op.auls_time ?? 0));
  const stsTime = 100 * included.reduce((sum, op, i) => sum + sTimeTerms[i] * weightOf(op), 0);

  const eir = {};
  for (const delta of targets) {
    eir[`${delta}%`] = round100(weighted(op => vlr(op.s_bmax) >= delta ? 1 : 0) * 100);
  }

  const allShapes = included.flatMap(op => op.duration_by_shape || []);
  const { worst, best } = worstBestShapes(allShapes);
  // Cross-operator PAOS = geometric mean of the per-operator PAOS values.
  // NOTE: PAOS is √(PKAI×PTEI), not "geomean of per-shape S_lat" — those are
  // different quantities and must not share a name.
  const paosGeo = geomean(included.map(op => (typeof op.paos === 'number' ? op.paos : op.paos?.paos)).filter(Number.isFinite));
  const pkaiGeo = geomean(included.map(op => op.paos?.pkai).filter(Number.isFinite));
  const pteiGeo = geomean(included.map(op => op.paos?.ptei).filter(Number.isFinite));
  const ppuiValues = included.map(op => op.ppui_u).filter(Number.isFinite);

  const flags = sortFlags(included.flatMap(op => op.flags || []).concat(
    operators.some(op => (op.baseline_quality ?? 'normal') !== 'normal') ? ['baseline_abnormal'] : []
  ));

  const waste = included.map(op => op.wasted_round_rate).filter(Number.isFinite);
  const termination = included.map(op => op.termination_reason).find(t => t === 'cycle_limit_reached') ?? included[included.length - 1]?.termination_reason;

  // Speedup aggregates mirror evaluate: EBGS is the geometric mean S_i,
  // the arithmetic mean is reported alongside it.
  const speedups = included.map(op => op.s_bmax).filter(Number.isFinite);
  const ebgs = geomean(speedups);

  return {
    sts_token: round(stsToken, 2),
    sts_time: round(stsTime, 2),
    s_equiv: round(sEquiv(stsToken), 4),
    auls: round(weighted(op => op.auls ?? 0), 4),
    ebgs: round(ebgs, 4),
    speedup_mean: speedups.length ? round(speedups.reduce((a, b) => a + b, 0) / speedups.length, 4) : null,
    compile_pass: round(median(included.map(op => op.compile_pass)), 1),
    task_pass_rate: round(median(included.map(op => op.task_pass_rate)), 1),
    candidate_pass_rate: round(median(included.map(op => op.candidate_pass_rate)), 1),
    gate_rate: round(median(included.map(op => op.gate_rate)), 1),
    vlr_mean: round(median(included.map(op => vlr(op.s_bmax))), 2),
    best_duration_us: round(median(included.map(op => (op.baseline_ms ?? 0) * 1000 / Math.max(op.s_bmax, 1))), 2),
    tokens_total_k: round(included.reduce((sum, op) => sum + (op.tokens_k ?? 0), 0), 1),
    api_cost_usd: null,
    api_cost_usd_reason: 'pricing_table_not_configured',
    rounds_median: round(median(included.map(op => op.rounds ?? 0)), 0),
    wall_time_median_s: round(median(included.map(op => op.wall_time_s ?? 0)), 1),
    wall_time_total_s: round(included.reduce((sum, op) => sum + (op.wall_time_s ?? 0), 0), 1),
    profiling_runs_median: round(median(included.map(op => op.profiling_runs ?? 0)), 0),
    tool_call_acc: null,
    tool_call_acc_reason: 'pending_backend',
    eir,
    paos: {
      paos: round(paosGeo, 4),
      pkai: round(pkaiGeo, 4),
      ptei: round(pteiGeo, 4),
      etps_gain: round(pteiGeo, 4),
      // Per-shape vectors are not part of the evaluator contract yet, so these
      // stay null rather than being back-filled from an unrelated quantity.
      s_lat_geo: null,
      worst_shape: worst,
      best_shape: best,
      ppui_u: ppuiValues.length ? round(ppuiValues.reduce((a, b) => a + b, 0) / ppuiValues.length, 3) : null
    },
    cost: {
      wasted_round_rate: waste.length ? round(waste.reduce((a, b) => a + b, 0) / waste.length, 3) : null,
      n_solutions: included.reduce((sum, op) => sum + (op.n_solutions ?? 0), 0),
      n_cycles: included.reduce((sum, op) => sum + (op.n_cycles ?? 0), 0),
      termination_reason: termination ?? null
    },
    flags,
    excluded_operators: operators.filter(op => (op.baseline_quality ?? 'normal') !== 'normal').map(op => op.operator)
  };
}

// ------------------------------- §4.5 reliability ---------------------------

/** Reliability@Δ: fraction of same-config runs with VLR(B_max) ≥ Δ; K < 3 → null. */
export function reliabilityAtDelta(runs, deltaPct) {
  if (!runs || runs.length < 3) return { value: null, reason: 'k_lt_3', k: runs?.length ?? 0 };
  const hits = runs.filter(run => vlr(run.s_bmax ?? 1) >= deltaPct).length;
  return { value: round(hits / runs.length, 3), reason: null, k: runs.length };
}

// ------------------------------- §7.3 radar ---------------------------------

/** Seven-dimension radar scores (0-100, higher = better). Reliability is null when K < 3. */
export function radarScores(entry) {
  const o = entry.overall;
  return [
    {key:'correctness',label:'Correctness',value:o.task_pass_rate ?? null},
    {key:'optimization',label:'Latency reduction',value:o.vlr_mean ?? null},
    {key:'token_eff',label:'Token efficiency',value:null},
    {key:'round_eff',label:'Round efficiency',value:null},
    {key:'time_eff',label:'Time efficiency',value:null},
    {key:'utilization',label:'Utilization',value:null},
    {key:'reliability',label:'Reliability',value:null,reason:'independent_repeats_required'}
  ];
}

// ------------------------------- verdict ------------------------------------

/** Drawer TL;DR verdict: positive / mixed / negative / excluded. */
export function verdictFor(entry) {
  const overall = entry.overall;
  if ((entry.baseline_quality ?? 'normal') !== 'normal') return 'excluded';
  if ((overall.flags || []).includes('fake_optimization') || (overall.vlr_mean ?? 0) < 0) return 'negative';
  if ((overall.vlr_mean ?? 0) >= 3 && !(overall.flags || []).includes('shape_regression')) return 'positive';
  return 'mixed';
}

export function verdictSentence(entry) {
  return `${entry.coverage_operators ?? 0} operators; STS coverage ${entry.sts_coverage ?? entry.coverage_operators}/${entry.coverage_operators}. STS is evaluated at 50M tokens. ${entry.estimated_runs ? `${entry.estimated_runs} runs use source-estimated token axes. ` : ''}Missing diagnostics remain unavailable; search gains alone do not establish end-to-end workload speedup.`;
}

/**
 * Local display helpers.
 *
 * Deliberately defined here instead of imported from `format.js`: that module
 * reaches `meta.js` -> `vendors.js` through extensionless imports, which bare
 * Node (used by `scripts/verify-anchors.mjs`) cannot resolve. Precision still
 * comes from the shared policy, so the numbers agree with the UI.
 */
function fixed(value, decimals) {
  return value == null || !Number.isFinite(value) ? '–' : Number(value).toFixed(decimals);
}
function ratio(value, decimals = PRECISION.ratio) {
  return value == null || !Number.isFinite(value) ? '–' : `${Number(value).toFixed(decimals)}×`;
}

// ------------------------------- helpers ------------------------------------

export function round(value, digits = 3) {
  if (value == null || !Number.isFinite(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function round100(value) { return Math.round(value * 10) / 10; }

/** Operator-level derived block from raw anytime + shape data. */
export function deriveOperator(op) {
  const sBmax = sAtBmax(op.anytime, 'tokens');
  const sBmaxTime = sAtBmax(op.anytime, 'time');
  const aulsValue = auls(op.anytime, 'tokens');
  const aulsTime = auls(op.anytime, 'time');
  const paos = paosFromShapes(op.duration_by_shape);
  const { worst, best } = worstBestShapes(op.duration_by_shape);
  const tokensTo = {};
  for (const delta of [1, 3, 5, 10]) {
    const hit = tokensToDelta(op.anytime, delta, 'tokens');
    tokensTo[`p${delta}`] = hit.value == null ? null : Math.round(hit.value);
    if (hit.reason) tokensTo[`p${delta}_reason`] = hit.reason;
  }
  const uBase = op.utilization?.u_base ?? null;
  const uBest = op.utilization?.u_best ?? null;
  const flags = computeFlags({
    sBmax,
    paos,
    shapes: op.duration_by_shape,
    terminationReason: op.termination_reason,
    wastedRoundRate: op.wasted_round_rate,
    epsilon: op.epsilon_used,
    sAtMidWall: sAtMidWall(op.anytime)
  });
  return {
    ...op,
    s_bmax: round(sBmax, 4),
    s_bmax_time: round(sBmaxTime, 4),
    vlr: round(vlr(sBmax), 2),
    auls: round(aulsValue, 4),
    auls_time: round(aulsTime, 4),
    sts_token: round(stsFromParts(sBmax, aulsValue), 2),
    sts_time: round(stsFromParts(sBmaxTime, aulsTime), 2),
    s_equiv: round(sEquiv(stsFromParts(sBmax, aulsValue)), 4),
    paos: paos == null ? null : round(paos, 4),
    worst_shape: worst,
    best_shape: best,
    tokens_to: tokensTo,
    ppui_u: ppui(uBase, uBest) == null ? null : round(ppui(uBase, uBest), 3),
    flags: sortFlags(flags)
  };
}
