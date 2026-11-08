// ============================================================================
// eval_summary.json → leaderboard entry.
//
// The evaluator (`evaluate/evaluate_operator_runs.py`) emits a *summary*: one
// file per evaluation root, holding suite metrics plus a `runs[]` array. The
// dashboard renders *entries*: one row per (agent × engine model).
//
// This module is the single implementation of that translation, so the two
// callers can never disagree:
//
//   • scripts/build-operator-bundle.mjs — builds the static bundle at build time
//   • src/services/datasets.js          — loads the internal dataset folder at
//                                         runtime, one chunk per dataset file
//
// Plain ESM, no Node and no DOM dependency, so the browser can run it. The only
// import is the vendor registry, which is dependency-free by design.
//
// Anything absent from the summary stays `null` and renders as "–": we never
// synthesise a plausible number. A visible gap is safer than an invisible lie.
// ============================================================================

import { modelPreset as presetFor, vendorFor } from './vendors.js';
import { computeFlags, sortFlags, EPS_DEFAULT } from './metrics.js';

/** Deployment-specific model presentation, keyed by engine_model id. */
export const MODEL_CATALOG = {
  'Qwen3.5-122B': { vendor: 'Alibaba', openness: 'open', vendor_mark: 'Q', color: '#2E86DE' },
  'Qwen3.6-35B': { vendor: 'Alibaba', openness: 'open', vendor_mark: 'Q', color: '#2E86DE' },
  'DeepSeek-V4-Pro': { vendor: 'DeepSeek', openness: 'open', vendor_mark: 'D', color: '#8C58C6' },
  'DeepSeek-V4-Flash-BF16': { vendor: 'DeepSeek', openness: 'open', vendor_mark: 'D', color: '#8C58C6' },
  'DeepSeek-V3.2': { vendor: 'DeepSeek', openness: 'open', vendor_mark: 'D', color: '#8C58C6' },
  'GPT-5.6-Sol': { vendor: 'OpenAI', openness: 'closed', vendor_mark: 'G', color: '#0E9B8E' },
  'gpt-5.6-terra': { vendor: 'OpenAI', openness: 'closed', vendor_mark: 'G', color: '#0E9B8E' },
  'Claude-Opus-4.8': { vendor: 'Anthropic', openness: 'closed', vendor_mark: 'C', color: '#6D5B52' },
  'claude-opus-5': { vendor: 'Anthropic', openness: 'closed', vendor_mark: 'C', color: '#6D5B52' },
  'Gemini-3.1-Pro': { vendor: 'Google', openness: 'closed', vendor_mark: 'G', color: '#3F67D5' },
  'GLM-5.2': { vendor: 'Zhipu', openness: 'open', vendor_mark: 'Z', color: '#3A7F96' },
  'glm-5.3': { vendor: 'Zhipu', openness: 'open', vendor_mark: 'Z', color: '#3A7F96' },
  'Kimi-K2.7': { vendor: 'Moonshot', openness: 'open', vendor_mark: 'K', color: '#D14B70' }
};

/** Model assumed when a summary names neither a model nor an id we know. */
export const DEFAULT_ENGINE_MODEL = {
  id: 'Qwen3.5-122B',
  vendor: 'Alibaba',
  openness: 'open',
  vendor_mark: 'Q',
  color: '#2E86DE'
};

/** Hardware / recipe labels. Presentation only — never mixed into metrics. */
export const CHIPS = ['Ascend 910C'];
export const RECIPE = { 'Qwen3.5-122B': { decode_b1: [] } };

/**
 * Is this parsed JSON an evaluator summary rather than a dashboard bundle?
 *
 * Summaries carry `runs[]` and never `entries`; bundles are the inverse. The
 * check is deliberately strict so a malformed bundle is not silently treated as
 * a summary and rendered as an empty entry.
 */
export function looksLikeSummary(raw) {
  return Boolean(
    raw && typeof raw === 'object'
    && Array.isArray(raw.runs)
    && !Array.isArray(raw.entries)
  );
}

// ------------------------------- conversions --------------------------------

/** Fraction (0–1) → percent. Pass-through for null. */
function toPercent(value) {
  return Number.isFinite(value) ? value * 100 : null;
}

/** Absolute token count → thousands of tokens. */
function toKilo(value) {
  return Number.isFinite(value) ? value / 1000 : null;
}

/** Milliseconds → microseconds (the UI displays µs). */
function toMicros(value) {
  return Number.isFinite(value) ? value * 1000 : null;
}

function sEquiv(stsToken) {
  return Number.isFinite(stsToken) ? 2 ** (stsToken / 100) : null;
}

/** Mean of one run-level accuracy field across runs. */
function rowsAccuracy(rows, field) {
  return mean(rows.map(row => row?.accuracy?.[field]));
}

function numbers(values) {
  return values.filter(value => Number.isFinite(value));
}

function mean(values) {
  const clean = numbers(values);
  if (!clean.length) return null;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function median(values) {
  const clean = [...numbers(values)].sort((a, b) => a - b);
  if (!clean.length) return null;
  const mid = Math.floor(clean.length / 2);
  return clean.length % 2 ? clean[mid] : (clean[mid - 1] + clean[mid]) / 2;
}

/** `<operator>-<agent>-<YYYYMMDD>-<HHMMSS>` → identity parts. */
export function parseTaskId(taskId) {
  const parts = String(taskId ?? '').split('-');
  const hasStamp = parts.length >= 3 && /^\d{8}$/.test(parts.at(-2) ?? '') && /^\d{6}$/.test(parts.at(-1) ?? '');
  const head = hasStamp ? parts.slice(0, -2) : parts;
  return {
    operator: head[0] ?? 'Operator',
    agent: head.slice(1).join('-') || 'agent',
    stamp: hasStamp ? `${parts.at(-2)}-${parts.at(-1)}` : null
  };
}

/**
 * Operator identity from the run's `task_id` prefix (`add_rms_norm` →
 * `AddRmsNorm`).
 *
 * Deliberately *not* a closed whitelist: an evaluation root that adds an
 * operator this repository has never seen must still get its own bucket.
 * Collapsing unknown prefixes into a shared catch-all would silently undercount
 * the operator roster and merge unrelated measurements into one row.
 */
export function toOperatorName(taskId) {
  const prefix = String(parseTaskId(taskId).operator ?? '').trim();
  if (!prefix) return 'Operator';
  const derived = prefix
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean)
    .map(part => part[0].toUpperCase() + part.slice(1))
    .join('');
  return derived || 'Operator';
}

/** Run-level STS/PAOS values are optional; return null instead of guessing. */
function runSts(row) {
  return row?.sts?.status === 'computed' ? row.sts : {};
}

function runPaos(row) {
  return row?.paos ?? {};
}

function profileValue(row, key) {
  const value = runPaos(row).profiling?.[key];
  return Number.isFinite(value) ? value : null;
}

/**
 * Did this run actually produce a profile?
 *
 * The evaluator records a per-side status (`baseline` / `agent`) under
 * `paos.profiling`. A run counts as profiled when either side reached
 * `computed`, which is the honest definition of "profiling runs" — the heatmap
 * exposes this metric, and returning a run count instead would mislabel it.
 */
function hasProfile(row) {
  const profiling = runPaos(row).profiling ?? {};
  return ['baseline', 'agent'].some(side => profiling[side]?.status === 'computed');
}

/**
 * Targets come from whatever keys the evaluator actually emitted
 * (`TokensTo1%`, `EIR_5%`, …). Nothing is hard-coded to a fixed tier list.
 */
function targetTiers(runRows) {
  const seen = new Set();
  for (const row of runRows) {
    for (const key of Object.keys(runSts(row))) {
      const match = /^TokensTo(.+)$/.exec(key);
      if (match) seen.add(match[1]);
    }
    for (const key of Object.keys(row?.sts ?? {})) {
      const match = /^TokensTo(.+)$/.exec(key);
      if (match) seen.add(match[1]);
    }
  }
  return [...seen];
}

// ------------------------------ model metadata ------------------------------

/**
 * Model metadata used for COLOURS / vendor marks, keyed by engine_model id.
 *
 * Resolution order, most authoritative first:
 *   1. what the evaluator itself declared for this model (it read the served
 *      model id off the run artifacts — it outranks anything we would guess),
 *   2. the deployment-specific `MODEL_CATALOG` above,
 *   3. the shared SOTA preset roster in `lib/vendors.js` (exact id match),
 *   4. a vendor family prefix match (`Qwen*`, `GLM*`, …) — mark + accent only,
 *      never an invented `openness`,
 *   5. a deterministic HSL colour so an unknown model still renders stably.
 *
 * Step 1 exists because a family-prefix guess is not evidence: a summary that
 * says a `GPT-*` model came from Alibaba is reporting what actually ran, and
 * silently overriding it with "GPT ⇒ OpenAI" would misattribute the benchmark.
 *
 * @param declared `{ vendor, openness }` from the summary's `generators` block
 */
function modelMetaFor(id, declared = {}) {
  const explicitVendor = typeof declared.vendor === 'string' && declared.vendor.trim()
    ? declared.vendor.trim() : null;
  const explicitOpenness = declared.openness ?? null;

  const known = MODEL_CATALOG?.[id];
  const preset = presetFor(id);

  // Presentation (mark + accent) still comes from the registries so the chip
  // renders; only the *facts* are taken from the declaration.
  const presentation = known
    ? { vendor_mark: known.vendor_mark, color: known.color }
    : preset
      ? { vendor_mark: preset.vendor_mark ?? preset.vendor?.[0] ?? null, color: preset.color }
      : (() => {
        let hash = 0;
        for (const char of String(id ?? '')) hash = (hash * 31 + char.charCodeAt(0)) % 360;
        return { vendor_mark: (id ?? '?')[0], color: `hsl(${hash}, 52%, 46%)` };
      })();

  const vendor = explicitVendor ?? known?.vendor ?? preset?.vendor ?? 'Other';
  // `preset.openness` is null for a family-prefix match — a guess, not a fact.
  const openness = explicitOpenness ?? known?.openness ?? preset?.openness ?? null;

  return withVendorId({ id, vendor, openness, ...presentation });
}

/**
 * Stamp the registry vendor id onto the emitted engine_model so the UI can
 * resolve the right brand mark without re-deriving it from a display label.
 *
 * The id is derived from the *resolved vendor name*, not from the model id, so
 * an explicitly declared "Alibaba" maps to the Alibaba mark even when the model
 * id would otherwise look like another vendor's. An unknown vendor resolves to
 * `generic`, which matches the `vendor: 'Other'` label it carries.
 */
function withVendorId(model) {
  return { ...model, vendor_id: vendorFor({ vendor: model.vendor }).id };
}

// ---------------------------------- entry -----------------------------------

/**
 * Build one leaderboard entry from one evaluation summary.
 *
 * @param summary parsed `eval_summary.json`
 * @param source  optional `{ id, agent, engine_model }` identity override —
 *                explicit reviewer intent beats anything read from the file.
 */
export function buildEntryFromSummary(summary, source = {}) {
  const runRows = Array.isArray(summary?.runs) ? summary.runs : [];
  const accuracySuite = summary?.accuracy ?? {};
  const costSuite = summary?.cost ?? {};
  const stsSuite = summary?.sts ?? {};
  const paosSuite = summary?.paos ?? {};

  const gated = runRows.filter(row => row?.sts?.status === 'computed');
  const tiers = targetTiers(runRows);
  const identities = runRows.map(row => parseTaskId(row.task_id));
  const operatorNames = [...new Set(runRows.map(row => toOperatorName(row.task_id)))];

  const eir = {};
  for (const tier of tiers) eir[tier] = toPercent(stsSuite[`EIR_${tier}`]);

  const perOperator = operatorNames.map((operatorName) => {
    const rows = runRows.filter(row => toOperatorName(row.task_id) === operatorName);
    const rowsGated = rows.filter(row => row?.sts?.status === 'computed');
    const speedups = numbers(rowsGated.map(row => row.sts.Speedup));

    const tokensTo = {};
    for (const tier of tiers) {
      const values = numbers(rows.map(row => row?.sts?.[`TokensTo${tier}`]));
      tokensTo[tier] = values.length ? median(values) : null;
    }

    // Real best-so-far trajectories exported by the evaluator. A run with a
    // missing baseline emits no points, so the sparkline stays empty rather
    // than drawing an invented shape.
    const bestSoFar = (axisKey, xKey) => {
      const points = rows
        .flatMap(row => (Array.isArray(row?.anytime?.[axisKey]) ? row.anytime[axisKey] : []))
        .filter(point => Number.isFinite(point?.[xKey]) && Number.isFinite(point?.Speedup))
        .sort((a, b) => a[xKey] - b[xKey]);
      const curve = [];
      let best = 1;
      for (const point of points) {
        best = Math.max(best, point.Speedup);
        const last = curve[curve.length - 1];
        if (last && last.x === point[xKey]) last.s = best;
        else curve.push({ x: point[xKey], s: best });
      }
      return curve;
    };
    const anytime = bestSoFar('token', 'Tokens').map(point => ({ cum_tokens_k: point.x / 1000, wall_sec: null, s: point.s }));
    const anytimeTime = bestSoFar('time', 'Seconds').map(point => ({ cum_tokens_k: null, wall_sec: point.x, s: point.s }));

    const paosValues = numbers(rowsGated.map(row => row?.paos?.PAOS_base));
    const etpsGains = numbers(rowsGated.map(row => row?.paos?.PTEI));

    // Flags are DERIVED, not copied: the evaluator does not export them, but the
    // rules are published (see FLAG vocabulary in lib/meta.js and the
    // methodology section). Running the shared rule engine here is what makes
    // that documentation true. Signals the evaluator omits are passed as null,
    // so the corresponding rule stays silent instead of guessing.
    const sBmaxValue = speedups.length ? Math.max(...speedups) : null;
    const paosValue = paosValues.length ? Math.max(...paosValues) : null;
    const flags = sortFlags(computeFlags({
      sBmax: sBmaxValue,
      paos: paosValue,
      // Per-shape latency is not exported yet, so shape_regression cannot be
      // evaluated. Passing null keeps it silent rather than inventing rows.
      shapes: null,
      terminationReason: rows.map(row => row?.accuracy?.FailureReason).find(Boolean) ?? null,
      wastedRoundRate: null,
      epsilon: EPS_DEFAULT,
      sAtMidWall: null
    }));

    return {
      operator: operatorName,
      flags,
      // `bound` is not measured by the evaluator; leave unknown rather than guessed.
      bound: null,
      baseline_ms: median(rows.map(row => row?.durations?.BaselineMs)),
      epsilon_used: null,
      compile_pass: null,
      // Correctness, per operator — same definitions as the evaluator's
      // accuracy block (TaskPass / CandidatePassRate / CorrectnessGate).
      task_pass_rate: toPercent(rowsAccuracy(rows, 'TaskPass')),
      candidate_pass_rate: toPercent(rowsAccuracy(rows, 'CandidatePassRate')),
      gate_rate: toPercent(rowsAccuracy(rows, 'CorrectnessGate')),
      tokens_k: toKilo(mean(rows.map(row => row?.cost?.TotalTokens))),
      rounds: mean(rows.map(row => row?.cost?.LLMRounds)),
      wall_time_s: mean(rows.map(row => row?.cost?.TotalWallClockSec)),
      // Count of runs in this operator that produced a profile, so the heatmap's
      // "Profiling Runs" mode shows a real number instead of a permanent gap.
      profiling_runs: rows.filter(hasProfile).length || null,
      wasted_round_rate: null,
      n_solutions: mean(rows.map(row => row?.accuracy?.CandidatesTotal)),
      n_cycles: null,
      termination_reason: null,
      baseline_quality: 'normal',
      recipe: {
        G: paosSuite.G ?? null,
        n_i: Array.isArray(paosSuite.calls_per_step) ? paosSuite.calls_per_step[0] ?? null : null,
        rep_shape: null
      },
      utilization: {
        vec_ratio_base: null,
        vec_ratio_best: mean(rows.map(row => profileValue(row, 'VecRatio'))),
        cube_ratio_best: mean(rows.map(row => profileValue(row, 'CubeRatio'))),
        mte2_ratio_best: mean(rows.map(row => profileValue(row, 'MTE2Ratio'))),
        u_base: null,
        u_best: mean(rows.map(row => profileValue(row, 'U_i'))),
        u_step: mean(rows.map(row => profileValue(row, 'U_step'))),
        s_i_aicore: mean(rows.map(row => profileValue(row, 'S_i_aicore'))),
        // Deliberately absent: the schema has no FLOPs / byte counters.
        l2_bw: null,
        l2_bw_reason: 'not_collected',
        hbm_bw: null,
        hbm_bw_reason: 'not_collected',
        overlap: null,
        overlap_reason: 'not_collected'
      },
      duration_by_shape: null,
      duration_by_shape_reason: 'per_shape_trace_not_exported',
      anytime,
      anytime_time: anytimeTime,
      anytime_failed: [],
      runs: rows.map(row => ({
        run_id: row.task_id,
        date: parseTaskId(row.task_id).stamp,
        s_bmax: runSts(row).Speedup ?? null,
        paos: row?.paos?.PAOS_base ?? null,
        status: row.status ?? null,
        // Run-level flags follow the same rules as the operator rollup; the
        // per-shape and plateau signals are unavailable at run level, so those
        // rules stay silent.
        flags: sortFlags(computeFlags({
          sBmax: runSts(row).Speedup ?? null,
          paos: row?.paos?.PAOS_base ?? null,
          shapes: null,
          terminationReason: row?.accuracy?.FailureReason ?? null,
          wastedRoundRate: null,
          epsilon: EPS_DEFAULT,
          sAtMidWall: null
        })),
        termination_reason: row?.accuracy?.FailureReason ?? null,
        cycles: [],
        baseline_ms: row?.durations?.BaselineMs ?? null,
        best_ms: row?.durations?.BestMs ?? null,
        // Who produced this specific run. Kept per run (not just per entry) so a
        // suite that mixed agents or models stays visible in the drawer.
        identity: row?.identity ?? null
      })),
      s_bmax: speedups.length ? Math.max(...speedups) : null,
      s_bmax_time: null,
      vlr: toPercent(median(rowsGated.map(row => row.sts.VLR))),
      auls: median(rowsGated.map(row => row.sts.AULS_token)),
      auls_time: median(rowsGated.map(row => row.sts.AULS_time)),
      sts_token: mean(rowsGated.map(row => row.sts.STS_token)),
      sts_time: mean(rowsGated.map(row => row.sts.STS_time)),
      s_equiv: sEquiv(mean(rowsGated.map(row => row.sts.STS_token))),
      best_duration_us: toMicros(median(rows.map(row => row?.durations?.BestMs))),
      paos: {
        paos: paosValues.length ? Math.max(...paosValues) : null,
        pkai: mean(rowsGated.map(row => row?.paos?.PKAI)),
        ptei: mean(rowsGated.map(row => row?.paos?.PTEI)),
        etps_base: mean(rowsGated.map(row => row?.paos?.eTPS_base)),
        etps_agent: mean(rowsGated.map(row => row?.paos?.eTPS_agent)),
        etps_gain: etpsGains.length ? mean(etpsGains) : null,
        s_i_lat: mean(rowsGated.map(row => row?.paos?.S_i_lat)),
        s_i_aicore: mean(rowsGated.map(row => profileValue(row, 'S_i_aicore'))),
        u_i: mean(rowsGated.map(row => profileValue(row, 'U_i'))),
        u_step: mean(rowsGated.map(row => profileValue(row, 'U_step'))),
        vec_ratio: mean(rowsGated.map(row => profileValue(row, 'VecRatio'))),
        cube_ratio: mean(rowsGated.map(row => profileValue(row, 'CubeRatio'))),
        mte2_ratio: mean(rowsGated.map(row => profileValue(row, 'MTE2Ratio'))),
        // Per-shape profiling vectors are not part of the evaluator contract yet.
        s_lat_geo: null,
        worst_shape: null,
        best_shape: null,
        shape_reason: 'per_shape_trace_not_exported',
        ppui_u: null
      },
      verification: {
        pass_rate: mean(rowsGated.map(row => row?.sts?.VerificationPassRate)),
        median_ms: median(rowsGated.map(row => row?.sts?.VerificationMedianMs)),
        cv: mean(rowsGated.map(row => row?.sts?.VerificationCV)),
        speedup_confirmed: mean(rowsGated.map(row => row?.sts?.SpeedupConfirmed)),
        vlr_confirmed: mean(rowsGated.map(row => row?.sts?.VLRConfirmed))
      },
      tokens_to: tokensTo
    };
  });

  // The run list behind the drawer's boxplot: the operator with the most runs is
  // the only one that can support a distribution, and the UI labels which
  // operator it used.
  const reliabilitySource = perOperator.reduce(
    (best, op) => ((op.runs?.length ?? 0) > (best.runs?.length ?? 0) ? op : best),
    { operator: null, runs: [] }
  );

  // Identity precedence, most authoritative first:
  //   1. `source` — an explicit override passed by the caller (scripts/tests),
  //   2. the per-run records in `runs[].identity` — the granular measured
  //      record, and the thing the suite-level rollup is derived *from*,
  //   3. the `generators` rollup — the evaluator's own summary statement,
  //   4. the run `task_id` — last resort for summaries predating identity.
  //
  // A summary can mix models across runs, so the run-level winner is the
  // majority rather than simply the first: ties break on id for determinism.
  // Preferring the granular record over the rollup means a summary whose two
  // halves disagree (a hand-edited file) resolves the same way every time
  // instead of depending on which field happened to be read first.
  const exported = summary?.generators ?? {};
  const exportedAgent = (exported.agents ?? [])[0] ?? {};
  const exportedModel = (exported.engine_models ?? [])[0] ?? {};

  const runModels = new Map();
  let runAgent = null;
  for (const row of runRows) {
    const identity = row?.identity ?? {};
    const model = identity.engine_model;
    if (model?.id) {
      const seen = runModels.get(model.id) ?? { count: 0, model, agent: identity.agent ?? null };
      seen.count += 1;
      runModels.set(model.id, seen);
    }
    if (!runAgent && identity.agent?.id) runAgent = identity.agent;
  }
  const majority = [...runModels.entries()]
    .sort((a, b) => b[1].count - a[1].count || String(a[0]).localeCompare(String(b[0])))[0]?.[1] ?? null;

  const declaredAgent = runAgent ?? exportedAgent;
  const declaredModel = majority?.model ?? exportedModel;

  const agent = {
    id: source.agent?.id ?? declaredAgent.id ?? identities[0]?.agent ?? 'agent',
    type: source.agent?.type ?? declaredAgent.type ?? 'general',
    strategy: source.agent?.strategy ?? declaredAgent.strategy ?? null,
    version: source.agent?.version ?? null
  };
  const engineModel = modelMetaFor(
    source.engine_model?.id ?? declaredModel.id ?? DEFAULT_ENGINE_MODEL.id,
    { vendor: source.engine_model?.vendor ?? declaredModel.vendor, openness: declaredModel.openness }
  );
  const entryId = source.id ?? `${agent.id}+${engineModel.id}`;

  const overallVerification = {
    pass_rate: mean(gated.map(row => row?.sts?.VerificationPassRate)),
    median_ms: median(gated.map(row => row?.sts?.VerificationMedianMs)),
    cv: mean(gated.map(row => row?.sts?.VerificationCV)),
    speedup_confirmed: mean(gated.map(row => row?.sts?.SpeedupConfirmed)),
    vlr_confirmed: mean(gated.map(row => row?.sts?.VLRConfirmed))
  };

  // K = number of *same-configuration* repeats, per §4.5: `gated.length` counts
  // gated runs across every operator, so three different operators each run once
  // would masquerade as K=3 and the drawer would draw a boxplot over three
  // unrelated kernels. The honest K is the largest run count for a single
  // operator — the same set `runs_for_reliability` exposes.
  const kGroup = reliabilitySource.runs?.length ?? 0;

  return {
    id: entryId,
    agent,
    engine_model: engineModel,
    weights: engineModel.openness ?? null,
    links: { report: '#' },
    k_group: kGroup,
    baseline_quality: 'normal',
    coverage_operators: perOperator.length,
    scenario: {
      runtime_scenario: paosSuite.runtime_scenario ?? null,
      G: paosSuite.G ?? null,
      batch_size: paosSuite.batch_size ?? null,
      chunk_len: paosSuite.chunk_len ?? null,
      T_other_ms: paosSuite.T_other_ms ?? null,
      calls_per_step: paosSuite.calls_per_step ?? null,
      input_source: paosSuite.input_source ?? null
    },
    overall: {
      sts_token: stsSuite.mean_STS_token ?? null,
      sts_time: stsSuite.mean_STS_time ?? null,
      s_equiv: sEquiv(stsSuite.mean_STS_token),
      auls: stsSuite.mean_AULS_token ?? null,
      auls_time: stsSuite.mean_AULS_time ?? null,
      ebgs: stsSuite.EBGS ?? null,
      // Correctness: TaskPassRate (soft) / GateRate (hard gate into perf ranking)
      // / RunCompletedRate — straight from the evaluator's accuracy block.
      task_pass_rate: toPercent(accuracySuite.TaskPassRate),
      gate_rate: toPercent(accuracySuite.GateRate),
      run_completed_rate: toPercent(accuracySuite.RunCompletedRate),
      candidate_pass_rate: toPercent(median(runRows.map(row => row?.accuracy?.CandidatePassRate))),
      compile_pass: null,
      vlr_mean: toPercent(stsSuite.mean_VLR),
      speedup_mean: stsSuite.mean_Speedup ?? null,
      speedup_median: stsSuite.median_Speedup ?? null,
      speedup_max: stsSuite.max_Speedup ?? null,
      best_duration_us: toMicros(median(runRows.map(row => row?.durations?.BestMs))),
      baseline_duration_ms: median(runRows.map(row => row?.durations?.BaselineMs)),
      tokens_total_k: toKilo(costSuite.TotalTokens),
      // Prices are applied by the UI (imported table) or by the build script
      // when a pricing file exists. Never invented here.
      api_cost_usd: null,
      api_cost_usd_reason: 'pricing_table_not_applied',
      rounds_median: costSuite.median_LLMRounds ?? null,
      wall_time_median_s: costSuite.median_WallClockSec ?? null,
      wall_time_total_s: costSuite.TotalWallClockSec ?? null,
      tool_calls_total: costSuite.TotalToolCalls ?? null,
      // Median of the per-operator profiling counts — matching both the label
      // ("Profiling Runs 中位") and lib/metrics.js, which derives the same field
      // as `median(op.profiling_runs)`. A run count here would put an entry-level
      // total under a per-operator median label.
      profiling_runs_median: median(perOperator.map(op => op.profiling_runs)),
      tool_call_acc: null,
      tool_call_acc_reason: 'not_collected',
      eir,
      paos: {
        paos: paosSuite.mean_PAOS_base ?? null,
        pkai: paosSuite.mean_PKAI ?? null,
        ptei: paosSuite.mean_PTEI ?? null,
        etps_base: paosSuite.mean_eTPS_base ?? null,
        etps_agent: paosSuite.mean_eTPS_agent ?? null,
        etps_gain: paosSuite.mean_PTEI ?? null,
        s_i_aicore: paosSuite.mean_S_i_aicore ?? null,
        u_i: paosSuite.mean_U_i ?? null,
        u_step: paosSuite.mean_U_step ?? null,
        vec_ratio: paosSuite.mean_VecRatio ?? null,
        cube_ratio: paosSuite.mean_CubeRatio ?? null,
        mte2_ratio: paosSuite.mean_MTE2Ratio ?? null,
        computed_runs: paosSuite.computed_runs ?? null,
        status: paosSuite.status ?? null,
        reason: paosSuite.reason ?? null,
        s_lat_geo: null,
        worst_shape: null,
        best_shape: null,
        shape_reason: 'per_shape_trace_not_exported',
        ppui_u: null
      },
      verification: overallVerification,
      cost: {
        n_solutions: runRows.length,
        total_tokens: costSuite.TotalTokens ?? null,
        total_wall_clock_sec: costSuite.TotalWallClockSec ?? null,
        mean_llm_rounds: costSuite.mean_LLMRounds ?? null,
        total_tool_calls: costSuite.TotalToolCalls ?? null,
        // Token breakdown for API-cost pricing (see src/lib/pricing.js).
        input_tokens: costSuite.TotalInputTokens ?? null,
        cached_input_tokens: costSuite.TotalCachedInputTokens ?? null,
        cache_write_tokens: costSuite.TotalCacheWriteTokens ?? null,
        output_tokens: costSuite.TotalOutputTokens ?? null,
        wasted_round_rate: null,
        n_cycles: null,
        termination_reason: null
      },
      // Entry flags = union of the per-operator flags, so the leaderboard Flag
      // column and the drawer verdict reflect the operators' own signals.
      flags: sortFlags(perOperator.flatMap(op => op.flags ?? [])),
      excluded_operators: [],
      // Reliability needs K>=3 *same-configuration* repeats; report the honest K
      // (largest per-operator run count), not the cross-operator run total.
      reliability: {
        k: kGroup,
        reason: kGroup < 3 ? 'insufficient_runs' : null,
        tiers: Object.fromEntries(tiers.map(tier => [tier, toPercent(stsSuite[`EIR_${tier}`])]))
      }
    },
    // The drawer's "Run cohort" block and the S-distribution boxplot read these
    // two fields. The evaluator summary has no suite-level run list, so the
    // richest operator (most runs) stands in — the same choice the synthetic
    // bundle generator makes, and the operator is named alongside it.
    runs_for_reliability: reliabilitySource.runs,
    reliability_operator: reliabilitySource.operator,
    per_operator: perOperator
  };
}

// ---------------------------------- bundle -----------------------------------

/**
 * Headline rollup over a set of entries, matching `meta` in the built bundle.
 *
 * Entries = one per summary. Agents / operators are unioned across sources (the
 * same agent in two roots is still one agent), while runs and tokens are summed,
 * because those are per-run measurements.
 */
export function summarizeEntries(entries) {
  const list = entries ?? [];
  const operators = [...new Set(list.flatMap(entry => entry.per_operator.map(op => op.operator)))].sort();
  const agents = [...new Map(list.map(entry => [entry.agent.id, entry.agent])).values()]
    .sort((a, b) => String(a.id).localeCompare(String(b.id)));
  const totals = list.reduce((accumulator, entry) => ({
    tokens: accumulator.tokens + (entry.overall.tokens_total_k ?? 0),
    runs: accumulator.runs + (entry.overall.cost?.n_solutions ?? 0)
  }), { tokens: 0, runs: 0 });
  const scenarios = [...new Set(list.map(entry => entry.scenario?.runtime_scenario).filter(Boolean))];
  const tiers = [...new Set(list.flatMap(entry => Object.keys(entry.overall.eir ?? {})))].sort();
  // Dedupe by model id: two roots evaluating the same model is one cluster.
  const modelClusters = [...new Map(list.map(entry => [entry.engine_model.id, {
    id: entry.engine_model.id,
    scenarios: [...new Set(list
      .filter(other => other.engine_model.id === entry.engine_model.id)
      .map(other => other.scenario?.runtime_scenario)
      .filter(Boolean))]
  }])).values()];

  return {
    operators,
    agents,
    runsTotal: totals.runs,
    tokensTotalK: totals.tokens,
    scenarios,
    tiers,
    modelClusters
  };
}
