// ============================================================================
// Display precision policy — the ONE place that decides how many decimals a
// value measured by the evaluator renders with.
//
// Before this file, precision was scattered: `meta.js` carried a `decimals`
// per metric (0 / 1 / 2 / 4), and components added their own `.toFixed(1)`,
// `.toFixed(3)`, `.toFixed(4)`. The same quantity could therefore read `2.1%`
// in one column and `2.10%` in another, and there was nowhere to look up what
// the right answer was. Every readout now resolves through this policy.
//
// WHY THIS IS ITS OWN MODULE (and not part of `format.js`)
// `meta.js` needs these numbers to stamp `metricMeta[key].decimals`, and
// `format.js` needs them for the ad-hoc formatters (percent / speedup / µs).
// `format.js` already imports `meta.js`, so putting the policy in `format.js`
// would create a `meta -> format -> meta` cycle — the overwrite would then run
// while `metricMeta` is still uninitialised and silently stamp `undefined`.
// A leaf module with no imports of its own breaks the cycle while keeping a
// single source of truth.
// ============================================================================

/**
 * Four kinds of number, four precisions.
 *
 * `measurement` — a directly measured quantity with a unit: µs / ms / s / % / $,
 *   plus unbounded scores (STS). Two decimals, because these are the values a
 *   reader compares across entries and one decimal hides real gaps
 *   (`1.4%` vs `1.6%` is a 15% relative difference).
 *
 * `ratio` — a dimensionless multiplier referenced at 1.0× (s_lat, s_bmax, single
 *   speedups) AND the utilization fractions (Vec Ratio, AI Core Utilization).
 *   Three decimals is deliberate and load-bearing on both counts:
 *     • a speedup: `1.004×` and `1.000×` are NOT the same claim, and rounding to
 *       two would erase exactly the narrow gain the "fake optimisation" check
 *       exists to catch;
 *     • a utilization fraction: real values run down to `0.001`, so two decimals
 *       would render it `0.00` and destroy the diagnostic outright.
 *
 * `ratioDetail` — the headline ratio figures (S_equiv / EBGS / PAOS). One more
 *   digit than `ratio` because they are themselves the result of an aggregation
 *   and are quoted as evidence.
 *
 * `count` — an integer count (rounds, tool calls, profiling runs). Zero
 *   decimals: `42.00` rounds is false precision, not extra detail.
 */
export const PRECISION = {
  measurement: 2,
  ratio: 3,
  ratioDetail: 4,
  count: 0
};

/**
 * Machine-readable form of the policy, keyed by the `PRECISION` field name.
 *
 * Used by `scripts/audit-precision.mjs`, which asserts that the live policy
 * still matches this description and that no component has hand-rolled a
 * precision of its own.
 */
export const PRECISION_KINDS = ['measurement', 'ratio', 'ratioDetail', 'count'];
