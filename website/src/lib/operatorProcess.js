// Operator Process display policy.
//
// Cost / effort fields can exist even when an agent never produced a useful
// kernel: the run still consumed tokens, rounds and wall time. Showing those
// values beside an empty token cell makes the heatmap look internally
// inconsistent, though. The customer-facing process view therefore exposes
// all four dimensions only when the operator has at least one correctness-
// gated candidate with a positive performance gain.

/** Did this operator produce a correctness-gated optimization above baseline? */
export function hasPositiveOptimization(operator) {
  return Number.isFinite(operator?.s_bmax) && operator.s_bmax > 1;
}

/** Resolve one heatmap value under the shared display policy. */
export function operatorProcessValue(operator, metric) {
  // Retain observed resource consumption even without a successful optimization.
  const value = operator?.[metric];
  return Number.isFinite(value) ? value : null;
}
