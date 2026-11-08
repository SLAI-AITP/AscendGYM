// Numeric and color formatting helpers — v2.0 display rules (§7.1/§7.2).
//
// Every default precision below comes from `lib/precision.js`. Do not inline a
// decimal count here or in a component: a literal that disagrees with the
// policy is exactly how the same quantity ended up rendering two ways.
import { metricMeta } from './meta';
import { PRECISION } from './precision';

export function signed(value, decimals = PRECISION.measurement) { return `${value >= 0 ? '+' : ''}${Number(value).toFixed(decimals)}`; }

export function formatInteger(value) { return Number.isFinite(value) ? Math.round(value).toLocaleString('en-US') : '–'; }

export function formatNumber(value, decimals = PRECISION.measurement) { return Number(value).toFixed(decimals); }

export function formatLogTick(value) { return value < 1 ? value.toString() : formatInteger(value); }

export function formatTokenTick(value) { return value >= 1000000 ? `${value / 1000000}M` : `${Math.round(value / 1000)}K`; }

/** §7.1 — speedup ratios: `1.021×` (three decimals — see `lib/precision.js`). */
export function formatSpeedup(value, decimals = PRECISION.ratio) {
  if (value == null || !Number.isFinite(value)) return '–';
  return `${Number(value).toFixed(decimals)}×`;
}

/**
 * §7.1 — headline ratios (S_equiv / EBGS / PAOS): the four-decimal tier.
 *
 * Named separately from `formatSpeedup` so a call site states which tier it
 * wants instead of passing a bare `4` that no reviewer can interpret.
 */
export function formatSpeedupDetail(value) {
  return formatSpeedup(value, PRECISION.ratioDetail);
}

/**
 * §7.1 — a percentage already expressed on the evaluator's 0–100 scale
 * (`vlr_mean`, `gate_rate`, `task_pass_rate`): `2.13%`.
 *
 * Does NOT multiply by 100 — the bundle stores these as `2.13`, not `0.0213`.
 * For a 0–1 fraction use `formatFractionPct`.
 */
export function formatPct(value, decimals = PRECISION.measurement) {
  if (value == null || !Number.isFinite(value)) return '–';
  return `${Number(value).toFixed(decimals)}%`;
}

/**
 * A 0–1 fraction rendered as a percentage (`wasted_round_rate`, verification
 * pass rate): `0.5` → `50.00%`.
 *
 * The two percentage helpers exist side by side because mixing them up is a
 * silent 100× error, and the bundle genuinely carries both conventions.
 */
export function formatFractionPct(value, decimals = PRECISION.measurement) {
  if (value == null || !Number.isFinite(value)) return '–';
  return `${(Number(value) * 100).toFixed(decimals)}%`;
}

/**
 * A signed 0–100 percentage: `+2.13%` / `-1.50%`.
 *
 * The sign is the signal — a negative VLR is a regression, and dropping the
 * `+` would make a gain and a loss read the same at a glance.
 */
export function formatSignedPct(value, decimals = PRECISION.measurement) {
  if (value == null || !Number.isFinite(value)) return '–';
  return `${value >= 0 ? '+' : ''}${Number(value).toFixed(decimals)}%`;
}

/** §7.1 — microsecond durations: `174.62 µs`. */
export function formatMicros(value, decimals = PRECISION.measurement) {
  if (value == null || !Number.isFinite(value)) return '–';
  return `${Number(value).toFixed(decimals)} µs`;
}

/** §7.1 — millisecond durations: `2.31 ms`. */
export function formatMillis(value, decimals = PRECISION.measurement) {
  if (value == null || !Number.isFinite(value)) return '–';
  return `${Number(value).toFixed(decimals)} ms`;
}

/** §7.1 — dollar amounts: `$1.42`. */
export function formatUsd(value, decimals = PRECISION.measurement) {
  if (value == null || !Number.isFinite(value)) return '–';
  return `$${Number(value).toFixed(decimals)}`;
}

/**
 * A bare measurement with no unit of its own — eTPS, verification CV and other
 * floats the drawer prints directly.
 */
export function formatMeasurement(value, decimals = PRECISION.measurement) {
  if (value == null || !Number.isFinite(value)) return '–';
  return Number(value).toFixed(decimals);
}

/** A bare ratio, `–` when absent. Kept for symmetry with `formatMeasurement`. */
export function formatRatio(value, decimals = PRECISION.ratio) {
  if (value == null || !Number.isFinite(value)) return '–';
  return Number(value).toFixed(decimals);
}

/** §7.1 — token volumes: `6.78M` / `508K` (input in thousands of tokens). */
export function formatTokensK(k) {
  if (k == null || !Number.isFinite(k)) return '–';
  if (k >= 1000) return `${(k / 1000).toFixed(PRECISION.measurement)}M`;
  return `${Math.round(k)}K`;
}

/**
 * Byte sizes for the import panel: `820 B` / `41.35 kB` / `1.13 MB`.
 *
 * Lives here rather than in the component because it is a number a reader sees,
 * so its decimal count must come from the policy — a hand-rolled `.toFixed(1)`
 * would be a second, unaudited precision rule. Bytes are exact integers, so the
 * sub-KB tier prints without decimals.
 */
export function formatBytes(bytes) {
  if (bytes == null || !Number.isFinite(bytes)) return '–';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(PRECISION.measurement)} kB`;
  return `${(bytes / 1024 / 1024).toFixed(PRECISION.measurement)} MB`;
}

/**
 * §7.1 — wall time: under one hour → seconds, otherwise hours. Both tiers use
 * the measurement precision (see `lib/precision.js`).
 */
export function formatWallTime(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '–';
  if (seconds < 3600) return `${seconds.toFixed(PRECISION.measurement)}s`;
  return `${(seconds / 3600).toFixed(PRECISION.measurement)}h`;
}

/** Fixed-hour readout for tables whose wall-time column uses one unit. */
export function formatWallHours(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '–';
  return `${(seconds / 3600).toFixed(PRECISION.measurement)}h`;
}

/** §7.1 — STS scores: measurement precision, unbounded. */
export function formatSts(value) {
  if (value == null || !Number.isFinite(value)) return '–';
  return Number(value).toFixed(PRECISION.measurement);
}

/** §7.2 — color tone for speedup-style values: bad (<1) / good (≥1.05) / neutral. */
export function speedupTone(value) {
  if (value == null || !Number.isFinite(value)) return 'neutral';
  if (value < 1) return 'bad';
  if (value >= 1.05) return 'good';
  return 'neutral';
}

/** Class helper so templates stay terse: :class="speedupClass(v)". */
export function speedupClass(value) { return `tone-${speedupTone(value)}`; }

/**
 * PAOS is nested ({paos, pkai, ptei, s_lat_geo, ...}) on both entries and
 * operators. Read the score itself so callers never do arithmetic on the
 * object by mistake. Returns null when absent.
 */
export function paosScore(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const score = value?.paos;
  return Number.isFinite(score) ? score : null;
}

export function formatMetric(value, key, withUnit = true) {
  const meta = metricMeta[key] || { decimals: PRECISION.measurement, unit: '' };
  if (value == null || !Number.isFinite(value)) return '–';
  const prefix = key === 'api_cost_usd' ? '$' : '';
  const suffix = withUnit && meta.unit ? ` ${meta.unit}` : (meta.unit === '%' ? '%' : '');
  return `${prefix}${Number(value).toFixed(meta.decimals)}${suffix}`;
}

export function niceMax(value) {
  if (value <= 0) return 1;
  if (value <= 10) return Math.ceil(value / 2) * 2;
  if (value <= 100) return Math.ceil(value / 10) * 10;
  if (value <= 1000) return Math.ceil(value / 100) * 100;
  return Math.ceil(value / 1000) * 1000;
}

export function qualityColor(quality) {
  const bad = '#E8B77E';
  const neutral = '#ECEBE4';
  const good = '#BEEBE4';
  if (quality <= .5) return mixHex(bad, neutral, quality * 2);
  return mixHex(neutral, good, (quality - .5) * 2);
}

export function mixHex(a, b, amount) {
  const parse = hex => hex.match(/[a-f\d]{2}/gi).map(value => Number.parseInt(value, 16));
  const [ar, ag, ab] = parse(a);
  const [br, bg, bb] = parse(b);
  const channel = (from, to) => Math.round(from + (to - from) * amount).toString(16).padStart(2, '0');
  return `#${channel(ar, br)}${channel(ag, bg)}${channel(bb)}`;
}

export function escapeHTML(value) { return String(value).replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' }[character])); }

export function attributeEscape(value) { return String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
