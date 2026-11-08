// Benchmark dataset (v2 bundle) + leaderboard filter/selection state.
// Module-singleton reactive store: importing this file from any component
// shares the same state without a state-management library.
//
// v2.0 changes vs v1:
//   • ranking entity = entry (agent × engine_model × version) — C1
//   • rank metric = sts_token (perf_score 的落地) — C2
//   • agent / engine_model preset filters for the non-orthogonal axes — C1
//   • quality-cost dual mode (A: cost×STS, B: STS×PAOS quadrant) — C5
import { reactive, computed } from 'vue';
import { loadBundle } from '../services/api';
import { metricMeta, applyMetricCatalog } from '../lib/meta';
import { applyPricing } from '../lib/pricing';
import { mergeRosters } from '../lib/datasets';
import { summarizeEntries } from '../lib/summary';
import { isEntryEnabled } from './datasets';
import { uploadedEntries } from './uploads';
import tokenPrice from '../data/Token_price.json';
import { configurationDisplayName } from '../lib/displayNames';

/**
 * Price table in effect: the repo's own `src/data/Token_price.json`.
 *
 * Prices used to be importable through the Imports page (persisted to
 * localStorage). That path is gone on purpose: the published artifact must show
 * the SAME cost numbers for everyone, and a price table living in one reviewer's
 * browser made that impossible to guarantee. Editing the JSON is now the single
 * way to change a price.
 *
 * Models without a verifiable price keep `null` in that file, and
 * `applyPricing` leaves their `api_cost_usd` null, so the UI shows "–" rather
 * than a number nobody can source.
 */
export const effectivePricing = computed(() => tokenPrice ?? { currency: 'USD', models: {} });

// Startup params may arrive via location.search or the hash query string
// (hash router), e.g. #/?export=1&theme=dark — merge both.
export function readStartupParams() {
  const merged = new URLSearchParams(window.location.search);
  const hash = window.location.hash;
  const hashQuery = hash.includes('?') ? hash.slice(hash.indexOf('?') + 1) : '';
  new URLSearchParams(hashQuery).forEach((value, key) => merged.set(key, value));
  return merged;
}

export const benchmark = reactive({
  data: null, // { meta, entries } — v2 bundle
  source: 'internal-datasets', // default board; see src/services/api.js
  loaded: false,

  rankMetric: 'sts_token',
  cohort: 'harness',
  preset: 'free', // 'free' | 'by-agent' | 'by-model'  (嵌套混淆筛选预设, C1)
  agentId: 'all',
  engineModelId: 'all',
  openness: 'all',
  search: '',

  scatterX: 'tokens', // 'tokens' | 'cost'

  heatMetric: 'tokens_k', // tokens_k | rounds | wall_time_s

  gainEntry: null,
  // Deep-view selections for the parked Duration / Hardware sections. Their
  // registry entries were removed from lib/sections.js, so nothing renders them
  // today; the state and the matching computeds (`selectedDuration`,
  // `utilSelection`) are kept so those components can be switched back on
  // without re-plumbing the store.
  durationEntry: null,
  durationOperator: null,
  durationView: 'absolute', // 'absolute' | 'normalized'
  utilEntry: null,
  utilOperator: null
});

/**
 * Load the benchmark bundle once. Both pages call this on mount, so a second
 * call must not re-fetch and clobber the store with a *different* bundle.
 */
export async function initBenchmark({ force = false } = {}) {
  if (benchmark.loaded && !force) return { data: benchmark.data, source: benchmark.source };
  const { data, source } = await loadBundle();
  // Overlay the evaluator's metric catalog so labels/definitions stay in sync
  // with the backend contract instead of being maintained twice.
  applyMetricCatalog(data.meta?.catalog);
  benchmark.data = data;
  benchmark.source = source;
  benchmark.loaded = true;
  // Default deep-view selections: strongest visible entry per section.
  const entries = data.entries;
  benchmark.gainEntry = entries[0]?.id ?? null;
  // Parked Duration / Hardware defaults — harmless while those sections are
  // unregistered, and ready when they come back.
  benchmark.durationEntry = entries[0]?.id ?? null;
  benchmark.durationOperator = entries[0]?.per_operator[0]?.operator ?? null;
  benchmark.utilEntry = entries[0]?.id ?? null;
  benchmark.utilOperator = entries[0]?.per_operator[0]?.operator ?? null;
  return { data, source };
}

// ------------------------------ value access -------------------------------

/** Dotted path lookup, e.g. "paos.paos" → entry.overall.paos.paos. */
export function readPath(object, path) {
  if (object == null) return null;
  return String(path).split('.').reduce((current, part) => (current == null ? null : current[part]), object);
}

/** Numeric rank value for a metric key; non-numeric entries sort last. */
export function entryMetric(entry, key) {
  const raw = readPath(entry.overall, key);
  const numeric = typeof raw === 'number' ? raw : Number(raw);
  if (raw == null || raw === '' || !Number.isFinite(numeric)) return -Infinity;
  return numeric;
}

// ------------------------------ entries -------------------------------------

/**
 * The shipped roster merged with the reviewer's own imports.
 *
 * Shipped entries win an id collision (see `mergeRosters`), so a local file can
 * never change what everyone else sees at the same URL — it only adds entries
 * the project does not have.
 */
export const mergedRoster = computed(() => mergeRosters(
  { entries: benchmark.data?.entries ?? [], files: benchmark.data?.meta?.source_files ?? [] },
  uploadedEntries.value
));

/** Just the reviewer-supplied entries, for the Imports panel's own listing. */
export const importedEntries = computed(() => (
  mergedRoster.value.entries.filter(entry => entry.origin === 'imported')
));

/**
 * Every entry the UI can show = the merged roster with the active price table
 * applied.
 *
 * Entries switched off in the Imports page are excluded here, which is what
 * makes the selection effective everywhere: the leaderboard, the hero counters
 * and every chart all read from this one computed.
 */
export const allEntries = computed(() => applyPricing(
  mergedRoster.value.entries.filter(entry => isEntryEnabled(entry.id) && (!entry.cohort || entry.cohort === benchmark.cohort)),
  effectivePricing.value
));

/** The full roster including entries switched off (shipped + imported). */
export const rosterEntries = computed(() => mergedRoster.value.entries);

/**
 * Bundle `meta` with the rollups recomputed over the *enabled* subset.
 *
 * The bundle's own `meta` describes everything the folder holds, so reading it
 * directly would contradict the filtered board (hero saying "28 entries" above a
 * table showing 5). Re-deriving the rollup with `summarizeEntries()` — the same
 * helper the build script uses — keeps the two consistent by construction.
 *
 * Fields that are not counts (chips, catalog, metric_directions) pass through.
 */
export const boardMeta = computed(() => {
  const base = benchmark.data?.meta;
  if (!base) return null;
  const enabled = allEntries.value;
  const { operators: ops, agents: agentRoster, runsTotal, tokensTotalK, scenarios, tiers, modelClusters } = summarizeEntries(enabled);
  const importedCount = enabled.filter(entry => entry.origin === 'imported').length;
  return {
    ...base,
    entries_evaluated: enabled.length,
    agents: agentRoster,
    operators: ops,
    model_clusters: modelClusters,
    runs_total: runsTotal,
    tokens_total_k: tokensTotalK,
    targets: tiers,
    // Headline provenance. `entries_imported` counts only what is ON the board,
    // so the hero note and the Imports panel agree even with some entries off.
    entries_imported: importedCount,
    source_files_built_in: mergedRoster.value.sources.builtIn,
    source_files_imported: mergedRoster.value.sources.imported,
    dataset_problems: mergedRoster.value.problems,
    // A single scenario is only meaningful when every enabled entry agrees.
    scenario: scenarios.length === 1 ? enabled[0]?.scenario ?? null : null
  };
});

// ------------------------------ filtering ----------------------------------

export const agents = computed(() => {
  const map = new Map();
  allEntries.value.forEach(entry => { if (entry.agent?.id) map.set(entry.agent.id, entry.agent); });
  return [...map.values()];
});
export const engineModels = computed(() => {
  const map = new Map();
  allEntries.value.forEach(entry => { if (entry.engine_model?.id) map.set(entry.engine_model.id, entry.engine_model); });
  return [...map.values()];
});

function matchesSearch(entry, term) {
  const haystack = `${entry.agent.id} ${entry.agent.strategy} ${entry.engine_model.id} ${entry.engine_model.vendor} ${configurationDisplayName(entry)}`.toLowerCase();
  return haystack.includes(term);
}

export const visibleEntries = computed(() => {
  if (!benchmark.loaded) return [];
  const term = benchmark.search;
  const filtered = allEntries.value.filter(entry => {
    if (benchmark.preset === 'by-agent' && benchmark.agentId !== 'all' && entry.agent.id !== benchmark.agentId) return false;
    if (benchmark.preset === 'by-model' && benchmark.engineModelId !== 'all' && entry.engine_model.id !== benchmark.engineModelId) return false;
    if (benchmark.openness !== 'all' && entry.engine_model.openness !== benchmark.openness) return false;
    if (term && !matchesSearch(entry, term)) return false;
    return true;
  });
  const meta = metricMeta[benchmark.rankMetric] || metricMeta.sts_token;
  const dir = meta.direction === 'down' ? -1 : 1;
  return filtered.sort((a, b) => {
    const av = readPath(a.overall, benchmark.rankMetric), bv = readPath(b.overall, benchmark.rankMetric);
    if (!Number.isFinite(av)) return Number.isFinite(bv) ? 1 : 0;
    if (!Number.isFinite(bv)) return -1;
    return (bv - av) * dir;
  });
});

// Several panels fall back to the full roster when the filters exclude
// everything, so single-entry deep views stay usable.
export const fallbackEntries = computed(() => visibleEntries.value.length ? visibleEntries.value : allEntries.value);

export const gainSelection = computed(() => {
  const list = fallbackEntries.value;
  return list.find(entry => entry.id === benchmark.gainEntry) || list[0] || null;
});

/** Parked Duration section: entry × operator pair for the shape views. */
export const selectedDuration = computed(() => {
  const list = fallbackEntries.value;
  const entry = list.find(item => item.id === benchmark.durationEntry) || list[0];
  if (!entry) return { entry: null, operator: null };
  const operator = entry.per_operator.find(item => item.operator === benchmark.durationOperator) || entry.per_operator[0] || null;
  return { entry, operator };
});

/** Parked Hardware section: primary entry plus up to two comparison entries. */
export const utilSelection = computed(() => {
  const list = fallbackEntries.value;
  if (!list.length) return [];
  const primary = list.find(entry => entry.id === benchmark.utilEntry) || list[0];
  const others = list.filter(entry => entry.id !== primary.id && entry.per_operator.some(op => op.operator === benchmark.utilOperator));
  return [primary, ...others.slice(0, 2)].map(entry => ({
    entry,
    operator: entry.per_operator.find(op => op.operator === benchmark.utilOperator) || entry.per_operator[0]
  })).filter(pair => pair.operator);
});

/** Operators evaluated by at least one *enabled* entry, in canonical order. */
export const operators = computed(() => {
  const declared = boardMeta.value?.operators ?? [];
  const seen = new Set(declared);
  allEntries.value.forEach(entry => (entry.per_operator ?? []).forEach(op => {
    if (op?.operator && !seen.has(op.operator)) seen.add(op.operator);
  }));
  return [...seen];
});

/** Find one entry by id across built-in + imported data (drawer deep links). */
export function findEntryById(id) {
  if (!id) return null;
  return allEntries.value.find(entry => entry.id === id) ?? null;
}
