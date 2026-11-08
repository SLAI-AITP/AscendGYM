// ============================================================================
// Internal evaluation datasets — parsing and bundle assembly.
//
// The UX project ships its own evaluation data in `src/data/eval_datasets_summary/`,
// one evaluator summary per file (same schema as `eval_opencode_summary.json`).
// Dropping a new `.json` there is the whole upload procedure — no code change and
// no in-app importer.
//
// This module is the runtime-agnostic half: it takes an already-parsed
// `{ path: json }` map and returns `{ entries, bundle, problems }`. The
// runtime-specific readers are:
//   • src/services/datasets.js          — browser, via `import.meta.glob`
//   • scripts/build-operator-bundle.mjs — Node, via fs
// so both agree on what a dataset is and how a broken file is reported.
//
// Keeping the folder *inside* the project makes the dashboard self-contained:
// `npm run build` emits a `dist/` that already carries the measured data, with
// no dependency on the parent repository.
// ============================================================================

import { looksLikeSummary, buildEntryFromSummary, summarizeEntries, MODEL_CATALOG, CHIPS, RECIPE } from './summary.js';

/** Folder holding the datasets, relative to the project root. */
export const DATASET_DIR = 'src/data/eval_datasets_summary';

/** Hardware / recipe labels. Presentation only — never mixed into metrics. */
export const DISPLAY = {
  chips: CHIPS,
  recipe: RECIPE,
  engineModel: { ...MODEL_CATALOG['Qwen3.5-122B'] }
};

/**
 * Directions keyed by the UI's metric keys.
 *
 * `looksLikeBundle()` in services/api.js requires this block, and both the rank
 * chart and the rubric read it to decide whether "higher" or "lower" is better.
 * It is a *contract* declaration, not a measurement, so it is written here once
 * rather than derived per file.
 */
export const METRIC_DIRECTIONS = {
  sts_token: 'up', s_equiv: 'up', ebgs: 'up', speedup_mean: 'up', vlr_mean: 'up',
  task_pass_rate: 'up', gate_rate: 'up', candidate_pass_rate: 'up',
  best_duration_us: 'down', baseline_duration_ms: 'down',
  tokens_total_k: 'down', rounds_median: 'down', wall_time_median_s: 'down',
  tool_calls_total: 'down', profiling_runs_median: 'down',
  api_cost_usd: 'down',
  'paos.paos': 'up', 'paos.pkai': 'up', 'paos.ptei': 'up'
};

/**
 * Files in the folder that are not datasets.
 *
 * `_template.json` is a copy-me starter. A leading underscore marks it as
 * non-data so a reader can open it without it polluting the leaderboard.
 */
export function isDatasetFile(filename) {
  const base = String(filename).split('/').pop() ?? '';
  return base.endsWith('.json') && !base.startsWith('_');
}

/**
 * Parse one raw file into an entry.
 * Returns a result object rather than throwing: one malformed file must not
 * take down the whole board.
 */
export function parseDataset(raw, label = '') {
  if (!looksLikeSummary(raw)) {
    return { ok: false, reason: 'Not an evaluator summary: the top-level runs array is missing' };
  }
  const entry = buildEntryFromSummary(raw, {});
  if (!entry) return { ok: false, reason: 'The summary could not be converted into an entry' };

  // Provenance: which file produced this entry, so the UI can name it.
  entry.source_file = label || null;
  entry.overall.sts_token = null;
  entry.overall.budget_gm = null;
  entry.overall.sts_token_reason = 'legacy_summary_requires_current_STS_recomputation';
  entry.sts_coverage = 0;
  for (const op of entry.per_operator) op.sts_token = null;
  return { ok: true, entry };
}

/**
 * Assemble a dashboard bundle from a set of entries.
 *
 * Mirrors the shape `scripts/build-operator-bundle.mjs` writes, because the UI
 * consumes one contract regardless of where the bundle came from. The rollup
 * numbers come from `summarizeEntries()` — the same helper the build script uses,
 * so a runtime-assembled board and a built one report identical totals.
 */
export function buildBundleFromEntries(entries, { files = [], problems = [], catalog = null } = {}) {
  const list = entries ?? [];
  const {
    operators, agents, runsTotal, tokensTotalK, scenarios, tiers, modelClusters
  } = summarizeEntries(list);

  return {
    meta: {
      status: 'measured',
      updated: new Date().toISOString().slice(0, 10),
      note: `Evaluation summaries from ${DATASET_DIR}; unavailable measurements remain null.`,
      entries_evaluated: list.length,
      agents,
      operators,
      chips: DISPLAY.chips,
      model_clusters: modelClusters,
      runs_total: runsTotal,
      tokens_total_k: tokensTotalK,
      recipe: DISPLAY.recipe,
      epsilon_default: null,
      targets: tiers,
      // A single scenario is only meaningful when every source agrees on it.
      scenario: scenarios.length === 1 ? list[0].scenario : null,
      source_files: files,
      dataset_problems: problems,
      catalog: catalog ?? { version: '1.0', categories: [], metrics: {} },
      metric_directions: METRIC_DIRECTIONS
    },
    entries: list
  };
}

/**
 * Convert a `{ path: parsedJson }` map into `{ entries, bundle, problems, files }`.
 *
 * Shared by the browser reader and the build script so both agree on what counts
 * as a dataset, which file wins a duplicated identity, and how a failure is
 * reported.
 */
export function datasetsFromMap(map, { dir = DATASET_DIR } = {}) {
  const entries = [];
  const problems = [];
  const files = [];
  const seen = new Set();
  let catalog = null;

  // Sorted for a deterministic board: folder iteration order is filesystem-dependent.
  for (const path of Object.keys(map).sort()) {
    const base = path.split('/').pop() ?? path;
    if (!isDatasetFile(base)) continue;
    files.push(base);

    const raw = map[path];
    const { ok, entry, reason } = parseDataset(raw, base);
    if (!ok) {
      problems.push(`${base}：${reason}`);
      continue;
    }

    // Two files declaring the same agent × model would double-count that
    // entry's runs and tokens in the headline totals, so the first wins and the
    // clash is surfaced rather than silently dropped.
    if (seen.has(entry.id)) {
      problems.push(`${base}：entry "${entry.id}" is provided by another file and was skipped`);
      continue;
    }
    seen.add(entry.id);
    entries.push(entry);

    // Metric semantics come from the evaluator itself; the first file that
    // carries a catalog wins (all are emitted by the same script and agree).
    if (!catalog && raw?.catalog && Object.keys(raw.catalog.metrics ?? {}).length) {
      catalog = raw.catalog;
    }
  }

  return {
    entries,
    files,
    problems,
    bundle: buildBundleFromEntries(entries, { files, problems, catalog }),
    dir
  };
}

/**
 * Merge the shipped roster with the reviewer's imported one.
 *
 * The shipped entries win an id collision, and the loser is NAMED. That is the
 * opposite of what `datasetsFromMap` does between two files in the same folder,
 * and deliberately so:
 *
 *   • inside the folder a clash means the project doubled up on one result;
 *   • here it means a reviewer re-ran something the project already ships, and
 *     the project's committed number is the reproducible one. A visitor cannot
 *     see which number came from where, so letting a local file win would mean
 *     the same URL shows different rankings on different machines.
 *
 * Imported entries are tagged `origin: 'imported'` so the UI can label them.
 * Shipped entries are tagged too, so a component never has to infer origin from
 * the absence of a field.
 *
 * Returns `{ entries, sources, importedIds, problems }`:
 *   • `entries`     — merged roster, shipped entries first
 *   • `sources`     — file names per origin, for the provenance lines
 *   • `importedIds` — ids that are reviewer-supplied (used for labelling/toggling)
 *   • `problems`    — rejected imported files plus collision notices
 */
export function mergeRosters(shipped, imported) {
  const builtInEntries = (shipped?.entries ?? []).map(entry => ({ ...entry, origin: 'built-in' }));
  const importedEntries = imported?.entries ?? [];
  const builtInIds = new Set(builtInEntries.map(entry => entry.id));

  const problems = [];
  const accepted = [];
  const seen = new Set(builtInIds);

  for (const entry of importedEntries) {
    if (seen.has(entry.id)) {
      problems.push(
        `${entry.source_file ?? entry.id}：entry "${entry.id}" duplicates an included identity，the included version takes precedence`
      );
      continue;
    }
    seen.add(entry.id);
    accepted.push({ ...entry, origin: 'imported' });
  }

  // Rejections recorded while parsing an imported file, prefixed so a reader can
  // tell a broken upload from a clash without opening the console.
  const parseProblems = (imported?.problems ?? []).map(text => text);

  return {
    entries: [...builtInEntries, ...accepted],
    importedIds: accepted.map(entry => entry.id),
    sources: {
      builtIn: shipped?.files ?? [],
      imported: imported?.files ?? []
    },
    problems: [...parseProblems, ...problems]
  };
}
