// Field audit: does every metric key the UI renders actually resolve in the
// real bundle? Run after changing a bundle, a component or lib/meta.js.
//
//   node scripts/audit-field-mapping.mjs
//
// Exits non-zero when a rendered key never resolves anywhere in the dataset,
// which means a column, heat cell or chart is permanently blank — a mapping
// bug, not an acceptable data gap.
//
// The resolvers below MIRROR how the components actually read the data. A key
// is not always a literal path: the rubric aggregates utilization across
// operators, and the leaderboard reads entry-level rollups while the per-operator
// table reads operator-level fields. Auditing literal paths alone reports false
// alarms and hides the real breaks.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const bundle = JSON.parse(fs.readFileSync(path.join(root, 'src', 'data', 'supplement_bundle.json'), 'utf8'));

/**
 * `src/lib/meta.js` uses extensionless imports (Vite resolves those, bare Node
 * does not), so key lists are extracted from the source text. That keeps this
 * audit runnable with plain `node` — no bundler, no build step.
 */
const metaSource = fs.readFileSync(path.join(root, 'src', 'lib', 'meta.js'), 'utf8');

function keysFromSource(name, kind = 'object') {
  const start = metaSource.indexOf(`export const ${name}`);
  if (start < 0) throw new Error(`audit: 找不到 export const ${name}`);
  const nextExport = metaSource.indexOf('\nexport ', start + 1);
  const body = metaSource.slice(start, nextExport < 0 ? undefined : nextExport);
  const keys = [];
  if (kind === 'array') {
    for (const match of body.matchAll(/\bkey:\s*'([^']+)'/g)) keys.push(match[1]);
    if (!keys.length) {
      for (const match of body.matchAll(/^\s*'([^']+)',?\s*$/gm)) keys.push(match[1]);
    }
  } else {
    for (const match of body.matchAll(/^ {2}([A-Za-z_][\w.]*):\s*\{/gm)) keys.push(match[1]);
  }
  return [...new Set(keys.filter(key => typeof key === 'string' && key))];
}

const rankMetricOptions = keysFromSource('rankMetricOptions', 'array');
const rubricColumns = keysFromSource('rubricColumns', 'array');
const processMeta = keysFromSource('processMeta', 'object');

const readPath = (object, dotted) =>
  String(dotted).split('.').reduce((current, part) => (current == null ? null : current[part]), object);

/**
 * Resolvers, one per real consumer. Each returns the values the UI would
 * actually paint, so "no values" means a genuinely blank view.
 */
const resolvers = {
  /** Leaderboard rank chart / table: entry-level rollup fields. */
  entryOverall: key => bundle.entries.map(entry => readPath(entry.overall, key)),

  /** Heatmap: literal field on each per-operator block. */
  operatorField: key => bundle.entries.flatMap(entry => entry.per_operator.map(op => op[key])),

  /**
   * MetricsDetailSection.value(): two columns are averaged across operators' nested
   * `utilization`, everything else comes from the entry rollup. This mirrors
   * that branch instead of assuming a flat path.
   */
  rubric: key => {
    if (key === 'vec_ratio_best' || key === 'u_best') {
      return bundle.entries.flatMap(entry => {
        const ops = entry.per_operator.filter(op => (op.baseline_quality ?? 'normal') === 'normal');
        const values = ops.map(op => op.utilization?.[key]).filter(Number.isFinite);
        return values.length ? [values.reduce((a, b) => a + b, 0) / values.length] : [];
      });
    }
    return bundle.entries.map(entry => readPath(entry.overall, key));
  }
};

const checks = [
  ['Re-rank options (entry rollup)', 'entryOverall', rankMetricOptions],
  ['Heatmap metrics (per operator)', 'operatorField', processMeta],
  ['Rubric columns (entry rollup + utilization aggregation)', 'rubric', rubricColumns]
];

const unavailable = new Set(['paos.paos', 'profiling_runs_median', 'vec_ratio_best', 'u_best']);
let dead = 0;
for (const [label, resolverName, keys] of checks) {
  const resolve = resolvers[resolverName];
  console.log(`\n=== ${label} ===`);
  for (const key of keys) {
    const values = resolve(key).filter(value => value != null && value !== '');
    if (!values.length && unavailable.has(key)) {
      console.log(`  unavailable ${key}: not exported in supplementary records`);
    } else if (!values.length) {
      dead += 1;
      console.log(`  DEAD  ${String(key).padEnd(24)} never resolves — permanently blank`);
    } else {
      console.log(`  ok    ${String(key).padEnd(24)} ${values.length} value(s)`);
    }
  }
}

console.log(`\n${dead ? `${dead} dead key(s) — fix lib/meta.js or the bundle mapping` : 'All rendered keys resolve.'}`);
process.exit(dead ? 1 : 0);
