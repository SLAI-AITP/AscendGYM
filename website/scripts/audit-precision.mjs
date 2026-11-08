// Precision audit: does every displayed measurement resolve its decimal count
// from `src/lib/precision.js`?
//
//   node scripts/audit-precision.mjs
//
// WHY THIS EXISTS
// Precision used to be scattered across `meta.js` (a `decimals` field) and the
// components (bare `.toFixed(1)` / `.toFixed(3)` / `.toFixed(4)` calls). The
// same quantity could therefore render at two different precisions in two
// places, and nothing failed — it just looked slightly wrong. This audit turns
// that silent drift into a build failure.
//
// WHAT COUNTS AS A VIOLATION
// A `.toFixed(<literal>)` inside a component or a display module, EXCEPT where
// the result is geometry (SVG coordinates, dash arrays, CSS ramp values) rather
// than a number a reader sees. Those are whitelisted explicitly below, by file
// and by intent, so the exemption is a reviewed decision rather than an
// accident.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const srcDir = path.join(root, 'src');

/** Files whose `.toFixed` calls build SVG geometry, not readouts. */
const GEOMETRY_ONLY = new Map([
  ['components/Sparkline.vue', 'path `d` coordinates only'],
  ['components/ScatterChart.vue', 'point transform + path `d` coordinates'],
  ['components/EntryDrawer.vue', 'anytime chart coordinates + plot geometry'],
  ['components/RatioFingerprint.vue', 'arc geometry via svgArcPath'],
  ['components/RatioLanes.vue', 'lane bead coordinates'],
  ['components/ShapeRings.vue', 'stroke-dasharray ramp'],
  ['components/GainChart.vue', 'impulse trace geometry'],
  ['lib/geometry.js', 'polar/arc point maths'],
  ['lib/markup.js', 'vendor glyph baseline offset']
]);

/**
 * Geometry calls INSIDE a file that also renders real readouts.
 *
 * `RankChart.vue` cannot be whitelisted wholesale: it formats the value label
 * through the shared formatters AND rounds arc radii for the SVG. Exempting the
 * file would blind the audit to a genuinely hard-coded readout added later, so
 * the exemption is matched on the call's own tail instead.
 */
const GEOMETRY_CALL_PATTERNS = [
  // `<circle ... r="${…toFixed(1)}">` and `stroke-width` on the value halo:
  // SVG user-unit geometry, not a number a reader compares.
  /(?:^|[\s"])r="\$\{[^}]*\.toFixed\(\d+\)\}"/,
  /stroke-width:\$\{[^}]*\.toFixed\(\d+\)\}px/
];

// A `.toFixed(` call is a violation when its argument is a numeric literal and
// the line is not geometry. `precision.js` itself legitimately holds the
// numbers, so it is exempt by definition.
//
// These paths are compared against `rel`, which is normalised to forward
// slashes below — using `path.join` here would produce backslashes on Windows
// and silently never match, exempting nothing.
const PRECISION_MODULE = 'lib/precision.js';
const FORMAT_MODULE = 'lib/format.js';
const METRICS_MODULE = 'lib/metrics.js';

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(item => {
    const full = path.join(dir, item.name);
    if (item.isDirectory()) return walk(full);
    return /\.(vue|js)$/.test(item.name) ? [full] : [];
  });
}

const files = walk(srcDir);
const violations = [];
let scanned = 0;
let geometryCallExempt = 0;

for (const file of files) {
  const rel = path.relative(srcDir, file).split(path.sep).join('/');
  const source = fs.readFileSync(file, 'utf8');
  const lines = source.split(/\r?\n/);

  lines.forEach((line, index) => {
    // A literal-argument `toFixed`, e.g. `.toFixed(2)`. A `.toFixed(variable)`
    // is allowed: it means the caller passed a policy value through.
    if (!/\.toFixed\(\s*\d+\s*\)/.test(line)) return;

    // Prose mentioning `.toFixed(2)` is documentation, not a call. Without this
    // the policy module's own explanation of *why* it exists trips the check.
    if (/^\s*(\/\/|\*|\/\*)/.test(line)) return;

    scanned += 1;

    // The policy owner and the shared formatters ARE the policy — they are
    // where a literal is supposed to live. `lib/metrics.js` cannot import the
    // formatters (bare Node loads it) so it carries its own tiny pair, also
    // justified.
    if (rel === PRECISION_MODULE || rel === FORMAT_MODULE || rel === METRICS_MODULE) return;

    if (GEOMETRY_ONLY.has(rel)) return;

    // Geometry call inside a file that ALSO renders readouts: exempt only when
    // the call site itself is an SVG attribute (see GEOMETRY_CALL_PATTERNS).
    if (GEOMETRY_CALL_PATTERNS.some(pattern => pattern.test(line))) {
      geometryCallExempt += 1;
      return;
    }

    violations.push({ rel, line: index + 1, text: line.trim() });
  });
}

console.log(`\n=== Precision policy audit (${scanned} literal toFixed call(s) seen) ===`);
console.log(`Policy: ${path.join('src', 'lib', 'precision.js')}`);

if (violations.length) {
  console.log('');
  for (const { rel, line, text } of violations) {
    console.log(`  DEAD  ${rel}:${line}`);
    console.log(`        ${text}`);
  }
  console.log(`\n${violations.length} hard-coded precision(s) outside the policy.`);
  console.log('Use the shared formatter from lib/format.js, or add the file to');
  console.log('GEOMETRY_ONLY in scripts/audit-precision.mjs when the value is not a readout.');
  process.exit(1);
}

const geometry = [...GEOMETRY_ONLY.keys()];
console.log(`  ok    0 unmanaged literal toFixed outside lib/precision.js, lib/format.js`);
console.log(`  ok    ${geometryCallExempt} SVG-attribute geometry call(s) exempt inside mixed files`);
console.log(`  ok    ${geometry.length} file(s) exempt as geometry-only:`);
for (const key of geometry) console.log(`          ${key}`);
console.log('\nAll displayed precision resolves from the policy.');
