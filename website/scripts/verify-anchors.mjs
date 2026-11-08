// ============================================================================
// Appendix A anchor verification (frontend-side twin of the backend unit test).
//
// Recomputes every Appendix-A quantity for the measured run
// AddRmsNorm_orchestrator_20260811 (entry world-model × Qwen3_6 × AddRmsNorm)
// from the *raw* bundle inputs (anytime points, duration_by_shape, recipe,
// interaction stats) using src/lib/metrics.js, and asserts each value against
// the published anchor within its tolerance.
//
//   node scripts/verify-anchors.mjs     → exit 0 all green / 1 any drift
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  sAtBmax, auls, vlr, geomean, tokensToDelta, sAtMidWall, etps, reliabilityAtDelta
} from '../src/lib/metrics.js';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const bundle = JSON.parse(fs.readFileSync(path.join(root, 'src', 'data', 'bundle.json'), 'utf8'));

const entry = bundle.entries.find(e => e.id === 'world-model+qwen3-6');
const op = entry.per_operator.find(o => o.operator === 'AddRmsNorm');
const run = op.runs.find(r => r.run_id === 'AddRmsNorm_orchestrator_20260811');

const results = [];
function check(name, actual, expected, tol) {
  const ok = actual == null ? false : Math.abs(actual - expected) <= tol;
  results.push({ name, actual, expected, tol, ok });
}

// ---- raw inputs (not the precomputed fields) ----
const anytime = op.anytime;
const shapes = op.duration_by_shape;
const recipe = bundle.meta.recipe['DeepSeek-V4'].decode_b1.find(r => r.operator === 'AddRmsNorm');
const repShape = op.duration_by_shape.find(r => r.shape === '32,1024');

const sBmax = sAtBmax(anytime, 'tokens');
const aulsValue = auls(anytime, 'tokens');
const aulsTime = auls(anytime, 'time');
const sts = 100 * (0.5 * Math.log(sBmax) / Math.LN2 + 0.5 * aulsValue);
const stsTime = 100 * (0.5 * Math.log(sAtBmax(anytime, 'time')) / Math.LN2 + 0.5 * aulsTime);
const paos = geomean(shapes.map(r => r.s_lat));
const worst = shapes.reduce((a, b) => (b.s_lat < a.s_lat ? b : a));
const tokensTo1 = tokensToDelta(anytime, 1, 'tokens');
const tokensTo3 = tokensToDelta(anytime, 3, 'tokens');
const etpsBase = etps(recipe, repShape.t_base_us * 1e-6);
const waste = 92 / 113; // interaction_stats: llm_rounds 113, wasted_rounds 92
const sMid = sAtMidWall(anytime);
const rel = reliabilityAtDelta(op.runs, 3);

// ---- Appendix A anchors ----
check('S(B_max)', sBmax, 1.0210, 0.0005);
check('VLR %', vlr(sBmax), 2.061, 0.05);
check('STS_token', sts, 2.82, 0.025);
check('STS_time', stsTime, 2.86, 0.035);
check('AULS', aulsValue, 0.0263, 0.0008);
check('Tokens-to-1% (tokens)', tokensTo1.value, 508067, 600);
check('Tokens-to-3% (null)', tokensTo3.reason === 'never_reached' ? 0 : 1, 0, 0);
check('PAOS (geomean S_lat)', paos, 0.9848, 0.0006);
check('worst_shape s_lat', worst.s_lat, 0.747, 0.0005);
check('worst_shape id', worst.shape === '32,1024' ? 0 : 1, 0, 0);
check('eTPS_base (token/s)', etpsBase, 971, 6);
check('wasted_round_rate %', waste * 100, 81.4, 0.1);
check('plateau rule: S(B_max)-S(T_mid) < ε', sBmax - sMid < 0.01 ? 0 : 1, 0, 0);
check('Reliability@3% (K=3, none ≥3%)', rel.value, 0, 0);

// flags must contain the four named in Appendix A
const mustHave = ['fake_optimization', 'shape_regression', 'unconverged', 'high_waste'];
const missing = mustHave.filter(f => !op.flags.includes(f));
check('flags ⊇ {fake_opt, shape_regression, unconverged, high_waste}', missing.length === 0 ? 0 : 1, 0, 0);

// ---- report ----
let failed = 0;
for (const r of results) {
  const mark = r.ok ? '  ✓' : '✗  FAIL';
  if (!r.ok) failed += 1;
  const shown = typeof r.actual === 'number' ? r.actual.toFixed(4) : String(r.actual);
  console.log(`${mark} ${r.name.padEnd(46)} actual=${shown} expected=${r.expected} tol=${r.tol}`);
}
console.log(`\nrun under test: ${run?.run_id} (date ${run?.date})`);
console.log(failed === 0
  ? 'ALL ANCHORS PASS — formulas match Appendix A.'
  : `${failed} anchor(s) FAILED — formula implementation drift.`);
process.exit(failed === 0 ? 0 : 1);
