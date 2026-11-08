// ============================================================================
// Static v2 bundle generator.
//
// Produces src/data/bundle.json implementing the v2.0 data contract (§6.1)
// as the offline fallback for GET /api/bundle.
//
// • world-model × Qwen3_6 × AddRmsNorm is the *measured* anchor
//   (AddRmsNorm_orchestrator_20260811) — raw values are transcribed from the
//   run artifacts; every derived field (STS/PAOS/flags) is recomputed with
//   src/lib/metrics.js so it matches Appendix A within rounding.
// • All other entries are deterministic synthetic data on the same contract
//   (clearly noted in meta.note) used to demonstrate the full UX surface.
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  deriveOperator, aggregateOverall, reliabilityAtDelta, vlr, round, sortFlags, computeFlags, sAtBmax, auls, etps
} from '../src/lib/metrics.js';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

// ------------------------------ seeded RNG ---------------------------------

function xmur3(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i += 1) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return (h ^= h >>> 16) >>> 0;
  };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rngFor = name => mulberry32(xmur3(name)());
const hex = (rng, len = 8) => Array.from({ length: len }, () => '0123456789abcdef'[Math.floor(rng() * 16)]).join('');

// ------------------------------ operator facts ------------------------------
// Baselines (L0) and shape tables are harness facts: identical across entries
// for the same operator. t_base_us = task duration of the baseline kernel.

const SHAPES = {
  AddRmsNorm: [
    { shape: '1,2,4,128,64', dtype: 'FP32', t_base: 6.98 },
    { shape: '2,16,32,64', dtype: 'FP32', t_base: 19.58 },
    { shape: '2,256,128', dtype: 'FP16', t_base: 7.5 },
    { shape: '2,64,128', dtype: 'BF16', t_base: 13.16 },
    { shape: '32,1024', dtype: 'BF16', t_base: 8.38 },
    { shape: '64,256', dtype: 'FP32', t_base: 5.24 }
  ],
  Swiglu: [
    { shape: '16,7168', dtype: 'BF16', t_base: 18.4 },
    { shape: '256,4096', dtype: 'BF16', t_base: 64.2 },
    { shape: '1024,7168', dtype: 'BF16', t_base: 121.8 },
    { shape: '4096,7168', dtype: 'FP16', t_base: 246.5 }
  ],
  SparseAttnSharedkv: [
    { shape: '1,1,64,1024,128', dtype: 'BF16', t_base: 148.2 },
    { shape: '1,1,256,2048,128', dtype: 'BF16', t_base: 289.6 },
    { shape: '1,1,1024,4096,128', dtype: 'FP16', t_base: 512.4 },
    { shape: '1,2,128,1024,128', dtype: 'BF16', t_base: 96.7 }
  ]
};

const OP_META = {
  AddRmsNorm: { bound: 'memory', baseline_ms: 0.1018, rep_shape: '32,1024', recipe: { G: 1, n_i: 123 } },
  Swiglu: { bound: 'memory', baseline_ms: 0.0642, rep_shape: '256,4096', recipe: { G: 1, n_i: 427 } },
  SparseAttnSharedkv: { bound: 'memory', baseline_ms: 0.2896, rep_shape: '1,1,256,2048,128', recipe: null }
};

const CYCLE_TITLES = [
  '基线向量化与分块 (tile & double buffer)',
  'Vector reduce 融合 (VADD→REDUCE)',
  'L2 分块尺寸与 cache 调优',
  'Broadcast 融合，减少 MTE2 读',
  '尾部处理与分支消除',
  'bf16×2 向量宽度对齐',
  '流水线重叠 (TPipe 双队列)',
  'L1 hit 策略与 tile 复用'
];

// ------------------------------ roster --------------------------------------

const ENGINE_MODELS = {
  'qwen3-6': { id: 'Qwen3_6', vendor: 'Alibaba', openness: 'open', vendor_mark: 'Q', color: '#0E9B8E' },
  'deepseek-v3-2': { id: 'DeepSeek-V3.2', vendor: 'DeepSeek', openness: 'open', vendor_mark: 'D', color: '#3F67D5' },
  'glm-5-2': { id: 'GLM-5.2', vendor: 'Zhipu', openness: 'open', vendor_mark: 'Z', color: '#8C58C6' },
  'gpt-5-6': { id: 'GPT-5.6', vendor: 'OpenAI', openness: 'closed', vendor_mark: 'G', color: '#E08A1E' },
  'claude-opus-4-8': { id: 'Claude-Opus-4.8', vendor: 'Anthropic', openness: 'closed', vendor_mark: 'C', color: '#C2554F' }
};

const AGENTS = {
  'world-model': { id: 'world-model', type: 'domain', strategy: 'world-model', version: 'v1' },
  'k-search': { id: 'k-search', type: 'domain', strategy: 'k-search-tree', version: 'v2' },
  'claude-code': { id: 'claude-code', type: 'general', strategy: 'general', version: '2.1' },
  'openhands': { id: 'openhands', type: 'general', strategy: 'general', version: '1.4' }
};

// --------------------------- shape factor helpers ----------------------------

function shapeRows(opName, factors) {
  return SHAPES[opName].map((row, i) => {
    const factor = factors[i] ?? 1;
    const tBest = round(row.t_base / factor, 2);
    return { shape: row.shape, dtype: row.dtype, t_base_us: row.t_base, t_best_us: tBest, s_lat: round(row.t_base / tBest, 3) };
  });
}

function utilization(opName, seed, uBase, uBest, vecBase, vecBest) {
  const rng = rngFor(`util:${seed}`);
  return {
    vec_ratio_base: round(vecBase, 3),
    vec_ratio_best: round(vecBest, 3),
    mac_ratio_base: null,
    mac_ratio_base_reason: 'zero_col_vector_op',
    mac_ratio_best: null,
    mac_ratio_best_reason: 'zero_col_vector_op',
    scalar_ratio_base: round(0.32 + rng() * 0.16, 3),
    scalar_ratio_best: round(0.30 + rng() * 0.16, 3),
    mte2_ratio_base: round(0.12 + rng() * 0.14, 3),
    mte2_ratio_best: round(0.12 + rng() * 0.16, 3),
    u_base: round(uBase, 3),
    u_best: round(uBest, 3),
    l2_bw: null,
    l2_bw_reason: 'not_collected',
    hbm_bw: null,
    hbm_bw_reason: 'not_collected',
    overlap: null,
    overlap_reason: 'not_collected'
  };
}

// ------------------------------ anytime builder -----------------------------

/**
 * Deterministic anytime trajectory for synthetic operators.
 * Returns {anytime, anytime_failed} — passed solutions form the
 * best-so-far curve that reaches `sTarget` at B_max.
 */
function genAnytime(seed, spec) {
  const { sTarget, tokensK, wallS, nSolutions, nCycles, baselineMs } = spec;
  const rng = rngFor(seed);
  const nPass = Math.max(3, Math.round(nSolutions * (0.30 + rng() * 0.12)));
  const nFail = nSolutions - nPass;

  // improvement schedule: m records at increasing levels, last = 1.0
  const nRecords = 3 + Math.floor(rng() * 2);
  const levels = Array.from({ length: nRecords }, (_, i) => (i === nRecords - 1 ? 1 : 0.3 + 0.68 * (i / (nRecords - 1)) * (0.8 + rng() * 0.4)));
  const last = levels.length - 1;
  levels[last] = 1;

  const frac = [];
  let f = 0;
  for (let j = 1; j <= nSolutions; j += 1) {
    f = Math.min(1, (j / nSolutions) * (0.9 + rng() * 0.2));
    frac.push(f);
  }

  const passIdx = [];
  for (let j = 0; j < nSolutions; j += 1) if (j % Math.ceil(nSolutions / nPass) === 0) passIdx.push(j);
  while (passIdx.length < nPass) passIdx.push(nSolutions - 1 - (passIdx.length - nPass + 1));
  const passSet = new Set(passIdx.slice(0, nPass));

  const sOf = j => {
    const record = Math.min(last, Math.floor((j / (nSolutions - 1)) * last + rng() * 0.25));
    return round(1 + (sTarget - 1) * levels[Math.max(0, Math.min(last, Math.round(record)))], 4);
  };

  const anytime = [{ cum_tokens_k: 0, wall_sec: 0, s: 1 }];
  const failed = [];
  let best = 1;
  for (let j = 0; j < nSolutions; j += 1) {
    const tk = round(tokensK * frac[j], 1);
    const ws = round(wallS * Math.min(1, frac[j] * (0.94 + rng() * 0.12)), 1);
    if (passSet.has(j)) {
      const s = Math.max(best, sOf(j));
      best = s;
      anytime.push({
        cum_tokens_k: tk,
        wall_sec: ws,
        s,
        solution_id: hex(rng),
        cycle_idx: 1 + Math.min(nCycles - 1, Math.floor((j / nSolutions) * nCycles))
      });
    } else {
      failed.push({
        cum_tokens_k: tk,
        wall_sec: ws,
        duration_ms: round(baselineMs * (0.86 + rng() * 0.3), 4),
        status: rng() < 0.42 ? 'compile_failed' : 'test_failed'
      });
    }
  }
  // force the final passed solution to land exactly on sTarget at B_max
  const tail = anytime[anytime.length - 1];
  tail.s = round(sTarget, 4);
  tail.cum_tokens_k = round(tokensK, 1);
  tail.wall_sec = round(wallS, 1);
  return { anytime, anytime_failed: failed };
}

// ------------------------------ runs builder --------------------------------

function genRuns(seed, { operator, agentId, sTarget, paosTarget, k, dates, termination, nCycles, shapes, baselineMs }) {
  const runs = [];
  for (let i = 0; i < k; i += 1) {
    const jitter = i === 0 ? 0 : (rngFor(`${seed}:run${i}:s`)() - 0.5) * 0.008;
    const s = i === 0 ? round(sTarget, 4) : round(sTarget * (1 + jitter), 4);
    const rows = shapes.map(row => {
      const f2 = i === 0 ? row.s_lat : round(row.s_lat * (1 + (rngFor(`${seed}:run${i}:f${row.shape}`)() - 0.5) * 0.012), 3);
      return { shape: row.shape, dtype: row.dtype, t_base_us: row.t_base_us, t_best_us: round(row.t_base_us / f2, 2), s_lat: f2 };
    });
    const paosRun = i === 0 ? round(paosTarget, 4) : round(paosTarget * (1 + (rngFor(`${seed}:run${i}:p`)() - 0.5) * 0.01), 4);
    const flags = sortFlags(computeFlags({
      sBmax: s,
      paos: paosRun,
      shapes: rows,
      terminationReason: termination,
      wastedRoundRate: 0.3,
      epsilon: 0.01
    }));
    const rng = rngFor(`${seed}:run${i}:cyc`);
    const nCyc = nCycles;
    const cycles = [];
    let acc = 0;
    for (let c = 0; c < nCyc; c += 1) {
      const share = 0.18 + rng() * 0.3;
      acc = Math.min(1, acc + share);
      cycles.push({
        idx: c + 1,
        action_title: CYCLE_TITLES[(c + i) % CYCLE_TITLES.length],
        attempts: 3 + Math.floor(rng() * 6),
        best_s: round(1 + (s - 1) * Math.pow(acc, 1.4), 4),
        tokens_k: round(acc * 100, 0)
      });
    }
    runs.push({
      run_id: `${operator}_${agentId}_${dates[i] ?? dates[0]}`,
      date: dates[i] ?? dates[0],
      s_bmax: s,
      paos: paosRun,
      flags,
      termination_reason: termination,
      cycles,
      best_solution_id: hex(rng),
      canonical_code_hash: hex(rng, 12),
      code_excerpt: codeExcerpt(operator)
    });
  }
  return runs;
}

function codeExcerpt(operator) {
  const fn = operator.toLowerCase().replace(/[^a-z0-9]+/g, '_');
  return [
    `// best_solution 392ea1f7 · canonical 9f3c1a2e (illustrative)`,
    `extern "C" __global__ __aicore__ void ${fn}(GM_ADDR in0, GM_ADDR in1, GM_ADDR out) {`,
    `  TPipe pipe;`,
    `  TQue<QuePosition::VECIN, 2> inQ;`,
    `  TQue<QuePosition::VECOUT, 2> outQ;`,
    `  pipe.InitBuffer(inQ, 2, TILE_BYTES);`,
    `  pipe.InitBuffer(outQ, 2, TILE_BYTES);`,
    `  // tile copy-in -> compute -> fused write-out`,
    `  CopyIn(); Compute(); WriteOut();`,
    `}`
  ].join('\n');
}

// --------------------------- E1 AddRmsNorm anchor ----------------------------
// Measured: Output/AddRmsNorm_orchestrator_20260811/{result.json,solutions.jsonl}
// + profiling/op_summary_*.csv  (Appendix A).

const ANCHOR_ANYTIME = [
  { cum_tokens_k: 0, wall_sec: 0, s: 1 },
  { cum_tokens_k: 508.067, wall_sec: 1900, s: 1.0172, solution_id: '392ea1f7', cycle_idx: 1 },
  { cum_tokens_k: 3012.1, wall_sec: 5200, s: 1.021, solution_id: '8b2c9d41', cycle_idx: 2 },
  { cum_tokens_k: 6062.5, wall_sec: 13841.6, s: 1.021, solution_id: 'c41a7e02', cycle_idx: 4 }
];

const ANCHOR_FAILED = [
  { cum_tokens_k: 233.4, wall_sec: 780, duration_ms: 0.1041, status: 'compile_failed' },
  { cum_tokens_k: 812.3, wall_sec: 2650, duration_ms: 0.101, status: 'test_failed' },
  { cum_tokens_k: 1147.8, wall_sec: 3720, duration_ms: 0.1062, status: 'compile_failed' },
  { cum_tokens_k: 1590.2, wall_sec: 4890, duration_ms: 0.0997, status: 'test_failed' },
  { cum_tokens_k: 1930.4, wall_sec: 6040, duration_ms: 0.1034, status: 'compile_failed' },
  { cum_tokens_k: 2481.6, wall_sec: 7420, duration_ms: 0.1008, status: 'test_failed' },
  { cum_tokens_k: 2874.9, wall_sec: 8610, duration_ms: 0.1049, status: 'compile_failed' },
  { cum_tokens_k: 3522.7, wall_sec: 10150, duration_ms: 0.0991, status: 'test_failed' },
  { cum_tokens_k: 4310.5, wall_sec: 11480, duration_ms: 0.1056, status: 'compile_failed' },
  { cum_tokens_k: 5204.3, wall_sec: 12720, duration_ms: 0.1022, status: 'test_failed' },
  { cum_tokens_k: 5771.9, wall_sec: 13540, duration_ms: 0.1037, status: 'compile_failed' },
  { cum_tokens_k: 6410.8, wall_sec: 14020, duration_ms: 0.1013, status: 'test_failed' }
];

const ANCHOR_SHAPES = shapeRows('AddRmsNorm', [1.017, 1.188, 0.987, 1.005, 0.747, 1.019]).map((row, i) => {
  // keep the measured t_best values from the contract sample exactly
  const exact = [6.86, 16.48, 7.6, 13.1, 11.22, 5.14][i];
  return { shape: row.shape, dtype: row.dtype, t_base_us: row.t_base_us, t_best_us: exact, s_lat: round(row.t_base_us / exact, 3) };
});

const ANCHOR_RUNS = [
  {
    run_id: 'AddRmsNorm_world-model_20260809',
    date: '2026-08-09',
    s_bmax: 1.019,
    paos: 0.9872,
    flags: sortFlags(['fake_optimization', 'shape_regression', 'unconverged', 'high_waste', 'plateau']),
    termination_reason: 'cycle_limit_reached',
    cycles: [
      { idx: 1, action_title: CYCLE_TITLES[0], attempts: 5, best_s: 1.0152, tokens_k: 512 },
      { idx: 2, action_title: CYCLE_TITLES[2], attempts: 6, best_s: 1.019, tokens_k: 2640 },
      { idx: 3, action_title: CYCLE_TITLES[3], attempts: 4, best_s: 1.019, tokens_k: 2210 },
      { idx: 4, action_title: CYCLE_TITLES[4], attempts: 3, best_s: 1.019, tokens_k: 1328 }
    ],
    best_solution_id: '77b0c2d4',
    canonical_code_hash: '5e21aa09c3f7',
    code_excerpt: codeExcerpt('AddRmsNorm')
  },
  {
    run_id: 'AddRmsNorm_orchestrator_20260811',
    date: '2026-08-11',
    s_bmax: 1.021,
    paos: 0.9848,
    flags: sortFlags(['fake_optimization', 'shape_regression', 'unconverged', 'high_waste', 'plateau']),
    termination_reason: 'cycle_limit_reached',
    cycles: [
      { idx: 1, action_title: CYCLE_TITLES[0], attempts: 5, best_s: 1.0172, tokens_k: 508 },
      { idx: 2, action_title: CYCLE_TITLES[1], attempts: 6, best_s: 1.021, tokens_k: 2504 },
      { idx: 3, action_title: CYCLE_TITLES[2], attempts: 4, best_s: 1.021, tokens_k: 2188 },
      { idx: 4, action_title: CYCLE_TITLES[6], attempts: 3, best_s: 1.021, tokens_k: 952 }
    ],
    best_solution_id: 'c41a7e02',
    canonical_code_hash: '9f3c1a2e77b1',
    code_excerpt: codeExcerpt('AddRmsNorm')
  },
  {
    run_id: 'AddRmsNorm_world-model_20260813',
    date: '2026-08-13',
    s_bmax: 1.0229,
    paos: 0.9811,
    flags: sortFlags(['fake_optimization', 'shape_regression', 'unconverged', 'high_waste', 'plateau']),
    termination_reason: 'cycle_limit_reached',
    cycles: [
      { idx: 1, action_title: CYCLE_TITLES[0], attempts: 5, best_s: 1.0181, tokens_k: 495 },
      { idx: 2, action_title: CYCLE_TITLES[3], attempts: 7, best_s: 1.0229, tokens_k: 2710 },
      { idx: 3, action_title: CYCLE_TITLES[2], attempts: 5, best_s: 1.0229, tokens_k: 2350 },
      { idx: 4, action_title: CYCLE_TITLES[7], attempts: 3, best_s: 1.0229, tokens_k: 880 }
    ],
    best_solution_id: 'd08f3b61',
    canonical_code_hash: 'b84d20ce11a9',
    code_excerpt: codeExcerpt('AddRmsNorm')
  }
];

// ------------------------------ operator spec -------------------------------

function opSpec(entryId, opName, spec) {
  const meta = OP_META[opName];
  const seed = `${entryId}:${opName}`;
  // Anchor operators carry a hand-transcribed measured shape table (factors
  // null) that buildEntry swaps in after this call.
  const shapes = spec.factors ? shapeRows(opName, spec.factors) : [];
  const anytimeBlock = spec.anytime
    ? { anytime: spec.anytime, anytime_failed: spec.anytime_failed ?? [] }
    : genAnytime(seed, {
      sTarget: spec.s_bmax, tokensK: spec.tokens_k, wallS: spec.wall_time_s,
      nSolutions: spec.n_solutions, nCycles: spec.n_cycles, baselineMs: meta.baseline_ms
    });
  const runs = spec.runs ?? genRuns(seed, {
    operator: opName,
    agentId: spec.agentId,
    sTarget: spec.s_bmax,
    paosTarget: spec.paos_target,
    k: spec.k,
    dates: spec.dates,
    termination: spec.termination,
    nCycles: spec.n_cycles,
    shapes,
    baselineMs: meta.baseline_ms
  });
  const op = {
    operator: opName,
    bound: meta.bound,
    baseline_ms: meta.baseline_ms,
    epsilon_used: 0.01,
    compile_pass: spec.compile_pass,
    tokens_k: spec.tokens_k,
    rounds: spec.rounds,
    wall_time_s: spec.wall_time_s,
    profiling_runs: spec.profiling_runs,
    wasted_round_rate: spec.waste,
    n_solutions: spec.n_solutions,
    n_cycles: spec.n_cycles,
    termination_reason: spec.termination,
    baseline_quality: spec.baseline_quality ?? 'normal',
    // Correctness signals named after the evaluator's accuracy block.
    // `gate_rate` is derived from the mock runs (share that produced S > 1);
    // this is prototype data, not a measurement.
    task_pass_rate: spec.func_pass_rate,
    candidate_pass_rate: spec.func_pass_rate,
    gate_rate: runs.length ? round((100 * runs.filter(run => (run.s_bmax ?? 0) > 1).length) / runs.length, 1) : null,
    recipe: meta.recipe ? { G: meta.recipe.G, n_i: meta.recipe.n_i, rep_shape: meta.rep_shape } : null,
    utilization: spec.utilization,
    duration_by_shape: shapes,
    ...anytimeBlock,
    runs
  };
  return op;
}

// ------------------------------ entries -------------------------------------

const E = {
  'world-model+qwen3-6': {
    agent: 'world-model', model: 'qwen3-6', kGroup: 3,
    ops: [
      {
        name: 'AddRmsNorm', agentId: 'world-model',
        // ---- measured anchor (Appendix A) ----
        anytime: ANCHOR_ANYTIME, anytime_failed: ANCHOR_FAILED,
        runs: ANCHOR_RUNS,
        factors: null,
        baseline_quality: 'normal',
        compile_pass: 100, func_pass_rate: 100,
        tokens_k: 6780, rounds: 113, wall_time_s: 13841.6, profiling_runs: 22,
        waste: round(92 / 113, 3), n_solutions: 22, n_cycles: 4,
        termination: 'cycle_limit_reached',
        paos_target: 0.9848, k: 3,
        dates: ['2026-08-09', '2026-08-11', '2026-08-13'],
        utilization: utilization('AddRmsNorm', 'wm:qwen3-6:AR', 0.709, 0.73, 0.064, 0.059)
      },
      {
        name: 'Swiglu', agentId: 'world-model',
        factors: [1.032, 1.011, 1.006, 1.014], s_bmax: 1.031, paos_target: 1.019,
        baseline_quality: 'normal',
        compile_pass: 100, func_pass_rate: 66.7,
        tokens_k: 910, rounds: 21, wall_time_s: 1892.4, profiling_runs: 6,
        waste: 0.38, n_solutions: 9, n_cycles: 2, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-14'],
        utilization: utilization('Swiglu', 'wm:qwen3-6:SW', 0.612, 0.641, 0.071, 0.077)
      }
    ]
  },
  'k-search+qwen3-6': {
    agent: 'k-search', model: 'qwen3-6', kGroup: 3,
    ops: [
      {
        name: 'AddRmsNorm', agentId: 'k-search',
        factors: [1.052, 1.031, 1.044, 1.029, 1.036, 1.021], s_bmax: 1.052, paos_target: 1.038,
        baseline_quality: 'normal',
        compile_pass: 94.1, func_pass_rate: 82.4,
        tokens_k: 5240, rounds: 84, wall_time_s: 9412.8, profiling_runs: 14,
        waste: 0.31, n_solutions: 17, n_cycles: 5, termination: 'improvement_stalled',
        k: 3, dates: ['2026-08-10', '2026-08-12', '2026-08-14'],
        utilization: utilization('AddRmsNorm', 'ks:qwen3-6:AR', 0.709, 0.744, 0.064, 0.069)
      },
      {
        name: 'Swiglu', agentId: 'k-search',
        factors: [1.062, 1.048, 1.041, 1.056], s_bmax: 1.084, paos_target: 1.051,
        baseline_quality: 'normal',
        compile_pass: 100, func_pass_rate: 83.3,
        tokens_k: 1820, rounds: 33, wall_time_s: 3120.5, profiling_runs: 8,
        waste: 0.24, n_solutions: 12, n_cycles: 3, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-15'],
        utilization: utilization('Swiglu', 'ks:qwen3-6:SW', 0.612, 0.663, 0.071, 0.081)
      }
    ]
  },
  'claude-code+gpt-5-6': {
    agent: 'claude-code', model: 'gpt-5-6', kGroup: 1,
    ops: [
      {
        name: 'AddRmsNorm', agentId: 'claude-code',
        factors: [1.008, 0.997, 1.004, 1.001, 1.006, 0.999], s_bmax: 1.014, paos_target: 1.002,
        baseline_quality: 'normal',
        compile_pass: 86.7, func_pass_rate: 60,
        tokens_k: 4180, rounds: 76, wall_time_s: 8204.3, profiling_runs: 12,
        waste: 0.44, n_solutions: 15, n_cycles: 4, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-12'],
        utilization: utilization('AddRmsNorm', 'cc:gpt:AR', 0.709, 0.718, 0.064, 0.062)
      },
      {
        name: 'Swiglu', agentId: 'claude-code',
        factors: [1.031, 0.968, 0.989, 1.012], s_bmax: 1.022, paos_target: 0.9997,
        baseline_quality: 'normal',
        compile_pass: 88.9, func_pass_rate: 55.6,
        tokens_k: 5230, rounds: 91, wall_time_s: 10233.7, profiling_runs: 13,
        waste: 0.49, n_solutions: 18, n_cycles: 4, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-13'],
        utilization: utilization('Swiglu', 'cc:gpt:SW', 0.612, 0.622, 0.071, 0.07)
      },
      {
        name: 'SparseAttnSharedkv', agentId: 'claude-code',
        factors: [1.019, 1.004, 1.013, 1.011], s_bmax: 1.008, paos_target: 1.011,
        baseline_quality: 'normal',
        compile_pass: 84.6, func_pass_rate: 61.5,
        tokens_k: 3860, rounds: 64, wall_time_s: 7112.9, profiling_runs: 11,
        waste: 0.41, n_solutions: 13, n_cycles: 4, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-14'],
        utilization: utilization('SparseAttnSharedkv', 'cc:gpt:SA', 0.534, 0.551, 0.082, 0.086)
      }
    ]
  },
  'claude-code+deepseek-v3-2': {
    agent: 'claude-code', model: 'deepseek-v3-2', kGroup: 1,
    ops: [
      {
        name: 'AddRmsNorm', agentId: 'claude-code',
        factors: [1.012, 1.001, 1.008, 1.003, 1.005, 1.008], s_bmax: 1.019, paos_target: 1.006,
        baseline_quality: 'normal',
        compile_pass: 91.7, func_pass_rate: 66.7,
        tokens_k: 3320, rounds: 58, wall_time_s: 5961.2, profiling_runs: 9,
        waste: 0.36, n_solutions: 12, n_cycles: 3, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-12'],
        utilization: utilization('AddRmsNorm', 'cc:ds:AR', 0.709, 0.716, 0.064, 0.063)
      },
      {
        // Swiglu pathological baseline (saturated): raw 4× inflation,
        // excluded from cross-operator aggregation — the v2.0 red line.
        name: 'Swiglu', agentId: 'claude-code',
        factors: [4.12, 4.02, 4.08, 3.95], s_bmax: 4.1, paos_target: 4.04,
        baseline_quality: 'saturated',
        compile_pass: 100, func_pass_rate: 100,
        tokens_k: 2910, rounds: 47, wall_time_s: 4102.6, profiling_runs: 5,
        waste: 0.58, n_solutions: 10, n_cycles: 3, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-13'],
        utilization: utilization('Swiglu', 'cc:ds:SW', 0.612, 0.615, 0.071, 0.071)
      }
    ]
  },
  'openhands+glm-5-2': {
    agent: 'openhands', model: 'glm-5-2', kGroup: 1,
    ops: [
      {
        name: 'AddRmsNorm', agentId: 'openhands',
        factors: [0.999, 0.986, 0.995, 0.991, 0.997, 0.99], s_bmax: 1.009, paos_target: 0.993,
        baseline_quality: 'normal',
        compile_pass: 81.8, func_pass_rate: 54.5,
        tokens_k: 2640, rounds: 44, wall_time_s: 4890.1, profiling_runs: 7,
        waste: 0.52, n_solutions: 11, n_cycles: 3, termination: 'cycle_limit_reached',
        k: 1, dates: ['2026-08-15'],
        utilization: utilization('AddRmsNorm', 'oh:glm:AR', 0.709, 0.711, 0.064, 0.061)
      }
    ]
  },
  'world-model+deepseek-v3-2': {
    agent: 'world-model', model: 'deepseek-v3-2', kGroup: 1,
    ops: [
      {
        name: 'AddRmsNorm', agentId: 'world-model',
        factors: [1.021, 1.008, 1.016, 1.011, 1.013, 1.012], s_bmax: 1.026, paos_target: 1.014,
        baseline_quality: 'normal',
        compile_pass: 93.8, func_pass_rate: 75,
        tokens_k: 4720, rounds: 79, wall_time_s: 8931.4, profiling_runs: 13,
        waste: 0.29, n_solutions: 16, n_cycles: 4, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-14'],
        utilization: utilization('AddRmsNorm', 'wm:ds:AR', 0.709, 0.733, 0.064, 0.067)
      },
      {
        name: 'Swiglu', agentId: 'world-model',
        factors: [1.031, 1.017, 1.024, 1.018], s_bmax: 1.041, paos_target: 1.022,
        baseline_quality: 'normal',
        compile_pass: 94.7, func_pass_rate: 68.4,
        tokens_k: 6110, rounds: 96, wall_time_s: 10488.2, profiling_runs: 15,
        waste: 0.33, n_solutions: 19, n_cycles: 4, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-15'],
        utilization: utilization('Swiglu', 'wm:ds:SW', 0.612, 0.635, 0.071, 0.074)
      }
    ]
  },
  'k-search+gpt-5-6': {
    agent: 'k-search', model: 'gpt-5-6', kGroup: 1,
    ops: [
      {
        name: 'AddRmsNorm', agentId: 'k-search',
        factors: [1.041, 1.022, 1.033, 1.026, 1.031, 1.024], s_bmax: 1.047, paos_target: 1.029,
        baseline_quality: 'normal',
        compile_pass: 94.4, func_pass_rate: 77.8,
        tokens_k: 5930, rounds: 92, wall_time_s: 10112.7, profiling_runs: 16,
        waste: 0.35, n_solutions: 18, n_cycles: 5, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-13'],
        utilization: utilization('AddRmsNorm', 'ks:gpt:AR', 0.709, 0.739, 0.064, 0.068)
      },
      {
        name: 'SparseAttnSharedkv', agentId: 'k-search',
        factors: [1.052, 1.034, 1.044, 1.038], s_bmax: 1.033, paos_target: 1.041,
        baseline_quality: 'normal',
        compile_pass: 92.9, func_pass_rate: 71.4,
        tokens_k: 4310, rounds: 71, wall_time_s: 7801.3, profiling_runs: 10,
        waste: 0.27, n_solutions: 14, n_cycles: 4, termination: 'improvement_stalled',
        k: 1, dates: ['2026-08-15'],
        utilization: utilization('SparseAttnSharedkv', 'ks:gpt:SA', 0.534, 0.566, 0.082, 0.089)
      }
    ]
  }
};

// ------------------------------ assemble ------------------------------------

function buildEntry(id, spec) {
  const rawOps = spec.ops.map(op => {
    const base = opSpec(id, op.name, {
      s_bmax: op.s_bmax, paos_target: op.paos_target,
      tokens_k: op.tokens_k, rounds: op.rounds, wall_time_s: op.wall_time_s,
      profiling_runs: op.profiling_runs, waste: op.waste, n_solutions: op.n_solutions,
      n_cycles: op.n_cycles, termination: op.termination, k: op.k, dates: op.dates,
      agentId: op.agentId, baseline_quality: op.baseline_quality,
      factors: op.factors, anytime: op.anytime, anytime_failed: op.anytime_failed,
      runs: op.runs, utilization: op.utilization,
      compile_pass: op.compile_pass, func_pass_rate: op.func_pass_rate
    });
    if (op.factors === null) base.duration_by_shape = ANCHOR_SHAPES;
    if (op.runs) base.runs = op.runs;
    return deriveOperator(base);
  });

  const overall = aggregateOverall(rawOps);
  // entry-level baseline quality: any abnormal run tints the row
  const baselineQuality = rawOps.some(op => op.baseline_quality === 'saturated') ? 'saturated'
    : rawOps.some(op => op.baseline_quality === 'abnormal') ? 'abnormal' : 'normal';
  const reliabilitySource = rawOps.reduce((a, b) => (b.runs.length > a.runs.length ? b : a));
  const reliability = reliabilityAtDelta(reliabilitySource.runs, 3);

  const meta0 = OP_META;
  const etpsRows = rawOps.map(op => {
    if (!op.recipe) return null;
    const rep = op.duration_by_shape.find(row => row.shape === op.recipe.rep_shape);
    const tBase = (rep?.t_base_us ?? 0) * 1e-6; // µs → s
    const tBest = (rep?.t_best_us ?? 0) * 1e-6;
    return {
      operator: op.operator,
      etps_base: tBase ? round(etps(op.recipe, tBase), 0) : null,
      etps_best: tBest ? round(etps(op.recipe, tBest), 0) : null,
      etps_gain: op.paos == null ? null : round(op.paos, 4),
      recipe: op.recipe,
      etps_reason: null
    };
  });

  return {
    id,
    agent: AGENTS[spec.agent],
    engine_model: ENGINE_MODELS[spec.model],
    weights: ENGINE_MODELS[spec.model].openness,
    links: { report: '#' },
    k_group: spec.kGroup,
    baseline_quality: baselineQuality,
    coverage_operators: rawOps.length,
    overall: {
      ...overall,
      reliability: { p3: reliability.value, p3_reason: reliability.reason ?? null, k: reliability.k }
    },
    per_operator: rawOps.map(op => ({ ...op, etps: etpsRows.find(row => row?.operator === op.operator) ?? { etps_base: null, etps_reason: 'not_in_recipe' } })),
    runs_for_reliability: reliabilitySource.runs,
    reliability_operator: reliabilitySource.operator
  };
}

const entries = Object.entries(E).map(([id, spec]) => buildEntry(id, spec));

const runsTotal = entries.reduce((sum, e) => sum + e.per_operator.reduce((s2, op) => s2 + op.runs.length, 0), 0);
const tokensTotal = entries.reduce((sum, e) => sum + (e.overall.tokens_total_k ?? 0), 0);

const bundle = {
  meta: {
    status: 'measured',
    updated: '2026-08-31',
    note: 'AddRmsNorm（world-model × Qwen3_6）为 217 真实 run 产物锚点（AddRmsNorm_orchestrator_20260811，附录 A）；其余 entry 为同一 v2 契约下的示意数据，用于演示完整交互面。',
    entries_evaluated: entries.length,
    agents: Object.values(AGENTS),
    operators: Object.keys(SHAPES),
    chips: ['Ascend 910C'],
    model_clusters: [{ id: 'DeepSeek-V4', scenarios: ['decode_b1'] }],
    runs_total: runsTotal,
    tokens_total_k: round(tokensTotal, 1),
    recipe: {
      'DeepSeek-V4': {
        decode_b1: [
          { operator: 'AddRmsNorm', G: 1, n_i: 123 },
          { operator: 'Swiglu', G: 1, n_i: 427 }
        ]
      }
    },
    epsilon_default: 0.01,
    metric_directions: {
      sts_token: 'up', s_equiv: 'up', ebgs: 'up', vlr_mean: 'up',
      task_pass_rate: 'up', gate_rate: 'up',
      best_duration_us: 'down', tokens_total_k: 'down', rounds_median: 'down',
      wall_time_median_s: 'down', profiling_runs_median: 'down',
      'paos.paos': 'up', vec_ratio: 'context', u: 'context'
    }
  },
  entries
};

const out = path.join(root, 'src', 'data', 'bundle.json');
fs.writeFileSync(out, `${JSON.stringify(bundle, null, 2)}\n`);

// ------------------------------ console summary -----------------------------

const fmt = n => (n == null ? '–' : Number(n).toFixed(2));
console.log(`bundle written: ${out}`);
console.log(`entries=${entries.length} runs=${runsTotal} tokens_total=${Math.round(tokensTotal)}k`);
for (const e of entries) {
  const o = e.overall;
  console.log(
    `${e.id.padEnd(28)} K=${e.k_group} sts=${fmt(o.sts_token)} vlr_mean=${fmt(o.vlr_mean)} gate=${fmt(o.gate_rate)} paos=${o.paos?.paos ?? '–'} ` +
    `flags=[${o.flags.join(',')}] rel@3%=${o.reliability?.p3 ?? 'k<3'}`
  );
}

// quick anchor echo (full check lives in scripts/verify-anchors.mjs)
const anchor = entries.find(e => e.id === 'world-model+qwen3-6').per_operator.find(op => op.operator === 'AddRmsNorm');
console.log('\n[anchor] world-model+Qwen3_6 × AddRmsNorm');
console.log(`  s_bmax=${anchor.s_bmax} vlr=${anchor.vlr}% auls=${anchor.auls} sts=${anchor.sts_token} sts_time=${anchor.sts_time}`);
console.log(`  paos=${anchor.paos} worst=${JSON.stringify(anchor.worst_shape)} tokens_to.p1=${anchor.tokens_to?.p1} flags=[${anchor.flags.join(',')}]`);
