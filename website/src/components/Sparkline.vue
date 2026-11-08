<template>
  <svg class="sparkline" :style="width > 120 ? { width: '100%', height: 'auto' } : {}" :viewBox="`0 0 ${width} ${height}`" role="img" :aria-label="label">
    <line x1="0" :y1="height - 2" :x2="width" :y2="height - 2" stroke="var(--border)" />
    <line x1="0" :y1="yOf(1)" :x2="width" :y2="yOf(1)" stroke="var(--border)" stroke-dasharray="2 3" />
    <path :d="path" fill="none" :stroke="color" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    <circle :cx="last.x" :cy="last.y" r="2.5" :fill="color" />
  </svg>
</template>

<script setup>
import { computed } from 'vue';

// C3 — the v2 sparkline is the anytime best-so-far curve S(T) on a
// logarithmic cumulative-token axis (the round axis of v1 cannot support
// AULS / Tokens-to-Δ). Shared by the leaderboard column (92×26) and the
// drawer (wide variant).
const props = defineProps({
  points: { type: Array, required: true }, // [{cum_tokens_k, s, wall_sec}]
  color: { type: String, default: 'var(--compute)' },
  width: { type: Number, default: 92 },
  height: { type: Number, default: 26 },
  label: { type: String, default: 'Anytime best-so-far speedup on log token axis' }
});

const geometry = computed(() => {
  const W = props.width;
  const H = props.height;
  const passed = props.points.filter(point => point.cum_tokens_k > 0);
  if (!passed.length) return { path: '', last: { x: W, y: H / 2 } };
  const t0 = passed[0].cum_tokens_k;
  const t1 = passed[passed.length - 1].cum_tokens_k;
  const sMax = Math.max(1, ...passed.map(p => p.s)) * 1.06;
  const xOf = t => t1 === t0 ? W - 3 : 3 + ((Math.log10(t) - Math.log10(t0)) / (Math.log10(t1) - Math.log10(t0))) * (W - 6);
  const yOf = s => 2 + (1 - (s - 1) / (sMax - 1)) * (H - 7);
  // step (best-so-far) rendering: horizontal to new x, vertical to new y
  const steps = passed.map((point, index) => {
    const x = xOf(point.cum_tokens_k);
    const y = yOf(point.s);
    if (!index) return `M${x.toFixed(1)},${y.toFixed(1)}`;
    return `L${x.toFixed(1)},${yOf(passed[index - 1].s).toFixed(1)} L${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const lastPoint = passed[passed.length - 1];
  return {
    path: steps.join(' '),
    last: { x: xOf(lastPoint.cum_tokens_k), y: yOf(lastPoint.s) }
  };
});

const path = computed(() => geometry.value.path);
const last = computed(() => geometry.value.last);

const yOf = s => {
  const passed = props.points.filter(point => point.cum_tokens_k > 0);
  const sMax = Math.max(1, ...passed.map(p => p.s)) * 1.06;
  return 2 + (1 - (s - 1) / (sMax - 1)) * (props.height - 7);
};
</script>
