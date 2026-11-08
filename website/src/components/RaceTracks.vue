<template>
  <div id="duration-bars" v-html="svg"></div>
</template>

<script setup>
import { modelDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import { benchmark, selectedDuration } from '../store/benchmark';
import { formatNumber, formatMicros, formatRatio, formatSpeedup, attributeEscape } from '../lib/format';

// C6 right panel — one shape group with two rails: base (dashed, muted) and
// best (colored by S_lat). Endpoint position encodes absolute duration.
const svg = computed(() => {
  const { entry, operator } = selectedDuration.value;
  if (!entry || !operator) return '';
  const rows = operator.duration_by_shape ?? [];
  if (!rows.length) return '';
  const width = 760;
  const groupH = 66;
  const margin = { top: 34, right: 68, bottom: 42, left: 168 };
  const height = margin.top + margin.bottom + rows.length * groupH;
  const plotW = width - margin.left - margin.right;
  const normalized = benchmark.durationView === 'normalized';
  const fastestBase = Math.min(...rows.map(row => row.t_base_us));
  const scale = value => (normalized ? value / fastestBase : value);
  const max = Math.max(...rows.map(row => scale(row.t_base_us))) * 1.08;
  const x = value => margin.left + (value / max) * plotW;

  const axis = [0, .25, .5, .75, 1].map(tick => `<circle class="rail-scale-node" cx="${x(max * tick)}" cy="${height - margin.bottom + 3}" r="2.5"/><text class="chart-tick" x="${x(max * tick)}" y="${height - 14}" text-anchor="middle">${formatNumber(max * tick, normalized ? 2 : 1)}${normalized ? '×' : ''}</text>`).join('');

  const toneColor = s => (s < 1 ? 'var(--danger)' : s >= 1.05 ? 'var(--compute)' : 'var(--muted)');

  const groups = rows.map((row, index) => {
    const groupTop = margin.top + groupH * index;
    const cyBase = groupTop + groupH / 2 - 11;
    const cyBest = groupTop + groupH / 2 + 11;
    const baseX = x(scale(row.t_base_us));
    const bestX = x(scale(row.t_base_us / row.s_lat));
    const color = toneColor(row.s_lat);
    const label = normalized
      ? `${formatRatio(scale(row.t_base_us))} / ${formatRatio(scale(row.t_base_us / row.s_lat))}`
      : `${formatMicros(row.t_base_us)} → ${formatMicros(row.t_base_us / row.s_lat)}`;
    const tooltip = attributeEscape(
      `<strong>${modelDisplayName(entry.engine_model)} · ${operator.operator}</strong><br>${row.shape} · ${row.dtype}<br>Base: <span class="tooltip-value">${formatMicros(row.t_base_us)}</span> → Best: <span class="tooltip-value">${formatMicros(row.t_base_us / row.s_lat)}</span><br>S_lat: <span class="tooltip-value">${formatSpeedup(row.s_lat)}</span>`
    );
    const baseBeads = [.25, .5, .75].map(ratio => `<circle class="duration-pulse" cx="${margin.left + (baseX - margin.left) * ratio}" cy="${cyBase}" r="1.6" fill="var(--faint)"/>`).join('');
    const bestBeads = [.25, .5, .75].map(ratio => `<circle class="duration-pulse" cx="${margin.left + (bestX - margin.left) * ratio}" cy="${cyBest}" r="1.6" fill="${color}"/>`).join('');
    return `<g class="duration-group" data-model-id="${entry.id}" data-tooltip-html="${tooltip}">
      ${index ? `<line class="chip-rail-separator" x1="12" y1="${groupTop}" x2="${width - margin.right}" y2="${groupTop}"/>` : ''}
      <text class="chart-label" x="${margin.left - 12}" y="${cyBase - 2}" text-anchor="end">${row.shape}</text>
      <text class="chart-tick" x="${margin.left - 12}" y="${cyBase + 10}" text-anchor="end">${row.dtype} · ${label}</text>
      <line class="duration-rail-base" x1="${margin.left}" y1="${cyBase}" x2="${width - margin.right}" y2="${cyBase}"/>
      <line class="duration-rail-signal duration-rail-mut" x1="${margin.left}" y1="${cyBase}" x2="${baseX}" y2="${cyBase}"/>
      ${baseBeads}
      <circle class="duration-bead" cx="${baseX}" cy="${cyBase}" r="5.5" fill="#fff" stroke="var(--faint)"/>
      <line class="duration-rail-base" x1="${margin.left}" y1="${cyBest}" x2="${width - margin.right}" y2="${cyBest}"/>
      <line class="duration-rail-signal" x1="${margin.left}" y1="${cyBest}" x2="${bestX}" y2="${cyBest}" stroke="${color}"/>
      ${bestBeads}
      <circle class="duration-bead" cx="${bestX}" cy="${cyBest}" r="6.5" fill="#fff" stroke="${color}"/>
      <circle cx="${bestX}" cy="${cyBest}" r="2.3" fill="${color}"/>
      <text class="chart-tick" x="14" y="${groupTop + 12}">SHAPE ${String(index + 1).padStart(2, '0')}</text>
    </g>`;
  }).join('');

  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Base versus best task duration race tracks by shape">
    <text class="chart-tick" x="${margin.left}" y="15">TIMING: BASE (DASHED) / BEST</text>
    ${groups}
    <line class="chart-axis" x1="${margin.left}" y1="${height - margin.bottom + 3}" x2="${width - margin.right}" y2="${height - margin.bottom + 3}"/>
    ${axis}
  </svg>`;
});
</script>
