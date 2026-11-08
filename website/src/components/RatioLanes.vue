<template>
  <div id="util-bars" v-html="svg"></div>
</template>

<script setup>
import { modelDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import { benchmark, utilSelection } from '../store/benchmark';
import { utilizationChannels } from '../lib/meta';
import { attributeEscape, formatRatio } from '../lib/format';

// C7 lanes: one row per collectible channel + the U (a/t) row.
// Each entry contributes a base (hollow) → best (filled) bead pair.
const svg = computed(() => {
  const selected = utilSelection.value.filter(pair => pair.operator);
  if (!selected.length) return '';
  const width = 650;
  const lanes = [...utilizationChannels, { key: 'u', label: 'U = a/t' }];
  const height = 380;
  const margin = { top: 42, right: 48, bottom: 20, left: 84 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const rowH = plotH / lanes.length;
  const x = value => margin.left + Math.min(1, Math.max(0, value)) * plotW;
  const scale = [0, .25, .5, .75, 1].map(tick => `<circle class="unit-scale-node" cx="${x(tick)}" cy="${margin.top - 13}" r="2"/><text class="chart-tick" x="${x(tick)}" y="${margin.top - 22}" text-anchor="middle">${tick}</text>`).join('');

  const lanesSvg = lanes.map((metric, metricIndex) => {
    const cy = margin.top + rowH * metricIndex + rowH / 2;
    const beads = selected.map(({ entry, operator }, entryIndex) => {
      const isU = metric.key === 'u';
      const base = operator.utilization?.[isU ? 'u_base' : `${metric.key}_base`] ?? null;
      const best = operator.utilization?.[isU ? 'u_best' : `${metric.key}_best`] ?? null;
      if (best == null) {
        return `<text class="chart-tick" x="${margin.left + 8}" y="${cy + (entryIndex - (selected.length - 1) / 2) * 10 + 3}">≈0（vector op）</text>`;
      }
      const beadY = cy + (entryIndex - (selected.length - 1) / 2) * 10;
      const baseX = base == null ? null : x(base);
      const bestX = x(best);
      const baseBead = baseX == null ? '' : `<circle class="unit-bead-hollow" cx="${baseX}" cy="${beadY}" r="6" stroke="${entry.engine_model.color}"/>`;
      const link = baseX == null ? '' : `<line class="unit-bead-link" x1="${baseX + 7}" y1="${beadY}" x2="${bestX - 8}" y2="${beadY}" stroke="${entry.engine_model.color}"/>`;
      return `<g data-model-id="${entry.id}" data-open-drawer data-tooltip-html="${attributeEscape(`<strong>${modelDisplayName(entry.engine_model)} · ${benchmark.utilOperator}</strong><br>${metric.label}: <span class="tooltip-value">${formatRatio(base)}</span> → <span class="tooltip-value">${formatRatio(best)}</span>`)}">${baseBead}${link}<circle class="unit-bead-halo" cx="${bestX}" cy="${beadY}" r="10" fill="color-mix(in srgb, ${entry.engine_model.color} 13%, transparent)"/><circle class="unit-bead" cx="${bestX}" cy="${beadY}" r="6" fill="#fff" stroke="${entry.engine_model.color}"/><circle cx="${bestX}" cy="${beadY}" r="2.3" fill="${entry.engine_model.color}"/><text class="chart-data" x="${bestX + 11}" y="${beadY + 3}" style="font-size:7px">${formatRatio(best)}</text></g>`;
    }).join('');
    const railNodes = [0, .25, .5, .75, 1].map(tick => `<circle class="unit-lane-node" cx="${x(tick)}" cy="${cy}" r="1.7"/>`).join('');
    return `<g><text class="chart-label" x="${margin.left - 14}" y="${cy + 3}" text-anchor="end">${metric.label}</text><line class="unit-lane" x1="${margin.left}" y1="${cy}" x2="${width - margin.right}" y2="${cy}"/>${railNodes}${beads}</g>`;
  }).join('');

  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Exact AI unit ratio lanes, base versus best"><text class="chart-tick" x="${margin.left}" y="12">UTILIZATION UNIT LANES · ratio (0–1) · open=base → filled=best</text>${scale}${lanesSvg}</svg>`;
});
</script>
