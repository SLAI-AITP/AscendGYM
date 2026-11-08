<template>
  <div>
    <div id="radar-chart" v-html="svg"></div>
    <div class="util-legend">
      <span v-for="pair in selection" :key="pair.entry.id" :data-model-id="pair.entry.id" data-open-drawer>
        <EntryLabel :entry="pair.entry" size="sm" />
      </span>
    </div>
  </div>
</template>

<script setup>
import { modelDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import EntryLabel from './EntryLabel.vue';
import { benchmark, utilSelection } from '../store/benchmark';
import { utilizationChannels } from '../lib/meta';
import { svgArcPath, svgPolarPoint } from '../lib/geometry';
import { escapeHTML, attributeEscape, formatRatio } from '../lib/format';
import { assetPath } from '../lib/markup';

const selection = computed(() => utilSelection.value);

const svg = computed(() => {
  const selected = selection.value.filter(pair => pair.operator);
  if (!selected.length) return '';
  const width = 520;
  const height = 360;
  const center = { x: 260, y: 165 };
  const radii = [72, 96, 120];
  const n = utilizationChannels.length;
  const sectorStart = index => -112 + index * (360 / n - 16);
  const sectorSweep = 56;
  const baseArcs = selected.map((_, modelIndex) => utilizationChannels.map((metric, metricIndex) => `<path class="fingerprint-base" d="${svgArcPath(center.x, center.y, radii[modelIndex], sectorStart(metricIndex), sectorStart(metricIndex) + sectorSweep)}"/>`).join('')).join('');
  // Per-channel visible maximum keeps small ratios (0.05–0.1) legible.
  const maxOf = channelKey => Math.max(
    ...selected.map(({ operator }) => operator.utilization?.[`${channelKey}_best`] ?? 0),
    1e-6
  );
  const signalArcs = selected.map(({ entry, operator }, modelIndex) => utilizationChannels.map((metric, metricIndex) => {
    const value = operator.utilization?.[`${metric.key}_best`] ?? null;
    const start = sectorStart(metricIndex);
    if (value == null) {
      const mid = sectorStart(metricIndex) + sectorSweep / 2;
      const [dx, dy] = svgPolarPoint(center.x, center.y, radii[modelIndex], mid);
      return `<text class="fingerprint-zero" x="${dx}" y="${dy + 3}" text-anchor="middle">≈0</text>`;
    }
    const end = start + sectorSweep * (value / maxOf(metric.key));
    const [dotX, dotY] = svgPolarPoint(center.x, center.y, radii[modelIndex], end);
    return `<g data-model-id="${entry.id}" data-open-drawer data-tooltip-html="${attributeEscape(`<strong>${modelDisplayName(entry.engine_model)} · ${benchmark.utilOperator}</strong><br>${metric.label} (best): <span class=&quot;tooltip-value&quot;>${formatRatio(value)}</span>`)}"><path class="fingerprint-signal" d="${svgArcPath(center.x, center.y, radii[modelIndex], start, end)}" stroke="${entry.engine_model.color}"/><circle class="fingerprint-node" cx="${dotX}" cy="${dotY}" r="3.4" fill="${entry.engine_model.color}"/></g>`;
  }).join('')).join('');
  const labels = utilizationChannels.map((metric, metricIndex) => {
    const [labelX, labelY] = svgPolarPoint(center.x, center.y, 148, sectorStart(metricIndex) + sectorSweep / 2);
    return `<text class="fingerprint-label" x="${labelX}" y="${labelY + 3}" text-anchor="middle">${metric.label}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="AI unit ratio fingerprint for ${escapeHTML(benchmark.utilOperator)}">${baseArcs}${signalArcs}${labels}<circle class="fingerprint-core" cx="${center.x}" cy="${center.y}" r="42"/><image href="${assetPath('assets/logo/mark-primary.svg')}" x="${center.x - 22}" y="${center.y - 22}" width="44" height="44"/><text class="chart-tick" x="${center.x}" y="${center.y + 58}" text-anchor="middle">AI UNIT SIGNAL</text></svg>`;
});
</script>
