<template>
  <div id="scatter-chart" class="scatter-chart" :aria-label="ariaLabel">
    <div class="scatter-plot">
      <div v-html="svg"></div>
      <span v-if="plottedEntries.length" class="scatter-axis-title scatter-y-title" aria-hidden="true">GM Speedup</span>
      <span v-if="plottedEntries.length" class="scatter-axis-title scatter-x-title" aria-hidden="true">{{ benchmark.scatterX === 'cost' ? 'API Cost' : 'Tokens' }}</span>
    </div>
  </div>
</template>

<script setup>
import { configurationDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import { benchmark, visibleEntries } from '../store/benchmark';
import { formatTokenTick, formatTokensK, escapeHTML, attributeEscape, formatUsd, formatSpeedup } from '../lib/format';
import { svgArcPath } from '../lib/geometry';
import { vendorMarkSVG } from '../lib/markup';

const ariaLabel = computed(() => `GM Speedup versus ${benchmark.scatterX === 'cost' ? 'API Cost' : 'Tokens'} scatter on a logarithmic x axis`);

const plottedEntries = computed(() => visibleEntries.value.filter(entry =>
  Number.isFinite(entry.overall.ebgs) && entry.overall.ebgs > 0 && (benchmark.scatterX === 'cost'
      ? Number.isFinite(entry.overall.api_cost_usd) && entry.overall.api_cost_usd > 0
      : Number.isFinite(entry.overall.tokens_total_k) && entry.overall.tokens_total_k > 0)
));

const svg = computed(() => {
  const entries = plottedEntries.value;
  if (!entries.length) return '<p class="evidence-note">No comparable measurements are available for the selected axes.</p>';
  const width = 1120;
  const height = 560;
  const margin = { top: 46, right: 58, bottom: 78, left: 92 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const maxSpeedup = Math.max(...entries.map(entry => entry.overall.ebgs));
  const roughStep = maxSpeedup * 1.15 / 5;
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const yStep = [1, 2, 2.5, 5, 10].find(step => step * magnitude >= roughStep) * magnitude;
  const yMax = Math.ceil(maxSpeedup * 1.15 / yStep) * yStep;
  const y = value => margin.top + plotH - (value / yMax) * plotH;

  let x, xTicks, frontierPath = '';

  {
    // Log-x cost/token axis with a Pareto frontier.
    const costMode = benchmark.scatterX === 'cost';
    const xValue = entry => {
      if (costMode) {
        const cost = entry.overall.api_cost_usd;
        return cost == null ? null : cost;
      }
      return (entry.overall.tokens_total_k ?? 0) * 1000;
    };
    if (!entries.some(entry => xValue(entry) != null && xValue(entry) > 0)) {
      return `<svg viewBox="0 0 ${width} ${height}"><text class="chart-tick" x="${width / 2}" y="${height / 2}" text-anchor="middle">API cost unavailable: complete billing components are not exported</text></svg>`;
    }
    const priced = entries.filter(entry => xValue(entry) != null && xValue(entry) > 0);
    const values = priced.map(entry => xValue(entry));
    const lo = Math.min(...values) * 0.7;
    const hi = Math.max(...values) * 1.4;
    const logLo = Math.log10(lo);
    const logHi = Math.log10(hi);
    x = value => margin.left + ((Math.log10(value) - logLo) / (logHi - logLo)) * plotW;
    const decades = [];
    for (let d = Math.floor(logLo); d <= Math.ceil(logHi); d += 1) {
      for (const m of [1, 2, 5]) {
        const v = m * 10 ** d;
        if (v >= lo && v <= hi) decades.push(v);
      }
    }
    xTicks = decades.length ? decades : [lo, hi];

    const frontier = [...priced].sort((a, b) => xValue(a) - xValue(b) || b.overall.ebgs - a.overall.ebgs).filter((entry, index, array) => {
      const cheaper = array.slice(0, index);
      return !cheaper.some(other => other.overall.ebgs >= entry.overall.ebgs);
    });
    frontierPath = frontier.map((entry, index) => `${index ? 'L' : 'M'}${x(xValue(entry)).toFixed(1)},${y(entry.overall.ebgs).toFixed(1)}`).join(' ');
  }

  const verticals = xTicks.map(tick => `<line class="constellation-tick" x1="${x(tick)}" y1="${height - margin.bottom - 8}" x2="${x(tick)}" y2="${height - margin.bottom + 8}"/><text class="chart-tick" x="${x(tick)}" y="${height - margin.bottom + 25}" text-anchor="middle">${benchmark.scatterX === 'cost' ? formatUsd(tick) : formatTokenTick(tick)}</text>`).join('');
  const yTicks = Array.from({ length: Math.round(yMax / yStep) + 1 }, (_, index) => index * yStep).map(tick => `<line class="constellation-tick" x1="${margin.left - 7}" y1="${y(tick)}" x2="${margin.left + 7}" y2="${y(tick)}"/><text class="chart-tick" x="${margin.left - 12}" y="${y(tick) + 5}" text-anchor="end">${Number(tick.toFixed(3))}×</text>`).join('');

  // Numbered marks preserve the original brand/orbit treatment; long model
  // names remain in the linked table rather than overlapping on dense plots.
  const iconScale = entries.length > 15 ? 0.8 : entries.length > 10 ? 0.9 : 1;
  const haloR = +(35 * iconScale).toFixed(1);
  const orbitR = +(29 * iconScale).toFixed(1);
  const discR = +(22 * iconScale).toFixed(1);
  const markSize = Math.max(11, Math.round(22 * iconScale));
  // Keep brand marks separate without moving the measured data. Each displaced
  // mark has a leader line to its original coordinate and a matching table number.
  const placed = [];
  const spacingX = haloR * 2 + 10;
  const spacingY = haloR * 2 + 22;
  const slots = [];
  for (let row = -6; row <= 6; row += 1) {
    for (let col = -12; col <= 12; col += 1) {
      slots.push({ dx:col * spacingX, dy:row * spacingY });
    }
  }
  slots.sort((a,b) => (a.dx ** 2 + a.dy ** 2) - (b.dx ** 2 + b.dy ** 2));
  const points = entries.map(entry => {
    const px = x(benchmark.scatterX === 'cost' ? entry.overall.api_cost_usd : (entry.overall.tokens_total_k ?? 1) * 1000);
    const py = y(entry.overall.ebgs);
    const muted = (entry.baseline_quality ?? 'normal') !== 'normal';
    const label = configurationDisplayName(entry);
    const estimated = entry.overall.api_cost_kind === 'scenario_estimate';
    const tooltip = `<strong>${escapeHTML(label)}</strong><br>GM Speedup: <span class="tooltip-value">${formatSpeedup(entry.overall.ebgs)}</span><br>Tokens: ${formatTokensK(entry.overall.tokens_total_k)}` + (benchmark.scatterX === 'cost'
      ? `<p>${estimated ? 'Estimated API cost' : 'API cost'}: ${formatUsd(entry.overall.api_cost_usd)}${estimated ? '<br>90% input / 10% output, no caching' : ''}${entry.overall.api_cost_price?.rate_basis === 'successor_proxy' ? '<br>DeepSeek successor API rate proxy' : ''}</p>` : '');
    const slot = slots.find(({dx,dy}) => {
      const cx = px + dx, cy = py + dy;
      return cx >= margin.left + haloR && cx <= width - margin.right - haloR
        && cy >= margin.top + haloR + 16 && cy <= height - margin.bottom - haloR
        && placed.every(other => Math.abs(cx - other.x) >= spacingX || Math.abs(cy - other.y) >= spacingY);
    });
    const cx = px + (slot?.dx ?? 0), cy = py + (slot?.dy ?? 0);
    placed.push({x:cx,y:cy});
    const number = visibleEntries.value.findIndex(item => item.id === entry.id) + 1;
    const displaced = Math.hypot(cx-px,cy-py) > 1;
    return `<g class="scatter-marker" data-entry-number="${number}" data-entry-id="${attributeEscape(entry.id)}" data-gm-speedup="${entry.overall.ebgs}" data-x-value="${benchmark.scatterX === 'cost' ? entry.overall.api_cost_usd : entry.overall.tokens_total_k * 1000}" data-x-coordinate="${px}" data-y-coordinate="${py}">
      ${displaced ? `<line class="scatter-leader" x1="${px}" y1="${py}" x2="${cx}" y2="${cy}" stroke="${entry.engine_model.color}"/><circle class="scatter-coordinate" cx="${px}" cy="${py}" r="3" fill="${entry.engine_model.color}"/>` : ''}
      <g transform="translate(${cx.toFixed(1)} ${cy.toFixed(1)})"><g class="scatter-point${muted ? ' scatter-point-muted' : ''}" style="--icon-scale:${iconScale}" data-model-id="${entry.id}" data-open-drawer data-tooltip-html="${attributeEscape(tooltip)}">
      <circle class="constellation-halo" r="${haloR}" fill="color-mix(in srgb, ${entry.engine_model.color} 7%, transparent)" stroke="color-mix(in srgb, ${entry.engine_model.color} 32%, transparent)"/>
      <path class="constellation-orbit" d="${svgArcPath(0,0,orbitR,-210,60)}" stroke="${entry.engine_model.color}"/>
      <circle r="${discR}" fill="#fff" stroke="${entry.engine_model.color}" ${entry.agent.type === 'general' ? 'stroke-dasharray="4 3"' : ''}/>
      ${vendorMarkSVG(entry,0,0,markSize)}
      <text class="scatter-label" x="0" y="${-haloR - 6}" text-anchor="middle">${number}</text>
      <title>${escapeHTML(label)}</title>
    </g></g></g>`;
  }).join('');

  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${ariaLabel.value}" data-y-metric="ebgs" data-x-metric="${benchmark.scatterX}" data-y-max="${yMax}">
    <defs><filter id="signal-glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="7"/></filter></defs>
    ${verticals}${yTicks}
    <line class="chart-axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"/>
    <line class="chart-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"/>
    ${frontierPath ? `<path class="pareto-glow" d="${frontierPath}"/><path class="pareto-line" d="${frontierPath}"/>` : ''}
    ${points}
  </svg>`;
});
</script>
