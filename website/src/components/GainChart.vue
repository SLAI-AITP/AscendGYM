<template>
  <div id="gain-chart" v-html="svg"></div>
</template>

<script setup>
import { modelDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import { gainSelection } from '../store/benchmark';
import { signed, escapeHTML, attributeEscape, formatSpeedup, formatMeasurement } from '../lib/format';

// C9 — per-operator VLR vectors from an explicit zero origin.
//   arrow tip   = representative VLR (1 − 1/s_bmax)
//   open ring   = geomean VLR from profiling (1 − 1/PAOS)
//   small tick  = worst shape (1 − 1/S_lat_worst), red when S_lat < 0.9
// The arrow↔ring gap is the direct picture of "representative vs geomean".
const svg = computed(() => {
  const entry = gainSelection.value;
  if (!entry) return '';
  const ops = entry.per_operator;
  const width = 960;
  const rowH = 44;
  const margin = { top: 30, right: 100, bottom: 40, left: 240 };
  const height = margin.top + margin.bottom + ops.length * rowH;
  const plotW = width - margin.left - margin.right;

  const vlrRep = ops.map(op => op.vlr ?? 0);
  // Per-shape geomean is a *different* quantity from PAOS (sqrt(PKAI x PTEI))
  // and is not exported yet, so the ring disappears when it is unavailable
  // instead of being substituted with an unrelated number.
  const vlrGeo = ops.map(op => (op.paos?.s_lat_geo ? (1 - 1 / op.paos.s_lat_geo) * 100 : null)).filter(Number.isFinite);
  const vlrWorst = ops.map(op => (op.worst_shape ? (1 - 1 / op.worst_shape.s_lat) * 100 : null)).filter(Number.isFinite);
  const lo = Math.min(0, ...vlrGeo, ...vlrWorst) - 6;
  const hi = Math.max(8, ...vlrRep) + 8;
  const x = value => margin.left + ((value - lo) / (hi - lo)) * plotW;
  const zero = x(0);

  const ticks = [];
  const tickStep = (hi - lo) > 40 ? 10 : 5;
  for (let t = Math.ceil(lo / tickStep) * tickStep; t <= hi; t += tickStep) ticks.push(t);
  const ticksSvg = ticks.map(t => `<line class="${t === 0 ? 'impulse-zero' : 'impulse-tick'}" x1="${x(t)}" y1="${margin.top}" x2="${x(t)}" y2="${height - margin.bottom}"/><text class="chart-tick" x="${x(t)}" y="${height - 16}" text-anchor="middle">${t}%</text>`).join('');

  const traces = ops.map((op, index) => {
    const cy = margin.top + rowH * index + rowH / 2;
    const rep = op.vlr ?? 0;
    const valueX = x(rep);
    // `flat` — the gain renders as `0.00%`. Such a row has no direction to point
    // at: the curve collapses to a single point and the arrow degenerates into a
    // zero-length spike sitting on the origin, which reads as a rendering fault
    // rather than as data. Draw a neutral origin dot instead — the honest picture
    // of "this operator produced no measurable gain".
    //
    // The threshold matches the label's own precision (`formatMeasurement` → 2
    // decimals), so the dot appears exactly when the number beside it would say
    // 0.00%. A sign is dropped too: `-0.00%` claims a direction that is not there.
    const flat = Math.abs(rep) < 0.005;
    const color = rep >= 0 ? 'var(--compute)' : 'var(--danger)';
    const direction = rep >= 0 ? 1 : -1;
    const mid = (zero + valueX) / 2;
    const pulseNodes = [.25, .5, .75].map((ratio, nodeIndex) => `<circle class="impulse-node" cx="${zero + (valueX - zero) * ratio}" cy="${cy + (nodeIndex % 2 ? 3 : -3)}" r="${nodeIndex === 1 ? 2.5 : 1.7}" fill="${color}"/>`).join('');
    const arrow = `${valueX},${cy} ${valueX - direction * 8},${cy - 5} ${valueX - direction * 6},${cy} ${valueX - direction * 8},${cy + 5}`;
    const marker = flat
      ? `<circle class="impulse-flat" cx="${valueX.toFixed(1)}" cy="${cy}" r="4.5"/>
      <text class="chart-data impulse-flat-value" x="${valueX + 12}" y="${cy + 3}" text-anchor="start">${formatMeasurement(rep)}%</text>`
      : `<path class="impulse-trace" d="M${zero} ${cy}C${mid} ${cy - 9 * direction},${mid} ${cy + 9 * direction},${valueX} ${cy}" stroke="${color}"/>
      ${pulseNodes}<polyline class="impulse-arrow" points="${arrow}" stroke="${color}"/>
      <text class="chart-data" x="${rep >= 0 ? valueX + 12 : valueX - 12}" y="${cy + 3}" text-anchor="${rep >= 0 ? 'start' : 'end'}">${signed(rep, 2)}%</text>`;

    // per-shape geomean endpoint ring (only when profiling vectors exist)
    const geo = op.paos?.s_lat_geo ? (1 - 1 / op.paos.s_lat_geo) * 100 : null;
    const ring = geo == null ? '' : `<circle class="impulse-geo-ring" cx="${x(geo)}" cy="${cy}" r="7.5"/>
      <text class="chart-tick" x="${x(geo)}" y="${cy - 14}" text-anchor="middle">geo ${signed(geo, 2)}%</text>`;

    // worst-shape protection tick
    const worst = op.worst_shape;
    const worstX = worst ? (1 - 1 / worst.s_lat) * 100 : null;
    const worstSvg = worstX == null ? '' : `<line class="impulse-worst${worst.s_lat < 0.9 ? ' impulse-worst-bad' : ''}" x1="${x(worstX)}" y1="${cy - 11}" x2="${x(worstX)}" y2="${cy + 11}"/>
      <text class="chart-tick${worst.s_lat < 0.9 ? ' tone-bad-text' : ''}" x="${x(worstX)}" y="${cy + 24}" text-anchor="middle">worst ${worst.shape} · ${formatSpeedup(worst.s_lat)}</text>`;

    const tooltip = `<strong>${escapeHTML(modelDisplayName(entry.engine_model))} · ${escapeHTML(op.operator)}</strong><br>Incumbent VLR: <span class=&quot;tooltip-value&quot;>${signed(rep, 2)}%</span> (${formatSpeedup(op.s_bmax)})${(op.flags || []).length ? `<br>flags: ${escapeHTML(op.flags.join(', '))}` : ''}`;

    return `<g class="impulse-row" data-model-id="${entry.id}" data-tooltip-html="${attributeEscape(tooltip)}">
      <text class="chart-label" x="${margin.left - 18}" y="${cy - 3}" text-anchor="end">${escapeHTML(op.operator)}<tspan class="chart-tick impulse-meta" x="${margin.left - 18}" dy="14">${op.bound ? `${escapeHTML(op.bound)}-bound · ` : ''}${op.rounds ?? '–'} rounds</tspan></text>
      <line class="impulse-rail" x1="${x(lo)}" y1="${cy}" x2="${x(hi)}" y2="${cy}"/>
      ${ring}${worstSvg}
      ${marker}
    </g>`;
  }).join('');

  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="VLR signal vectors by operator for ${escapeHTML(modelDisplayName(entry.engine_model))}">
    ${ticksSvg}${traces}
    <text class="chart-tick" x="${margin.left}" y="14">VLR AT 50M TOKENS</text>
  </svg>`;
});
</script>
