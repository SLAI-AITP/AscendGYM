<template>
  <div ref="chartRoot" id="rank-chart" class="rank-chart" :aria-label="`${metricLabel} ranking`" v-html="svg"></div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { benchmark, visibleEntries, entryMetric } from '../store/benchmark';
import { metricMeta } from '../lib/meta';
import { escapeHTML, attributeEscape, formatTokensK, formatWallTime, formatSts, formatSpeedup, formatSpeedupDetail, formatPct, formatMicros, formatMillis, formatUsd, formatInteger, formatMeasurement } from '../lib/format';
import { vendorMarkSVG } from '../lib/markup';
import { harnessDisplayName, modelDisplayName, configurationDisplayName } from '../lib/displayNames';
import { isDesktopViewport, pageScale } from '../lib/displayScale';

const props = defineProps({ metricLabel: { type:String, default:'STS' } });
const chartRoot = ref(null);
const chartWidth = ref(1080);
const desktopLayout = ref(isDesktopViewport());
let resizeObserver;
let valueMeasureContext;

// Measure rendered SVG bounds as well as estimating text space below. This
// includes actual font metrics and rotated names, not just glyph counts.
function fitContent() {
  const element = chartRoot.value?.querySelector('svg');
  // Align the multiplication sign to the rendered harness name, rather than
  // estimating its centre from character counts (Codex and OpenCode differ).
  element?.querySelectorAll('.rank-axis-label').forEach(label => {
    const harness = label.querySelector('.rank-axis-harness');
    const cross = label.querySelector('.rank-axis-cross');
    if (!harness || !cross) return;
    const harnessBox = harness.getBBox();
    const centre = harnessBox.x + harnessBox.width / 2;
    cross.setAttribute('text-anchor', 'middle');
    cross.setAttribute('x', centre);
    const crossBox = cross.getBBox();
    cross.setAttribute('x', centre + (centre - crossBox.x - crossBox.width / 2));
  });
  if (!desktopLayout.value) return;
  const content = element?.querySelector('.rank-plot-content');
  if (!content) return;
  const box = content.getBBox();
  const width = Number(element.dataset.canvasWidth);
  const height = Number(element.dataset.canvasHeight);
  const left = Math.min(0, box.x - 16);
  // Trim unused space above the plot while retaining clearance for labels.
  const top = box.y - 16;
  const right = Math.max(width, box.x + box.width + 16);
  const bottom = Math.max(height, box.y + box.height + 16);
  element.setAttribute('viewBox', `${left} ${top} ${right-left} ${bottom-top}`);
  // Publish the visible axis size after SVG fitting so companion charts can
  // match it without changing this chart's established density and layout.
  const tick = element.querySelector('.chart-tick');
  if (tick && right > left) {
    const size = Number.parseFloat(getComputedStyle(tick).fontSize)
      * element.getBoundingClientRect().width / pageScale() / (right - left);
    if (Number.isFinite(size) && size > 0) {
      chartRoot.value.closest('#app')?.style.setProperty('--benchmark-axis-display-size', `${size}px`);
    }
  }
}

onMounted(() => {
  resizeObserver = new ResizeObserver(() => {
    // Preserve the existing mobile presentation; desktop fills its panel.
    desktopLayout.value = isDesktopViewport();
    chartWidth.value = desktopLayout.value
      ? Math.max(320, chartRoot.value?.clientWidth ?? 1080) : 1080;
    nextTick(fitContent);
  });
  resizeObserver.observe(chartRoot.value);
  document.fonts?.ready.then(fitContent);
});
onBeforeUnmount(() => resizeObserver?.disconnect());

// Rank-metric value formatting for core labels and grid ticks. Keys match the
// evaluator's metric names; the decimal count for every branch comes from
// `lib/precision.js` via the shared formatters. GM chart labels use the compact
// three-decimal ratio tier; the table keeps its four-decimal detail tier.
function formatRankValue(metric, value) {
  // A missing value is a *null* in the summary, but `entryMetric` reports it as
  // `-Infinity` so it sorts last — an infinity is not a measurement, and must
  // never reach the formatters (they would print “-Infinity” or “NaN”).
  if (value == null || !Number.isFinite(value)) return '–';
  switch (metric) {
    case 'sts_token': return formatSts(value);
    case 'budget_gm': case 's_equiv': case 'ebgs': case 'speedup_mean': return formatSpeedup(value);
    case 'paos.paos': case 'paos.pkai': case 'paos.ptei': return formatSpeedupDetail(value);
    case 'vlr_mean': case 'task_pass_rate': case 'gate_rate':
    case 'candidate_pass_rate': case 'compile_pass': return formatPct(value);
    case 'best_duration_us': return formatMicros(value);
    case 'baseline_duration_ms': return formatMillis(value);
    case 'tokens_total_k': return formatTokensK(value);
    case 'wall_time_median_s': return formatWallTime(value);
    case 'api_cost_usd': return formatUsd(value);
    case 'rounds_median': return formatInteger(value);
    default: return formatMeasurement(value, metricMeta[metric]?.decimals);
  }
}

// y-domain per metric: Task Pass Rate spans 50–100%; other percent/score
// metrics start at 0. Ratio metrics
// (S_equiv / EBGS / PAOS) are referenced at 1.0× so small gains stay visible.
// Only measured values take part: an entry with a missing rank metric has no
// position to contribute, and folding its sentinel in would blow the domain up.
function domainFor(metric, values) {
  const clean = values.filter(Number.isFinite);
  const max = Math.max(...clean, 0);
  if (['budget_gm', 's_equiv', 'ebgs', 'speedup_mean', 'paos.paos', 'paos.pkai', 'paos.ptei'].includes(metric)) {
    const step = niceStep((Math.max(max, 1.01) - 1) / 5);
    return [1, 1 + Math.ceil((Math.max(max, 1.01) - 1) / step) * step, step];
  }
  if (metric === 'tokens_total_k') {
    // Data are in thousands of tokens; ticks use whole millions consistently.
    const step = Math.max(1, niceStep(max / 1000 / 5)) * 1000;
    return [0, Math.max(step, Math.ceil(max / step) * step), step];
  }
  if (metric === 'wall_time_median_s') {
    const step = Math.max(1, niceStep(max / 3600 / 4)) * 3600;
    return [0, Math.max(step, Math.ceil(max / step) * step), step];
  }
  if (metric === 'task_pass_rate') return [50, 100];
  if (metric === 'rounds_median') return [0, 1500];
  if (['gate_rate', 'candidate_pass_rate', 'compile_pass'].includes(metric)) return [0, 100];
  if (metric === 'vlr_mean') return [0, Math.max(nice(max), 5)];
  // A roster where every value is missing collapses to a zero-height band; the
  // fallback keeps `y()` from dividing by it (the cores render as empty stubs
  // either way, but the grid must not turn into NaN coordinates).
  const upper = nice(max);
  return [0, upper > 0 ? upper : 1];
}

// Prefer readable 1/2/3/5 multiples, e.g. 0.2×, 500M tokens, or 3 hours.
function niceStep(value) {
  if (!(value > 0)) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  return [1, 2, 3, 5, 10].find(factor => factor * magnitude >= value) * magnitude;
}

function nice(value) {
  if (value <= 10) return Math.ceil(value / 2) * 2;
  if (value <= 100) return Math.ceil(value / 10) * 10;
  if (value <= 1000) return Math.ceil(value / 100) * 100;
  return Math.ceil(value / 1000) * 1000;
}

// Grid labels go through `formatRankValue`, which returns its own placeholder for
// a missing value. A fallback tick is never reached in practice (a domain always
// spans a positive interval), but rendering “NaN” on a degenerate axis would be
// worse than an honest dash.
function formatAxisValue(metric, value) {
  if (metric === 'tokens_total_k') return `${formatInteger(value / 1000)}M`;
  if (metric === 'wall_time_median_s') return `${formatInteger(value / 3600)}h`;
  if (['budget_gm', 's_equiv', 'ebgs', 'speedup_mean', 'paos.paos', 'paos.pkai', 'paos.ptei'].includes(metric)) {
    return formatSpeedup(value).replace(/(\.\d*?[1-9])0+×$/, '$1×').replace(/\.0+×$/, '×');
  }
  const text = formatRankValue(metric, value);
  return text === 'NaN' ? '–' : text;
}

// Every ring uses a fixed 100-unit circumference. STS is a score (not a
// percentage); its ring uses a fixed 0–100 display scale, while the axis and
// tooltip retain the uncapped score. GATE/VLR are measured percentages. Never
// normalize any ring by the visible roster: filtering must not change its fill.
function metricRing(name, value, x, y, radius, color, active = true) {
  const measured = value != null && Number.isFinite(value);
  const fill = measured && active ? Math.min(Math.abs(value), 100) : 0;
  const negative = measured && value < 0;
  const stroke = negative ? 'var(--bad, #b54a44)' : color;
  // Flat caps preserve the arc length of small percentages; a round cap would
  // add visible length, especially for VLR values of only 2–5%.
  return `<g class="rank-metric-ring" data-ring="${name}" data-value="${measured ? value : ''}" data-scale="100" aria-label="${name}: ${measured ? (name === 'STS' ? formatSts(value) : formatPct(value)) : 'Unavailable'}">
    <circle class="core-ring-track" cx="${x}" cy="${y}" r="${radius}"/>
    ${measured && active ? `<circle class="core-ring-fill" cx="0" cy="0" r="${radius}" pathLength="100" stroke-dasharray="${fill} 100" stroke="${stroke}" transform="translate(${x} ${y}) rotate(-90)${negative ? ' scale(1 -1)' : ''}"/>` : ''}
  </g>`;
}

const svg = computed(() => {
  const entries = visibleEntries.value;
  if (!entries.length) return '';

  // ---- roster density ---------------------------------------------------
  // Geometry and type step down at 10 and 20 entries. Past those sizes the
  // cores start to touch and the value labels merge into one band, so the whole
  // field is scaled rather than merely re-spaced. Two separate factors: the
  // cores can shrink harder than the type before they stop reading.
  const count = entries.length;
  const densityScale = count > 20 ? 0.66 : count > 10 ? 0.82 : 1;
  const textScale = count > 20 ? 0.82 : count > 10 ? 0.92 : 1;
  const annotationScale = desktopLayout.value ? 1.3 : 1;

  const top = desktopLayout.value ? 96 : 72;
  const metric = metricMeta[benchmark.rankMetric] || metricMeta.sts_token;
  const values = entries.map(entry => entryMetric(entry, benchmark.rankMetric));
  const [d0, d1, tickStep] = domainFor(benchmark.rankMetric, values);
  const ticks = tickStep
    ? Array.from({ length: Math.round((d1 - d0) / tickStep) + 1 }, (_, index) => d0 + index * tickStep)
    : [0, .25, .5, .75, 1].map(tick => d0 + (d1 - d0) * tick);
  const axisLabelScale = desktopLayout.value ? 1.331 : 1;
  const axisFont = 11 * annotationScale * axisLabelScale;
  const axisLabels = ticks.map(tick => formatAxisValue(benchmark.rankMetric, tick));
  const left = desktopLayout.value ? Math.max(58,...axisLabels.map(label => label.length * axisFont * .64 + 28)) : 52;

  // One configuration per tick, drawn from the SAME sorted entry as its core.
  // Three lines retain both identities when a model occurs in several harnesses.
  const names = entries.map(entry => [harnessDisplayName(entry.agent), '×', modelDisplayName(entry.engine_model)]);
  const nameFont = 10.5 * textScale * annotationScale * axisLabelScale;
  // Tighten inter-line whitespace so the larger desktop labels fit without
  // enlarging the SVG viewBox and cancelling their visible size increase.
  const lineHeight = nameFont * (desktopLayout.value ? .85 : 1.12);
  const angle = desktopLayout.value ? 50 : 75;
  const radians = angle * Math.PI / 180;
  const sin = Math.sin(radians), cos = Math.cos(radians);
  const nameWidths = names.map(lines => Math.max(...lines.map(name => name.length)) * nameFont * .6);
  const widestName = Math.max(...nameWidths);
  const nameAscent = nameFont * .85;
  const labelReach = widestName * cos + nameAscent * sin;
  const right = Math.ceil((desktopLayout.value ? 8 : 18) + labelReach);
  // Keep a single, aligned row. Reserve enough room for every rotated
  // three-line block before fitting the complete SVG to its panel.
  const labelThickness = 2 * lineHeight + nameFont;
  const minimumStep = (labelThickness + (desktopLayout.value ? 4 : 3)) / sin;
  const width = desktopLayout.value
    ? Math.max(chartWidth.value, left + right + count * minimumStep) : chartWidth.value;
  const labelDrop = widestName * sin + (2 * lineHeight + nameFont * .25) * cos;
  const height = desktopLayout.value ? Math.max(620,Math.min(820,width * .76))
    : Math.round(252 + 36 + labelDrop + 16);
  const baseline = desktopLayout.value ? height - Math.ceil(labelDrop + 52) : 252;
  const nameAnchor = baseline + 36;

  const step = (width - left - right) / Math.max(count, 1);
  const coreScale = desktopLayout.value ? Math.min(densityScale,step / 82) : densityScale;
  const plotHeight = baseline - top;
  const y = value => baseline - ((value - d0) / (d1 - d0)) * plotHeight;
  // ---- core geometry ----------------------------------------------------
  const radiusScale = desktopLayout.value ? 1.1 : 1;
  const rOut = 35 * coreScale * radiusScale; // outer ring = STS
  const innerScale = desktopLayout.value ? .8 : 1;
  const rVlr = 22 * coreScale * innerScale * radiusScale;
  const rMid = (rOut + rVlr) / 2; // distinct middle ring = GATE
  // Desktop VLR stroke has half-width 2.1×scale. Keep only 0.65×scale
  // clearance, and inscribe the square image in the backing circle so even
  // rectangular vendor marks cannot cover the innermost metric ring.
  const rHub = rVlr - (desktopLayout.value ? 2.75 : 5) * coreScale;
  // SVG user-unit geometry. Kept unrounded: the value goes straight into an
  // attribute, and rounding here would be a rendering tweak masquerading as a
  // display-precision decision (which `lib/precision.js` owns).
  const baseMarkSize = Math.max(12, Math.round(22 * coreScale));
  const markSize = desktopLayout.value ? rHub * Math.SQRT2 : baseMarkSize;
  const enlargeValue = desktopLayout.value
    && ['sts_token', 'budget_gm', 's_equiv', 'ebgs', 'speedup_mean'].includes(benchmark.rankMetric);
  const valueFont = 13 * textScale * annotationScale * (enlargeValue ? 1.1 : 1);
  const compactRatioLabels = desktopLayout.value
    && ['budget_gm', 's_equiv', 'ebgs', 'speedup_mean'].includes(benchmark.rankMetric);
  const valueHalo = (compactRatioLabels ? 3 : 5) * textScale;
  const grid = ticks.map(tickValue => {
    const tickY = y(tickValue);
    return `<path class="rank-field-rail" d="M${left} ${tickY}H${width - right}"/><rect class="rank-field-node" x="${left - 3}" y="${tickY - 3}" width="6" height="6"/><text class="chart-tick" style="font-size:${axisFont}px" x="${left - 11}" y="${tickY + 3}" text-anchor="end">${formatAxisValue(benchmark.rankMetric, tickValue)}</text>`;
  }).join('');
  const circuit = entries.slice(0, -1).map((entry, index) => {
    const x1 = left + step * index + step / 2;
    const x2 = left + step * (index + 1) + step / 2;
    const mid = (x1 + x2) / 2;
    return `<path class="rank-circuit" d="M${x1} ${baseline + 2}H${mid - 8}l8 8 8-8H${x2}"/>`;
  }).join('');
  const valueLabels = [];
  const configurationLabels = [];
  if (compactRatioLabels) {
    valueMeasureContext ??= document.createElement('canvas').getContext('2d');
    if (valueMeasureContext) valueMeasureContext.font = `760 ${valueFont}px Arial`;
  }
  const cores = entries.map((entry, index) => {
    const x = left + step * index + step / 2;
    const value = values[index];
    const muted = (entry.baseline_quality ?? 'normal') !== 'normal';
    const rings = (cy, active = true) => [
      metricRing('STS', entry.overall.sts_token, x, cy, rOut, 'var(--compute)', active),
      metricRing('GATE', entry.overall.gate_rate, x, cy, rMid, 'var(--gate-ring)', active),
      metricRing('VLR', entry.overall.vlr_mean, x, cy, rVlr, 'var(--cost)', active)
    ].join('');
    // Share the same starting edge at every viewport width. Centering each
    // line makes the rotated block's top depend on the length of its name.
    const labelLines = names[index].map((line, row) => `<tspan class="${row === 0 ? 'rank-axis-harness' : row === 1 ? 'rank-axis-cross' : 'rank-axis-model'}" x="0" y="${row * lineHeight}">${escapeHTML(line)}</tspan>`).join('');
    const label = `<g class="rank-axis-label" aria-label="${attributeEscape(configurationDisplayName(entry))}" transform="translate(${x} ${nameAnchor}) rotate(${angle})"><text class="chart-label rank-name-rotated" style="font-size:${nameFont}px" text-anchor="start">${labelLines}</text></g>
      <text class="chart-tick" style="font-size:${axisFont}px" x="${x}" y="${baseline + 18}" text-anchor="middle">${String(index + 1).padStart(2, '0')}</text>`;
    const selectedMetricTooltip = ['sts_token', 'gate_rate', 'vlr_mean'].includes(benchmark.rankMetric) ? ''
      : `<br>${metric.label}: <span class=&quot;tooltip-value&quot;>${formatRankValue(benchmark.rankMetric, value)}</span>${Number.isFinite(value) ? '' : ' Unavailable'}`;
    const tooltip = `<strong>#${index + 1} · ${escapeHTML(configurationDisplayName(entry))}</strong>${selectedMetricTooltip}<br>STS: ${formatSts(entry.overall.sts_token)} · GATE: ${formatPct(entry.overall.gate_rate)} · VLR: ${formatPct(entry.overall.vlr_mean)}${muted ? '<br>⬜ Invalid baseline; excluded from cross-operator aggregation' : ''}`;
    // Keep axis labels outside the moving cores so hover never breaks their
    // alignment. Retain the same tooltip and drawer interaction on the names.
    configurationLabels.push(`<g data-model-id="${entry.id}" data-open-drawer data-tooltip-html="${attributeEscape(tooltip)}">${label}</g>`);

    // An entry whose rank metric is missing has no position on the axis, so it
    // gets no orbit — drawing one from a sentinel coordinate would fly the whole
    // card out of the canvas (the bug this branch replaces). It keeps its tick,
    // its lane stub, its brand mark and its name so the row stays identifiable,
    // and says “no measured value” on hover instead of a fabricated rank.
    if (!Number.isFinite(value)) {
      const emptyY = baseline - rOut;
      return `<g class="rank-core rank-core-empty${muted ? ' rank-core-muted' : ''}" style="--core-scale:${coreScale}" data-model-id="${entry.id}" data-open-drawer data-tooltip-html="${attributeEscape(tooltip)}">
        <path class="rank-lane rank-lane-empty" d="M${x} ${baseline}V${emptyY + rOut + 3}" stroke="${entry.engine_model.color}"/>
        <circle class="rank-lane-node" cx="${x}" cy="${baseline}" r="${(3 * coreScale).toFixed(1)}" fill="${entry.engine_model.color}"/>
        ${rings(emptyY, false)}
        <circle class="core-hub" cx="${x}" cy="${emptyY}" r="${rHub}" fill="var(--surface)"/>
        ${vendorMarkSVG(entry, x, emptyY, markSize)}
        <text class="rank-value rank-value-empty" style="font-size:${valueFont}px" x="${x}" y="${emptyY - rOut - 11}" text-anchor="middle">–</text>
      </g>`;
    }

    const topY = y(value);
    const valueText = formatRankValue(benchmark.rankMetric,value);
    const measured = compactRatioLabels ? valueMeasureContext?.measureText(valueText) : null;
    const valueWidth = measured
      ? measured.width - valueFont * .04 * (valueText.length - 1)
      : valueText.length * valueFont * .64 + valueHalo;
    const ascent = measured?.fontBoundingBoxAscent ?? measured?.actualBoundingBoxAscent ?? valueFont * .8;
    const descent = measured?.fontBoundingBoxDescent ?? measured?.actualBoundingBoxDescent ?? valueFont * .2;
    let valueY = topY - rOut - 11;
    if (compactRatioLabels) {
      // Move only as far as the actual text bounds require. A fixed 26px jump
      // could overshoot a near miss twice and leave GM labels far from cores.
      let collisions;
      while ((collisions = valueLabels.filter(other =>
        Math.abs(x-other.x) < (valueWidth+other.width)/2 + 1
        && valueY - ascent - valueHalo/2 < other.y + other.descent + valueHalo/2 + 2
        && valueY + descent + valueHalo/2 > other.y - other.ascent - valueHalo/2 - 2
      )).length) {
        valueY = Math.min(...collisions.map(other => other.y - other.ascent - descent - valueHalo - 3));
      }
    } else {
      while (desktopLayout.value && valueLabels.some(other => Math.abs(x-other.x) < (valueWidth+other.width)/2 + 4
        && Math.abs(valueY-other.y) < valueFont + 8)) valueY -= valueFont + 10;
    }
    valueLabels.push({x,y:valueY,width:valueWidth,ascent,descent});
    return `<g class="rank-core${muted ? ' rank-core-muted' : ''}" style="--core-scale:${coreScale}" data-model-id="${entry.id}" data-open-drawer data-tooltip-html="${attributeEscape(tooltip)}">
      <path class="rank-lane" d="M${x} ${baseline}V${topY + rOut + 3}" stroke="${entry.engine_model.color}"/>
      <circle class="rank-lane-node" cx="${x}" cy="${baseline}" r="${(3 * coreScale).toFixed(1)}" fill="${entry.engine_model.color}"/>
      ${rings(topY)}
      <circle class="core-hub" cx="${x}" cy="${topY}" r="${rHub}" fill="var(--surface)"/>
      ${vendorMarkSVG(entry, x, topY, markSize)}
      <text class="rank-value" style="font-size:${valueFont}px;stroke-width:${valueHalo}px" x="${x}" y="${valueY}" text-anchor="middle">${valueText}</text>
    </g>`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" data-canvas-width="${width}" data-canvas-height="${height}" role="img" aria-label="${attributeEscape(props.metricLabel)} ranking for ${entries.length} entries">
    <defs><filter id="core-glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
    <g class="rank-plot-content">${grid}${circuit}<line class="chart-axis" x1="${left}" y1="${baseline}" x2="${width - right}" y2="${baseline}"/>
    ${cores}<g class="rank-axis-labels">${configurationLabels.join('')}</g>
    </g>
  </svg>`;
});
watch(svg,() => nextTick(fitContent),{flush:'post'});
</script>
