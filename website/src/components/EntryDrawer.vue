<template>
  <aside class="model-drawer" :class="{ 'is-open': isOpen }" :aria-hidden="String(!isOpen)" aria-labelledby="drawer-identity">
    <button class="drawer-backdrop" type="button" aria-label="Close entry detail" @click="closeDrawer"></button>
    <div class="drawer-panel" role="dialog" aria-modal="true" aria-labelledby="drawer-identity">
      <div class="drawer-header">
        <div id="drawer-identity" class="drawer-identity">
          <EntryLabel v-if="entry" :entry="entry" size="lg" stacked-icons :show-agent-badge="false" :show-vendor-in-subtitle="false" />
        </div>
        <button ref="closeButton" class="icon-button" type="button" aria-label="Close entry detail" @click="closeDrawer"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg></button>
      </div>
      <div class="drawer-scroll" id="drawer-content">
        <template v-if="entry">
          <!-- ④ anytime full chart -->
          <section class="drawer-section">
            <h3 class="kernel-speedup-title">Kernel Speedup</h3>
            <div class="kernel-chart-controls">
              <label id="drawer-kernel-label" for="drawer-kernel">Kernel</label>
              <KernelSelect v-model="anyOperator" :options="entry.per_operator.map(op => op.operator)" :active="isOpen" />
              <div class="segmented kernel-axis-toggle" role="group" aria-label="Kernel speedup axis">
                <button type="button" :class="{ 'is-active': anyAxis === 'token' }" @click="anyAxis = 'token'">Token</button>
                <button type="button" :class="{ 'is-active': anyAxis === 'time' }" @click="anyAxis = 'time'">Wall Time</button>
              </div>
            </div>
            <div v-if="anytimeSvg" class="drawer-anytime" v-html="anytimeSvg"></div>
            <p v-else class="drawer-note kernel-trajectory-empty" role="status">{{ anyAxis === 'time' ? 'Wall Time' : 'Token' }} trajectory is unavailable for this kernel in the source records.</p>
          </section>

          <!-- ⑤ per-operator table with expansion -->
          <section class="drawer-section">
            <h3 class="kernel-speedup-title">Kernel results</h3>
            <p class="shape-caption">Select a row to view supported shapes.</p>
            <div class="table-wrap kernel-result-wrap">
              <table class="drawer-op-table">
                <colgroup><col v-for="column in 6" :key="column" /></colgroup>
                <thead><tr><th>Kernel</th><th>STS</th><th>VLR</th><th>Tokens</th><th>Rounds</th><th class="kernel-trajectory-heading" title="Best-so-far speedup versus cumulative tokens (log scale)">Speed Up</th></tr></thead>
                <tbody>
                  <template v-for="op in entry.per_operator" :key="op.operator">
                    <tr class="op-row-toggle" :class="{ 'row-muted': op.baseline_quality !== 'normal' }" @click="expandedOp = expandedOp === op.operator ? null : op.operator">
                      <td><button type="button" class="kernel-shape-open" :title="op.operator" :aria-expanded="expandedOp === op.operator" @click.stop="expandedOp = expandedOp === op.operator ? null : op.operator"><span class="kernel-shape-name">{{ op.operator }}</span><span aria-hidden="true">{{ expandedOp === op.operator ? '⌄' : '›' }}</span></button><span v-if="op.baseline_quality !== 'normal'" class="flag-dot" data-level="gray"> ⬜</span></td>
                      <td>{{ formatSts(op.sts_token) }}</td>
                      <td :class="speedupClass(1 + (op.vlr ?? 0) / 100)">{{ formatSignedPct(op.vlr) }}</td>
                      <td>{{ formatTokensK(op.tokens_k) }}</td>
                      <td>{{ op.rounds != null ? formatInteger(op.rounds) : '–' }}</td>
                      <td><Sparkline :points="op.anytime" :color="entry.engine_model.color" preserveAspectRatio="none" label="Best-so-far speedup versus cumulative tokens (log scale)" /></td>
                    </tr>
                    <tr v-if="expandedOp === op.operator" class="shape-expansion-row"><td colspan="6" class="op-expand-cell"><KernelShapeDetails :operator="op" /></td></tr>
                  </template>
                </tbody>
              </table>
            </div>
          </section>

        </template>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed, ref, watch, nextTick } from 'vue';
import EntryLabel from './EntryLabel.vue';
import Sparkline from './Sparkline.vue';
import KernelShapeDetails from './KernelShapeDetails.vue';
import KernelSelect from './KernelSelect.vue';
import { ui, closeDrawer } from '../store/ui';
import { boardMeta, findEntryById } from '../store/benchmark';
import {
  formatSpeedup, formatTokensK, formatWallHours, formatSts, formatSignedPct,
  formatInteger, speedupClass, attributeEscape
} from '../lib/format';

const isOpen = computed(() => Boolean(ui.drawerModelId));

// Keep the last opened entry rendered while the close transition fades out.
const lastEntryId = ref(null);
watch(() => ui.drawerModelId, id => { if (id) lastEntryId.value = id; });

const entry = computed(() => {
  const id = ui.drawerModelId || lastEntryId.value;
  return findEntryById(id);
});

const expandedOp = ref(null);
const anyAxis = ref('token');
const anyOperator = ref(null);
watch(() => ui.drawerModelId, id => {
  expandedOp.value = null;
  const found = findEntryById(id);
  if (found) {
    anyOperator.value = [...found.per_operator].sort((a, b) => (b.tokens_k ?? 0) - (a.tokens_k ?? 0))[0]?.operator ?? null;
    anyAxis.value = 'token';
  }
});

// VLR target tiers come from the evaluator itself (meta.targets → ['1%','5%','10%']).
// Nothing is hard-coded here, so a changed `--targets` flows straight through.
const tiers = computed(() => boardMeta.value?.targets ?? []);
function tierValue(map, tier) {
  const value = map?.[tier];
  return Number.isFinite(value) ? value : null;
}
// ------------------------- ④ anytime SVG -------------------------
const anyOp = computed(() => (entry.value?.per_operator ?? []).find(op => op.operator === anyOperator.value) ?? entry.value?.per_operator?.[0] ?? null);

const anytimeSvg = computed(() => {
  const op = anyOp.value;
  if (!op || !entry.value) return '';
  const width = 700;
  const height = 340;
  const margin = { top: 30, right: 50, bottom: 56, left: 90 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const key = anyAxis.value === 'time' ? 'wall_sec' : 'cum_tokens_k';
  const tokenMode = anyAxis.value === 'token';
  const series = !tokenMode && op.anytime_time?.length ? op.anytime_time : (op.anytime ?? []);
  const passed = series.filter(p => Number.isFinite(p[key]) && p[key]>=0 && Number.isFinite(p.s) && p.s>0).sort((a,b) => a[key]-b[key]);
  if (!passed.length) return '';
  const positive = passed.filter(p => p[key]>0);
  const referenceOnly = !positive.length;
  const logMode = tokenMode && !referenceOnly;
  // A source with only (0, 1x) gets a reference point at zero, never a made-up
  // consumed budget. The unit domain is geometry, not another data point.
  const t1 = referenceOnly ? 1 : positive[positive.length-1][key];
  // Keep the observed endpoint fixed for single-point and baseline-only data.
  // Zero references occupy the left edge of the logarithmic token plot.
  const first = referenceOnly ? 0 : positive[0][key];
  const t0 = logMode ? (passed[0][key]===0 || first===t1 ? first/10 : first) : 0;
  const sMax = Math.max(1.04, ...passed.map(p => p.s)) * 1.04;
  const x = t => margin.left + (logMode
    ? (Math.log10(Math.max(t,t0))-Math.log10(t0))/(Math.log10(t1)-Math.log10(t0))
    : t/t1) * plotW;
  const y = s => margin.top + (1 - (s - 1) / (sMax - 1)) * plotH;

  // ticks
  const xTicks = referenceOnly ? [0] : logMode
    ? [...new Set([t0,...positive.map(p => p[key]).filter((v,i,arr) => i===0 || i===arr.length-1 || arr.length<=6)])]
    : [t0, (t0 + t1) / 2, t1];
  const xTickSvg = xTicks.map(t => `<line class="chart-tick-line" x1="${x(t)}" y1="${height - margin.bottom}" x2="${x(t)}" y2="${height - margin.bottom + 6}"/><text class="chart-tick" x="${x(t)}" y="${height - margin.bottom + 20}" text-anchor="middle">${tokenMode ? formatTokensK(t) : formatWallHours(t)}</text>`).join('');
  const yTicks = [1, 1 + (sMax - 1) / 2, sMax].map(s => `<line class="chart-tick-line" x1="${margin.left - 5}" y1="${y(s)}" x2="${margin.left}" y2="${y(s)}"/><text class="chart-tick" x="${margin.left - 9}" y="${y(s) + 3}" text-anchor="end">${formatSpeedup(s)}</text>`).join('');

  // Δ vertical lines (CTG-Δ on the token axis only)
  let deltaLines = '';
  if (tokenMode) {
    for (const tier of tiers.value) {
      const value = tierValue(op.tokens_to, tier);
      if (value == null) continue;
      const tx = x(value / 1000);
      deltaLines += `<line class="delta-line" x1="${tx}" y1="${margin.top}" x2="${tx}" y2="${height - margin.bottom}"/><text class="delta-label" x="${tx}" y="${margin.top - 6}" text-anchor="middle">Δ${tier}</text>`;
    }
  }

  // best-so-far step line
  const steps = passed.map((point, index) => {
    const px = x(point[key]);
    const py = y(point.s);
    if (!index) return `M${px.toFixed(1)},${py.toFixed(1)}`;
    return `L${px.toFixed(1)},${y(passed[index - 1].s).toFixed(1)} L${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');

  // failed gray dots: position from duration (unverified), hollow style
  const failed = (op.anytime_failed ?? []).map(point => {
    const t = point[key];
    if (!t || t < t0) return '';
    const s = Math.min(sMax, op.baseline_ms ? op.baseline_ms / point.duration_ms : 1);
    const px = x(t);
    const py = y(s);
    const tooltip = attributeEscape(`<strong>Failed candidate (unverified)</strong><br>${point.status}<br>${tokenMode ? `${formatTokensK(t)} tokens` : formatWallHours(t)} · ${point.duration_ms} ms`);
    return `<circle class="failed-dot" cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="3.5" data-tooltip-html="${tooltip}"/>`;
  }).join('');

  const color = entry.value.engine_model.color;
  return `<svg viewBox="0 0 ${width} ${height}" role="img" data-axis="${tokenMode ? 'token' : 'time'}" data-source-run-id="${attributeEscape(op.trajectory_run_id ?? '')}" aria-label="Kernel speedup over ${tokenMode ? 'cumulative tokens' : 'wall time'}">
    ${xTickSvg}${yTicks}
    <line class="chart-axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"/>
    <line class="chart-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"/>
    ${deltaLines}
    <path class="anytime-area" d="${steps} L${x(t1).toFixed(1)},${y(1)} L${x(t0).toFixed(1)},${y(1)} Z" style="fill:color-mix(in srgb, ${color} 8%, transparent)"/>
    <path class="anytime-line" d="${steps}" style="stroke:${color}"/>
    ${failed}
    <circle cx="${x(passed[passed.length-1][key]).toFixed(1)}" cy="${y(passed[passed.length - 1].s).toFixed(1)}" r="4" style="fill:${color}"/>
    <text class="chart-tick" x="${margin.left + plotW / 2}" y="${height - 10}" text-anchor="middle">${tokenMode ? 'Cumulative tokens' : 'Cumulative wall time (h)'}</text>
  </svg>`;
});

const closeButton = ref(null);
watch(isOpen, async open => {
  if (open) {
    await nextTick();
    closeButton.value?.focus();
  }
});
</script>
