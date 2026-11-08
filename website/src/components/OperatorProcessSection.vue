<template>
  <section class="section page-shell" id="operators" aria-labelledby="operators-title">
    <div class="section-heading kernel-panel-heading">
      <div><p class="section-index">{{ section.index }} / {{ section.kicker }}</p><h2 id="operators-title">{{ section.title }}</h2></div>
      <BenchmarkSelect id="heat-metric" v-model="benchmark.heatMetric" label="Metric" :options="metricOptions" />
      <p v-for="(line, i) in sectionSummary(section)" :key="i" class="kernel-panel-description">{{ line }}</p>
    </div>
    <div class="component-surface heatmap-wrap paginated-surface" id="process-heatmap">
      <div ref="heatScroller" id="kernel-results-scroll-area" class="pagination-table-scroll" role="region" aria-label="Scrollable kernel results" tabindex="0" @scroll.passive="syncScrollPosition">
      <div v-if="visibleEntries.length" class="heatmap" role="grid" :aria-label="`${activeMeta.label} by entry and operator`" :style="{ gridTemplateColumns: `var(--process-label-width, 220px) repeat(${operators.length}, minmax(var(--process-cell-width, 120px), 1fr))` }">
        <div class="heat-head" role="columnheader"><span>Harness × Model</span></div>
        <div v-for="operatorName in operators" :key="operatorName" class="heat-head" role="columnheader" :data-operator="operatorName"><span>{{ kernelHeaderName(operatorName) }}</span></div>
        <template v-for="entry in pagedEntries" :key="entry.id">
          <div class="heat-row-label" role="rowheader" :data-model-id="entry.id" data-open-drawer>
            <EntryLabel
              :entry="entry"
              size="sm"
              stacked-icons
              :show-agent-badge="false"
              :show-vendor-in-subtitle="false"
            />
          </div>
          <div
            v-for="operatorName in operators"
            :key="`${entry.id}-${operatorName}`"
            class="heat-cell"
            role="gridcell"
            :class="{ 'heat-cell-empty': cellValue(entry, operatorName) == null }"
            :data-model-id="entry.id"
            :data-operator="operatorName"
            :style="cellStyle(entry, operatorName)"
            :data-tooltip-html="cellTooltip(entry, operatorName)"
          >
            <span v-if="cellValue(entry, operatorName) != null" class="heat-value">{{ formatHeatValue(entry, operatorName) }}</span>
            <span v-else class="heat-missing">Unavailable</span>
          </div>
        </template>
      </div>
      </div>
      <input
        v-if="maxHorizontalScroll > 0"
        class="kernel-horizontal-scrollbar"
        type="range"
        min="0"
        :max="maxHorizontalScroll"
        step="1"
        :value="horizontalScroll"
        :style="{ '--kernel-scroll-thumb-width': `${scrollThumbWidth}px` }"
        aria-label="Scroll kernel results horizontally"
        aria-controls="kernel-results-scroll-area"
        :aria-valuetext="`${Math.round(horizontalScroll / maxHorizontalScroll * 100)}% scrolled`"
        @input="scrollFromSlider"
      >
      <TablePagination v-model:page="page" :total="total" :page-size="pageSize" label="Kernel Panel table pagination" />
    </div>
  </section>
</template>

<script setup>
import { harnessDisplayName, modelDisplayName } from '../lib/displayNames';
import { computed, ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue';
import TablePagination from './TablePagination.vue';
import { usePagination } from '../lib/usePagination';
import EntryLabel from './EntryLabel.vue';
import BenchmarkSelect from './BenchmarkSelect.vue';
import { benchmark, visibleEntries, operators } from '../store/benchmark';
import { SECTIONS, sectionSummary } from '../lib/sections';
import { operatorProcessValue } from '../lib/operatorProcess';

const section = SECTIONS.operators;
import { processMeta } from '../lib/meta';
import { formatInteger, qualityColor, formatWallTime, formatTokensK } from '../lib/format';

const activeMeta = computed(() => processMeta[benchmark.heatMetric]);
const metricOptions = Object.entries(processMeta).map(([value, meta]) => ({ value, label: meta.label }));
const { page, total, pageSize, pagedItems: pagedEntries } = usePagination(visibleEntries);

const heatScroller = ref(null);
const horizontalScroll = ref(0);
const scrollViewportWidth = ref(0);
const scrollContentWidth = ref(0);
const maxHorizontalScroll = computed(() => Math.max(0, scrollContentWidth.value - scrollViewportWidth.value));
const scrollThumbWidth = computed(() => Math.min(scrollViewportWidth.value,
  Math.max(36, scrollViewportWidth.value ** 2 / Math.max(1, scrollContentWidth.value))));
let scrollResizeObserver;
function syncScrollPosition() {
  horizontalScroll.value = heatScroller.value?.scrollLeft ?? 0;
}
function updateScrollMetrics() {
  const element = heatScroller.value;
  if (!element) return;
  scrollViewportWidth.value = element.clientWidth;
  scrollContentWidth.value = element.scrollWidth;
  syncScrollPosition();
}
function scrollFromSlider(event) {
  if (heatScroller.value) heatScroller.value.scrollLeft = Number(event.target.value);
  syncScrollPosition();
}
onMounted(() => {
  scrollResizeObserver = new ResizeObserver(updateScrollMetrics);
  scrollResizeObserver.observe(heatScroller.value);
  updateScrollMetrics();
});
watch([operators, pagedEntries], async () => {
  await nextTick();
  updateScrollMetrics();
}, { flush: 'post' });
onBeforeUnmount(() => scrollResizeObserver?.disconnect());

// Format only the heading; measurement lookups keep the original kernel IDs.
function kernelHeaderName(name) {
  return name.split('_').map(part => part.charAt(0).toUpperCase() + part.slice(1)).join('');
}

const extent = computed(() => {
  const values = visibleEntries.value.flatMap(entry => (
    entry.per_operator
      .map(op => operatorProcessValue(op, benchmark.heatMetric))
      .filter(Number.isFinite)
  ));
  if (!values.length) return { min: 0, max: 0 };
  return { min: Math.min(...values), max: Math.max(...values) };
});

function cellValue(entry, operatorName) {
  const item = entry.per_operator.find(op => op.operator === operatorName);
  return operatorProcessValue(item, benchmark.heatMetric);
}

function cellQuality(entry, operatorName) {
  const value = cellValue(entry, operatorName);
  if (value == null) return null;
  const { min, max } = extent.value;
  return max === min ? .5 : 1 - (value - min) / (max - min);
}

function cellStyle(entry, operatorName) {
  const quality = cellQuality(entry, operatorName);
  if (quality == null) return {};
  // No `toFixed` here: `--heat-level` is a CSS interpolation ramp key, not a
  // displayed measurement, so rounding it is a rendering concern.
  return { '--heat-color': qualityColor(quality), '--heat-level': quality };
}

function formatHeatValue(entry, operatorName) {
  const value = cellValue(entry, operatorName);
  const key = benchmark.heatMetric;
  if (key === 'tokens_k') return formatTokensK(value);
  if (key === 'wall_time_s') return formatWallTime(value);
  return formatInteger(value);
}

/**
 * Unit suffix for the tooltip.
 *
 * The token and wall-time formatters already embed their magnitude ("283K") or
 * suffix ("2918.5s"), so appending `unit` as well would render "283K k tokens".
 * Only the bare-integer metrics need the unit spelled out.
 */
function heatUnitSuffix() {
  const key = benchmark.heatMetric;
  if (key === 'tokens_k' || key === 'wall_time_s') return '';
  return processMeta[key]?.unit ?? '';
}

function cellTooltip(entry, operatorName) {
  const meta = activeMeta.value;
  if (cellValue(entry, operatorName) == null) {
    return `<strong>${harnessDisplayName(entry.agent)} × ${modelDisplayName(entry.engine_model)} · ${operatorName}</strong><br><small>No measurement available in the selected source records</small>`;
  }
  const unit = heatUnitSuffix();
  return `<strong>${harnessDisplayName(entry.agent)} × ${modelDisplayName(entry.engine_model)} · ${operatorName}</strong><br>${meta.label}: <span class="tooltip-value">${formatHeatValue(entry, operatorName)}${unit ? ` ${unit}` : ''}</span><br><small>Lower is better</small>`;
}
</script>
