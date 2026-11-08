<template>
  <section class="section page-shell" id="rubric" aria-labelledby="rubric-title">
    <div class="section-heading">
      <div><p class="section-index">{{ section.index }} / {{ section.kicker }}</p>
        <h2 id="rubric-title">{{ section.title }}</h2>
        <!-- The summary is a FUNCTION here: it quotes the live column count so
             the copy can never claim more signals than the table renders. -->
        <p v-for="(line, i) in sectionSummary(section, { columns: columns.length })" :key="i">{{ line }}</p></div>
      <div class="ramp-legend"><span>GOOD</span><i></i><span>BAD</span></div>
    </div>
    <div class="component-surface rubric-wrap paginated-surface" id="rubric-table">
      <div class="pagination-table-scroll">
      <table v-if="visibleEntries.length" class="rubric-table">
        <caption class="sr-only">Color-coded entry rubric (v2 signals)</caption>
        <thead><tr>
          <th>Entry</th>
          <th v-for="column in columns" :key="column.key" :class="{ 'col-context': column.context }">
            {{ column.label }}<button class="metric-help" type="button" :aria-label="column.definition" :data-tooltip-html="`<strong>${column.label}</strong><br>${column.definition}`">?</button>
          </th>
        </tr></thead>
        <tbody>
          <tr
            v-for="entry in pagedEntries"
            :key="entry.id"
            :data-model-id="entry.id"
            data-open-drawer
            :class="{ 'row-muted': entry.baseline_quality !== 'normal' }"
          >
            <td><EntryLabel :entry="entry" size="sm" :show-agent-badge="false" :show-vendor-in-subtitle="false" /></td>
            <td v-for="column in columns" :key="column.key" :class="{ 'td-context': column.context }">
              <span
                class="rubric-signal"
                :class="{ 'rubric-score': column.key === 'sts_token', 'rubric-context': column.context || value(entry, column.key) == null }"
                :style="column.context || value(entry, column.key) == null ? {} : { '--quality-color': qualityColor(quality(entry, column.key)) }"
              >
                <span class="rubric-bars" aria-hidden="true"><i v-for="segment in 5" :key="segment" :class="{ 'is-on': segment <= litSegments(entry, column.key) }"></i></span>
                <b>{{ display(entry, column.key) }}</b>
              </span>
            </td>
          </tr>
        </tbody>
      </table>
      </div>
      <TablePagination v-model:page="page" :total="total" :page-size="pageSize" label="Metrics table pagination" />
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue';
import TablePagination from './TablePagination.vue';
import { usePagination } from '../lib/usePagination';
import EntryLabel from './EntryLabel.vue';
import { visibleEntries } from '../store/benchmark';
import { SECTIONS, sectionSummary } from '../lib/sections';

const section = SECTIONS.rubric;
import { metricMeta, rubricColumns } from '../lib/meta';
import { formatMetric, qualityColor, formatSts, formatRatio, formatTokensK, formatWallTime } from '../lib/format';

const columns = rubricColumns;
const { page, total, pageSize, pagedItems: pagedEntries } = usePagination(visibleEntries);

function value(entry, key) {
  if (key === 'vec_ratio_best' || key === 'u_best') {
    const ops = entry.per_operator.filter(op => (op.baseline_quality ?? 'normal') === 'normal');
    const values = ops.map(op => op.utilization?.[key]).filter(Number.isFinite);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  }
  return entry.overall?.[key] ?? null;
}

function display(entry, key) {
  const v = value(entry, key);
  if (v == null) return '–';
  if (key === 'sts_token') return formatSts(v);
  // Vec / U are utilization ratios (0–1), so they follow the ratio tier.
  if (key === 'vec_ratio_best' || key === 'u_best') return formatRatio(v);
  if (key === 'tokens_total_k') return formatTokensK(v);
  if (key === 'wall_time_median_s') return formatWallTime(v);
  return formatMetric(v, key, false);
}

function range(key) {
  const values = visibleEntries.value.map(entry => value(entry, key)).filter(Number.isFinite);
  if (!values.length) return [0, 0];
  return [Math.min(...values), Math.max(...values)];
}

function quality(entry, key) {
  const v = value(entry, key);
  if (v == null) return .5;
  const [min, max] = range(key);
  const meta = metricMeta[key];
  let score = max === min ? .5 : (v - min) / (max - min);
  if (meta?.direction === 'down') score = 1 - score;
  return score;
}

function litSegments(entry, key) {
  if (value(entry, key) == null) return 0;
  return Math.max(1, Math.round(quality(entry, key) * 5));
}
</script>
