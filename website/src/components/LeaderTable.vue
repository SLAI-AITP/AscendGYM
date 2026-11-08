<template>
  <table class="leader-table" id="leader-table">
    <caption class="sr-only">AscendGYM rankings by harness × model</caption>
    <colgroup><col><col><col><col><col><col><col><col><col><col></colgroup>
    <thead><tr>
      <th scope="col">Rank</th>
      <th scope="col" class="entry-col">Harness × Model</th>
      <th scope="col"><button type="button" @click="sortBy('sts_token')">STS{{ sortIndicator('sts_token') }}</button></th>
      <th scope="col"><button type="button" @click="sortBy('ebgs')">GM Speedup{{ sortIndicator('ebgs') }}</button></th>
      <th scope="col"><button type="button" @click="sortBy('vlr_mean')">VLR{{ sortIndicator('vlr_mean') }}</button></th>
      <th scope="col"><button type="button" @click="sortBy('task_pass_rate')">Pass Rate{{ sortIndicator('task_pass_rate') }}</button></th>
      <th scope="col"><button type="button" @click="sortBy('tokens_total_k')">Cumulative Tokens{{ sortIndicator('tokens_total_k') }}</button></th>
      <th scope="col"><button type="button" @click="sortBy('rounds_median')">Rounds{{ sortIndicator('rounds_median') }}</button></th>
      <th scope="col"><button type="button" @click="sortBy('wall_time_median_s')">Wall Time{{ sortIndicator('wall_time_median_s') }}</button></th>
      <th scope="col">Speedup × Token</th>
    </tr></thead>
    <tbody>
      <tr
        v-for="(entry, index) in displayedEntries"
        :key="entry.id"
        class="leader-row"
        :class="{ 'row-muted': entry.baseline_quality !== 'normal' }"
        :data-model-id="entry.id"
        data-open-drawer
        tabindex="0"
      >
        <td><span class="rank-badge" :class="{ top: startIndex + index < 3 }">{{ startIndex + index + 1 }}</span></td>
        <td class="entry-col"><EntryLabel :entry="entry" size="sm" stacked-icons :show-agent-badge="false" :show-vendor-in-subtitle="false" /></td>
        <td class="score-cell">{{ formatSts(entry.overall.sts_token) }}<small v-if="entry.sts_coverage &lt; entry.coverage_operators" class="evidence-tag">{{ entry.sts_coverage }}/{{ entry.coverage_operators }} operators</small></td>
        <td :class="speedupClass(entry.overall.ebgs)">{{ formatSpeedup(entry.overall.ebgs) }}</td>
        <td :class="speedupClass(1 + (entry.overall.vlr_mean ?? 0) / 100)">{{ formatPct(entry.overall.vlr_mean) }}</td>
        <td :class="{ 'tone-bad': (entry.overall.task_pass_rate ?? 100) < 100 }">{{ formatPct(entry.overall.task_pass_rate) }}</td>
        <td>{{ formatTokensK(entry.overall.tokens_total_k) }}</td>
        <td>{{ entry.overall.rounds_median != null ? formatInteger(entry.overall.rounds_median) : '–' }}</td>
        <td>{{ formatWallHours(entry.overall.wall_time_median_s) }}</td>
        <td><Sparkline :points="primaryOperator(entry).anytime" :color="entry.engine_model.color" /></td>
      </tr>
    </tbody>
  </table>

  <div class="leader-cards" id="leader-cards" aria-label="Mobile leaderboard cards">
    <article
      v-for="(entry, index) in displayedEntries"
      :key="entry.id"
      class="leader-card"
      :class="{ 'row-muted': entry.baseline_quality !== 'normal' }"
      :data-model-id="entry.id"
      data-open-drawer
      tabindex="0"
    >
      <div class="leader-card-head">
        <div class="model-label">
          <span class="rank-badge" :class="{ top: startIndex + index < 3 }">{{ startIndex + index + 1 }}</span>
          <EntryLabel :entry="entry" size="sm" stacked-icons :show-agent-badge="false" :show-vendor-in-subtitle="false" />
        </div>
        <div class="leader-card-score">{{ formatSts(entry.overall.sts_token) }}<small>STS</small><small v-if="entry.sts_coverage &lt; entry.coverage_operators" class="evidence-tag">{{ entry.sts_coverage }}/{{ entry.coverage_operators }} operators</small></div>
      </div>
      <div class="leader-card-grid">
        <div><span>VLR</span><b :class="speedupClass(1 + (entry.overall.vlr_mean ?? 0) / 100)">{{ formatPct(entry.overall.vlr_mean) }}</b></div>
        <div><span>Pass Rate</span><b>{{ formatPct(entry.overall.task_pass_rate) }}</b></div>
        <div><span>Tokens</span><b>{{ formatTokensK(entry.overall.tokens_total_k) }}</b></div>
      </div>
    </article>
  </div>
  <TablePagination v-model:page="page" :total="total" :page-size="pageSize" label="STS table pagination" />
</template>

<script setup>
import { computed, ref } from 'vue';
import TablePagination from './TablePagination.vue';
import { usePagination } from '../lib/usePagination';
import EntryLabel from './EntryLabel.vue';
import Sparkline from './Sparkline.vue';
import { formatTokensK, formatWallHours, formatSts, formatSpeedup, formatPct, formatInteger, speedupClass } from '../lib/format';
import { metricMeta } from '../lib/meta';

const props = defineProps({
  entries: { type: Array, required: true }
});

// Table ordering is independent of the chart's RE-RANK selection.
const sortMetric = ref('sts_token');
const sortedEntries = computed(() => {
  const direction = metricMeta[sortMetric.value]?.direction === 'down' ? 1 : -1;
  return [...props.entries].sort((a, b) => {
    const av = a.overall?.[sortMetric.value], bv = b.overall?.[sortMetric.value];
    if (!Number.isFinite(av)) return Number.isFinite(bv) ? 1 : 0;
    if (!Number.isFinite(bv)) return -1;
    return (av - bv) * direction
      || ((b.overall?.sts_token ?? -Infinity) - (a.overall?.sts_token ?? -Infinity));
  });
});
const { page, total, pageSize, startIndex, pagedItems: displayedEntries } = usePagination(sortedEntries);
function sortIndicator(metric) {
  return sortMetric.value === metric ? (metricMeta[metric]?.direction === 'down' ? ' ↑' : ' ↓') : '';
}
function sortBy(metric) {
  sortMetric.value = metric;
}

// Precision is decided centrally in `lib/precision.js`; this component only
// chooses *which* formatter each column needs, never a decimal count.

// The leaderboard sparkline uses the entry's highest-token operator
// (where the agent spent the most effort).
function primaryOperator(entry) {
  return [...(entry.per_operator ?? [])].sort((a, b) => (b.tokens_k ?? 0) - (a.tokens_k ?? 0))[0] ?? { anytime: [] };
}
</script>
