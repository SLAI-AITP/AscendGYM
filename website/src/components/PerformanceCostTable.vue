<template>
  <div class="pagination-table-scroll">
    <table class="scatter-details-table" id="scatter-details-table">
      <caption class="sr-only">Performance and cost by configuration</caption>
      <colgroup><col><col><col><col><col></colgroup>
      <thead><tr>
        <th scope="col">Rank</th>
        <th scope="col" class="entry-col">Harness × Model</th>
        <th scope="col">GM Speedup</th>
        <th scope="col">Cumulative Tokens</th>
        <th scope="col">API Costs</th>
      </tr></thead>
      <tbody>
        <tr v-for="(entry,index) in pagedEntries" :key="entry.id" :data-model-id="entry.id" data-open-drawer tabindex="0" @keydown.enter.prevent="openDrawer(entry.id)" @keydown.space.prevent="openDrawer(entry.id)">
          <td><span class="rank-badge" :class="{ top: startIndex + index < 3 }">{{ startIndex + index + 1 }}</span></td>
          <td class="entry-col"><EntryLabel :entry="entry" size="sm" stacked-icons :show-agent-badge="false" :show-vendor-in-subtitle="false" /></td>
          <td>{{ formatSpeedup(entry.overall.ebgs) }}</td>
          <td>{{ formatTokensK(entry.overall.tokens_total_k) }}</td>
          <td>{{ entry.overall.api_cost_kind === 'scenario_estimate' ? '≈ ' : '' }}{{ formatUsd(entry.overall.api_cost_usd) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
  <TablePagination v-model:page="page" :total="total" :page-size="pageSize" label="Performance and cost table pagination" />
</template>

<script setup>
import TablePagination from './TablePagination.vue';
import EntryLabel from './EntryLabel.vue';
import { usePagination } from '../lib/usePagination';
import { visibleEntries } from '../store/benchmark';
import { openDrawer } from '../store/ui';
import { formatTokensK, formatUsd, formatSpeedup } from '../lib/format';

// Keep the chart's entry numbering when moving between table pages.
const { page, total, pageSize, startIndex, pagedItems: pagedEntries } = usePagination(visibleEntries);
</script>
