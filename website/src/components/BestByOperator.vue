<template>
  <div class="best-list">
    <div v-if="winners.length && !winners.some(row => row.op)" class="best-row"><span>–</span><span class="best-empty">No measured operators</span></div>
    <div
      v-for="row in pagedWinners"
      :key="row.operator"
      class="best-row"
      :data-model-id="row.entry.id"
      data-open-drawer
    >
      <span>{{ row.operator }}</span>
      <EntryLabel :entry="row.entry" size="sm" stacked-icons :show-agent-badge="false" :show-vendor-in-subtitle="false" />
      <span class="best-values">
        <span class="data-value" :class="speedupClass(row.op.s_bmax)">{{ row.op.s_bmax != null ? formatSpeedupDetail(row.op.s_bmax) : '–' }}</span>
        <small>Speedup @50M · {{ row.op.s_bmax ? formatSpeedup(row.op.s_bmax) : '–' }}</small>
        <span v-if="row.op.flags?.includes('fake_optimization')" class="flag-dot" data-level="red" aria-label="fake optimization">🟥</span>
      </span>
    </div>
  </div>
  <TablePagination v-model:page="page" :total="total" :page-size="pageSize" label="Best kernel results pagination" />
</template>

<script setup>
import { computed } from 'vue';
import TablePagination from './TablePagination.vue';
import { usePagination } from '../lib/usePagination';
import EntryLabel from './EntryLabel.vue';
import { operators, fallbackEntries } from '../store/benchmark';
import { speedupClass, paosScore, formatSpeedup, formatSpeedupDetail } from '../lib/format';

// Honest winner: highest PAOS = sqrt(PKAI x PTEI) (not the representative
// speedup), so a fake-optimization run cannot top the list on a single number.
const winners = computed(() => {
  const entries = fallbackEntries.value;
  return operators.value.map(operatorName => {
    const candidates = entries
      .map(entry => ({ entry, op: entry.per_operator.find(item => item.operator === operatorName) }))
      .filter(pair => pair.op && pair.op.baseline_quality === 'normal' && pair.op.s_bmax != null)
      .sort((a, b) => b.op.s_bmax - a.op.s_bmax);
    const winner = candidates[0];
    return winner ? { operator: operatorName, ...winner } : { operator: operatorName, entry: null, op: null };
  });
});
const measuredWinners = computed(() => winners.value.filter(row => row.op));
const { page, total, pageSize, pagedItems: pagedWinners } = usePagination(measuredWinners, row => `${row.operator}:${row.entry.id}`);
</script>
