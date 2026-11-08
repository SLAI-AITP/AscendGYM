<template>
  <section class="section page-shell" id="quality-cost" aria-labelledby="scatter-title">
    <div class="section-heading">
      <div>
        <p class="section-index">{{ section.kicker }}</p>
        <h2 id="scatter-title">{{ section.title }}</h2>
      </div>
      <div class="segmented" role="group" aria-label="Performance cost axis">
        <button
          v-for="option in xOptions"
          :key="option.value"
          type="button"
          :class="{ 'is-active': benchmark.scatterX === option.value }"
          :aria-pressed="benchmark.scatterX === option.value"
          @click="benchmark.scatterX = option.value"
        >{{ option.label }}</button>
      </div>
    </div>
    <div class="component-surface performance-cost-chart-surface">
      <div class="surface-title-row">
        <div><h3 class="scatter-chart-title">{{ chartTitle }}</h3></div>
        <span class="pareto-key"><i></i> Pareto frontier</span>
      </div>
      <ScatterChart />
    </div>
    <div class="component-surface performance-cost-table-surface paginated-surface">
      <PerformanceCostTable />
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue';
import ScatterChart from './ScatterChart.vue';
import PerformanceCostTable from './PerformanceCostTable.vue';
import { benchmark } from '../store/benchmark';
import { SECTIONS } from '../lib/sections';

const section = SECTIONS['quality-cost'];
const chartTitle = computed(() => `GM Speedup × ${benchmark.scatterX === 'cost' ? 'Cost' : 'Tokens'}`);

const xOptions = [
  { value: 'tokens', label: 'Tokens' },
  { value: 'cost', label: 'API Cost' }
];

</script>
