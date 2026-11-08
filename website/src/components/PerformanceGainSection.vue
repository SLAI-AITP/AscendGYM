<template>
  <section class="section page-shell" id="gain" aria-labelledby="gain-title">
    <div class="section-heading gain-heading">
      <div>
        <p class="section-index">{{ section.index }} / {{ section.kicker }}</p>
        <h2 id="gain-title">{{ section.title }}</h2>
      </div>
      <BenchmarkSelect id="gain-model" v-model="benchmark.gainEntry" label="Harness × Model" :options="entryOptions" />
      <p v-for="(line, i) in sectionSummary(section)" :key="i" class="gain-description">{{ line }}</p>
    </div>
    <div class="split-grid">
      <div class="component-surface">
        <div class="surface-title-row gain-chart-identity">
          <EntryLabel v-if="gainSelection" :entry="gainSelection" stacked-icons :show-agent-badge="false" :show-vendor-in-subtitle="false" />
        </div>
        <GainChart />
      </div>
      <div class="component-surface best-combinations-surface">
        <div class="surface-title-row"><div><h3 class="best-combination-title">Best combination per kernel</h3><p>Best correctness-validated speedup within the 50M-token budget.</p></div></div>
        <BestByOperator />
      </div>
    </div>
  </section>
</template>

<script setup>
import { harnessDisplayName, modelDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import GainChart from './GainChart.vue';
import EntryLabel from './EntryLabel.vue';
import BenchmarkSelect from './BenchmarkSelect.vue';
import BestByOperator from './BestByOperator.vue';
import { benchmark, gainSelection, fallbackEntries } from '../store/benchmark';
import { SECTIONS, sectionSummary } from '../lib/sections';

const section = SECTIONS.gain;

const entryOptions = computed(() => fallbackEntries.value.map(entry => ({ value: entry.id, label: `${harnessDisplayName(entry.agent)} × ${modelDisplayName(entry.engine_model)}` })));
</script>
