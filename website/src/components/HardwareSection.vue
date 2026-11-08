<template>
  <section class="section page-shell" id="hardware" aria-labelledby="hardware-title">
    <div class="section-heading">
      <div><p class="section-index">{{ section.index }} / {{ section.kicker }}</p><h2 id="hardware-title">{{ section.title }}</h2><p v-for="(line, i) in sectionSummary(section)" :key="i">{{ line }}</p></div>
        <label>Operator
          <select id="util-operator" v-model="benchmark.utilOperator">
            <option v-for="operatorName in operators" :key="operatorName" :value="operatorName">{{ operatorName }}</option>
          </select>
        </label>
        <label>Entry (primary)
          <select v-model="benchmark.utilEntry">
            <option v-for="entry in entriesWithOperator" :key="entry.id" :value="entry.id">{{ harnessDisplayName(entry.agent) }} × {{ modelDisplayName(entry.engine_model) }}</option>
          </select>
        </label>
    </div>
    <aside class="context-note"><strong>Read with bottleneck context.</strong> Vec, Scalar, and MTE2 ratios are read from profiler summaries. Near-zero MAC activity is expected for vector kernels. Memory bandwidth and overlap are unavailable until the corresponding traces are collected. U approximates active-resource time divided by task duration.</aside>
    <div class="split-grid">
      <div class="component-surface">
        <div class="surface-title-row"><div><h3>AI Unit ratio fingerprint</h3><p>Four observed activity channels; unavailable channels remain empty.</p></div></div>
        <RatioFingerprint />
      </div>
      <div class="component-surface">
        <div class="surface-title-row"><div><h3>Exact ratio unit lanes</h3><p>Open markers: baseline; filled markers: best candidate.</p></div></div>
        <RatioLanes />
      </div>
    </div>
    <div class="roofline-note"><span class="optional-tag">OPTIONAL ROOFLINE</span><p>FLOP and byte counts are unavailable in the current records; arithmetic intensity and roofline distance cannot be computed.</p></div>
    <div class="skeleton section-skeleton" aria-hidden="true"><i></i><i></i><i></i></div>
    <div class="empty-state section-empty" aria-hidden="true"><span class="empty-glyph">◇</span><strong>No hardware counters</strong><p>Requires Vec, MAC, Scalar, MTE2, and utilization measurements from profiler summaries.</p></div>
  </section>
</template>

<script setup>
import { harnessDisplayName, modelDisplayName } from '../lib/displayNames';
import { computed, watch } from 'vue';
import RatioFingerprint from './RatioFingerprint.vue';
import RatioLanes from './RatioLanes.vue';
import { benchmark, operators, fallbackEntries } from '../store/benchmark';
import { sectionSummary } from '../lib/sections';

// This section is parked: it is no longer in lib/sections.js, so it has no
// registry entry to read. The descriptor is local so the component stays
// self-contained — re-enabling it means restoring the `hardware` entry in
// SECTIONS *and* in MAIN_SECTION_ORDER, plus the import in LeaderboardPage.
const section = { id: 'hardware', index: '07', kicker: 'HARDWARE UTILIZATION', title: 'Utilization fingerprint', summary: 'Diagnostic context; higher activity does not necessarily indicate better performance.' };

const entriesWithOperator = computed(() =>
  fallbackEntries.value.filter(entry => entry.per_operator.some(op => op.operator === benchmark.utilOperator && (op.utilization?.vec_ratio_best != null)))
);

// Keep the selected entry valid when the operator (or filters) change.
watch(() => [benchmark.utilOperator, benchmark.preset, benchmark.agentId, benchmark.engineModelId, benchmark.openness, benchmark.search], () => {
  if (!entriesWithOperator.value.some(entry => entry.id === benchmark.utilEntry)) {
    benchmark.utilEntry = entriesWithOperator.value[0]?.id ?? null;
  }
});
</script>
