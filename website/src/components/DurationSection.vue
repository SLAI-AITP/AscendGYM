<template>
  <section class="section page-shell" id="duration" aria-labelledby="duration-title">
    <div class="section-heading">
      <div>
        <p class="section-index">{{ section.index }} / {{ section.kicker }}</p>
        <h2 id="duration-title">{{ section.title }}</h2>
        <p v-for="(line, i) in sectionSummary(section)" :key="i">{{ line }}</p>
      </div>
      <div class="control-row">
        <label>Operator
          <select id="duration-operator" v-model="benchmark.durationOperator">
            <option v-for="operatorName in operators" :key="operatorName" :value="operatorName">{{ operatorName }}</option>
          </select>
        </label>
        <label>Entry
          <select id="duration-model" v-model="benchmark.durationEntry">
            <option v-for="entry in entriesWithOperator" :key="entry.id" :value="entry.id">{{ harnessDisplayName(entry.agent) }} × {{ modelDisplayName(entry.engine_model) }}</option>
          </select>
        </label>
      </div>
    </div>
    <div class="split-grid duration-grid">
      <div class="component-surface">
        <div class="surface-title-row">
          <div><h3>Shape timing rings (base → best)</h3><p>Outer ring: baseline (dashed); inner ring: best; arc length: speedup</p></div>
          <div class="ramp-legend s-lat-legend"><span class="tone-bad-text">S&lt;1</span><i></i><span class="tone-good-text">S≥1.05</span></div>
        </div>
        <ShapeRings />
      </div>
      <div class="component-surface">
        <div class="surface-title-row">
          <div><h3>Base vs best race tracks</h3><p>Endpoints encode baseline and best latency for each shape</p></div>
          <div class="segmented">
            <button
              v-for="option in viewOptions"
              :key="option.value"
              type="button"
              :class="{ 'is-active': benchmark.durationView === option.value }"
              @click="benchmark.durationView = option.value"
            >{{ option.label }}</button>
          </div>
        </div>
        <RaceTracks />
      </div>
    </div>
    <div class="skeleton section-skeleton" aria-hidden="true"><i></i><i></i><i></i></div>
    <div class="empty-state section-empty" aria-hidden="true"><span class="empty-glyph">µs</span><strong>No profiler duration matrix</strong><p>Select an entry and operator with both baseline and candidate per-shape timings.</p></div>
  </section>
</template>

<script setup>
import { harnessDisplayName, modelDisplayName } from '../lib/displayNames';
import { computed, watch } from 'vue';
import ShapeRings from './ShapeRings.vue';
import RaceTracks from './RaceTracks.vue';
import { benchmark, operators, fallbackEntries } from '../store/benchmark';
import { sectionSummary } from '../lib/sections';

// This section is parked: it is no longer in lib/sections.js, so it has no
// registry entry to read. The descriptor is local so the component stays
// self-contained — re-enabling it means restoring the `duration` entry in
// SECTIONS *and* in MAIN_SECTION_ORDER, plus the import in LeaderboardPage.
const section = { id: 'duration', index: '06', kicker: 'TASK DURATION', title: 'Latency by shape', summary: 'Per-shape baseline and candidate latency. Colors indicate regression below 1 and improvement at or above 1.05.' };

// Only offer entries that actually evaluated the selected operator.
const entriesWithOperator = computed(() =>
  fallbackEntries.value.filter(entry => entry.per_operator.some(op => op.operator === benchmark.durationOperator))
);

// Keep the selected entry valid when the operator (or filters) change.
watch(() => [benchmark.durationOperator, benchmark.preset, benchmark.agentId, benchmark.engineModelId, benchmark.openness, benchmark.search], () => {
  if (!entriesWithOperator.value.some(entry => entry.id === benchmark.durationEntry)) {
    benchmark.durationEntry = entriesWithOperator.value[0]?.id ?? null;
  }
});

const viewOptions = [
  { value: 'absolute', label: 'µs' },
  { value: 'normalized', label: 'Fastest = 1×' }
];
</script>
