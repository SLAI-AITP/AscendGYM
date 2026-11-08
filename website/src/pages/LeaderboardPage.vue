<template>
  <a class="skip-link" href="#leaderboard" @click.prevent="scrollToId('leaderboard')">Skip to leaderboard</a>

  <SiteHeader />

  <main>
    <EntryRankingSection />
    <QualityCostSection />
    <OperatorProcessSection />
    <PerformanceGainSection />
  </main>

  <SiteFooter />
</template>

<script setup>
import { onMounted } from 'vue';
import SiteHeader from '../components/SiteHeader.vue';
import EntryRankingSection from '../components/EntryRankingSection.vue';
import QualityCostSection from '../components/QualityCostSection.vue';
import OperatorProcessSection from '../components/OperatorProcessSection.vue';
import PerformanceGainSection from '../components/PerformanceGainSection.vue';
// Parked components — imported by nobody, kept for a possible re-introduction:
//   MetricsDetailSection — removed with the Rubric navigation and registry entry.
//   DurationSection / HardwareSection — their registry entries were removed from
//     lib/sections.js, so rendering them would read SECTIONS.duration === undefined
//     and crash. Restore both SECTIONS entries *and* the MAIN_SECTION_ORDER keys
//     before importing them again.
//   MethodologySection — moved to pages/MethodologyPage.vue so the long-scroll
//     keeps only ranking views; the formulas stay one nav click away.
import SiteFooter from '../components/SiteFooter.vue';
import { scrollToId, applyStartupParams, configureExportPage } from '../store/ui';
import { initBenchmark } from '../store/benchmark';

onMounted(async () => {
  const { params, exportBreakpoint } = applyStartupParams();
  // v2: pull the bundle (live /api/bundle first, static fallback) before
  // the export page sizing so the captured master reflects real heights.
  await initBenchmark();
  configureExportPage(params, exportBreakpoint);
});
</script>
