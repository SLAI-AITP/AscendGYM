<template>
  <a class="skip-link" href="#imports" @click.prevent="scrollToId('imports')">Skip to imports</a>

  <SiteHeader variant="page" />

  <main>
    <div class="page-shell page-intro">
      <p class="section-index">{{ section.index }} / {{ section.kicker }}</p>
      <h1 id="imports-title" class="page-title">{{ section.title }}</h1>
      <p v-for="(line, i) in summaryLines" :key="i" class="page-lede">{{ line }}</p>
      <div class="page-quicknav" aria-label="Navigate to leaderboard sections">
        <span class="micro-label">Leaderboard sections</span>
        <button v-for="item in sectionLinks" :key="item.id" type="button" class="page-quicklink" @click="goToSection(item.id)">
          {{ item.label }}
        </button>
      </div>
    </div>

    <ImportSection :show-heading="false" />
  </main>

  <SiteFooter />
</template>

<script setup>
import { computed, onMounted } from 'vue';
import SiteHeader from '../components/SiteHeader.vue';
import ImportSection from '../components/ImportSection.vue';
import SiteFooter from '../components/SiteFooter.vue';
import { MAIN_SECTIONS, SECTIONS, sectionSummary } from '../lib/sections';
import { useSectionNav } from '../lib/useSectionNav';
import { applyStartupParams, scrollToId } from '../store/ui';
import { initBenchmark } from '../store/benchmark';

const section = SECTIONS.imports;

// The page lede is the section's registry summary — the same copy the section
// renders when it carries its own heading, so the two can never disagree.
const summaryLines = computed(() => sectionSummary(section));

// Quick links back to every content section of the leaderboard page, so this
// page keeps the same 1:1 section vocabulary as the top navigation.
const sectionLinks = computed(() => MAIN_SECTIONS.map(section => ({ id: section.id, label: section.label })));

const { goToSection } = useSectionNav();

onMounted(async () => {
  applyStartupParams();
  await initBenchmark();
});
</script>
