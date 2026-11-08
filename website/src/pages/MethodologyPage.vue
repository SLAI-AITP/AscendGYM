<template>
  <a class="skip-link" href="#methodology" @click.prevent="scrollToId('methodology')">Skip to methodology</a>

  <SiteHeader variant="page" />

  <main>
    <div class="page-shell page-intro">
      <p class="section-index">{{ section.index }} / {{ section.kicker }}</p>
      <h1 id="methodology-title" class="page-title">{{ section.title }}</h1>
      <div class="page-quicknav" aria-label="Navigate to leaderboard sections">
        <span class="micro-label">Leaderboard sections</span>
        <button v-for="item in sectionLinks" :key="item.id" type="button" class="page-quicklink" @click="goToSection(item.id)">
          {{ item.label }}
        </button>
      </div>
    </div>

    <MethodologySection :show-heading="false" />
  </main>

  <SiteFooter />
</template>

<script setup>
import { computed, onMounted } from 'vue';
import SiteHeader from '../components/SiteHeader.vue';
import MethodologySection from '../components/MethodologySection.vue';
import SiteFooter from '../components/SiteFooter.vue';
import { MAIN_SECTIONS, SECTIONS } from '../lib/sections';
import { useSectionNav } from '../lib/useSectionNav';
import { applyStartupParams, scrollToId } from '../store/ui';
import { initBenchmark } from '../store/benchmark';

const section = SECTIONS.methodology;

// Quick links back to every content section of the leaderboard page, so this
// page keeps the same 1:1 section vocabulary as the top navigation.
const sectionLinks = computed(() => MAIN_SECTIONS.map(item => ({ id: item.id, label: item.label })));

const { goToSection } = useSectionNav();

onMounted(async () => {
  applyStartupParams();
  // The metric catalog is rendered from `meta.catalog`, which arrives with the
  // bundle — so the data must be loaded before this page paints.
  await initBenchmark();
});
</script>
