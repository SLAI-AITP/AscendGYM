<template>
  <header class="site-header" :class="`header-${variant}`" id="top">
    <nav class="top-nav page-shell" aria-label="Primary navigation">
      <a class="mini-lockup" href="#top" aria-label="AscendGYM home" @click.prevent="home">
        <img :src="assetPath('assets/logo/mark-primary.svg')" alt="" width="34" height="34">
        <span>Ascend<b>GYM</b></span>
      </a>
      <!-- Nav is generated from lib/sections.js so every entry maps 1:1 to a
           rendered section (or a standalone page) and can never drift. -->
      <div class="nav-links">
        <template v-for="item in navItems" :key="item.id">
          <a
            v-if="!item.route"
            :href="`#${item.id}`"
            :class="{ 'is-active': onLeaderboardPage && activeId === item.id }"
            :aria-current="onLeaderboardPage && activeId === item.id ? 'true' : undefined"
            @click.prevent="goToSection(item.id)"
          >{{ item.label }}</a>
          <!-- Standalone page (Methodology / Imports). Every page here is
               read-only now, so there is no per-page lock to render. -->
          <RouterLink
            v-else
            :to="pageLocation(item.id)"
            :class="{ 'is-active': currentRouteName === item.id }"
            @click="currentRouteName === item.id && goToPage(item.id)"
          >{{ item.label }}</RouterLink>
        </template>
      </div>
      <div class="nav-actions">
        <button
          class="theme-switch"
          :class="{ 'is-dark': ui.theme === 'dark' }"
          type="button"
          role="switch"
          aria-label="Dark mode"
          :aria-checked="ui.theme === 'dark'"
          :title="`Switch to ${ui.theme === 'light' ? 'Dark' : 'Light'} mode`"
          @click="toggleTheme"
        >
          <span class="theme-option" :class="{ 'is-active': ui.theme === 'light' }">
            <svg class="theme-symbol" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" />
            </svg>
            <span>Light</span>
          </span>
          <span class="theme-option" :class="{ 'is-active': ui.theme === 'dark' }">
            <svg class="theme-symbol" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M20.5 13.1A8.5 8.5 0 0 1 10.9 3.5 8.5 8.5 0 1 0 20.5 13.1Z" />
            </svg>
            <span>Dark</span>
          </span>
        </button>
      </div>
    </nav>

    <!-- Hero only on the home page; standalone tool pages bring their own title. -->
    <template v-if="variant === 'home'">
    <div class="hero-grid" aria-hidden="true"></div>
    <div class="hero page-shell">
      <div class="hero-mark-wrap">
        <img class="hero-mark light-logo" :src="assetPath('assets/logo/mark-primary.svg')" alt="AscendGYM mark — a compute core inside an orbital ring" width="96" height="96">
        <img class="hero-mark dark-logo" :src="assetPath('assets/logo/mark-dark.svg')" alt="AscendGYM mark — a compute core inside an orbital ring" width="96" height="96">
      </div>
      <p class="eyebrow">ASCEND KERNEL OPTIMIZATION BENCHMARK</p>
      <h1>Ascend<span>GYM</span></h1>
      <p class="hero-tagline">A Benchmark for Evaluating Kernel Optimization Capabilities on Real-World Training and Inference Workloads on Ascend</p>
      <div class="resource-row" aria-label="Project resources">
        <a class="resource-chip" :href="assetPath('AscendGYM_TechReport.pdf')" target="_blank" rel="noopener"><span class="resource-icon resource-brand arxiv"><img :src="assetPath('assets/resources/arxiv.jpg')" alt="arXiv" width="32" height="32"></span> Technical Report <span aria-hidden="true">↗</span></a>
        <!-- Hugging Face remains a placeholder until a dataset URL is available. -->
        <span class="resource-chip is-placeholder"><span class="resource-icon resource-brand hf"><img :src="assetPath('assets/resources/huggingface.svg')" alt="" width="32" height="32"></span> Hugging Face</span>
        <a class="resource-chip" href="https://github.com/SLAI-AITP/AscendGYM" target="_blank" rel="noopener"><span class="resource-icon resource-brand gh"><img :src="assetPath('assets/resources/github.svg')" alt="" width="32" height="32"></span> GitHub <span aria-hidden="true">↗</span></a>
      </div>

      <div class="status-row">
        <span class="status-pill"><span class="pulse-dot"></span> Updated {{ meta?.updated ?? '—' }}</span>
        <span class="placeholder-note">{{ provenanceNote }}</span>
      </div>

      <dl class="headline-stats" id="headline-stats">
        <div v-for="stat in headlineStats" :key="stat[0]">
          <dt>{{ stat[0] }}</dt><dd>{{ stat[1] }}<small v-if="stat[2]" class="headline-stat-note">{{ stat[2] }}</small></dd>
        </div>
      </dl>
    </div>
    </template>
  </header>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { RouterLink } from 'vue-router';
import { benchmark, boardMeta } from '../store/benchmark';
import { ui, toggleTheme } from '../store/ui';
import { assetPath } from '../lib/markup';
import { formatTokensK } from '../lib/format';
import { MAIN_SECTIONS, NAV_ITEMS } from '../lib/sections';
import { useSectionNav } from '../lib/useSectionNav';

const props = defineProps({
  /** 'home' renders the hero; 'page' renders the nav bar only. */
  variant: { type: String, default: 'home' }
});

const { goToSection, goToPage, pageLocation, onLeaderboardPage, currentRouteName } = useSectionNav();

const navItems = NAV_ITEMS;

function home() {
  if (currentRouteName.value === 'leaderboard') {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    return;
  }
  goToPage('leaderboard');
}

// ------------------------------- scroll spy ---------------------------------
// Highlight the nav entry for whichever section is currently in view, so the
// navigation visibly corresponds to the content below it.
const activeId = ref(MAIN_SECTIONS[0].id);
let observer = null;

onMounted(() => {
  if (props.variant !== 'home' || typeof IntersectionObserver === 'undefined') return;
  const elements = MAIN_SECTIONS
    .map(section => document.getElementById(section.id))
    .filter(Boolean);
  if (!elements.length) return;
  observer = new IntersectionObserver((entries) => {
    // Pick the entry closest to the top of the viewport among those visible.
    const visible = entries.filter(entry => entry.isIntersecting);
    if (!visible.length) return;
    const top = visible.reduce((best, entry) => (
      entry.boundingClientRect.top < best.boundingClientRect.top ? entry : best
    ));
    activeId.value = top.target.id;
  }, { rootMargin: '-20% 0px -70% 0px', threshold: 0 });
  elements.forEach(element => observer.observe(element));
});

onBeforeUnmount(() => {
  observer?.disconnect();
});

const meta = computed(() => boardMeta.value);

// Capability summary displayed below the measurement snapshot date.
const provenanceNote = 'HARNESS COMPARISON · MODEL MEASUREMENT · JOINT PERFORMANCE–COST METRICS';

// Requested capacity labels; unique evaluated model IDs span all published cohorts.
// Cases is the supplied benchmark total; cumulative tokens use the enabled cohort.
const headlineStats = computed(() => {
  const m = meta.value;
  if (!m) return [];

  const modelCount = new Set((benchmark.data?.entries ?? [])
    .map(entry => entry.engine_model?.id?.trim()).filter(Boolean)).size;
  return [
    ['Cases', '951'],
    ['Agents', '4+'],
    ['Kernels', '48', '17 for now'],
    ['Models', modelCount],
    ['Cumulative Tokens', m.tokens_total_k ? formatTokensK(m.tokens_total_k) : '–']
  ];
});
</script>
