<template>
  <section class="section page-shell" id="leaderboard" aria-labelledby="leaderboard-title">
    <div class="section-heading">
      <div>
        <p class="section-index">{{ section.index }} / {{ section.kicker }}</p>
        <h2 id="leaderboard-title">{{ section.title }}</h2>
        <!-- Subtitle comes from SECTIONS.leaderboard.summary. Never inline it:
             a hand-written <p> here silently orphans the registry field. -->
        <p v-for="(line, i) in sectionSummary(section)" :key="i" class="section-description">{{ line }}</p>
        <p class="metric-definition-link"><RouterLink :to="{ ...pageLocation('methodology'), hash: '#sts-metric' }">Metric Definition</RouterLink></p>
      </div>
      <div class="direction-badge"><span>{{ directionArrow }}</span> {{ directionText }}</div>
    </div>

    <div class="leader-controls" aria-label="Leaderboard controls">
      <label>Evaluation mode
        <select class="evaluation-mode-select" v-model="benchmark.cohort" @change="benchmark.agentId = 'all'; benchmark.engineModelId = 'all'; benchmark.search = ''">
          <option value="harness">Harness-Model</option><option value="default">Model Only</option>
        </select>
      </label>
      <label class="search-field">Search
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4 4" /></svg>
        <input id="model-search" type="search" placeholder="Search by name" autocomplete="off" :value="benchmark.search" @input="benchmark.search = $event.target.value.trim().toLowerCase()">
      </label>
      <ConfigurationFilter />
      <label>Re-rank
        <select id="rank-metric" v-model="benchmark.rankMetric">
          <option v-for="option in rankMetricOptions" :key="option" :value="option">{{ option === 'sts_token' ? 'STS' : (metricMeta[option]?.label || option) }}</option>
        </select>
      </label>
      <!-- Keep the five primary controls on the first row. -->
      <label>Chip
        <select id="chip-filter"><option v-for="chip in chips" :key="chip">{{ chip }}</option></select>
      </label>
      <label v-if="benchmark.preset === 'by-agent'">Harness
        <select v-model="benchmark.agentId">
          <option value="all">All harnesses</option>
          <option v-for="agent in agents" :key="agent.id" :value="agent.id">{{ harnessDisplayName(agent) }} ({{ agent.type }})</option>
        </select>
      </label>
      <label v-if="benchmark.preset === 'by-model'">Model
        <select v-model="benchmark.engineModelId">
          <option value="all">All models</option>
          <option v-for="model in engineModels" :key="model.id" :value="model.id">{{ modelDisplayName(model) }}</option>
        </select>
      </label>
    </div>

    <div class="component-surface chart-surface">
      <div class="surface-title-row">
        <div>
          <h3 class="rank-chart-title">{{ rankMetricLabel }}</h3>
          <p class="rank-budget-note">The token budget <MathText tex="B_{\max}" /> is set to 50M tokens.</p>
          <div class="rank-ring-legend" aria-label="Metric rings, outer to inner">
            <span class="ring-key-sts">STS <small>(0–100 score scale)</small></span>
            <span class="ring-key-gate">GATE <small>(%)</small></span>
            <span class="ring-key-vlr">VLR <small>(%)</small></span>
          </div>
        </div>
      </div>
      <RankChart :metric-label="rankMetricLabel" />
    </div>

    <div class="table-wrap component-surface leader-table-surface">
      <LeaderTable :entries="visibleEntries" />
    </div>
  </section>
</template>

<script setup>
import { harnessDisplayName, modelDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import { RouterLink } from 'vue-router';
import RankChart from './RankChart.vue';
import MathText from './MathText.vue';
import LeaderTable from './LeaderTable.vue';
import ConfigurationFilter from './ConfigurationFilter.vue';
import { benchmark, visibleEntries, agents, engineModels } from '../store/benchmark';
import { metricMeta, rankMetricOptions } from '../lib/meta';
import { SECTIONS, sectionSummary } from '../lib/sections';
import { useSectionNav } from '../lib/useSectionNav';

const section = SECTIONS.leaderboard;

// Deep-link to the STS definition and formula on the Metrics page.
const { pageLocation } = useSectionNav();

const rankMetricLabel = computed(() => benchmark.rankMetric === 'sts_token' ? 'STS' : (metricMeta[benchmark.rankMetric]?.label || 'STS'));
// The badge must describe the *selected* metric, not a hard-coded STS claim.
const directionArrow = computed(() => (metricMeta[benchmark.rankMetric]?.direction === 'down' ? '↓' : '↑'));
const directionText = computed(() => {
  const meta = metricMeta[benchmark.rankMetric];
  if (meta?.direction === 'down') return 'Lower is better';
  return 'Higher is better';
});
const chips = computed(() => benchmark.data?.meta.chips ?? ['Ascend 910C']);
</script>
