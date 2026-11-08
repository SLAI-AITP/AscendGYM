<template>
  <div id="duration-matrix">
    <div v-if="rows.length" class="shape-rings" role="grid" :aria-label="`Shape durations for ${selection.operator.operator}`">
      <div
        v-for="row in rows"
        :key="row.shape"
        class="shape-ring-cell"
        role="gridcell"
        :style="{ '--ring-color': ringColor(row) }"
        :data-tooltip-html="tooltip(row)"
      >
        <svg class="timing-dial" viewBox="0 0 84 84" aria-hidden="true">
          <circle class="timing-dial-base" cx="42" cy="42" r="34" />
          <circle
            class="timing-dial-signal"
            cx="42" cy="42" r="26"
            pathLength="100"
            :stroke-dasharray="`${Math.max(5, Math.min(100, (row.s_lat / 1.3) * 100)).toFixed(1)} 100`"
          />
        </svg>
        <span class="shape-ring-center">
          <b class="matrix-value">{{ formatMicros(row.t_best_us) }}<small>µs</small></b>
          <i :class="speedupClass(row.s_lat)">{{ formatSpeedup(row.s_lat) }}</i>
        </span>
        <span class="shape-ring-label">{{ row.shape }} <em>{{ row.dtype }}</em></span>
      </div>
      <div class="shape-rings-summary">
        <!-- Shape-level geomean is NOT PAOS (PAOS = sqrt(PKAI x PTEI)). -->
        <span class="micro-label">Geomean S_lat(per shape)</span>
        <b :class="speedupClass(geomean)">{{ geomean ? formatSpeedup(geomean) : '–' }}</b>
        <span v-if="worst" class="worst-chip" :class="{ 'is-bad': worst.s_lat < 0.9 }">worst {{ worst.shape }} · {{ formatSpeedup(worst.s_lat) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { modelDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import { selectedDuration } from '../store/benchmark';
import { attributeEscape, speedupClass, formatSpeedup, formatMicros } from '../lib/format';

const selection = computed(() => selectedDuration.value);
const rows = computed(() => selection.value.operator?.duration_by_shape ?? []);
const geomean = computed(() => selection.value.operator?.paos ?? null);
const worst = computed(() => selection.value.operator?.worst_shape ?? null);

function tone(row) {
  if (row.s_lat < 1) return 'var(--danger)';
  if (row.s_lat >= 1.05) return 'var(--compute)';
  return 'var(--muted)';
}

function ringColor(row) { return tone(row); }

function tooltip(row) {
  const { entry, operator } = selection.value;
  return attributeEscape(
    `<strong>${modelDisplayName(entry.engine_model)} · ${operator.operator}</strong><br>${row.shape} · ${row.dtype}<br>Base: <span class="tooltip-value">${formatMicros(row.t_base_us)}</span><br>Best: <span class="tooltip-value">${formatMicros(row.t_best_us)}</span><br>S_lat: <span class="tooltip-value">${formatSpeedup(row.s_lat)}</span>`
  );
}
</script>
