<template>
  <div class="kernel-shapes">
    <strong class="shape-list-heading">Shapes<span v-if="rows.length"> ({{ rows.length }})</span></strong>
    <table v-if="rows.length" class="shape-list-table">
      <thead><tr><th>Shape</th><th>Dtype</th><th>Baseline<br>(µs)</th><th>Optimized<br>(µs)</th><th>Speedup</th></tr></thead>
      <tbody>
        <tr v-for="row in rows" :key="row.case_id">
          <td><span class="shape-case-id">{{ row.case_id }}</span><span>{{ dimensions(row) }}</span></td>
          <td>{{ dtypes(row) }}</td>
          <td>{{ formatMeasurement(row.t_base_us) }}</td>
          <td>{{ formatMeasurement(row.t_best_us) }}</td>
          <td :class="row.speedup == null ? '' : speedupClass(row.speedup)">{{ formatSpeedup(row.speedup) }}<small v-if="row.correctness && row.correctness !== 'pass'">{{ statusLabel(row.correctness) }}</small></td>
        </tr>
      </tbody>
    </table>
    <p v-else class="shape-caption" role="status">Shape data not yet available.</p>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { formatMeasurement, formatSpeedup, speedupClass } from '../lib/format';
const props = defineProps({ operator: { type: Object, required: true } });
const rows = computed(() => (props.operator.duration_by_shape ?? []).map((row, i) => ({
  ...row,
  case_id: row.case_id ?? `Case ${i + 1}`,
  speedup: row.correctness === 'pass' && Number.isFinite(row.t_base_us) && row.t_base_us > 0 && Number.isFinite(row.t_best_us) && row.t_best_us > 0
    ? row.t_base_us / row.t_best_us : null
})));
const dimensions = row => row.inputs?.length
  ? row.inputs.map(input => `${input.name ?? 'Input'}: ${JSON.stringify(input.shape)}`).join('; ')
  : row.shape ?? '–';
const dtypes = row => row.inputs?.length
  ? [...new Set(row.inputs.map(input => input.dtype ?? '–'))].join(', ')
  : row.dtype ?? '–';
const statusLabel = status => ({ fail: 'Failed', error: 'Error', not_measured: 'Not measured' }[status] ?? 'Not reported');
</script>
