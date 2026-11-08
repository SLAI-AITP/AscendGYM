<template>
  <nav v-if="total > 0" class="table-pagination" :aria-label="label">
    <span class="pagination-range" aria-live="polite">{{ start }}–{{ end }} of {{ total }}</span>
    <div class="pagination-controls">
      <button type="button" class="pagination-step" :disabled="page <= 1" aria-label="Previous page" @click="change(page - 1)">‹ <span>Previous</span></button>
      <template v-for="item in pages" :key="item.key">
        <span v-if="item.gap" class="pagination-gap" aria-hidden="true">…</span>
        <button v-else type="button" :aria-label="`Page ${item.value}`" :aria-current="item.value === page ? 'page' : undefined" @click="change(item.value)">{{ item.value }}</button>
      </template>
      <button type="button" class="pagination-step" :disabled="page >= count" aria-label="Next page" @click="change(page + 1)"><span>Next</span> ›</button>
    </div>
  </nav>
</template>

<script setup>
import { computed } from 'vue';
const props = defineProps({ page: { type: Number, default: 1 }, total: { type: Number, required: true }, pageSize: { type: Number, default: 5 }, label: { type: String, default: 'Table pagination' } });
const emit = defineEmits(['update:page']);
const count = computed(() => Math.max(1, Math.ceil(props.total / props.pageSize)));
const start = computed(() => (props.page - 1) * props.pageSize + 1);
const end = computed(() => Math.min(props.page * props.pageSize, props.total));
const pages = computed(() => {
  const values = Array.from({ length: count.value }, (_, i) => i + 1)
    .filter(value => count.value <= 7 || value === 1 || value === count.value || Math.abs(value - props.page) <= 1);
  return values.flatMap((value, index) => [
    ...(index && value - values[index - 1] > 1 ? [{ key: `gap-${value}`, gap: true }] : []),
    { key: value, value }
  ]);
});
function change(value) { emit('update:page', Math.max(1, Math.min(count.value, value))); }
</script>
