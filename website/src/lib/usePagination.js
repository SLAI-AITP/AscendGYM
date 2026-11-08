import { computed, ref, toValue, watch } from 'vue';

export const TABLE_PAGE_SIZE = 5;

/** Page only the displayed rows; callers retain full data for chart scales. */
export function usePagination(items, key = item => item.id) {
  const page = ref(1);
  const total = computed(() => toValue(items).length);
  const pageCount = computed(() => Math.max(1, Math.ceil(total.value / TABLE_PAGE_SIZE)));
  const startIndex = computed(() => (page.value - 1) * TABLE_PAGE_SIZE);
  const pagedItems = computed(() => toValue(items).slice(startIndex.value, startIndex.value + TABLE_PAGE_SIZE));
  watch(() => JSON.stringify(toValue(items).map(key)), () => { page.value = 1; }, { flush: 'sync' });
  watch(page, value => {
    const next = Number.isFinite(value) ? Math.max(1, Math.min(pageCount.value, Math.floor(value))) : 1;
    if (next !== value) page.value = next;
  }, { flush: 'sync' });
  return { page, total, pageCount, startIndex, pagedItems, pageSize: TABLE_PAGE_SIZE };
}
