<template>
  <span class="flag-cell" :class="{ 'is-empty': !flags?.length }">
    <span
      v-for="flag in flags"
      :key="flag"
      class="flag-dot"
      :data-level="flagMeta[flag]?.level"
      :data-tooltip-html="tooltipFor(flag)"
      :aria-label="flagMeta[flag]?.label"
    >{{ flagMeta[flag]?.icon }}</span>
    <span v-if="!flags?.length" class="flag-none" aria-label="no flags">–</span>
  </span>
</template>

<script setup>
import { flagMeta } from '../lib/meta';
import { attributeEscape } from '../lib/format';

defineProps({
  flags: { type: Array, default: () => [] }
});

function tooltipFor(flag) {
  const meta = flagMeta[flag];
  if (!meta) return '';
  return attributeEscape(`<strong>${meta.icon} ${meta.label}</strong><br>${meta.rule}`);
}
</script>
