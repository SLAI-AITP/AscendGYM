<template>
  <div
    ref="el"
    class="data-tooltip"
    :class="{ 'is-visible': tooltip.visible }"
    role="tooltip"
    :aria-hidden="String(!tooltip.visible)"
    :style="position"
    v-html="tooltip.html"
  ></div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue';
import { tooltip } from '../store/ui';
import { pageScale } from '../lib/displayScale';

const el = ref(null);
const position = ref({});

// Clamp against the viewport after the content has rendered so the final
// measurement matches the original tooltip behaviour.
watch(
  [() => tooltip.visible, () => tooltip.html, () => tooltip.x, () => tooltip.y],
  async () => {
    await nextTick();
    if (!tooltip.visible || !el.value) return;
    const width = el.value.offsetWidth || 250;
    const height = el.value.offsetHeight || 120;
    const scale = pageScale();
    const x = Math.min(tooltip.x / scale + 14, window.innerWidth / scale - width - 12);
    const y = Math.min(tooltip.y / scale + 14, window.innerHeight / scale - height - 12);
    position.value = { left: `${Math.max(10, x)}px`, top: `${Math.max(10, y)}px` };
  }
);
</script>
