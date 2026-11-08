<template>
  <span class="vendor-chip" :data-size="size" :data-vendor="visual.vendor" :style="{ '--model-color': visual.color }">
    <img v-if="visual.asset" :src="logo" :alt="alt || `${visual.label} mark`" loading="eager">
    <!-- No brand mark for this vendor: render a monogram in the same ring
         instead of silently borrowing another vendor's logo. -->
    <span v-else class="vendor-glyph" aria-hidden="true">{{ visual.glyph }}</span>
  </span>
</template>

<script setup>
import { computed } from 'vue';
import { assetPath } from '../lib/markup';
import { vendorVisual } from '../lib/vendors';

const props = defineProps({
  // v2: pass the engine_model object directly
  model: { type: Object, required: true },
  size: { type: String, default: 'sm' },
  alt: { type: String, default: '' },
  // Optional: paint the ring in this colour instead of the model's own.
  color: { type: String, default: null }
});

const visual = computed(() => vendorVisual(props.model, { color: props.color }));
const logo = computed(() => assetPath(`assets/vendors/${visual.value.asset}`));
</script>
