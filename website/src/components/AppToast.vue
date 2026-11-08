<template>
  <div class="toast" :class="{ 'is-visible': toast.visible }" role="status" aria-live="polite">{{ toast.message }}</div>
</template>

<script setup>
import { toast } from '../store/ui';
</script>
