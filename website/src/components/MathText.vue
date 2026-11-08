<template>
  <component :is="display ? 'div' : 'span'" :class="['metric-math', { 'report-formula': display }]" v-html="html" />
</template>

<script setup>
import { computed } from 'vue';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const props = defineProps({ tex: { type: String, required: true }, display: Boolean });
const html = computed(() => katex.renderToString(props.tex, {
  displayMode: props.display, output: 'htmlAndMathml', throwOnError: true, trust: false, strict: 'error'
}));
</script>
