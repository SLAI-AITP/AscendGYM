<template>
  <span class="harness-chip" :data-size="size" role="img" :aria-label="`${name} mark`" :title="name">
    <img v-if="logo" :src="assetPath(logo)" alt="" loading="eager">
    <span v-else aria-hidden="true">{{ agentInitials(agent.id) }}</span>
  </span>
</template>

<script setup>
import { computed } from 'vue';
import { harnessDisplayName } from '../lib/displayNames';
import { agentInitials } from '../lib/meta';
import { assetPath } from '../lib/markup';

const props = defineProps({
  agent: { type: Object, required: true },
  size: { type: String, default: 'sm' }
});
const marks = {
  'claude-code': 'assets/harnesses/claude-code.svg',
  codex: 'assets/vendors/openai.svg',
  opencode: 'assets/harnesses/opencode.svg'
};
const name = computed(() => harnessDisplayName(props.agent));
const logo = computed(() => marks[props.agent.id] ?? null);
</script>
