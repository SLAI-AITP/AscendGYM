<template>
  <div ref="root" class="configuration-filter" @keydown.esc.stop.prevent="close(true)">
    <span class="control-label" id="configuration-filter-label">Filter</span>
    <button ref="trigger" class="configuration-filter-trigger" type="button"
      :aria-label="`Configuration filter: ${selectionLabel}`" :aria-expanded="isOpen"
      aria-controls="configuration-filter-panel" @click="toggle">
      <span>{{ selectionLabel }}</span>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>
    </button>
    <div v-if="isOpen" id="configuration-filter-panel" class="configuration-filter-panel" role="group" aria-label="Configuration filters">
      <div class="configuration-filter-level" role="group" aria-label="Configuration scope" @keydown="moveFocus">
        <button v-for="option in presetOptions" :key="option.value" type="button"
          :class="{ 'is-selected': draftPreset === option.value }" :aria-pressed="draftPreset === option.value"
          @click="draftPreset = option.value" @keydown.right.prevent="focusSecondLevel">
          <span>{{ option.label }}</span><span aria-hidden="true">›</span>
        </button>
      </div>
      <div ref="secondLevel" class="configuration-filter-level" role="group" :aria-label="`${presetLabel(draftPreset)} model openness`" @keydown="moveFocus">
        <button v-for="option in opennessOptions" :key="option.value" type="button"
          :class="{ 'is-selected': benchmark.preset === draftPreset && benchmark.openness === option.value }"
          :aria-pressed="benchmark.preset === draftPreset && benchmark.openness === option.value"
          @click="apply(option.value)" @keydown.left.prevent="focusFirstLevel">
          <span>{{ option.label }}</span><span aria-hidden="true" v-if="benchmark.preset === draftPreset && benchmark.openness === option.value">✓</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { benchmark } from '../store/benchmark';

const presetOptions = [
  { value:'free', label:'All configurations' },
  { value:'by-agent', label:'By harness' },
  { value:'by-model', label:'By model' }
];
const opennessOptions = [
  { value:'all', label:'All' },
  { value:'open', label:'Open' },
  { value:'closed', label:'Closed' }
];
const root = ref(null), trigger = ref(null), secondLevel = ref(null);
const isOpen = ref(false), draftPreset = ref(benchmark.preset);
const presetLabel = value => presetOptions.find(option => option.value === value)?.label ?? 'All configurations';
const selectionLabel = computed(() => `${presetLabel(benchmark.preset)} / ${opennessOptions.find(option => option.value === benchmark.openness)?.label ?? 'All'}`);

function toggle() {
  draftPreset.value = benchmark.preset;
  isOpen.value = !isOpen.value;
}
function close(restoreFocus = false) {
  isOpen.value = false;
  if (restoreFocus) trigger.value?.focus();
}
function apply(openness) {
  benchmark.preset = draftPreset.value;
  benchmark.openness = openness;
  if (benchmark.preset !== 'by-agent') benchmark.agentId = 'all';
  if (benchmark.preset !== 'by-model') benchmark.engineModelId = 'all';
  close(true);
}
function focusSecondLevel() { secondLevel.value?.querySelector('button')?.focus(); }
function focusFirstLevel() { root.value?.querySelector('.configuration-filter-level button[aria-pressed="true"]')?.focus(); }
function moveFocus(event) {
  if (!['ArrowDown','ArrowUp','Home','End'].includes(event.key)) return;
  const buttons = [...event.currentTarget.querySelectorAll('button')];
  const index = buttons.indexOf(document.activeElement);
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
    : (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
  event.preventDefault();
  buttons[next]?.focus();
}
function onOutsidePointer(event) { if (!root.value?.contains(event.target)) close(); }
function onFocusOut(event) { if (!root.value?.contains(event.target)) close(); }
onMounted(() => {
  document.addEventListener('pointerdown', onOutsidePointer);
  document.addEventListener('focusin', onFocusOut);
});
onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', onOutsidePointer);
  document.removeEventListener('focusin', onFocusOut);
});
</script>
