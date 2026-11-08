<template>
  <div ref="root" class="kernel-select" @keydown.esc="escape">
    <select id="drawer-kernel" class="kernel-select-native" :value="modelValue" @change="$emit('update:modelValue', $event.target.value)">
      <option v-for="option in options" :key="option" :value="option">{{ option }}</option>
    </select>
    <div class="kernel-select-custom">
      <button ref="trigger" type="button" class="kernel-select-trigger" role="combobox" aria-labelledby="drawer-kernel-label drawer-kernel-value" aria-haspopup="listbox" :aria-expanded="open" aria-controls="drawer-kernel-options" @click="open ? close() : show()" @keydown.down.prevent="show()" @keydown.up.prevent="show()">
        <span id="drawer-kernel-value" :title="modelValue">{{ modelValue }}</span>
        <svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m5 8 5 5 5-5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" /></svg>
      </button>
      <div v-if="open" id="drawer-kernel-options" class="kernel-select-menu" role="listbox" aria-labelledby="drawer-kernel-label" @keydown="move" @keydown.tab="close()">
        <button v-for="(option, i) in options" :key="option" :ref="el => { items[i] = el; }" type="button" role="option" tabindex="-1" :aria-selected="option === modelValue" @click="choose(option)">
          <span>{{ option }}</span><svg v-if="option === modelValue" width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m4 10 4 4 8-8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch, onMounted, onBeforeUnmount } from 'vue';
const props = defineProps({ modelValue: String, options: { type: Array, default: () => [] }, active: Boolean });
const emit = defineEmits(['update:modelValue']);
const root = ref(null), trigger = ref(null), open = ref(false), items = ref([]);
let focusedIndex = 0;
async function show() {
  open.value = true;
  focusedIndex = Math.max(0, props.options.indexOf(props.modelValue));
  await nextTick();
  items.value[focusedIndex]?.focus();
}
function close(restore = false) {
  open.value = false;
  if (restore) trigger.value?.focus();
}
function choose(value) { emit('update:modelValue', value); close(true); }
function escape(event) {
  if (!open.value) return;
  event.stopPropagation();
  event.preventDefault();
  close(true);
}
function move(event) {
  const last = props.options.length - 1;
  if (event.key === 'ArrowDown') focusedIndex = Math.min(last, focusedIndex + 1);
  else if (event.key === 'ArrowUp') focusedIndex = Math.max(0, focusedIndex - 1);
  else if (event.key === 'Home') focusedIndex = 0;
  else if (event.key === 'End') focusedIndex = last;
  else return;
  event.preventDefault();
  items.value[focusedIndex]?.focus();
}
function outside(event) { if (!root.value?.contains(event.target)) close(); }
watch(() => props.active, active => { if (!active) close(); });
onMounted(() => document.addEventListener('pointerdown', outside));
onBeforeUnmount(() => document.removeEventListener('pointerdown', outside));
</script>
