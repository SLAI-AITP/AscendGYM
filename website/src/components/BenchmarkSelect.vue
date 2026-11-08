<template>
  <div ref="root" class="benchmark-select" @keydown.esc="escape">
    <span :id="`${id}-label`" class="benchmark-select-label">{{ label }}</span>
    <select :id="id" class="benchmark-select-native" :aria-label="label" :value="modelValue" @change="choose($event.target.value, false)">
      <option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option>
    </select>
    <div class="benchmark-select-custom">
      <button ref="trigger" type="button" class="benchmark-select-trigger" role="combobox" :aria-labelledby="`${id}-label ${id}-value`" aria-haspopup="listbox" :aria-expanded="open" :aria-controls="`${id}-options`" @click="open ? close() : show()" @keydown.down.prevent="show()" @keydown.up.prevent="show()">
        <span :id="`${id}-value`">{{ selectedLabel }}</span>
        <svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m5 8 5 5 5-5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" /></svg>
      </button>
      <div v-if="open" :id="`${id}-options`" class="benchmark-select-menu" role="listbox" :aria-labelledby="`${id}-label`" @keydown="move" @keydown.tab="close()">
        <button v-for="(option, i) in options" :key="option.value" :ref="el => { items[i] = el; }" type="button" role="option" tabindex="-1" :aria-selected="option.value === modelValue" @click="choose(option.value)">
          <span>{{ option.label }}</span><svg v-if="option.value === modelValue" width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m4 10 4 4 8-8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, nextTick, onMounted, onBeforeUnmount } from 'vue';
const props = defineProps({ id: { type: String, required: true }, label: String, modelValue: String, options: { type: Array, default: () => [] } });
const emit = defineEmits(['update:modelValue']);
const root = ref(null), trigger = ref(null), open = ref(false), items = ref([]);
const selectedLabel = computed(() => props.options.find(option => option.value === props.modelValue)?.label ?? '');
let focusedIndex = 0;
async function show() {
  if (!props.options.length) return;
  open.value = true;
  focusedIndex = Math.max(0, props.options.findIndex(option => option.value === props.modelValue));
  await nextTick();
  items.value[focusedIndex]?.focus();
}
function close(restore = false) { open.value = false; if (restore) trigger.value?.focus(); }
function choose(value, restore = true) { emit('update:modelValue', value); close(restore); }
function escape(event) { if (open.value) { event.stopPropagation(); event.preventDefault(); close(true); } }
function move(event) {
  const last = props.options.length - 1;
  if (event.key === 'ArrowDown') focusedIndex = Math.min(last, focusedIndex + 1);
  else if (event.key === 'ArrowUp') focusedIndex = Math.max(0, focusedIndex - 1);
  else if (event.key === 'Home') focusedIndex = 0;
  else if (event.key === 'End') focusedIndex = last;
  else return;
  event.preventDefault(); items.value[focusedIndex]?.focus();
}
function outside(event) { if (!root.value?.contains(event.target)) close(); }
onMounted(() => document.addEventListener('pointerdown', outside));
onBeforeUnmount(() => document.removeEventListener('pointerdown', outside));
</script>
