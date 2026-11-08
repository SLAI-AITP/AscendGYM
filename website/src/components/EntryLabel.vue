<template>
  <span class="model-label entry-label" :class="{ 'entry-label-stacked': stackedIcons }">
    <template v-if="stackedIcons">
      <VendorChip class="entry-model-icon" :model="entry.engine_model" :size="size" />
      <strong class="entry-model-name">{{ modelDisplayName(entry.engine_model) }}<span class="weight-tag">{{ entry.engine_model.openness }}</span></strong>
      <HarnessChip class="entry-harness-icon" :agent="entry.agent" :size="size" />
      <small class="entry-harness-name">{{ subtitle }}</small>
    </template>
    <template v-else>
    <span
      v-if="showAgentBadge"
      class="agent-badge"
      :data-size="size"
      :data-type="entry.agent.type"
      :data-tooltip-html="agentTooltip"
    >{{ initials }}</span>
    <VendorChip :model="entry.engine_model" :size="size" />
    <span class="model-text">
      <strong>{{ modelDisplayName(entry.engine_model) }}<span class="weight-tag">{{ entry.engine_model.openness }}</span></strong>
      <small>{{ subtitle }}</small>
    </span>
    </template>
    <!-- Marks a reviewer-supplied row. Deliberately an explicit tag rather than
         a subtle tint: an imported entry can outrank a shipped one, and a reader
         comparing two boards must be able to tell whose number they are looking
         at without opening the drawer. -->
    <span v-if="isImported" class="origin-tag" title="Locally imported dataset; available only in this browser">Import</span>
  </span>
</template>

<script setup>
import { harnessDisplayName, modelDisplayName } from '../lib/displayNames';
import { computed } from 'vue';
import VendorChip from './VendorChip.vue';
import HarnessChip from './HarnessChip.vue';
import { agentInitials } from '../lib/meta';
import { attributeEscape } from '../lib/format';

const props = defineProps({
  entry: { type: Object, required: true },
  size: { type: String, default: 'sm' },
  showAgentBadge: { type: Boolean, default: true },
  stackedIcons: { type: Boolean, default: false },
  showVendorInSubtitle: { type: Boolean, default: true }
});

const initials = computed(() => agentInitials(props.entry.agent.id));

const subtitle = computed(() => (
  props.showVendorInSubtitle && props.entry.engine_model.vendor
    ? `${harnessDisplayName(props.entry.agent)} · ${props.entry.engine_model.vendor}`
    : harnessDisplayName(props.entry.agent)
));

const isImported = computed(() => props.entry.origin === 'imported');

const agentTooltip = computed(() => attributeEscape(
  `<strong>${harnessDisplayName(props.entry.agent)}</strong><br>Agent type: ${props.entry.agent.type === 'domain' ? 'Domain-specific' : 'General-purpose'} · Strategy ${props.entry.agent.strategy} · ${props.entry.agent.version}`
));
</script>
