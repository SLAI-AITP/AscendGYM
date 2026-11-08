<template>
  <router-view />
  <EntryDrawer />
  <DataTooltip />
  <AppToast />
</template>

<script setup>
import { onMounted, watch } from 'vue';
import { useRoute } from 'vue-router';
import EntryDrawer from './components/EntryDrawer.vue';
import DataTooltip from './components/DataTooltip.vue';
import AppToast from './components/AppToast.vue';
import { ui, showTooltip, hideTooltip, openDrawer, closeDrawer, focusSearch } from './store/ui';

const route = useRoute();

// Mirror reactive UI state onto the body/html attribute contract that
// styles.css already understands (kept unchanged from the original build).
watch(() => ui.theme, theme => { document.documentElement.dataset.theme = theme; }, { immediate: true });
watch(() => ui.exportMode, value => document.body.classList.toggle('export-mode', value));
watch(() => ui.exportBreakpoint, value => {
  document.documentElement.classList.remove('export-tablet', 'export-mobile');
  if (value) document.documentElement.classList.add(`export-${value}`);
});
// The design-system board styles itself through body.ds-body.
watch(() => route.name, name => document.body.classList.toggle('ds-body', name === 'design-system'), { immediate: true });
// Drawer open state locks background scroll.
watch(() => ui.drawerModelId, id => { document.body.style.overflow = id ? 'hidden' : ''; });

// Linked highlighting (C1 keeps the v1 contract: one stable identity per
// entry across the whole page). The attribute drives the generic dimming
// rule; matching elements get .hl-match via class toggling so ANY entry id
// (live backend data included) restores full opacity — the v1 stylesheet
// only enumerated its original mock ids.
watch(() => ui.highlightModel, value => {
  if (value) {
    document.body.dataset.highlightModel = value;
    document.querySelectorAll('[data-model-id]').forEach(el => {
      el.classList.toggle('hl-match', el.dataset.modelId === value);
    });
  } else {
    delete document.body.dataset.highlightModel;
    document.querySelectorAll('.hl-match').forEach(el => el.classList.remove('hl-match'));
  }
}, { immediate: true });

onMounted(() => {
  // Event delegation for chart fragments rendered through v-html: they carry
  // data-open-drawer / data-model-id / data-tooltip-html attributes, exactly
  // like the original prototype.
  document.addEventListener('click', event => {
    const target = event.target.closest('[data-open-drawer]');
    if (target) openDrawer(target.dataset.modelId);
  });
  document.addEventListener('pointerover', event => {
    const target = event.target.closest('[data-model-id]');
    if (target) ui.highlightModel = target.dataset.modelId;
  });
  document.addEventListener('pointerout', event => {
    const target = event.target.closest('[data-model-id]');
    if (!target || (event.relatedTarget && target.contains(event.relatedTarget))) return;
    ui.highlightModel = null;
  });
  document.addEventListener('pointermove', event => {
    const target = event.target.closest('[data-tooltip-html]');
    if (target) showTooltip(target.dataset.tooltipHtml, event.clientX, event.clientY);
    else hideTooltip();
  });
  document.body.addEventListener('pointerleave', hideTooltip);
  document.addEventListener('keydown', event => {
    const target = event.target;
    if (event.key === '/' && !/INPUT|SELECT|TEXTAREA/.test(target.tagName)) {
      event.preventDefault();
      focusSearch();
    }
    if (event.key === 'Escape' && ui.drawerModelId) closeDrawer();
  });
});
</script>
