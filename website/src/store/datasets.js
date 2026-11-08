// ============================================================================
// Internal dataset selection.
//
// The dataset folder ships every evaluation the project has. A reviewer often
// wants to look at a subset — one agent, one vendor, a single model — without
// editing files. Two things are tracked here:
//
//   • `disabledIds` — which entries are switched OFF. The board, the hero
//     counters and every chart are computed from the enabled set only.
//   • `filter*` — transient view state for the Imports list. Filtering only
//     narrows what the list *shows*; it never changes what is enabled.
//
// Keeping them separate matters: narrowing a filter to inspect one dataset must
// not silently drop the others from the leaderboard.
//
// Disabled ids persist to localStorage (this is a reading preference, not data),
// while the filters reset on reload — a stale filter that hides rows is far more
// confusing than having to retype it.
// ============================================================================

import { reactive, computed } from 'vue';
import { migrateKey } from '../lib/storage';

const STORAGE_KEY = 'ascendgym.datasets.v1';
const LEGACY_STORAGE_KEY = 'perfbench.datasets.v1';

function readStorage() {
  if (typeof window === 'undefined') return null;
  migrateKey(LEGACY_STORAGE_KEY, STORAGE_KEY);
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeStorage() {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
      disabledIds: datasetSelection.disabledIds
    }));
  } catch {
    // Storage disabled — the selection still applies for this session.
  }
}

const stored = readStorage();

export const datasetSelection = reactive({
  /**
   * Entry ids switched OFF. Storing the *disabled* set — rather than the
   * enabled one — means a dataset file added later is on by default, which is
   * what a reader expects from a folder that ships the project's own results.
   */
  disabledIds: Array.isArray(stored?.disabledIds) ? stored.disabledIds : [],

  // View-only filters for the Imports list.
  filterAgent: 'all',
  filterModel: 'all',
  filterText: ''
});

/** Is this entry part of the board? Reactive: reads `disabledIds`. */
export function isEntryEnabled(id) {
  return !datasetSelection.disabledIds.includes(id);
}

export function setEntryEnabled(id, enabled) {
  const next = new Set(datasetSelection.disabledIds);
  if (enabled) next.delete(id);
  else next.add(id);
  datasetSelection.disabledIds = [...next];
  writeStorage();
}

/** Enable or disable a whole set of ids in one write (avoids N storage hits). */
export function setEntriesEnabled(ids, enabled) {
  const next = new Set(datasetSelection.disabledIds);
  for (const id of ids) {
    if (enabled) next.delete(id);
    else next.add(id);
  }
  datasetSelection.disabledIds = [...next];
  writeStorage();
}

export function resetSelection() {
  datasetSelection.disabledIds = [];
  writeStorage();
}

export function setFilter(patch) {
  Object.assign(datasetSelection, patch);
}

export function clearFilters() {
  datasetSelection.filterAgent = 'all';
  datasetSelection.filterModel = 'all';
  datasetSelection.filterText = '';
}

/** True when at least one filter is narrowing the list. */
export const isFiltered = computed(() => (
  datasetSelection.filterAgent !== 'all'
  || datasetSelection.filterModel !== 'all'
  || datasetSelection.filterText.trim() !== ''
));

/** Counts against a supplied roster, for the panel's status line. */
export function selectionCounts(entries) {
  const list = entries ?? [];
  const enabled = list.filter(entry => isEntryEnabled(entry.id)).length;
  return { total: list.length, enabled, disabled: list.length - enabled };
}
