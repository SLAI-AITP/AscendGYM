// Bundle access layer (§6.0).
//
// Load order:
//   1. an explicit `?dataset=` request (bundled alias, or a URL to fetch),
//   2. the live `GET /api/bundle` endpoint (short timeout),
//   3. the project's own datasets in `src/data/eval_datasets_summary/` — the
//      default. Adding a dataset is a file drop; see README §3.4.
//
// A built `operator_bundle.json` is no longer required at runtime: the folder is
// parsed in the browser with the same code the build script uses, so `npm run
// dev` and a built `dist/` both reflect whatever the folder currently holds.
//
// Drawer drill-downs use the §6.2 sub-endpoints when `source === 'live'`;
// with a static bundle those details are embedded per operator (`runs[]`).

import { loadInternalDatasets } from './datasets';

const BASE = (import.meta.env.BASE_URL || '/').replace(/\/$/, '');

// Remembered across in-app navigation: switching routes (e.g. to /imports)
// drops the query string, so the chosen dataset would silently reset.
const DATASET_STORAGE_KEY = 'ascendgym.dataset';
const LEGACY_DATASET_STORAGE_KEY = 'perfbench.dataset';

/**
 * Move the remembered dataset from its pre-rename key.
 *
 * sessionStorage rather than localStorage, which is why `lib/storage.js` (the
 * localStorage helper the stores use) does not cover it.
 *
 * Called EAGERLY at module scope, not from `readStoredDataset()`. Hanging it off
 * the read path looked equivalent but was not: when the URL already carries
 * `?dataset=…`, `readStoredDataset()` is never reached, so a legacy key would
 * survive untouched — and the next in-app route change (which drops the query
 * string) would read the empty new key and silently reset the dataset.
 */
let datasetKeyMigrated = false;
function migrateDatasetKey() {
  if (datasetKeyMigrated || typeof window === 'undefined') return;
  datasetKeyMigrated = true;
  try {
    if (window.sessionStorage.getItem(DATASET_STORAGE_KEY) != null) {
      // New key already authoritative — drop a stale legacy key if one exists.
      window.sessionStorage.removeItem(LEGACY_DATASET_STORAGE_KEY);
      return;
    }
    const legacy = window.sessionStorage.getItem(LEGACY_DATASET_STORAGE_KEY);
    if (legacy == null) return;
    window.sessionStorage.setItem(DATASET_STORAGE_KEY, legacy);
    window.sessionStorage.removeItem(LEGACY_DATASET_STORAGE_KEY);
  } catch {
    // Storage disabled — the dataset still resolves from the URL each load.
  }
}

migrateDatasetKey();

/**
 * Aliases resolved against a bundled file instead of the network.
 * `synthetic` is the deterministic UX prototype data (not a measurement);
 * `operator` / `measured` remain as aliases for the internal dataset folder,
 * which is already the default, so an old bookmark still resolves.
 */
const DATASETS = {
  synthetic: { loader: () => import('../data/bundle.json'), source: 'local-bundle' },
  demo: { loader: () => import('../data/bundle.json'), source: 'local-bundle' }
};

/** The internal dataset folder, assembled in the browser. */
async function loadMeasuredBundle() {
  const { bundle, problems } = await loadInternalDatasets();
  return { data: bundle, source: 'internal-datasets', problems };
}

/**
 * An alias that means "the internal datasets" (the default board). Resolved to
 * the folder rather than to a checked-in bundle file, so it can never go stale.
 */
const MEASURED_ALIASES = new Set(['operator', 'operator-bundle', 'measured', 'internal', 'datasets']);

function readDatasetFromUrl() {
  if (typeof window === 'undefined') return null;
  const merged = new URLSearchParams(window.location.search);
  const hash = window.location.hash;
  const hashQuery = hash.includes('?') ? hash.slice(hash.indexOf('?') + 1) : '';
  new URLSearchParams(hashQuery).forEach((value, key) => merged.set(key, value));
  return merged.get('dataset') || merged.get('bundle') || null;
}

function readStoredDataset() {
  if (typeof window === 'undefined') return null;
  migrateDatasetKey();
  try {
    return window.sessionStorage.getItem(DATASET_STORAGE_KEY);
  } catch {
    return null;
  }
}

function storeDataset(value) {
  if (typeof window === 'undefined' || !value) return;
  try {
    window.sessionStorage.setItem(DATASET_STORAGE_KEY, value);
  } catch {
    // sessionStorage unavailable — navigation just falls back to the default.
  }
}

function resolveRequestedDataset() {
  const fromUrl = readDatasetFromUrl();
  if (fromUrl) {
    storeDataset(fromUrl);
    return fromUrl;
  }
  return readStoredDataset();
}

/** The dataset currently in effect (URL param first, then the remembered one). */
export function activeDataset() {
  return readDatasetFromUrl() ?? readStoredDataset();
}

function looksLikeBundle(data) {
  return Boolean(data && Array.isArray(data.entries) && data.meta && data.meta.metric_directions);
}

export async function loadBundle() {
  const requestedDataset = resolveRequestedDataset();

  // An alias for the internal folder short-circuits the network: the folder IS
  // the default, so there is nothing to fetch.
  if (requestedDataset && MEASURED_ALIASES.has(requestedDataset)) return loadMeasuredBundle();

  if (requestedDataset) {
    try {
      const bundled = DATASETS[requestedDataset];
      if (bundled) {
        const local = (await bundled.loader()).default;
        return { data: local, source: bundled.source };
      }

      const candidate = /^https?:\/\//.test(requestedDataset) || requestedDataset.startsWith('data:')
        ? requestedDataset
        : new URL(requestedDataset, window.location.href).toString();

      const res = await fetch(candidate, { headers: { Accept: 'application/json' } });
      if (res.ok) {
        const data = await res.json();
        if (looksLikeBundle(data)) return { data, source: 'external-json' };
      }
    } catch {
      // Ignore dataset-specific failures and fall back to the default.
    }
  }

  if (requestedDataset === 'live' && typeof fetch === 'function') {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 1600);
      const res = await fetch(`${BASE}/api/bundle`, {
        signal: controller.signal,
        headers: { Accept: 'application/json' }
      });
      clearTimeout(timer);
      if (res.ok) {
        const data = await res.json();
        if (looksLikeBundle(data)) return { data, source: 'live' };
      }
    } catch {
      // network / 404 / timeout → fall through to the internal datasets
    }
  }
  // Default = the project's own dataset folder, not the synthetic prototype.
  return loadMeasuredBundle();
}

/** §6.2 sub-endpoint helper (live mode only). */
export async function fetchJson(url) {
  const res = await fetch(`${BASE}${url}`, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error(`GET ${url} → HTTP ${res.status}`);
  return res.json();
}
