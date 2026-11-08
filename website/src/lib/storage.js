// ============================================================================
// localStorage key migration.
//
// The project was renamed from `perfbench` to `ascendgym`. Renaming a storage
// key is NOT a cosmetic change: the old value stays under the OLD key, so a
// returning reviewer would silently lose their saved dataset selection, price
// table and developer flag. A rename without this file is data loss that looks
// like "the site forgot my settings".
//
// Each store calls `migrateKey(LEGACY_KEY, STORAGE_KEY)` before its first read.
// The move is copy-then-delete, and it runs at most once per key per session,
// so a stale legacy key left behind by a failed read cannot resurrect itself.
//
// Deliberately not handled here: `perfbench.dataset` (sessionStorage, and only
// a navigation alias — see `services/api.js`, which migrates it inline).
// ============================================================================

/** Keys already checked this session, so we do not re-scan on every read. */
const checked = new Set();

/**
 * Move a value from `legacyKey` to `key` when the new key is still empty.
 *
 * Never overwrites an existing new-key value: if a user has already used the
 * renamed build, their current settings win over whatever the old key holds.
 */
export function migrateKey(legacyKey, key) {
  if (typeof window === 'undefined') return;
  if (checked.has(key)) return;
  checked.add(key);
  try {
    const current = window.localStorage.getItem(key);
    if (current != null) return; // already migrated, or written by this build
    const legacy = window.localStorage.getItem(legacyKey);
    if (legacy == null) return;
    window.localStorage.setItem(key, legacy);
    window.localStorage.removeItem(legacyKey);
  } catch {
    // Storage disabled or full: the app still works, it just will not persist.
  }
}
