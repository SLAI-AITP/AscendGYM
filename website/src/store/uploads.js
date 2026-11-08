// ============================================================================
// Reviewer-imported evaluation data.
//
// The Imports page accepts evaluator summaries the project does not ship —
// someone else's run, an ablation, a re-run with a different scaffold — and
// merges them into the board so they can be compared side by side with the
// built-in results.
//
// WHY WE STORE THE RAW SUMMARY, NOT THE PARSED ENTRY:
// the board must be assembled by exactly one parser. If we persisted entries,
// the stored ones would have been built by whatever `lib/summary.js` looked like
// the day they were uploaded, and a later fix to the metric derivation would
// silently apply to the shipped folder but not to imports — two different
// rules for the same number. Keeping the source JSON means every load re-derives
// imports through the same code path as `src/data/eval_datasets_summary/`.
//
// WHY localStorage, WHEN PRICES ARE NOT STORED THERE:
// the two are opposites on purpose, and the distinction is about whose claim the
// number is. A price is the project's claim about a vendor, so a personal copy
// would make the published board irreproducible — hence `Token_price.json` on
// disk. An imported dataset is the VISITOR's own measurement, addressed to
// themselves; losing it on reload would be data loss. It is never merged into
// the shipped bundle and never affects anyone else's view.
//
// The size guard exists because 28 shipped summaries already weigh ~1.1 MB, and
// localStorage typically allows ~5 MB per origin (stored as UTF-16, so roughly
// double the file size). A reviewer importing many large runs would otherwise hit
// a quota error at an arbitrary moment; we refuse the file up front, by name,
// instead.
// ============================================================================

import { reactive, computed, ref } from 'vue';
import { parseDataset } from '../lib/datasets';
import { formatBytes } from '../lib/format';
import { migrateKey } from '../lib/storage';
import { validateSubmitter } from '../lib/submission';

const STORAGE_KEY = 'ascendgym.uploads.v1';
const LEGACY_STORAGE_KEY = 'perfbench.uploads.v1';

/**
 * One imported upload may be at most this large.
 *
 * 4 MB sits under the practical localStorage ceiling while leaving room for the
 * selection state and a couple of files. A single evaluator summary is ~41 kB
 * today, so the limit only fires on pathological input — but when it does, it
 * fires BEFORE the write, so the failure is a clean rejection rather than a
 * half-written store.
 */
const MAX_FILE_BYTES = 4 * 1024 * 1024;

/** Total budget for all imports combined, leaving headroom for other keys. */
const MAX_TOTAL_BYTES = 4.5 * 1024 * 1024;

function readStorage() {
  if (typeof window === 'undefined') return null;
  migrateKey(LEGACY_STORAGE_KEY, STORAGE_KEY);
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && Array.isArray(parsed.files) ? parsed : null;
  } catch {
    return null;
  }
}

/** True when the value can actually be persisted without blowing the quota. */
function tryWrite(files) {
  if (typeof window === 'undefined') return { ok: false, reason: 'no-storage' };
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ schema: 'ascendgym-uploads-v1', files }));
    return { ok: true };
  } catch (error) {
    return { ok: false, reason: error?.name === 'QuotaExceededError' ? 'quota' : 'failed' };
  }
}

const stored = readStorage();

/**
 * `files[]` holds the original JSON and optional submitter details.
 * Parsed benchmark entries are always derived again when loaded.
 */
export const uploads = reactive({
  files: Array.isArray(stored?.files) ? stored.files : []
});

/** Last import attempt's outcome, for the panel's status line. */
export const lastImport = ref(null);

/**
 * Files this session refused, kept only until reload.
 *
 * Deliberately NOT persisted: a rejected file holds no data, and re-showing
 * "bad.json was rejected" days later would describe an attempt the reviewer has
 * long since fixed. A toast covers the moment; this list covers the next few
 * minutes, which is when the reviewer is still looking at the panel.
 *
 * Without it a multi-file drop that partly failed would lose the failures
 * entirely once the toast faded — the panel would show fewer rows than files
 * dropped, with nothing saying which ones went missing.
 */
export const rejectedUploads = ref([]);

/**
 * Parse the current uploads into one ROW per file.
 *
 * A row carries its own outcome rather than the panel re-deriving it from a flat
 * problem list, because the three failure modes need different wording and
 * different fixes:
 *
 *   `ok`       — parsed and it is the first file to claim this identity
 *   `duplicate`— parses fine, but another UPLOAD already claimed the identity
 *   `invalid`  — not an evaluator summary at all
 *
 * (A fourth case, colliding with the project's own roster, can only be decided
 * after merging, so the store leaves that to `mergeRosters`.)
 *
 * Failures are per-file and collected, never thrown: one bad import must not
 * blank the whole board.
 */
export const uploadedEntries = computed(() => {
  const entries = [];
  const rows = [];
  const seen = new Set();

  for (const file of uploads.files) {
    const { ok, entry, reason } = parseDataset(file.raw, file.name);
    if (!ok) {
      rows.push({ name: file.name, status: 'invalid', entry: null, message: reason, addedAt: file.addedAt });
      continue;
    }
    if (seen.has(entry.id)) {
      rows.push({
        name: file.name,
        status: 'duplicate',
        entry: null,
        message: `Identity "${entry.id}" is already provided by another imported file`,
        addedAt: file.addedAt
      });
      continue;
    }
    seen.add(entry.id);
    entries.push(entry);
    rows.push({ name: file.name, status: 'ok', entry, message: null, addedAt: file.addedAt });
  }

  return { entries, rows, files: uploads.files.map(file => file.name) };
});

/** Total bytes currently held, for the budget readout. */
export const uploadedBytes = computed(() => uploads.files
  .reduce((sum, file) => sum + byteLength(file.raw), 0));

function byteLength(value) {
  try {
    return new TextEncoder().encode(JSON.stringify(value)).length;
  } catch {
    return 0;
  }
}

/**
 * Human-readable byte size. Re-exported from `lib/format.js` so the panel does
 * not have to know which module owns the display policy.
 */
export const bytesLabel = formatBytes;

export { MAX_FILE_BYTES, MAX_TOTAL_BYTES };

/**
 * Why this text cannot be imported, or null when it looks usable.
 *
 * Structural checks only — whether it is a *good* evaluation is the evaluator's
 * question, not ours. A file that parses into an entry is accepted; one that
 * does not is refused with the specific reason so the message is actionable
 * ("顶层缺少 runs 数组" rather than "Import失败").
 */
export function validateUpload(name, raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return 'The top-level JSON value must be an object';
  }
  // A dashboard bundle is a plausible thing to drop here, so catch it by name
  // instead of reporting the generic "not a summary".
  if (Array.isArray(raw.entries)) {
    return 'This is a dashboard bundle containing entries[]. Import an evaluator eval_summary.json instead.';
  }
  const { ok, reason } = parseDataset(raw, name);
  if (!ok) return reason;
  return null;
}

/**
 * Add or replace an uploaded file.
 *
 * Replace-by-name is the intended flow for the same run changed and re-uploaded:
 * importing `ablation.json` twice should leave ONE entry, not two rows that look
 * like a duplicate agent and confuse the totals.
 *
 * Returns `{ ok, replaced, error }`; `error` is a sentence safe to show a user.
 */
export function addUpload(name, raw, submitter = null) {
  /** Record the refusal everywhere the UI reads it, then report it. */
  const refuse = message => {
    lastImport.value = { ok: false, name, message };
    noteRejection(name, message);
    return { ok: false, replaced: false, error: message };
  };

  const problem = validateUpload(name, raw);
  if (problem) return refuse(problem);
  const details = submitter == null ? null : validateSubmitter(submitter);
  if (details && !details.ok) return refuse(details.message);

  const size = byteLength(raw);
  if (size > MAX_FILE_BYTES) {
    return refuse(`${name} size ${bytesLabel(size)}exceeds the per-file limit ${bytesLabel(MAX_FILE_BYTES)}`);
  }

  const previous = uploads.files;
  const replaced = previous.some(file => file.name === name);
  const next = [
    ...previous.filter(file => file.name !== name),
    { name, addedAt: new Date().toISOString(), raw, ...(details ? { submitter: details.value } : {}) }
  ];

  const total = next.reduce((sum, file) => sum + byteLength(file.raw), 0);
  if (total > MAX_TOTAL_BYTES) {
    return refuse(`Total size after import: ${bytesLabel(total)}exceeds the limit ${bytesLabel(MAX_TOTAL_BYTES)} — remove some datasets first`);
  }

  const write = tryWrite(next);
  if (!write.ok) {
    return refuse(write.reason === 'quota'
      ? 'Browser storage is full. Remove some imported datasets and retry.'
      : 'Browser storage write failed; the import was not saved');
  }

  uploads.files = next;
  // A file that was refused earlier and now imports cleanly must not keep
  // showing up as rejected.
  rejectedUploads.value = rejectedUploads.value.filter(item => item.name !== name);
  lastImport.value = { ok: true, name, replaced };
  return { ok: true, replaced, error: null };
}

/** Remove one upload by file name. */
export function removeUpload(name) {
  const next = uploads.files.filter(file => file.name !== name);
  tryWrite(next);
  uploads.files = next;
  rejectedUploads.value = rejectedUploads.value.filter(item => item.name !== name);
  lastImport.value = { ok: true, name, removed: true };
}

/** Drop every import, returning the board to the shipped roster. */
export function clearUploads() {
  tryWrite([]);
  uploads.files = [];
  rejectedUploads.value = [];
  lastImport.value = { ok: true, cleared: true };
}

/** Record a refused file for the session (see `rejectedUploads`). */
export function noteRejection(name, message) {
  const next = rejectedUploads.value.filter(item => item.name !== name);
  next.push({ name, message });
  rejectedUploads.value = next;
}
