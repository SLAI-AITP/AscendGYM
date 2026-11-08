<template>
  <section class="section page-shell" id="imports" aria-labelledby="imports-title">
    <!-- Omitted on the standalone Imports page, which supplies its own title. -->
    <div v-if="showHeading" class="section-heading">
      <div>
        <p class="section-index">{{ section.index }} / {{ section.kicker }}</p>
        <h2 id="imports-title">{{ section.title }}</h2>
        <p v-for="(line, i) in sectionSummary(section)" :key="i">{{ line }}</p>
      </div>
      <div class="direction-badge"><span>⇩</span> Published measurements · imports remain local</div>
    </div>

    <article class="component-surface import-panel import-panel-json">
      <header class="import-panel-head">
        <div>
          <h3>1. Evaluation Results JSON</h3>
          <p>Refer to the core JSON content below to select the corresponding evaluation results file from your recorded runs.</p>
        </div>
      </header>
      <figure class="evaluation-json-preview">
        <figcaption>
          <span class="evaluation-terminal-file"><span class="evaluation-terminal-dots" aria-hidden="true"><i></i><i></i><i></i></span><code>eval_summary.json</code></span>
          <span>Claude Code × Claude-Opus-5</span>
        </figcaption>
        <pre tabindex="0" aria-label="Evaluation results JSON core excerpt"><code>{{ evaluationJson }}</code></pre>
        <div class="evaluation-json-note">Core excerpt · additional runs and fields omitted</div>
      </figure>
    </article>

    <!-- --------------------- reviewer imports -------------------------
         Bring your own evaluator summary and compare it against the shipped
         results. Two rules keep this from turning into a way to edit the board:

           • imported entries are ADDITIVE — an id the project already ships wins,
             so a local file can never change what everyone else sees;
           • they live in localStorage only: never uploaded, never committed.

         The panel stores the RAW summary JSON and re-parses it on load, so
         imports always go through the same derivation as the built-in folder.
         Storing the parsed entry would freeze whatever the parser did on upload
         day, and a later metric fix would apply to only half the board. -->
    <article class="component-surface import-panel evaluation-upload-panel" id="evaluation-upload">
      <header class="import-panel-head">
        <div>
          <h3 id="evaluation-upload-title">2. Import Evaluation Data</h3>
          <p>Select the generated <code>eval_summary.json</code> file from your evaluation output folder and upload it here.</p>
        </div>
      </header>

      <p v-if="submitterError" id="submitter-error" class="submitter-error" role="alert">{{ submitterError }}</p>
      <div
        class="upload-drop"
        :class="{ 'is-over': dragOver }"
        @dragover.prevent="dragOver = true"
        @dragleave="dragOver = false"
        @drop.prevent="onDrop"
      >
        <span class="upload-entry-icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="26" height="26" fill="none"><path d="M12 16V4m-4 4 4-4 4 4M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" /></svg></span>
        <span class="upload-entry-copy"><strong>Upload your evaluation results</strong><small>JSON files from your evaluation output folder</small></span>
        <button class="evaluation-file-button" type="button" :aria-expanded="submitterOpen" aria-controls="submitter-details" :aria-describedby="submitterError ? 'submitter-error' : undefined" @click="pickFiles"><svg width="17" height="17" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M3 5h5l2 2h7v9H3V5Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" /></svg>Choose Files</button>
        <button v-if="uploads.files.length" class="import-secondary" type="button" @click="clearAllUploads">Remove all</button>
      </div>

      <Transition name="submitter-panel">
        <form v-if="submitterOpen" id="submitter-details" class="submitter-details" aria-labelledby="submitter-details-title" novalidate @submit.prevent="pickFiles">
          <h4 id="submitter-details-title">Submitter Details</h4>
          <div class="submitter-fields">
            <label for="submitter-organization">Organization <span aria-hidden="true">*</span><input id="submitter-organization" v-model="submitter.organization" name="organization" autocomplete="organization" placeholder="Institution or company" required :aria-invalid="submitterAttempted && !submitter.organization.trim()" :aria-describedby="submitterError ? 'submitter-error' : undefined" /></label>
            <label for="submitter-email">Email Address <span aria-hidden="true">*</span><input id="submitter-email" v-model="submitter.email" name="email" type="email" autocomplete="email" placeholder="name@organization.org" required :aria-invalid="submitterAttempted && (!submitter.email.trim() || !submitterCheck.ok && !!submitter.organization.trim() && !!submitter.name.trim())" :aria-describedby="submitterError ? 'submitter-error' : undefined" /></label>
            <label for="submitter-name">Full Name <span aria-hidden="true">*</span><input id="submitter-name" v-model="submitter.name" name="name" autocomplete="name" placeholder="Your full name" required :aria-invalid="submitterAttempted && !submitter.name.trim()" :aria-describedby="submitterError ? 'submitter-error' : undefined" /></label>
          </div>
          <div class="submitter-actions"><span>Complete all fields to continue.</span><button class="evaluation-file-button" type="submit">Upload Evaluation Results <span aria-hidden="true">→</span></button></div>
        </form>
      </Transition>

      <p v-if="importMessage" class="import-hint" :class="importMessage.ok ? 'tone-ok-text' : 'tone-bad-text'">
        {{ importMessage.text }}
      </p>

      <ul v-if="uploads.files.length" class="import-datasets">
        <li v-for="row in uploadedRows" :key="row.name" :class="{ 'is-off': row.status !== 'ok' || !isEntryEnabled(row.entry.id) }">
          <input
            :id="`up-${row.name}`"
            class="dataset-toggle"
            type="checkbox"
            :checked="row.status === 'ok' && isEntryEnabled(row.entry.id)"
            :disabled="row.status !== 'ok'"
            @change="setEnabled(row.entry.id, $event.target.checked)"
          >
          <label :for="`up-${row.name}`">
            <b class="mono">{{ row.name }}</b>
            <small v-if="row.status === 'ok'">
              {{ harnessDisplayName(row.entry.agent) }} × {{ modelDisplayName(row.entry.engine_model) }}
              · {{ row.entry.coverage_operators }} Operator
              · Imported on {{ String(row.addedAt).slice(0, 10) }}
            </small>
            <small v-else class="tone-bad-text">{{ row.message }}</small>
          </label>
          <button class="upload-remove" type="button" :aria-label="`Remove ${row.name}`" @click="removeUploaded(row.name)">✕</button>
        </li>
      </ul>
      <p v-else class="import-hint">
        No locally imported datasets. The board currently uses the included measurements.
      </p>

      <p v-if="collidedIds.size" class="import-hint tone-warn-text">
        The following imports duplicate included identities. The published measurements take precedence.
      </p>

      <ul v-if="rejectedUploads.length" class="import-datasets import-rejected">
        <li v-for="row in rejectedUploads" :key="`rej-${row.name}`">
          <label>
            <b class="mono">{{ row.name }}</b>
            <small class="tone-bad-text">Rejected:{{ row.message }}</small>
          </label>
        </li>
      </ul>
    </article>

    <Teleport to="body">
      <dialog ref="submissionDialog" class="submission-dialog" aria-labelledby="submission-dialog-title" aria-describedby="submission-dialog-description" @click="closeOnBackdrop">
        <button class="submission-dialog-close" type="button" aria-label="Close submission notice" @click="closeSubmissionDialog"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" /></svg></button>
        <span class="submission-dialog-icon" aria-hidden="true"><svg width="26" height="26" viewBox="0 0 24 24" fill="none"><rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" stroke-width="1.6" /><path d="m4 7 8 6 8-6" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" /></svg></span>
        <h3 id="submission-dialog-title">Leaderboard Submission</h3>
        <p id="submission-dialog-description">To request an evaluation and submit your results to the leaderboard, please contact:</p>
        <a class="submission-dialog-email" href="mailto:lidongfang@hit.edu.cn">lidongfang@hit.edu.cn <span aria-hidden="true">↗</span></a>
        <div class="submission-dialog-actions"><button class="evaluation-file-button" type="button" autofocus @click="closeSubmissionDialog">Got it</button></div>
      </dialog>
    </Teleport>
  </section>
</template>

<script setup>
import { harnessDisplayName, modelDisplayName } from '../lib/displayNames';
import { computed, onBeforeUnmount, ref } from 'vue';
import { mergedRoster } from '../store/benchmark';
import {
  uploads, uploadedEntries, lastImport, rejectedUploads,
  removeUpload, clearUploads
} from '../store/uploads';
import { validateSubmitter } from '../lib/submission';
import { showToast } from '../store/ui';
import { SECTIONS, sectionSummary } from '../lib/sections';
import { isEntryEnabled, setEntryEnabled } from '../store/datasets';
import evaluationResultsCore from '../data/evaluation_results_core.json';

const evaluationJson = JSON.stringify(evaluationResultsCore, null, 2)
  .replace(/\{\n\s+"id": "([^"]+)"\n\s+\}/g, '{ "id": "$1" }');

defineProps({
  /** The standalone Imports page renders its own heading. */
  showHeading: { type: Boolean, default: true }
});

const section = SECTIONS.imports;

// ------------------------------- uploads ------------------------------------
//
// The board is assembled by `mergeRosters()` in the store, not here: this panel
// only reports what is on it. Keeping the merge out of the component means the
// leaderboard, hero counters and every chart see the same roster without this
// panel having to be mounted.

const submissionDialog = ref(null);
const dragOver = ref(false);
const submitterOpen = ref(false);
const submitterAttempted = ref(false);
const submitter = ref({ organization: '', email: '', name: '' });
const submitterCheck = computed(() => validateSubmitter(submitter.value));
const submitterError = computed(() => submitterAttempted.value && !submitterCheck.value.ok ? submitterCheck.value.message : '');

function requireSubmitter() {
  submitterOpen.value = true;
  submitterAttempted.value = true;
  return submitterCheck.value.ok ? { ...submitterCheck.value.value } : null;
}

/**
 * Ids the project already ships that an import also claimed.
 *
 * Extracted from the merge report rather than recomputed: `mergeRosters()` is
 * the single place that decides who wins, and re-deriving the verdict here could
 * drift from what the board actually shows.
 */
const collidedIds = computed(() => new Set(
  mergedRoster.value.problems
    .map(text => text.match(/entry "([^"]+)" duplicates an included identity/)?.[1])
    .filter(Boolean)
));

/** Per-file rows for the imported list, with the merge verdict folded in. */
const uploadedRows = computed(() => uploadedEntries.value.rows.map(row => {
  const id = row.entry?.id ?? null;
  // Losing to a built-in entry is not the file's fault, so it gets its own
  // message: the file is valid and was accepted, it just did not win the id.
  if (row.status === 'ok' && id && collidedIds.value.has(id)) {
    return { ...row, status: 'collides', message: `Identity "${id}" duplicates an included dataset; the included version takes precedence` };
  }
  return row;
}));

/** Rows that made it onto the board, for the header counter. */
const onBoardRows = computed(() => uploadedRows.value.filter(row => row.status === 'ok'));

const importMessage = computed(() => {
  const last = lastImport.value;
  if (!last) return null;
  if (last.cleared) return { ok: true, text: 'All imports removed; included measurements restored.' };
  if (last.removed) return { ok: true, text: `Removed ${last.name}.` };
  if (!last.ok) return { ok: false, text: `Import failed:${last.message}` };
  // Counts ACCEPTED FILES against ENTRIES ON BOARD, which differ: two uploads can
  // describe the same entry, and reporting "2 entries" there would be a number
  // the leaderboard does not show.
  const accepted = last.accepted ?? 1;
  const onBoard = onBoardRows.value.length;
  if (!onBoard) return { ok: false, text: `Read ${accepted} file${accepted === 1 ? '' : 's'}, but no entry was added to the board; see details below.` };
  return { ok: true, text: `Imported ${accepted} files —— ${onBoard} entries added to the comparison.` };
});

function pickFiles() {
  if (!requireSubmitter()) return;
  if (!submissionDialog.value?.open) submissionDialog.value?.showModal();
}

function closeSubmissionDialog() {
  submissionDialog.value?.close();
}

function closeOnBackdrop(event) {
  if (event.target !== submissionDialog.value) return;
  const bounds = submissionDialog.value.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeSubmissionDialog();
}

onBeforeUnmount(closeSubmissionDialog);

function onDrop(event) {
  dragOver.value = false;
  if (event.dataTransfer?.files?.length) pickFiles();
}

function removeUploaded(name) {
  removeUpload(name);
  showToast(`Removed ${name}`);
}

function clearAllUploads() {
  const count = uploads.files.length;
  clearUploads();
  showToast(`Removed all ${count} imported datasets`);
}

function setEnabled(id, enabled) {
  if (id) setEntryEnabled(id, enabled);
}
</script>

<style scoped>
.submission-dialog { box-sizing:border-box; width:min(480px,calc(100vw - 40px)); max-height:calc(100dvh - 40px); margin:auto; padding:32px; overflow:auto; border:1px solid var(--border); border-radius:18px; background:var(--surface); color:var(--ink); box-shadow:var(--shadow-floating); font-family:var(--font-sans); }
.submission-dialog::backdrop { background:rgba(16,28,27,.38); backdrop-filter:blur(5px); }
.submission-dialog-close { position:absolute; top:16px; right:16px; display:grid; place-items:center; width:32px; height:32px; padding:0; border:1px solid var(--border); border-radius:9px; color:var(--muted); background:var(--surface); cursor:pointer; }
.submission-dialog-icon { display:grid; place-items:center; width:52px; height:52px; margin-bottom:20px; border:1px solid color-mix(in srgb,var(--compute) 22%,transparent); border-radius:15px; background:color-mix(in srgb,var(--compute) 9%,var(--surface)); color:var(--compute-dark); }
.submission-dialog h3 { margin:0 32px 12px 0; font-size:24px; font-weight:700; letter-spacing:-.025em; line-height:1.25; }
.submission-dialog p { margin:0; color:var(--muted); font-size:16px; line-height:1.65; }
.submission-dialog-email { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:20px; padding:14px 16px; border:1px solid color-mix(in srgb,var(--compute) 22%,var(--border)); border-radius:10px; background:color-mix(in srgb,var(--compute) 5%,var(--surface)); color:var(--compute-dark); font-size:17px; font-weight:650; text-decoration:none; overflow-wrap:anywhere; }
.submission-dialog-actions { display:flex; justify-content:flex-end; margin-top:26px; }
.submission-dialog .evaluation-file-button { min-height:42px; padding:10px 24px; font-size:14px; }
.submission-dialog-close:hover,.submission-dialog-email:hover { border-color:var(--compute); }
.submission-dialog :focus-visible { outline:2px solid var(--compute); outline-offset:3px; }
@media(max-width:600px) { .submission-dialog { padding:26px 22px; } .submission-dialog h3 { font-size:22px; } .submission-dialog-email { font-size:16px; } }
</style>
