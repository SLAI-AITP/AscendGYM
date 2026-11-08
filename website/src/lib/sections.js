// ============================================================================
// Section registry — the single source of truth linking the top navigation to
// the sections rendered below it.
//
// Every section reads its own number/kicker from here and the header builds
// its nav from the same list, so the two can no longer drift apart (the earlier
// bug was a hand-written nav that skipped sections 02 / 03 / 05 / 06).
//
// `route` marks a section that lives on its own page instead of the main
// leaderboard scroll. Standalone pages are listed in `PAGE_SECTION_ORDER` and
// are reachable only from the top navigation.
//
// ---------------------------------------------------------------------------
// To remove a section, delete it from BOTH its order list and `SECTIONS`.
// A key listed in an order without a registry entry resolves to `undefined`,
// which previously propagated into the nav and crashed the page with
// "Cannot read properties of undefined (reading 'id')". The assertion at the
// bottom of this file now fails loudly at import time instead of silently
// shipping a broken nav.
// ---------------------------------------------------------------------------
//
// `summary` is the SUBTITLE RENDERED UNDER THE SECTION TITLE. Components must
// not hand-write that <p>: doing so silently orphaned this field, so editing
// the registry changed nothing on screen. A summary is one of:
//
//   • a string            — a single line;
//   • an array of strings — genuinely two lines (Quality × cost);
//   • a function          — when the text quotes a live count. The rubric
//                           names how many columns it renders, and that number
//                           has to come from `rubricColumns`, not from prose.
//                           It receives the caller's context, e.g. { columns }.
//
// Always render it through `sectionSummary()`, never `section.summary`.
// ---------------------------------------------------------------------------
// ============================================================================

/** Ordered content sections rendered on the main leaderboard page. */
export const MAIN_SECTION_ORDER = [
  'leaderboard',
  'quality-cost',
  'operators',
  'gain'
];

/**
 * Ordered standalone pages. They are not part of the leaderboard scroll, but
 * the top navigation links to them exactly like a content section, so they
 * still carry an index / label / kicker from the same registry.
 */
export const PAGE_SECTION_ORDER = ['methodology', 'imports'];

export const SECTIONS = {
  leaderboard: {
    id: 'leaderboard',
    index: '01',
    kicker: 'STS',
    label: 'STS',
    title: 'Speedup-Token Score (STS)',
    summary: 'Speedup-Token Score (STS) is the core metric of AscendGYM, jointly characterizing performance under a specified cost budget.'
  },
  'quality-cost': {
    id: 'quality-cost',
    index: '02',
    kicker: '02 / PERF × COST',
    label: 'Perf×Cost',
    title: 'Performance × Cost',
    summary: 'GM speedup versus cumulative tokens or estimated API cost; higher speedup and lower expenditure are preferred.'
  },
  operators: {
    id: 'operators',
    index: '04',
    kicker: 'KERNEL PANEL',
    label: 'Details',
    title: 'Kernel Panel',
    summary: 'Kernel-level results'
  },
  gain: {
    id: 'gain',
    index: '05',
    kicker: 'PERFORMANCE GAIN',
    label: 'Performance',
    title: 'Performance Gain',
    summary: 'Correctness-validated performance gains by harness, model, and kernel.'
  },
  methodology: {
    id: 'methodology',
    index: '05',
    kicker: 'METRICS',
    label: 'Metrics',
    title: 'Evaluation Metrics',
    summary: 'Metric definitions aligned with Section 3.3 of the AscendGYM Technical Report.',
    route: '/methodology' // lives on its own page
  },
  imports: {
    id: 'imports',
    index: '06',
    kicker: 'IMPORT DATA',
    label: 'Import',
    title: 'Import Your Evaluation Data',
    summary: 'Set up the environment provided in the open-source repository, including the sandbox and supporting code. Load the default or a custom harness, generate the evaluation files, and upload them to display your results on the AscendGYM leaderboard.',
    route: '/imports' // lives on its own page
  }
};

/**
 * Resolve an order list against the registry.
 *
 * Fail fast on a registry/order mismatch: an `undefined` entry means the order
 * names a section that no longer exists, and the failure would otherwise
 * surface far away as a crash inside the nav or the scroll spy.
 */
function resolveOrder(order, listName) {
  return order.map(key => {
    const section = SECTIONS[key];
    if (!section) {
      throw new Error(
        `sections.js: ${listName} 引用了未注册的章节 "${key}"。` +
        `请在 SECTIONS 中补回该章节，或从 ${listName} 中删除它。`
      );
    }
    return section;
  });
}

/**
 * Render a section's `summary` as a list of display lines.
 *
 * Normalising here means every caller uses the same `v-for` and no component
 * has to know which of the three summary shapes the registry chose. An absent
 * summary yields `[]` rather than `undefined`, so `v-for` cannot blow up.
 */
export function sectionSummary(section, context = {}) {
  const value = typeof section?.summary === 'function' ? section.summary(context) : section?.summary;
  if (Array.isArray(value)) return value;
  return value == null ? [] : [value];
}

/**
 * Fail fast when a section ships without usable subtitle copy.
 *
 * A missing summary renders as a silently empty <p> — which reads as a design
 * decision, not a bug. That is precisely how this field went unused for months.
 * Same philosophy as `resolveOrder` below: throw at import time.
 */
function assertSummaries() {
  const offenders = Object.entries(SECTIONS)
    .filter(([, section]) => {
      const value = section.summary;
      if (typeof value === 'function') return false;
      if (Array.isArray(value)) {
        return !value.length || value.some(line => typeof line !== 'string' || !line.trim());
      }
      return typeof value !== 'string' || !value.trim();
    })
    .map(([key]) => key);

  if (offenders.length) {
    throw new Error(
      `sections.js: 章节 ${offenders.join(' / ')} 缺少 summary——子标题会渲染成空白。` +
      `请在 SECTIONS 中补上 summary，或从注册表中Remove该章节。`
    );
  }
}

assertSummaries();

/** Main-page sections in render order (the leaderboard long-scroll). */
export const MAIN_SECTIONS = resolveOrder(MAIN_SECTION_ORDER, 'MAIN_SECTION_ORDER');

/** Standalone pages in navigation order. */
export const PAGE_SECTIONS = resolveOrder(PAGE_SECTION_ORDER, 'PAGE_SECTION_ORDER');

/**
 * Top-navigation model: content sections followed by the standalone pages,
 * both derived from the registry above so the nav can never drift from the
 * headings that actually render.
 */
export const NAV_ITEMS = [...MAIN_SECTIONS, ...PAGE_SECTIONS];
