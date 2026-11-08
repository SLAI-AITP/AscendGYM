// ============================================================================
// MathML builders for the Methodology section.
//
// WHY MathML AND NOT A LATEX LIBRARY
// The formulas are fractions, sums, radicals and sub/superscripts — the exact
// shapes MathML exists for. Chromium has shipped a native MathML Core
// implementation for years, and this project targets Chromium (Electron/VS Code
// webview and modern browsers), so real math typesetting costs zero dependencies
// and zero client-side JS at render time. Pulling in KaTeX would add ~270 kB to
// a page whose whole payload is ~230 kB, to render five lines of algebra.
//
// WHY `v-html` IS REQUIRED
// Vue's template compiler turns a literal `<math>` tag into
// `createElementVNode('math')`, and `document.createElement('math')` produces an
// HTMLUnknownElement in the XHTML namespace — the fraction then collapses instead
// of laying out (measured: 20px vs 39px). Only parsing real markup (or
// createElementNS) yields a MathMLElement. So the markup is built here as a
// string and injected, which is how every other SVG fragment in this project is
// rendered too (see lib/markup.js).
//
// Every string below is a hand-written literal. NOTHING here is interpolated
// from data, so there is no injection surface — the module takes no arguments.
// ============================================================================

/** Escape-free by construction: these are static literals, never user input. */

const MINUS = '\u2212'; // real minus sign, not a hyphen
const THIN = '<mspace width="0.28em"/>';
const QUAD = '<mspace width="1em"/>';

/** `x` as an italic math identifier. */
const id = x => `<mi>${x}</mi>`;
/**
 * A multi-letter LABEL (a subscript like "best", an acronym like "AULS").
 * `<mtext>` keeps it upright and spaces it as words, which is what a label
 * needs — unlike a function name, see `fn`.
 */
const op = x => `<mtext>${x}</mtext>`;
/**
 * A FUNCTION name: ln / log / exp.
 * MathML Core wants these as `<mi mathvariant="normal">` — upright, but still
 * spaced as a math identifier so the following operand keeps its italic
 * spacing. `<mtext>` would be wrong: that element is for natural language.
 */
const fn = x => `<mi mathvariant="normal">${x}</mi>`;
const num = x => `<mn>${x}</mn>`;
/** Subscript: base with a subscript. */
const sub = (base, s) => `<msub>${base}${s}</msub>`;
/** Superscript: base with a superscript. */
const sup = (base, s) => `<msup>${base}${s}</msup>`;
const frac = (n, d) => `<mfrac>${n}${d}</mfrac>`;
const row = (...parts) => `<mrow>${parts.join('')}</mrow>`;

const mo = (text, attrs = '') => `<mo${attrs}>${text}</mo>`;

/**
 * A large operator with its limits UNDERNEATH (Σ when summing over an index).
 * `<munder>` places the limits below the operator, which is the display-style
 * convention; `<msub>` would put them to the right and read as a subscript.
 * The trailing space keeps the summand from colliding with the operator.
 */
const sumUnder = index => `<munder><mo>∑</mo>${index}</munder><mspace width="0.16em"/>`;
/** Σ with bounds both below and above (Σ_{k=1}^{K}). */
const sumBounds = (from, to) =>
  `<munderover><mo>∑</mo>${from}${to}</munderover><mspace width="0.16em"/>`;
/** A radical around `inner`. */
const sqrt = inner => `<msqrt>${inner}</msqrt>`;

/** `=` with default spacing (an `<mo>` is a relation and gets operator spacing). */
const eq = mo('=');
const plus = mo('+');
const comma = mo(',');

/** L⁰ — baseline latency, written as a superscript zero on the italic L. */
const L0 = sup(id('L'), num('0'));
/** t_i⁰ and t_i¹ — per-operator baseline / optimised latency. */
const tiBase = sup(sub(id('t'), id('i')), num('0'));
const tiAgent = sup(sub(id('t'), id('i')), num('1'));

/**
 * A display equation: one line of MathML with a label the stylesheet can hang a
 * number on. Kept as a string because it is injected through `v-html`.
 */
export const DISPLAY = (inner, label) =>
  `<math display="block" class="math-block"><mrow>${inner}</mrow></math>` +
  (label ? `<span class="math-label">${label}</span>` : '');

// ---------------------------------------------------------------------------
// STS — token efficiency axis
// ---------------------------------------------------------------------------

/** S(B) = L⁰ / L_best(B) */
export const STS_SPEEDUP = DISPLAY(
  row(
    id('S'), mo('('), id('B'), mo(')'), eq,
    frac(L0, sub(sub(id('L'), op('best')), mo('(') + id('B') + mo(')')))
  ),
  'Speedup'
);

/** VLR(B) = 1 − 1/S = 1 − L_best/L⁰ */
export const STS_VLR = DISPLAY(
  row(
    op('VLR'), mo('('), id('B'), mo(')'), eq,
    num('1'), mo(MINUS), frac(num('1'), row(id('S'), mo('('), id('B'), mo(')'))),
    eq,
    num('1'), mo(MINUS), frac(sub(id('L'), op('best')), L0)
  ),
  'Latency reduction'
);

/** g_k = log₂ S(B_k),  γ_k = ln(B_k/B_{k−1}) / ln(B_K/B_0) */
export const STS_AULS_TERMS = DISPLAY(
  row(
    sub(id('g'), id('k')), eq,
    sub(fn('log'), num('2')), id('S'), mo('('), sub(id('B'), id('k')), mo(')'),
    QUAD,
    sub(id('\u03B3'), id('k')), eq,
    frac(
      row(fn('ln'), mo('('), frac(sub(id('B'), id('k')), sub(id('B'), row(id('k'), mo(MINUS), num('1')))), mo(')')),
      row(fn('ln'), mo('('), frac(sub(id('B'), id('K')), sub(id('B'), num('0'))), mo(')'))
    )
  ),
  'Log-axis weight'
);

/** AULS = Σ_{k=1..K} γ_k (g_{k−1} + g_k)/2 */
export const STS_AULS = DISPLAY(
  row(
    op('AULS'), eq,
    sumBounds(row(id('k'), eq, num('1')), id('K')),
    sub(id('\u03B3'), id('k')),
    frac(
      row(sub(id('g'), row(id('k'), mo(MINUS), num('1'))), plus, sub(id('g'), id('k'))),
      num('2')
    )
  ),
  'Area under the curve'
);

/** STS_token = 100·[0.5·log₂S(B_max) + 0.5·AULS_token] */
export const STS_TOKEN = DISPLAY(
  row(
    sub(op('STS'), op('token')), eq,
    num('100'),
    mo('('),
    num('0.5'), sub(fn('log'), num('2')), id('S'),
    mo('('), sub(id('B'), op('max')), mo(')'),
    plus,
    num('0.5'), sub(op('AULS'), op('token')),
    mo(')')
  ),
  'Leaderboard score'
);

/** S_equiv = 2^(STS_token / 100) */
export const STS_SEQUIV = DISPLAY(
  row(
    sub(id('S'), op('equiv')), eq,
    sup(num('2'), row(sub(op('STS'), op('token')), mo('/'), num('100')))
  ),
  'Equivalent speedup'
);

// ---------------------------------------------------------------------------
// PAOS — profiling value axis
// ---------------------------------------------------------------------------

/** S_i^lat = t_i⁰ / t_i¹ */
export const PAOS_SLAT = DISPLAY(
  row(sup(sub(id('S'), id('i')), op('lat')), eq, frac(tiBase, tiAgent)),
  'Operator latency speedup'
);

/** T_step⁰ = T_other + Σ_i n_i·t_i⁰ */
export const PAOS_TSTEP = DISPLAY(
  row(
    sup(sub(id('T'), op('step')), num('0')), eq,
    sub(id('T'), op('other')), plus,
    sumUnder(id('i')),
    sub(id('n'), id('i')), THIN, tiBase,
    QUAD,
    sup(sub(id('T'), op('step')), num('1')), eq,
    sub(id('T'), op('other')), plus,
    sumUnder(id('i')),
    sub(id('n'), id('i')), THIN, tiAgent
  ),
  'Step latency (before / after)'
);

/** PTEI = T_step⁰ / T_step¹,  eTPS = G / T_step */
export const PAOS_PTEI = DISPLAY(
  row(
    op('PTEI'), eq,
    frac(sup(sub(id('T'), op('step')), num('0')), sup(sub(id('T'), op('step')), num('1'))),
    comma,
    QUAD,
    op('eTPS'), eq,
    frac(id('G'), sub(id('T'), op('step')))
  ),
  'Throughput gain / estimated throughput'
);

/** w_i = n_i·t_i⁰ / T_step⁰,  PKAI = exp(Σ w_i·ln S_i^lat) */
export const PAOS_PKAI = DISPLAY(
  row(
    sub(id('w'), id('i')), eq,
    frac(row(sub(id('n'), id('i')), THIN, tiBase), sup(sub(id('T'), op('step')), num('0'))),
    comma,
    QUAD,
    op('PKAI'), eq,
    fn('exp'),
    mo('('),
    sumUnder(id('i')),
    sub(id('w'), id('i')), fn('ln'), sup(sub(id('S'), id('i')), op('lat')),
    mo(')')
  ),
  'Weighted operator speedup index'
);

/** PAOS = sqrt(PKAI × PTEI) */
export const PAOS_SCORE = DISPLAY(
  row(
    op('PAOS'), eq,
    sqrt(row(op('PKAI'), mo('\u00D7'), op('PTEI')))
  ),
  'Legacy profiling score'
);
