// ============================================================================
// Vendor identity registry — the single source of truth for *icon elements*.
//
// Two rows of the UI read from here:
//
//   1. The circular model chip (`VendorChip.vue`, `lib/markup.js` and the SVG
//      charts) needs a brand mark for every engine model it renders.
//   2. The Imports page needs a preset roster so a reviewer can declare prices
//      and add runs for models they did not measure themselves.
//
// Deliberately dependency-free plain ESM: `scripts/build-operator-bundle.mjs`
// imports this file too, so it must not touch `import.meta.env`, Vue or the
// DOM. Asset *paths* are resolved in the view layer (`lib/markup.js`).
// ============================================================================

/**
 * Brand marks shipped in `public/assets/vendors/`.
 *
 * `color` is a *presentation accent*, not a measurement: it paints the ring
 * when a model carries no explicit colour of its own (imported entries usually
 * do not). A model's own `color` always wins — the charts use it to link an
 * entry across sections, so it must stay stable.
 */
export const VENDORS = {
  anthropic: {
    id: 'anthropic', label: 'Anthropic', asset: 'anthropic.svg', glyph: 'CL', color: '#D97757',
    aliases: ['anthropic', 'claude'],
    // Conservative: only ids that genuinely start with the family token. A
    // wrong logo is worse than no logo.
    family: /^(claude|opus|sonnet|haiku)/i
  },
  openai: {
    id: 'openai', label: 'OpenAI', asset: 'openai.svg', glyph: 'GPT', color: '#10A37F',
    aliases: ['openai', 'open ai', 'gpt', 'chatgpt'],
    family: /^(gpt|o[1-9]|chatgpt|codex)/i
  },
  google: {
    id: 'google', label: 'Google', asset: 'google.svg', glyph: 'GEM', color: '#4285F4',
    aliases: ['google', 'google deepmind', 'deepmind', 'gemini'],
    family: /^(gemini|gemma|palm)/i
  },
  deepseek: {
    id: 'deepseek', label: 'DeepSeek', asset: 'deepseek.svg', glyph: 'DS', color: '#4D6BFE',
    aliases: ['deepseek'],
    family: /^deepseek/i
  },
  alibaba: {
    id: 'alibaba', label: 'Alibaba', asset: 'alibabacloud.svg', glyph: 'QW', color: '#FF6A00',
    aliases: ['alibaba', 'alibaba cloud', 'aliyun', 'qwen', 'tongyi'],
    family: /^(qwen|qwq|qvq|tongyi)/i
  },
  zhipu: {
    id: 'zhipu', label: 'Zhipu', asset: 'zai.svg', glyph: 'GLM', color: '#3A7F96',
    aliases: ['zhipu', 'zhipu ai', 'z.ai', 'zai', 'glm'],
    family: /^(glm|chatglm|cogvlm)/i
  },
  moonshot: {
    id: 'moonshot', label: 'Moonshot', asset: 'kimi.svg', glyph: 'KM', color: '#D14B70',
    aliases: ['moonshot', 'moonshot ai', 'kimi'],
    family: /^(kimi|moonshot)/i
  },
  minimax: {
    id: 'minimax', label: 'MiniMax', asset: 'minimax.svg', glyph: 'MM', color: '#EE3D5A',
    aliases: ['minimax', 'mini max'],
    family: /^(minimax|abab|hailuo)/i
  },
  xai: {
    id: 'xai', label: 'xAI', asset: 'xai.svg', glyph: 'X', color: '#14161A',
    aliases: ['xai', 'x.ai', 'x ai', 'grok'],
    family: /^grok/i
  },
  mistral: {
    id: 'mistral', label: 'Mistral AI', asset: 'mistral.svg', glyph: 'MI', color: '#FA520F',
    aliases: ['mistral', 'mistral ai'],
    family: /^(mistral|mixtral|magistral|codestral|devstral|pixtral)/i
  },
  meta: {
    id: 'meta', label: 'Meta', asset: 'meta.svg', glyph: 'LL', color: '#0866FF',
    aliases: ['meta', 'meta ai', 'meta platforms', 'llama'],
    family: /^(llama|meta)/i
  },
  bytedance: {
    id: 'bytedance', label: 'ByteDance', asset: 'bytedance.svg', glyph: 'DB', color: '#325AB4',
    aliases: ['bytedance', 'byte dance', 'doubao', 'seed', 'skylark'],
    family: /^(doubao|seed|skylark)/i
  },
  baidu: {
    id: 'baidu', label: 'Baidu', asset: 'baidu.svg', glyph: 'EN', color: '#2932E1',
    aliases: ['baidu', 'ernie', 'wenxin'],
    family: /^(ernie|wenxin)/i
  },
  // Fallback identity: no mark to claim, so the chip renders a monogram in the
  // neutral ink. Never borrow another vendor's logo — that is a data-fidelity
  // bug, not a cosmetic shortcut.
  generic: {
    id: 'generic', label: 'Other', asset: null, glyph: '', color: '#6B7280',
    aliases: [], family: null
  }
};

export const VENDOR_LIST = Object.values(VENDORS);

/** Vendors with a shipped brand mark, in registry order (for the icon board). */
export const VENDORS_WITH_MARK = VENDOR_LIST.filter(vendor => vendor.asset);

const normalize = value => String(value ?? '').trim().toLowerCase().replace(/[\s._-]/g, '');

/** vendor id / label / alias → vendor record. */
const BY_ALIAS = new Map();
for (const vendor of VENDOR_LIST) {
  BY_ALIAS.set(normalize(vendor.id), vendor);
  BY_ALIAS.set(normalize(vendor.label), vendor);
  for (const alias of vendor.aliases) BY_ALIAS.set(normalize(alias), vendor);
}

/**
 * Resolve the vendor behind an engine model.
 *
 * Order matters: an explicit `vendor_id` beats a declared `vendor` label,
 * which beats a guess from the model id. Returns `VENDORS.generic` (never
 * `null`) so callers can always render something.
 */
export function vendorFor(model) {
  if (!model) return VENDORS.generic;
  if (typeof model === 'string') return vendorFromId(model);

  if (model.vendor_id && VENDORS[model.vendor_id]) return VENDORS[model.vendor_id];
  const declared = BY_ALIAS.get(normalize(model.vendor));
  if (declared) return declared;

  const guessed = vendorFromId(model.id);
  if (guessed !== VENDORS.generic) return guessed;
  // Last resort: a one-letter `vendor_mark` can still tie to a known vendor.
  const byMark = model.vendor_mark
    ? VENDOR_LIST.find(vendor => vendor !== VENDORS.generic && vendor.glyph.startsWith(String(model.vendor_mark).toUpperCase()))
    : null;
  return byMark ?? VENDORS.generic;
}

/** Guess a vendor from a model id using each vendor's leading family token. */
export function vendorFromId(modelId) {
  const id = String(modelId ?? '').trim();
  if (!id) return VENDORS.generic;
  for (const vendor of VENDOR_LIST) {
    if (vendor.family?.test(id)) return vendor;
  }
  return VENDORS.generic;
}

/**
 * Everything the view layer needs to draw one model chip.
 * `asset` is a bare filename — `lib/markup.js` turns it into a URL.
 */
export function vendorVisual(model, { color = null } = {}) {
  const vendor = vendorFor(model);
  const modelColor = color ?? (typeof model === 'object' ? model?.color : null);
  return {
    vendor: vendor.id,
    label: vendor.label,
    asset: vendor.asset,
    glyph: vendor.glyph || initialsOf(typeof model === 'object' ? model?.id : model),
    color: modelColor || vendor.color
  };
}

function initialsOf(modelId) {
  const parts = String(modelId ?? '').split(/[\s._-]+/).filter(Boolean);
  if (parts.length >= 2) return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
  return String(modelId ?? '?').slice(0, 2).toUpperCase();
}

// ---------------------------------------------------------------------------
// Preset roster of mainstream SOTA engine models.
//
// `engine_model.id` is the join key everywhere (cost, colour, chip). Presets
// give an imported run a stable colour + vendor mark without the reviewer
// having to spell them out, and give the Imports page something to click.
// ---------------------------------------------------------------------------
export const SOTA_MODELS = [
  // --- Anthropic ---------------------------------------------------------
  { id: 'Claude-Opus-4.8', vendor: 'Anthropic', openness: 'closed', color: '#6D5B52' },
  { id: 'Claude-Sonnet-4.8', vendor: 'Anthropic', openness: 'closed', color: '#8A6F63' },
  { id: 'Claude-Haiku-4.5', vendor: 'Anthropic', openness: 'closed', color: '#A98D7E' },
  // --- OpenAI ------------------------------------------------------------
  { id: 'GPT-5.6-Sol', vendor: 'OpenAI', openness: 'closed', color: '#0E9B8E' },
  { id: 'GPT-5.6', vendor: 'OpenAI', openness: 'closed', color: '#10A37F' },
  { id: 'GPT-5.2-mini', vendor: 'OpenAI', openness: 'closed', color: '#3FBF9F' },
  // --- Google ------------------------------------------------------------
  { id: 'Gemini-3.1-Pro', vendor: 'Google', openness: 'closed', color: '#3F67D5' },
  { id: 'Gemini-3.1-Flash', vendor: 'Google', openness: 'closed', color: '#5B8DEF' },
  // --- DeepSeek ----------------------------------------------------------
  { id: 'DeepSeek-V4-Pro', vendor: 'DeepSeek', openness: 'open', color: '#8C58C6' },
  { id: 'DeepSeek-V3.2', vendor: 'DeepSeek', openness: 'open', color: '#4D6BFE' },
  { id: 'DeepSeek-R2', vendor: 'DeepSeek', openness: 'open', color: '#6E6BE8' },
  // --- Alibaba / Qwen ----------------------------------------------------
  { id: 'Qwen3.5-122B', vendor: 'Alibaba', openness: 'open', color: '#2E86DE' },
  { id: 'Qwen3.7-Max', vendor: 'Alibaba', openness: 'open', color: '#D97706' },
  { id: 'Qwen3.5-Coder', vendor: 'Alibaba', openness: 'open', color: '#4C9EE8' },
  // --- Zhipu / GLM -------------------------------------------------------
  { id: 'GLM-5.2', vendor: 'Zhipu', openness: 'open', color: '#3A7F96' },
  { id: 'GLM-5.2-Air', vendor: 'Zhipu', openness: 'open', color: '#5AA0B6' },
  // --- Moonshot / Kimi ---------------------------------------------------
  { id: 'Kimi-K2.7', vendor: 'Moonshot', openness: 'open', color: '#D14B70' },
  { id: 'Kimi-K3', vendor: 'Moonshot', openness: 'open', color: '#B33E60' },
  // --- MiniMax -----------------------------------------------------------
  { id: 'MiniMax-M3', vendor: 'MiniMax', openness: 'open', color: '#EE3D5A' },
  { id: 'MiniMax-M2.1', vendor: 'MiniMax', openness: 'open', color: '#F0667C' },
  // --- xAI ---------------------------------------------------------------
  { id: 'Grok-5', vendor: 'xAI', openness: 'closed', color: '#14161A' },
  { id: 'Grok-5-mini', vendor: 'xAI', openness: 'closed', color: '#4A4F57' },
  // --- Mistral -----------------------------------------------------------
  { id: 'Mistral-Large-3', vendor: 'Mistral AI', openness: 'open', color: '#FA520F' },
  { id: 'Magistral-2', vendor: 'Mistral AI', openness: 'open', color: '#F58442' },
  // --- Meta --------------------------------------------------------------
  { id: 'Llama-4.2-Maverick', vendor: 'Meta', openness: 'open', color: '#0866FF' },
  { id: 'Llama-4.2-Scout', vendor: 'Meta', openness: 'open', color: '#4D8DFF' },
  // --- ByteDance / Baidu -------------------------------------------------
  { id: 'Doubao-2.5-Pro', vendor: 'ByteDance', openness: 'closed', color: '#325AB4' },
  { id: 'ERNIE-6.0-Turbo', vendor: 'Baidu', openness: 'closed', color: '#2932E1' }
];

const PRESET_BY_ID = new Map(SOTA_MODELS.map(model => [normalize(model.id), model]));

/**
 * Look up a preset by engine_model id.
 *
 * Exact (case/separator-insensitive) match first. A family prefix in the id
 * still yields the vendor's mark and accent colour, but the *openness* is left
 * null rather than guessed — we do not know it for a model we have not listed.
 */
export function modelPreset(modelId) {
  const id = String(modelId ?? '').trim();
  if (!id) return null;

  const exact = PRESET_BY_ID.get(normalize(id));
  if (exact) return { ...exact, preset: true };

  const family = vendorFromId(id);
  if (family === VENDORS.generic) return null;
  return { id, vendor: family.label, openness: null, vendor_mark: family.glyph, color: family.color, preset: false };
}

/** Preset model ids grouped by vendor label — used by the icon/library board. */
export function sotaModelsByVendor() {
  const groups = new Map();
  for (const model of SOTA_MODELS) {
    if (!groups.has(model.vendor)) groups.set(model.vendor, []);
    groups.get(model.vendor).push(model);
  }
  return [...groups.entries()].map(([vendor, models]) => ({
    vendor,
    vendorId: vendorFor({ vendor }).id,
    models
  }));
}
