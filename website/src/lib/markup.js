// HTML-string builders for identity fragments embedded inside chart SVG
// (v-html renders these directly, so they keep the manual escaping of the
// original prototype). Vue templates use the EntryLabel / VendorChip
// components instead.
import { escapeHTML, attributeEscape, formatTokensK, formatPct, formatSts, formatSpeedupDetail } from './format';
import { agentInitials, flagMeta } from './meta';
import { vendorVisual } from './vendors';
import { harnessDisplayName, modelDisplayName } from './displayNames';

// Public-dir assets are addressed relative to the deployment base so the
// build works from any sub-path.
export function assetPath(path) {
  const base = import.meta.env.BASE_URL || '/';
  return `${base.replace(/\/$/, '')}/${path}`.replace(/^\.\//, '');
}

/** Chip markup shared by the SVG-string builders and the Vue component. */
export function vendorChipHTML(model, size = 'sm') {
  const visual = vendorVisual(model);
  const inner = visual.asset
    ? `<img src="${assetPath(`assets/vendors/${visual.asset}`)}" alt="" loading="eager">`
    : `<span class="vendor-glyph" aria-hidden="true">${escapeHTML(visual.glyph)}</span>`;
  return `<span class="vendor-chip" data-size="${size}" data-vendor="${visual.vendor}" style="--model-color:${visual.color}">${inner}</span>`;
}

/**
 * Brand mark resolved to a URL. Returns `null` when the vendor ships no mark,
 * so callers fall back to a monogram instead of borrowing another vendor's
 * logo — a wrong logo is a data-fidelity bug.
 */
export function vendorLogoPath(entry) {
  const visual = vendorVisual(entry?.engine_model);
  return visual.asset ? assetPath(`assets/vendors/${visual.asset}`) : null;
}

/** `<image>`-or-monogram mark centred on (x, y) inside an SVG chart. */
export function vendorMarkSVG(entry, x, y, size = 22) {
  const model = entry?.engine_model;
  const visual = vendorVisual(model);
  if (visual.asset) {
    const half = size / 2;
    return `<image href="${assetPath(`assets/vendors/${visual.asset}`)}" x="${x - half}" y="${y - half}" width="${size}" height="${size}" preserveAspectRatio="xMidYMid meet"/>`;
  }
  return `<g class="chart-vendor-glyph"><circle cx="${x}" cy="${y}" r="${size / 2}" fill="${visual.color}"/><text x="${x}" y="${y + 3.5}" text-anchor="middle" font-size="${Math.round(size * 0.5)}" font-family="var(--font-mono)" fill="#fff">${escapeHTML(visual.glyph)}</text></g>`;
}

export function agentBadge(entry, size = 'sm') {
  const agent = entry.agent;
  const tooltip = `<strong>${escapeHTML(harnessDisplayName(agent))}</strong><br>Agent type: ${agent.type === 'domain' ? 'Domain-specific' : 'General-purpose'} · Strategy ${escapeHTML(agent.strategy)} · ${escapeHTML(agent.version)}`;
  return `<span class="agent-badge" data-size="${size}" data-type="${agent.type}" data-tooltip-html="${attributeEscape(tooltip)}">${agentInitials(agent.id)}</span>`;
}

export function entryLabel(entry, size = 'sm') {
  const model = entry.engine_model;
  return `<span class="model-label entry-label">${agentBadge(entry, size)}${vendorChipHTML(model, size)}<span class="model-text"><strong>${escapeHTML(modelDisplayName(model))}<span class="weight-tag">${escapeHTML(model.openness)}</span></strong><small>${escapeHTML(harnessDisplayName(entry.agent))} · ${escapeHTML(model.vendor)}</small></span></span>`;
}

export function flagsHTML(flags) {
  return (flags || []).map(flag => {
    const meta = flagMeta[flag];
    if (!meta) return '';
    return `<span class="flag-dot" data-level="${meta.level}" data-tooltip-html="${attributeEscape(`<strong>${meta.icon} ${escapeHTML(meta.label)}</strong><br>${escapeHTML(meta.rule)}`)}" aria-label="${escapeHTML(meta.label)}">${meta.icon}</span>`;
  }).join('');
}

export function tooltipEntry(entry) {
  const overall = entry.overall;
  return `<div class="tooltip-title">${entryLabel(entry, 'sm')}</div><div class="tooltip-grid"><span>STS Token</span><span class="tooltip-value">${formatSts(overall.sts_token)}</span><span>S_equiv</span><span class="tooltip-value">${formatSpeedupDetail(overall.s_equiv)}</span><span>EBGS</span><span class="tooltip-value">${formatSpeedupDetail(overall.ebgs)}</span><span>VLR</span><span class="tooltip-value">${formatPct(overall.vlr_mean)}</span><span>Gate</span><span class="tooltip-value">${formatPct(overall.gate_rate)}</span><span>PAOS</span><span class="tooltip-value">${formatSpeedupDetail(overall.paos?.paos)}</span><span>Cumulative Tokens</span><span class="tooltip-value">${formatTokensK(overall.tokens_total_k)}</span></div>${flagsHTML(overall.flags)}`;
}
