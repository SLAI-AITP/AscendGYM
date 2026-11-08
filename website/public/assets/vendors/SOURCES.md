# Vendor mark sources

- OpenAI, Anthropic, Google, DeepSeek and Alibaba Cloud: Simple Icons SVG collection, retrieved 2026-07-18. Simple Icons is distributed under CC0; brand marks may remain subject to trademark rules.
- KIMI: official KIMI Brand Guidelines (`moonshotai.github.io/Branding-Guide/`), K-only light-background SVG, retrieved 2026-07-18.
- Z.AI: official-source mark mirrored by Wikimedia Commons (`File:Z.ai (company logo).svg`); simplified to the visible geometric paths for compact UI use.
- MiniMax, Mistral AI, X (xAI / Grok), Meta, QWen, ByteDance and Baidu: Simple Icons SVG collection, retrieved 2026-09-23 (CC0).

These marks are used only to identify benchmark vendors in this illustrative prototype. They do not imply endorsement.

## Rendering rules

- Marks render inside a circular chip (`.vendor-chip`) with a colour ring; the
  registry is `src/lib/vendors.js`.
- A vendor with no mark here falls back to a **monogram in the same ring**. It
  never falls back to another vendor's logo — misattributing an entry is a
  data-fidelity bug, not a cosmetic shortcut.
