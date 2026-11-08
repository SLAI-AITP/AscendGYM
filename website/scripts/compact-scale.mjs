import { DISPLAY_SCALE } from '../src/lib/displayScale.js';

// CSS zoom scales lengths and layout, while viewport units and media queries
// still refer to the unzoomed viewport. Compensate for those two exceptions so
// 100% browser zoom reproduces the original page at Chrome's 67% preset.
export default function compactScale() {
  return {
    postcssPlugin: 'ascendgym-compact-scale',
    Once(root) {
      root.walkDecls(declaration => {
        if (!declaration.value.includes('url(')) {
          declaration.value = declaration.value.replace(
            /(-?(?:\d*\.)?\d+)(vw|vh|vmin|vmax)\b/g,
            'calc($1$2 / var(--site-scale, 1))'
          );
        }
      });
      root.walkAtRules('media', rule => {
        if (/\bprint\b/.test(rule.params)) return;
        rule.params = rule.params.replace(
          /((?:min|max)-width\s*:\s*)(\d+(?:\.\d+)?)px/g,
          (_, prefix, width) => `${prefix}${Number(width) * DISPLAY_SCALE}px`
        );
      });
    }
  };
}
