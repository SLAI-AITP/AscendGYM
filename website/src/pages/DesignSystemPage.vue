<template>
  <header class="ds-nav">
    <router-link to="/">← Interactive prototype</router-link>
    <span>AscendGYM · Design system v1.0</span>
  </header>

  <main class="boards">
    <section class="board token-board" data-board="01">
      <div class="board-header">
        <div><span>FOUNDATION / 01</span><h1>Token swatch board</h1><p>Paper-white instrument panel. One compute accent, one cost accent, stable model colors.</p></div>
        <img :src="assetPath('assets/logo/lockup-primary.svg')" alt="AscendGYM" width="300">
      </div>

      <div class="board-section"><h2>Primary theme · light</h2><div class="swatches">
        <div class="swatch"><i style="--swatch:#FBFBF9"></i><b>Background</b><code>#FBFBF9</code></div>
        <div class="swatch"><i style="--swatch:#FFFFFF"></i><b>Surface</b><code>#FFFFFF</code></div>
        <div class="swatch"><i style="--swatch:#E7E7E2"></i><b>Hairline</b><code>#E7E7E2</code></div>
        <div class="swatch"><i style="--swatch:#14161A"></i><b>Text</b><code>#14161A</code></div>
        <div class="swatch"><i style="--swatch:#6B7280"></i><b>Muted</b><code>#6B7280</code></div>
        <div class="swatch"><i style="--swatch:#0E9B8E"></i><b>Compute</b><code>#0E9B8E</code></div>
        <div class="swatch"><i style="--swatch:#E08A1E"></i><b>Cost</b><code>#E08A1E</code></div>
      </div></div>

      <div class="board-section dark-token-panel"><h2>Parity theme · dark</h2><div class="swatches">
        <div class="swatch"><i style="--swatch:#0E1111"></i><b>Background</b><code>#0E1111</code></div>
        <div class="swatch"><i style="--swatch:#151918"></i><b>Surface</b><code>#151918</code></div>
        <div class="swatch"><i style="--swatch:#2B3330"></i><b>Hairline</b><code>#2B3330</code></div>
        <div class="swatch"><i style="--swatch:#F3F4F1"></i><b>Text</b><code>#F3F4F1</code></div>
        <div class="swatch"><i style="--swatch:#A4ACA8"></i><b>Muted</b><code>#A4ACA8</code></div>
        <div class="swatch"><i style="--swatch:#36C9B8"></i><b>Compute</b><code>#36C9B8</code></div>
        <div class="swatch"><i style="--swatch:#F1A94F"></i><b>Cost</b><code>#F1A94F</code></div>
      </div></div>

      <div class="board-grid two">
        <div class="board-section"><h2>Metric ramps</h2>
          <div class="ramp-row"><div><b>Sequential · ↑ is better</b><span>Correctness Gate</span></div><div class="ramp ramp-good-bad"></div><code>96.0% · 90.0% · 82.0%</code></div>
          <div class="ramp-row"><div><b>Sequential · ↓ is better</b><span>Cumulative Tokens — inverted</span></div><div class="ramp ramp-good-bad"></div><code>690k · 1,000k · 1,310k</code></div>
          <div class="ramp-row"><div><b>Diverging · baseline = 0</b><span>Speedup Gain</span></div><div class="ramp ramp-diverging"></div><code>−8% · 0% · +52%</code></div>
          <div class="ramp-row"><div><b>Neutral / dual-hue</b><span>Hardware utilization</span></div><div class="ramp ramp-util"></div><code>diagnostic · contextual</code></div>
        </div>
        <div class="board-section"><h2>Stable model colors</h2><div class="model-swatches">
          <span><i style="--m:#0E9B8E"><img :src="assetPath('assets/vendors/openai.svg')" alt=""></i>{{ modelDisplayName('GPT-5.6 Sol') }}</span>
          <span><i style="--m:#6D5B52"><img :src="assetPath('assets/vendors/anthropic.svg')" alt=""></i>{{ modelDisplayName('Claude Opus 4.8') }}</span>
          <span><i style="--m:#3F67D5"><img :src="assetPath('assets/vendors/google.svg')" alt=""></i>{{ modelDisplayName('Gemini 3.1 Pro') }}</span>
          <span><i style="--m:#8C58C6"><img :src="assetPath('assets/vendors/deepseek.svg')" alt=""></i>{{ modelDisplayName('DeepSeek V4-Pro') }}</span>
          <span><i style="--m:#D97706"><img :src="assetPath('assets/vendors/alibabacloud.svg')" alt=""></i>{{ modelDisplayName('Qwen3.7-Max') }}</span>
          <span><i style="--m:#3A7F96"><img :src="assetPath('assets/vendors/zai.svg')" alt=""></i>{{ modelDisplayName('GLM-5.2') }}</span>
          <span><i style="--m:#D14B70"><img :src="assetPath('assets/vendors/kimi.svg')" alt=""></i>{{ modelDisplayName('Kimi K2.7') }}</span>
        </div></div>
      </div>

      <div class="board-grid two">
        <div class="board-section"><h2>Type scale</h2><div class="type-specimens"><p class="type-display">AscendGYM</p><p class="type-h1">Performance, precisely ranked.</p><p class="type-body">Research-grade engineering evidence without dashboard chrome.</p><p class="type-data">87.4 · +41.5% · 58.3 µs · $14.90</p><code>DISPLAY 64/65 · H1 40/44 · BODY 15/25 · DATA 13/18</code></div></div>
        <div class="board-section"><h2>Spacing · radius · motion</h2><div class="spacing-scale"><i style="width:4px"></i><i style="width:8px"></i><i style="width:12px"></i><i style="width:16px"></i><i style="width:24px"></i><i style="width:32px"></i><i style="width:48px"></i><i style="width:64px"></i></div><code>4 · 8 · 12 · 16 · 24 · 32 · 48 · 64 px</code><div class="radius-row"><i>4</i><i>7</i><i>10</i><i>14</i><i class="pill">pill</i></div><code>Motion · 150 / 220 / 400 ms · ease-out</code></div>
      </div>
    </section>

    <section class="board primitives-board" data-board="02">
      <div class="board-header"><div><span>FOUNDATION / 02</span><h1>Component primitives</h1><p>Same semantics and affordances in light and dark. Numerals always use tabular mono.</p></div><span class="board-number">02</span></div>
      <div class="primitive-themes">
        <div class="primitive-panel light-sample">
          <div class="primitive-title"><b>LIGHT / PRIMARY</b><span>#FBFBF9</span></div>
          <div class="primitive-group"><label>Buttons</label><div class="primitive-row"><button class="p-primary">Run benchmark</button><button class="p-secondary">View methodology</button><button class="p-ghost">Ghost action</button></div></div>
          <div class="primitive-group"><label>Resources &amp; status</label><div class="primitive-row"><span class="p-resource"><i>aX</i> arXiv ↗</span><span class="p-resource"><i>HF</i> Hugging Face ↗</span><span class="p-status"><i></i> Rolling · updated</span></div></div>
          <div class="primitive-group"><label>Model identity</label><div class="primitive-row"><span class="p-model"><i style="--m:#0E9B8E"><img :src="assetPath('assets/vendors/openai.svg')" alt=""></i><b>{{ modelDisplayName('GPT-5.6 Sol') }}</b><em>CLOSED</em></span><span class="p-model"><i style="--m:#8C58C6"><img :src="assetPath('assets/vendors/deepseek.svg')" alt=""></i><b>{{ modelDisplayName('DeepSeek V4-Pro') }}</b><em>OPEN</em></span></div></div>
          <div class="primitive-group"><label>Controls</label><div class="primitive-row"><select><option>PerfScore</option></select><span class="p-segment"><b>Default</b><i>Tokens</i></span><span class="p-search">⌕ Search models · /</span></div></div>
          <div class="primitive-group"><label>Table row + numeric style</label><div class="p-table"><span>#1</span><span class="p-model"><i style="--m:#0E9B8E"><img :src="assetPath('assets/vendors/openai.svg')" alt=""></i><b>{{ modelDisplayName('GPT-5.6 Sol') }}</b></span><strong>87.4</strong><code>91.2%</code><code>+41.5%</code><code>58.3 µs</code></div></div>
          <div class="primitive-group"><label>Rubric ramp</label><div class="rubric-examples"><i style="--fill:#BEEBE4">96.0</i><i style="--fill:#DBEEE9">91.2</i><i style="--fill:#ECEBE4">88.7</i><i style="--fill:#F2DDC6">84.3</i><i style="--fill:#E8B77E">79.8</i></div></div>
          <div class="primitive-group two-up"><div><label>Tooltip</label><div class="p-tooltip"><b>{{ modelDisplayName('GPT-5.6 Sol') }}</b><span>PerfScore <code>87.4</code></span><span>Total API Cost <code>$14.90</code></span></div></div><div><label>Convergence sparkline</label><svg class="p-spark" viewBox="0 0 180 60"><path d="M3 53L28 46L54 34L80 26L106 20L132 15L177 10"/><circle cx="177" cy="10" r="4"/></svg></div></div>
        </div>
        <div class="primitive-panel dark-sample">
          <div class="primitive-title"><b>DARK / PARITY</b><span>#0E1111</span></div>
          <div class="primitive-group"><label>Buttons</label><div class="primitive-row"><button class="p-primary">Run benchmark</button><button class="p-secondary">View methodology</button><button class="p-ghost">Ghost action</button></div></div>
          <div class="primitive-group"><label>Resources &amp; status</label><div class="primitive-row"><span class="p-resource"><i>aX</i> arXiv ↗</span><span class="p-resource"><i>HF</i> Hugging Face ↗</span><span class="p-status"><i></i> Rolling · updated</span></div></div>
          <div class="primitive-group"><label>Model identity</label><div class="primitive-row"><span class="p-model"><i style="--m:#36C9B8"><img :src="assetPath('assets/vendors/openai.svg')" alt=""></i><b>{{ modelDisplayName('GPT-5.6 Sol') }}</b><em>CLOSED</em></span><span class="p-model"><i style="--m:#A477D2"><img :src="assetPath('assets/vendors/deepseek.svg')" alt=""></i><b>{{ modelDisplayName('DeepSeek V4-Pro') }}</b><em>OPEN</em></span></div></div>
          <div class="primitive-group"><label>Controls</label><div class="primitive-row"><select><option>PerfScore</option></select><span class="p-segment"><b>Default</b><i>Tokens</i></span><span class="p-search">⌕ Search models · /</span></div></div>
          <div class="primitive-group"><label>Table row + numeric style</label><div class="p-table"><span>#1</span><span class="p-model"><i style="--m:#36C9B8"><img :src="assetPath('assets/vendors/openai.svg')" alt=""></i><b>{{ modelDisplayName('GPT-5.6 Sol') }}</b></span><strong>87.4</strong><code>91.2%</code><code>+41.5%</code><code>58.3 µs</code></div></div>
          <div class="primitive-group"><label>Rubric ramp</label><div class="rubric-examples"><i style="--fill:#BEEBE4">96.0</i><i style="--fill:#DBEEE9">91.2</i><i style="--fill:#ECEBE4">88.7</i><i style="--fill:#F2DDC6">84.3</i><i style="--fill:#E8B77E">79.8</i></div></div>
          <div class="primitive-group two-up"><div><label>Tooltip</label><div class="p-tooltip"><b>{{ modelDisplayName('GPT-5.6 Sol') }}</b><span>PerfScore <code>87.4</code></span><span>Total API Cost <code>$14.90</code></span></div></div><div><label>Convergence sparkline</label><svg class="p-spark" viewBox="0 0 180 60"><path d="M3 53L28 46L54 34L80 26L106 20L132 15L177 10"/><circle cx="177" cy="10" r="4"/></svg></div></div>
        </div>
      </div>
    </section>

    <section class="board identity-board" data-board="03">
      <div class="board-header"><div><span>IDENTITY / 03</span><h1>Ascend + kernel</h1><p>A single compute-core glyph held inside a precise orbital ring. Legible at 24px and in one color.</p></div><span class="board-number">03</span></div>
      <div class="logo-hero"><img :src="assetPath('assets/logo/mark-primary.svg')" alt="Primary AscendGYM mark"><div><img :src="assetPath('assets/logo/lockup-primary.svg')" alt="AscendGYM primary lockup"><p>Primary accent-on-light lockup</p></div></div>
      <div class="logo-variants">
        <figure><div><img :src="assetPath('assets/logo/mark-primary.svg')" alt="Primary logo"></div><figcaption><b>Primary</b><code>accent on paper</code></figcaption></figure>
        <figure><div><img :src="assetPath('assets/logo/mark-monochrome.svg')" alt="Monochrome logo"></div><figcaption><b>Monochrome</b><code>single-ink safe</code></figcaption></figure>
        <figure class="dark-logo-figure"><div><img :src="assetPath('assets/logo/mark-dark.svg')" alt="Dark theme logo"></div><figcaption><b>Dark parity</b><code>accent on #0E1111</code></figcaption></figure>
        <figure><div class="favicon-preview"><img :src="assetPath('assets/logo/favicon.svg')" alt="Favicon"><img :src="assetPath('assets/logo/favicon.svg')" alt="Favicon at 32 pixels"></div><figcaption><b>Favicon</b><code>64px · 32px</code></figcaption></figure>
      </div>
      <div class="logo-clearspace"><div class="clearspace-mark"><span>X</span><img :src="assetPath('assets/logo/mark-primary.svg')" alt="Logo clearspace diagram"><i></i></div><div><h2>Clearspace &amp; minimums</h2><p>Minimum clearspace equals one inner cell width. Minimum UI size is 24px; favicon art is optically corrected at 16–32px.</p><code>MARK 24px MIN · LOCKUP 132px MIN</code></div></div>
    </section>

    <section class="board states-board" data-board="04">
      <div class="board-header"><div><span>INTERACTION / 04</span><h1>State completeness matrix</h1><p>Default, hover/active, loading, empty, mobile, dark and light are accounted for across the live sections A–F, I–J; G/H are parked with their sections.</p></div><span class="board-number">04</span></div>
      <table class="state-matrix"><thead><tr><th>Component</th><th>Default</th><th>Hover / active</th><th>Loading</th><th>Empty</th><th>Mobile</th><th>Dark</th><th>Light</th></tr></thead><tbody>
        <tr><td>A Hero</td><td>Done</td><td>Done</td><td>Done</td><td>Done</td><td>Done</td><td>Done</td><td>Done</td></tr>
        <tr><td>B1 Ranked field</td><td>Done</td><td>Linked</td><td>Done</td><td>Done</td><td>Scroll</td><td>Done</td><td>Done</td></tr>
        <tr><td>B2 Table</td><td>Done</td><td>Sort + drawer</td><td>Done</td><td>Done</td><td>Cards</td><td>Done</td><td>Done</td></tr>
        <tr><td>C Scatter</td><td>Done</td><td>Tooltip + linked</td><td>Done</td><td>Done</td><td>Log axis</td><td>Done</td><td>Done</td></tr>
        <tr><td>D Rubric</td><td>Done</td><td>Tooltip + linked</td><td>Done</td><td>Done</td><td>Scroll</td><td>Done</td><td>Done</td></tr>
        <tr><td>E Heatmap</td><td>Done</td><td>Row + cell</td><td>Done</td><td>Done</td><td>Scroll</td><td>Done</td><td>Done</td></tr>
        <tr><td>F Gain</td><td>Done</td><td>Tooltip</td><td>Done</td><td>Done</td><td>Stacked</td><td>Done</td><td>Done</td></tr>
        <tr><td>G Duration</td><td>Parked</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
        <tr><td>H Utilization</td><td>Parked</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
        <tr><td>H Roofline</td><td>Omitted*</td><td>—</td><td>—</td><td>Reason</td><td>—</td><td>—</td><td>—</td></tr>
        <tr><td>I Detail drawer</td><td>Done</td><td>Open / close</td><td>Shell</td><td>Shell</td><td>Full-screen</td><td>Done</td><td>Done</td></tr>
        <tr><td>J Method + footer</td><td>Done</td><td>Copy</td><td>Done</td><td>Done</td><td>Stacked</td><td>Done</td><td>Done</td></tr>
      </tbody></table>
      <p class="matrix-note">* Roofline is deliberately omitted because the supplied schema has no FLOP or byte-count fields. Drawing it would fabricate arithmetic intensity.</p>
      <p class="matrix-note">G / H are parked: their sections were removed from <code>lib/sections.js</code>, so the components exist but are not mounted. Restore the registry entries and the <code>LeaderboardPage</code> imports to bring them back.</p>
      <div class="state-samples">
        <article><span>DEFAULT</span><div class="sample-chart"><i style="height:58%"></i><i style="height:84%"></i><i style="height:66%"></i><i style="height:48%"></i></div></article>
        <article><span>HOVER / LINKED</span><div class="sample-chart dim"><i style="height:58%"></i><i class="active" style="height:84%"></i><i style="height:66%"></i><i style="height:48%"></i></div></article>
        <article><span>LOADING</span><div class="sample-loading"><i></i><i></i><i></i></div></article>
        <article><span>EMPTY</span><div class="sample-empty"><i>00</i><b>No data</b></div></article>
      </div>
    </section>

    <section class="board icons-board" data-board="05">
      <div class="board-header">
        <div>
          <span>IDENTITY / 05</span>
          <h1>Vendor marks &amp; icon elements</h1>
          <p>
            Model icons use <code>lib/vendors.js</code> registry: a consistent ring with the vendor mark.
            Vendors without an available mark use their own initials.
          </p>
        </div>
        <span class="board-number">05</span>
      </div>

      <div class="board-section">
        <h2>Vendor marks · {{ markedVendors.length }} of {{ allVendors.length }} vendors ship a mark</h2>
        <div class="mark-grid">
          <figure v-for="vendor in markedVendors" :key="vendor.id">
            <VendorChip :model="{ id: vendor.label, vendor: vendor.label, vendor_id: vendor.id, color: vendor.color }" size="md" />
            <figcaption>
              <b>{{ vendor.label }}</b>
              <code>{{ vendor.asset }}</code>
            </figcaption>
          </figure>
          <!-- Monogram fallback: no brand mark, same ring. -->
          <figure class="is-fallback">
            <VendorChip :model="{ id: 'Unknown-7B', vendor: null }" size="md" />
            <figcaption>
              <b>Other / unknown</b>
              <code>monogram · never another vendor's logo</code>
            </figcaption>
          </figure>
        </div>
      </div>

      <div class="board-section">
        <h2>Chip sizes</h2>
        <div class="mark-sizes">
          <span><VendorChip :model="sampleModel" size="sm" /><code>sm · 28px · table cell</code></span>
          <span><VendorChip :model="sampleModel" size="md" /><code>md · 36px · panel</code></span>
          <span><VendorChip :model="sampleModel" size="lg" /><code>lg · 48px · drawer header</code></span>
          <span><VendorChip :model="{ id: 'Unknown-7B', vendor: null }" size="lg" /><code>lg · monogram fallback</code></span>
        </div>
      </div>

      <div class="board-section">
        <h2>Preset SOTA roster · {{ presetModelCount }} models · {{ presetVendorCount }} vendors</h2>
        <div class="mark-roster">
          <div v-for="group in presetGroups" :key="group.vendor" class="mark-roster-row">
            <span class="mark-roster-vendor"><VendorChip :model="{ id: group.vendor, vendor: group.vendor }" size="sm" />{{ group.vendor }}</span>
            <div class="mark-roster-models">
              <span v-for="model in group.models" :key="model.id" class="mark-roster-model">
                <VendorChip :model="model" size="sm" />
                <b>{{ modelDisplayName(model) }}</b>
                <em>{{ model.openness }}</em>
              </span>
            </div>
          </div>
        </div>
      </div>

      <div class="board-section">
        <h2>Icon elements</h2>
        <div class="icon-element-grid">
          <div>
            <label>Agent badge · domain / general</label>
            <div class="primitive-row"><span class="agent-badge" data-type="domain">WM</span><span class="agent-badge" data-type="general">CC</span><span class="agent-badge" data-size="lg" data-type="domain">OP</span></div>
          </div>
          <div>
            <label>Flag vocabulary</label>
            <div class="primitive-row"><span class="flag-dot" data-level="red">🟥</span><span class="flag-dot" data-level="amber">🟧</span><span class="flag-dot" data-level="blue">🟦</span><span class="flag-dot" data-level="gray">⬜</span></div>
          </div>
          <div>
            <label>Rank + open/closed tag</label>
            <div class="primitive-row"><span class="rank-badge top">1</span><span class="rank-badge">2</span><span class="weight-tag">open</span><span class="weight-tag">closed</span></div>
          </div>
          <div>
            <label>Resource + status</label>
            <div class="primitive-row"><span class="resource-chip is-placeholder"><span class="resource-icon hf">HF</span> Hugging Face</span><span class="status-pill"><span class="pulse-dot"></span> Measured</span></div>
          </div>
        </div>
      </div>
    </section>
  </main>
</template>

<script setup>
import { assetPath } from '../lib/markup';
import { modelDisplayName } from '../lib/displayNames';
import VendorChip from '../components/VendorChip.vue';
import { VENDORS, VENDORS_WITH_MARK, SOTA_MODELS, sotaModelsByVendor } from '../lib/vendors';

const allVendors = Object.values(VENDORS);
const markedVendors = VENDORS_WITH_MARK;
const presetGroups = sotaModelsByVendor();
const presetModelCount = SOTA_MODELS.length;
const presetVendorCount = presetGroups.length;
// One representative mark for the size ladder.
const sampleModel = SOTA_MODELS.find(model => model.id === 'Qwen3.5-122B') ?? SOTA_MODELS[0];
</script>
