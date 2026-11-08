<template>
  <footer class="site-footer" id="source">
    <div class="page-shell">
      <div class="footer-lockup">
        <img :src="assetPath('assets/logo/mark-monochrome.svg')" alt="" width="54" height="54">
        <strong>AscendGYM</strong>
      </div>
      <div class="footer-grid">
        <div>
          <h2>Cite this benchmark</h2>
          <pre id="bibtex"><code>@misc{ascendgym_2026,
  title  = {AscendGYM},
  year   = {2026},
  note   = {AscendC operator optimization benchmark}
}</code></pre>
          <button class="secondary-button" type="button" @click="copyText(bibtex, 'BibTeX copied')">Copy BibTeX</button>
        </div>
        <div class="footer-links">
          <h2>Resources</h2>
          <a :href="assetPath('AscendGYM_TechReport.pdf')" target="_blank" rel="noopener">Technical report ↗</a>
          <a href="https://github.com/SLAI-AITP/AscendGYM/tree/main/datasets" id="dataset" target="_blank" rel="noopener">Dataset ↗</a>
          <a href="https://github.com/SLAI-AITP/AscendGYM" target="_blank" rel="noopener">Repository ↗</a>
        </div>
        <div class="footer-links">
          <h2>Project</h2>
          <a href="https://github.com/SLAI-AITP/AscendGYM#license" target="_blank" rel="noopener">Licensing &amp; third-party notices ↗</a>
          <a href="https://github.com/SLAI-AITP/AscendGYM" target="_blank" rel="noopener">Project repository ↗</a>
        </div>
      </div>
      <div class="footer-bottom">
        <span>© 2026 AscendGYM</span><span>Designed for technical truth, not marketing gloss.</span><a href="#top" @click.prevent="scrollToId('top')">Back to top ↑</a>
      </div>
    </div>
  </footer>
</template>

<script setup>
import { scrollToId, copyText } from '../store/ui';
import { assetPath } from '../lib/markup';

const bibtex = `@misc{ascendgym_2026,
  title  = {AscendGYM},
  year   = {2026},
  note   = {AscendC operator optimization benchmark}
}`;
</script>
