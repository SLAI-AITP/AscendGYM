<template>
  <section class="section page-shell" id="methodology" aria-labelledby="method-title">
    <div v-if="showHeading" class="section-heading"><div><p class="section-index">{{ section.index }} / {{ section.kicker }}</p><h2 id="method-title">{{ section.title }}</h2></div></div>
    <div class="method-grid">
      <section class="metric-category-panel" aria-labelledby="metric-categories-title">
        <h3 id="metric-categories-title">Categories</h3>
        <ul class="metric-categories">
          <li v-for="(category, index) in categories" :key="category.title"><span class="metric-category-number">{{ String(index + 1).padStart(2, '0') }}</span><h4>{{ category.title }}</h4><p>{{ category.goal }}</p></li>
        </ul>
      </section>
      <div class="method-copy metric-definition-panels">
        <section class="method-group metric-definition-panel" aria-labelledby="joint-metrics-title">
          <h3 id="joint-metrics-title" class="metric-panel-title">Joint Performance–Budget Metrics</h3>
          <section id="sts-metric" aria-label="Speedup-Token Score (STS)">
          <h4>Speedup–Token Score (STS)</h4>
          <p>Speedup–Token Score (STS) rewards the best correctness-validated speedup within a fixed token budget and penalizes improvements obtained later in that budget.</p>
          <MathText :tex="reportMath.sts" display />
          <MathText :tex="reportMath.delta" display />
          <dl class="var-list">
            <div><dt><MathText tex="B_{\max}" /></dt><dd>Per-task token budget; the current leaderboard uses 50 million tokens.</dd></div>
            <div><dt><MathText tex="n" /></dt><dd>Number of candidate evaluations completed within the token budget.</dd></div>
            <div><dt><MathText tex="B_k" /></dt><dd>Cumulative token consumption when evaluation <MathText tex="k" /> completes, including failed and non-improving attempts.</dd></div>
            <div><dt><MathText tex="S_k" /></dt><dd>Best correctness-validated speedup after evaluation <MathText tex="k" />. The reference implementation is retained: <MathText tex="S_0=1" /> and <MathText tex="S_k\ge1" />.</dd></div>
            <div><dt><MathText tex="\Delta g_k" /></dt><dd>Incremental gain in log-speedup at evaluation <MathText tex="k" />.</dd></div>
          </dl>
          <p><b>Aggregation:</b> Average independent repeats within each kernel, then weight valid kernels equally. Report incomplete coverage explicitly.</p>
          <p>Higher STS indicates better budget-aware optimization under matched evaluation conditions.</p>
          </section>
          <h4>Cost-to-Gain (CTG)</h4>
          <p>Cost-to-Gain (CTG) measures the additional tokens required to increase speedup by a specified fraction relative to reference checkpoint <MathText tex="r" />.</p>
          <MathText :tex="reportMath.ctg" display />
          <dl class="var-list">
            <div><dt><MathText tex="r" /></dt><dd>Reference checkpoint, with token consumption <MathText tex="B_r" /> and incumbent speedup <MathText tex="S_r" />.</dd></div>
            <div><dt><MathText tex="\delta" /></dt><dd>Target relative speedup gain: 1%, 5%, or 10%.</dd></div>
            <div><dt><MathText tex="B_k-B_r" /></dt><dd>Additional token consumption accumulated since the reference checkpoint.</dd></div>
          </dl>
          <p>Lower CTG indicates greater token efficiency. An unattained target has <MathText tex="\mathrm{CTG}=+\infty" />. An unavailable token axis is missing evidence, not an unattained target. Relative speedup gains differ from latency-reduction thresholds.</p>
        </section>
        <section class="method-group metric-definition-panel" aria-labelledby="performance-metrics-title">
          <h3 id="performance-metrics-title" class="metric-panel-title">Performance Metrics</h3>
          <h4>GM Speedup</h4>
          <p>GM Speedup is the equal-weight geometric mean of best correctness-validated speedups across tasks. In the supplied supplementary tables, it summarizes terminal selected trajectories. Higher values indicate greater speedup.</p>
          <MathText :tex="reportMath.gm" display />
          <dl class="var-list">
            <div><dt><MathText tex="N" /></dt><dd>Number of evaluation tasks; each task receives equal weight.</dd></div>
            <div><dt><MathText tex="S_{i,n_i}" /></dt><dd>Best correctness-validated speedup on task <MathText tex="i" /> after its <MathText tex="n_i" /> evaluations, under the evaluation conditions and budget specified for the metric.</dd></div>
          </dl>
          <h4>GM @50M</h4>
          <p>GM @50M is the geometric mean of correctness-validated incumbent speedups within a 50-million-token budget per task, over available kernel trajectories. Higher values indicate better performance at this fixed budget. Missing trajectories remain unavailable and require explicit coverage reporting.</p>
          <h4><MathText tex="\mathrm{fast}_p" /></h4>
          <p>The fraction of tasks with a correctness-validated speedup strictly greater than <MathText tex="p" />. Higher values indicate broader task coverage above the chosen threshold.</p>
          <MathText :tex="reportMath.fast" display />
          <dl class="var-list">
            <div><dt><MathText tex="p" /></dt><dd>Speedup threshold, with <MathText tex="p\ge1" />. Only speedups strictly greater than this threshold count.</dd></div>
            <div><dt><MathText tex="\mathbf{1}[\cdot]" /></dt><dd>Indicator function: 1 when the condition holds and 0 otherwise.</dd></div>
          </dl>
          <p>Tasks without improvement retain speedup 1 and do not contribute to <MathText tex="\mathrm{fast}_p" /> for <MathText tex="p\ge1" />. Comparisons require matched evaluation conditions and budgets.</p>
          <h4>VLR</h4>
          <p>VLR expresses latency reduction relative to the reference implementation. Higher values indicate greater latency reduction. It is a dashboard diagnostic, distinct from the relative speedup gain used by CTG.</p>
          <MathText :tex="reportMath.vlr" display />
          <dl class="var-list"><div><dt><MathText tex="S" /></dt><dd>Correctness-validated speedup: reference latency divided by optimized latency.</dd></div></dl>
          <p>Harness and Default/Luna experiments remain separate cohorts; comparisons require matched task coverage and evaluation conditions.</p>
        </section>
        <section class="method-group metric-definition-panel" aria-labelledby="budget-metrics-title">
          <h3 id="budget-metrics-title" class="metric-panel-title">Budget Metrics</h3>
          <p>Budget metrics quantify the resources and time required to obtain optimization results.</p>
          <h4>Cumulative / Mean Tokens</h4>
          <p>Full selected-run token consumption, reported as either cumulative totals or per-task means. Totals and means are distinct quantities. Lower consumption indicates lower token cost under matched conditions.</p>
          <h4>LLM Rounds / Wall-clock Time</h4>
          <p>LLM rounds count model interactions during optimization; wall-clock time measures elapsed run time. Lower values indicate lower interaction or time costs under matched conditions. Median aggregates are explicitly labeled in the leaderboard.</p>
          <p>All budget metrics include unsuccessful attempts. The Kernel Panel retains observed costs even when no improvement is found. Source-estimated token trajectories retain their annotations.</p>
        </section>
        <section class="method-group metric-definition-panel" aria-labelledby="reliability-metrics-title">
          <h3 id="reliability-metrics-title" class="metric-panel-title">Reliability Metrics</h3>
          <h4>Candidate Pass Rate</h4>
          <p>The fraction of evaluated candidates that compile, execute, and pass all required correctness checks. Compilation failures, runtime errors, timeouts, and incorrect outputs count as failures. Higher values indicate more reliable candidate generation.</p>
          <p>The supplied group tables average candidate pass rates within kernels before averaging across kernels.</p>
          <h4>Task Pass Rate</h4>
          <p>The fraction of attempted tasks with a valid baseline measurement and at least one correct search candidate. Tasks with no submitted or passing candidate remain in the denominator. Higher values indicate broader coverage of correct task outcomes.</p>
          <p>Independent repeats, when available, additionally assess the reproducibility and stability of optimization outcomes.</p>
        </section>
      </div>
    </div>
  </section>
</template>
<script setup>
import { SECTIONS } from '../lib/sections';
import { reportMath } from '../lib/reportMath';
import MathText from './MathText.vue';
const section = SECTIONS.methodology;
defineProps({showHeading:{type:Boolean,default:true}});
// AscendGYM Technical Report, Section 3.3.1–3.3.4 (PDF pages 5–6).
const categories = [
  { title: 'Joint Performance–Budget Metrics', goal: 'Evaluate speedup gains and the token cost of obtaining them with STS and CTG.' },
  { title: 'Performance Metrics', goal: 'Measure the magnitude and task coverage of validated speedups with GM Speedup and fastₚ.' },
  { title: 'Budget Metrics', goal: 'Quantify token consumption, LLM rounds, and wall-clock time, including failed attempts.' },
  { title: 'Reliability Metrics', goal: 'Assess candidate correctness and the coverage of correct outcomes across tasks.' }
];
</script>
