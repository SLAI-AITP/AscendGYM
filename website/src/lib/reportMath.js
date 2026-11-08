// AscendGYM Technical Report, Section 3.3, equations (1)–(5), PDF pages 5–6.
export const reportMath = {
  sts: String.raw`\mathrm{STS}=100\log_2 S_n-50\sum_{k=1}^{n}\frac{B_k}{B_{\max}}\Delta g_k`,
  delta: String.raw`\Delta g_k=\log_2 S_k-\log_2 S_{k-1}`,
  ctg: String.raw`\mathrm{CTG}(\delta;r)=\min_{\substack{r\le k\le n\\ S_k\ge(1+\delta)S_r}}(B_k-B_r),\quad\delta\in\{0.01,0.05,0.10\}`,
  fast: String.raw`\mathrm{fast}_p=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left[S_{i,n_i}>p\right],\quad p\ge1`,
  gm: String.raw`\mathrm{GM\ Speedup}=\left(\prod_{i=1}^{N}S_{i,n_i}\right)^{1/N}`,
  // Dashboard diagnostic retained from the former metric-reference table.
  vlr: String.raw`\mathrm{VLR}=100\left(1-\frac{1}{S}\right)\%`
};
