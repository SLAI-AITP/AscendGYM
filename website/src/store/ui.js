// Global UI state: theme, export mode, drawer, toast, tooltip.
// These mirror the body/html attribute contract of the original stylesheet
// (body[data-highlight-model], html[data-theme]).
import { reactive } from 'vue';
import { benchmark, readStartupParams, findEntryById } from './benchmark';

export const ui = reactive({
  theme: 'light',
  exportMode: false,
  exportBreakpoint: null,
  highlightModel: null, // v2: entry id
  drawerModelId: null   // v2: entry id
});

export const toast = reactive({ message: '', visible: false });
let toastTimer;

export function showToast(message) {
  toast.message = message;
  toast.visible = true;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.visible = false; }, 1800);
}

// Tooltip carries raw cursor coordinates; DataTooltip.vue clamps the final
// position against the viewport once the content has rendered.
export const tooltip = reactive({ html: '', visible: false, x: 0, y: 0 });

export function showTooltip(html, clientX, clientY) {
  tooltip.html = html;
  tooltip.visible = true;
  tooltip.x = clientX;
  tooltip.y = clientY;
}

export function hideTooltip() {
  tooltip.visible = false;
}

export function openDrawer(entryId) {
  // Resolve against the merged roster so imported entries open too.
  if (!findEntryById(entryId)) return;
  ui.drawerModelId = entryId;
}

export function closeDrawer() {
  ui.drawerModelId = null;
}

export function toggleTheme() {
  ui.theme = ui.theme === 'light' ? 'dark' : 'light';
}

export async function copyText(text, confirmation) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement('textarea');
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }
  showToast(confirmation);
}

export function focusSearch() {
  document.querySelector('#leaderboard')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  setTimeout(() => document.querySelector('#model-search')?.focus(), 220);
}

export function scrollToId(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Reads theme / export / breakpoint params once at startup.
export function applyStartupParams() {
  const params = readStartupParams();
  if (params.get('theme') === 'dark') ui.theme = 'dark';
  if (params.get('export') === '1') ui.exportMode = true;
  const exportBreakpoints = { tablet: { width: 834, height: 1000 }, mobile: { width: 390, height: 844 } };
  const breakpointName = ['tablet', 'mobile'].includes(params.get('breakpoint')) ? params.get('breakpoint') : null;
  if (breakpointName) ui.exportBreakpoint = breakpointName;
  return { params, exportBreakpoint: exportBreakpoints[breakpointName] };
}

// Print master injection: keeps each long-scroll export route on one page.
export function configureExportPage(params, exportBreakpoint) {
  if (params.get('export') !== '1') return;
  requestAnimationFrame(() => {
    const width = exportBreakpoint?.width || 1440;
    // Desktop paged media expands vertical rhythm more than responsive page sizes.
    // Keep a bounded allowance per export route so each long-scroll master remains one page.
    const breakpointName = params.get('breakpoint');
    const printAllowance = breakpointName === 'mobile' ? 0.68 : breakpointName === 'tablet' ? 0.92 : 1.45;
    const height = Math.max(Math.ceil(document.documentElement.scrollHeight * printAllowance), exportBreakpoint?.height || 1000);
    const printOverride = document.createElement('style');
    printOverride.id = 'export-page-size';
    printOverride.textContent = `@page{size:${width / 96}in ${height / 96}in;margin:0}@media print{html,body{width:${width}px!important}.site-header,.section,.site-footer{width:${width}px!important}}`;
    document.head.append(printOverride);
  });
}
