// Chrome's displayed 67% preset is exactly two thirds.
export const DISPLAY_SCALE = 2 / 3;

export function pageScale() {
  return Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
}

export function isDesktopViewport() {
  return window.innerWidth / pageScale() >= 981;
}
