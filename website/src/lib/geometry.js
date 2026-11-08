// SVG polar helpers used by the orbit, radar and timing-ring primitives.

export function svgPolarPoint(cx, cy, radius, angleDegrees) {
  const angle = (angleDegrees - 90) * Math.PI / 180;
  return [(cx + radius * Math.cos(angle)).toFixed(2), (cy + radius * Math.sin(angle)).toFixed(2)];
}

export function svgArcPath(cx, cy, radius, startAngle, endAngle) {
  const start = svgPolarPoint(cx, cy, radius, endAngle);
  const end = svgPolarPoint(cx, cy, radius, startAngle);
  const largeArcFlag = Math.abs(endAngle - startAngle) <= 180 ? 0 : 1;
  return `M${start[0]} ${start[1]}A${radius} ${radius} 0 ${largeArcFlag} 0 ${end[0]} ${end[1]}`;
}
