// Generated from the supplementary measurement tables by scripts/sync-supplement.py.
import { applyShapeResults } from '../lib/shapeResults';
export async function loadInternalDatasets() {
  const [measurements, shapes] = await Promise.all([
    import('../data/supplement_bundle.json'), import('../data/shape_results.json')
  ]);
  const bundle = applyShapeResults(measurements.default, shapes.default);
  return { bundle, entries: bundle.entries, files: bundle.meta.source_files, problems: [], dir: 'eval_plot/png_600dpi_v2' };
}
