import assert from 'node:assert/strict';
import fs from 'node:fs';
import { applyShapeResults } from '../src/lib/shapeResults.js';
const bundle = JSON.parse(fs.readFileSync(new URL('../src/data/supplement_bundle.json', import.meta.url)));
const shapes = JSON.parse(fs.readFileSync(new URL('../src/data/shape_results.json', import.meta.url)));
const before = JSON.stringify(bundle);
const merged = applyShapeResults(bundle, shapes);
let count = 0;
for (const [i, entry] of merged.entries.entries()) {
  assert.deepEqual(entry.overall, bundle.entries[i].overall);
  for (const [j, op] of entry.per_operator.entries()) {
    const { duration_by_shape, ...rest } = op;
    const { duration_by_shape: previous, ...original } = bundle.entries[i].per_operator[j];
    assert.deepEqual(rest, original, 'shapes must not change trajectories, ranks or other metrics');
    const record = shapes.records.find(row => row.entry_id === entry.id && row.operator === op.operator && row.run_id === op.trajectory_run_id);
    assert.deepEqual(duration_by_shape, record.duration_by_shape);
    count += duration_by_shape.length;
    for (const row of duration_by_shape) {
      if (row.correctness === 'pass') assert(row.t_base_us > 0 && row.t_best_us > 0);
      else assert.equal(row.t_best_us, null, 'unmeasured cases must not gain an optimized timing');
    }
  }
}
assert.equal(count, shapes.meta.case_count);
assert.equal(JSON.stringify(bundle), before, 'source bundle must remain untouched');
const wrongRun = structuredClone(bundle);
wrongRun.entries[0].per_operator[0].trajectory_run_id = 'different-run';
assert.deepEqual(applyShapeResults(wrongRun, shapes).entries[0].per_operator[0], wrongRun.entries[0].per_operator[0]);
console.log(`PASS: ${count} shape records, exact three-part joins, missing timings preserved and all benchmark metrics unchanged.`);
