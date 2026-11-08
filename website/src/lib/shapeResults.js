const keyFor = (entryId, operator, runId) => JSON.stringify([entryId, operator, runId]);

// A shape measurement belongs to one configuration, kernel and selected run.
// Do not attach it to a different run or infer missing timings from aggregates.
export function applyShapeResults(bundle, shapes) {
  const index = new Map(shapes.records.map(record => [keyFor(record.entry_id, record.operator, record.run_id), record]));
  return {
    ...bundle,
    entries: bundle.entries.map(entry => ({
      ...entry,
      per_operator: entry.per_operator.map(operator => {
        const record = index.get(keyFor(entry.id, operator.operator, operator.trajectory_run_id));
        return record ? { ...operator, duration_by_shape: record.duration_by_shape } : operator;
      })
    }))
  };
}
