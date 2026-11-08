"""Validate the supplied CSV and build run-specific shape details, without changing metrics."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'public/templates/shape-results.csv'
OUTPUT = ROOT / 'src/data/shape_results.json'
FIELDS = ['entry_id', 'operator', 'run_id', 'case_id', 'shape', 'dtype', 't_base_us', 't_best_us', 'correctness']


def build():
    bundle = json.loads((ROOT / 'src/data/supplement_bundle.json').read_text())
    runs = {(e['id'], op['operator']): op['trajectory_run_id'] for e in bundle['entries'] for op in e['per_operator']}
    records, seen, statuses = {}, set(), Counter()
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Unexpected CSV columns: {reader.fieldnames}')
        for line, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f'CSV row {line}: malformed column count')
            row = {key: value.strip() for key, value in raw.items()}
            key = (row['entry_id'], row['operator'])
            if key not in runs or runs[key] != row['run_id']:
                raise ValueError(f'CSV row {line}: entry/kernel/run does not match the published trajectory')
            case_key = (*key, row['run_id'], row['case_id'])
            if case_key in seen:
                raise ValueError(f'CSV row {line}: duplicate case identity {case_key}')
            seen.add(case_key)
            if not all(row[field] for field in ['case_id', 'shape', 'dtype']):
                raise ValueError(f'CSV row {line}: case_id, shape and dtype are required')
            if row['correctness'] not in ['pass', 'fail', 'error', 'not_measured']:
                raise ValueError(f'CSV row {line}: invalid correctness status')
            shape = {field: row[field] for field in ['case_id', 'shape', 'dtype', 'correctness']}
            shape['source_row'] = line
            for field in ['t_base_us', 't_best_us']:
                value = float(row[field]) if row[field] else None
                if value is not None and (not math.isfinite(value) or value <= 0):
                    raise ValueError(f'CSV row {line}: {field} must be positive and finite')
                if row['correctness'] == 'pass' and value is None:
                    raise ValueError(f'CSV row {line}: passed cases require both timings')
                shape[field] = value
            record = records.setdefault(key, {field: row[field] for field in ['entry_id', 'operator', 'run_id']})
            record.setdefault('duration_by_shape', []).append(shape)
            statuses[row['correctness']] += 1
    return {'meta': {'source': 'public/templates/shape-results.csv', 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                     'case_count': len(seen), 'kernel_count': len(records), 'status_counts': dict(statuses)},
            'records': list(records.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = build()
    if args.check:
        if json.loads(OUTPUT.read_text()) != result:
            raise SystemExit('Shape bundle is stale; run npm run data:shapes.')
    else:
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f"PASS: {result['meta']['case_count']} shapes, {result['meta']['kernel_count']} matching kernel/run groups; {result['meta']['status_counts']}.")
