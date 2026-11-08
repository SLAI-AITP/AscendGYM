"""Validate a filled shape-results template using Python's standard library."""
import argparse
import json
import math
from pathlib import Path


def validate(data, allow_incomplete=False):
    errors = []

    def check(ok, path, message):
        if not ok:
            errors.append(f'{path}: {message}')

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    def required(value, path):
        check(text(value) or (allow_incomplete and value is None), path, 'nonempty text required')

    if not isinstance(data, dict):
        return ['root: JSON object required']
    check(data.get('schema_version') == 'ascendgym-shape-results-v1', 'schema_version', 'unsupported schema')
    check(data.get('latency_unit') == 'us', 'latency_unit', 'must be us (microseconds)')
    records = data.get('records')
    if not isinstance(records, list) or not records:
        return errors + ['records: nonempty array required']
    seen = set()
    for index, record in enumerate(records):
        path = f'records[{index}]'
        if not isinstance(record, dict):
            errors.append(f'{path}: object required')
            continue
        for field in ['entry_id', 'harness_id', 'model_id', 'operator', 'run_id', 'source_summary', 'candidate_id', 'candidate_source_hash']:
            required(record.get(field), f'{path}.{field}')
        key = tuple(str(record.get(k)) for k in ['entry_id', 'operator', 'run_id'])
        check(key not in seen, path, 'duplicate entry/operator/run')
        seen.add(key)
        selection = record.get('candidate_selection')
        check(selection in ['best_correct_within_budget', 'terminal_correct'] or (allow_incomplete and selection is None), path, 'candidate_selection must state the actual selection policy')
        budget, consumed = record.get('token_budget'), record.get('candidate_tokens_at_selection')
        check(positive(budget), f'{path}.token_budget', 'positive token budget required')
        check((type(consumed) in (int, float) and math.isfinite(consumed) and consumed >= 0) or (allow_incomplete and consumed is None), f'{path}.candidate_tokens_at_selection', 'nonnegative measured tokens required')
        if selection == 'best_correct_within_budget' and positive(budget) and type(consumed) in (int, float):
            check(consumed <= budget, path, 'selected candidate exceeds token budget')
        protocol = record.get('measurement_protocol')
        if not isinstance(protocol, dict):
            errors.append(f'{path}.measurement_protocol: object required')
        else:
            for field in ['device_model', 'cann_version', 'driver_version', 'latency_statistic', 'synchronization', 'baseline_implementation', 'benchmark_revision', 'case_manifest_uri']:
                required(protocol.get(field), f'{path}.measurement_protocol.{field}')
            for field, minimum in [('warmup_iterations', 0), ('timed_iterations', 1)]:
                value = protocol.get(field)
                check((type(value) is int and value >= minimum) or (allow_incomplete and value is None), f'{path}.measurement_protocol.{field}', f'integer >= {minimum} required')
        rows = record.get('duration_by_shape')
        if not isinstance(rows, list):
            errors.append(f'{path}.duration_by_shape: array required')
            continue
        count = record.get('expected_case_count')
        check((type(count) is int and count > 0 and count == len(rows)) or (allow_incomplete and count is None), f'{path}.expected_case_count', 'must equal the complete case count, including failures')
        case_ids = set()
        for i, row in enumerate(rows):
            row_path = f'{path}.duration_by_shape[{i}]'
            if not isinstance(row, dict):
                errors.append(f'{row_path}: object required')
                continue
            cid = row.get('case_id')
            required(cid, f'{row_path}.case_id')
            if cid is not None:
                check(str(cid) not in case_ids, row_path, 'duplicate case_id')
                case_ids.add(str(cid))
            status = row.get('correctness')
            check(status in ['pass', 'fail', 'error', 'not_measured'], row_path, 'invalid correctness status')
            if not allow_incomplete:
                check(status != 'not_measured', row_path, 'case has not been measured')
            inputs = row.get('inputs')
            check(isinstance(inputs, list) and bool(inputs), f'{row_path}.inputs', 'all input tensors required')
            for j, tensor in enumerate(inputs if isinstance(inputs, list) else []):
                if not isinstance(tensor, dict):
                    errors.append(f'{row_path}.inputs[{j}]: object required')
                    continue
                for field in ['name', 'dtype']:
                    required(tensor.get(field), f'{row_path}.inputs[{j}].{field}')
                dims = tensor.get('shape')
                check((isinstance(dims, list) and all(type(d) is int and d >= 0 for d in dims)) or (allow_incomplete and dims is None), f'{row_path}.inputs[{j}].shape', 'resolved integer dimensions required; [] means scalar')
            for field in ['t_base_us', 't_best_us']:
                value = row.get(field)
                check(value is None or positive(value), f'{row_path}.{field}', 'must be null or a positive finite latency in us')
                if status == 'pass':
                    check(positive(value), f'{row_path}.{field}', 'passed cases require both latencies')
            if status in ['fail', 'error']:
                required(row.get('error'), f'{row_path}.error')
            if status != 'not_measured':
                required(row.get('evidence_uri'), f'{row_path}.evidence_uri')
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true', help='Check a draft template while permitting unfilled values.')
    args = parser.parse_args()
    try:
        data = json.loads(args.file.read_text())
        errors = validate(data, args.allow_incomplete)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    if errors:
        print('\n'.join(errors[:30]))
        parser.exit(1, f'Validation failed: {len(errors)} error(s).\n')
    print(f'Validated {len(data["records"])} record(s)' + (' (draft; incomplete values allowed).' if args.allow_incomplete else '.'))
