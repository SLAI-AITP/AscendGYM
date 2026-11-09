#!/usr/bin/env python3
"""Reclassify one run from a successful manual final recheck.

Scenario
--------
A batch has finished and ``result.json`` still says ``success=false`` because
final verification repeatedly failed on a fixed NPU, while the same best
version passed on another NPU.  A completed recheck was written under
``_final_recheck_*``; all requested recheck evaluations passed.  This tool
applies that evidence safely:

* validates that the recheck is ``valid=true`` and every evaluation passed;
* preserves the original failure under ``result.json.recheck_history``;
* replaces ``final_verification`` and sets ``success=true``;
* updates the trailing ``run_finished`` event in ``events.jsonl``;
* keeps a timestamped backup of the original ``result.json``;
* refreshes ``TIMING.md`` when a compatible batch timing reporter exists.

It never touches other operators in the batch.  It refuses a non-passing
recheck, and by default refuses to reclassify an already-successful run
(use ``--force`` only for an intentional additional recheck).

Usage
-----
    python3 tools/apply_final_recheck.py \
        --run-dir <runs_dir>/<tag>/<run-id> \
        --npu node-0:npu0 --yes

    # preview only
    python3 tools/apply_final_recheck.py --run-dir ... --dry-run --force

If ``--recheck-dir`` is omitted, the newest ``_final_recheck_*`` directory
under the run is used.  If ``--npu`` is omitted, the server is extracted from
``assigned_server`` fields in the recheck result when present.

The recheck count is not hard-coded: the tool accepts any completed recheck
whose ``requested`` is a positive integer and whose evaluations all passed.
The tool only applies an existing recheck; it does not run the recheck itself.
Use ``reprofile_best.py`` to generate a native/bare recheck and optionally
call this tool with ``--apply``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def newest_recheck_dir(run_dir: Path) -> Path:
    candidates = sorted(
        (path for path in run_dir.glob("_final_recheck_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"未找到 _final_recheck_* 目录: {run_dir}")
    return candidates[-1]


def extract_npu(recheck: dict[str, Any]) -> str:
    matches = re.findall(
        r'"assigned_server"\s*:\s*"([^"]+)"',
        json.dumps(recheck, ensure_ascii=False),
    )
    if not matches:
        # run_final_reprofile keeps the full sandbox response in
        # backend_result.profiling_log_path rather than in the compact result.
        for item in recheck.get("evaluations") or []:
            backend = item.get("backend_result") if isinstance(item, dict) else None
            log_path = backend.get("profiling_log_path") if isinstance(backend, dict) else None
            if not log_path:
                continue
            path = Path(log_path)
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            matches.extend(re.findall(r'"assigned_server"\s*:\s*"([^"]+)"', text))
    if not matches:
        return "unknown"
    counts: dict[str, int] = {}
    for item in matches:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get)


def validate_recheck(
    main: dict[str, Any],
    recheck: dict[str, Any],
    expected_version: str | None,
) -> tuple[list[str], str]:
    errors: list[str] = []
    if not main.get("best_verified"):
        errors.append("result.json has no best_verified; refusing to reclassify a baseline-only run")
    if int(main.get("candidate_evaluations") or 0) <= 0:
        errors.append("candidate_evaluations <= 0; this tool only reclassifies runs with a best candidate")
    if recheck.get("valid") is not True:
        errors.append(f"recheck.valid is not true: {recheck.get('valid')!r}")
    requested = recheck.get("requested")
    if not isinstance(requested, int) or requested <= 0:
        errors.append(f"recheck.requested must be a positive int, got {requested!r}")
    # The evidence count is intentionally not fixed to 5.  Whatever count the
    # recheck actually requested is acceptable as long as it is positive and
    # every recorded evaluation passed.
    evaluations = recheck.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        errors.append("recheck.evaluations is empty")
        evaluations = []
    for index, item in enumerate(evaluations, start=1):
        if not isinstance(item, dict):
            errors.append(f"evaluation {index} is not an object")
            continue
        if item.get("status") != "passed" or item.get("correctness_passed") is not True:
            errors.append(
                f"evaluation {index} did not pass: status={item.get('status')!r} "
                f"correctness_passed={item.get('correctness_passed')!r}"
            )
    version = expected_version
    if version is None:
        best = main.get("best_verified") or {}
        version = best.get("version")
    if version:
        mismatched = [
            item.get("evaluation_id")
            for item in evaluations
            if item.get("version") not in (None, version)
        ]
        if mismatched:
            errors.append(
                f"recheck evaluations are not for {version}: {mismatched}"
            )
    return errors, version or "unknown"


def update_events(run_dir: Path, npu: str, recheck_dir: Path, timestamp: str) -> bool:
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return False
    lines = events_path.read_text(encoding="utf-8").splitlines()
    updated = False
    for index in range(len(lines) - 1, -1, -1):
        try:
            row = json.loads(lines[index])
        except json.JSONDecodeError:
            continue
        if row.get("event_type") != "run_finished":
            continue
        row["success"] = True
        row["manual_recheck"] = {
            "target_npu": npu,
            "recheck_dir": str(recheck_dir),
            "timestamp": timestamp,
        }
        lines[index] = json.dumps(row, ensure_ascii=False)
        updated = True
        break
    if updated:
        events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return updated


def refresh_timing(run_dir: Path) -> None:
    script = Path(__file__).with_name("timing_report.py")
    batch_dir = run_dir.parent
    if not script.is_file() or not (batch_dir / "TIMING.md").is_file():
        return
    try:
        subprocess.run(
            [sys.executable, str(script), "--batch-dir", str(batch_dir)],
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"warning: TIMING.md refresh failed: {exc!r}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--recheck-dir", default=None,
                        help="默认使用最新的 _final_recheck_* 目录")
    parser.add_argument("--npu", default=None,
                        help="复测使用的 target NPU；默认从 assigned_server 提取")
    parser.add_argument("--version", default=None,
                        help="期望版本（如 v4）；默认取 result.json.best_verified.version")
    parser.add_argument("--yes", action="store_true",
                        help="执行写入；不加时只打印计划")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不写入")
    parser.add_argument("--force", action="store_true",
                        help="即使 result.json 已是 success=true 也执行")
    parser.add_argument("--no-timing-refresh", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        print(f"result.json 不存在: {result_path}", file=sys.stderr)
        return 1
    main = read_json(result_path)
    if main.get("success") is True and not args.force:
        print("result.json 已经是 success=true；如需再次记录复测，请加 --force")
        return 0

    recheck_dir = (
        Path(args.recheck_dir).expanduser().resolve()
        if args.recheck_dir else newest_recheck_dir(run_dir)
    )
    recheck_path = recheck_dir / "result.json"
    if not recheck_path.is_file():
        print(f"recheck result.json 不存在: {recheck_path}", file=sys.stderr)
        return 1
    recheck = read_json(recheck_path)
    errors, version = validate_recheck(main, recheck, args.version)
    if errors:
        print("复测结果不满足改判条件：", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        return 2

    npu = args.npu or extract_npu(recheck)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    old_fv = main.get("final_verification") or {}
    history = list(main.get("recheck_history") or [])
    history_entry = {
        "timestamp": timestamp,
        "reason": "manual_recheck_on_different_npu",
        "target_npu": npu,
        "recheck_dir": str(recheck_dir),
        "recheck_result": str(recheck_path),
        "previous_success": main.get("success"),
        "previous_operational_status": main.get("operational_status"),
        "previous_termination_reason": main.get("termination_reason"),
        "previous_raw_termination_reason": main.get("raw_termination_reason"),
        "previous_final_verification": old_fv,
        "new_final_verification": recheck,
    }
    # Native runs identify the best version by solution_id, not a version
    # string; only write ``version`` when the evidence actually carries one.
    if version != "unknown":
        history_entry["version"] = version
    new_fv = dict(recheck)
    new_fv.update({
        "source": "manual_recheck",
        "target_npu": npu,
        "recheck_dir": str(recheck_dir),
        "timestamp": timestamp,
    })
    if version != "unknown":
        new_fv["version"] = version
    else:
        new_fv.pop("version", None)
    updated = dict(main)
    updated["recheck_history"] = history + [history_entry]
    updated["final_verification"] = new_fv
    updated["success"] = True
    updated["operational_status"] = "completed"

    print(f"run: {run_dir.name}")
    print(f"version: {version}  target_npu: {npu}")
    print(f"recheck: {recheck_path}  ({len(recheck.get('evaluations') or [])} passed)")
    print(f"previous: success={main.get('success')} final_valid={old_fv.get('valid')} "
          f"reason={main.get('termination_reason')}")
    print(f"updated : success=True final_valid={new_fv.get('valid')} "
          f"history_entries={len(updated['recheck_history'])}")

    if args.dry_run or not args.yes:
        print("dry-run: 未写入；确认后加 --yes")
        return 0

    backup = result_path.with_name(
        f"{result_path.name}.bak-before-recheck-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    if backup.exists():
        backup = result_path.with_name(
            f"{result_path.name}.bak-before-recheck-{time.strftime('%Y%m%d-%H%M%S-%f')}"
        )
    shutil.copy2(result_path, backup)
    write_json_atomic(result_path, updated)
    events_updated = update_events(run_dir, npu, recheck_dir, timestamp)
    if not args.no_timing_refresh:
        refresh_timing(run_dir)
    print(f"backup: {backup}")
    print(f"events run_finished updated: {events_updated}")
    print("reclassified successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
