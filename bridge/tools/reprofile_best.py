#!/usr/bin/env python3
"""Re-run final verification for the best version of a native or bare run.

This is the native/bare replacement for the previous reprofile helper.
It loads a finished run, restores/uses its best snapshot, runs ``count`` final
verification evaluations (optionally pinned to one NPU), and writes a
``_final_recheck_<timestamp>/result.json`` that ``apply_final_recheck.py`` can
write back.

It does not run the agent again and does not consume candidate budget.

Examples
--------
    # native: dry-run plan (needs SANDBOX_URL or --sandbox-url)
    python3 tools/reprofile_best.py --run-dir <runs>/<TAG>/<RUN_ID> \
        --count 5 --sandbox-url "$SANDBOX_URL" --dry-run

    # native: run on a specific NPU, then apply the result
    python3 tools/reprofile_best.py --run-dir <runs>/<TAG>/<RUN_ID> \
        --count 5 --npu node-0:npu0 --sandbox-url "$SANDBOX_URL" --apply

    # bare: same interface; the bridge run config is read from
    # <run-dir>.json (or --config <path>)

If ``--count`` is omitted, the count defaults to the run's original
``final_verification.requested``, then ``stop.final_reprofile_count``.
``--dry-run`` also validates the snapshot/staging/package inputs without
starting the evaluator or Backend.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BRIDGE_DIR = Path(__file__).resolve().parents[1]
if str(BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(BRIDGE_DIR))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_last_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("evaluator stdout contains no JSON object")


def assigned_server(value: Any) -> str | None:
    """Best-effort recursive extraction of the sandbox assigned_server field."""
    if isinstance(value, dict):
        for key in ("assigned_server", "assignedServer", "node"):
            if isinstance(value.get(key), str) and value[key]:
                return value[key]
        for item in value.values():
            found = assigned_server(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = assigned_server(item)
            if found:
                return found
    return None


def parse_npu(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    node, _, npu = value.partition(":")
    if not npu:
        return None, re.sub(r"^npu", "", node, flags=re.IGNORECASE)
    return node or None, re.sub(r"^npu", "", npu, flags=re.IGNORECASE)


def set_preferred_npu(env: dict[str, str], npu: str | None) -> None:
    node, npu_id = parse_npu(npu)
    if node:
        env["SANDBOX_PREFERRED_NODE_ID"] = node
    if npu_id:
        env["SANDBOX_PREFERRED_NPU_ID"] = str(npu_id)


def package_metadata(config: dict, args) -> dict:
    source = Path(config["input"]["source_dir"])
    package_json = source.parent / "package.json"
    package = read_json(package_json) if package_json.is_file() else {}
    return {
        "ops_repo": args.ops_repo or package.get("ops_repo") or config["input"].get("ops_repo") or "",
        "data_dir": args.data_dir or package.get("data_dir"),
        "data_cases": package.get("data_cases"),
        "experimental": bool(package.get("experimental")),
    }


def assigned_server_from_log(path: Any) -> str | None:
    """Read the sandbox worker id from a profiling log when the API response omits it."""
    if not path:
        return None
    try:
        text = Path(str(path)).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = re.findall(r'"assigned_server"\s*:\s*"([^"]+)"', text)
    if not matches:
        return None
    counts: dict[str, int] = {}
    for item in matches:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get)


def preflight(harness: str, run_dir: Path, result: dict, config: dict, args) -> list[str]:
    """Validate inputs that both dry-run and a real recheck need."""
    errors: list[str] = []
    best = result.get("best_verified") or {}
    if harness == "native":
        solution_id = best.get("solution_id")
        if not solution_id:
            errors.append("result.json has no best_verified.solution_id")
        elif not (run_dir / "snapshots" / str(solution_id) / "files").is_dir():
            errors.append(f"native best snapshot not found: snapshots/{solution_id}/files")
        evaluator = config.get("evaluator") or {}
        if not evaluator.get("command"):
            errors.append("config has no evaluator.command")
        if not (args.sandbox_url or os.environ.get("SANDBOX_URL")):
            errors.append("native recheck needs --sandbox-url or SANDBOX_URL")
        try:
            package = package_metadata(config, args)
        except (KeyError, OSError, ValueError) as exc:
            errors.append(f"cannot read package metadata: {exc}")
        else:
            if not package["ops_repo"]:
                errors.append("package metadata has no ops_repo; pass --ops-repo")
            data_dir = package["data_dir"]
            if data_dir and not Path(str(data_dir)).expanduser().is_dir():
                errors.append(f"package data_dir not found: {data_dir} (pass --data-dir)")
    else:
        if not (args.version or best.get("version")):
            errors.append("bare result has no best_verified.version")
        for name in ("op_path", "op_path_version"):
            if not (run_dir / "_stage" / name).is_dir():
                errors.append(f"bare staged directory missing: {run_dir}/_stage/{name}")
        backend_root = Path(str((config.get("backend") or {}).get("root") or ""))
        if not (backend_root / "app.py").is_file():
            errors.append(f"bare Backend app.py not found: {backend_root / 'app.py'}")
        sandbox = config.get("sandbox") or {}
        if not args.sandbox_url and not (sandbox.get("host") and sandbox.get("port") is not None):
            errors.append("bare config has no sandbox host/port; pass --sandbox-url")
    return errors


def resolve_count(result: dict, config: dict, requested: int | None) -> int:
    """Default to the original final-verification count, then the run budget."""
    if requested is not None:
        return requested
    final = result.get("final_verification") or {}
    value = final.get("requested")
    if isinstance(value, int) and value > 0:
        return value
    value = (config.get("stop") or {}).get("final_reprofile_count")
    if isinstance(value, int) and value > 0:
        return value
    return 1


def native_evaluator(config: dict, args) -> tuple[list[str], dict]:
    evaluator = config.get("evaluator") or {}
    command = evaluator.get("command")
    if not command:
        raise SystemExit("native config has no evaluator.command")
    return list(command), evaluator


def native_recheck(run_dir: Path, result: dict, config: dict, doc: dict, args) -> bool:
    best = result.get("best_verified") or {}
    solution_id = best.get("solution_id")
    if not solution_id:
        raise SystemExit("native result.json has no best_verified.solution_id")
    snapshot = run_dir / "snapshots" / solution_id / "files"
    if not snapshot.is_dir():
        raise SystemExit(f"native best snapshot not found: {snapshot}")

    package = package_metadata(config, args)
    command, evaluator = native_evaluator(config, args)
    profiling = evaluator.get("profiling")
    timeout = int(evaluator.get("per_call_timeout_seconds") or 5400)
    env = os.environ.copy()
    sandbox_url = args.sandbox_url or env.get("SANDBOX_URL")
    if not sandbox_url:
        raise SystemExit("native recheck needs --sandbox-url or SANDBOX_URL")
    env["SANDBOX_URL"] = sandbox_url
    env["SANDBOX_OPS_REPO"] = str(package["ops_repo"])
    env["SANDBOX_OP_NAME"] = str(result.get("op_name") or config["input"]["op_name"])
    if profiling:
        total = int(profiling.get("total_runs") or 0)
        warmup = int(profiling.get("warmup_runs") or 0)
        env["SANDBOX_PROFILING_RUNS"] = str(max(1, total - warmup))
        env["SANDBOX_WARMUP_RUNS"] = str(warmup)
    if package["data_dir"]:
        env["SANDBOX_TEST_DATA_DIR"] = str(package["data_dir"])
        env["SANDBOX_TEST_DATA_CASES"] = str(package["data_cases"] or "all")
    if package["experimental"]:
        env["SANDBOX_EXPERIMENTAL"] = "1"
    set_preferred_npu(env, args.npu)

    out_dir: Path = doc.pop("_out_dir")
    evaluations = []
    for index in range(doc["requested"]):
        evaluation_dir = out_dir / "evaluations" / f"RF{index + 1:03d}"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "schema_version": "ascend-gym-evaluator-v1",
            "evaluation_kind": "final_verification",
            "source_dir": str(snapshot),
            "artifacts_dir": str(evaluation_dir),
        }
        if profiling:
            request["profiling"] = dict(profiling)
        write_json(evaluation_dir / "request.json", request)
        try:
            proc = subprocess.run(command, input=json.dumps(request), text=True,
                                  capture_output=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            evaluations.append({"evaluation_id": f"RF{index + 1:03d}",
                                "status": "timeout", "correctness_passed": None,
                                "diagnostic": f"evaluator exceeded {timeout}s"})
            break
        except OSError as exc:
            evaluations.append({"evaluation_id": f"RF{index + 1:03d}",
                                "status": "sandbox_error", "correctness_passed": None,
                                "diagnostic": f"evaluator could not start: {exc}"})
            break
        (evaluation_dir / "stdout.json").write_text(proc.stdout or "", encoding="utf-8")
        (evaluation_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            evaluations.append({"evaluation_id": f"RF{index + 1:03d}",
                                "status": "sandbox_error", "correctness_passed": None,
                                "diagnostic": f"evaluator exited {proc.returncode}"})
            break
        try:
            observed = parse_last_json(proc.stdout)
        except ValueError as exc:
            evaluations.append({"evaluation_id": f"RF{index + 1:03d}",
                                "status": "protocol_error", "correctness_passed": None,
                                "diagnostic": str(exc)})
            break
        observed_protocol = (observed.get("raw") or {}).get("profiling_protocol")
        protocol_ok = not profiling or observed_protocol == profiling
        status = observed.get("status") or "failed"
        if status == "passed" and not protocol_ok:
            status = "profiling_protocol_mismatch"
        passed = status == "passed" and observed.get("correctness_passed") is True
        row = {
            "evaluation_id": f"RF{index + 1:03d}",
            "status": status,
            "correctness_passed": observed.get("correctness_passed") is True,
            "duration_ms": observed.get("duration_ms"),
            "profiling_protocol": observed_protocol,
            "backend_result": observed,
        }
        if doc.get("version"):
            row["version"] = doc["version"]
        server = assigned_server(observed.get("raw") or observed)
        if server:
            row["assigned_server"] = server
        evaluations.append(row)
        if not passed:
            break
    doc["evaluations"] = evaluations
    doc["valid"] = len(evaluations) == doc["requested"] and all(
        row.get("status") == "passed" and row.get("correctness_passed") is True
        for row in evaluations)
    return doc["valid"]


def backend_call(api_url: str, name: str, parameters: dict, timeout: float) -> dict:
    import requests
    body = "<tool_call>\n" + json.dumps(
        {"name": name, "parameters": parameters}, ensure_ascii=False
    ) + "\n</tool_call>"
    response = requests.post(api_url, json=body, headers={"Content-Type": "application/json"},
                             timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("backend response is not an object")
    return value


def bare_recheck(run_dir: Path, result: dict, config: dict, doc: dict, args) -> bool:
    from bare_bench.runner import backend_env, free_port, stop_backend, wait_health

    best = result.get("best_verified") or {}
    version = args.version or best.get("version")
    if not version:
        raise SystemExit("bare result.json has no best_verified.version")
    doc["version"] = version

    staged = {
        "op_path_root": str(run_dir / "_stage" / "op_path"),
        "version_root": str(run_dir / "_stage" / "op_path_version"),
    }
    if not Path(staged["op_path_root"]).is_dir() or not Path(staged["version_root"]).is_dir():
        raise SystemExit(f"bare staged directories missing under {run_dir}/_stage")
    backend = config.get("backend") or {}
    backend_root = Path(str(backend.get("root") or ""))
    app = backend_root / "app.py"
    if not app.is_file():
        raise SystemExit(f"bare Backend app.py not found: {app}")
    timeout = float((config.get("client") or {}).get("backend_timeout_seconds") or 3600)
    env = backend_env(config, staged)
    env["PYTHONUNBUFFERED"] = "1"
    if args.sandbox_url:
        host, port = args.sandbox_url.rsplit(":", 1)
        env["PROFILING_HOST"] = host.split("//")[-1]
        env["PROFILING_PORT"] = port
    set_preferred_npu(env, args.npu)
    port = free_port()
    log_path = doc.pop("_out_dir") / "backend.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(backend.get("python") or sys.executable), str(app),
        "--port", str(port),
        "--op-path", staged["op_path_root"],
        "--op-path-version", staged["version_root"],
        "--min-improvement-epsilon", str(backend.get("min_improvement_epsilon", 0.0)),
    ]
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=str(backend_root), env=env,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_health(port, int(backend.get("start_timeout_seconds", 90)), log_path,
                    required_tools=("ascend_code_profiling", "restore_op_version"))
        api_url = f"http://127.0.0.1:{port}/api/"
        op_name = result.get("op_name") or config["input"]["op_name"]
        evaluations = []
        try:
            restore = backend_call(api_url, "restore_op_version",
                                   {"op_name": op_name, "version": version}, timeout)
        except Exception as exc:
            restore = None
            evaluations.append({"evaluation_id": "RF001", "status": "transport_unknown",
                                "correctness_passed": None, "version": version,
                                "diagnostic": f"restore: {type(exc).__name__}: {exc}"})
        if restore is not None and not restore.get("success"):
            evaluations.append({"evaluation_id": "RF001", "status": "restore_failed",
                                "correctness_passed": None, "version": version,
                                "backend_result": restore})
        elif restore is not None:
            for index in range(doc["requested"]):
                try:
                    raw = backend_call(api_url, "ascend_code_profiling",
                                       {"op_name": op_name}, timeout)
                except Exception as exc:
                    evaluations.append({
                        "evaluation_id": f"RF{index + 1:03d}",
                        "status": "transport_unknown",
                        "correctness_passed": None,
                        "version": version,
                        "diagnostic": f"profiling: {type(exc).__name__}: {exc}",
                    })
                    break
                inner = raw.get("tool_call_result") if isinstance(raw.get("tool_call_result"), dict) else {}
                passed = bool(raw.get("success") and inner.get("status") == "success")
                protocol = inner.get("profiling_protocol")
                status = "passed" if passed and protocol == config.get("profiling") else (
                    "profiling_protocol_mismatch" if passed else (inner.get("status") or "failed"))
                row = {
                    "evaluation_id": f"RF{index + 1:03d}",
                    "status": status,
                    "correctness_passed": bool(passed),
                    "duration_us": inner.get("exec_time_us"),
                    "profiling_protocol": protocol,
                    "version": version,
                    "backend_result": raw,
                    "restore_result": restore if index == 0 else None,
                }
                server = assigned_server(raw) or assigned_server_from_log(
                    inner.get("profiling_log_path"))
                if server:
                    row["assigned_server"] = server
                evaluations.append(row)
                if status != "passed":
                    break
        doc["evaluations"] = evaluations
        doc["valid"] = len(evaluations) == doc["requested"] and all(
            row.get("status") == "passed" and row.get("correctness_passed") is True
            for row in evaluations)
        return doc["valid"]
    finally:
        stop_backend(proc)
        handle.close()


def run_apply(run_dir: Path, out_dir: Path, doc: dict, args) -> int:
    command = [
        sys.executable, str(Path(__file__).with_name("apply_final_recheck.py")),
        "--run-dir", str(run_dir), "--recheck-dir", str(out_dir), "--yes",
    ]
    if args.npu:
        command += ["--npu", args.npu]
    if doc.get("version"):
        command += ["--version", str(doc["version"])]
    if args.force_apply:
        command += ["--force"]
    if args.no_timing_refresh:
        command += ["--no-timing-refresh"]
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None,
                        help="run config; native defaults to <run-dir>/config.resolved.json, "
                             "bare defaults to <run-dir>.json")
    parser.add_argument("--count", type=int, default=None,
                        help="number of final verification evaluations; default follows the "
                             "original final_verification.requested, then final_reprofile_count")
    parser.add_argument("--version", default=None,
                        help="version to recheck; default from result.best_verified.version")
    parser.add_argument("--npu", default=None,
                        help="target NPU, e.g. node-0:npu0")
    parser.add_argument("--sandbox-url", default=None,
                        help="native: sandbox URL; defaults to SANDBOX_URL")
    parser.add_argument("--ops-repo", default=None, help="override package ops_repo")
    parser.add_argument("--data-dir", default=None, help="override package data_dir")
    parser.add_argument("--recheck-dir", default=None,
                        help="output directory; default <run-dir>/_final_recheck_<timestamp>")
    parser.add_argument("--apply", action="store_true",
                        help="call apply_final_recheck.py after a valid recheck")
    parser.add_argument("--force-apply", action="store_true",
                        help="pass --force to apply_final_recheck.py")
    parser.add_argument("--no-timing-refresh", action="store_true",
                        help="pass --no-timing-refresh to apply_final_recheck.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without starting the backend/evaluator")
    args = parser.parse_args()

    if args.count is not None and args.count <= 0:
        parser.error("--count must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        print(f"result.json not found: {result_path}", file=sys.stderr)
        return 1
    result = read_json(result_path)
    schema = result.get("schema_version")
    if schema == "native-bench-result-v1":
        harness = "native"
        config_path = Path(args.config) if args.config else run_dir / "config.resolved.json"
    elif schema == "klens-bare-client-result-v1":
        harness = "bare"
        config_path = Path(args.config) if args.config else Path(str(run_dir) + ".json")
    else:
        print(f"unsupported result schema: {schema!r}", file=sys.stderr)
        return 1
    if not config_path.is_file():
        print(f"run config not found: {config_path}", file=sys.stderr)
        return 1
    config = read_json(config_path)
    args.count = resolve_count(result, config, args.count)
    best = result.get("best_verified") or {}
    version = args.version or best.get("version")
    if harness == "bare" and not version:
        print("bare result has no best_verified.version", file=sys.stderr)
        return 1

    errors = preflight(harness, run_dir, result, config, args)
    if errors:
        print("preflight failed:", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        return 1
    if args.apply and result.get("success") is True and not args.force_apply:
        print("warning: result.json is already success=true; --apply will not write "
              "unless --force-apply is set", file=sys.stderr)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.recheck_dir).expanduser().resolve() if args.recheck_dir else \
        run_dir / f"_final_recheck_{stamp}"
    if out_dir.exists():
        print(f"recheck directory already exists: {out_dir}", file=sys.stderr)
        return 1
    doc = {
        "schema_version": "ascend-gym-final-recheck-v1",
        "harness": harness,
        "run_id": run_dir.name,
        "requested": args.count,
        "target_npu": args.npu,
        "best_solution_id": best.get("solution_id"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "valid": False,
        "evaluations": [],
        "_out_dir": out_dir,
    }
    if version is not None:
        doc["version"] = version
    print(f"run: {run_dir}")
    print(f"harness: {harness}  version: {version or 'unknown'}  target_npu: {args.npu or 'auto'}")
    print(f"recheck: {out_dir}  requested={args.count}")
    if args.dry_run:
        print("dry-run: no backend/evaluator started, no files written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        if harness == "native":
            native_recheck(run_dir, result, config, doc, args)
        else:
            bare_recheck(run_dir, result, config, doc, args)
    finally:
        doc.pop("_out_dir", None)
        doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(out_dir / "result.json", doc)

    print(f"recheck valid={doc['valid']}  evaluations={len(doc['evaluations'])}/{doc['requested']}")
    if not doc["valid"]:
        print("recheck did not pass; apply_final_recheck.py was not called", file=sys.stderr)
        return 2
    if args.apply:
        return run_apply(run_dir, out_dir, doc, args)
    print(f"recheck result: {out_dir / 'result.json'}")
    print("apply with: python3 tools/apply_final_recheck.py --run-dir ... --recheck-dir ... --yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
