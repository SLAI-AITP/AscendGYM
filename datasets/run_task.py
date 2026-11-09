#!/usr/bin/env python3
"""Run one frozen dataset test through sandbox-client; never recollect golden."""
import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--client", default="sandbox-client")
    parser.add_argument("--compile-timeout", type=int, default=1800)
    parser.add_argument("--run-timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry = json.loads((ROOT / "registry.json").read_text())
    entries = registry["tasks"] + registry["preserved_golden_tasks"]
    item = next((x for x in entries if x["id"] == args.task), None)
    if item is None:
        parser.error("Unknown task; see datasets/registry.json")
    spec = json.loads((ROOT / item["spec"]).read_text())
    source = ROOT / spec["source_dir"]
    if not spec.get("test_script"):
        parser.error("This preserved task has no declared executable test.")
    test = source / spec["test_script"]
    if not source.is_dir() or not test.is_file():
        parser.error("Incomplete dataset task: " + args.task)
    command = [args.client, "--custom-op-dir", str(source), "--exec-code-path", str(test),
               "--host", args.host, "--port", str(args.port), "--ops-repo", spec["ops_repo"],
               "--compile-timeout", str(args.compile_timeout), "--run-timeout", str(args.run_timeout)]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not shutil.which(args.client):
        parser.error("sandbox-client is not installed in this environment; activate the AscendForge runtime.")
    return subprocess.run(command).returncode

if __name__ == "__main__":
    raise SystemExit(main())
