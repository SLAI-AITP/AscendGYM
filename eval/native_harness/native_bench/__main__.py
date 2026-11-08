import argparse
import json
import sys
from pathlib import Path

from .adapters import probe
from .common import dump, write_json
from .config import load_config
from .runner import NativeRun
from .migrate import from_v4
from .replay import replay_usage


def main():
    p = argparse.ArgumentParser(description="Native coding-agent benchmark: no custom model/tool loop")
    sub = p.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    stop = sub.add_parser("stop", help="request a manual stop of this run only")
    stop.add_argument("run")
    sub.add_parser("doctor", help="read CLI availability/version without invoking models")
    migrate = sub.add_parser("migrate-v4", help="generate native task config; does not run an agent")
    migrate.add_argument("--config", required=True)
    migrate.add_argument("--agent", choices=("claude", "codex", "opencode"), required=True)
    migrate.add_argument("--output", required=True)
    replay = sub.add_parser("replay-usage", help="reparse native logs without invoking a model or altering original evidence")
    replay.add_argument("--run", required=True)
    replay.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        if args.action == "doctor":
            print(dump([probe(k) for k in ("claude", "codex", "opencode")]))
            return 0
        if args.action == "stop":
            root = Path(args.run).resolve()
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest.get("schema_version") != "native-bench-run-v1" or (root / "result.json").exists():
                raise ValueError("not an active native benchmark run")
            write_json(root / "STOP", {"run_id": manifest["run_id"], "reason": "manual_stop"})
            print("Stop requested: " + str(root))
            return 0
        if args.action == "migrate-v4":
            cfg, notes = from_v4(args.config, args.agent)
            target = Path(args.output).resolve()
            report = target.with_suffix(".migration.json")
            if target.exists() or report.exists():
                raise ValueError("migration output must be new")
            write_json(target, cfg)
            write_json(report, notes)
            print(dump(notes))
            return 0
        if args.action == "replay-usage":
            target = Path(args.output).resolve()
            if target.exists():
                raise ValueError("replay output must be new")
            report = replay_usage(args.run)
            write_json(target, report)
            print(dump(report))
            return 0
        result = NativeRun(load_config(args.config), args.output).run()
        print(dump({k: result[k] for k in ("run_id", "success", "termination_reason", "candidate_evaluations", "profiling_count", "usage")}))
        return 0 if result["success"] else 1
    except (ValueError, OSError, KeyError) as exc:
        print(f"native benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
