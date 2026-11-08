"""Reparse observable native CLI events without calling any model."""
import json
from pathlib import Path

from .common import sha
from .usage import UsageLedger, replay_transport_usage


def replay_usage(run):
    root = Path(run).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    agent = manifest["agent"]
    ledger = UsageLedger(agent.get("event_format", agent["kind"]))
    ledger.meter_url = ""
    path = root / "agent.stdout.jsonl"
    raw = path.read_bytes()
    for index, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("not a JSON object")
            ledger.consume(event)
        except ValueError:
            ledger.stream_errors.append("unparsed native stdout at line " + str(index))
    export = root / "opencode.export.json"
    if ledger.kind == "opencode" and export.exists():
        try:
            ledger.consume_export(json.loads(export.read_text()))
        except ValueError:
            pass
    # Native process outcome is determined from recorded process-exit evidence,
    # not from whether optimization succeeded or the old parser accepted usage.
    exits = [e for line in (root / "events.jsonl").read_text().splitlines()
             if (e := json.loads(line)).get("event_type") == "agent_exited"]
    if not exits:
        completed = False
    else:
        completed = exits[-1].get("returncode") == 0 and not exits[-1].get("forced")
    report = ledger.report(completed)
    saved = json.loads((root / "usage.json").read_text())
    if saved.get("source") == "provider.transport.usage":
        report = replay_transport_usage(report, saved, json.loads((root / "solutions.json").read_text()))
    return {"schema_version": "native-bench-usage-replay-v1", "run_id": result["run_id"],
            "source_sha256": sha(raw), "source": str(path), "original_artifacts_modified": False,
            "usage": report}
