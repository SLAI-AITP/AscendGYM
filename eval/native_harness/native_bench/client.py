"""Agent-facing evaluation CLI. Native agent shell tools invoke this directly."""
import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--connection", required=True)
    p.add_argument("action", choices=["baseline", "candidate", "status"])
    p.add_argument("--request-id", help="Reuse this ID after an uncertain transport result; never resubmit with a new ID")
    a = p.parse_args()
    conn = json.loads(Path(a.connection).read_text())
    rid = a.request_id or str(uuid.uuid4())
    payload = {} if a.action == "status" else {"kind": a.action, "request_id": rid}
    # Print request identity before dispatch so interruption does not lose it.
    print("evaluation request_id=" + rid, file=sys.stderr, flush=True)
    endpoint = "/status" if a.action == "status" else "/evaluate"
    request = urllib.request.Request(conn["url"] + endpoint, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + conn["token"], "Content-Type": "application/json"})
    try:
        # Evaluator has its own infrastructure timeout. There is no run wall-clock limit.
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode(), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Transport outcome unknown ({type(exc).__name__}); do not blindly resubmit. request_id={rid}", file=sys.stderr)
        return 3
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if data.get("status") not in ("rejected", "error", "busy", "pending") else 1


if __name__ == "__main__":
    sys.exit(main())
