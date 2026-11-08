import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(dump(value) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def path_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or str(p) != name or "\\" in name or ":" in name:
        raise ValueError("unsafe relative source path")
    return name


class Journal:
    def __init__(self, root, run_id):
        self.root, self.run_id = Path(root), run_id
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.sequence = 0

    def append(self, name, row):
        with self.lock:
            with (self.root / name).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()

    def event(self, kind, **data):
        with self.lock:
            self.sequence += 1
            row = {"event_id": self.sequence, "run_id": self.run_id,
                   "event_type": kind, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_sec": time.monotonic() - self.started, **data}
            self.append("events.jsonl", row)
            return row
