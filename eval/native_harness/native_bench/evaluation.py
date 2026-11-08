"""Trusted evaluation gateway. It has no read/edit tools and no optimization policy."""
import hmac
import json
import os
import secrets
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.contract import checked_evaluation
from .common import dump, path_name, sha, write_json


def source_hash(files):
    return sha(json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def tool_evaluation(row):
    """Return task evidence, not observer-only token counters or duplicated history."""
    keys = ("evaluation_id", "solution_id", "source_hash", "status", "correctness_passed", "duration_ms",
            "diagnostic", "profiling_report", "hardware_metrics", "artifacts_ref", "backup_created")
    return {k: row[k] for k in keys}


def read_files(root, names):
    root = Path(root).resolve()
    result = {}
    for name in names:
        path_name(name)
        p = root / name
        for parent in [p, *p.parents]:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError("symlink in submitted source")
        if not p.is_file() or root not in p.resolve().parents:
            raise ValueError("missing/unsafe submitted source: " + name)
        result[name] = p.read_bytes().decode("utf-8")
    if sum(len(v.encode()) for v in result.values()) > 8_000_000:
        raise ValueError("submitted source exceeds 8 MB")
    return result


class CommandEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, snapshot, kind, output):
        request = {"schema_version": "ascend-gym-evaluator-v1", "evaluation_kind": kind,
                   "source_dir": str(snapshot), "artifacts_dir": str(output)}
        if "profiling" in self.cfg:
            request["profiling"] = dict(self.cfg["profiling"])
        write_json(output / "evaluator_request.json", request)
        with (output / "stdout.json").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
            p = subprocess.Popen(self.cfg["command"], stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                 text=True, start_new_session=True)
            try:
                p.communicate(dump(request), timeout=self.cfg.get("per_call_timeout_seconds", 1800))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                raise RuntimeError("evaluator transport timeout; remote execution state may be unknown")
        if p.returncode:
            raise RuntimeError("evaluator exited %s; see stderr.txt (no automatic resubmission)" % p.returncode)
        result = checked_evaluation(json.loads((output / "stdout.json").read_text()))
        if "profiling" in self.cfg and result["raw"].get("profiling_protocol") != self.cfg["profiling"]:
            raise RuntimeError("evaluator did not acknowledge the requested profiling protocol; refusing mixed measurements")
        return result


class EvaluationGateway:
    def __init__(self, cfg, root, initial, journal, usage, evaluator=None):
        self.cfg, self.root, self.initial = cfg, Path(root), initial
        self.workspace = self.root / "workspace"
        self.journal, self.usage = journal, usage
        self.evaluator = evaluator or CommandEvaluator(cfg["evaluator"])
        self.limit = cfg["stop"]["candidate_evaluations_max"]
        self.lock, self.eval_lock = threading.RLock(), threading.Lock()
        self.requests, self.rows, self.snapshots = {}, [], {}
        self.used = 0
        self.baseline, self.best = None, None
        self.stop_reason = None
        self.active = False
        self.stop_ready = threading.Event()
        self.token = secrets.token_urlsafe(32)
        self._save_snapshot("S000", initial)

    def _save_snapshot(self, sid, files):
        folder = self.root / "snapshots" / sid
        if folder.exists():
            raise ValueError("snapshot already exists")
        for name, content in files.items():
            p = folder / "files" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content.encode())
        manifest = {"snapshot_id": sid, "source_hash": source_hash(files),
                    "files": {n: sha(c.encode()) for n, c in files.items()}}
        write_json(folder / "manifest.json", manifest)
        self.snapshots[sid] = manifest
        return folder / "files"

    def close_admission(self, reason):
        with self.lock:
            if self.stop_reason is None:
                self.stop_reason = reason
                self.journal.event("admission_closed", reason=reason)
            if not self.active:
                self.stop_ready.set()

    def status(self):
        with self.lock:
            return {"baseline_verified": self.baseline is not None, "candidate_evaluations_used": self.used,
                    "remaining_evaluations": max(0, self.limit - self.used),
                    "active_evaluation": self.active, "stop_required": bool(self.stop_reason),
                    "stop_reason": self.stop_reason, "best_verified": None if self.best is None else
                    {k: self.best[k] for k in ("evaluation_id", "solution_id", "duration_ms", "source_hash")}}

    def evaluate(self, request):
        if not isinstance(request, dict) or set(request) != {"kind", "request_id"} or request["kind"] not in ("baseline", "candidate"):
            raise ValueError("expected kind=baseline|candidate and request_id")
        rid = request["request_id"]
        if not isinstance(rid, str) or not rid or len(rid) > 200:
            raise ValueError("invalid request_id")
        with self.lock:
            if rid in self.requests:
                old, result = self.requests[rid]
                if old != request:
                    raise ValueError("request_id reused with different arguments")
                return result
            if self.stop_reason:
                return {"status": "rejected", **self.status()}
            if self.active:
                return {"status": "busy", "message": "wait for the existing evaluation; no budget consumed"}
            if request["kind"] == "baseline" and self.baseline is not None:
                return {"status": "ok", "evaluation": tool_evaluation(self.baseline), **self.status()}
            if request["kind"] == "candidate" and self.baseline is None:
                raise ValueError("baseline must pass first")
            self.active = True
            self.requests[rid] = (dict(request), {"status": "pending", "request_id": rid})
        with self.eval_lock:
            try:
                kind = request["kind"]
                current = read_files(self.workspace, self.initial)
                # A second read detects ordinary concurrent edits; agent is instructed
                # to keep the source quiescent while submitting. Isolation is external.
                if current != read_files(self.workspace, self.initial):
                    raise ValueError("source changed during snapshot; submit again after edits finish")
                readonly = set(self.initial) - set(self.cfg["input"]["editable_paths"])
                if any(current[n] != self.initial[n] for n in readonly):
                    raise ValueError("protected source/test changed")
                if kind == "baseline" and current != self.initial:
                    raise ValueError("baseline requires unmodified initial source")
                with self.lock:
                    if self.stop_reason:
                        result = {"status": "rejected", **self.status(), "active_evaluation": False}
                        self.requests[rid] = (dict(request), result)
                        return result
                    if kind == "candidate":
                        self.used += 1  # Reserve before dispatch, including compile/test failure.
                        sid = f"S{self.used:03d}"
                        snapshot = self._save_snapshot(sid, current)
                    else:
                        sid, snapshot = "S000", self.root / "snapshots/S000/files"
                row = self._measure(sid, snapshot, kind, rid)
                with self.lock:
                    if row["status"] in ("sandbox_error", "timeout"):
                        self.close_admission("evaluation_state_unknown")
                    elif kind == "baseline" and row["status"] != "passed":
                        self.close_admission("baseline_failed")
                    elif self.used >= self.limit:
                        self.close_admission("candidate_budget_exhausted")
                    result = {"status": "ok", "evaluation": tool_evaluation(row), **self.status(), "active_evaluation": False}
                    self.requests[rid] = (dict(request), result)
                self.journal.event("evaluation_tool_response", request=request, response=result)
                return result
            except ValueError as exc:
                result = {"status": "rejected", "message": str(exc), **self.status(), "active_evaluation": False}
                with self.lock:
                    self.requests[rid] = (dict(request), result)
                self.journal.event("evaluation_rejected", request=request, response=result)
                return result
            finally:
                with self.lock:
                    self.active = False
                    if self.stop_reason:
                        self.stop_ready.set()

    def _measure(self, sid, snapshot, kind, request_id):
        eid = f"E{len(self.rows):04d}"
        output = self.root / "evaluations" / eid
        output.mkdir(parents=True, exist_ok=False)
        checkpoint = self.usage.checkpoint()
        submitted = self.journal.event("evaluation_started", evaluation_id=eid, snapshot_id=sid,
                                       evaluation_kind=kind, request_id=request_id,
                                       source_hash=self.snapshots[sid]["source_hash"], usage_checkpoint=checkpoint)
        try:
            observed = checked_evaluation(self.evaluator.evaluate(snapshot, kind, output))
        except Exception as exc:
            observed = {"status": "sandbox_error", "correctness_passed": None, "duration_ms": None,
                        "profiling_report": "", "hardware_metrics": {},
                        "diagnostic": f"{type(exc).__name__}: {exc}", "raw": {"state_unknown": True}}
        completed_ns = time.monotonic_ns()
        completed_elapsed = time.monotonic() - self.journal.started
        completion = self.usage.completion_checkpoint(completed_ns, kind)
        write_json(output / "result.json", observed)
        row = {"evaluation_id": eid, "solution_id": sid, "source_hash": self.snapshots[sid]["source_hash"],
               "evaluation_kind": kind, "status": observed["status"], "duration_ms": observed["duration_ms"],
               "correctness_passed": observed["correctness_passed"], "usage_checkpoint": checkpoint,
               "cumulative_tokens": completion["cumulative_tokens"],
               "completion_usage_checkpoint": {k: v for k, v in completion.items() if k != "records"},
               "submitted_elapsed_sec": submitted["elapsed_sec"],
               "solution_elapsed_sec": completed_elapsed,
               "diagnostic": observed["diagnostic"], "profiling_report": observed["profiling_report"],
               "hardware_metrics": observed["hardware_metrics"],
               "artifacts_ref": f"evaluations/{eid}", "backup_created": observed["status"] == "passed"}
        with self.lock:
            self.rows.append(row)
            if row["status"] == "passed":
                if kind == "baseline":
                    self.baseline = row
                if kind != "final_verification" and (self.best is None or row["duration_ms"] < self.best["duration_ms"]):
                    self.best = row
                write_json(self.root / "backups" / (eid + ".json"), {
                    "evaluation_id": eid, "snapshot_ref": f"snapshots/{sid}",
                    "source_hash": row["source_hash"], "evaluation_ref": f"evaluations/{eid}/result.json"})
        self.journal.append("evaluations.jsonl", row)
        self.journal.event("evaluation_finished", evaluation=row)
        return row

    def final_verification(self, count):
        with self.eval_lock:
            if self.best is None or self.stop_reason in ("evaluation_state_unknown", "baseline_failed", "manual_stop", "agent_error"):
                return []
            sid = self.best["solution_id"]
            root = self.root / "snapshots" / sid / "files"
            files = read_files(root, self.initial)
            if source_hash(files) != self.snapshots[sid]["source_hash"]:
                raise ValueError("immutable snapshot corrupted")
            checks = []
            for i in range(count):
                checks.append(self._measure(sid, root, "final_verification", f"controller-final-{i}"))
                if checks[-1]["status"] in ("sandbox_error", "timeout"):
                    break
            return checks

    def serve(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + gateway.token):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > 8192:
                        raise ValueError("request size outside bounds")
                    request = json.loads(self.rfile.read(length))
                    if self.path == "/evaluate":
                        result = gateway.evaluate(request)
                    elif self.path == "/status":
                        if request != {}:
                            raise ValueError("status requires an empty object")
                        result = gateway.status()
                    else:
                        self.send_error(404)
                        return
                    code = 200
                except (ValueError, TypeError, KeyError) as exc:
                    result, code = {"status": "error", "message": str(exc)}, 400
                except Exception as exc:
                    gateway.close_admission("gateway_error")
                    result, code = {"status": "error", "message": type(exc).__name__}, 500
                body = dump(result).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                    self.wfile.flush()
                    gateway.journal.event("gateway_response_sent", endpoint=self.path,
                                          request_id=request.get("request_id"),
                                          note="Delivered to CLI client, not proof of model consumption")
                except (BrokenPipeError, ConnectionResetError):
                    gateway.journal.event("gateway_client_disconnected", endpoint=self.path)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread
