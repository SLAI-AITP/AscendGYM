"""Native CLI counters and provider usage at trusted evaluation completion boundaries."""
import threading
import json
import os
import time
import uuid
import urllib.request
import urllib.error
from urllib.parse import urlsplit


FIELDS = ("input_tokens", "uncached_input_tokens", "cache_read_tokens", "cache_write_tokens",
          "output_tokens", "reasoning_tokens", "total_tokens")


def number(value):
    return value if type(value) is int and value >= 0 else None


def normalized(raw, kind):
    if kind == "claude":
        uncached = number(raw.get("input_tokens", raw.get("inputTokens")))
        read = number(raw.get("cache_read_input_tokens", raw.get("cacheReadInputTokens")))
        write = number(raw.get("cache_creation_input_tokens", raw.get("cacheCreationInputTokens")))
        inp = uncached + read + write if None not in (uncached, read, write) else None
        out = number(raw.get("output_tokens", raw.get("outputTokens")))
        reason = number(raw.get("reasoning_tokens"))
    elif kind == "codex":
        inp, out = number(raw.get("input_tokens")), number(raw.get("output_tokens"))
        read, write = number(raw.get("cached_input_tokens")), number(raw.get("cache_write_input_tokens"))
        uncached = inp - read if None not in (inp, read) and inp >= read else None
        reason = number(raw.get("reasoning_output_tokens"))
    elif kind == "opencode":
        uncached = number(raw.get("input"))
        read, write = number(raw.get("cache", {}).get("read")), number(raw.get("cache", {}).get("write"))
        inp = uncached + read + write if None not in (uncached, read, write) else None
        # OpenCode v1.2.27 getUsage stores outputTokens INCLUDING reasoningTokens.
        # Preserve reasoning as a breakdown, never add it a second time.
        out, reason = number(raw.get("output")), number(raw.get("reasoning"))
    else:
        return {k: number(raw.get(k)) for k in FIELDS}
    total = inp + out if inp is not None and out is not None else None
    return dict(zip(FIELDS, (inp, uncached, read, write, out, reason, total)))


def total(rows):
    rows = list(rows)
    return {k: sum(r[k] for r in rows) if rows and all(r[k] is not None for r in rows) else None
            for k in FIELDS}


class UsageLedger:
    """Versioned native formats with conservative checkpoint coverage.

    Claude: finalized stream message usage, reconciled against native result.
    Codex: turn.completed contains turn aggregate, NOT per model response usage.
    OpenCode: step_finish tokens, reconciled with `export SESSION_ID` at exit.
    """
    def __init__(self, kind):
        self.kind = kind
        self.lock = threading.RLock()
        self.rows, self.raw_rows = {}, {}
        self.open_messages, self.current = {}, {}
        self.session_id = None
        self.models = set()
        self.terminal = None
        self.terminal_source = None
        self.terminal_error = False
        self.child_seen = False
        self.stream_errors = []
        self.native_warnings = []
        self.codex_turn = 0
        self.version = None
        self.meter_url = os.environ.get("NATIVE_BENCH_USAGE_URL", "")
        self.meter_token = os.environ.get("NATIVE_BENCH_USAGE_TOKEN", "")
        self.meter_epoch = None
        self.completion_checkpoints = []
        self.last_transport_receipt = None

    def reset_meter(self):
        if not self.meter_url:
            return
        parts = urlsplit(self.meter_url)
        if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1") or parts.username or parts.query or parts.fragment:
            raise ValueError("completion usage meter must share the local monotonic clock")
        request = urllib.request.Request(self.meter_url, data=b"{}", method="POST",
            headers={"Authorization": "Bearer " + self.meter_token, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.meter_epoch = json.load(response)["epoch"]

    def completion_checkpoint(self, at_ns=None, kind=None):
        at_ns = time.monotonic_ns() if at_ns is None else at_ns
        result = {"available": False, "cumulative_tokens": None, "as_of_ns": at_ns,
                  "reason": "run-scoped model transport meter is not configured", "epoch": self.meter_epoch}
        if self.meter_url:
            try:
                request = urllib.request.Request(self.meter_url + "?at_ns=" + str(at_ns),
                    headers={"Authorization": "Bearer " + self.meter_token})
                with urllib.request.urlopen(request, timeout=5) as response:
                    result = json.load(response)
                if result.get("epoch") != self.meter_epoch:
                    raise ValueError("transport meter epoch changed")
                result = transport_snapshot(result["records"], at_ns, result["epoch"])
                with self.lock:
                    if self.last_transport_receipt is None or at_ns >= self.last_transport_receipt["as_of_ns"]:
                        self.last_transport_receipt = result
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = {"available": False, "cumulative_tokens": None, "as_of_ns": at_ns,
                          "reason": "transport usage unavailable: " + type(exc).__name__, "epoch": self.meter_epoch}
        if kind in ("baseline", "candidate"):
            with self.lock:
                self.completion_checkpoints.append({"kind": kind, **{k: v for k, v in result.items() if k != "records"}})
        return result

    def consume(self, record):
        with self.lock:
            try:
                self._consume(record)
            except (KeyError, TypeError, ValueError) as exc:
                self.stream_errors.append(type(exc).__name__ + ": " + str(exc))

    def _consume(self, r):
        typ = r.get("type")
        if self.kind == "claude":
            if typ == "system" and r.get("subtype") == "init":
                self.session_id = r.get("session_id")
                self.version = r.get("claude_code_version")
                if r.get("model"):
                    self.models.add(r["model"])
            if r.get("parent_tool_use_id"):
                self.child_seen = True
            if typ == "stream_event":
                e = r["event"]
                lane = (r.get("session_id"), r.get("parent_tool_use_id"))
                if e["type"] == "message_start":
                    msg = e["message"]
                    key = str((lane, msg["id"]))
                    self.current[lane] = key
                    self.open_messages[key] = dict(msg.get("usage", {}))
                    if msg.get("model"):
                        self.models.add(msg["model"])
                elif e["type"] == "message_delta" and lane in self.current:
                    self.open_messages[self.current[lane]].update(e.get("usage", {}))
                elif e["type"] == "message_stop" and lane in self.current:
                    key = self.current.pop(lane)
                    raw = self.open_messages.pop(key)
                    self.raw_rows[key] = raw
                    self.rows[key] = normalized(raw, self.kind)
            if typ == "result":
                models = r.get("modelUsage") or {}
                self.models.update(models)
                if models:
                    self.terminal = total(normalized(v, self.kind) for v in models.values())
                    self.terminal_source = "claude.result.modelUsage"
                elif not self.child_seen:
                    self.terminal = normalized(r.get("usage") or {}, self.kind)
                    self.terminal_source = "claude.result.usage (main loop)"
                self.terminal_error = bool(r.get("is_error"))
        elif self.kind == "codex":
            if typ == "thread.started":
                self.session_id = r.get("thread_id")
            if typ == "turn.started":
                self.codex_turn += 1
            if typ == "turn.completed":
                key = r.get("turn_id") or f"turn-{self.codex_turn}"
                self.raw_rows[key] = r.get("usage") or {}
                self.rows[key] = normalized(self.raw_rows[key], self.kind)
                self.terminal = total(self.rows.values())
                self.terminal_source = "codex.exec.turn.completed.usage"
            if typ == "turn.failed":
                self.terminal_error = True
            elif typ == "error":
                # Codex also emits type=error for retry/reconnect notifications.
                # Only a failed turn / missing final usage / process failure is
                # terminal. Preserve warnings rather than discard a later native
                # turn.completed report. This is NOT a provider billing audit.
                self.native_warnings.append(r)
        elif self.kind == "opencode":
            if r.get("sessionID"):
                self.session_id = r["sessionID"]
            if typ == "step_finish":
                part = r["part"]
                key = part["id"]
                self.raw_rows[key] = part["tokens"]
                self.rows[key] = normalized(part["tokens"], self.kind)
            if typ == "tool_use" and r.get("part", {}).get("tool") == "task":
                self.child_seen = True
            if typ == "error":
                self.terminal_error = True
        elif self.kind == "custom" and typ == "benchmark.usage":
            # An external CLI adapter may emit native counts with a stable ID.
            key = r["usage_id"]
            self.raw_rows[key] = r["usage"]
            self.rows[key] = normalized(r["usage"], "custom")
            if r.get("final"):
                self.terminal = total(self.rows.values())
                self.terminal_source = "custom.command.native_usage"

    def consume_export(self, data):
        """OpenCode's own per-session export; never account-wide stats subtraction."""
        with self.lock:
            info = data.get("info", {})
            if info.get("id") != self.session_id:
                raise ValueError("OpenCode export session identity mismatch")
            found = {}
            for message in data.get("messages", []):
                m = message.get("info", {})
                if m.get("role") == "assistant":
                    if m.get("modelID"):
                        self.models.add(m["modelID"])
                    found[m["id"]] = normalized(m.get("tokens") or {}, "opencode")
            self.terminal = total(found.values())
            self.terminal_source = "opencode.export.session.assistant.tokens"

    def checkpoint(self):
        with self.lock:
            observed = total(self.rows.values())
            provisional = (self.kind in ("claude", "opencode", "custom") and bool(self.rows)
                           and not self.open_messages and not self.child_seen and not self.stream_errors
                           and observed["total_tokens"] is not None)
            return {"observed": observed, "provisional": provisional,
                    "cumulative_tokens": observed["total_tokens"] if provisional else None,
                    "usage_record_ids": sorted(self.rows),
                    "reason": "native stream counters; require final reconciliation" if provisional else
                              "per-submission native usage is not observable or complete"}

    def report(self, completed):
        transport = self.completion_checkpoint() if self.meter_url else None
        with self.lock:
            observed = total(self.rows.values())
            end = self.terminal
            if self.kind == "opencode" and end is None and completed and not self.child_seen:
                end = observed
                source = "opencode.run.step_finish.tokens"
            else:
                source = self.terminal_source
            available = bool(completed and not self.terminal_error and end and
                             end["total_tokens"] is not None and not self.stream_errors and
                             not (self.kind == "opencode" and self.child_seen))
            reconciled = bool(available and end["total_tokens"] == observed["total_tokens"] and
                              not self.open_messages and not self.child_seen and self.kind != "codex")
            report = {"available": available, "total": end if end else {k: None for k in FIELDS},
                    "source": source, "observed_stream_subtotal": observed,
                    "submission_checkpoints_reconciled": reconciled,
                    "submission_checkpoints_usable": False,
                    "checkpoint_limitation": "No causal native usage-to-evaluation binding; receive-time subtotals are not exact submission token counts.",
                    "session_id": self.session_id, "models_observed": sorted(self.models),
                    "native_usage_records": len(self.rows), "child_activity_seen": self.child_seen,
                    "observed_model_request_count": len(self.rows) if self.kind in ("claude", "opencode") and not self.child_seen and not self.open_messages and not self.stream_errors and completed else None,
                    "native_turn_count": self.codex_turn if self.kind == "codex" else None,
                    "errors": list(self.stream_errors),
                    "native_warnings": list(self.native_warnings),
                    "note": "Counts reported by native CLI, not independently measured provider billing. Unknown stays null."}
            if transport is not None:
                proof = self.last_transport_receipt
                checkpoints = [reconcile_completion(c, proof) for c in self.completion_checkpoints]
                candidates = [c for c in checkpoints if c["kind"] == "candidate"]
                usable = bool(candidates and all(c["available"] for c in candidates))
                report.update(native_cli_total=report["total"], available=transport["available"],
                    total=transport.get("total", {k: None for k in FIELDS}),
                    source="provider.transport.usage", transport_receipt=transport,
                    completion_checkpoints=checkpoints,
                    submission_checkpoints_reconciled=usable,
                    submission_checkpoints_usable=usable,
                    checkpoint_limitation=None if usable else
                        "a candidate completion has missing usage, active requests, or no covering transport receipt",
                    note="Candidate completion counts and whole-session totals have separate availability; no token estimates.")
                if proof is not None and proof is not transport:
                    report["completion_receipt"] = proof
            return report



def transport_snapshot(records, at_ns, epoch):
    if type(at_ns) is not int or at_ns < 0 or not isinstance(records, list):
        raise ValueError("invalid transport snapshot")
    selected, errors, seen = [], [], set()
    for record in records:
        rid, started, finished = record["request_id"], record["started_ns"], record["finished_ns"]
        if type(rid) is not int or rid < 1 or rid in seen or type(started) is not int or started < 0:
            raise ValueError("invalid transport request identity or start")
        seen.add(rid)
        if finished is not None and (type(finished) is not int or finished < started):
            raise ValueError("invalid transport completion")
        if started > at_ns:
            continue
        if finished is None or finished > at_ns:
            errors.append("model request active at completion boundary")
            continue
        usage = record.get("usage")
        if not isinstance(usage, dict) or number(usage.get("total_tokens")) is None:
            errors.append("provider usage missing for a completed request")
            continue
        inp, out, count = usage.get("input_tokens"), usage.get("output_tokens"), usage["total_tokens"]
        if inp is not None and out is not None and count != inp + out:
            raise ValueError("inconsistent provider token totals")
        if any(v is not None and (number(v) is None or v > count) for v in (inp, out)):
            raise ValueError("inconsistent provider token breakdown")
        if any(v is not None and number(v) is None for v in usage.values()):
            raise ValueError("invalid provider usage")
        selected.append(usage)
    sums = total(selected) if selected else {k: 0 for k in FIELDS}
    available = not errors
    return {"available": available, "cumulative_tokens": sums["total_tokens"] if available else None,
            "total": sums if available else {k: None for k in FIELDS},
            "observed_subtotal": sums, "as_of_ns": at_ns, "epoch": epoch,
            "records": records, "reason": "; ".join(sorted(set(errors))) if errors else None}


class TransportMeter:
    def __init__(self):
        self.lock = threading.RLock()
        self.records = []
        self.epoch = uuid.uuid4().hex

    def reset(self):
        with self.lock:
            if any(r["finished_ns"] is None for r in self.records):
                raise ValueError("cannot reset active model requests")
            self.records = []
            self.epoch = uuid.uuid4().hex
            return {"epoch": self.epoch}

    def begin(self):
        with self.lock:
            record = {"request_id": len(self.records) + 1, "started_ns": time.monotonic_ns(),
                      "finished_ns": None, "usage": None}
            self.records.append(record)
            return record

    def finish(self, record, usage):
        with self.lock:
            if record["finished_ns"] is not None:
                return
            record["usage"] = usage
            record["finished_ns"] = time.monotonic_ns()

    def snapshot(self, at_ns=None):
        with self.lock:
            records = [{**r, "usage": dict(r["usage"]) if r["usage"] is not None else None} for r in self.records]
            now = time.monotonic_ns()
            if at_ns is not None and at_ns > now:
                raise ValueError("usage cutoff is in the future")
            return transport_snapshot(records, now if at_ns is None else at_ns, self.epoch)


def provider_usage(raw, kind):
    if kind == "anthropic":
        inp = number(raw.get("input_tokens"))
        read, write = number(raw.get("cache_read_input_tokens", 0)), number(raw.get("cache_creation_input_tokens", 0))
        uncached = inp
        inp = inp + read + write if None not in (inp, read, write) else None
        out = number(raw.get("output_tokens"))
    else:
        inp = number(raw.get("prompt_tokens", raw.get("input_tokens")))
        out = number(raw.get("completion_tokens", raw.get("output_tokens")))
        details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
        read = number(details.get("cached_tokens", 0))
        write = number(raw.get("cache_write_input_tokens", 0))
        uncached = inp - read if None not in (inp, read) and inp >= read else None
    declared = raw.get("total_tokens")
    if declared is not None:
        count = number(declared)
        if count is None or any(v is not None and v > count for v in (inp, out)):
            return None
        if inp is not None and out is not None and inp + out != count:
            return None
    elif inp is not None and out is not None:
        count = inp + out
    else:
        return None
    reason = number((raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}).get("reasoning_tokens"))
    return dict(zip(FIELDS, (inp, uncached, read, write, out, reason, count)))


def reconcile_completion(previous, proof):
    if not proof or "records" not in proof or previous.get("epoch") != proof["epoch"] or previous["as_of_ns"] > proof["as_of_ns"]:
        return dict(previous)
    checkpoint = transport_snapshot(proof["records"], previous["as_of_ns"], proof["epoch"])
    return {"kind": previous["kind"], **{k: v for k, v in checkpoint.items() if k != "records"},
            "recovered_from_transport_receipt": bool(previous.get("recovered_from_transport_receipt")
                or (not previous["available"] and checkpoint["available"]))}


def replay_transport_usage(native, saved, rows):
    receipt = saved["transport_receipt"]
    proof = saved.get("completion_receipt", receipt)
    if "records" in receipt:
        current = transport_snapshot(receipt["records"], receipt["as_of_ns"], receipt["epoch"])
    else:
        current = {"available": False, "total": {k: None for k in FIELDS}}
    checkpoints = []
    for row in rows:
        if row["evaluation_kind"] not in ("baseline", "candidate"):
            continue
        previous = row["completion_usage_checkpoint"]
        covered = bool(proof and "records" in proof and previous.get("epoch") == proof.get("epoch")
                       and previous["as_of_ns"] <= proof["as_of_ns"])
        if previous["available"] and not covered:
            raise ValueError("completion usage lacks a covering receipt")
        checkpoint = reconcile_completion({"kind": row["evaluation_kind"], **previous}, proof)
        if checkpoint["cumulative_tokens"] != previous.get("cumulative_tokens") or checkpoint["cumulative_tokens"] != row.get("cumulative_tokens"):
            raise ValueError("completion token receipt mismatch")
        checkpoints.append(checkpoint)
    candidates = [c for c in checkpoints if c["kind"] == "candidate"]
    usable = bool(candidates and all(c["available"] for c in candidates))
    native.update(native_cli_total=native["total"], available=current["available"], total=current["total"],
        source="provider.transport.usage", transport_receipt=receipt, completion_checkpoints=checkpoints,
        submission_checkpoints_reconciled=usable, submission_checkpoints_usable=usable,
        checkpoint_limitation=None if usable else "a candidate completion lacks complete transport evidence")
    if "completion_receipt" in saved:
        native["completion_receipt"] = proof
    return native
