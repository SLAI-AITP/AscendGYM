import json
import copy
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from core.metrics import compute_metrics
from .adapters import command, probe
from .common import Journal, dump, sha, write_json
from .config import validate_config, candidate_target
from .evaluation import EvaluationGateway, read_files, source_hash
from .usage import UsageLedger


def kill_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


class NativeRun:
    def __init__(self, config, output):
        validate_config(config)
        config = copy.deepcopy(config)
        config.setdefault("search", {})["candidate_solutions_target"] = candidate_target(config)
        self.cfg, self.root = config, Path(output).resolve()
        source = Path(config["input"]["source_dir"]).resolve()
        if self.root.exists() or self.root == source or source in self.root.parents:
            raise ValueError("output must be new and outside input source")
        candidates = sorted(source.rglob("*"))
        if any(p.is_symlink() for p in candidates):
            raise ValueError("symlinks not allowed in input bundle")
        names = [p.relative_to(source).as_posix() for p in candidates if p.is_file()]
        if not names or len(names) > 256 or any(n == ".bench" or n.startswith(".bench/") for n in names):
            raise ValueError("provide a curated source bundle (1..256 files, no .bench/)")
        self.initial = read_files(source, names)
        if not set(config["input"]["editable_paths"] + config["input"]["test_paths"]) <= self.initial.keys():
            raise ValueError("editable/test file missing from source")
        spec = Path(config["input"]["spec_path"]).read_text(encoding="utf-8")
        if not spec.strip():
            raise ValueError("operator description is empty")
        self.root.mkdir(parents=True)
        self.workspace = self.root / "workspace"
        for name, content in self.initial.items():
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode())
            if name not in config["input"]["editable_paths"]:
                path.chmod(0o444)
        self.run_id = str(uuid.uuid4())
        self.journal = Journal(self.root, self.run_id)
        self.usage = UsageLedger(config["agent"].get("event_format", config["agent"]["kind"]))
        self.gateway = EvaluationGateway(config, self.root, self.initial, self.journal, self.usage)
        self.task_dir = self.workspace / ".bench"
        self.task_dir.mkdir()
        (self.task_dir / "operator_spec.md").write_text(spec)
        reference = Path(__file__).resolve().parents[1] / "core/references/metrics_master.md"
        self.manifest = {"schema_version": "native-bench-run-v1", "run_id": self.run_id,
            "input": {"files": {n: sha(v.encode()) for n, v in self.initial.items()},
                      "source_hash": source_hash(self.initial), "spec_sha256": sha(spec.encode())},
            "metrics_master_sha256": sha(reference.read_bytes()) if reference.exists() else None,
            "task": config["input"], "agent": config["agent"], "stop": config["stop"],
            "search": config["search"], "profiling": config["evaluator"].get("profiling"),
            "isolation": config.get("isolation", "process_only"),
            "security_note": "Process-only mode is cooperative, NOT an OS security boundary. Use external isolation for untrusted agents.",
            "evidence_kind": "synthetic" if config["agent"].get("evidence_kind") == "synthetic" else config["evaluator"]["evidence_kind"]}
        package = Path(__file__).resolve().parent
        self.manifest["harness_sources_sha256"] = {p.name: sha(p.read_bytes())
            for p in sorted(package.iterdir()) if p.suffix in (".py", ".txt")}
        write_json(self.task_dir / "input_manifest.json", self.manifest)
        write_json(self.root / "config.resolved.json", config)
        self.process = None

    def _prompt(self, connection):
        base = [sys.executable, str(Path(__file__).with_name("client.py").resolve()),
                "--connection", str(connection)]
        values = {"op_name": self.cfg["input"].get("op_name", "operator"), "workspace": str(self.workspace),
            "spec_path": str(self.task_dir / "operator_spec.md"), "manifest_path": str(self.task_dir / "input_manifest.json"),
            "editable_paths": dump(self.cfg["input"]["editable_paths"]), "test_paths": dump(self.cfg["input"]["test_paths"]),
            "candidate_evaluations_max": self.cfg["stop"]["candidate_evaluations_max"],
            "candidate_solutions_target": candidate_target(self.cfg),
            "profiling_protocol": (dump(self.cfg["evaluator"]["profiling"])
                if "profiling" in self.cfg["evaluator"] else "由评测器固定定义；不自行推测采样参数。"),
            **{kind + "_command": shlex.join(base + [kind]) for kind in ("baseline", "candidate", "status")}}
        prompt = Path(__file__).with_name("prompt.txt").read_text().format(**values)
        if self.cfg.get("task_note"):
            prompt += "\n任务补充：\n" + self.cfg["task_note"] + "\n"
        return prompt

    def _read_stream(self, pipe):
        with (self.root / "agent.stdout.jsonl").open("wb") as out:
            for line_number, line in enumerate(iter(pipe.readline, b""), 1):
                out.write(line)
                out.flush()
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("native event is not an object")
                    self.usage.consume(record)
                    self.journal.event("native_event", raw_line=line_number, payload=record)
                except (ValueError, UnicodeDecodeError) as exc:
                    self.journal.event("unparsed_stdout", raw_line=line_number, error=str(exc))
                    self.usage.stream_errors.append("unparsed native stdout at line " + str(line_number))

    def run(self):
        # Preflight before starting any listener: bad CLI/flags must not leak it.
        agent = self.cfg["agent"]
        if agent["kind"] != "custom":
            self.manifest["agent_probe"] = probe(agent["kind"], agent.get("executable"))
            if not self.manifest["agent_probe"]["available"]:
                write_json(self.root / "manifest.json", self.manifest)
                raise ValueError("native CLI unavailable: " + agent["kind"])
        command(agent, "", self.workspace, self.root)
        self.usage.reset_meter()
        server, server_thread = self.gateway.serve()
        connection = self.task_dir / "connection.json"
        write_json(connection, {"url": f"http://127.0.0.1:{server.server_port}", "token": self.gateway.token})
        connection.chmod(0o600)
        prompt = self._prompt(connection)
        (self.root / "prompt.txt").write_text(prompt)
        self.manifest["prompt_sha256"] = sha(prompt.encode())
        write_json(self.root / "manifest.json", self.manifest)
        argv, stdin = command(agent, prompt, self.workspace, self.root)
        write_json(self.root / "launch.json", {"argv": argv, "cwd": str(self.workspace), "stdin": stdin,
                                               "note": "Initial task only; native agent owns the loop and system prompt"})
        environment = os.environ.copy()
        # Existing CLI auth is inherited, never copied into artifacts.
        # Secrets must not be supplied as argv/config values.
        environment["NATIVE_BENCH_CONNECTION"] = str(connection)
        environment["NATIVE_BENCH_RUN_ID"] = self.run_id
        failure, forced, native_completed = None, False, False
        checks, reader = [], None
        self.journal.event("agent_started", adapter=agent["kind"])
        try:
            with (self.root / "agent.stderr.txt").open("wb") as stderr:
                self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                    cwd=self.workspace, env=environment, start_new_session=True)
                reader = threading.Thread(target=self._read_stream, args=(self.process.stdout,), daemon=True)
                reader.start()
                try:
                    self.process.stdin.write(stdin.encode())
                    self.process.stdin.close()
                except BrokenPipeError:
                    self.process.stdin.close()
                closing_since = None
                while self.process.poll() is None:
                    if (self.root / "STOP").exists():
                        self.gateway.close_admission("manual_stop")
                    if self.gateway.stop_ready.is_set():
                        if closing_since is None:
                            closing_since = time.monotonic()
                            self.journal.event("agent_closing", reason=self.gateway.stop_reason)
                        # This grace starts ONLY after another stop condition, not
                        # at run start. It permits a native final usage response.
                        if time.monotonic() - closing_since >= self.cfg["stop"].get("shutdown_grace_seconds", 20):
                            forced = True
                            kill_group(self.process, signal.SIGINT)
                            try:
                                self.process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                kill_group(self.process, signal.SIGTERM)
                                try:
                                    self.process.wait(timeout=3)
                                except subprocess.TimeoutExpired:
                                    kill_group(self.process, signal.SIGKILL)
                                    self.process.wait()
                    time.sleep(.05)
                native_completed = self.process.returncode == 0 and not forced
                self.gateway.close_admission("agent_finished" if native_completed else "agent_error")
                # Reap remaining descendants; no background agent process is allowed
                # to continue optimizing after its run is declared ended.
                kill_group(self.process, signal.SIGTERM)
                reader.join(timeout=5)
                if reader.is_alive():
                    kill_group(self.process, signal.SIGKILL)
                    reader.join(timeout=2)
                    self.usage.stream_errors.append("native stdout remained open after process exit")
                self.journal.event("agent_exited", returncode=self.process.returncode, forced=forced)
        except KeyboardInterrupt:
            self.gateway.close_admission("manual_stop")
            forced = True
            if self.process is not None:
                kill_group(self.process, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    kill_group(self.process, signal.SIGKILL)
                    self.process.wait()
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            self.gateway.close_admission("agent_error")
            if self.process is not None and self.process.poll() is None:
                kill_group(self.process, signal.SIGKILL)
                self.process.wait()
        finally:
            server.shutdown()
            server.server_close()  # Waits for accepted evaluations; they have infrastructure timeouts.
            server_thread.join()
            if reader is not None:
                reader.join(timeout=2)
            if self.process is not None:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
                if self.process.stdout and (reader is None or not reader.is_alive()):
                    self.process.stdout.close()
        if self.usage.kind == "opencode" and self.usage.session_id:
            self._opencode_export()
        usage = self.usage.report(native_completed)
        write_json(self.root / "usage.json", usage)
        self.journal.event("usage_finalized", usage=usage)
        if failure is None and self.gateway.stop_reason not in ("agent_error", "manual_stop", "gateway_error"):
            try:
                checks = self.gateway.final_verification(self.cfg["stop"].get("final_reprofile_count", 1))
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)}
        return self._finalize(usage, checks, failure, forced)

    def _opencode_export(self):
        agent = self.cfg["agent"]
        argv = [agent.get("executable", "opencode"), "export", self.usage.session_id]
        try:
            # Only a native local export command, never a new model conversation.
            p = subprocess.run(argv, cwd=self.workspace, capture_output=True, timeout=60)
            (self.root / "opencode.export.json").write_bytes(p.stdout)
            (self.root / "opencode.export.stderr.txt").write_bytes(p.stderr)
            if p.returncode:
                raise ValueError("native export failed")
            self.usage.consume_export(json.loads(p.stdout))
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            self.journal.event("usage_export_unavailable", error=str(exc))

    def _finalize(self, usage, checks, failure, forced):
        reconciled = {(c["kind"], c["as_of_ns"]): c for c in usage.get("completion_checkpoints", [])}
        recovered = False
        for row in self.gateway.rows:
            previous = row.get("completion_usage_checkpoint", {})
            checkpoint = reconciled.get((row["evaluation_kind"], previous.get("as_of_ns")))
            if checkpoint is not None:
                if not previous.get("available") and checkpoint["available"]:
                    recovered = True
                    self.journal.event("completion_usage_reconciled", evaluation_id=row["evaluation_id"],
                                       cumulative_tokens=checkpoint["cumulative_tokens"], as_of_ns=checkpoint["as_of_ns"])
                row["completion_usage_checkpoint"] = {k: v for k, v in checkpoint.items() if k != "kind"}
                row["cumulative_tokens"] = checkpoint["cumulative_tokens"]
        rows = [dict(r) for r in self.gateway.rows]
        for r in rows:
            if not r.get("completion_usage_checkpoint", {}).get("available"):
                r["cumulative_tokens"] = None
        if recovered and (self.root / "evaluations.jsonl").exists():
            (self.root / "evaluations.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        write_json(self.root / "solutions.json", rows)
        final_count = self.cfg["stop"].get("final_reprofile_count", 1)
        final_valid = len(checks) == final_count and all(r["status"] == "passed" for r in checks)
        try:
            original_unchanged = read_files(self.cfg["input"]["source_dir"], self.initial) == self.initial
        except (OSError, ValueError):
            original_unchanged = False
        good_stop = self.gateway.stop_reason in ("agent_finished", "candidate_budget_exhausted")
        success = bool(good_stop and self.gateway.baseline and self.gateway.used and final_valid
                       and failure is None and original_unchanged)
        context = {"operational_ok": success, "usage_complete": usage["available"] and usage["submission_checkpoints_usable"],
                   "candidate_usage_complete": usage["submission_checkpoints_usable"],
                   "verification_required": final_count > 0, "verification_valid": final_valid}
        metrics = compute_metrics(rows, self.cfg.get("metrics", {}), self.manifest["evidence_kind"], context)
        if not usage["submission_checkpoints_usable"]:
            metrics["STS_token"] = {"status": "unavailable", "reason": usage["checkpoint_limitation"]}
        metrics["token_accounting_note"] = "Candidate coordinates use timestamped provider receipts; unavailable session-tail totals do not invalidate proven candidate counts."
        metrics["official_ranking_eligible"] = False
        write_json(self.root / "metrics.json", metrics)
        result = {"schema_version": "native-bench-result-v1", "run_id": self.run_id, "success": success,
                  "evidence_kind": self.manifest["evidence_kind"], "termination_reason": self.gateway.stop_reason,
                  "error": failure, "agent_forced_shutdown": forced, "usage": usage,
                  "candidate_evaluations": self.gateway.used, "profiling_count": len(rows),
                  "baseline": self.gateway.baseline, "best_verified": self.gateway.best,
                  "final_verification": {"requested": final_count, "valid": final_valid, "evaluations": checks},
                  "original_input_unchanged": original_unchanged,
                  "elapsed_seconds": time.monotonic() - self.journal.started,
                  "stopping_policy": "evaluation count / native completion / manual stop / fatal error; no token or run wall-clock caps",
                  "trace_scope": "Native CLI observable events and provider usage counters, not undisclosed provider prompts or hidden reasoning"}
        write_json(self.root / "result.json", result)
        self.journal.event("run_finished", success=success, result_ref="result.json")
        return result
