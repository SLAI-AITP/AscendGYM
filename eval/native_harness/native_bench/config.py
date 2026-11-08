import json
import math
import os
import sys
from pathlib import Path

from .adapters import KINDS
from .common import path_name
from core.metrics import sts, paos_single
from core.profiling import validate_profiling


def candidate_target(cfg):
    return cfg.get("search", {}).get("candidate_solutions_target", cfg["stop"]["candidate_evaluations_max"])


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    validate_config(cfg)
    def _expand(value):
        return os.path.expanduser(os.path.expandvars(value))

    for name in ("source_dir", "spec_path"):
        raw = _expand(cfg["input"][name])
        candidate = Path(raw)
        cfg["input"][name] = str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
    for section in ("agent", "evaluator"):
        if "command" in cfg[section]:
            cfg[section]["command"] = [os.path.expanduser(os.path.expandvars(
                arg.replace("{python}", sys.executable).replace("{config_dir}", str(path.parent))))
                                       for arg in cfg[section]["command"]]
    return cfg


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a JSON object")
    required = {"input", "agent", "evaluator", "stop"}
    allowed = required | {"metrics", "isolation", "task_note", "search"}
    if not required <= cfg.keys() or set(cfg) - allowed:
        raise ValueError("native config requires input/agent/evaluator/stop; v4 budget and unknown fields are rejected")
    sections = {
        "input": {"source_dir", "spec_path", "editable_paths", "test_paths", "op_name"},
        "agent": {"kind", "executable", "model", "extra_args", "allowed_tools", "permission_mode", "sandbox", "command", "event_format", "evidence_kind"},
        "evaluator": {"command", "evidence_kind", "per_call_timeout_seconds", "profiling"},
    }
    for name, keys in sections.items():
        if not isinstance(cfg[name], dict) or set(cfg[name]) - keys:
            raise ValueError("unsupported fields in " + name + "; limits and secrets do not belong here")
    if not isinstance(cfg["stop"], dict):
        raise ValueError("stop must be an object")
    for name in ("source_dir", "spec_path"):
        if not isinstance(cfg["input"].get(name), str) or not cfg["input"][name]:
            raise ValueError("input." + name + " required")
    stop = cfg["stop"]
    if set(stop) - {"candidate_evaluations_max", "final_reprofile_count", "shutdown_grace_seconds"}:
        raise ValueError("token/time/run-turn limits are not termination conditions in native mode")
    if type(stop.get("candidate_evaluations_max")) is not int or stop["candidate_evaluations_max"] < 1:
        raise ValueError("candidate_evaluations_max must be positive")
    search = cfg.get("search", {})
    if not isinstance(search, dict) or set(search) - {"candidate_solutions_target"}:
        raise ValueError("search only accepts candidate_solutions_target")
    target = candidate_target(cfg)
    if type(target) is not int or not 1 <= target <= stop["candidate_evaluations_max"]:
        raise ValueError("candidate_solutions_target must be an integer in [1, candidate_evaluations_max]")
    if "profiling" in cfg["evaluator"]:
        validate_profiling(cfg["evaluator"]["profiling"])
    if type(stop.get("final_reprofile_count", 1)) is not int or stop.get("final_reprofile_count", 1) < 0:
        raise ValueError("invalid final_reprofile_count")
    grace = stop.get("shutdown_grace_seconds", 20)
    if type(grace) not in (int, float) or not math.isfinite(grace) or grace < 0 or grace > 300:
        raise ValueError("shutdown grace must be 0..300 seconds, only after a non-time stop condition")
    kind = cfg["agent"].get("kind")
    if kind not in KINDS:
        raise ValueError("unknown agent adapter")
    if cfg["agent"].get("event_format", kind) not in KINDS:
        raise ValueError("unknown native event format")
    if kind != "custom" and ("command" in cfg["agent"] or "event_format" in cfg["agent"]):
        raise ValueError("command/event_format overrides belong to custom adapters")
    for key in ("extra_args", "allowed_tools"):
        value = cfg["agent"].get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise ValueError("agent." + key + " must be an argv/name list")
    if cfg["agent"].get("evidence_kind", "real") not in ("real", "synthetic"):
        raise ValueError("invalid agent evidence_kind")
    timeout = cfg["evaluator"].get("per_call_timeout_seconds", 1800)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive finite per-call infrastructure timeout required")
    if cfg.get("isolation", "process_only") not in ("process_only", "externally_isolated"):
        raise ValueError("isolation must describe actual external deployment")
    metrics = cfg.get("metrics", {})
    if not isinstance(metrics, dict) or set(metrics) - {"token_checkpoints", "time_checkpoints_sec", "recipe"}:
        raise ValueError("unsupported metrics fields")
    for key in ("token_checkpoints", "time_checkpoints_sec"):
        checkpoints = metrics.get(key, [])
        if not isinstance(checkpoints, list):
            raise ValueError("metrics checkpoints must be arrays, not stop limits")
        sts([], 1., checkpoints, "unused")
    try:
        paos_single(1., 1., metrics.get("recipe"))
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid metric recipe") from exc
    if kind == "custom" and not cfg["agent"].get("command"):
        raise ValueError("custom adapter requires command argv")
    for section in ("agent", "evaluator"):
        command = cfg[section].get("command")
        if command is not None and (not isinstance(command, list) or not command or
                                    not all(isinstance(v, str) and v for v in command)):
            raise ValueError("command must be nonempty argv, never a shell string")
    if not cfg["evaluator"].get("command"):
        raise ValueError("trusted evaluator command required")
    if cfg["evaluator"].get("evidence_kind") not in ("real", "synthetic"):
        raise ValueError("explicit evaluator evidence_kind required")
    for name in ("editable_paths", "test_paths"):
        paths = cfg["input"].get(name)
        if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
            raise ValueError("nonempty unique " + name + " required")
        for p in paths:
            path_name(p)
    if set(cfg["input"]["editable_paths"]) & set(cfg["input"]["test_paths"]):
        raise ValueError("tests may not be editable")
