"""Explicit, non-executing migration of v4 task/evaluator configuration."""
import copy
import json
import sys
from pathlib import Path

from .config import validate_config
from core.profiling import default_profiling


def from_v4(path, agent_kind):
    path = Path(path).resolve()
    old = json.loads(path.read_text(encoding="utf-8"))
    backend = old.get("backend", {})
    if backend.get("kind") != "command" or not backend.get("command"):
        raise ValueError("migration requires v4 command backend; write a thin evaluator adapter for other backends")
    inp = {k: copy.deepcopy(v) for k, v in old["input"].items()
           if k in ("source_dir", "spec_path", "editable_paths", "test_paths", "op_name")}
    for name in ("source_dir", "spec_path"):
        inp[name] = str((path.parent / inp[name]).resolve())
    argv = [v.replace("{config_dir}", str(path.parent)).replace("{python}", sys.executable)
            for v in backend["command"]]
    # Resolving every arbitrary argv as a path would corrupt flags/URLs. Require
    # explicit absolute path operands and report the remaining manual check.
    for value in argv:
        if value.startswith("./") or value.startswith("../"):
            raise ValueError("make v4 evaluator path arguments absolute before migration")
    cfg = {
        "input": inp,
        "agent": {"kind": agent_kind},
        "search": {"candidate_solutions_target": old["budget"]["candidate_evaluations_max"]},
        "evaluator": {"command": argv, "evidence_kind": backend.get("evidence_kind"),
                      "per_call_timeout_seconds": backend.get("timeout_seconds", 1800)},
        "stop": {"candidate_evaluations_max": old["budget"]["candidate_evaluations_max"],
                 "final_reprofile_count": old["budget"].get("final_reprofile_count", 1),
                 "shutdown_grace_seconds": 30},
        "metrics": copy.deepcopy(old.get("metrics", {})),
        "isolation": "process_only",
    }
    uses_add_rms_norm = any(Path(v).name == "sandbox_add_rms_norm.py" for v in argv)
    if uses_add_rms_norm:
        cfg["evaluator"]["profiling"] = default_profiling()
    if agent_kind == "claude":
        cfg["agent"].update(permission_mode="acceptEdits", allowed_tools=["Read", "Edit", "Write", "Glob", "Grep", "Bash"])
    elif agent_kind == "codex":
        cfg["agent"].update(sandbox="workspace-write", extra_args=["-c", "sandbox_workspace_write.network_access=true", "-c", 'approval_policy="never"'])
    validate_config(cfg)
    notes = {"source": str(path), "copied": ["input paths", "evaluator command", "metrics checkpoints"],
             "dropped_budget_fields": sorted(set(old["budget"]) - {"candidate_evaluations_max", "final_reprofile_count"}),
             "old_agent_configuration_discarded": True,
             "checks_required": ["CLI login, actual model and permissions", "absolute evaluator argument paths on target node",
                                 "sandbox availability, fixed NPU, frozen test/golden hashes", "external filesystem/network isolation"],
             "executed": False}
    if uses_add_rms_norm:
        notes["checks_required"].extend([
            "Use the UPDATED sandbox_add_rms_norm.py and core/profiling.py; old bridge versions will be rejected",
            "New sampling protocol is 8 total / 3 warmup / mean; remove stale 4/1/median rules from the NEW experiment spec",
            "If bridge config has a profiling object, it must exactly match evaluator.profiling; legacy profile_repeats is overridden and recorded",
        ])
    return cfg, notes
