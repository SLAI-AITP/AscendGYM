#!/usr/bin/env python3
"""Compute accuracy, STS, and concise PAOS metrics from operator traces.

Two on-disk layouts are read, because the harness has changed shape while old
archives are still on disk:

  native_bench (current; `<root>/<batch-tag>/<run-id>/`)
      result.json          run summary (schema native-bench-result-v1)
      solutions.json       the evaluated solutions (same records as evaluations.jsonl)
      events.jsonl         adapter-native event stream (rounds / tool calls)
      evaluations/<EID>/profile_output/**/op_summary_*.csv
      manifest.json / config.resolved.json / usage.json / launch.json

  legacy (`<root>/<run-id>/`)
      result.json + candidates.jsonl (or solutions.jsonl) + profiling/<EID>/…

The *output* schema does not depend on the layout: the readers below normalise
both shapes onto the same internal rows, so the dataset files keep the exact
structure the dashboard already consumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_TARGETS = (0.01, 0.05, 0.10)

# Where `--out-dir` writes when no explicit directory is given: the dashboard's
# own dataset folder, so a fresh run lands exactly where the UI reads from.
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "src" / "data" / "eval_datasets_summary"

# Directories that can never be a run themselves but would otherwise match the
# "is this a run?" test one level down. `snapshots/S00x/` carries its own
# manifest and `backups/` carries per-evaluation JSON, so both must be pruned
# rather than descended into.
SKIP_DIR_NAMES = frozenset({
    ".git", ".per-run", "__pycache__", "backups", "evaluations", "node_modules",
    "snapshots", "trace-html", "workspace",
})

# How deep below `--root` a run directory may sit. Depth 2 covers the native
# `<root>/<batch-tag>/<run-id>/`; depth 1 keeps the legacy `<root>/<run-id>/`.
MAX_RUN_DEPTH = 3


# ---------------------------- Windows long paths ----------------------------
#
# A native run buries its profiling CSV under
# `<run>/evaluations/<EID>/profile_output/mindstudio_profiler_output/`, which on
# Windows can push the full path past MAX_PATH (260). The failure is not a clean
# error: `Path.exists()` returns False and `open()` raises FileNotFoundError as
# though the file were absent, even though the directory clearly lists it.
#
# Relative roots make it worse — `--root ..\eval_result` still resolves against
# the current working directory, so the *effective* length is
# `len(cwd) + 1 + len(relative)`. Two defences are applied:
#
#   1. every root is made absolute up front (`resolve_root`), so the length no
#      longer depends on where the caller happened to be standing,
#   2. reads go through `long_path`, which prefixes an absolute path with `\\?\`
#      when it would breach MAX_PATH, opting that call out of the limit.
#
# Without this, a relative `--root` silently produced runs whose profiling and
# token axes were null while an absolute `--root` on the same data did not.

def long_path(path: Path) -> Path:
    """Opt an absolute path out of the Windows MAX_PATH limit when needed."""
    if os.name != "nt":
        return path
    text = str(path)
    if len(text) < 240 or text.startswith("\\\\?\\"):
        return path
    if text.startswith("\\\\"):  # UNC share: \\server\share -> \\?\UNC\server\share
        return Path("\\\\?\\UNC" + text[1:])
    return Path("\\\\?\\" + text)


def safe_read_text(path: Path, encoding: str = "utf-8", errors: str = "ignore") -> str:
    """`Path.read_text` that survives a path longer than MAX_PATH."""
    try:
        return path.read_text(encoding=encoding, errors=errors)
    except OSError:
        return long_path(path).read_text(encoding=encoding, errors=errors)


def resolve_root(root: Path) -> Path:
    """Absolute form of `--root`, so path length never depends on the cwd."""
    return Path(os.path.abspath(root))


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(safe_read_text(path))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        lines = safe_read_text(path).splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def load_list(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON file that is a *list* of records (`solutions.json`)."""
    if not path.exists():
        return []
    try:
        value = json.loads(safe_read_text(path))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def first_number(*values: Any) -> Optional[float]:
    for value in values:
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None


def first_text(*values: Any) -> Optional[str]:
    """First non-empty string among `values`, trimmed."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def candidate_duration(row: Dict[str, Any]) -> Optional[float]:
    return first_number(row.get("duration_ms"), (row.get("eval_result") or {}).get("duration_ms"))


def is_passing_row(row: Dict[str, Any]) -> bool:
    """Was this evaluation a correctness pass?

    The native harness pairs `status=passed` with `correctness_passed=true` and
    reports failures under their own status (`test_failed`, `compile_failed`,
    `profiling_failed`). Some exports carry only one of the two signals, so a row
    with either is accepted — but a row that explicitly says
    `correctness_passed=false` never is.
    """
    if row.get("correctness_passed") is False:
        return False
    if row.get("correctness_passed") is True:
        return True
    return row.get("status") == "passed"


def is_valid_duration(value: Optional[float]) -> bool:
    return value is not None and value > 0


def is_run_dir(path: Path) -> bool:
    """Does this directory hold one operator run?

    `result.json` is the unambiguous marker. A run killed before its summary was
    written still carries the per-evaluation rows, so an `evaluations.jsonl`
    beside the resolved config counts too — the run stays visible in the suite
    instead of silently disappearing from it.
    """
    if (path / "result.json").exists():
        return True
    return (path / "evaluations.jsonl").exists() and (
        (path / "config.resolved.json").exists() or (path / "manifest.json").exists()
    )


def discover_run_dirs(root: Path) -> List[Path]:
    """Every run directory at or below `root`, in a deterministic order.

    The native harness nests runs one level deeper than the legacy layout
    (`<root>/<batch-tag>/<run-id>/`), so the walk is recursive. It stops as soon
    as a directory *is* a run, which also keeps a nested copy from being counted
    twice.
    """
    if not root.exists():
        return []
    runs: List[Path] = []

    def walk(directory: Path, depth: int) -> None:
        if is_run_dir(directory):
            runs.append(directory)
            return
        if depth >= MAX_RUN_DEPTH:
            return
        for child in sorted(directory.iterdir()):
            if child.is_dir() and child.name not in SKIP_DIR_NAMES and not child.name.startswith("."):
                walk(child, depth + 1)

    walk(root, 0)
    return runs


# --------------------------- dataset file naming ---------------------------
#
# The dashboard reads one summary file per (agent × engine model) pair from
# `src/data/eval_datasets_summary/`. That folder's naming convention is what
# this section reproduces, so a freshly generated file is picked up by the UI
# without anyone hand-renaming it:
#
#   claude-code × DeepSeek-V4-Pro → claude-code-deepseek-v4-pro.json
#   k-search    × Qwen3.5-122B    → k-search-qwen3-5-122b.json
#   world-model × GPT-5.6-Sol     → world-model-gpt-5-6-sol.json
#
# The rule is "lowercase, every run of non-alphanumerics becomes a single
# hyphen": the version dot in `Qwen3.5-122B` is a separator just like the
# hyphen already in `DeepSeek-V4-Pro`, which is why the pair reads
# `qwen3-5-122b` and not `qwen3.5-122b`.

def slugify(value: Any) -> str:
    """Lowercase, hyphen-separated form of an id, safe on every filesystem."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def dataset_filename(agent_id: Any, model_id: Any, suffix: str = ".json") -> str:
    """`{agent}-{model}.json`, degrading gracefully when an id is unknown.

    A run whose model could not be read is still a valid dataset, so the model
    segment is simply dropped (`k-search.json`) rather than inventing a name.
    Only when *both* halves are unknown does the file fall back to a neutral
    stem, which keeps the caller honest: an `unidentified.json` in the folder is
    a visible prompt to fix the harness, not a plausible-looking fake pair.
    """
    parts = [part for part in (slugify(agent_id), slugify(model_id)) if part]
    return f"{'-'.join(parts) or 'unidentified'}{suffix}"


def identity_of(run: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(agent id, engine model id) for one evaluated run — the bucketing key."""
    identity = run.get("identity") or {}
    return (
        ((identity.get("agent") or {}).get("id")),
        ((identity.get("engine_model") or {}).get("id")),
    )


def bucket_runs_by_identity(runs: Iterable[Dict[str, Any]]) -> Dict[Tuple[Optional[str], Optional[str]], List[Dict[str, Any]]]:
    """Group runs into one bucket per (agent × engine model) pair.

    Nothing is dropped: a run whose identity could not be resolved still gets a
    bucket (keyed on `None`s), so an empty or mislabelled dataset file is
    reported instead of silently disappearing from the suite.
    """
    buckets: Dict[Tuple[Optional[str], Optional[str]], List[Dict[str, Any]]] = {}
    for run in runs:
        buckets.setdefault(identity_of(run), []).append(run)
    return buckets


def load_run_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Per-evaluation rows: every baseline / candidate / final verification.

    `solutions.json` and `evaluations.jsonl` are the same records in the native
    layout (a JSON array and its JSONL stream); the legacy layout used
    `candidates.jsonl` / `solutions.jsonl`. First non-empty reader wins.
    """
    for name in ("solutions.json", "evaluations.jsonl", "candidates.jsonl", "solutions.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        rows = load_list(path) if name.endswith(".json") else read_jsonl(path)
        if rows:
            return rows
    return []


def load_trace_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Backwards-compatible alias for the row reader."""
    return load_run_rows(run_dir)


def run_usage(result: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """The run-level usage block: `result.json` first, then `usage.json`."""
    usage = result.get("usage")
    if isinstance(usage, dict) and usage:
        return usage
    fallback = load_json(run_dir / "usage.json")
    return fallback if fallback else {}


def run_elapsed_seconds(result: Dict[str, Any]) -> Optional[float]:
    """Total wall-clock seconds the harness spent on this run."""
    return first_number(result.get("elapsed_seconds"), result.get("total_run_duration_seconds"))


def row_observed_tokens(row: Dict[str, Any]) -> Optional[float]:
    """A row's cumulative token cost, from whichever axis the harness filled.

    Order of preference:

      1. `cumulative_tokens` — the legacy explicit axis,
      2. `usage_checkpoint.observed.total_tokens`,
      3. the sum of the observed components that are present.

    The native adapters usually fill *components* rather than a total. Summing
    `uncached_input + cache_read + output` reproduces the run's own
    `usage.total.total_tokens` exactly (verified against every run in the
    archive), so it reconstructs the measured total instead of estimating it.
    Only `input_tokens` is deliberately excluded: when a checkpoint reports both
    it and `uncached_input_tokens` they are the same quantity, and adding both
    would double-count.

    A checkpoint that carries only `uncached_input_tokens` therefore still yields
    a lower bound — every alternative is null in that case, and a stated lower
    bound is more useful than an absent axis. A checkpoint with nothing at all
    stays `None`, so the axis is reported unavailable rather than fabricated.
    """
    observed = (row.get("usage_checkpoint") or {}).get("observed") or {}
    explicit = first_number(row.get("cumulative_tokens"), observed.get("total_tokens"))
    if explicit is not None:
        return explicit
    components = [
        safe_float(observed.get(name))
        for name in ("uncached_input_tokens", "cache_read_tokens", "output_tokens")
    ]
    present = [value for value in components if value is not None]
    return sum(present) if present else None


# ----------------------- Codex rollout token axis ---------------------------
#
# Some adapters do not stamp a token counter onto each evaluation. The codex CLI
# is one: it reports a single run-level total, so every candidate's
# `usage_checkpoint` is null and `STS_token` cannot be computed from the run
# summary alone.
#
# The CLI does, however, keep its own session log. Under
# `.per-run/<run-id>/codex-home/sessions/<Y>/<M>/<D>/rollout-*.jsonl` it writes
# a `token_usage_record` for every model response, carrying
# `thread_token_usage.total_tokens` (the session cumulative) and a timestamp.
# Those records are the missing axis.
#
# The milestone timestamps need no cross-source guessing: both the rollout's
# `timestamp` and the harness' `evaluation_started.timestamp_utc` are UTC, and
# the last record at or before an evaluation is that evaluation's cost.
#
# Two caveats are handled rather than papered over:
#
#   * A run may own several rollouts (a failed attempt plus a resumed one), each
#     numbering its tokens from zero. Only the rollout overlapping the
#     evaluation window feeds the run's total, so the window filter also picks
#     the right file. The winner is confirmed by requiring its final cumulative
#     value to equal the run's own `usage.total.total_tokens`; without that
#     match the rollout is discarded rather than trusted.
#   * The resulting curve is *time-aligned*, not submission-aligned: a point is
#     the cumulative spend when an evaluation began, which includes any thinking
#     between the previous evaluation and this one. The marker recorded on the
#     run says so.

ROLLOUT_TOKEN_TAG = '"token_usage_record"'


def parse_utc(text: Any) -> Optional[float]:
    """ISO-8601 UTC timestamp → epoch seconds, or None when unreadable."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def codex_rollout_dirs(run_dir: Path, root: Path) -> List[Path]:
    """Per-run codex CLI homes that may hold this run's session logs."""
    homes: List[Path] = []
    for base in (run_dir.parent, root):
        per_run = base / ".per-run"
        if not per_run.is_dir():
            continue
        try:
            entries = sorted(per_run.iterdir())
        except OSError:
            continue
        for entry in entries:
            # A retry re-runs a run, so `<run-id>-r2` belongs to the same task;
            # `startswith` is what ties those back to the run.
            if entry.is_dir() and entry.name.startswith(run_dir.name):
                homes.append(entry / "codex-home" / "sessions")
    return homes


def read_rollout_token_series(run_dir: Path, root: Path) -> List[Tuple[float, float, float]]:
    """[(epoch seconds, cumulative total tokens, evaluation_id)] for a run.

    Every candidate is attributed the cumulative spend *as of its own
    evaluation*, chosen from the single rollout that overlaps the run's
    evaluation window.
    """
    # 1. the run's evaluation milestones, in time order
    milestones: List[Tuple[float, str]] = []
    for event in _iterate_native_events(run_dir):
        if event.get("event_type") != "evaluation_started":
            continue
        stamp = parse_utc(event.get("timestamp_utc"))
        if stamp is not None:
            milestones.append((stamp, str(event.get("evaluation_id"))))
    if not milestones:
        return []
    milestones.sort()
    window_start, window_end = milestones[0][0], milestones[-1][0]

    # 2. the rollout whose token records fall inside that window
    sessions = [sessions for home in codex_rollout_dirs(run_dir, root) for sessions in [home] if home.is_dir()]
    series: List[Tuple[float, float]] = []
    for sessions_dir in sessions:
        try:
            rollouts = sorted(sessions_dir.rglob("rollout-*.jsonl"))
        except OSError:
            continue
        for rollout in rollouts:
            candidate: List[Tuple[float, float]] = []
            try:
                with long_path(rollout).open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        if ROLLOUT_TOKEN_TAG not in line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") != "token_usage_record":
                            continue
                        payload = event.get("payload") or {}
                        stamp = parse_utc(event.get("timestamp"))
                        total = safe_float((payload.get("thread_token_usage") or {}).get("total_tokens"))
                        if stamp is not None and total is not None:
                            candidate.append((stamp, total))
            except OSError:
                continue
            if not candidate:
                continue
            inside = [point for point in candidate if window_start <= point[0] <= window_end]
            # The rollout that actually drove this run is the one covering the
            # evaluation window; a stale sibling attempt covers a different one.
            if len(inside) > len(series):
                series = candidate
    if not series:
        return []
    series.sort()

    # 3. attribute each milestone the latest record at or before it
    out: List[Tuple[float, float, float]] = []
    cursor = 0
    for stamp, evaluation_id in milestones:
        while cursor + 1 < len(series) and series[cursor + 1][0] <= stamp:
            cursor += 1
        if series[cursor][0] <= stamp:
            out.append((stamp, series[cursor][1], evaluation_id))
    return out


def codex_rollout_tokens(run_dir: Path, root: Path, run_total: Optional[float]) -> Tuple[Dict[str, float], Optional[str]]:
    """{evaluation_id: cumulative tokens} for a codex run, validated.

    The series is only accepted when its last cumulative value equals the run's
    own `usage.total.total_tokens`. A mismatch means the rollout belongs to a
    different attempt, or the log is partial, so nothing is returned — a wrong
    axis is worse than an absent one.
    """
    series = read_rollout_token_series(run_dir, root)
    if not series:
        return {}, None
    if run_total is not None:
        final = series[-1][1]
        if final != run_total:
            return {}, None
    return {evaluation_id: total for _, total, evaluation_id in series}, "codex_rollout:time_aligned"


# ------------------------- generating agent / model -------------------------
#
# Who produced the operator is not a metric, but a leaderboard is meaningless
# without it: the same speedup from different agents is a different result.
# Both facts are on disk, just spread across the run's config, its usage block
# and its launch logs:
#
#   * the agent:   the run config's `agent.kind`, plus the wrapper a `command`
#                  agent invokes (that wrapper is what names e.g. opencode).
#   * the model:   whichever of the sources below is present, most
#                  authoritative first (see `engine_model_identity`).

# `agent.kind` → leaderboard identity. `command` kinds are resolved through the
# wrapper filename instead, so a new command-driven agent needs no edit here.
AGENT_ALIASES = {
    "claude": {"id": "claude-code", "type": "general", "strategy": "agentic-search"},
    "codex": {"id": "codex", "type": "general", "strategy": "agentic-search"},
    "opencode": {"id": "opencode", "type": "general", "strategy": "agentic-search"},
}

# Model ids the harness reports are deployment tags, not product names.
# `local/qwen3.5-122b-a10b` is what the gateway served; the canonical name is
# what a reader recognises. Unknown tags keep their own string so a new model is
# never silently relabelled as something it is not.
#
# The MoE active-parameter suffix is folded away (`qwen3.5-122b-a10b` →
# `Qwen3.5-122B`), matching the names the dashboard's dataset folder already
# uses; the raw tag survives on the run as `engine_model.raw`.
MODEL_ALIASES = {
    "qwen3.5-122b-a10b": "Qwen3.5-122B",
    "qwen3.6-35b-a3b": "Qwen3.6-35B",
}

# Presentation metadata for canonical model ids (vendor / openness). Kept
# minimal and explicit; unknown models report null rather than a guess.
MODEL_CATALOG = {
    "Qwen3.5-122B": {"vendor": "Alibaba", "openness": "open"},
    "Qwen3.6-35B": {"vendor": "Alibaba", "openness": "open"},
}

# Tokens that name the harness / agent rather than the model, used to find where
# a run id's model segment begins when read from the right.
RUN_ID_MARKER_TOKENS = frozenset({
    "agent", "claude", "claude-code", "codex", "harness", "native", "opencode", "command",
})

# A model tag's leading product family. Checked after the run-id scan, so a
# trailing version suffix is never mistaken for a model.
MODEL_FAMILY_RE = re.compile(
    r"^(?:qwen|qwq|qvq|tongyi|gpt|o[1-9]|deepseek|kimi|moonshot|glm|claude|gemini|llama|mistral)",
    re.IGNORECASE,
)


def config_candidates(run_dir: Path, root: Path) -> List[Path]:
    """Every config-ish document that may describe this run, best first.

    The native harness writes `config.resolved.json` and `manifest.json` inside
    the run directory; the legacy layout kept one document *next to* it, named
    after the run.
    """
    documents = [
        run_dir / "config.resolved.json",
        run_dir / "manifest.json",
        run_dir / "workspace" / ".bench" / "input_manifest.json",
        run_dir.parent / f"{run_dir.name}.json",
        root / f"{run_dir.name}.json",
    ]
    return [document for document in documents if document.exists()]


def agent_identity(run_dir: Path, root: Path) -> Dict[str, Any]:
    """Which agent produced this run, from the run's own config."""
    block: Dict[str, Any] = {}
    for document in config_candidates(run_dir, root):
        config = load_json(document)
        candidate = config.get("agent")
        if isinstance(candidate, dict) and candidate:
            block = candidate
            break

    kind = first_text(block.get("kind"))
    command = block.get("command")
    wrapper = Path(str(command[-1])).name if isinstance(command, list) and command else None
    if wrapper is None:
        executable = first_text(block.get("executable"))
        wrapper = Path(executable).name if executable else None

    key = (kind or "").lower()
    if key == "command" and wrapper:
        # `opencode_agent_wrapper.py` → `opencode`
        key = wrapper.split("_")[0].lower() or key

    known = AGENT_ALIASES.get(key)
    if known:
        return {**known, "kind": kind, "wrapper": wrapper}
    return {"id": key or None, "type": "general", "strategy": None, "kind": kind, "wrapper": wrapper}


def normalise_model_tag(raw: Any) -> Optional[str]:
    """Gateway tag → canonical model id; unknown tags keep their own string."""
    text = first_text(raw)
    if not text:
        return None
    tag = text.split("/")[-1]
    return MODEL_ALIASES.get(tag.lower(), tag)


def model_identity_from(raw: Any) -> Tuple[Optional[str], Optional[str]]:
    """(canonical id, raw gateway tag) for one model statement."""
    text = first_text(raw)
    if not text:
        return None, None
    return normalise_model_tag(text), text


def _model_from_agent_block(run_dir: Path, root: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """`agent.model` in any config document — the opencode adapter sets it."""
    for document in config_candidates(run_dir, root):
        config = load_json(document)
        block = config.get("agent")
        model = first_text(block.get("model")) if isinstance(block, dict) else None
        if model is None:
            model = first_text(config.get("model"))
        canonical, raw = model_identity_from(model)
        if canonical:
            return canonical, raw, f"{document.name}:agent.model"
    return None, None, None


def _model_from_usage(run_dir: Path, result: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """`usage.models_observed`, which the claude adapter fills in.

    A run that genuinely switched models lists more than one; returning the first
    would silently pick whichever the adapter happened to record first, so the
    list is handed back for the caller to disambiguate by served turns.
    """
    for source, usage in (
        ("result.json", result.get("usage") or {}),
        ("usage.json", load_json(run_dir / "usage.json")),
    ):
        observed = usage.get("models_observed")
        if not isinstance(observed, list):
            continue
        stated = [first_text(entry) for entry in observed]
        unique = sorted({tag for tag in (normalise_model_tag(entry) for entry in stated) if tag})
        if len(unique) == 1:
            return unique[0], next((entry for entry in stated if entry), None), f"{source}:usage.models_observed"
        if len(unique) > 1:
            winner = _dominant_model_from_events(run_dir, unique)
            if winner:
                raw = next((entry for entry in stated if normalise_model_tag(entry) == winner), None)
                return winner, raw, f"{source}:dominant_turns"
    return None, None, None


def _dominant_model_from_events(run_dir: Path, candidates: List[str]) -> Optional[str]:
    """Break a multi-model tie by counting the turns each model actually served.

    Only the claude adapter stamps a model onto each assistant event; for the
    others there is no tie to break, so the answer is `None` rather than a coin
    flip on list order.
    """
    counts: Dict[str, int] = {}
    for event in _iterate_native_events(run_dir):
        if event.get("event_type") != "native_event":
            continue
        payload = event.get("payload") or {}
        message = payload.get("message")
        if not isinstance(message, dict):
            continue
        tag = normalise_model_tag(message.get("model"))
        if tag:
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _model_from_launch_log(run_dir: Path, root: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """`native agent model: <tag>` — the harness' own statement of what it ran."""
    for log in (run_dir.parent / f"{run_dir.name}.launch.log", root / f"{run_dir.name}.launch.log"):
        if not log.exists():
            continue
        try:
            text = safe_read_text(log)
        except OSError:
            continue
        for line in text.splitlines():
            match = re.search(r"native agent model:\s*([^\s（(]+)", line)
            if match:
                canonical, raw = model_identity_from(match.group(1))
                if canonical:
                    return canonical, raw, f"{log.name}:native agent model"
    return None, None, None


def _model_from_codex_descriptor(run_dir: Path, root: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Codex materialises its endpoint config per run under `.per-run/`."""
    for base in (run_dir.parent, root):
        per_run = base / ".per-run"
        if not per_run.is_dir():
            continue
        for entry in sorted(per_run.iterdir()):
            if not entry.is_dir() or not entry.name.startswith(run_dir.name):
                continue
            descriptor = load_json(entry / "model.json")
            stated = descriptor.get("name") or descriptor.get("key")
            if stated is None:
                toml = entry / "codex-home" / "config.toml"
                if toml.exists():
                    match = re.search(r'^\s*model\s*=\s*"([^"]+)"', safe_read_text(toml), re.M)
                    stated = match.group(1) if match else None
                descriptor_name = f"{entry.name}/codex-home/config.toml"
            else:
                descriptor_name = f"{entry.name}/model.json"
            canonical, raw = model_identity_from(stated)
            if canonical:
                return canonical, raw, descriptor_name
    return None, None, None


def _model_from_model_turns(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Legacy `model_turns/*.json` tally — most-served model wins."""
    counts: Dict[str, int] = {}
    turns_dir = run_dir / "model_turns"
    if turns_dir.is_dir():
        for turn_path in sorted(turns_dir.glob("*.json")):
            raw = load_json(turn_path).get("raw") or {}
            served = raw.get("model")
            if isinstance(served, str) and served.strip():
                counts[served.strip()] = counts.get(served.strip(), 0) + 1
            usage = raw.get("modelUsage")
            if isinstance(usage, dict):
                for key in usage:
                    if isinstance(key, str) and key.strip():
                        counts[key.strip()] = counts.get(key.strip(), 0) + 1
    if not counts:
        return None, None, None
    raw_id = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    canonical, raw = model_identity_from(raw_id)
    return canonical, raw, "model_turns"


def _model_from_run_id(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Last resort: the harness mints run ids as `<op>-<agent>-<model>-<stamp>`.

    The model segment is read from the right, because the operator name in front
    of it is arbitrary (`add_rms_norm-codex-native-codex-qwen3.5-122b-a10b-…`):
    tokens are collected backwards from the stamp until one names the harness or
    the agent, and the result is only accepted when it starts with a model
    family. A run id that does not follow the convention yields nothing.
    """
    tokens = run_dir.name.split("-")
    if len(tokens) < 4 or not re.fullmatch(r"\d{8}", tokens[-2] or "") or not re.fullmatch(r"\d{6}", tokens[-1] or ""):
        return None, None, None
    collected: List[str] = []
    for token in reversed(tokens[:-2]):
        if token.lower() in RUN_ID_MARKER_TOKENS:
            break
        collected.append(token)
    candidate = "-".join(reversed(collected))
    if not candidate or not MODEL_FAMILY_RE.match(candidate):
        return None, None, None
    canonical, raw = model_identity_from(candidate)
    return canonical, raw, "run_id"


def engine_model_identity(run_dir: Path, root: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    """Which model served the agent, with the evidence for that answer recorded.

    Sources are tried most authoritative first and the winner is named in
    `model_source`, so a reviewer can tell a read fact from a name parsed out of
    a directory label:

      1. `agent.model` in the resolved config (opencode records it explicitly),
      2. `usage.models_observed` in result.json / usage.json (claude does),
      3. the harness' `native agent model:` launch line (codex does),
      4. the codex per-run model descriptor under `.per-run/`,
      5. the legacy `model_turns/*.json` served-model tally,
      6. the run id, which the harness mints from the very pair we are trying to
         recover.

    A run no source can name reports `id: null` rather than a plausible guess.
    """
    for reader in (
        lambda: _model_from_agent_block(run_dir, root),
        lambda: _model_from_usage(run_dir, result),
        lambda: _model_from_launch_log(run_dir, root),
        lambda: _model_from_codex_descriptor(run_dir, root),
        lambda: _model_from_model_turns(run_dir),
        lambda: _model_from_run_id(run_dir),
    ):
        canonical, raw, source = reader()
        if canonical:
            meta = MODEL_CATALOG.get(canonical, {})
            return {
                "id": canonical,
                "raw": raw,
                "vendor": meta.get("vendor"),
                "openness": meta.get("openness"),
                "model_source": source,
            }
    return {"id": None, "raw": None, "vendor": None, "openness": None, "model_source": None}


def run_identity(run_dir: Path, root: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "agent": agent_identity(run_dir, root),
        "engine_model": engine_model_identity(run_dir, root, result),
    }


def valid_baseline_row(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("evaluation_kind") == "baseline" and is_passing_row(row):
            duration = candidate_duration(row)
            if is_valid_duration(duration):
                return row
    return None


def baseline_duration(result: Dict[str, Any], rows: List[Dict[str, Any]]) -> Optional[float]:
    row = valid_baseline_row(rows)
    if row is not None:
        return candidate_duration(row)
    # The run summary is accepted only when a valid baseline row cannot be
    # archived; this preserves compatibility with result-only exports. Both the
    # native (`baseline.duration_ms`) and flat (`baseline_duration_ms`) shapes are
    # recognised, as is the legacy nested `durations.BaselineMs`.
    value = first_number(
        (result.get("baseline") or {}).get("duration_ms"),
        result.get("baseline_duration_ms"),
        (result.get("durations") or {}).get("BaselineMs"),
    )
    return value if is_valid_duration(value) else None


def passed_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("evaluation_kind") == "candidate" and is_passing_row(row) and is_valid_duration(candidate_duration(row))
    ]


def completed_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("status") not in (None, "")]


def median_positive(values: Iterable[Any]) -> Optional[float]:
    parsed = [value for value in (safe_float(item) for item in values) if value is not None and value > 0]
    return float(median(parsed)) if parsed else None


# --------------------------- rounds and tool calls --------------------------
#
# Neither quantity is a headline metric, but both are published ("LLM Rounds",
# "Tool Calls") and the native harness records them only inside each adapter's
# own event stream. Counting therefore has to follow the adapter; the counter
# knows three dialects and reports `None` for anything it does not recognise:
#
#   claude    `assistant` message events            → rounds
#             `tool_use` content blocks             → tool calls
#   codex     completed `agent_message` items       → rounds
#             completed `command_execution` items   → tool calls
#   opencode  `step_finish` records                 → rounds
#             `tool_use` records                    → tool calls
#
# These are the adapters' own turn boundaries, not a re-derivation of them;
# where an adapter exposes no boundary the field stays null.

def _iterate_native_events(run_dir: Path) -> Iterable[Dict[str, Any]]:
    """Stream `events.jsonl` without materialising it (files reach GBs)."""
    path = run_dir / "events.jsonl"
    if not path.exists():
        return
    try:
        with long_path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event
    except OSError:
        return


def _count_claude_interactions(run_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    rounds = 0
    tool_calls = 0
    for event in _iterate_native_events(run_dir):
        if event.get("event_type") != "native_event":
            continue
        payload = event.get("payload") or {}
        if payload.get("type") != "assistant":
            continue
        rounds += 1
        message = payload.get("message") or {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls += 1
    return (rounds or None), (tool_calls or None)


def _count_codex_interactions(run_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    # Codex streams each item twice (started + completed); count unique ids of
    # the completed form only, so a round is not counted twice.
    seen: Dict[str, set] = {"agent_message": set(), "command_execution": set()}
    for event in _iterate_native_events(run_dir):
        if event.get("event_type") != "native_event":
            continue
        payload = event.get("payload") or {}
        if payload.get("type") != "item.completed":
            continue
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") not in seen:
            continue
        identifier = item.get("id") or f"{item.get('type')}:{len(seen[item['type']])}"
        seen[item["type"]].add(identifier)
    return (len(seen["agent_message"]) or None), (len(seen["command_execution"]) or None)


def _count_opencode_interactions(run_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    path = run_dir / "agent.stdout.jsonl"
    if not path.exists():
        return None, None
    rounds = 0
    tool_calls = 0
    try:
        with long_path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "step_finish":
                    rounds += 1
                elif kind == "tool_use":
                    tool_calls += 1
    except OSError:
        return None, None
    return (rounds or None), (tool_calls or None)


def interaction_counts(run_dir: Path, agent_id: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """(LLM rounds, tool calls) as the run's own adapter recorded them."""
    counters = {
        "claude-code": _count_claude_interactions,
        "codex": _count_codex_interactions,
        "opencode": _count_opencode_interactions,
    }
    counter = counters.get(agent_id or "")
    return counter(run_dir) if counter else (None, None)


def result_metrics(result: Dict[str, Any], rows: List[Dict[str, Any]], run_dir: Path, agent_id: Optional[str]) -> Dict[str, Any]:
    usage = run_usage(result, run_dir)
    totals = usage.get("total") or {}
    if not totals:
        # Legacy flat export: the usage block sits directly on result.json.
        totals = result.get("token_usage") or {}

    cumulative_tokens = [safe_float(row.get("cumulative_tokens")) for row in rows]
    observed_tokens = [row_observed_tokens(row) for row in rows]
    elapsed = [safe_float(row.get("solution_elapsed_sec")) for row in rows]
    positive = [value for value in observed_tokens if value is not None]
    usable = [value for value in cumulative_tokens if value is not None]

    reported_total = safe_float(totals.get("total_tokens"))
    trajectory_total = max(
        (value for value in (max(positive, default=None), max(usable, default=None)) if value is not None),
        default=None,
    )
    # A cumulative run total cannot be smaller than a checkpoint observed
    # during that run.  A few archived retries have a stale/zero final usage
    # block while their candidate trajectory retains the larger measured
    # counter.  Taking the maximum preserves the monotonic accounting invariant
    # without estimating anything; both inputs are native measurements.
    total_tokens = max(
        (value for value in (reported_total, trajectory_total) if value is not None),
        default=None,
    )
    total_wall_clock = first_number(
        run_elapsed_seconds(result),
        result.get("total_run_duration_seconds"),
        max((value for value in elapsed if value is not None), default=None),
    )
    search_wall_clock = first_number(result.get("total_search_duration_seconds"), max((value for value in elapsed if value is not None), default=None))
    rounds, tool_calls = interaction_counts(run_dir, agent_id)
    row_rounds = [value for value in (safe_float(row.get("llm_rounds_cumulative")) for row in rows) if value is not None]
    row_tools = [value for value in (safe_float(row.get("tool_calls_cumulative")) for row in rows) if value is not None]
    llm_rounds = first_number(
        result.get("llm_rounds"),
        rounds,
        max(row_rounds, default=None),
    )
    tool_calls_total = first_number(
        result.get("tool_calls"),
        tool_calls,
        max(row_tools, default=None),
    )
    metrics = {
        "TotalTokens": total_tokens,
        "TotalWallClockSec": total_wall_clock,
        "SearchWallClockSec": search_wall_clock,
        "LLMRounds": llm_rounds,
        "ToolCalls": tool_calls_total,
        # Token breakdown, so a pricing table can be applied downstream:
        # cost = input*uncached + cache_read*discount + cache_write*write + output*out
        "InputTokens": safe_float(totals.get("uncached_input_tokens") if totals.get("uncached_input_tokens") is not None else totals.get("input_tokens")),
        "CachedInputTokens": safe_float(totals.get("cache_read_tokens")),
        "CacheWriteTokens": safe_float(totals.get("cache_write_tokens")),
        "OutputTokens": safe_float(totals.get("output_tokens")),
        "UsageAvailable": usage.get("available"),
    }
    if trajectory_total is not None and (reported_total is None or trajectory_total > reported_total):
        metrics["cost_source"] = "candidate_usage_checkpoint:max"
    return metrics


def best_curve(rows: List[Dict[str, Any]], axis: str, baseline: float, value_of=None) -> Optional[List[Tuple[float, float]]]:
    """Best-so-far (budget, duration) curve over the rows that passed.

    `value_of` overrides how the x axis is read, so the token axis can fall back
    to the harness' observed counters without a second copy of this integration.
    """
    read = value_of or (lambda row, name: safe_float(row.get(name)))
    points: List[Tuple[float, float]] = []
    for row in rows:
        x = read(row, axis)
        duration = candidate_duration(row)
        if is_passing_row(row) and x is not None and x > 0 and is_valid_duration(duration):
            points.append((x, duration))
    if not points:
        return None
    points.sort(key=lambda item: item[0])
    deduplicated: Dict[float, float] = {}
    for x, duration in points:
        deduplicated[x] = min(duration, deduplicated.get(x, math.inf))
    curve: List[Tuple[float, float]] = []
    best = baseline
    for x, duration in sorted(deduplicated.items()):
        best = min(best, duration)
        curve.append((x, best))
    return curve


def auls(curve: Optional[List[Tuple[float, float]]], baseline: float) -> Optional[float]:
    if not curve:
        return None
    points = [(curve[0][0], baseline)] + curve
    deduplicated: Dict[float, float] = {}
    for x, duration in points:
        deduplicated[x] = min(duration, deduplicated.get(x, math.inf))
    points = sorted(deduplicated.items())
    if len(points) < 2 or points[0][0] <= 0 or points[-1][0] <= points[0][0]:
        return None
    denominator = math.log(points[-1][0] / points[0][0])
    if denominator <= 0:
        return None
    area = 0.0
    for previous, current in zip(points, points[1:]):
        x0, duration0 = previous
        x1, duration1 = current
        weight = math.log(x1 / x0) / denominator
        gain0 = math.log2(baseline / duration0)
        gain1 = math.log2(baseline / duration1)
        area += weight * (gain0 + gain1) / 2.0
    return area


def vlr(baseline: float, duration: float) -> float:
    return 1.0 - (duration / baseline)


def target_value(rows: List[Dict[str, Any]], baseline: float, axis: str, target: float, value_of=None) -> Optional[float]:
    read = value_of or (lambda row, name: safe_float(row.get(name)))
    values: List[Tuple[float, float]] = []
    for row in rows:
        axis_value = read(row, axis)
        duration = candidate_duration(row)
        if is_passing_row(row) and axis_value is not None and axis_value > 0 and is_valid_duration(duration):
            values.append((axis_value, duration))
    for axis_value, duration in sorted(values):
        if vlr(baseline, duration) >= target:
            return axis_value
    return None


def target_suffix(target: float) -> str:
    percent = target * 100.0
    return f"{percent:g}%"


def empty_sts(targets: Tuple[float, ...], reason: str) -> Dict[str, Any]:
    sts: Dict[str, Any] = {
        "status": "unavailable",
        "reason": reason,
        "Speedup": None,
        "VLR": None,
        "EndGain": None,
        "AULS_token": None,
        "AULS_time": None,
        "STS_token": None,
        "STS_time": None,
        "S_equiv": None,
    }
    for target in targets:
        suffix = target_suffix(target)
        sts[f"TokensTo{suffix}"] = None
        sts[f"TimeTo{suffix}"] = None
    return sts


def verification_metrics(result: Dict[str, Any], rows: List[Dict[str, Any]], baseline: float) -> Dict[str, Any]:
    """Final re-profile statistics.

    Two shapes are accepted. `final_verification.evaluations[]` is the native
    one: each entry re-evaluates the winning solution, so its `duration_ms` is a
    sample and the entries are counted directly. The legacy shape carried
    pre-aggregated `samples_ms` / `passed` / `completed` counters on the block.
    """
    verification = result.get("final_verification") or {}
    entries = verification.get("evaluations")
    if not isinstance(entries, list) or not entries:
        entries = [row for row in rows if row.get("evaluation_kind") == "final_verification"]

    samples = [value for value in (safe_float(candidate_duration(entry)) for entry in entries) if value is not None and value > 0]
    requested = safe_float(verification.get("requested"))
    completed = first_number(len(entries) or None, verification.get("completed"))
    passed = first_number(
        sum(1 for entry in entries if is_passing_row(entry)) if entries else None,
        verification.get("passed"),
    )
    # A confirmation is trusted only when every requested repeat ran and matched:
    # fewer valid samples than requested means the median is not confirmed.
    valid = verification.get("valid")
    if valid is None:
        valid = completed is not None and requested is not None and completed >= requested
    confirmed = (
        median_positive(samples)
        if valid is True and passed is not None and completed is not None and passed == completed and samples
        else None
    )
    mean = sum(samples) / len(samples) if samples else None
    std = math.sqrt(sum((value - mean) ** 2 for value in samples) / len(samples)) if samples and mean is not None else None
    return {
        "VerificationPassRate": (passed / completed) if passed is not None and completed else None,
        "VerificationMedianMs": median_positive(samples),
        "VerificationCV": (std / mean) if std is not None and mean else None,
        "SpeedupConfirmed": (baseline / confirmed) if confirmed else None,
        "VLRConfirmed": vlr(baseline, confirmed) if confirmed else None,
    }


def discover_recipe(run_dir: Path, root: Path) -> Optional[Path]:
    paths = [
        run_dir / "recipe.json",
        root / "recipes" / f"{run_dir.name}.json",
        root / "recipes" / "recipe.json",
        run_dir.parent / "recipes" / f"{run_dir.name}.json",
    ]
    return next((path for path in paths if path.exists()), None)


def profile_csv_paths(run_dir: Path, evaluation_id: Any) -> List[Path]:
    """`op_summary_*.csv` for one evaluation, in either layout.

    Native runs keep the profile under `evaluations/<EID>/` and the legacy
    harness under `profiling/<EID>/`; both nest the CSV inside `profile_output/`.
    """
    if not evaluation_id:
        return []
    paths: List[Path] = []
    for artifact_root in ("evaluations", "profiling"):
        profile_dir = run_dir / artifact_root / str(evaluation_id) / "profile_output"
        if profile_dir.is_dir():
            paths.extend(sorted(profile_dir.glob("**/op_summary_*.csv")))
    return paths


def profile_summary(run_dir: Path, evaluation_id: Any, duration_ms: Optional[float]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": "unavailable",
        "evaluation_id": evaluation_id,
        "ActiveTimeUs": None,
        "S_i_aicore": None,
        "U_i": None,
        "VecRatio": None,
        "CubeRatio": None,
        "MTE2Ratio": None,
    }
    if not evaluation_id:
        summary["reason"] = "missing_evaluation_id"
        return summary
    csv_paths = profile_csv_paths(run_dir, evaluation_id)
    if not csv_paths:
        summary["reason"] = "missing_op_summary_csv"
        return summary
    rows: List[Dict[str, Any]] = []
    for csv_path in csv_paths:
        try:
            with long_path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
        except (OSError, csv.Error):
            continue
    if not rows:
        summary["reason"] = "empty_op_summary_csv"
        return summary

    def values(*names: str, positive: bool = False) -> List[float]:
        parsed: List[float] = []
        for row in rows:
            candidates = [safe_float(row.get(name)) for name in names]
            candidates = [value for value in candidates if value is not None]
            if positive:
                candidates = [value for value in candidates if value > 0]
            if candidates:
                parsed.append(candidates[0])
        return parsed

    active_values = values("aicore_time(us)", "aiv_time(us)", positive=True)
    active_time = float(median(active_values)) if active_values else None
    task_times = values("Task Duration(us)", "_time(us)", positive=True)
    task_time = float(median(task_times)) if task_times else duration_ms
    vec_ratio = values("aiv_vec_ratio", positive=False)
    cube_ratio = values("cube_utilization(%)", "cube_ratio", positive=False)
    mte2_ratio = values("aic_mte2_ratio", "aiv_mte2_ratio", "mte2_ratio", positive=False)
    summary.update({
        "status": "computed",
        "source": str(csv_paths[0]),
        "samples": len(rows),
        "ActiveTimeUs": active_time,
        "U_i": (active_time / (task_time * 1000.0)) if active_time is not None and task_time and task_time > 0 else None,
        "VecRatio": float(median(vec_ratio)) if vec_ratio else None,
        "CubeRatio": float(median(cube_ratio)) if cube_ratio else None,
        "MTE2Ratio": float(median(mte2_ratio)) if mte2_ratio else None,
    })
    return summary


def recipe_ops(recipe: Dict[str, Any], default_calls: float) -> List[Dict[str, Any]]:
    ops = recipe.get("ops")
    if not isinstance(ops, list) or not ops:
        return [{"calls_per_step": default_calls, "source": "default"}]
    normalized = []
    for op in ops:
        if not isinstance(op, dict):
            normalized.append({"calls_per_step": default_calls, "source": "default"})
            continue
        calls = first_number(op.get("calls_per_step"), op.get("n_i"), default_calls)
        normalized.append({**op, "calls_per_step": calls, "source": "recipe" if first_number(op.get("calls_per_step"), op.get("n_i")) is not None else "default"})
    return normalized


def compute_paos(
    run_dir: Path,
    root: Path,
    baseline: Optional[float],
    best: Optional[float],
    gate: int,
    paos_g: Optional[float] = None,
    paos_batch_size: Optional[float] = None,
    paos_scenario: Optional[str] = None,
    paos_chunk_len: Optional[float] = None,
    paos_t_other_ms: Optional[float] = None,
    paos_calls_per_step: Optional[float] = None,
    baseline_evaluation_id: Any = None,
    best_evaluation_id: Any = None,
) -> Dict[str, Any]:
    recipe_path = discover_recipe(run_dir, root)
    recipe = load_json(recipe_path) if recipe_path is not None else {}
    default_t_other = paos_t_other_ms if paos_t_other_ms is not None else 0.0
    default_calls = paos_calls_per_step if paos_calls_per_step is not None else 1.0
    recipe_scenario = recipe.get("runtime_scenario") or recipe.get("scenario")
    recipe_batch_size = first_number(recipe.get("batch_size"))
    recipe_chunk_len = first_number(recipe.get("chunk_len"))
    scenario = paos_scenario or recipe_scenario
    batch_size = paos_batch_size if paos_batch_size is not None else recipe_batch_size
    chunk_len = paos_chunk_len if paos_chunk_len is not None else recipe_chunk_len
    if paos_g is not None:
        g = paos_g
        g_source = "cli"
    elif safe_float(recipe.get("G")) is not None:
        g = safe_float(recipe.get("G"))
        g_source = "recipe"
    elif scenario == "decode" and batch_size is not None:
        g = batch_size
        g_source = "computed"
    elif scenario == "prefill" and batch_size is not None and chunk_len is not None:
        g = batch_size * chunk_len
        g_source = "computed"
    else:
        g = None
        g_source = "missing"
    recipe_t_other = safe_float(recipe.get("T_other_ms"))
    t_other = paos_t_other_ms if paos_t_other_ms is not None else (recipe_t_other if recipe_t_other is not None else default_t_other)
    ops = recipe_ops(recipe, default_calls)
    if paos_calls_per_step is not None:
        ops = [{**op, "calls_per_step": paos_calls_per_step, "source": "cli"} for op in ops]
    output: Dict[str, Any] = {
        "status": "unavailable",
        "G": g,
        "runtime_scenario": scenario,
        "batch_size": batch_size,
        "chunk_len": chunk_len,
        "T_other_ms": t_other,
        "calls_per_step": [op["calls_per_step"] for op in ops],
        "T_step_base_ms": None,
        "T_step_agent_ms": None,
        "eTPS_base": None,
        "eTPS_agent": None,
        "S_i_lat": None,
        "PTEI": None,
        "PKAI": None,
        "PAOS_base": None,
        "profiling": {
            "baseline": profile_summary(run_dir, baseline_evaluation_id, baseline),
            "agent": profile_summary(run_dir, best_evaluation_id, best),
            "S_i_aicore": None,
            "U_i": None,
            "U_step": None,
            "VecRatio": None,
            "CubeRatio": None,
            "MTE2Ratio": None,
        },
        "input_source": {
            "recipe": str(recipe_path) if recipe_path is not None else "default",
            "G": g_source,
            "runtime_scenario": "cli" if paos_scenario is not None else ("recipe" if recipe_scenario is not None else "missing"),
            "batch_size": "cli" if paos_batch_size is not None else ("recipe" if recipe_batch_size is not None else "missing"),
            "chunk_len": "cli" if paos_chunk_len is not None else ("recipe" if recipe_chunk_len is not None else "missing"),
            "T_other_ms": "cli" if paos_t_other_ms is not None else ("recipe" if recipe_t_other is not None else "default"),
            "calls_per_step": "cli" if paos_calls_per_step is not None else ("recipe" if recipe.get("ops") else "default"),
        },
    }
    if not gate or baseline is None or best is None:
        output["reason"] = "correctness_gate_failed"
        return output
    if g is None:
        output["reason"] = "missing_runtime_G"
        return output
    if (
        g is None
        or g <= 0
        or (batch_size is not None and batch_size <= 0)
        or (chunk_len is not None and chunk_len <= 0)
        or t_other is None
        or t_other < 0
        or any(op["calls_per_step"] < 0 for op in ops)
    ):
        output["reason"] = "invalid_T_other_ms"
        return output
    baseline_step = t_other + sum(op["calls_per_step"] * baseline for op in ops)
    agent_step = t_other + sum(op["calls_per_step"] * best for op in ops)
    if baseline_step <= 0 or agent_step <= 0:
        output["reason"] = "invalid_step_time"
        output["recipe"] = str(recipe_path)
        return output
    speedup = baseline / best
    pkai = math.exp(sum((op["calls_per_step"] * baseline / baseline_step) * math.log(speedup) for op in ops))
    ptei = baseline_step / agent_step
    output.update({
        "status": "computed",
        "recipe": str(recipe_path) if recipe_path is not None else None,
        "G": g,
        "T_step_base_ms": baseline_step,
        "T_step_agent_ms": agent_step,
        "eTPS_base": g / baseline_step,
        "eTPS_agent": g / agent_step,
        "S_i_lat": speedup,
        "PTEI": ptei,
        "PKAI": pkai,
        "PAOS_base": math.sqrt(pkai * ptei),
    })
    baseline_profile = output["profiling"]["baseline"]
    agent_profile = output["profiling"]["agent"]
    baseline_active = safe_float(baseline_profile.get("ActiveTimeUs"))
    agent_active = safe_float(agent_profile.get("ActiveTimeUs"))
    if baseline_active is not None and agent_active is not None and agent_active > 0:
        output["profiling"]["S_i_aicore"] = baseline_active / agent_active
    output["profiling"].update({
        "U_i": agent_profile.get("U_i"),
        "U_step": (sum(op["calls_per_step"] for op in ops) * agent_active / (agent_step * 1000.0)) if agent_active is not None and agent_step > 0 else None,
        "VecRatio": agent_profile.get("VecRatio"),
        "CubeRatio": agent_profile.get("CubeRatio"),
        "MTE2Ratio": agent_profile.get("MTE2Ratio"),
    })
    return output


def compute_run_metrics(
    run_dir: Path,
    root: Path,
    targets: Tuple[float, ...],
    paos_g: Optional[float] = None,
    paos_batch_size: Optional[float] = None,
    paos_scenario: Optional[str] = None,
    paos_chunk_len: Optional[float] = None,
    paos_t_other_ms: Optional[float] = None,
    paos_calls_per_step: Optional[float] = None,
) -> Dict[str, Any]:
    result = load_json(run_dir / "result.json")
    rows = load_run_rows(run_dir)
    identity = run_identity(run_dir, root, result)
    agent_id = (identity.get("agent") or {}).get("id")

    baseline = baseline_duration(result, rows)
    baseline_row = valid_baseline_row(rows)
    passed = passed_candidates(rows)
    best_row = min(passed, key=lambda row: candidate_duration(row) or math.inf) if passed else None
    candidate_rows = [row for row in rows if row.get("evaluation_kind") == "candidate"]
    completed = completed_rows(candidate_rows)
    gate = int(baseline is not None and bool(passed))
    task_pass = int(baseline is not None or bool(passed))
    accuracy: Dict[str, Any] = {
        "TaskPass": task_pass,
        "CandidatePassRate": (len(passed) / len(completed)) if completed else None,
        "CorrectnessGate": gate,
        # The native harness reports `success`; the legacy one used
        # `operational_status`. Either is a statement about the run, not a guess.
        "RunCompleted": int(result.get("success") is True or result.get("operational_status") == "completed"),
        "FailureReason": first_text(result.get("termination_reason"), result.get("error")),
        "CandidatesTotal": len(candidate_rows),
        "CandidatesPassed": len(passed),
    }
    costs = result_metrics(result, rows, run_dir, agent_id)
    # Where a run's own per-candidate counters are empty, the codex CLI session
    # log can still supply the axis; see the rollout section above for the
    # validation that gates it.
    rollout_tokens, rollout_source = codex_rollout_tokens(run_dir, root, safe_float(costs.get("TotalTokens")))
    if rollout_tokens:
        # Some Codex adapters persisted the per-response rollout counters but
        # left `usage.json.total.total_tokens` empty.  In that case the final
        # aligned checkpoint is the run total we actually observed.  Reuse it
        # for the cost summary as well as the STS axis so the dashboard does
        # not show a blank Cumulative Tokens value beside a computed token
        # curve.  Never replace a harness-reported total: when one exists it
        # has already been used above to validate the rollout exactly.
        if costs.get("TotalTokens") is None:
            costs["TotalTokens"] = max(rollout_tokens.values())
        costs["cost_source"] = rollout_source
    best = min((candidate_duration(row) for row in passed), default=None)
    durations: Dict[str, Any] = {
        "BaselineMs": baseline,
        "BestMs": best,
        "Speedup": (baseline / best) if baseline is not None and best else None,
        "BaselineEvaluationId": (baseline_row or {}).get("evaluation_id") or (result.get("baseline") or {}).get("evaluation_id"),
        "BestEvaluationId": (best_row or {}).get("evaluation_id") or (result.get("best_verified") or {}).get("evaluation_id"),
        "BaselineEvidence": "candidate_trace" if baseline_row else ("result_summary" if baseline is not None else None),
    }
    run: Dict[str, Any] = {
        "task_id": run_dir.name,
        "status": "passed" if gate else ("blocked" if baseline is None else "failed"),
        "identity": identity,
        "accuracy": accuracy,
        "cost": costs,
        "durations": durations,
        "sts": empty_sts(targets, "correctness_gate_failed"),
        "paos": compute_paos(
            run_dir,
            root,
            baseline,
            best,
            gate,
            paos_g,
            paos_batch_size,
            paos_scenario,
            paos_chunk_len,
            paos_t_other_ms,
            paos_calls_per_step,
            durations["BaselineEvaluationId"],
            durations["BestEvaluationId"],
        ),
    }
    if baseline is None:
        run["sts"]["reason"] = "missing_valid_baseline"
        return run
    if not passed:
        run["status"] = "failed"
        run["sts"]["reason"] = "no_passed_candidate"
        return run

    # The token axis is the harness' own cumulative counter where it exists; the
    # observed-checkpoint fallback and its reconstruction rule are documented on
    # `row_observed_tokens`. Only the search trajectory is consulted — the final
    # re-profiles repeat the winner and would otherwise fabricate a degenerate
    # axis (codex writes the run total on those rows and nothing on the
    # candidates). When no row carries any token counter the axis stays absent
    # and STS_token reports unavailable.
    search_rows = [row for row in rows if row.get("evaluation_kind") != "final_verification"]

    def token_value(row: Dict[str, Any], _axis: str = "cumulative_tokens") -> Optional[float]:
        own = row_observed_tokens(row)
        if own is not None:
            return own
        return rollout_tokens.get(str(row.get("evaluation_id")))

    token_of = token_value if any(token_value(row) is not None for row in search_rows) else None

    token_curve = best_curve(search_rows, "cumulative_tokens", baseline, token_of)
    time_curve = best_curve(search_rows, "solution_elapsed_sec", baseline)
    speedup = baseline / best
    # `EndGain` is the half of STS_token that needs no token curve: it is
    # `log2 S(B_max)`, the final verified speedup on a log scale. The other half
    # (`AULS`) is a curve integral, so it is the only part a missing token axis
    # can actually take away. Exposing EndGain separates the two, so a run whose
    # adapter never recorded per-candidate tokens still reports the peak-gain
    # term it *does* have instead of presenting an undifferentiated `null`.
    end_gain = math.log2(speedup) if speedup and speedup > 0 else None
    run["sts"] = {
        "status": "computed",
        "Speedup": speedup,
        "VLR": vlr(baseline, best),
        "EndGain": end_gain,
        "AULS_token": auls(token_curve, baseline),
        "AULS_time": auls(time_curve, baseline),
        "SpeedupSearch": speedup,
        "VLRSearch": vlr(baseline, best),
    }
    for target in targets:
        suffix = target_suffix(target)
        run["sts"][f"TokensTo{suffix}"] = target_value(search_rows, baseline, "cumulative_tokens", target, token_of)
        run["sts"][f"TimeTo{suffix}"] = target_value(search_rows, baseline, "solution_elapsed_sec", target)
    run["sts"].update(verification_metrics(result, rows, baseline))
    # `status: computed` describes the block; a null STS_token still needs its own
    # explanation, or a reader sees a computed metric family with a silent gap on
    # the headline number. The reason distinguishes “the adapter recorded no token
    # counter at all” from “tokens exist but too few passing points to integrate”.
    if run["sts"]["AULS_token"] is None:
        run["sts"]["reason"] = "missing_token_axis" if token_of is None else "insufficient_token_points"
    elif rollout_tokens:
        run["sts"]["token_source"] = rollout_source
    run["sts"]["STS_token"] = 100.0 * (0.5 * end_gain + 0.5 * run["sts"]["AULS_token"]) if (end_gain is not None and run["sts"]["AULS_token"] is not None) else None
    run["sts"]["STS_time"] = 100.0 * (0.5 * end_gain + 0.5 * run["sts"]["AULS_time"]) if (end_gain is not None and run["sts"]["AULS_time"] is not None) else None
    run["sts"]["S_equiv"] = 2.0 ** (run["sts"]["STS_token"] / 100.0) if run["sts"]["STS_token"] is not None else None
    run["accuracy"]["BaselineEvidence"] = "candidate_trace" if baseline_row else "result_summary"
    # Real best-so-far trajectory, exported so dashboards plot measured points
    # instead of interpolating a plausible-looking curve.
    run["anytime"] = {
        "BaselineMs": baseline,
        "token": [{"Tokens": x, "BestMs": d, "Speedup": (baseline / d) if d else None} for x, d in (token_curve or [])],
        "time": [{"Seconds": x, "BestMs": d, "Speedup": (baseline / d) if d else None} for x, d in (time_curve or [])],
    }
    return run


def metric_catalog() -> Dict[str, Any]:
    """Friendly metric metadata for dashboards.

    The evaluator is the single source of truth for metric *semantics*; this
    catalog carries the presentation contract (label, short mark, unit,
    direction, plain-language definition) so the UI does not have to invent or
    drift from its own wording.
    """
    return {
        "version": "1.0",
        "categories": [
            {"key": "accuracy", "label": "正确性", "subtitle": "Correctness"},
            {"key": "identity", "label": "生成者", "subtitle": "Generator identity"},
            {"key": "cost", "label": "成本", "subtitle": "Cost & effort"},
            {"key": "sts", "label": "STS · token 效率轴", "subtitle": "Speedup–Token Surface"},
            {"key": "paos", "label": "PAOS · 价值轴", "subtitle": "Profile Agent Optimization Score"},
            {"key": "durations", "label": "延迟", "subtitle": "Latency"},
        ],
        "metrics": {
            # -- identity ---------------------------------------------------
            "Agent": {"label": "Agent", "short": "AGENT", "unit": "", "direction": "context", "category": "identity",
                      "definition": "生成算子的 coding agent 身份，取自 run 配置的 agent.kind（command 型由 wrapper 文件名解析）。"},
            "EngineModel": {"label": "Engine Model", "short": "MODEL", "unit": "", "direction": "context", "category": "identity",
                            "definition": "驱动该 agent 的 LLM，优先取 run 配置的 agent.model，其次 usage.models_observed、harness 启动日志与 run id；具体来源记在 identity.engine_model.model_source。"},
            # -- accuracy ---------------------------------------------------
            "TaskPassRate": {"label": "Task Pass Rate", "short": "TASK %", "unit": "%", "direction": "up", "category": "accuracy",
                             "definition": "至少存在一个有效 baseline 或通过候选的任务占比。"},
            "GateRate": {"label": "Correctness Gate Rate", "short": "GATE %", "unit": "%", "direction": "up", "category": "accuracy",
                         "definition": "通过正确性硬门槛的任务占比；只有这些任务进入 STS / PAOS 性能榜。"},
            "RunCompletedRate": {"label": "Run Completed Rate", "short": "DONE %", "unit": "%", "direction": "up", "category": "accuracy",
                                 "definition": "完整跑完（operational_status=completed）的任务占比。"},
            "TaskPass": {"label": "Task Pass", "short": "PASS", "unit": "", "direction": "up", "category": "accuracy",
                         "definition": "任务级 0/1 标记：存在有效 baseline 或通过候选即为 1。"},
            "CandidatePassRate": {"label": "Candidate Pass Rate", "short": "CAND %", "unit": "%", "direction": "up", "category": "accuracy",
                                  "definition": "通过功能测试的候选数 / 已完成的候选总数。"},
            "CorrectnessGate": {"label": "Correctness Gate", "short": "GATE", "unit": "", "direction": "up", "category": "accuracy",
                                "definition": "任务级硬门槛：存在通过候选才允许计入性能汇总。"},
            "RunCompleted": {"label": "Run Completed", "short": "DONE", "unit": "", "direction": "up", "category": "accuracy",
                             "definition": "运行是否正常结束。"},
            "FailureReason": {"label": "Failure Reason", "short": "REASON", "unit": "", "direction": "context", "category": "accuracy",
                              "definition": "原样保留的终止原因，用于解释失败或阻塞。"},
            # -- cost -------------------------------------------------------
            "TotalTokens": {"label": "Cumulative Tokens", "short": "TOKENS", "unit": "token", "direction": "down", "category": "cost",
                            "definition": "该任务累计消耗的 prompt+completion token，是 STS 的成本横轴。"},
            "median_TotalTokens": {"label": "Median Tokens", "short": "TOKENS", "unit": "token", "direction": "down", "category": "cost",
                                   "definition": "各任务累计 token 的中位数。"},
            "LLMRounds": {"label": "LLM Rounds", "short": "ROUNDS", "unit": "", "direction": "down", "category": "cost",
                          "definition": "Agent 与模型交互的轮数。"},
            "mean_LLMRounds": {"label": "Mean LLM Rounds", "short": "ROUNDS", "unit": "", "direction": "down", "category": "cost",
                               "definition": "各任务 LLM 轮数的算术平均。"},
            "ToolCalls": {"label": "Tool Calls", "short": "TOOLS", "unit": "", "direction": "down", "category": "cost",
                          "definition": "Agent 发起的工具调用次数。"},
            "TotalWallClockSec": {"label": "Wall Time", "short": "WALL", "unit": "s", "direction": "down", "category": "cost",
                                  "definition": "任务端到端 wall-clock 耗时（秒）。"},
            # -- durations --------------------------------------------------
            "BaselineMs": {"label": "Baseline Latency", "short": "L0 µs", "unit": "ms", "direction": "down", "category": "durations",
                           "definition": "优化前基线算子延迟 L0，取自有效 baseline 候选。"},
            "BestMs": {"label": "Best Latency", "short": "LBEST µs", "unit": "ms", "direction": "down", "category": "durations",
                       "definition": "通过候选中的最小延迟 L_best。"},
            # -- sts --------------------------------------------------------
            "EBGS": {"label": "EBGS", "short": "EBGS", "unit": "×", "direction": "up", "category": "sts",
                     "definition": "跨任务专家基线几何平均加速比 exp(mean(ln S_i))，代表峰值收益。"},
            "mean_Speedup": {"label": "Mean Speedup", "short": "SPD", "unit": "×", "direction": "up", "category": "sts",
                             "definition": "各任务已验证加速比 S_i = L0 / L_best 的算术平均。"},
            "median_Speedup": {"label": "Median Speedup", "short": "S~", "unit": "×", "direction": "up", "category": "sts",
                               "definition": "各任务已验证加速比的中位数，比均值更抗离群。"},
            "max_Speedup": {"label": "Max Speedup", "short": "S↑", "unit": "×", "direction": "up", "category": "sts",
                            "definition": "任务集合中的最佳加速比。"},
            "Speedup": {"label": "Speedup", "short": "S", "unit": "×", "direction": "up", "category": "sts",
                        "definition": "已验证加速比 S = L0 / L_best，只统计通过功能测试的解。"},
            "VLR": {"label": "VLR", "short": "VLR", "unit": "%", "direction": "up", "category": "sts",
                    "definition": "Validated Latency Reduction 延迟降低率 1 − L_best / L0。"},
            "EndGain": {"label": "End Gain", "short": "log₂S", "unit": "", "direction": "up", "category": "sts",
                        "definition": "最终收益项 log₂S(B_max)，即 S(B_max)=L0/L_best 的对数；是 STS_token 中不依赖 token 曲线的分量，token 轴缺失时仍可计算。"},
            "AULS_token": {"label": "AULS (token)", "short": "AULS", "unit": "", "direction": "up", "category": "sts",
                           "definition": "对数 token 轴上 best-so-far 加速曲线的归一化面积，衡量“多早拿到收益”。"},
            "AULS_time": {"label": "AULS (time)", "short": "AULS", "unit": "", "direction": "up", "category": "sts",
                          "definition": "对数 wall-clock 轴上的 anytime 面积，衡量时间维度上的效率。"},
            "STS_token": {"label": "STS Token", "short": "STS", "unit": "", "direction": "up", "category": "sts",
                          "definition": "主榜分：100·[0.5·EndGain + 0.5·AULS_token]，融合最终收益与过程效率；任一分量缺失则为 null。"},
            "STS_time": {"label": "STS Time", "short": "STS", "unit": "", "direction": "up", "category": "sts",
                         "definition": "同 STS_token，横轴换成 wall-clock，用于基础设施效率诊断。"},
            "S_equiv": {"label": "S_equiv", "short": "S_EQUIV", "unit": "×", "direction": "up", "category": "sts",
                        "definition": "2^(STS_token/100)，把综合分翻译回可解释的“等效加速比”。"},
            "EIR": {"label": "EIR", "short": "EIR %", "unit": "%", "direction": "up", "category": "sts",
                    "definition": "达到指定 VLR 阈值（1% / 5% / 10%）的任务比例，衡量改进广度。"},
            "TokensTo": {"label": "Tokens-to-Δ", "short": "TO Δ", "unit": "token", "direction": "down", "category": "sts",
                         "definition": "首次达到 VLR ≥ Δ（1% / 5% / 10%）所需的累计 token 数。"},
            "TimeTo": {"label": "Time-to-Δ", "short": "T Δ", "unit": "s", "direction": "down", "category": "sts",
                       "definition": "首次达到 VLR ≥ Δ（1% / 5% / 10%）所需的 wall-clock 秒数。"},
            # -- paos -------------------------------------------------------
            "S_i_lat": {"label": "Kernel Speedup", "short": "S_lat", "unit": "×", "direction": "up", "category": "paos",
                        "definition": "算子延迟加速比 t_base / t_best，是 PAOS 的原子量。"},
            "T_step_base_ms": {"label": "T_step (base)", "short": "T0", "unit": "ms", "direction": "down", "category": "paos",
                               "definition": "优化前 step 总耗时 T_other + Σ(n_i · t_i^0)，按 recipe 的调用次数合成。"},
            "T_step_agent_ms": {"label": "T_step (agent)", "short": "T1", "unit": "ms", "direction": "down", "category": "paos",
                                "definition": "优化后 step 总耗时，只有被优化算子换成了最佳通过候选。"},
            "PTEI": {"label": "PTEI", "short": "PTEI", "unit": "×", "direction": "up", "category": "paos",
                     "definition": "eTPSGain = T_step^0 / T_step^1，step 级 token 吞吐增益。"},
            "PKAI": {"label": "PKAI", "short": "PKAI", "unit": "×", "direction": "up", "category": "paos",
                     "definition": "token-time 加权算子加速指数 exp(Σ w_i·ln S_i^lat)，权重小的算子影响力也小。"},
            "PAOS_base": {"label": "PAOS", "short": "PAOS", "unit": "×", "direction": "up", "category": "paos",
                          "definition": "价值轴主分 √(PKAI × PTEI)：既奖励算得快，也奖励优化在关键路径上。"},
            "eTPS_base": {"label": "eTPS (base)", "short": "eTPS0", "unit": "token/ms", "direction": "up", "category": "paos",
                          "definition": "优化前估算模型 token 吞吐 G / T_step^0。"},
            "eTPS_agent": {"label": "eTPS (agent)", "short": "eTPS1", "unit": "token/ms", "direction": "up", "category": "paos",
                           "definition": "优化后估算模型 token 吞吐 G / T_step^1。"},
            "S_i_aicore": {"label": "AI Core Speedup", "short": "S_aicore", "unit": "×", "direction": "up", "category": "paos",
                           "definition": "AI 活跃时间加速比 a_base / a_agent，反映是否真正更省算力。"},
            "U_i": {"label": "AI Active Utilization", "short": "U_i", "unit": "", "direction": "up", "category": "paos",
                    "definition": "算子级 AI 活跃时间利用率 a_i / t_i（诊断上下文，不参与 PAOS 主分）。"},
            "U_step": {"label": "Step Utilization", "short": "U_step", "unit": "", "direction": "up", "category": "paos",
                       "definition": "step 级 AI 活跃利用率 Σ(n_i·a_i) / T_step。"},
            "VecRatio": {"label": "Vector Ratio", "short": "VEC", "unit": "", "direction": "context", "category": "paos",
                         "definition": "AI Vector 时间 / Task Duration，用于判断算子是否 vector-bound。"},
            "CubeRatio": {"label": "Cube Ratio", "short": "CUBE", "unit": "", "direction": "context", "category": "paos",
                          "definition": "Cube 利用率，用于判断算子是否 cube-bound。"},
            "MTE2Ratio": {"label": "MTE2 Ratio", "short": "MTE2", "unit": "", "direction": "context", "category": "paos",
                          "definition": "MTE2（搬运）占比，用于识别访存瓶颈。"},
        },
    }


def valid_values(runs: Iterable[Dict[str, Any]], category: str, metric: str) -> List[float]:
    values = []
    for run in runs:
        value = safe_float((run.get(category) or {}).get(metric))
        if value is not None:
            values.append(value)
    return values


def compute_runs(
    root: Path,
    targets: Tuple[float, ...],
    paos_g: Optional[float] = None,
    paos_batch_size: Optional[float] = None,
    paos_scenario: Optional[str] = None,
    paos_chunk_len: Optional[float] = None,
    paos_t_other_ms: Optional[float] = None,
    paos_calls_per_step: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Evaluate every discoverable run under `root` (one pass, no aggregation).

    `discover_run_dirs` already walks in sorted order, so the result is stable
    across runs on an unchanged root.
    """
    return [
        compute_run_metrics(run_dir, root, targets, paos_g, paos_batch_size, paos_scenario, paos_chunk_len, paos_t_other_ms, paos_calls_per_step)
        for run_dir in discover_run_dirs(root)
    ]


def summarize_runs(
    root: Path,
    runs: List[Dict[str, Any]],
    targets: Tuple[float, ...],
) -> Dict[str, Any]:
    """Suite-level aggregate over an already-evaluated set of runs.

    Split out from `summarize_suite` so the same aggregation can be applied to
    any subset — the whole root, or one (agent × engine model) bucket — without
    re-reading the traces or re-deriving a second, drifting definition of the
    suite metrics.
    """
    total = len(runs)
    accuracy = {
        "TaskPassRate": sum((run["accuracy"]["TaskPass"] for run in runs), 0) / total if total else None,
        "GateRate": sum((run["accuracy"]["CorrectnessGate"] for run in runs), 0) / total if total else None,
        "RunCompletedRate": sum((run["accuracy"]["RunCompleted"] for run in runs), 0) / total if total else None,
    }
    speedups = valid_values(runs, "sts", "Speedup")
    vlrs = valid_values(runs, "sts", "VLR")
    end_gains = valid_values(runs, "sts", "EndGain")
    auls_token = valid_values(runs, "sts", "AULS_token")
    auls_time = valid_values(runs, "sts", "AULS_time")
    sts_token = valid_values(runs, "sts", "STS_token")
    sts_time = valid_values(runs, "sts", "STS_time")
    sts: Dict[str, Any] = {
        "EBGS": math.exp(sum(math.log(value) for value in speedups) / len(speedups)) if speedups and all(value > 0 for value in speedups) else None,
        "mean_Speedup": (sum(speedups) / len(speedups)) if speedups else None,
        "median_Speedup": float(median(speedups)) if speedups else None,
        "max_Speedup": max(speedups) if speedups else None,
        "mean_VLR": sum(vlrs) / len(vlrs) if vlrs else None,
        "mean_EndGain": sum(end_gains) / len(end_gains) if end_gains else None,
        "mean_AULS_token": sum(auls_token) / len(auls_token) if auls_token else None,
        "mean_AULS_time": sum(auls_time) / len(auls_time) if auls_time else None,
        "mean_STS_token": sum(sts_token) / len(sts_token) if sts_token else None,
        "mean_STS_time": sum(sts_time) / len(sts_time) if sts_time else None,
    }
    for target in targets:
        suffix = target_suffix(target)
        target_vlrs = [value for value in vlrs if value is not None]
        eir = sum(value >= target for value in target_vlrs) / len(target_vlrs) if target_vlrs else None
        sts[f"EIR_{suffix}"] = eir
        token_values = valid_values(runs, "sts", f"TokensTo{suffix}")
        time_values = valid_values(runs, "sts", f"TimeTo{suffix}")
        sts[f"median_TokensTo{suffix}"] = float(median(token_values)) if token_values else None
        sts[f"median_TimeTo{suffix}"] = float(median(time_values)) if time_values else None
    paos_values = valid_values(runs, "paos", "PAOS_base")
    ptei_values = valid_values(runs, "paos", "PTEI")
    pkai_values = valid_values(runs, "paos", "PKAI")
    etps_base_values = valid_values(runs, "paos", "eTPS_base")
    etps_agent_values = valid_values(runs, "paos", "eTPS_agent")
    profile_values = lambda metric: [
        value
        for run in runs
        for value in [safe_float(((run.get("paos") or {}).get("profiling") or {}).get(metric))]
        if value is not None
    ]
    aicore_speedups = profile_values("S_i_aicore")
    active_utilizations = profile_values("U_i")
    step_utilizations = profile_values("U_step")
    vec_ratios = profile_values("VecRatio")
    cube_ratios = profile_values("CubeRatio")
    mte2_ratios = profile_values("MTE2Ratio")
    paos = {
        "computed_runs": len(paos_values),
        "mean_PAOS_base": sum(paos_values) / len(paos_values) if paos_values else None,
        "mean_PTEI": sum(ptei_values) / len(ptei_values) if ptei_values else None,
        "mean_PKAI": sum(pkai_values) / len(pkai_values) if pkai_values else None,
        "mean_eTPS_base": sum(etps_base_values) / len(etps_base_values) if etps_base_values else None,
        "mean_eTPS_agent": sum(etps_agent_values) / len(etps_agent_values) if etps_agent_values else None,
        "mean_S_i_aicore": sum(aicore_speedups) / len(aicore_speedups) if aicore_speedups else None,
        "mean_U_i": sum(active_utilizations) / len(active_utilizations) if active_utilizations else None,
        "mean_U_step": sum(step_utilizations) / len(step_utilizations) if step_utilizations else None,
        "mean_VecRatio": sum(vec_ratios) / len(vec_ratios) if vec_ratios else None,
        "mean_CubeRatio": sum(cube_ratios) / len(cube_ratios) if cube_ratios else None,
        "mean_MTE2Ratio": sum(mte2_ratios) / len(mte2_ratios) if mte2_ratios else None,
        "status": "computed" if paos_values else "unavailable",
    }
    if not paos_values:
        paos["reason"] = "no_run_with_complete_recipe_and_correctness_gate"
    # Scenario metadata is recorded once at suite level so the UI can label the
    # PAOS number with the runtime scenario it was computed under. Without this
    # the same PAOS value is ambiguous across decode / prefill workloads.
    computed_run = next((run for run in runs if ((run.get("paos") or {}).get("PAOS_base") is not None)), None)
    if computed_run is not None:
        reference = computed_run.get("paos") or {}
        paos["runtime_scenario"] = reference.get("runtime_scenario")
        paos["G"] = reference.get("G")
        paos["batch_size"] = reference.get("batch_size")
        paos["chunk_len"] = reference.get("chunk_len")
        paos["T_other_ms"] = reference.get("T_other_ms")
        paos["calls_per_step"] = reference.get("calls_per_step")
        paos["input_source"] = reference.get("input_source")
    cost_rows = [run.get("cost") or {} for run in runs]
    cost_tokens = [value for value in (safe_float(row.get("TotalTokens")) for row in cost_rows) if value is not None]
    cost_wall = [value for value in (safe_float(row.get("TotalWallClockSec")) for row in cost_rows) if value is not None]
    cost_rounds = [value for value in (safe_float(row.get("LLMRounds")) for row in cost_rows) if value is not None]
    cost_tools = [value for value in (safe_float(row.get("ToolCalls")) for row in cost_rows) if value is not None]
    cost = {
        "TotalTokens": sum(cost_tokens) if cost_tokens else None,
        "median_TotalTokens": float(median(cost_tokens)) if cost_tokens else None,
        "TotalWallClockSec": sum(cost_wall) if cost_wall else None,
        "median_WallClockSec": float(median(cost_wall)) if cost_wall else None,
        "mean_LLMRounds": (sum(cost_rounds) / len(cost_rounds)) if cost_rounds else None,
        "median_LLMRounds": float(median(cost_rounds)) if cost_rounds else None,
        "TotalToolCalls": sum(cost_tools) if cost_tools else None,
        "median_ToolCalls": float(median(cost_tools)) if cost_tools else None,
    }
    # Suite-level token breakdown for the same pricing formula.
    for key in ("InputTokens", "CachedInputTokens", "CacheWriteTokens", "OutputTokens"):
        values = [value for value in (safe_float(row.get(key)) for row in cost_rows) if value is not None]
        cost[f"Total{key}"] = sum(values) if values else None

    # Who generated the operators: unioned across runs, so a suite that mixed
    # agents or models reports every one of them instead of only the first.
    def distinct(field: str, key: str) -> List[str]:
        return sorted({
            (run.get("identity") or {}).get(field, {}).get(key)
            for run in runs
            if ((run.get("identity") or {}).get(field) or {}).get(key)
        })

    # Dedupe whole identity blocks (not just the ids) so the resolved
    # type / strategy / vendor travels with the id.
    agents = {(run.get("identity") or {}).get("agent", {}).get("id"): (run.get("identity") or {}).get("agent")
              for run in runs if ((run.get("identity") or {}).get("agent") or {}).get("id")}
    engine_models = {(run.get("identity") or {}).get("engine_model", {}).get("id"): (run.get("identity") or {}).get("engine_model")
                     for run in runs if ((run.get("identity") or {}).get("engine_model") or {}).get("id")}

    return {
        "root": str(root),
        "n_runs": total,
        "generators": {
            "agents": [
                {key: value for key, value in agents[agent_id].items() if key not in ("raw", "turns", "all")}
                for agent_id in sorted(agents)
            ],
            "engine_models": [
                {key: engine_models[model_id].get(key) for key in ("id", "vendor", "openness")}
                for model_id in sorted(engine_models)
            ],
            "agent_ids": distinct("agent", "id"),
            "engine_model_ids": distinct("engine_model", "id"),
        },
        "accuracy": accuracy,
        "cost": cost,
        "sts": sts,
        "paos": paos,
        "catalog": metric_catalog(),
        "runs": runs,
    }


def summarize_suite(
    root: Path,
    targets: Tuple[float, ...],
    paos_g: Optional[float] = None,
    paos_batch_size: Optional[float] = None,
    paos_scenario: Optional[str] = None,
    paos_chunk_len: Optional[float] = None,
    paos_t_other_ms: Optional[float] = None,
    paos_calls_per_step: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate `root` and aggregate every run into a single suite summary."""
    runs = compute_runs(root, targets, paos_g, paos_batch_size, paos_scenario, paos_chunk_len, paos_t_other_ms, paos_calls_per_step)
    return summarize_runs(root, runs, targets)


def split_into_datasets(
    root: Path,
    runs: List[Dict[str, Any]],
    targets: Tuple[float, ...],
) -> List[Dict[str, Any]]:
    """Split evaluated runs into one dataset per (agent × engine model) pair.

    Returns a list of `{filename, summary}` — the exact shape the dashboard's
    dataset folder expects, with the file name derived from the run identities
    rather than requested on the command line. Ordering is deterministic
    (by agent id, then model id) so regenerating an unchanged root produces
    byte-identical names instead of a shuffled folder.
    """
    datasets = []
    for (agent_id, model_id), bucket in sorted(
        bucket_runs_by_identity(runs).items(),
        key=lambda item: (item[0][0] or "", item[0][1] or ""),
    ):
        datasets.append({
            "agent_id": agent_id,
            "engine_model_id": model_id,
            "filename": dataset_filename(agent_id, model_id),
            "summary": summarize_runs(root, bucket, targets),
        })
    return datasets


def write_datasets(datasets: List[Dict[str, Any]], directory: Path) -> List[Path]:
    """Write each split dataset to `directory`; returns the paths written."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset in datasets:
        path = directory / dataset["filename"]
        path.write_text(json.dumps(dataset["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


def use_utf8_stdout() -> None:
    """Keep `print` from crashing on a legacy console code page.

    The metric catalog carries real units (`µs`), and a Windows console still
    defaults to GBK / cp1252 often enough that dumping the summary to stdout
    would raise UnicodeEncodeError instead of showing it. Writing files was
    never affected — `encoding="utf-8"` is explicit there.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):  # already wrapped / not a TextIOWrapper
            pass


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(
        description=(
            "Compute accuracy, STS, and concise PAOS metrics from Ascend operator traces. "
            "By default the result is split into one dataset file per "
            "(agent × engine model) pair, named after those identities in the same "
            "convention as src/data/eval_datasets_summary/."
        )
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "operator", help="Directory containing operator run directories (a batch root or a single batch/run directory).")
    parser.add_argument("--output", type=Path, default=None, help="Write one combined summary JSON here instead of splitting per identity.")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"Directory for the split dataset files. Default: {DEFAULT_DATASET_DIR}.")
    parser.add_argument("--no-split", action="store_true", help="Keep runs in a single summary (no per-identity files); print to stdout unless --output is given.")
    parser.add_argument("--dry-run", action="store_true", help="Show the file names that would be written, without writing them.")
    parser.add_argument("--targets", nargs="*", type=float, default=list(DEFAULT_TARGETS), help="VLR targets, for example 0.01 0.05 0.10.")
    parser.add_argument("--paos-g", type=float, default=None, help="Explicit benchmark G override; prefer --paos-scenario with its required parameters.")
    parser.add_argument("--paos-scenario", choices=("decode", "prefill"), default=None, help="Runtime scenario used to compute G.")
    parser.add_argument("--paos-batch-size", type=float, default=None, help="Benchmark batch size; required to compute G when no explicit G is given.")
    parser.add_argument("--paos-chunk-len", type=float, default=None, help="Prefill chunk length; required for G=batch_size*chunk_len.")
    parser.add_argument("--paos-t-other-ms", type=float, default=None, help="PAOS unoptimized step time in ms; overrides recipe value, default 0.")
    parser.add_argument("--paos-calls-per-step", type=float, default=None, help="Default PAOS calls per step n_i when recipe is missing or incomplete; default 1.")
    args = parser.parse_args()

    # Absolute from here on. A relative root makes the effective path length
    # depend on the caller's cwd, which on Windows can push a deep profiling CSV
    # past MAX_PATH and make it read as missing (`safe_read_text` / `long_path`).
    args.root = resolve_root(args.root)

    runs = compute_runs(args.root, tuple(args.targets), args.paos_g, args.paos_batch_size, args.paos_scenario, args.paos_chunk_len, args.paos_t_other_ms, args.paos_calls_per_step)
    if not runs:
        parser.error(
            f"no runs found under {args.root} "
            "(each run needs result.json, or evaluations.jsonl plus config.resolved.json/manifest.json)"
        )

    if args.no_split:
        summary = summarize_runs(args.root, runs, tuple(args.targets))
        rendered = json.dumps(summary, indent=2, ensure_ascii=False)
        if args.output is None:
            print(rendered)
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote summary to {args.output}")
        return

    datasets = split_into_datasets(args.root, runs, tuple(args.targets))
    directory = args.out_dir or DEFAULT_DATASET_DIR
    directory = Path(os.path.abspath(directory))
    if args.dry_run:
        print(f"Would write {len(datasets)} dataset file(s) to {directory}:")
        for dataset in datasets:
            print(f"  {dataset['filename']}  ({dataset['summary']['n_runs']} runs)")
        return
    for path, dataset in zip(write_datasets(datasets, directory), datasets):
        print(f"Wrote {path}  ({dataset['summary']['n_runs']} runs)")


if __name__ == "__main__":
    main()
