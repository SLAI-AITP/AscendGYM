"""metrics_master STS formulae, with explicit missing-data and unit handling."""
import math

TOKEN_FIELDS = ("input_tokens", "uncached_input_tokens", "cache_read_tokens",
                "cache_write_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def hardware_observations(raw):
    """Normalize explicit scalar fields; never silently average across shapes."""
    aliases = {
        "aicore_time_us": ["aicore_time_us", "aicore_time(us)", "AI Core Time (us)"],
        "aiv_time_us": ["aiv_time_us", "aiv_time(us)", "AI Vector Time (us)"],
        "core_count": ["core_count", "Block Num", "Block Dim"],
        "mix_core_count": ["Mix Block Num", "Mix Block Dim"],
        "vec_ratio": ["vec_ratio", "aiv_vec_ratio", "Vec Ratio"],
        "cube_ratio": ["cube_ratio", "aic_mac_ratio", "Mac Ratio"],
        "mte2_ratio": ["mte2_ratio", "aic_mte2_ratio", "MTE2 Ratio"],
    }
    out = {}
    for key, names in aliases.items():
        values = [raw[n] for n in names if type(raw.get(n)) in (float, int)
                  and math.isfinite(raw[n]) and raw[n] >= 0]
        out[key] = values[0] if values else None
    cube, vector = out["aicore_time_us"], out["aiv_time_us"]
    if cube and vector:
        kind, active = "mixed", None
    elif cube:
        kind, active = "aicore", cube
    elif vector:
        kind, active = "aiv", vector
    else:
        kind, active = "unavailable", None
    out.update(active_time_kind=kind, active_time_us=active, normalization="unspecified")
    out["availability"] = {k: "observed" if v is not None else "unavailable"
                           for k, v in out.items() if k not in {"active_time_kind", "normalization"}}
    return out


def normalize_usage(raw, input_includes_cache=True):
    raw = raw or {}
    def get(*keys):
        for key in keys:
            if key in raw and raw[key] is not None:
                v = raw[key]
                if type(v) is not int or v < 0:
                    raise ValueError(f"invalid provider token count: {key}")
                return v
        return None
    inp = get("input_tokens", "prompt_tokens", "input")
    out = get("output_tokens", "completion_tokens", "output")
    details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    read = get("cache_read_tokens", "cache_read_input_tokens", "cacheRead")
    if read is None:
        read = details.get("cached_tokens")
    write = get("cache_write_tokens", "cache_creation_input_tokens", "cacheWrite")
    reason = get("reasoning_tokens", "reasoning")
    if reason is None:
        reason = (raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}).get("reasoning_tokens")
    if inp is not None and not input_includes_cache:
        inp += (read or 0) + (write or 0)
    total = get("total_tokens")
    if total is None and inp is not None and out is not None:
        total = inp + out
    return {"available": total is not None, "input_tokens": inp,
            "uncached_input_tokens": None if inp is None else max(0, inp - (read or 0) - (write or 0)),
            "cache_read_tokens": read, "cache_write_tokens": write,
            "output_tokens": out, "reasoning_tokens": reason, "total_tokens": total}


def sts(rows, baseline_ms, checkpoints, axis):
    if not checkpoints:
        return {"status": "unavailable", "reason": "configure a fixed positive budget endpoint"}
    if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in checkpoints):
        raise ValueError("STS checkpoints must be finite and positive")
    if sorted(set(checkpoints)) != checkpoints:
        raise ValueError("STS checkpoints must increase strictly")
    if type(baseline_ms) not in (float, int) or not math.isfinite(baseline_ms) or baseline_ms <= 0:
        raise ValueError("STS baseline must be finite and positive")
    if any(r.get(axis) is None for r in rows):
        return {"status": "unavailable", "reason": f"incomplete {axis} coverage"}
    if any(type(r[axis]) not in (float, int) or not math.isfinite(r[axis]) or r[axis] < 0 for r in rows):
        raise ValueError("STS event coordinates must be finite and nonnegative")
    passed = [r for r in rows if r["status"] == "passed"]
    if any(type(r.get("duration_ms")) not in (float, int)
           or not math.isfinite(r["duration_ms"]) or r["duration_ms"] <= 0 for r in passed):
        return {"status": "unavailable", "reason": "invalid passed-candidate duration"}
    budget_max = checkpoints[-1]
    durations = {}
    for row in passed:
        coordinate = row[axis]
        if coordinate <= budget_max:
            durations[coordinate] = min(durations.get(coordinate, baseline_ms), row["duration_ms"])
    best = baseline_ms
    log_gain = 0.0
    improvements = []
    penalties = []
    for coordinate, duration in sorted(durations.items()):
        if duration >= best:
            continue
        gain = math.log2(baseline_ms / duration)
        delta = gain - log_gain
        penalties.append(50 * (coordinate / budget_max) * delta)
        improvements.append({"budget": coordinate, "best_duration_ms": duration,
                             "speedup": baseline_ms / duration, "delta_log2_speedup": delta})
        best, log_gain = duration, gain
    base_score = 100 * log_gain
    delay_penalty = math.fsum(penalties)
    score = base_score - delay_penalty
    curve = []
    for budget in checkpoints:
        duration = min([baseline_ms] + [d for coordinate, d in durations.items() if coordinate <= budget])
        curve.append({"budget": budget, "best_duration_ms": duration, "speedup": baseline_ms / duration})
    first = {}
    for delta in (.01, .03, .05, .10):
        hits = [row[axis] for row in passed if row[axis] <= budget_max
                and 1 - row["duration_ms"] / baseline_ms >= delta - 1e-12]
        first[str(delta)] = min(hits) if hits else None
    return {"status": "observed", "axis": axis, "curve": curve,
            "improvements": improvements, "budget_max": budget_max,
            "base_score": base_score, "delay_penalty": delay_penalty,
            "STS": score, "S_equiv": 2 ** (score / 100),
            "S_at_Bmax": baseline_ms / best, "VLR": 1 - best / baseline_ms,
            "first_hit": first, "formula": "linear-budget-delay-v1",
            "integration": "linear-budget event delay penalty"}


def paos_single(baseline_ms, best_ms, recipe):
    """Single task recipe with fixed unoptimized overhead. Never infer launch counts."""
    if not recipe:
        return {"status": "unavailable", "reason": "missing workload recipe G/n_i/shape alignment"}
    g, n, other = recipe["tokens_per_step"], recipe["calls_per_step"], recipe["other_step_time_ms"]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (g, n, other)) or g <= 0 or n <= 0 or other < 0:
        raise ValueError("invalid recipe G/n_i/T_other")
    if recipe.get("latency_scope") != "single_invocation":
        return {"status": "unavailable", "reason": "recipe requires single-invocation, shape-aligned latency; batch aggregate is not valid"}
    base_step, agent_step = n * baseline_ms + other, n * best_ms + other
    pkai, ptei = baseline_ms / best_ms, base_step / agent_step
    return {"status": "observed", "scope": "single evaluated kernel + fixed overhead",
            "PKAI": pkai, "PTEI": ptei, "PAOS": math.sqrt(pkai*ptei),
            "eTPS_base": g * 1000 / base_step, "eTPS_agent": g * 1000 / agent_step,
            "eTPSGain": ptei, "profile_only": other == 0, "recipe": recipe,
            "PPUI": None, "PAOS_full": None,
            "limitations": "No assumed active-time normalization, FLOPs or memory bytes."}


def compute_metrics(rows, config, evidence_kind, run_context=None):
    context = run_context or {}
    bases = [r for r in rows if r["evaluation_kind"] == "baseline" and r["status"] == "passed"]
    if not bases:
        return {"status": "unavailable", "reason": "baseline not verified", "evidence_kind": evidence_kind,
                "official_ranking_eligible": False, "single_run_evidence_eligible": False}
    base = bases[0]["duration_ms"]
    search = [r for r in rows if r["evaluation_kind"] == "candidate"]
    best = min([base] + [r["duration_ms"] for r in search if r["status"] == "passed"])
    metrics = {"evidence_kind": evidence_kind, "official_ranking_eligible": False,
            "ranking_note": "single-run measurements do not establish suite eligibility, exclusions or Reliability",
            "single_run_evidence_eligible": bool(evidence_kind == "real" and context.get("operational_ok")
                and context.get("usage_complete") and (not context.get("verification_required") or context.get("verification_valid"))),
            "run_validity": context,
            "STS_token": sts(search, base, config.get("token_checkpoints", []), "cumulative_tokens"),
            "STS_time": sts(search, base, config.get("time_checkpoints_sec", []), "solution_elapsed_sec"),
            "raw_speedup": base / best, "PAOS": paos_single(base, best, config.get("recipe")),
            "Reliability": {"status": "unavailable", "reason": "requires >=3 independent runs, not three profiles"},
            "aggregate_EBGS_EIR": {"status": "not_applicable", "reason": "single-task run; task weights and exclusions belong to suite evaluator"},
            "candidate_pass_rate": sum(r["status"] == "passed" for r in search) / len(search) if search else None}
    if context.get("candidate_usage_complete", context.get("usage_complete")) is False:
        metrics["STS_token"] = {"status": "unavailable", "reason": "incomplete candidate-completion provider usage coverage"}
    if context.get("operational_ok") is False and metrics["STS_time"].get("status") == "observed":
        metrics["STS_time"].update(status="partial", reason="run did not complete normally; descriptive observed trajectory only")
    return metrics
