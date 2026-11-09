#!/usr/bin/env python3
"""Regression test: NPU kv_rms_norm_rope_cache vs golden data collected from the same operator.

Inputs are LOADED from golden_input/, never regenerated from a seed: the comparison is only
meaningful if the inputs are byte-identical to the ones used at collection time.

Tolerance: golden is frozen while the sources keep changing, so the check carries slack —
|ref - cur| / max(|ref|, 1) <= 1e-4 for every element.

This suite covers the V1 (kv-combined, N=1) + Norm-cache-mode dialect. In that configuration
the kernel writes exactly two in-place buffers — k_cache (RoPE'd RMSNorm K) and ckv_cache
(RMSNorm V) — at the `index` rows; the k_rope / c_kv outputs are NOT produced (is_output_kv is
ineffective in Norm non-quant, per the operator README), so only those two buffers are compared.
"""
import json
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "kv_rms_norm_rope_cache"
OP_CAMEL = "KvRmsNormRopeCache"
DEVICE = "npu:0"

# Positional order of the generated wrapper (binding.py), op_def order, excluding the trailing
# out_tmpls buffer list. Inputs (op_def order) then attrs (op_def order); optional inputs are
# passed as None.
PARAM_NAMES = [
    "kv", "gamma", "cos", "sin", "index", "k_cache", "ckv_cache",
    "k_rope_scale", "c_kv_scale", "k_rope_offset", "c_kv_offset", "v",
    "epsilon", "cache_mode", "is_output_kv",
]
OUTPUT_KEYS = ["k_cache", "ckv_cache"]

# Both compared outputs are bound by the (fp16/bf16) input dtype — the kernel result is stored
# in the input dtype, so no separate intermediate float32 tolerance is needed here.
INTERMEDIATE_OUTPUT_KEYS = set()

ATOL_DEFAULT = 1e-4
ERR_RATE_DEFAULT = 0.0

ATOL_BY_DTYPE = {torch.float16: 2 ** -8, torch.bfloat16: 2 ** -6, torch.float32: 2 ** -11}

# Per-case tolerance overrides. {case_id: {"atol": float or None, "err_rate": float}}
TOLERANCE_OVERRIDES = {}


def resolve_data_root():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "test_data"),
        os.path.join(here, OP_NAME, "test_data"),
        os.path.join(os.getcwd(), OP_NAME, "test_data"),
        os.path.join(os.getcwd(), "test_data"),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "golden_result", OP_CAMEL)):
            return path
    raise FileNotFoundError(
        "golden data not found. Tried:\n  " + "\n  ".join(candidates)
        + f"\nRun datamining/{OP_NAME}_collect.sh first."
    )


DATA_ROOT = resolve_data_root()
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def compare_tensor(ref_t, cur_t, atol=ATOL_DEFAULT, err_rate=ERR_RATE_DEFAULT):
    """Two independent knobs:
        atol     — per-element threshold on diff / max(|ref|, 1)
        err_rate — allowed fraction of elements exceeding atol
    """
    if tuple(ref_t.shape) != tuple(cur_t.shape):
        return float("inf"), 1.0, f"SHAPE {tuple(cur_t.shape)}!={tuple(ref_t.shape)}"
    if ref_t.dtype != cur_t.dtype:
        return float("inf"), 1.0, f"DTYPE {cur_t.dtype}!={ref_t.dtype}"

    ref_np = ref_t.float().cpu().detach().numpy().flatten().astype("float64")
    cur_np = cur_t.float().cpu().detach().numpy().flatten().astype("float64")
    diff = np.abs(ref_np - cur_np)
    rel = diff / np.maximum(np.abs(ref_np), 1.0)

    max_diff = float(np.max(diff)) if diff.size else 0.0
    error_rate = float((rel > atol).sum()) / max(ref_np.size, 1)
    status = "PASS" if error_rate <= err_rate else "FAIL"
    return max_diff, error_rate, status


def tolerance_for(case_id, output_key, input_dtype, ref_dtype):
    override = TOLERANCE_OVERRIDES.get(case_id)
    if override is None:
        return ATOL_DEFAULT, ERR_RATE_DEFAULT
    atol = override.get("atol", ATOL_DEFAULT)
    if atol is None:
        dtype = input_dtype if output_key in INTERMEDIATE_OUTPUT_KEYS else ref_dtype
        atol = max(ATOL_DEFAULT, ATOL_BY_DTYPE.get(dtype, 0.0))
    return atol, override.get("err_rate", ERR_RATE_DEFAULT)


def build_out_tmpls(input_data):
    """Template buffers for the two non-in-place outputs (k_rope, c_kv) that the wrapper
    allocates via empty_like from out_tmpls[2] and out_tmpls[3] (full OpDef output order
    [k_cache, ckv_cache, k_rope, c_kv]). Positions 0/1 are the in-place cache buffers and are
    never read from the template list, so they are small dummies."""
    dtype = input_data["kv"].dtype
    B, N, S, _ = input_data["kv"].shape
    Dk = input_data["cos"].shape[-1]
    Dv = input_data["gamma"].shape[0]
    dummy = torch.empty((1,), dtype=dtype, device=DEVICE)
    k_rope = torch.empty((B, N, S, Dk), dtype=dtype, device=DEVICE)
    c_kv = torch.empty((B, N, S, Dv), dtype=dtype, device=DEVICE)
    return [dummy, dummy, k_rope, c_kv]


def run_case(case_id):
    name = f"case_{case_id}"
    input_data = torch.load(os.path.join(INPUT_DIR, f"{name}.pt"), weights_only=False)
    golden = torch.load(os.path.join(RESULT_DIR, f"{name}.pt"), weights_only=False)

    call_kwargs = {}
    for pname in PARAM_NAMES:
        if pname not in input_data:
            raise KeyError(f"{name}: missing param '{pname}' in golden_input")
        val = input_data[pname]
        if val is None:
            call_kwargs[pname] = None
        elif isinstance(val, torch.Tensor):
            call_kwargs[pname] = val.to(DEVICE)
        else:
            call_kwargs[pname] = val

    out_tmpls = build_out_tmpls(input_data)
    positional = [call_kwargs[p] for p in PARAM_NAMES] + [out_tmpls]
    outputs = ops.npu_kv_rms_norm_rope_cache(*positional)
    torch.npu.synchronize()

    if outputs is None:
        # in-place outputs: fall back to the buffers we passed in
        outputs = [call_kwargs["k_cache"], call_kwargs["ckv_cache"]]
    elif isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    input_dtype = input_data["kv"].dtype
    results = []
    for i, okey in enumerate(OUTPUT_KEYS):
        if okey not in golden:
            continue
        if i >= len(outputs):
            results.append((okey, float("nan"), 1.0, "MISSING"))
            continue
        atol, err_rate = tolerance_for(case_id, okey, input_dtype, golden[okey].dtype)
        results.append((okey,) + compare_tensor(
            golden[okey].to(DEVICE), outputs[i], atol=atol, err_rate=err_rate
        ))
    return results


def main():
    manifest_path = os.path.join(DATA_ROOT, "cases_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as fh:
            collected = [c["id"] for c in json.load(fh)["cases"]]
    else:
        collected = sorted(
            int(f[len("case_"):-len(".pt")])
            for f in os.listdir(RESULT_DIR)
            if f.startswith("case_") and f.endswith(".pt")
        )

    if not collected:
        print(f"No golden cases in {RESULT_DIR} — run the collect script first.")
        sys.exit(1)

    print(f"Golden data: {DATA_ROOT}   (atol={ATOL_DEFAULT:.0e})")
    print(f"{'case':<12}{'output':<24}{'max_diff':<18}{'err_rate':<14}{'status':<8}")
    print("-" * 74)

    all_pass = True
    for case_id in collected:
        try:
            for okey, max_diff, error_rate, status in run_case(case_id):
                if status != "PASS":
                    all_pass = False
                print(f"{'case_' + str(case_id):<12}{okey:<24}{max_diff:<18.10f}{error_rate:<14.6f}{status:<8}")
        except Exception as exc:
            all_pass = False
            print(f"{'case_' + str(case_id):<12}{'-':<24}{'ERR':<18}{'-':<14}FAIL  {type(exc).__name__}: {str(exc)[:200]}")

    print("-" * 74)
    if all_pass:
        print(f"ALL CHECKS PASSED ({len(collected)} cases)")
    else:
        print("REGRESSION DETECTED — do NOT re-collect golden data to make this pass.")
        sys.exit(1)


if __name__ == "__main__":
    main()
