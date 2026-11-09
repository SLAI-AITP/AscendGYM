#!/usr/bin/env python3
"""Regression test: NPU qkv_rms_norm_rope_cache vs golden data collected from the same operator.

Inputs are LOADED from golden_input/, never regenerated from a seed: the comparison is only
meaningful if the inputs are byte-identical to the ones used at collection time.

Tolerance: golden is frozen while the sources keep changing, so the check carries slack —
|ref - cur| / max(|ref|, 1) <= 1e-4 for every element. int8 caches (cases 2 and 6) hold
quantized values; k/v_offset asymmetric quant (case 6) produces small negative values, all
relative-checked against the int8 ref so any rounding shift that changes a value trips the test.
"""
import json
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "qkv_rms_norm_rope_cache"
OP_CAMEL = "QkvRmsNormRopeCache"
DEVICE = "npu:0"

# Positional order of the generated wrapper (binding.py), excluding 'out'.
PARAM_NAMES = [
    "qkv", "q_gamma", "k_gamma", "cos", "sin", "index",
    "q_out", "k_cache", "v_cache",
    "k_scale", "v_scale", "k_offset", "v_offset",
    "qkv_size", "head_nums", "epsilon", "cache_mode", "is_output_qkv",
]
# Order in which the operator returns its outputs.
OUTPUT_KEYS = [
    "q_out", "k_cache", "v_cache",
    "q_out_before_quant", "k_out_before_quant", "v_out_before_quant",
]

# No float32-stored intermediates here: all 6 outputs are in the input dtype (or int8 cache).
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


def run_case(case_id):
    name = f"case_{case_id}"
    input_data = torch.load(os.path.join(INPUT_DIR, f"{name}.pt"), weights_only=False)
    golden = torch.load(os.path.join(RESULT_DIR, f"{name}.pt"), weights_only=False)

    # Build positional args — the wrapper (binding.py) calls the raw func by keyword.
    call_kwargs = {}
    for pname in PARAM_NAMES:
        if pname not in input_data:
            raise KeyError(f"{name}: missing param '{pname}' in golden_input")
        val = input_data[pname]
        if isinstance(val, torch.Tensor):
            call_kwargs[pname] = val.to(DEVICE)
        elif isinstance(val, list):
            call_kwargs[pname] = [v.to(DEVICE) if isinstance(v, torch.Tensor) else v for v in val]
        else:
            call_kwargs[pname] = val

    # Size the output buffers from the golden tensors — this also checks shape and dtype.
    # q_out/k_cache/v_cache are in-place: the input tensors (positions 6/7/8) must equal the
    # output buffers, exactly as in collect. Use zeros_like, not empty_like: the kernel only
    # writes the cache slots referenced by `index`, so unwritten slots must stay 0 (deterministic)
    # rather than whatever garbage empty_like leaves behind.
    out_bufs = [torch.zeros_like(golden[k]).to(DEVICE) for k in OUTPUT_KEYS]
    call_kwargs["q_out"] = out_bufs[0]
    call_kwargs["k_cache"] = out_bufs[1]
    call_kwargs["v_cache"] = out_bufs[2]

    positional = [call_kwargs[p] for p in PARAM_NAMES] + [out_bufs]
    outputs = ops.npu_qkv_rms_norm_rope_cache(*positional)
    torch.npu.synchronize()

    outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
    input_dtype = input_data["qkv"].dtype

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




def _run_suite(runs):
    """重复整套测试 runs 次（warmup + 测量）；单次 FAIL 不中断后续轮次。"""
    all_ok = True
    for run_idx in range(1, runs + 1):
        print(f"\n{'=' * 98}")
        print(f"Run {run_idx}/{runs}")
        print(f"{'=' * 98}")
        try:
            main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                all_ok = False
    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="repeated suite runs for warmup + measurement")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of repeated runs (each run iterates all cases)")
    args = parser.parse_args()
    raise SystemExit(0 if _run_suite(args.runs) else 1)
