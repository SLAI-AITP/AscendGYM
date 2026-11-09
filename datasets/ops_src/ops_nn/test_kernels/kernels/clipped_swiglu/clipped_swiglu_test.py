#!/usr/bin/env python3
"""Reference regression test: NPU operator vs golden data collected from the same operator.

This is the canonical test script for the sandbox workflow. To adapt it to another operator,
change only OP_NAME/OP_CAMEL, PARAM_NAMES, OUTPUT_KEYS, INTERMEDIATE_OUTPUT_KEYS and the
ops.npu_<op_name>(...) call in run_case(). Leave path resolution, the comparator and the exit code alone.

Inputs are LOADED from golden_input/, never regenerated from a seed: the comparison is only
meaningful if the inputs are byte-identical to the ones used at collection time.

Tolerance: golden is frozen while the sources keep changing, so the check carries slack —
|ref - cur| / max(|ref|, 1) <= 1e-4 for every element. Note this is stricter than one ulp for
fp16/bf16 outputs of order 1 (spacing 2**-10 and 2**-8 there); the slack is real for float32 and
for small-magnitude outputs. A legitimate rounding shift in bf16 needs a per-case override.

Naming: the operator is called as ops.npu_<op_name> verbatim — the npu_ prefix is part of the
registered name; no _custom/_v2/_kernel suffixes, no wrapper function, no alias.
"""
import json
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "clipped_swiglu"
OP_CAMEL = "ClippedSwiglu"
DEVICE = "npu:0"

# Positional order expected by ops.npu_clipped_swiglu(...), excluding the output buffers.
# Binding signature: (x, group_index, dim, alpha, limit, bias, interleaved, out_tmpl)
PARAM_NAMES = ["x", "group_index", "dim", "alpha", "limit", "bias", "interleaved"]
# Order in which the operator returns its outputs.
OUTPUT_KEYS = ["y"]

# Outputs that are intermediate statistics: stored as float32 but derived from the input tensor,
# so their real precision floor is the INPUT dtype, not their storage dtype.
INTERMEDIATE_OUTPUT_KEYS = set()

# Default tolerance. atol applies per element to diff / max(|ref|, 1); err_rate is the allowed
# fraction of elements exceeding it. Do not widen these globally.
ATOL_DEFAULT = 1e-4
ERR_RATE_DEFAULT = 0.0

# Dtype floor, a couple of ulp of headroom — used only when an override sets "atol": None.
ATOL_BY_DTYPE = {torch.float16: 2 ** -8, torch.bfloat16: 2 ** -7, torch.float32: 2 ** -11}

# Per-case tolerance overrides, for cases where an intentional source change shifted the rounding
# (or where the kernel's reduction order is nondeterministic). Each entry needs a comment naming
# the case, the output tensor and the cause. {case_id: {"atol": float or None, "err_rate": float}}
TOLERANCE_OVERRIDES = {}


def resolve_data_root():
    """Find test_data/ in both layouts: sandbox (CWD/<op>/test_data) and local (script dir/test_data).

    Do not branch on __file__.startswith("tmp") — that breaks when the client's temp naming changes.
    Absolute host paths never resolve inside the sandbox, so everything is relative to here or CWD.
    """
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
    Never compare error_rate against atol; they are different quantities.
    """
    if tuple(ref_t.shape) != tuple(cur_t.shape):
        return float("inf"), 1.0, f"SHAPE {tuple(cur_t.shape)}!={tuple(ref_t.shape)}"
    if ref_t.dtype != cur_t.dtype:
        return float("inf"), 1.0, f"DTYPE {cur_t.dtype}!={ref_t.dtype}"

    ref_np = ref_t.float().cpu().detach().numpy().flatten().astype("float64")
    cur_np = cur_t.float().cpu().detach().numpy().flatten().astype("float64")
    diff = np.abs(ref_np - cur_np)
    rel = diff / np.maximum(np.abs(ref_np), 1.0)

    max_diff = float(diff.max()) if diff.size else 0.0
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

    call_args = []
    for pname in PARAM_NAMES:
        if pname not in input_data:
            raise KeyError(f"{name}: missing param '{pname}' in golden_input")
        val = input_data[pname]
        call_args.append(val.to(DEVICE) if isinstance(val, torch.Tensor) else val)

    # Size the output buffers from the golden tensors — this also checks shapes and dtypes.
    out_bufs = [torch.empty_like(golden[k]).to(DEVICE) for k in OUTPUT_KEYS if k in golden]
    out_arg = out_bufs[0] if len(out_bufs) == 1 else out_bufs

    outputs = ops.npu_clipped_swiglu(*call_args, out_arg)
    torch.npu.synchronize()

    outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
    input_dtype = input_data[PARAM_NAMES[0]].dtype

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
    print(f"{'case':<12}{'output':<16}{'max_diff':<18}{'err_rate':<14}{'status':<8}")
    print("-" * 68)

    all_pass = True
    for case_id in collected:
        for okey, max_diff, error_rate, status in run_case(case_id):
            if status != "PASS":
                all_pass = False
            print(f"{'case_' + str(case_id):<12}{okey:<16}{max_diff:<18.10f}{error_rate:<14.6f}{status:<8}")

    print("-" * 68)
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
