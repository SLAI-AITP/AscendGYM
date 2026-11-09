#!/usr/bin/env python3
"""Regression test: NPU mla_prolog vs golden data collected from the same operator.

Inputs are LOADED from golden_input/, never regenerated from a seed: the comparison is only
meaningful if the inputs are byte-identical to the ones used at collection time.

Tolerance: golden is frozen while the sources keep changing, so the check carries slack —
|ref - cur| / max(|ref|, 1) <= 1e-4 for every element.

This is a single-kernel MLA prolog operator (non-quantized BF16 path). All four outputs
(query, query_rope, kv_cache, kr_cache) are replayed from the same NPU kernel, so an exact
match is expected. kv_cache / kr_cache are in-place outputs: the test passes the saved
pre-call cache tensors in and compares the written-through result. query / query_rope are
buffered template outputs (2-list out_tmpls).
"""
import json
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

# Re-create FRACTAL_NZ (internal format) tensors via npu_format_cast for the NZ-declared
# weights; allow_internal_format must be on or the cast falls back to ND.
torch_npu.npu.config.allow_internal_format = True

OP_NAME = "mla_prolog"
OP_CAMEL = "MlaProlog"
DEVICE = "npu:0"

# Positional order of the generated wrapper (binding.py), op_def order, excluding the
# trailing out_tmpls buffer list.
PARAM_NAMES = [
    "token_x", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
    "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "rope_sin", "rope_cos", "cache_index",
    "kv_cache", "kr_cache",
    "dequant_scale_x", "dequant_scale_w_dq", "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr", "quant_scale_ckv", "quant_scale_ckr", "smooth_scales_cq",
    "rmsnorm_epsilon_cq", "rmsnorm_epsilon_ckv", "cache_mode",
]
OUTPUT_KEYS = ["query", "query_rope", "kv_cache", "kr_cache"]

# These OpDef inputs are declared FORMAT_FRACTAL_NZ. torch.save / .cpu() do not persist the
# NPU format, so after loading from golden_input they come back as ND — re-apply the NZ cast
# (format 29) before the call, matching how they were passed at collection time.
NZ_WEIGHTS = {"weight_dq", "weight_uq_qr", "weight_dkv_kr"}

# All outputs are direct BF16 kernel results stored in BF16 — no float32 intermediates here.
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

    call_kwargs = {}
    for pname in PARAM_NAMES:
        if pname not in input_data:
            raise KeyError(f"{name}: missing param '{pname}' in golden_input")
        val = input_data[pname]
        if val is None:
            call_kwargs[pname] = None
        elif isinstance(val, torch.Tensor):
            t = val.to(DEVICE)
            if pname in NZ_WEIGHTS:
                t = torch_npu.npu_format_cast(t.contiguous(), 29)
            call_kwargs[pname] = t
        else:
            call_kwargs[pname] = val

    # Zero-init the two template output buffers (OpDef output order: query, query_rope) —
    # deterministic unwritten slots, matching collect. kv_cache / kr_cache are passed in as
    # in-place inputs and written through by the kernel.
    out_bufs = [
        torch.zeros_like(golden["query"]).to(DEVICE),
        torch.zeros_like(golden["query_rope"]).to(DEVICE),
    ]

    positional = [call_kwargs[p] for p in PARAM_NAMES] + [out_bufs]
    outputs = ops.npu_mla_prolog(*positional)
    torch.npu.synchronize()

    outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
    input_dtype = input_data["token_x"].dtype

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
