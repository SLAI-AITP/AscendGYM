#!/usr/bin/env python3
"""
Regression test for the gather_pa_kv_cache NPU operator (multi-case).

Compares NPU output against pre-collected golden data for every case. Runs the
operator through the sandbox (NPU). Precision requirement: atol=1e-4 minimum
(gather is an exact copy, so diffs should be ~0 for floating dtypes; integer
dtypes must match exactly).

Golden data lives in:
  kernels/gather_pa_kv_cache/golden_{input,output}_case_{N}/
  kernels/gather_pa_kv_cache/golden_params_case_{N}.json

Run via running/gather_pa_kv_cache_test.sh (sandbox) or directly (local).
"""

import os
import sys
import json
import argparse

import torch
import torch_npu  # noqa: F401
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_kernel_dir():
    """Return the operator kernel dir (where golden_*_case_* live) in the current mode."""
    if os.path.basename(__file__).startswith("tmp"):
        # Sandbox: __file__ renamed to tmpXXX.py, SCRIPT_DIR == CWD == project root.
        return os.path.join(SCRIPT_DIR, "gather_pa_kv_cache")
    # Local: __file__ == .../kernels/gather_pa_kv_cache/gather_pa_kv_cache_test.py
    return SCRIPT_DIR


def _make_cache_tensor(shape, dtype, device):
    if dtype in (torch.int8, torch.int32, torch.int64):
        return torch.randint(-50, 50, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def load_case(kernel_dir, case_id):
    """Load params json and the saved golden outputs for a case."""
    with open(os.path.join(kernel_dir, f"golden_params_case_{case_id}.json")) as f:
        params = json.load(f)
    params["dtype"] = getattr(torch, params["dtype"].split(".")[-1])
    if params.get("index_dtype"):
        params["index_dtype"] = getattr(torch, params["index_dtype"].split(".")[-1])
    out_dir = os.path.join(kernel_dir, f"golden_output_case_{case_id}")
    golden_key = torch.load(os.path.join(out_dir, "key_ref.pt"), weights_only=False)
    golden_value = torch.load(os.path.join(out_dir, "value_ref.pt"), weights_only=False)
    return params, golden_key, golden_value


def regenerate_inputs(params, device="npu"):
    """Regenerate the exact same inputs as the collect script (same seed)."""
    torch.manual_seed(params["seed"])
    idx_dtype = params.get("index_dtype", torch.int32)

    key_cache = _make_cache_tensor((params["num_blocks"], params["block_size"],
                                    params["num_heads"], params["head_dim_k"]),
                                   params["dtype"], device)
    value_cache = _make_cache_tensor((params["num_blocks"], params["block_size"],
                                      params["num_heads"], params["head_dim_v"]),
                                     params["dtype"], device)
    block_tables = torch.randint(0, params["num_blocks"],
                                 (params["batch"], params["max_blocks"]),
                                 dtype=idx_dtype, device=device)
    lens = torch.tensor(params["lens"], dtype=idx_dtype, device=device)
    if params["is_seq_lens_cumsum"]:
        # Kernel convention: is_seq_lens_cumsum=True expects a PREFIX-SUM seq_lens of
        # length batch+1: [S0, S0+l0, ..., total]. cumsum() upcasts int32->int64 on the
        # NPU and the operator requires matching index dtype -> force back to idx_dtype.
        prefix = torch.cat([torch.zeros(1, dtype=idx_dtype, device=device),
                            torch.cumsum(lens, dim=0)]).to(idx_dtype)
        seq_lens = prefix
        total_tokens = int(seq_lens[-1].item())
    else:
        seq_lens = lens
        total_tokens = params["total_tokens"]

    key_ref = torch.empty((total_tokens, params["num_heads"], params["head_dim_k"]),
                          dtype=params["dtype"], device=device)
    value_ref = torch.empty((total_tokens, params["num_heads"], params["head_dim_v"]),
                            dtype=params["dtype"], device=device)

    seq_offset = None
    if params.get("has_seq_offset"):
        seq_offset = torch.arange(params["batch"], dtype=idx_dtype, device=device)

    return {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "key_ref": key_ref,
        "value_ref": value_ref,
        "seq_offset": seq_offset,
    }


def _compare_array(name, cur, golden, dtype):
    """Compare one output tensor against golden. Returns bool ok."""
    if dtype == torch.int8:
        cur_f = cur.to(torch.int32).numpy()
        ref_f = golden.to(torch.int32).numpy()
        max_diff = int(np.abs(cur_f.astype(np.int64) - ref_f.astype(np.int64)).max())
        ok = (max_diff == 0)
        print(f"  {name}: {'PASS' if ok else 'FAIL'} (max_abs_diff={max_diff})")
        return ok

    cur_f = cur.float().numpy()
    ref_f = golden.float().numpy()
    diff = np.abs(cur_f - ref_f)
    max_diff = float(diff.max())
    ok = max_diff <= 1e-4
    print(f"  {name}: {'PASS' if ok else 'FAIL'} (max_diff={max_diff:.3e})")
    if not ok:
        flat_idx = int(np.argmax(diff))
        idx = np.unravel_index(flat_idx, cur_f.shape)
        print(f"    worst @ {idx}: golden={ref_f[idx]:.6f} npu={cur_f[idx]:.6f}")
    return ok


def run_case(case_id, kernel_dir):
    """Run the NPU operator for one case and compare against golden."""
    params, golden_key, golden_value = load_case(kernel_dir, case_id)
    inputs = regenerate_inputs(params, device="npu")

    gsrc = params.get("golden_source", "npu")
    nfail = params.get("npu_failed", False)
    tag = " (CPU-ref baseline; NPU op broken at collect time)" if nfail else " (NPU baseline)"
    print(f"\n--- case {case_id} | dtype={params['dtype']} | "
          f"cache_mode={params.get('cache_mode', 'Norm')} | cumsum={params['is_seq_lens_cumsum']} | "
          f"golden={gsrc}{tag} ---")

    ops = __import__("gtsrusops", fromlist=["npu_gather_pa_kv_cache"])
    kwargs = dict(
        key_cache=inputs["key_cache"],
        value_cache=inputs["value_cache"],
        block_tables=inputs["block_tables"],
        seq_lens=inputs["seq_lens"],
        key=inputs["key_ref"],
        value=inputs["value_ref"],
        cache_mode=params.get("cache_mode", "Norm"),
        is_seq_lens_cumsum=params["is_seq_lens_cumsum"],
        out=[inputs["key_ref"], inputs["value_ref"]],
    )
    if inputs["seq_offset"] is not None:
        kwargs["seq_offset"] = inputs["seq_offset"]
    ops.npu_gather_pa_kv_cache(**kwargs)
    torch.npu.synchronize()

    npu_key = inputs["key_ref"].cpu()
    npu_value = inputs["value_ref"].cpu()

    ok1 = _compare_array("key_ref", npu_key, golden_key, params["dtype"])
    ok2 = _compare_array("value_ref", npu_value, golden_value, params["dtype"])
    return ok1 and ok2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=10)
    ap.add_argument("--case", type=int, default=-1, help="Test a single case if >=0")
    args = ap.parse_args()

    kernel_dir = resolve_kernel_dir()
    case_ids = [args.case] if args.case >= 0 else list(range(args.max_cases))

    results = {}
    all_ok = True
    for cid in case_ids:
        try:
            ok = run_case(cid, kernel_dir)
            results[cid] = ok
            all_ok = all_ok and ok
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else repr(e)
            results[cid] = False
            all_ok = False
            print(f"[ERROR] case {cid}: {msg}")
            if ("executor is nullptr" in msg or "acl api failed" in msg
                    or "aclnnGatherPaKvCache" in msg):
                print("KERNEL_ERROR - gather_pa_kv_cache kernel failed (see collect log)")

    print("\n" + "=" * 50)
    for cid in case_ids:
        print(f"case {cid}: {'PASS' if results.get(cid) else 'FAIL'}")
    if all_ok:
        print("\nALL CHECKS PASSED!")
        sys.exit(0)
    print("\nREGRESSION DETECTED (or kernel error)!")
    sys.exit(1)


if __name__ == "__main__":
    main()
