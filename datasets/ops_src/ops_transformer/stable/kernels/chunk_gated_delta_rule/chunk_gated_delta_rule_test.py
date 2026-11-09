#!/usr/bin/env python3
"""
Regression test for chunk_gated_delta_rule NPU operator.
Compares NPU output against pre-collected golden data.

In sandbox: golden files are in cwd-relative golden_output_case_0/
In local runs: golden files are in script-relative golden_output_case_0/
"""

import os
import sys
import json
import argparse

import torch
import torch_npu  # noqa: F401

import gtsrusops as ops

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# In sandbox: __file__ = /tmp/tmpXXX/tmpXXXXXX.py (script renamed to tmp pattern)
#   => SCRIPT_DIR = CWD = tmp_dir, operator dir is copied here with golden data
#   => golden data at: SCRIPT_DIR/chunk_gated_delta_rule/golden_output_case_0/
# In local: __file__ = .../kernels/chunk_gated_delta_rule/test.py
#   => SCRIPT_DIR = .../kernels/chunk_gated_delta_rule/, golden data at SCRIPT_DIR/golden_output_case_0/
if os.path.basename(__file__).startswith("tmp"):
    # Sandbox: operator dir (with golden data) is copied to CWD
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "chunk_gated_delta_rule", "golden_output_case_0")
    INPUT_DIR = os.path.join(SCRIPT_DIR, "chunk_gated_delta_rule", "golden_input_case_0")
else:
    # Local: script has original name, SCRIPT_DIR = script dir
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "golden_output_case_0")
    INPUT_DIR = os.path.join(SCRIPT_DIR, "golden_input_case_0")


def generate_inputs(seed=2026):
    """Generate test inputs on NPU."""
    torch.manual_seed(seed)

    B, seqlen, nk, nv, dk, dv = 1, 64, 4, 4, 128, 128
    chunk_size, scale = 64, 1.0 / (dk**0.5)

    T = seqlen
    DEVICE = "npu"
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=DEVICE)
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=DEVICE)
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16, device=DEVICE)
    beta = torch.rand((T, nv), dtype=torch.bfloat16, device=DEVICE)
    g = torch.rand((T, nv), dtype=torch.float32, device=DEVICE) * -1.0
    initial_state = torch.zeros((B, nv, dv, dk), dtype=torch.bfloat16, device=DEVICE)
    actual_seq_lengths = torch.tensor([seqlen], dtype=torch.int32, device=DEVICE)

    # L2 normalize q and k
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)

    return {
        "query": query, "key": key, "value": value,
        "beta": beta, "g": g,
        "initial_state": initial_state,
        "actual_seq_lengths": actual_seq_lengths,
    }, {
        "B": B, "seqlen": seqlen, "nk": nk, "nv": nv,
        "dk": dk, "dv": dv, "chunk_size": chunk_size,
        "scale": scale,
    }


def run_test():
    """Run regression test: compare NPU output vs golden data."""
    import numpy as np

    # Load golden data (CPU tensors saved by collect script)
    golden_out = torch.load(os.path.join(OUTPUT_DIR, "out.pt"), weights_only=False)
    golden_state = torch.load(os.path.join(OUTPUT_DIR, "final_state.pt"), weights_only=False)

    inputs, params = generate_inputs()

    T, nv, dv = params["seqlen"], params["nv"], params["dv"]
    B = params["B"]
    DEVICE = "npu"
    out_tmpls = [
        torch.empty((T, nv, dv), dtype=torch.bfloat16, device=DEVICE),
        torch.empty((B, nv, dv, params["dk"]), dtype=torch.bfloat16, device=DEVICE),
    ]

    print("Running NPU operator for test...")
    npu_out, npu_state = ops.npu_chunk_gated_delta_rule(
        inputs["query"], inputs["key"], inputs["value"], inputs["beta"],
        inputs["initial_state"], inputs["actual_seq_lengths"], inputs["g"],
        params["scale"], out_tmpls,
    )
    torch.npu.synchronize()

    # Compare
    def compare(name, ref, cur, atol=1e-4):
        ref_f = ref.float().cpu().detach().numpy().flatten()
        cur_f = cur.float().cpu().detach().numpy().flatten()
        diff = np.abs(ref_f - cur_f)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        ok = max_diff <= atol
        print(f"{name}: {'PASS' if ok else 'FAIL'} (max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f})")
        if not ok:
            top_idx = np.argsort(diff)[-5:][::-1]
            for i in top_idx:
                print(f"  [{i}] golden={ref_f[i]:.6f}, npu={cur_f[i]:.6f}, diff={diff[i]:.6f}")
        return ok

    print("\n=== Regression Test: NPU vs Golden ===")
    ok1 = compare("out", golden_out, npu_out)
    ok2 = compare("final_state", golden_state, npu_state)

    if not (ok1 and ok2):
        print("\nREGRESSION DETECTED!")
        sys.exit(1)
    print("\nALL CHECKS PASSED!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run regression test")
    args = parser.parse_args()

    # Default: run test
    run_test()


if __name__ == "__main__":
    main()