#!/usr/bin/env python3
"""Entry point for collecting golden data via --collect flag."""

import os
import sys
import json
import argparse

import torch
import torch_npu  # noqa: F401

import gtsrusops as ops

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Save relative to sandbox cwd (tmp_dir) so --save-file can find them.
# These dirs are created in the sandbox's working directory.
OUTPUT_DIR = "golden_output_case_0"
INPUT_DIR = "golden_input_case_0"


def generate_inputs(seed=2026):
    """Generate test inputs on NPU."""
    torch.manual_seed(seed)

    B, seqlen, nk, nv, dk, dv = 1, 64, 4, 4, 128, 128
    chunk_size, scale = 64, 1.0 / (dk**0.5)

    T = seqlen
    # Allocate directly on NPU device
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


def collect_golden():
    """Run NPU op and save outputs as golden data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    inputs, params = generate_inputs()

    # Save input tensors (copy from NPU to CPU for local storage)
    for name, tensor in inputs.items():
        torch.save(tensor.cpu(), os.path.join(INPUT_DIR, f"{name}.pt"))
    print(f"Saved inputs to: {INPUT_DIR}")

    # Build output templates on NPU
    T, nv, dv = params["seqlen"], params["nv"], params["dv"]
    B = params["B"]
    DEVICE = "npu"
    out_tmpls = [
        torch.empty((T, nv, dv), dtype=torch.bfloat16, device=DEVICE),
        torch.empty((B, nv, dv, params["dk"]), dtype=torch.bfloat16, device=DEVICE),
    ]

    print("Running NPU operator...")
    # Inputs are already on NPU (allocated in generate_inputs)

    golden_out, golden_state = ops.npu_chunk_gated_delta_rule(
        inputs["query"], inputs["key"], inputs["value"], inputs["beta"],
        inputs["initial_state"], inputs["actual_seq_lengths"], inputs["g"],
        params["scale"], out_tmpls,
    )
    torch.npu.synchronize()

    # =============================================================================
    # Save to relative path so --save-file can find them:
    # collect_golden.sh requests: golden_output_case_0/out.pt, golden_output_case_0/final_state.pt
    # =============================================================================
    out_path = os.path.join(OUTPUT_DIR, "out.pt")
    state_path = os.path.join(OUTPUT_DIR, "final_state.pt")
    try:
        torch.save(golden_out.cpu(), out_path)
        print(f"SAVED out.pt: {os.path.getsize(out_path)} bytes", flush=True)
    except Exception as e:
        print(f"ERROR saving out.pt: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
    try:
        torch.save(golden_state.cpu(), state_path)
        print(f"SAVED final_state.pt: {os.path.getsize(state_path)} bytes", flush=True)
    except Exception as e:
        print(f"ERROR saving final_state.pt: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
    print(f"Saved golden outputs to: {OUTPUT_DIR}")
    print(f"  out.pt: {tuple(golden_out.shape)}")
    print(f"  final_state.pt: {tuple(golden_state.shape)}")

    # Save params
    params_path = os.path.join(SCRIPT_DIR, "golden_params_case_0.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print("Golden data collection complete!")


if __name__ == "__main__":
    collect_golden()