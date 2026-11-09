#!/usr/bin/env python3
"""
Regression test for gather_pa_kv_cache NPU operator.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# In sandbox: __file__ = /tmp/tmpXXX/tmpXXXXXX.py (script renamed to tmp pattern)
#   => SCRIPT_DIR = CWD = tmp_dir, golden data is at gather_pa_kv_cache/golden_output_case_0/
# In local: __file__ = .../kernels/gather_pa_kv_cache/tests/pytest/main.py
#   => SCRIPT_DIR = .../kernels/gather_pa_kv_cache/tests/pytest/, golden data at ../../golden_output_case_0/
if os.path.basename(__file__).startswith("tmp"):
    # Sandbox: script renamed to tmp-like name, SCRIPT_DIR = CWD = project root
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "gather_pa_kv_cache", "golden_output_case_0")
    INPUT_DIR = os.path.join(SCRIPT_DIR, "gather_pa_kv_cache", "golden_input_case_0")
else:
    # Local: script has original name, SCRIPT_DIR = script dir
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "golden_output_case_0")
    INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "golden_input_case_0")


def generate_inputs(seed=2026):
    """Generate test inputs on NPU."""
    torch.manual_seed(seed)

    # Default test case parameters
    # keyCache shape: [num_blocks, block_size, num_heads, head_dim_k]
    # valueCache shape: [num_blocks, block_size, num_heads, head_dim_v]
    # blockTables shape: [batch_size, max_blocks_per_seq]
    # seqLens shape: [batch_size] (if is_seq_lens_cumsum=True, this is cumulative)
    
    DEVICE = "npu"
    dtype = torch.float16
    
    num_blocks = 128
    block_size = 16
    batch_size = 4
    num_heads = 16
    head_dim_k = 128
    head_dim_v = 128
    max_blocks_per_seq = 32
    
    # Generate keyCache and valueCache (4D tensors)
    key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim_k, 
                            dtype=dtype, device=DEVICE)
    value_cache = torch.randn(num_blocks, block_size, num_heads, head_dim_v, 
                              dtype=dtype, device=DEVICE)
    
    # Generate block_tables: map each position to a block
    block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq), 
                                 dtype=torch.int32, device=DEVICE)
    
    # seq_lens: sequence lengths for each batch
    # With cumsum, values are cumulative (e.g., [64, 128, 192, 256])
    seq_lens = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device=DEVICE)
    seq_lens_cumsum = torch.cumsum(seq_lens, dim=0)  # [16, 48, 96, 160]
    
    # key_ref and value_ref (output tensors)
    total_tokens = seq_lens_cumsum[-1].item()  # 160
    key_ref = torch.empty((total_tokens, num_heads, head_dim_k), 
                          dtype=dtype, device=DEVICE)
    value_ref = torch.empty((total_tokens, num_heads, head_dim_v), 
                            dtype=dtype, device=DEVICE)
    
    # seq_offset (optional, for block table offset)
    seq_offset = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
    
    return {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens_cumsum,
        "key_ref": key_ref,
        "value_ref": value_ref,
        "seq_offset": seq_offset,
    }, {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim_k": head_dim_k,
        "head_dim_v": head_dim_v,
        "max_blocks_per_seq": max_blocks_per_seq,
        "is_seq_lens_cumsum": True,
    }


def run_test():
    """Run regression test: compare NPU output vs golden data."""
    import numpy as np

    # Load golden data (CPU tensors saved by collect script)
    golden_key_ref = torch.load(os.path.join(OUTPUT_DIR, "key_ref.pt"), weights_only=False)
    golden_value_ref = torch.load(os.path.join(OUTPUT_DIR, "value_ref.pt"), weights_only=False)

    inputs, params = generate_inputs()

    print("Running NPU operator for test...")
    # Run the NPU operator
    torch_npu.npu_gather_pa_kv_cache(
        inputs["key_cache"],
        inputs["value_cache"],
        inputs["block_tables"],
        inputs["seq_lens"],
        seq_offset=inputs["seq_offset"],
        key=inputs["key_ref"],
        value=inputs["value_ref"],
    )
    torch.npu.synchronize()

    npu_key_ref = inputs["key_ref"]
    npu_value_ref = inputs["value_ref"]

    # Compare
    def compare(name, ref, cur, atol=1e-3):
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
    ok1 = compare("key_ref", golden_key_ref, npu_key_ref)
    ok2 = compare("value_ref", golden_value_ref, npu_value_ref)

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