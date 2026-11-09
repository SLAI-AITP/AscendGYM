#!/usr/bin/env python3
"""
Golden data collection script for gather_pa_kv_cache NPU operator.
This script generates golden input and output data for regression testing.

In sandbox: data is saved to cwd-relative directories
In local runs: data is saved relative to this script's directory

Usage:
    python gather_pa_kv_cache_collect.py --case 0

This generates:
    golden_input_case_0/input.pt
    golden_output_case_0/key_ref.pt
    golden_output_case_0/value_ref.pt
"""

import os
import sys
import argparse

import torch
import torch_npu  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path resolution: in sandbox vs local
if os.path.basename(__file__).startswith("tmp"):
    # Sandbox: script renamed to tmp-like name, SCRIPT_DIR = CWD
    KERNEL_DIR = os.path.join(SCRIPT_DIR, "gather_pa_kv_cache")
else:
    # Local: script in tests/pytest/, go up to kernel dir
    KERNEL_DIR = os.path.join(SCRIPT_DIR, "..", "..")

INPUT_DIR = os.path.join(KERNEL_DIR, "golden_input_case_0")
OUTPUT_DIR = os.path.join(KERNEL_DIR, "golden_output_case_0")


def generate_case_params(case_id):
    """Generate test parameters for different cases."""
    cases = [
        # Case 0: Basic test with typical parameters
        {
            "num_blocks": 128,
            "block_size": 16,
            "batch_size": 4,
            "num_heads": 16,
            "head_dim_k": 128,
            "head_dim_v": 128,
            "max_blocks_per_seq": 32,
            "is_seq_lens_cumsum": True,
            "seed": 2026,
        },
        # Case 1: Different head dimensions
        {
            "num_blocks": 64,
            "block_size": 32,
            "batch_size": 2,
            "num_heads": 8,
            "head_dim_k": 64,
            "head_dim_v": 64,
            "max_blocks_per_seq": 16,
            "is_seq_lens_cumsum": True,
            "seed": 42,
        },
        # Case 2: Larger configuration
        {
            "num_blocks": 256,
            "block_size": 8,
            "batch_size": 8,
            "num_heads": 32,
            "head_dim_k": 64,
            "head_dim_v": 64,
            "max_blocks_per_seq": 64,
            "is_seq_lens_cumsum": True,
            "seed": 123,
        },
    ]
    
    if case_id >= len(cases):
        case_id = 0  # Default to case 0
    
    return cases[case_id]


def collect_golden_data(case_id=0, dtype=torch.float16):
    """Collect golden input and output data for the given case."""
    
    params = generate_case_params(case_id)
    torch.manual_seed(params["seed"])
    
    DEVICE = "npu"
    
    num_blocks = params["num_blocks"]
    block_size = params["block_size"]
    batch_size = params["batch_size"]
    num_heads = params["num_heads"]
    head_dim_k = params["head_dim_k"]
    head_dim_v = params["head_dim_v"]
    max_blocks_per_seq = params["max_blocks_per_seq"]
    
    print(f"Collecting golden data for case {case_id}")
    print(f"Parameters: num_blocks={num_blocks}, block_size={block_size}, "
          f"batch_size={batch_size}, num_heads={num_heads}")
    
    # Generate keyCache and valueCache (4D tensors)
    key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim_k, 
                            dtype=dtype, device=DEVICE)
    value_cache = torch.randn(num_blocks, block_size, num_heads, head_dim_v, 
                              dtype=dtype, device=DEVICE)
    
    # Generate block_tables
    block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq), 
                                 dtype=torch.int32, device=DEVICE)
    
    # seq_lens: sequence lengths for each batch (with cumsum)
    seq_lens_list = [block_size * (i + 1) for i in range(batch_size)]
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    seq_lens_cumsum = torch.cumsum(seq_lens, dim=0)
    
    # Allocate output tensors
    total_tokens = seq_lens_cumsum[-1].item()
    key_ref = torch.empty((total_tokens, num_heads, head_dim_k), 
                          dtype=dtype, device=DEVICE)
    value_ref = torch.empty((total_tokens, num_heads, head_dim_v), 
                            dtype=dtype, device=DEVICE)
    
    # seq_offset
    seq_offset = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
    
    print(f"Input shapes:")
    print(f"  key_cache: {key_cache.shape}")
    print(f"  value_cache: {value_cache.shape}")
    print(f"  block_tables: {block_tables.shape}")
    print(f"  seq_lens: {seq_lens_cumsum.shape}")
    print(f"  key_ref (output): {key_ref.shape}")
    print(f"  value_ref (output): {value_ref.shape}")
    
    # Run the NPU operator to generate golden output
    print("Running NPU operator to generate golden output...")
    torch_npu.npu_gather_pa_kv_cache(
        key_cache,
        value_cache,
        block_tables,
        seq_lens_cumsum,
        seq_offset=seq_offset,
        key=key_ref,
        value=value_ref,
    )
    torch.npu.synchronize()
    
    # Create output directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save input data (on CPU for portability)
    input_data = {
        "key_cache": key_cache.cpu(),
        "value_cache": value_cache.cpu(),
        "block_tables": block_tables.cpu(),
        "seq_lens": seq_lens_cumsum.cpu(),
        "seq_offset": seq_offset.cpu(),
        "params": params,
    }
    input_path = os.path.join(INPUT_DIR, "input.pt")
    torch.save(input_data, input_path)
    print(f"Saved golden input to {input_path}")
    
    # Save output data (on CPU)
    output_data = {
        "key_ref": key_ref.cpu(),
        "value_ref": value_ref.cpu(),
    }
    output_path_key = os.path.join(OUTPUT_DIR, "key_ref.pt")
    output_path_value = os.path.join(OUTPUT_DIR, "value_ref.pt")
    torch.save(key_ref.cpu(), output_path_key)
    torch.save(value_ref.cpu(), output_path_value)
    print(f"Saved golden output to {OUTPUT_DIR}")
    
    # Save metadata
    metadata = {
        "case_id": case_id,
        "params": params,
        "total_tokens": total_tokens,
        "input_shape": {
            "key_cache": list(key_cache.shape),
            "value_cache": list(value_cache.shape),
            "block_tables": list(block_tables.shape),
            "seq_lens": list(seq_lens_cumsum.shape),
        },
        "output_shape": {
            "key_ref": list(key_ref.shape),
            "value_ref": list(value_ref.shape),
        },
    }
    metadata_path = os.path.join(INPUT_DIR, "metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print("Golden data collection completed successfully!")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Collect golden data for gather_pa_kv_cache")
    parser.add_argument("--case", type=int, default=0, help="Case ID (0-2)")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type (float16, bfloat16)")
    args = parser.parse_args()
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.float16)
    
    collect_golden_data(case_id=args.case, dtype=dtype)


if __name__ == "__main__":
    main()