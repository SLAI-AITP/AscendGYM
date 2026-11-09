#!/usr/bin/env python3
"""
Tensor data generation utilities for gather_pa_kv_cache tests.
"""

import torch
import numpy as np


def gen_tensor_data(params, tensor_name):
    """
    Generate random tensor data based on template parameters.
    
    Args:
        params: dict with shape, dtype information
        tensor_name: name of the tensor to generate
        
    Returns:
        torch.Tensor with random data
    """
    shape = params["shape_input"].get(tensor_name)
    if shape is None:
        shape = params["shape_output"].get(tensor_name)
    
    if shape is None:
        raise ValueError(f"Unknown tensor: {tensor_name}")
    
    dtype_str = params["dtype_input"].get(tensor_name, "fp16")
    dtype = _str_to_dtype(dtype_str)
    
    tensor = torch.randn(shape, dtype=dtype)
    return tensor


def _str_to_dtype(dtype_str):
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }
    return dtype_map.get(dtype_str, torch.float16)


def generate_block_tables(num_blocks, batch_size, max_blocks_per_seq, device="cpu"):
    """Generate random block table mapping."""
    return torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq), 
                         dtype=torch.int32, device=device)


def generate_seq_lens_cumsum(batch_size, block_size, device="cpu"):
    """Generate cumulative sequence lengths."""
    seq_lens = torch.tensor([block_size * (i + 1) for i in range(batch_size)], 
                            dtype=torch.int32, device=device)
    return torch.cumsum(seq_lens, dim=0)


def generate_test_params(case_id=0):
    """Generate test parameters for different cases."""
    cases = [
        # Case 0: Basic fp16 test
        {
            "num_blocks": 128,
            "block_size": 16,
            "batch_size": 4,
            "num_heads": 16,
            "head_dim_k": 128,
            "head_dim_v": 128,
            "max_blocks_per_seq": 32,
            "dtype": "fp16",
        },
        # Case 1: bf16 test
        {
            "num_blocks": 64,
            "block_size": 32,
            "batch_size": 2,
            "num_heads": 8,
            "head_dim_k": 64,
            "head_dim_v": 64,
            "max_blocks_per_seq": 16,
            "dtype": "bf16",
        },
        # Case 2: fp32 test
        {
            "num_blocks": 256,
            "block_size": 8,
            "batch_size": 8,
            "num_heads": 32,
            "head_dim_k": 64,
            "head_dim_v": 64,
            "max_blocks_per_seq": 64,
            "dtype": "fp32",
        },
    ]
    
    return cases[case_id % len(cases)]


if __name__ == "__main__":
    # Test data generation
    params = generate_test_params(0)
    print("Generated params:", params)
    
    block_tables = generate_block_tables(params["num_blocks"], params["batch_size"], 
                                         params["max_blocks_per_seq"])
    print("block_tables shape:", block_tables.shape)
    print("block_tables:", block_tables)
    
    seq_lens = generate_seq_lens_cumsum(params["batch_size"], params["block_size"])
    print("seq_lens (cumsum):", seq_lens)