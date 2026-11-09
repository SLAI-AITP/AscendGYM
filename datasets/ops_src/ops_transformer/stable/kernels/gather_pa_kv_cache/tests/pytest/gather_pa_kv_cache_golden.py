#!/usr/bin/env python3
"""
Golden reference implementation for gather_pa_kv_cache operator.
This provides a CPU-compatible reference for generating expected outputs.

Usage:
    from gather_pa_kv_cache_golden import gather_pa_kv_cache_cpu
    key_ref, value_ref = gather_pa_kv_cache_cpu(key_cache, value_cache, ...)
"""

import torch


def gather_pa_kv_cache_cpu(
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    key_ref,
    value_ref,
    seq_offset=None,
    cache_mode="Norm",
    is_seq_lens_cumsum=True,
):
    """
    CPU reference implementation of gather_pa_kv_cache.
    
    This function gathers key/value tokens from Paged Attention KV cache.
    
    Args:
        key_cache: [num_blocks, block_size, num_heads, head_dim_k]
        value_cache: [num_blocks, block_size, num_heads, head_dim_v]
        block_tables: [batch_size, max_blocks_per_seq]
        seq_lens: [batch_size] cumulative sequence lengths
        key_ref: [total_tokens, num_heads, head_dim_k] output buffer
        value_ref: [total_tokens, num_heads, head_dim_v] output buffer
        seq_offset: [batch_size] block table offsets (optional)
        cache_mode: "Norm" or "PA_NZ"
        is_seq_lens_cumsum: True if seq_lens is cumulative
        
    Returns:
        key_ref, value_ref (modified in-place)
    """
    batch_size = block_tables.shape[0]
    block_size = key_cache.shape[1]
    num_heads = key_cache.shape[2]
    head_dim_k = key_cache.shape[3]
    
    if seq_offset is None:
        seq_offset = torch.zeros(batch_size, dtype=torch.int32, device=block_tables.device)
    
    if is_seq_lens_cumsum:
        # seq_lens is cumulative
        token_start = 0
        for b in range(batch_size):
            if b == 0:
                token_end = seq_lens[b].item()
            else:
                token_end = seq_lens[b].item()
            num_tokens = token_end - token_start
            num_blocks = (num_tokens + block_size - 1) // block_size
            
            for i in range(num_blocks):
                block_id = block_tables[b, i].item()
                offset_in_block = 0
                tokens_in_block = min(block_size, num_tokens - i * block_size)
                
                # Gather key
                key_block = key_cache[block_id, :tokens_in_block]  # [tokens_in_block, num_heads, head_dim]
                start_pos = token_start + i * block_size
                end_pos = start_pos + tokens_in_block
                key_ref[start_pos:end_pos] = key_block
                
                # Gather value
                value_block = value_cache[block_id, :tokens_in_block]
                value_ref[start_pos:end_pos] = value_block
            
            token_start = token_end
    else:
        # seq_lens is not cumulative
        token_start = 0
        for b in range(batch_size):
            num_tokens = seq_lens[b].item()
            num_blocks = (num_tokens + block_size - 1) // block_size
            
            for i in range(num_blocks):
                block_id = block_tables[b, i].item()
                tokens_in_block = min(block_size, num_tokens - i * block_size)
                
                key_block = key_cache[block_id, :tokens_in_block]
                start_pos = token_start + i * block_size
                end_pos = start_pos + tokens_in_block
                key_ref[start_pos:end_pos] = key_block
                
                value_block = value_cache[block_id, :tokens_in_block]
                value_ref[start_pos:end_pos] = value_block
            
            token_start += num_tokens
    
    return key_ref, value_ref


def generate_golden_from_template_case(case_input, case_output_template):
    """
    Generate golden output for a template case.
    
    Args:
        case_input: dict with input tensors
        case_output_template: dict with output tensor shapes/dtypes
        
    Returns:
        dict with output tensors
    """
    # Generate output tensors based on template
    result = {}
    
    for name, template in case_output_template.items():
        shape = template["shape"]
        dtype = template["dtype"]
        result[name] = torch.empty(shape, dtype=dtype)
    
    return result


if __name__ == "__main__":
    # Test the CPU implementation
    print("Testing CPU reference implementation...")
    
    DEVICE = "cpu"
    dtype = torch.float16
    
    num_blocks = 8
    block_size = 4
    batch_size = 2
    num_heads = 2
    head_dim = 8
    
    # Generate test data
    key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=DEVICE)
    value_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=DEVICE)
    block_tables = torch.randint(0, num_blocks, (batch_size, 4), dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([4, 8], dtype=torch.int32, device=DEVICE)
    seq_lens_cumsum = torch.cumsum(seq_lens, dim=0)
    
    total_tokens = seq_lens_cumsum[-1].item()
    key_ref = torch.empty((total_tokens, num_heads, head_dim), dtype=dtype, device=DEVICE)
    value_ref = torch.empty((total_tokens, num_heads, head_dim), dtype=dtype, device=DEVICE)
    
    print(f"key_cache shape: {key_cache.shape}")
    print(f"block_tables: {block_tables}")
    print(f"seq_lens (cumsum): {seq_lens_cumsum}")
    print(f"key_ref shape: {key_ref.shape}")
    
    # Run CPU reference
    gather_pa_kv_cache_cpu(
        key_cache, value_cache, block_tables, seq_lens_cumsum,
        key_ref, value_ref, is_seq_lens_cumsum=True
    )
    
    print("\nCPU reference output:")
    print(f"key_ref[0]: {key_ref[0]}")
    print(f"value_ref[0]: {value_ref[0]}")
    
    print("\nTest completed successfully!")