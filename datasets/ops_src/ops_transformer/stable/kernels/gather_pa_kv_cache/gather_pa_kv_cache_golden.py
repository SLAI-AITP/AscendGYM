#!/usr/bin/env python3
"""
CPU reference implementation of gather_pa_kv_cache.

Used to compute independent golden data for the AscendC operator regression
tests (instead of relying solely on NPU output, which may be unavailable or
incorrect while the operator is broken). Source of truth logic adapted from
the repo's tests:
    runtime/ascend_c/ops-transformer/attention/gather_pa_kv_cache/tests/pytest/

GatherPaKvCache semantics (Norm mode):
  * key_cache    : [num_blocks, block_size, num_heads, head_dim_k]
  * value_cache  : [num_blocks, block_size, num_heads, head_dim_v]
  * block_tables : [batch_size, max_blocks_per_seq]
  * seq_lens     : [batch_size]
      - if is_seq_lens_cumsum=True : cumulative lengths, output dim0 = seq_lens[-1]
      - if is_seq_lens_cumsum=False: per-batch lengths,   output dim0 = sum(seq_lens)
  * seq_offset   : [batch_size] optional block-table offset (blocks), same unit as
                   the paged-attention block index.
  * key_ref / value_ref : output buffers, written in-place:
      key_ref   : [total_tokens, num_heads, head_dim_k]
      value_ref : [total_tokens, num_heads, head_dim_v]
"""

import torch


def _tokens_in_block(num_tokens, i, block_size):
    return min(block_size, num_tokens - i * block_size)


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
    """Gather key/value tokens from paged KV cache. Writes into key_ref/value_ref."""
    if cache_mode is None or cache_mode == "":
        cache_mode = "Norm"
    if cache_mode != "Norm":
        raise ValueError(f"CPU reference only supports cache_mode='Norm', got {cache_mode}")

    batch_size = block_tables.shape[0]
    block_size = key_cache.shape[1]

    if seq_offset is None:
        seq_offset = torch.zeros(batch_size, dtype=torch.int32, device=block_tables.device)

    # Kernel semantics (op_kernel/gather_pa_kv_cache_nd.h): when is_seq_lens_cumsum=True,
    # seq_lens is a PREFIX-SUM array of length batch+1:
    #     seq_lens = [S0, S0+l0, S0+l0+l1, ..., S0+sum(l)]   (S0 = initial output offset)
    # The kernel reads tokenStart = seq_lens[0] and tokenEnd = seq_lens[b+1] for seq b.
    token_start = int(seq_lens[0].item()) if is_seq_lens_cumsum else 0
    for b in range(batch_size):
        if is_seq_lens_cumsum:
            token_end = int(seq_lens[b + 1].item())
            num_tokens = token_end - token_start
        else:
            num_tokens = int(seq_lens[b].item())
            token_end = token_start + num_tokens

        num_blocks_for_seq = (num_tokens + block_size - 1) // block_size

        for i in range(num_blocks_for_seq):
            block_id = int(block_tables[b, i].item())
            tok = _tokens_in_block(num_tokens, i, block_size)

            # block_tables[b, i] directly indexes key_cache first dim (a paged block).
            key_ref[token_start + i * block_size:
                    token_start + i * block_size + tok] = key_cache[block_id, :tok]
            value_ref[token_start + i * block_size:
                      token_start + i * block_size + tok] = value_cache[block_id, :tok]

        token_start = token_end

    return key_ref, value_ref


def build_case_params(case_id):
    """
    Deterministic parameters for each test case, mirroring template/GatherPaKvCache.json.
    seq_lens holds the lengths in 'length' form (non-cumulative). The caller converts to
    cumulative when is_seq_lens_cumsum=True. The returned total_tokens is the output dim0.
    """
    cases = [
        # case 0: fp16, typical
        dict(dtype=torch.float16, num_blocks=128, block_size=16, batch=4, num_heads=16,
             head_dim_k=128, head_dim_v=128, max_blocks=32,
             lens=[16, 32, 48, 64], is_seq_lens_cumsum=True, seed=2026, has_seq_offset=False),
        # case 1: fp16, different dims
        dict(dtype=torch.float16, num_blocks=64, block_size=32, batch=2, num_heads=8,
             head_dim_k=64, head_dim_v=64, max_blocks=16,
             lens=[32, 64], is_seq_lens_cumsum=True, seed=42, has_seq_offset=False),
        # case 2: bf16, larger
        dict(dtype=torch.bfloat16, num_blocks=256, block_size=8, batch=8, num_heads=32,
             head_dim_k=64, head_dim_v=64, max_blocks=64,
             lens=[32] * 8, is_seq_lens_cumsum=True, seed=123, has_seq_offset=False),
        # case 3: bf16 (op supports fp16/bf16/int8 only) + seq_offset
        dict(dtype=torch.bfloat16, num_blocks=128, block_size=16, batch=4, num_heads=16,
             head_dim_k=64, head_dim_v=64, max_blocks=32,
             lens=[16, 16, 16, 16], is_seq_lens_cumsum=True, seed=7, has_seq_offset=True),
        # case 4: fp16, non-cumulative seq_lens
        dict(dtype=torch.float16, num_blocks=128, block_size=16, batch=4, num_heads=16,
             head_dim_k=128, head_dim_v=128, max_blocks=32,
             lens=[16, 32, 48, 64], is_seq_lens_cumsum=False, seed=2026, has_seq_offset=False),
        # case 5: fp16, int64 index (block_tables/seq_lens int64) + seq_offset
        dict(dtype=torch.float16, num_blocks=512, block_size=4, batch=16, num_heads=8,
             head_dim_k=128, head_dim_v=128, max_blocks=128,
             lens=[32] * 16, is_seq_lens_cumsum=True, seed=99, has_seq_offset=True, index_dtype=torch.int64),
        # case 6: fp16, batch=1
        dict(dtype=torch.float16, num_blocks=128, block_size=16, batch=1, num_heads=16,
             head_dim_k=128, head_dim_v=128, max_blocks=64,
             lens=[64], is_seq_lens_cumsum=True, seed=5, has_seq_offset=False),
        # case 7: fp16, asymmetric head dims (k=144, v=128)
        dict(dtype=torch.float16, num_blocks=256, block_size=16, batch=8, num_heads=16,
             head_dim_k=144, head_dim_v=128, max_blocks=64,
             lens=[64] * 8, is_seq_lens_cumsum=True, seed=77, has_seq_offset=False),
        # case 8: int8
        dict(dtype=torch.int8, num_blocks=128, block_size=32, batch=4, num_heads=16,
             head_dim_k=64, head_dim_v=64, max_blocks=32,
             lens=[64, 64, 64, 64], is_seq_lens_cumsum=True, seed=13, has_seq_offset=False),
        # case 9: fp16, PA_NZ mode (norm-style gather layout for CPU-ref parity)
        dict(dtype=torch.float16, num_blocks=128, block_size=16, batch=4, num_heads=16,
             head_dim_k=128, head_dim_v=128, max_blocks=32,
             lens=[16, 32, 48, 64], is_seq_lens_cumsum=True, seed=2026, has_seq_offset=False,
             cache_mode="Norm"),
    ]
    c = cases[case_id].copy()
    c["total_tokens"] = sum(c["lens"])
    return c


if __name__ == "__main__":
    torch.manual_seed(2026)
    DEVICE = "cpu"
    p = build_case_params(0)
    kc = torch.randn(p["num_blocks"], p["block_size"], p["num_heads"], p["head_dim_k"], dtype=p["dtype"])
    vc = torch.randn(p["num_blocks"], p["block_size"], p["num_heads"], p["head_dim_v"], dtype=p["dtype"])
    bt = torch.randint(0, p["num_blocks"], (p["batch"], p["max_blocks"]), dtype=torch.int32)
    lens = torch.tensor(p["lens"], dtype=torch.int32)
    if p["is_seq_lens_cumsum"]:
        lens = torch.cumsum(lens, dim=0)
    total = int(lens[-1].item()) if p["is_seq_lens_cumsum"] else p["total_tokens"]
    kr = torch.empty((total, p["num_heads"], p["head_dim_k"]), dtype=p["dtype"])
    vr = torch.empty((total, p["num_heads"], p["head_dim_v"]), dtype=p["dtype"])
    gather_pa_kv_cache_cpu(kc, vc, bt, lens, kr, vr, is_seq_lens_cumsum=p["is_seq_lens_cumsum"])
    print("CPU reference self-test OK. key_ref shape:", tuple(kr.shape))
