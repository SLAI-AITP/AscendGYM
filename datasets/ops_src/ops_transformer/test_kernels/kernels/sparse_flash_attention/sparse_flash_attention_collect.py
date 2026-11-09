#!/usr/bin/env python3
"""Collect golden data for sparse_flash_attention by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

SHAPE MODEL — from op_host/sparse_flash_attention_def.cpp + docs/aclnnSparseFlashAttention.md
+ tests/pytest/sparse_flash_attention_paramset.py (proven NPU configs). This is a specialised
MLA-absorb "flash attention" kernel: query layering is BSND, KV index is shared (N2=1).

  B=batch, S1=query seq len, S2=kv seq len, N1=query heads, N2=kv heads(=1, MLA),
  D=head dim (512), Dr=rope head dim (64), K=sparse size (num block slots).

  query              (B,S1,N1,D)        fp16|bf16      Q (MLA absorb: D=512)
  key                (B,S2,N2,D)        fp16|bf16      K base (D=512)
  value              (B,S2,N2,D)        fp16|bf16      V (== key in the suite's convention)
  sparse_indices     (B,S1,N2,blockcount) int32        block selection per query row, -1 = none
  block_table        (B,...)            int32 opt      only used for PA_BSND layout (None here)
  actual_seq_lengths_query (B,)         int32          per-batch query lengths
  actual_seq_lengths_kv    (B,)         int32          per-batch kv lengths
  query_rope         (B,S1,N1,Dr)       fp16|bf16      Q rotary (Dr=64)
  key_rope           (B,S2,N2,Dr)       fp16|bf16      K rotary (Dr=64)
  sinks              (N1,)              fp32 opt       dropped by aclnn V1 (passed as None)

  attrs: scale_value (float, default 1/sqrt(D+Dr) for MLA absorb), sparse_block_size (default 1),
    layout_query/layout_kv (BSND), sparse_mode (3 = local lower-tri, 0 = full mask),
    pre_tokens/next_tokens (INT64_MAX), attention_mode (2 = MLA-absorb),
    return_softmax_lse (bool)

  outputs (3 buffered templates, out_tmpls list):
    attention_out    (B,S1,N1,D)        fp16|bf16
    softmax_max      (B,N2,S1,N1/N2)    fp32   (only meaningful if return_softmax_lse)
    softmax_sum      (B,N2,S1,N1/N2)    fp32   (only meaningful if return_softmax_lse)

CONSTRAINTS (from docs + paramset): N2=1, D=512, Dr=64, attention_mode=2, layout BSND,
  sparse_block_size in [1,128] power of 2, K a multiple of sparse_block_size,
  actual_seq_lengths[i] <= S1 / S2 and > 0. sparse_indices entries in [0, ceil(seq/spb)-1]
  or -1 (see tests/pytest/sparse_flash_attention_golden.py::_generate_sparse_indices).

CALL (binding.py wrapper, op_def order): query,key,value,sparse_indices,block_table,
  actual_seq_lengths_query,actual_seq_lengths_kv,query_rope,key_rope,sinks,scale_value,
  sparse_block_size,layout_query,layout_kv,sparse_mode,pre_tokens,next_tokens,attention_mode,
  return_softmax_lse + out_tmpls (list of 3 buffers, OpDef output order).

# case_0: fp16 B1 S1=4 S2=8  N1=8   K16 mode3 lse=True  spb=1
# case_1: bf16 B2 S1=3 S2=6  N1=16  K32 mode0 lse=False spb=1
# case_2: fp16 B2 S1=4 S2=4  N1=8   K8  mode3 lse=True  spb=1
# case_3: bf16 B1 S1=8 S2=32 N1=32  K64 mode0 lse=False spb=1
# case_4: fp16 B3 S1=2 S2=8  N1=16  K16 mode3 lse=True  spb=1
# case_5: bf16 B1 S1=5 S2=16 N1=8   K16 mode3 lse=False spb=2 (non-default block size)
# case_6: fp16 B2 S1=6 S2=12 N1=64  K32 mode0 lse=False spb=1
# case_7: bf16 B2 S1=4 S2=16 N1=32  K32 mode0 lse=True  spb=1
# case_8: fp16 B1 S1=6 S2=6  N1=16  K8  mode3 lse=False spb=1
# case_9: bf16 B3 S1=8 S2=32 N1=16  K64 mode3 lse=True  spb=1
"""
import inspect
import json
import math
import os
import sys

import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "sparse_flash_attention"
OP_CAMEL = "SparseFlashAttention"
DEVICE = "npu:0"
SEED = 817
D = 512
DR = 64
N2 = 1
SCALE_DEFAULT = 1.0 / math.sqrt(D + DR)  # MLA-absorb: 1/sqrt(576)

# Positional order of the generated wrapper (binding.py), op_def order: inputs then attrs,
# excluding the trailing out_tmpls buffer list.
PARAM_NAMES = [
    "query", "key", "value", "sparse_indices",
    "block_table", "actual_seq_lengths_query", "actual_seq_lengths_kv",
    "query_rope", "key_rope", "sinks",
    "scale_value", "sparse_block_size", "layout_query", "layout_kv",
    "sparse_mode", "pre_tokens", "next_tokens", "attention_mode",
    "return_softmax_lse",
]
OUTPUT_KEYS = ["attention_out", "softmax_max", "softmax_sum"]
BLOCKCOUNT = "blockcount"  # key name for sparse_blockcount inside case dict


def _num_blocks(S2, sparse_block_size, sparse_mode, s_index, S1):
    """Number of valid KV blocks a query row may attend to (golden model)."""
    if sparse_mode == 0:
        threshold = S2
    elif sparse_mode == 3:
        threshold = S2 - S1 + s_index + 1
    else:
        raise ValueError("unsupported sparse_mode %r" % sparse_mode)
    if threshold <= 0:
        return 0
    return math.ceil(threshold / sparse_block_size)


def make_inputs(case):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED + case["id"])
    dt = torch.float16 if case["q_type"] == "fp16" else torch.bfloat16
    B = case["B"]
    S1, S2 = case["S1"], case["S2"]
    N1 = case["N1"]
    K = case["K"]
    spb = case["spb"]
    sparse_blockcount = K // spb
    sparse_mode = case["mode"]
    actual_q = [S1] * B
    actual_kv = [S2] * B

    # Q/K/V (MLA absorb, D=512): keep values small + finite so the kernel never sees NaN/Inf.
    query = (torch.randn((B, S1, N1, D), generator=g, device=DEVICE) * 0.3).to(dt)
    key = (torch.randn((B, S2, N2, D), generator=g, device=DEVICE) * 0.3).to(dt)
    value = key.clone()
    query_rope = (torch.randn((B, S1, N1, DR), generator=g, device=DEVICE) * 0.3).to(dt)
    key_rope = (torch.randn((B, S2, N2, DR), generator=g, device=DEVICE) * 0.3).to(dt)

    # sparse_indices: port of golden.py::_generate_sparse_indices for the BSND/BSND non-PA
    # layout. Build in BNSD [B,N2,S1,blockcount] then permute to the operator BSND layout.
    ind = torch.full((B, N2, S1, sparse_blockcount), -1, dtype=torch.int32, device=DEVICE)
    for b in range(B):
        for n in range(N2):
            for s in range(S1):
                valid_blocks_max = _num_blocks(S2, spb, sparse_mode, s, S1)
                if valid_blocks_max <= 0:
                    continue
                topk = min(valid_blocks_max, sparse_blockcount)
                perm = torch.randperm(valid_blocks_max - 1, generator=g, dtype=torch.int32,
                                      device=DEVICE)
                ind[b, n, s, : topk - 1] = perm[: topk - 1]
                ind[b, n, s, topk - 1] = valid_blocks_max - 1
    sparse_indices = ind.permute(0, 2, 1, 3).contiguous()  # [B,S1,N2,blockcount]

    actual_seq_lengths_query = torch.tensor(actual_q, dtype=torch.int32, device=DEVICE)
    actual_seq_lengths_kv = torch.tensor(actual_kv, dtype=torch.int32, device=DEVICE)

    return {
        "query": query,
        "key": key,
        "value": value,
        "sparse_indices": sparse_indices,
        "block_table": None,  # non-PA: aclnn drops / ignores it
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "sinks": None,  # aclnn V1 drops 'sinks' — never reaching the kernel
        "scale_value": SCALE_DEFAULT,
        "sparse_block_size": int(spb),
        "layout_query": "BSND",
        "layout_kv": "BSND",
        "sparse_mode": int(sparse_mode),
        "pre_tokens": (1 << 63) - 1,
        "next_tokens": (1 << 63) - 1,
        "attention_mode": 2,
        "return_softmax_lse": bool(case["lse"]),
    }


def make_output_bufs(case):
    """Three buffered outputs (OpDef order): attention_out, softmax_max, softmax_sum."""
    dt = torch.float16 if case["q_type"] == "fp16" else torch.bfloat16
    B, S1, S2, N1 = case["B"], case["S1"], case["S2"], case["N1"]
    out = torch.zeros((B, S1, N1, D), dtype=dt, device=DEVICE)
    sm = torch.zeros((B, N2, S1, N1 // N2), dtype=torch.float32, device=DEVICE)
    return [out, sm, sm.clone()]


def build_call_args(case):
    inputs = make_inputs(case)
    out_bufs = make_output_bufs(case)
    call_args = [inputs[p] for p in PARAM_NAMES] + [out_bufs]
    return call_args, inputs, out_bufs


def collect_case(case):
    call_args, inputs, out_bufs = build_call_args(case)
    outputs = ops.npu_sparse_flash_attention(*call_args)
    torch.npu.synchronize()

    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    else:
        outputs = list(outputs)
    if len(outputs) != len(OUTPUT_KEYS):
        raise RuntimeError(f"case_{case['id']}: got {len(outputs)} outputs, expected {OUTPUT_KEYS}")

    saved_out = {}
    for i, k in enumerate(OUTPUT_KEYS):
        t = outputs[i].detach().cpu()
        if not torch.isfinite(t.float()).all():
            raise RuntimeError(f"case_{case['id']}: output '{k}' has NaN/Inf — bad golden data")
        saved_out[k] = t

    saved_in = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    return saved_in, saved_out


def describe_call(call_args):
    parts = []
    names = PARAM_NAMES + ["out_tmpls"]
    for name, val in zip(names, call_args):
        if isinstance(val, torch.Tensor):
            parts.append(f"{name}=Tensor{tuple(val.shape)}:{str(val.dtype).replace('torch.', '')}")
        elif isinstance(val, list):
            desc = ", ".join(f"Tensor{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}"
                             for t in val)
            parts.append(f"{name}=[{desc}]")
        elif val is None:
            parts.append(f"{name}=None")
        else:
            parts.append(f"{name}={val!r}")
    return ", ".join(parts)


def check_signature():
    params = list(inspect.signature(ops.npu_sparse_flash_attention).parameters)
    expected = [p for p in params if p not in ("out", "out_tmpl", "out_tmpls")]
    if expected != PARAM_NAMES:
        print("Signature mismatch between this script and the generated wrapper:")
        print("  wrapper:      %s" % ", ".join(params))
        print("  PARAM_NAMES:  %s" % ", ".join(PARAM_NAMES))
        sys.exit(1)


DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)

CASES = [
    {"id": 0, "q_type": "fp16", "B": 1, "S1": 4, "S2": 8, "N1": 8, "K": 16, "spb": 1,
     "mode": 3, "lse": True},
    {"id": 1, "q_type": "bf16", "B": 2, "S1": 3, "S2": 6, "N1": 16, "K": 32, "spb": 1,
     "mode": 0, "lse": False},
    {"id": 2, "q_type": "fp16", "B": 2, "S1": 4, "S2": 4, "N1": 8, "K": 8, "spb": 1,
     "mode": 3, "lse": True},
    {"id": 3, "q_type": "bf16", "B": 1, "S1": 8, "S2": 32, "N1": 32, "K": 64, "spb": 1,
     "mode": 0, "lse": False},
    {"id": 4, "q_type": "fp16", "B": 3, "S1": 2, "S2": 8, "N1": 16, "K": 16, "spb": 1,
     "mode": 3, "lse": True},
    {"id": 5, "q_type": "bf16", "B": 1, "S1": 5, "S2": 16, "N1": 8, "K": 16, "spb": 2,
     "mode": 3, "lse": False},
    {"id": 6, "q_type": "fp16", "B": 2, "S1": 6, "S2": 12, "N1": 64, "K": 32, "spb": 1,
     "mode": 0, "lse": False},
    {"id": 7, "q_type": "bf16", "B": 2, "S1": 4, "S2": 16, "N1": 32, "K": 32, "spb": 1,
     "mode": 0, "lse": True},
    {"id": 8, "q_type": "fp16", "B": 1, "S1": 6, "S2": 6, "N1": 16, "K": 8, "spb": 1,
     "mode": 3, "lse": False},
    {"id": 9, "q_type": "bf16", "B": 3, "S1": 8, "S2": 32, "N1": 16, "K": 64, "spb": 1,
     "mode": 3, "lse": True},
]


def main():
    check_signature()
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    manifest = {"op_name": OP_NAME, "op_camel": OP_CAMEL, "output_keys": OUTPUT_KEYS, "cases": []}

    for case in CASES:
        name = f"case_{case['id']}"
        try:
            saved_in, saved_out = collect_case(case)
        except TypeError as exc:
            rejected = "incompatible function arguments" in str(exc)
            print(f"{name}: {'signature mismatch' if rejected else 'TypeError after the call'}")
            if rejected:
                print(f"  passed: {describe_call(build_call_args(case)[0])}")
                doc = ops.npu_sparse_flash_attention.__doc__
                print(f"  binding expects: {doc.splitlines()[0] if doc else '(no docstring)'}")
            print(f"  {type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(1)
        except Exception as exc:
            print(f"{name}: COLLECT FAILED — {type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(1)
        torch.save(saved_in, os.path.join(INPUT_DIR, f"{name}.pt"))
        torch.save(saved_out, os.path.join(RESULT_DIR, f"{name}.pt"))
        manifest["cases"].append({
            "id": case["id"],
            "q_type": case["q_type"],
            "shape": [case["B"], case["S1"], case["N1"], D],
            "attrs": {k: v for k, v in case.items() if k not in ("id", "q_type")},
        })
        print(f"{name}: OK  input={os.path.getsize(os.path.join(INPUT_DIR, name + '.pt'))}B  "
              f"result={os.path.getsize(os.path.join(RESULT_DIR, name + '.pt'))}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
