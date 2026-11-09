#!/usr/bin/env python3
"""Collect golden data for lightning_indexer by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

SHAPE MODEL — from op_host/lightning_indexer_def.cpp, op_host/lightning_indexer_infershape.cpp,
op_host/lightning_indexer_tiling.cpp and op_host/op_api/aclnn_lightning_indexer.h.

  Indices = Top-k{ [1]@( (W @ [1]^T) * ReLU(Q @ K^T) ) };  W is the per-token (per-head) weight.
  head dim D is FIXED to 128; key head num N2 is FIXED to 1; query head num N1 <= 64.

  query  FP16/BF16   BSND:(B,S1,N1,128)  or TND:(T,N1,128)
  key    FP16/BF16   BSND:(B,S2,1,128)   or TND:(T2,1,128)   or PA_BSND:(block_num,block_size,1,128)
  weights FP16/BF16/FP32  BSND:(B,S1,N1) or TND:(T,N1)      (must equal query/key dtype unless fp32)
  actualSeqLengthsQuery  INT32 (B,)  — required for TND layout (prefix sums cumulating to T);
                                      optional (None) for BSND layout
  actualSeqLengthsKey    INT32 (B,)  — required for TND/PA_BSND key; optional for BSND key
  blockTable             INT32 (B, maxBlockNumPerSeq) — required for PA_BSND key
  sparseIndicesOut  INT32   BSND:(B,S1,N2,sparseCount)  TND:(T,N2,sparseCount)
  sparseValuesOut   query-dtype — same shape as indices when return_values=True, else (0,)

LAYOUT RULES (validated by aclnn): when layout_key is non-PA, layout_query == layout_key.
sparse_count is in [1,2048] (or the 3-8k sparsities); it must equal the indices output last dim.
sparse_mode in {0=defaultMask, 3=rightDownCausal}; pre_tokens/next_tokens keep the default 2^63-1.
return_values True only when layout_key != PA_BSND. When return_values=True and sparse_count exceeds
the usable (masked) window, the kernel fills the tail positions with a garbage/padding sentinel —
sparse_values may therefore legitimately contain NaN in padded slots (by design; the regression
comparator only checks the leading finite region robustly). Indices padding uses a finite sentinel.

CASE MATRIX (D=128, N2=1 throughout; sm=sparse_mode; rv=return_values):

# case_0:  BSND/BSND fp16 N1=1  B1 S1=4  S2=64  sc=64   sm=0 rv=True   full window, values, defaultMask
# case_1:  BSND/BSND bf16 N1=8  B2 S1=16 S2=128 sc=2048 sm=3 rv=False GQA, causal, default sparseCount, indices-only
# case_2:  BSND/BSND fp16 N1=32 B1 S1=8  S2=128 sc=128  sm=3 rv=False large GQA (g=32), causal, indices-only
# case_3:  BSND/BSND fp16 N1=4  B2 S1=16 S2=64  sc=64   sm=0 rv=True   weights fp32 (query/key fp16), full, values
# case_4:  BSND/PA_BSND bf16 N1=8 B2 S1=32 key(16 blk x16) sc=128 sm=3 rv=False PageAttention, indices-only
# case_5:  TND/TND fp16 N1=2  B1 T=64  sc=64  sm=0 rv=True   TND single-batch full window, values
# case_6:  TND/TND bf16 N1=4  B1 T=64  sc=128 sm=3 rv=False TND single-batch bf16, causal, indices-only
# case_7:  TND/TND fp16 N1=8  B3 T=48  sc=16  sm=0 rv=False  TND multi-batch weights fp32, indices-only
#   (equal batches 16 each, sc=16 -> every window fully filled, NO padding: multi-batch TND
#    pads sparse_indices with a run-to-run-nondeterministic sentinel, which a stable suite
#    must not freeze; equal-batch + sc<=batch keeps it fully deterministic)
# case_8:  TND/PA_BSND fp16 N1=2 B1 T=16 key(2 blk x16) sc=64 sm=3 rv=False TND+PageAttention, indices-only
#   (single batch: sparse_indices padding is deterministic for B=1, unlike the multi-batch TND case_7)
# case_9:  BSND/BSND fp16 N1=64 B1 S1=1  S2=16  sc=16  sm=0 rv=True   max heads N1=64, single token, full, values
"""
import json
import os
import sys

import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "lightning_indexer"
OP_CAMEL = "LightningIndexer"
DEVICE = "npu:0"
SEED = 2027

PRE_TOKENS = 2 ** 63 - 1   # only the default is supported
NEXT_TOKENS = 2 ** 63 - 1

# Positional order of the generated wrapper (binding.py), excluding 'out'.
PARAM_NAMES = [
    "query", "key", "weights",
    "actual_seq_lengths_query", "actual_seq_lengths_key", "block_table",
    "layout_query", "layout_key", "sparse_count", "sparse_mode",
    "pre_tokens", "next_tokens", "return_values",
]
OUTPUT_KEYS = ["sparse_indices", "sparse_values"]

# Each case: id, ref-id for input generation, dtype (query/key), qlayout, klayout,
# B, S1 or T, N1, S2 (BSND key) or block-count/block-size (PA) or T2 (TND key),
# sparse_count, sparse_mode, return_values, weights_dtype (defaults to dtype).
CASES = [
    {"id": 0, "dtype": torch.float16,  "q_layout": "BSND", "k_layout": "BSND",
     "B": 1, "S1": 4, "N1": 1, "S2": 64, "sc": 64,  "sm": 0, "rv": True},
    {"id": 1, "dtype": torch.bfloat16, "q_layout": "BSND", "k_layout": "BSND",
     "B": 2, "S1": 16, "N1": 8, "S2": 128, "sc": 2048, "sm": 3, "rv": False},
    {"id": 2, "dtype": torch.float16,  "q_layout": "BSND", "k_layout": "BSND",
     "B": 1, "S1": 8, "N1": 32, "S2": 128, "sc": 128, "sm": 3, "rv": False},
    {"id": 3, "dtype": torch.float16,  "q_layout": "BSND", "k_layout": "BSND",
     "B": 2, "S1": 16, "N1": 4, "S2": 64, "sc": 64,  "sm": 0, "rv": True,
     "weights_dtype": torch.float32},
    {"id": 4, "dtype": torch.bfloat16, "q_layout": "BSND", "k_layout": "PA_BSND",
     "B": 2, "S1": 32, "N1": 8, "blocks": 16, "blk": 16, "max_blocks": 8,
     "sc": 128, "sm": 3, "rv": False},
    {"id": 5, "dtype": torch.float16,  "q_layout": "TND", "k_layout": "TND",
     "B": 1, "T": 64, "N1": 2, "sc": 64, "sm": 0, "rv": True},
    {"id": 6, "dtype": torch.bfloat16, "q_layout": "TND", "k_layout": "TND",
     "B": 1, "T": 64, "N1": 4, "sc": 128, "sm": 3, "rv": False},
    {"id": 7, "dtype": torch.float16,  "q_layout": "TND", "k_layout": "TND",
     "B": 3, "T": 48, "N1": 8, "sc": 16, "sm": 0, "rv": False,
     "weights_dtype": torch.float32, "seq": [16, 32, 48]},
    {"id": 8, "dtype": torch.float16,  "q_layout": "TND", "k_layout": "PA_BSND",
     "B": 1, "T": 16, "N1": 2, "blocks": 2, "blk": 16, "max_blocks": 2,
     "seq": [16], "seq_key": [32], "block_table_override": [[0, 1]],
     "sc": 64, "sm": 3, "rv": False},
    {"id": 9, "dtype": torch.float16,  "q_layout": "BSND", "k_layout": "BSND",
     "B": 1, "S1": 1, "N1": 64, "S2": 16, "sc": 16, "sm": 0, "rv": True},
]

DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def _rand(shape, dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=torch.float32, device=DEVICE).to(dtype)
    return torch.randn(shape, dtype=dtype, device=DEVICE)


def cumulative_lens(B, T):
    """B positive ints, strictly increasing, last == T."""
    if B == 1:
        return torch.tensor([T], dtype=torch.int32)
    cuts = sorted(torch.randint(1, T, (B - 1,)).tolist())
    while len(set(cuts)) != B - 1:
        cuts = sorted(torch.randint(1, T, (B - 1,)).tolist())
    parts = [cuts[0]] + [b - a for a, b in zip(cuts, cuts[1:])] + [T - cuts[-1]]
    return torch.tensor(list(parts), dtype=torch.int32)


def make_inputs(case):
    torch.manual_seed(SEED + case["id"])
    dtype = case["dtype"]
    wdtype = case.get("weights_dtype", dtype)
    ql, kl = case["q_layout"], case["k_layout"]

    inputs = {}

    if ql == "BSND":
        B, S1, N1 = case["B"], case["S1"], case["N1"]
        query = _rand((B, S1, N1, 128), dtype)
        weights = _rand((B, S1, N1), wdtype)
        inputs["actual_seq_lengths_query"] = None
    else:  # TND
        B, T, N1 = case["B"], case["T"], case["N1"]
        query = _rand((T, N1, 128), dtype)
        weights = _rand((T, N1), wdtype)
        inputs["actual_seq_lengths_query"] = (
            torch.tensor(case["seq"], dtype=torch.int32) if case.get("seq")
            else cumulative_lens(B, T)).to(DEVICE)

    inputs["query"] = query
    inputs["weights"] = weights

    if kl == "BSND":
        key = _rand((case["B"], case["S2"], 1, 128), dtype)
        inputs["key"] = key
        inputs["actual_seq_lengths_key"] = None
        inputs["block_table"] = None
    elif kl == "TND":
        key = _rand((case["T"], 1, 128), dtype)
        inputs["key"] = key
        inputs["actual_seq_lengths_key"] = (
            torch.tensor(case["seq"], dtype=torch.int32) if case.get("seq")
            else cumulative_lens(case["B"], case["T"])).to(DEVICE)
        inputs["block_table"] = None
    else:  # PA_BSND
        blk = case["blk"]
        key = _rand((case["blocks"], blk, 1, 128), dtype)
        inputs["key"] = key
        # per-batch token counts (PA is NOT cumulative). Each batch uses nblk_i full blocks, so
        # count_i = nblk_i * blk <= max_blocks*blk (the block_table columns it owns). A fixed
        # seq_key override forces every batch to the same full-window count (no padding).
        max_blocks = case["max_blocks"]
        if case.get("seq_key"):
            counts = list(case["seq_key"])
        else:
            nblk = torch.randint(1, max_blocks + 1, (case["B"],)).tolist()
            counts = [n * blk for n in nblk]
        inputs["actual_seq_lengths_key"] = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
        if case.get("block_table_override"):
            bt = torch.tensor(case["block_table_override"], dtype=torch.int32, device=DEVICE)
        else:
            bt = torch.randint(0, case["blocks"], (case["B"], max_blocks), dtype=torch.int32, device=DEVICE)
        inputs["block_table"] = bt

    inputs["layout_query"] = ql
    inputs["layout_key"] = kl
    inputs["sparse_count"] = int(case["sc"])
    inputs["sparse_mode"] = int(case["sm"])
    inputs["pre_tokens"] = PRE_TOKENS
    inputs["next_tokens"] = NEXT_TOKENS
    inputs["return_values"] = bool(case["rv"])
    return inputs


def out_shape(case, sparse_count):
    ql = case["q_layout"]
    if ql == "BSND":
        return (case["B"], case["S1"], 1, sparse_count)
    return (case["T"], 1, sparse_count)


def make_output_bufs(case, inputs):
    oshape = out_shape(case, inputs["sparse_count"])
    return [
        torch.empty(oshape, dtype=torch.int32, device=DEVICE),
        torch.empty(oshape if inputs["return_values"] else (0,),
                    dtype=case["dtype"], device=DEVICE),
    ]


def build_call_args(case):
    inputs = make_inputs(case)
    return [inputs[p] for p in PARAM_NAMES], inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)
    out_bufs = make_output_bufs(case, inputs)
    out_arg = out_bufs[0] if len(out_bufs) == 1 else out_bufs

    outputs = ops.npu_lightning_indexer(*call_args, out_arg)
    torch.npu.synchronize()

    if outputs is None:
        outputs = out_bufs
    elif isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    else:
        outputs = list(outputs)
    if len(outputs) != len(OUTPUT_KEYS):
        raise RuntimeError(f"case_{case['id']}: got {len(outputs)} outputs, expected {OUTPUT_KEYS}")

    saved_out = {k: outputs[i].cpu() for i, k in enumerate(OUTPUT_KEYS)}
    # sparse_indices must always be finite. sparse_values carries padding sentinels (NaN) whenever
    # sparse_count > the masked window — which happens in the causal (sm=3) cases and in ANY
    # multi-batch run (each batch's masked window < sc). It is only finiteness-checked for the
    # single-batch / uniform BSND full-window sm=0 cases (rv=True) where every slot is real data.
    if not torch.isfinite(saved_out["sparse_indices"].float()).all():
        raise RuntimeError(f"case_{case['id']}: sparse_indices has NaN/Inf — bad golden data")
    full_window = case["rv"] and (
        case["q_layout"] == "BSND" or case["B"] == 1
    )
    if full_window and case["sm"] == 0 and \
            not torch.isfinite(saved_out["sparse_values"].float()).all():
        raise RuntimeError(f"case_{case['id']}: sparse_values non-finite in a full-window case — "
                           f"check inputs")

    saved_in = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    return saved_in, saved_out


def describe_call(call_args):
    parts = []
    for name, val in zip(PARAM_NAMES, call_args):
        if isinstance(val, torch.Tensor):
            parts.append(f"{name}=Tensor{tuple(val.shape)}:{str(val.dtype).replace('torch.', '')}")
        else:
            parts.append(f"{name}={val!r}")
    return ", ".join(parts)


def check_signature():
    import inspect
    try:
        params = list(inspect.signature(ops.npu_lightning_indexer).parameters)
    except (TypeError, ValueError):
        print("[WARN] cannot introspect the wrapper signature — skipping the arity check")
        return
    expected = [p for p in params if p != "out"]
    if expected != PARAM_NAMES:
        print("Signature mismatch between this script and the generated wrapper:")
        print("  wrapper:      %s" % ", ".join(expected))
        print("  PARAM_NAMES:  %s" % ", ".join(PARAM_NAMES))
        sys.exit(1)


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
                doc = ops.npu_lightning_indexer.__doc__
                print(f"  binding expects: {doc.splitlines()[0] if doc else '(no docstring)'}")
            print(f"  {type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(1)
        except Exception as exc:
            print(f"{name}: COLLECT FAILED — {type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(1)
        in_path, out_path = os.path.join(INPUT_DIR, f"{name}.pt"), os.path.join(RESULT_DIR, f"{name}.pt")
        torch.save(saved_in, in_path)
        torch.save(saved_out, out_path)
        manifest["cases"].append({
            "id": case["id"], "dtype": str(case["dtype"]).replace("torch.", ""),
            "q_layout": case["q_layout"], "k_layout": case["k_layout"],
            "shape": [case["B"], case.get("S1", case.get("T")), case["N1"]],
            "attrs": {k: (int(v) if isinstance(v, (int,)) and not isinstance(v, bool) else
                          bool(v) if isinstance(v, bool) else str(v).replace("torch.", ""))
                      for k, v in case.items() if k not in ("id", "dtype", "q_layout", "k_layout")},
        })
        print(f"{name}: OK  input={os.path.getsize(in_path)}B  result={os.path.getsize(out_path)}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
