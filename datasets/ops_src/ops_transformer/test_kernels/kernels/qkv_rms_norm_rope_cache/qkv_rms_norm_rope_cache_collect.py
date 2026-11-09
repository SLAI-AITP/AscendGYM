#!/usr/bin/env python3
"""Collect golden data for qkv_rms_norm_rope_cache by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

SHAPE MODEL — from op_host/qkv_rms_norm_rope_cache_def.cpp + docs/aclnnQkvRmsNormRopeCache.md
+ qkv_rms_norm_rope_cache_tiling.cpp. B=batches, S=seq per batch, D=head dim, Nq/Nk/Nv=head counts.

  qkv      (B*S, (Nq+Nk+Nv)*D)   fp16/bf16            Nqkv = Nq+Nk+Nv, Nk=Nv, all D equal
  q_gamma  (D,)                  == qkv dtype         RMSNorm gamma for q
  k_gamma  (D,)                  == qkv dtype         RMSNorm gamma for k
  cos/sin  (B*S, D)              == qkv dtype         D*2 must be divisible by 32
  index    (B*S,)                int64                unique values in [0, blockNum*blockSize)
  q_out    (B*S, Nq*D)           == qkv dtype         in-place output
  k_cache  4D (blockNum, (Nk*D)//16, blockSize, 16)   non-quant (fp16/bf16); k_scale=None
           4D (blockNum, (Nk*D)//32, blockSize, 32)   quant (int8); k_scale required
  v_cache  same as k_cache (same blockNum/blockSize)
  k_scale/v_scale  (Nk, D)/(Nv, D)  fp32              None unless int8 cache
  k_offset/v_offset (Nk, D)/(Nv, D)  fp32             only for asymmetric int8 quant

  attrs: qkv_size=[B,S,Nqkv,D]; head_nums=[Nq,Nk,Nv]; epsilon; cache_mode="PA_NZ";
         is_output_qkv=True (the 3 "before_quant" outputs are only defined when this is True)

CONSTRAINTS (from qkv_rms_norm_rope_cache_tiling.cpp GetShapeAttrsInfoInner + Check*Valid):
  qkvDim (D) must be exactly 128      "Dqkv must be 128."
  Nk == Nv and Nq % Nk == 0
  cos/sin shape (B*S, D) or (B, D); index 1D int64 size == B*S
  q_out (B*S, Nq*D) same dtype as qkv
  k_cache/v_cache 4D: (blockNum, Nk*D//16, blockSize, 16) fp16/bf16 or
      (blockNum, Nk*D//32, blockSize, 32) int8; blockSize%%32==0;
      blockNum >= Ceil(S/blockSize)*B; v_cache blockNum/blockSize == k_cache's
  k_scale/v_scale (Nk,D)/(Nv,D) fp32 present IFF cache int8
  k_offset/v_offset present only for asymmetric int8 quant

# case_0: fp16 B16 S3 D128 Nq16 Nk1 Nv1  nonquant blocksz32 bn16  -> q_out(48,2048) kc(16,8,32,16)
# case_1: bf16 B1  S2 D128 Nq8  Nk2 Nv2  nonquant blocksz32 bn1   -> q_out(2,1024)  kc(1,16,32,16)
# case_2: fp16 int8-quant (sym) B2 S1 D128 Nq4 Nk1 Nv1 blocksz64 bn1 kc(1,4,64,32) kscale(v)
# case_3: fp16 B1 S4 D128 Nq32 Nk4 Nv4 nonquant blocksz32 bn1 kc(1,32,32,16)
# case_4: bf16 B1 S1 D128 Nq1 Nk1 Nv1 nonquant blocksz32 bn1 kc(1,8,32,16) tiny
# case_5: fp16 B1 S64 D128 Nq8 Nk8 Nv8 nonquant blocksz32 bn2 kc(2,64,32,16) multi-block
# case_6: bf16 int8-quant (asym) B3 S2 D128 Nq4 Nk2 Nv2 blocksz32 bn3 kc(3,8,32,32) k/voffset
# case_7: fp16 B1 S8 D128 Nq16 Nk1 Nv1 nonquant blocksz32 bn1 kc(1,8,32,16)
# case_8: bf16 B2 S16 D128 Nq8 Nk8 Nv8 nonquant blocksz64 bn2 kc(2,64,64,16) multi-batch
# case_9: fp16 B1 S16 D128 Nq40 Nk2 Nv2 nonquant blocksz128 bn1 kc(1,16,128,16) big blockSize
"""
import inspect
import json
import os
import sys

import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "qkv_rms_norm_rope_cache"
OP_CAMEL = "QkvRmsNormRopeCache"
DEVICE = "npu:0"
SEED = 2028

# Positional order of the generated wrapper (binding.py), excluding 'out'.
# q_out/k_cache/v_cache are in-place: same buffers appear as inputs and as outputs.
PARAM_NAMES = [
    "qkv", "q_gamma", "k_gamma", "cos", "sin", "index",
    "q_out", "k_cache", "v_cache",
    "k_scale", "v_scale", "k_offset", "v_offset",
    "qkv_size", "head_nums", "epsilon", "cache_mode", "is_output_qkv",
]
OUTPUT_KEYS = [
    "q_out", "k_cache", "v_cache",
    "q_out_before_quant", "k_out_before_quant", "v_out_before_quant",
]

DT = {"fp16": torch.float16, "bf16": torch.bfloat16}

CASES = [
    # case_0: fp16, docs-example config, non-quant
    {"id": 0, "dt": "fp16", "B": 16, "S": 3, "D": 128, "Nq": 16, "Nk": 1, "Nv": 1,
     "blockSize": 32, "quant": False, "eps": 1e-6},
    # case_1: bf16, multi-head k, non-quant, single batch
    {"id": 1, "dt": "bf16", "B": 1, "S": 2, "D": 128, "Nq": 8, "Nk": 2, "Nv": 2,
     "blockSize": 32, "quant": False, "eps": 1e-6},
    # case_2: fp16 input, int8 symmetric quant k/v, single token per batch
    {"id": 2, "dt": "fp16", "B": 2, "S": 1, "D": 128, "Nq": 4, "Nk": 1, "Nv": 1,
     "blockSize": 64, "quant": True, "asym": False, "eps": 1e-6},
    # case_3: fp16, many q heads, non-quant
    {"id": 3, "dt": "fp16", "B": 1, "S": 4, "D": 128, "Nq": 32, "Nk": 4, "Nv": 4,
     "blockSize": 32, "quant": False, "eps": 1e-6},
    # case_4: bf16, single token, tiny single head
    {"id": 4, "dt": "bf16", "B": 1, "S": 1, "D": 128, "Nq": 1, "Nk": 1, "Nv": 1,
     "blockSize": 32, "quant": False, "eps": 1e-6},
    # case_5: fp16, long seq -> multi-block (bn=2), many heads
    {"id": 5, "dt": "fp16", "B": 1, "S": 64, "D": 128, "Nq": 8, "Nk": 8, "Nv": 8,
     "blockSize": 32, "quant": False, "eps": 1e-6},
    # case_6: bf16 input, int8 asymmetric quant (scale+offset), multi-batch, multi-head k
    {"id": 6, "dt": "bf16", "B": 3, "S": 2, "D": 128, "Nq": 4, "Nk": 2, "Nv": 2,
     "blockSize": 32, "quant": True, "asym": True, "eps": 1e-6},
    # case_7: fp16, single-head k, non-quant
    {"id": 7, "dt": "fp16", "B": 1, "S": 8, "D": 128, "Nq": 16, "Nk": 1, "Nv": 1,
     "blockSize": 32, "quant": False, "eps": 1e-5},
    # case_8: bf16, multi-batch, big blockSize
    {"id": 8, "dt": "bf16", "B": 2, "S": 16, "D": 128, "Nq": 8, "Nk": 8, "Nv": 8,
     "blockSize": 64, "quant": False, "eps": 1e-6},
    # case_9: fp16, big blockSize=128, many q heads, non-quant
    {"id": 9, "dt": "fp16", "B": 1, "S": 16, "D": 128, "Nq": 40, "Nk": 2, "Nv": 2,
     "blockSize": 128, "quant": False, "eps": 1e-12},
]


def block_num(case):
    """blockNum >= ceil(S/blockSize)*B — the minimum number of cache blocks."""
    import math
    return math.ceil(case["S"] / case["blockSize"]) * case["B"]


def make_inputs(case):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED + case["id"])
    dt = DT[case["dt"]]
    B, S, D, Nq, Nk, Nv = case["B"], case["S"], case["D"], case["Nq"], case["Nk"], case["Nv"]
    Nqkv = Nq + Nk + Nv
    BS = B * S

    qkv = (torch.randn((BS, Nqkv * D), generator=g, device=DEVICE) * 0.5).to(dt)
    q_gamma = (torch.rand((D,), generator=g, device=DEVICE) * 0.5 + 0.5).to(dt)  # positive
    k_gamma = (torch.rand((D,), generator=g, device=DEVICE) * 0.5 + 0.5).to(dt)

    # RoPE cos/sin tables per token row (half-and-half: first half cos-ish, second half sin-ish)
    pos = torch.arange(BS, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    freq = torch.arange(D, device=DEVICE, dtype=torch.float32).unsqueeze(0) / max(D, 1)
    ang = pos * freq
    cos = torch.cos(ang).to(dt)
    sin = torch.sin(ang).to(dt)

    # unique cache-slot indices in [0, blockNum*blockSize). Use .clone() (unconditional copy): a
    # view into the shuffled full-length arange would keep storage == n_slots, and the binding's
    # ConvertType re-views the tensor to its storage shape, which would then mismatch the logical
    # B*S shape.
    blk_num = block_num(case)
    n_slots = blk_num * case["blockSize"]
    perm = torch.randperm(n_slots, generator=g, device=DEVICE)
    index = perm[:BS].clone()

    # quant params
    k_scale = v_scale = k_offset = v_offset = None
    if case.get("quant"):
        k_scale = (torch.rand((Nk, D), generator=g, device=DEVICE) * 0.1 + 0.05).to(torch.float32)
        v_scale = (torch.rand((Nv, D), generator=g, device=DEVICE) * 0.1 + 0.05).to(torch.float32)
        if case.get("asym"):
            k_offset = (torch.rand((Nk, D), generator=g, device=DEVICE) * 0.2).to(torch.float32)
            v_offset = (torch.rand((Nv, D), generator=g, device=DEVICE) * 0.2).to(torch.float32)

    return {
        "qkv": qkv,
        "q_gamma": q_gamma,
        "k_gamma": k_gamma,
        "cos": cos,
        "sin": sin,
        "index": index,
        "q_out": None,       # filled by make_output_bufs (in-place)
        "k_cache": None,
        "v_cache": None,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "k_offset": k_offset,
        "v_offset": v_offset,
        "qkv_size": [B, S, Nqkv, D],
        "head_nums": [Nq, Nk, Nv],
        "epsilon": case["eps"],
        "cache_mode": "PA_NZ",
        "is_output_qkv": True,
    }


def make_output_bufs(case):
    """6 buffers in OUTPUT_KEYS order. q_out/k_cache/v_cache are the in-place buffers that are
    also passed as inputs in PARAM_NAMES positions 6/7/8."""
    dt = DT[case["dt"]]
    B, S, D, Nq, Nk, Nv = case["B"], case["S"], case["D"], case["Nq"], case["Nk"], case["Nv"]
    BS = B * S
    blk_num = block_num(case)
    bs = case["blockSize"]

    cache_dtype = torch.int8 if case.get("quant") else dt
    d_per_block = 32 if case.get("quant") else 16
    # zero-init every buffer: the kernel only writes the cache slots referenced by `index`, so
    # any unwritten slots must stay 0 (torch.empty would leave non-deterministic garbage and
    # the regression test would fail on them between runs).
    k_cache = torch.zeros((blk_num, (Nk * D) // d_per_block, bs, d_per_block), dtype=cache_dtype, device=DEVICE)
    v_cache = torch.zeros((blk_num, (Nv * D) // d_per_block, bs, d_per_block), dtype=cache_dtype, device=DEVICE)
    q_out = torch.zeros((BS, Nq * D), dtype=dt, device=DEVICE)
    proto = torch.zeros((BS, Nq * D), dtype=dt, device=DEVICE)  # before_quant outs share q shape
    k_proto = torch.zeros((BS, Nk * D), dtype=dt, device=DEVICE)
    v_proto = torch.zeros((BS, Nv * D), dtype=dt, device=DEVICE)
    return [q_out, k_cache, v_cache, proto, k_proto, v_proto]


def build_call_args(case):
    inputs = make_inputs(case)
    out_bufs = make_output_bufs(case)
    # in-place inputs == their output buffers
    inputs["q_out"] = out_bufs[0]
    inputs["k_cache"] = out_bufs[1]
    inputs["v_cache"] = out_bufs[2]
    positional = [inputs[p] for p in PARAM_NAMES]
    return positional, inputs, out_bufs


def collect_case(case):
    call_args, inputs, out_bufs = build_call_args(case)

    outputs = ops.npu_qkv_rms_norm_rope_cache(*call_args, out_bufs)
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
    for k, t in saved_out.items():
        if not torch.isfinite(t.float()).all():
            raise RuntimeError(f"case_{case['id']}: output '{k}' has NaN/Inf — bad golden data")

    saved_in = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    return saved_in, saved_out


def check_signature():
    params = list(inspect.signature(ops.npu_qkv_rms_norm_rope_cache).parameters)
    expected = [p for p in params if p != "out"]
    if expected != PARAM_NAMES:
        print("Signature mismatch between this script and the generated wrapper:")
        print("  wrapper:      %s" % ", ".join(expected))
        print("  PARAM_NAMES:  %s" % ", ".join(PARAM_NAMES))
        sys.exit(1)
    if "out" not in params:
        print("[WARN] the wrapper has no 'out' parameter — every output is in-place")


def main():
    for case in CASES:
        assert case["D"] % 16 == 0, f"case {case['id']}: D not 16-aligned"
        Nk, Nv, D = case["Nk"], case["Nv"], case["D"]
        res = 32 if case.get("quant") else 16
        assert (Nk * D) % res == 0 and (Nv * D) % res == 0
        assert case["blockSize"] % 32 == 0
    check_signature()
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    manifest = {"op_name": OP_NAME, "op_camel": OP_CAMEL, "output_keys": OUTPUT_KEYS, "cases": []}

    for case in CASES:
        name = f"case_{case['id']}"
        try:
            saved_in, saved_out = collect_case(case)
        except TypeError as exc:
            print(f"{name}: TypeError — {str(exc)[:400]}")
            sys.exit(1)
        except Exception as exc:
            print(f"{name}: COLLECT FAILED — {type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(1)
        in_path = os.path.join(INPUT_DIR, f"{name}.pt")
        out_path = os.path.join(RESULT_DIR, f"{name}.pt")
        torch.save(saved_in, in_path)
        torch.save(saved_out, out_path)
        manifest["cases"].append({
            "id": case["id"],
            "dtype": case["dt"],
            "shape": [case["B"] * case["S"], (case["Nq"] + case["Nk"] + case["Nv"]) * case["D"]],
            "attrs": {k: v for k, v in case.items() if k not in ("id", "dt")},
        })
        print(f"{name}: OK  input={os.path.getsize(in_path)}B  result={os.path.getsize(out_path)}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


if __name__ == "__main__":
    main()
