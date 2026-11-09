#!/usr/bin/env python3
"""Collect golden data for mla_prolog by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

SHAPE MODEL — from op_host/mla_prolog_def.cpp + docs/aclnnMlaProlog.md + README.md
(parameter/constraint tables). MLA "prolog": compute query (q^N), rope-rotated query
(q^R) and write k^C / k^R into two paged caches, given token_x and weight matrices.

  Non-quantized BF16 path (all computation tensors BF16, cache_index INT64):
    token_x        (B,S,He) | (T,He)        BF16    input activations
    weight_dq      (He,Hcq)                 BF16    W^DQ  (FRACTAL_NZ, passed as logical ND)
    weight_uq_qr   (Hcq,N*(D+Dr))           BF16    W^UQ + W^QR (fused)
    weight_uk      (N,D,Hckv)               BF16    W^UK
    weight_dkv_kr  (He,Hckv+Dr)             BF16    W^DKV + W^KR (fused)
    rmsnorm_gamma_cq   (Hcq,)               BF16
    rmsnorm_gamma_ckv  (Hckv,)              BF16
    rope_sin / rope_cos  (B,S,Dr) | (T,Dr)  BF16
    cache_index     (B,S) | (T)             INT64   slot indices in [0, BlockNum*BlockSize)
    kv_cache  (BlockNum,BlockSize,1,Hckv)   BF16    in-place: written through (k^C)
    kr_cache  (BlockNum,BlockSize,1,Dr)     BF16    in-place: written through (k^R)
    7 optional dequant/quant/smooth scales -> None (non-quantized)

  attrs: rmsnorm_epsilon_cq (double, default 1e-5), rmsnorm_epsilon_ckv (double, 1e-5),
    cache_mode (str; doc-valid values "PA_BSND"/"PA_NZ", recommended "PA_BSND" — used here).

  outputs (OpDef order):
    query       (B,S,N,Hckv) | (T,N,Hckv)   BF16   buffered template 0 (q^N)
    query_rope  (B,S,N,Dr)   | (T,N,Dr)     BF16   buffered template 1 (q^R)
    kv_cache / kr_cache  in-place (inputs written through)

CONSTRAINTS (from docs): He in {1024..8192 step 1024}, Hcq=1536 fixed, N in {1,2,4,8,16,
  32,64,128}, Hckv=512 fixed, D=128 fixed, Dr=64 fixed, Nkv=1 fixed, BlockSize in
  [16,1024] multiple of 16, BlockNum=ceil(T/BlockSize), cache_index in [0,BlockNum*BlockSize).
  DynamicFormatFlag(true) -> ND weight tensors are auto-converted to FRACTAL_NZ.

CALL (binding.py wrapper, op_def order; no op_host/op_api so the aclnn call uses op_host
order, which matches the aclnnMlaProlog signature): 19 inputs then 3 attrs, then the
trailing out_tmpls buffer list (2 buffers, OpDef output order: query, query_rope).
   token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
   rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache,
   dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr,
   quant_scale_ckv, quant_scale_ckr, smooth_scales_cq,
   rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode
Returns tuple (query, query_rope, kv_cache, kr_cache). kv_cache/kr_cache are the same
in-place input tensors, written through by the kernel.

# case_0: BF16 BS B1 S1      N4   BlockSize16  (tiny, single token)
# case_1: BF16 T-fused T2    N2   BlockSize16  (2D token_x + 1D cache_index rank variant)
# case_2: BF16 BS B1 S4      N8   BlockSize32
# case_3: BF16 BS B2 S1      N4   BlockSize64
# case_4: BF16 BS B1 S8      N16  BlockSize32
# case_5: BF16 BS B2 S4      N8   BlockSize32
# case_6: BF16 BS B2 S2      N2   BlockSize16  (min heads)
# case_7: BF16 BS B4 S4 T16  N4   BlockSize16  (full block, 16 slots)
# case_8: BF16 BS B2 S16 T32 N16  BlockSize16  (multi-block: BlockNum=2)
# case_9: BF16 BS B3 S3 T9   N8   BlockSize64
"""
import inspect
import json
import os
import sys

import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

# Allow creating FRACTAL_NZ (internal format) tensors via npu_format_cast — otherwise the
# weight tensors declared FORMAT_FRACTAL_NZ stay ND and aclnnMlaProlog rejects them.
torch_npu.npu.config.allow_internal_format = True

OP_NAME = "mla_prolog"
OP_CAMEL = "MlaProlog"
DEVICE = "npu:0"
SEED = 202601
HE = 1024      # hidden size (smallest allowed)
HCQ = 1536     # q low-rank dim (fixed)
HCKV = 512     # kv low-rank dim (fixed)
D = 128        # qk dim without rope (fixed)
DR = 64        # rope dim (fixed)
NKV = 1        # kv heads (fixed)

# Positional order of the generated wrapper (binding.py), op_def order: inputs then attrs,
# excluding the trailing out_tmpls buffer list.
PARAM_NAMES = [
    "token_x", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
    "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "rope_sin", "rope_cos", "cache_index",
    "kv_cache", "kr_cache",
    "dequant_scale_x", "dequant_scale_w_dq", "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr", "quant_scale_ckv", "quant_scale_ckr", "smooth_scales_cq",
    "rmsnorm_epsilon_cq", "rmsnorm_epsilon_ckv", "cache_mode",
]
OUTPUT_KEYS = ["query", "query_rope", "kv_cache", "kr_cache"]


def _cache_index(B, S, T, BlockNum, BlockSize):
    """Distinct slot indices in [0, BlockNum*BlockSize) for the (B,S) | (T) tokens."""
    idx = torch.arange(T, dtype=torch.int64, device=DEVICE)
    if B and S:
        return idx.view(B, S)
    return idx  # T layout: 1-D


def make_inputs(case):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED + case["id"])
    dt = torch.bfloat16
    B, S, N = case["B"], case["S"], case["N"]
    fused = case.get("fused", False)
    T = B * S if not fused else case["T"]
    BlockSize = case["bsz"]
    BlockNum = max(1, -(-T // BlockSize))  # ceil(T/BlockSize), >=1 so cache non-empty

    # token_x / rope: (B,S,...) or (T,...)
    if fused:
        token_x = (torch.randn((T, HE), generator=g, device=DEVICE) * 0.1).to(dt)
        rope_sin = (torch.randn((T, DR), generator=g, device=DEVICE) * 0.1).to(dt)
        rope_cos = (torch.randn((T, DR), generator=g, device=DEVICE) * 0.1).to(dt)
        cache_index = _cache_index(0, 0, T, BlockNum, BlockSize)
    else:
        token_x = (torch.randn((B, S, HE), generator=g, device=DEVICE) * 0.1).to(dt)
        rope_sin = (torch.randn((B, S, DR), generator=g, device=DEVICE) * 0.1).to(dt)
        rope_cos = (torch.randn((B, S, DR), generator=g, device=DEVICE) * 0.1).to(dt)
        cache_index = _cache_index(B, S, T, BlockNum, BlockSize)

    # weights: dq / uq_qr / dkv_kr are declared FORMAT_FRACTAL_NZ in the OpDef, so they
    # must be cast to NZ storage (format 29 = FRACTAL_NZ) while keeping the logical ND
    # shape (matches official mla_prolog_v2 pytest). weight_uk stays ND.
    weight_dq = (torch.randn((HE, HCQ), generator=g, device=DEVICE) * 0.05).to(dt)
    weight_dq = torch_npu.npu_format_cast(weight_dq.contiguous(), 29)
    weight_uq_qr = (torch.randn((HCQ, N * (D + DR)), generator=g, device=DEVICE) * 0.05).to(dt)
    weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr.contiguous(), 29)
    weight_uk = (torch.randn((N, D, HCKV), generator=g, device=DEVICE) * 0.05).to(dt)
    weight_dkv_kr = (torch.randn((HE, HCKV + DR), generator=g, device=DEVICE) * 0.05).to(dt)
    weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr.contiguous(), 29)

    # rmsnorm gamma = ones (identity scaling)
    rmsnorm_gamma_cq = torch.ones((HCQ,), dtype=dt, device=DEVICE)
    rmsnorm_gamma_ckv = torch.ones((HCKV,), dtype=dt, device=DEVICE)

    # paged caches (in-place), zero-init for determinism of unwritten slots
    kv_cache = torch.zeros((BlockNum, BlockSize, NKV, HCKV), dtype=dt, device=DEVICE)
    kr_cache = torch.zeros((BlockNum, BlockSize, NKV, DR), dtype=dt, device=DEVICE)

    return {
        "token_x": token_x,
        "weight_dq": weight_dq,
        "weight_uq_qr": weight_uq_qr,
        "weight_uk": weight_uk,
        "weight_dkv_kr": weight_dkv_kr,
        "rmsnorm_gamma_cq": rmsnorm_gamma_cq,
        "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv,
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
        "cache_index": cache_index,
        "kv_cache": kv_cache,
        "kr_cache": kr_cache,
        "dequant_scale_x": None,          # non-quantized: must be null
        "dequant_scale_w_dq": None,
        "dequant_scale_w_uq_qr": None,
        "dequant_scale_w_dkv_kr": None,
        "quant_scale_ckv": None,
        "quant_scale_ckr": None,
        "smooth_scales_cq": None,
        "rmsnorm_epsilon_cq": 1e-5,
        "rmsnorm_epsilon_ckv": 1e-5,
        "cache_mode": "PA_BSND",
    }


def make_output_bufs(case):
    """Two buffered template outputs (OpDef output order): query, query_rope."""
    dt = torch.bfloat16
    B, S, N = case["B"], case["S"], case["N"]
    if case.get("fused", False):
        T = case["T"]
        query = torch.zeros((T, N, HCKV), dtype=dt, device=DEVICE)
        query_rope = torch.zeros((T, N, DR), dtype=dt, device=DEVICE)
    else:
        query = torch.zeros((B, S, N, HCKV), dtype=dt, device=DEVICE)
        query_rope = torch.zeros((B, S, N, DR), dtype=dt, device=DEVICE)
    return [query, query_rope]


def build_call_args(case):
    inputs = make_inputs(case)
    out_bufs = make_output_bufs(case)
    call_args = [inputs[p] for p in PARAM_NAMES] + [out_bufs]
    return call_args, inputs, out_bufs


def collect_case(case):
    call_args, inputs, out_bufs = build_call_args(case)

    # Snapshot inputs BEFORE the call — kv_cache / kr_cache are in-place outputs and the
    # kernel writes through them, so the saved golden_input must be the pre-call tensors.
    saved_in = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    outputs = ops.npu_mla_prolog(*call_args)
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
    params = list(inspect.signature(ops.npu_mla_prolog).parameters)
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
    {"id": 0, "B": 1, "S": 1, "N": 4, "bsz": 16, "fused": False},
    {"id": 1, "B": 1, "S": 1, "T": 2, "N": 2, "bsz": 16, "fused": True},
    {"id": 2, "B": 1, "S": 4, "N": 8, "bsz": 32, "fused": False},
    {"id": 3, "B": 2, "S": 1, "N": 4, "bsz": 64, "fused": False},
    {"id": 4, "B": 1, "S": 8, "N": 16, "bsz": 32, "fused": False},
    {"id": 5, "B": 2, "S": 4, "N": 8, "bsz": 32, "fused": False},
    {"id": 6, "B": 2, "S": 2, "N": 2, "bsz": 16, "fused": False},
    {"id": 7, "B": 4, "S": 4, "N": 4, "bsz": 16, "fused": False},
    {"id": 8, "B": 2, "S": 16, "N": 16, "bsz": 16, "fused": False},
    {"id": 9, "B": 3, "S": 3, "N": 8, "bsz": 64, "fused": False},
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
                doc = ops.npu_mla_prolog.__doc__
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
            "shape": [case["B"], case["S"], case["N"], HCKV],
            "attrs": {k: v for k, v in case.items() if k not in ("id", "B", "S")},
        })
        print(f"{name}: OK  input={os.path.getsize(os.path.join(INPUT_DIR, name + '.pt'))}B  "
              f"result={os.path.getsize(os.path.join(RESULT_DIR, name + '.pt'))}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
