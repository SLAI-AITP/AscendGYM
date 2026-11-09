#!/usr/bin/env python3
"""Collect golden data for interleave_rope by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

Shape constraints from tiling code:
  - x: (B, N, S, 64) — N>=1, S>=1, B>=1
  - cos/sin: (B, 1, cosS, 64) — cosB MUST equal x's B, cosN=1, cosS=1 or S
  - D must be exactly 64

Tiling branches (from DoOpTiling):
  1. BN1D: seq >= 512 AND cosS == seq  (seq-per-core split)
  2. FIXED_BNSD: B==32, N==32, S==1, cosN==1, cosS==1  (batch-per-block=4)
  3. B11D: cosS==1 and (B >= coreNum OR B >= N*S)  (batch split)
  4. B11D: cosS==1 and B < coreNum AND B < N*S  (NS split)
  5. B1SD: cosS != 1 and cosS != seq  (batch split) — NOT reachable due to shape check

All shapes are multiples of 32 for alignment.
"""
import json, os, sys
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "interleave_rope"
OP_CAMEL = "InterleaveRope"
DEVICE = "npu:0"
SEED = 2026

# Positional order expected by ops.npu_interleave_rope(...), excluding output buffers.
PARAM_NAMES = ["x", "cos", "sin"]
OUTPUT_KEYS = ["y"]

# 10 cases spanning dtypes / sizes / cos/sin broadcast patterns
# cos/sin batch MUST equal x batch (enforced in tiling code: cosB_ != batchSize_)
# cos/sin N must be 1 (enforced in tiling code: cosN_ != 1)
# cosS must be 1 or equal to seq (enforced in tiling code)
# D must be 64
CASES = [
    {"id": 0, "dtype": torch.float16,  "batch": 32, "seq": 32,   "cos_s": 1},   # B11D, batch split
    {"id": 1, "dtype": torch.float16,  "batch": 32, "seq": 64,   "cos_s": 1},   # B11D, batch split
    {"id": 2, "dtype": torch.float16,  "batch": 32, "seq": 128,  "cos_s": 1},   # B11D, batch split
    {"id": 3, "dtype": torch.bfloat16, "batch": 64, "seq": 32,   "cos_s": 1},   # B11D, batch split
    {"id": 4, "dtype": torch.bfloat16, "batch": 64, "seq": 64,   "cos_s": 1},   # B11D, batch split
    {"id": 5, "dtype": torch.float16,  "batch": 128,"seq": 32,   "cos_s": 1},   # B11D, batch split
    {"id": 6, "dtype": torch.float16,  "batch": 32, "seq": 512,  "cos_s": 512}, # BN1D (seq >= 512)
    {"id": 7, "dtype": torch.bfloat16, "batch": 128,"seq": 64,   "cos_s": 1},   # B11D, batch split
    {"id": 8, "dtype": torch.float16,  "batch": 64, "seq": 1024, "cos_s": 1024},# BN1D (seq >= 512)
    {"id": 9, "dtype": torch.bfloat16, "batch": 32, "seq": 1,    "cos_s": 1},   # B11D, batch split
]

DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def make_inputs(case):
    """Build every entry of PARAM_NAMES for this case.
    
    Key constraint from tiling code:
      cosB_ must equal batchSize_  (cos/sin batch == x batch)
      cosN_ must be 1
      cosS_ must be 1 or seqLength_
      cosD_ must equal hiddenDim_ (64)
    """
    torch.manual_seed(SEED + case["id"])
    dtype = case["dtype"]
    batch, seq = case["batch"], case["seq"]
    cos_s = case["cos_s"]
    # Shape is always (B, N, S, D) with N=1, D=64
    # cos/sin batch MUST equal x batch (not 1!)
    x_shape = (batch, 1, seq, 64)
    cos_shape = (batch, 1, cos_s, 64)
    sin_shape = (batch, 1, cos_s, 64)
    return {
        "x": torch.randn(x_shape, dtype=dtype, device=DEVICE),
        "cos": torch.randn(cos_shape, dtype=dtype, device=DEVICE),
        "sin": torch.randn(sin_shape, dtype=dtype, device=DEVICE),
    }


def make_output_bufs(case):
    """(shape, dtype) per output, in OUTPUT_KEYS order."""
    batch, seq = case["batch"], case["seq"]
    dtype = case["dtype"]
    return [torch.empty((batch, 1, seq, 64), dtype=dtype, device=DEVICE)]


def build_call_args(case):
    inputs = make_inputs(case)
    positional = [inputs[p] for p in PARAM_NAMES]
    return positional, inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)

    out_bufs = make_output_bufs(case)
    out_arg = out_bufs[0] if len(out_bufs) == 1 else out_bufs

    outputs = ops.npu_interleave_rope(*call_args, out_arg)
    torch.npu.synchronize()

    outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
    if len(outputs) != len(OUTPUT_KEYS):
        raise RuntimeError(f"case_{case['id']}: got {len(outputs)} outputs, expected {OUTPUT_KEYS}")

    saved_out = {k: outputs[i].cpu() for i, k in enumerate(OUTPUT_KEYS)}
    for k, t in saved_out.items():
        if not torch.isfinite(t.float()).all():
            raise RuntimeError(f"case_{case['id']}: output '{k}' has NaN/Inf — bad golden data")

    saved_in = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    return saved_in, saved_out


def describe_call(call_args):
    """Compact argument summary."""
    parts = []
    for name, val in zip(PARAM_NAMES, call_args):
        if isinstance(val, torch.Tensor):
            parts.append(f"{name}=Tensor{tuple(val.shape)}:{str(val.dtype).replace('torch.', '')}")
        else:
            parts.append(f"{name}={val!r}")
    return ", ".join(parts)


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    manifest = {"op_name": OP_NAME, "op_camel": OP_CAMEL, "output_keys": OUTPUT_KEYS, "cases": []}

    for case in CASES:
        name = f"case_{case['id']}"
        try:
            saved_in, saved_out = collect_case(case)
        except TypeError as exc:
            print(f"{name}: signature mismatch")
            print(f"  passed: {describe_call(build_call_args(case)[0])}")
            print(f"  binding expects: {ops.npu_interleave_rope.__doc__.splitlines()[0] if ops.npu_interleave_rope.__doc__ else '?'}")
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
            "shape": [case["batch"], 1, case["seq"], 64],
            "attrs": {k: v for k, v in case.items() if k not in ("id", "dtype", "batch", "seq")},
        })
        print(f"{name}: OK  input={os.path.getsize(in_path)}B  result={os.path.getsize(out_path)}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
