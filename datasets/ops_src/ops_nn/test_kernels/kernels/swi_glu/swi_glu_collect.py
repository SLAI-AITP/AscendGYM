#!/usr/bin/env python3
"""Collect golden data for swi_glu by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

# case_0: in=[x:float16[2, 32], dim=-1] -> out=[y:float16[2, 16]]
# case_1: in=[x:float16[1, 512], dim=-1] -> out=[y:float16[1, 256]]
# case_2: in=[x:float16[4, 64, 32], dim=-1] -> out=[y:float16[4, 64, 16]]
# case_3: in=[x:float16[8, 128, 64], dim=1] -> out=[y:float16[8, 64, 32]]
# case_4: in=[x:float16[2, 256, 128], dim=-1] -> out=[y:float16[2, 256, 64]]
# case_5: in=[x:float32[2, 16, 32, 64], dim=-1] -> out=[y:float32[2, 16, 32, 32]]
# case_6: in=[x:float32[64, 256], dim=-1] -> out=[y:float32[64, 128]]
# case_7: in=[x:float32[1, 2, 4, 128, 64], dim=2] -> out=[y:float32[1, 2, 64, 64]]
# case_8: in=[x:bfloat16[32, 1024], dim=-1] -> out=[y:bfloat16[32, 512]]
# case_9: in=[x:bfloat16[2, 64, 128], dim=-1] -> out=[y:bfloat16[2, 64, 64]]
"""
import json, os, sys
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "swi_glu"
OP_CAMEL = "SwiGlu"
DEVICE = "npu:0"
SEED = 2026

# Positional order expected by ops.npu_swi_glu_custom(...), excluding output buffers.
PARAM_NAMES = ["x", "dim"]
OUTPUT_KEYS = ["y"]

# 10 cases spanning dtypes / ranks / sizes
CASES = [
    {"id": 0, "dtype": torch.float16,  "shape": (2, 32),         "dim": -1},
    {"id": 1, "dtype": torch.float16,  "shape": (1, 512),        "dim": -1},
    {"id": 2, "dtype": torch.float16,  "shape": (4, 64, 32),     "dim": -1},
    {"id": 3, "dtype": torch.float16,  "shape": (8, 128, 64),    "dim": 1},
    {"id": 4, "dtype": torch.float16,  "shape": (2, 256, 128),   "dim": -1},
    {"id": 5, "dtype": torch.float32,  "shape": (2, 16, 32, 64), "dim": -1},
    {"id": 6, "dtype": torch.float32,  "shape": (64, 256),       "dim": -1},
    {"id": 7, "dtype": torch.float32,  "shape": (1, 2, 4, 128, 64), "dim": 2},
    {"id": 8, "dtype": torch.bfloat16, "shape": (32, 1024),      "dim": -1},
    {"id": 9, "dtype": torch.bfloat16, "shape": (2, 64, 128),    "dim": -1},
]

DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def make_inputs(case):
    """Build every entry of PARAM_NAMES for this case."""
    torch.manual_seed(SEED + case["id"])
    dtype, shape = case["dtype"], case["shape"]
    return {
        "x": torch.randn(shape, dtype=torch.float32, device=DEVICE).to(dtype),
        "dim": case["dim"],
    }


def make_output_bufs(case):
    """(shape, dtype) per output, in OUTPUT_KEYS order."""
    dtype, shape = case["dtype"], case["shape"]
    out_shape = list(shape)
    dim = case["dim"]
    if dim < 0:
        dim = len(out_shape) + dim
    out_shape[dim] = out_shape[dim] // 2
    return [(tuple(out_shape), dtype)]


def build_call_args(case):
    inputs = make_inputs(case)
    return [inputs[p] for p in PARAM_NAMES], inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)

    out_bufs = []
    for s, d in make_output_bufs(case):
        out_bufs.append(torch.empty(s, dtype=d, device=DEVICE))

    out_arg = out_bufs[0] if len(out_bufs) == 1 else out_bufs

    outputs = ops.npu_swi_glu(*call_args, out_arg)
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
            print(f"  binding expects: {ops.npu_swi_glu_custom.__doc__.splitlines()[0] if ops.npu_swi_glu_custom.__doc__ else '?'}")
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
            "shape": list(case["shape"]),
            "attrs": {k: v for k, v in case.items() if k not in ("id", "dtype", "shape")},
        })
        print(f"{name}: OK  input={os.path.getsize(in_path)}B  result={os.path.getsize(out_path)}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
