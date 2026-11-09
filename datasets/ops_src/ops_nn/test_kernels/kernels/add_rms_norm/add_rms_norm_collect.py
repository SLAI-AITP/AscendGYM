#!/usr/bin/env python3
"""Collect golden data for add_rms_norm by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

# case_0: in=[x1:float16[2, 16], x2:float16[2, 16], gamma:float16[16], epsilon=1e-06] -> out=[y:float16, rstd:float32, x:float16]
# case_1: in=[x1:float16[1, 256], x2:float16[1, 256], gamma:float16[256], epsilon=1e-06] -> out=[y:float16, rstd:float32, x:float16]
# case_2: in=[x1:float16[4, 64, 32], x2:float16[4, 64, 32], gamma:float16[32], epsilon=1e-06] -> out=[y:float16, rstd:float32, x:float16]
# case_3: in=[x1:float16[8, 128, 64], x2:float16[8, 128, 64], gamma:float16[64], epsilon=1e-05] -> out=[y:float16, rstd:float32, x:float16]
# case_4: in=[x1:float16[2, 256, 128], x2:float16[2, 256, 128], gamma:float16[128], epsilon=1e-06] -> out=[y:float16, rstd:float32, x:float16]
# case_5: in=[x1:float32[2, 16, 32, 64], x2:float32[2, 16, 32, 64], gamma:float32[64], epsilon=1e-06] -> out=[y:float32, rstd:float32, x:float32]
# case_6: in=[x1:float32[64, 256], x2:float32[64, 256], gamma:float32[256], epsilon=1e-06] -> out=[y:float32, rstd:float32, x:float32]
# case_7: in=[x1:float32[1, 2, 4, 128, 64], x2:float32[1, 2, 4, 128, 64], gamma:float32[64], epsilon=1e-06] -> out=[y:float32, rstd:float32, x:float32]
# case_8: in=[x1:bfloat16[32, 1024], x2:bfloat16[32, 1024], gamma:bfloat16[1024], epsilon=1e-06] -> out=[y:bfloat16, rstd:float32, x:bfloat16]
# case_9: in=[x1:bfloat16[2, 64, 128], x2:bfloat16[2, 64, 128], gamma:bfloat16[128], epsilon=1e-06] -> out=[y:bfloat16, rstd:float32, x:bfloat16]
"""
import json, os, sys
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "add_rms_norm"
OP_CAMEL = "AddRmsNorm"
DEVICE = "npu:0"
SEED = 2026

# Positional order expected by ops.npu_add_rms_norm(...), excluding output buffers.
PARAM_NAMES = ["x1", "x2", "gamma", "epsilon"]
OUTPUT_KEYS = ["y", "rstd", "x"]

# 10 cases spanning dtypes / ranks / sizes
CASES = [
    {"id": 0, "dtype": torch.float16,  "shape": (2, 16),         "epsilon": 1e-6},
    {"id": 1, "dtype": torch.float16,  "shape": (1, 256),        "epsilon": 1e-6},
    {"id": 2, "dtype": torch.float16,  "shape": (4, 64, 32),     "epsilon": 1e-6},
    {"id": 3, "dtype": torch.float16,  "shape": (8, 128, 64),    "epsilon": 1e-5},
    {"id": 4, "dtype": torch.float16,  "shape": (2, 256, 128),   "epsilon": 1e-6},
    {"id": 5, "dtype": torch.float32,  "shape": (2, 16, 32, 64), "epsilon": 1e-6},
    {"id": 6, "dtype": torch.float32,  "shape": (64, 256),       "epsilon": 1e-6},
    {"id": 7, "dtype": torch.float32,  "shape": (1, 2, 4, 128, 64), "epsilon": 1e-6},
    {"id": 8, "dtype": torch.bfloat16, "shape": (32, 1024),      "epsilon": 1e-6},
    {"id": 9, "dtype": torch.bfloat16, "shape": (2, 64, 128),    "epsilon": 1e-6},
]

DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def make_inputs(case):
    """Build every entry of PARAM_NAMES for this case."""
    torch.manual_seed(SEED + case["id"])
    dtype, shape = case["dtype"], case["shape"]
    return {
        "x1": torch.randn(shape, dtype=torch.float32, device=DEVICE).to(dtype),
        "x2": torch.randn(shape, dtype=torch.float32, device=DEVICE).to(dtype),
        "gamma": torch.randn(shape[-1:], dtype=torch.float32, device=DEVICE).to(dtype),
        "epsilon": case["epsilon"],
    }


def make_output_bufs(case):
    """(shape, dtype) per output, in OUTPUT_KEYS order."""
    dtype = case["dtype"]
    shape = list(case["shape"])
    # rstd: last dim reduced to 1, rest same
    rstd_shape = list(shape)
    rstd_shape[-1] = 1
    return [
        (tuple(shape), dtype),      # y
        (tuple(rstd_shape), torch.float32),  # rstd (always float32)
        (tuple(shape), dtype),      # x
    ]


def build_call_args(case):
    inputs = make_inputs(case)
    return [inputs[p] for p in PARAM_NAMES], inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)

    out_bufs = []
    for s, d in make_output_bufs(case):
        out_bufs.append(torch.empty(s, dtype=d, device=DEVICE))

    out_arg = out_bufs  # 3 outputs -> list

    outputs = ops.npu_add_rms_norm(*call_args, out_arg)
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
            print(f"  binding expects: {ops.npu_add_rms_norm.__doc__.splitlines()[0] if ops.npu_add_rms_norm.__doc__ else '?'}")
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
