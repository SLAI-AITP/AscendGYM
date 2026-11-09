#!/usr/bin/env python3
"""Collect golden data for moe_fused_topk by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

# case_0: x:float16[16,4], add_num:float16[4], mapping_num=None, mapping_table=None,
#         group_num=1, group_topk=1, top_n=1, top_k=4, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[16,4], indices:int32[16,4]
# case_1: x:bfloat16[32,8], add_num:bfloat16[8], mapping_num=None, mapping_table=None,
#         group_num=2, group_topk=2, top_n=1, top_k=2, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[32,2], indices:int32[32,2]
# case_2: x:float32[8,2], add_num:float32[2], mapping_num=None, mapping_table=None,
#         group_num=1, group_topk=1, top_n=1, top_k=2, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[8,2], indices:int32[8,2]
# case_3: x:float16[64,32], add_num:float16[32], mapping_num=None, mapping_table=None,
#         group_num=4, group_topk=4, top_n=1, top_k=4, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[64,4], indices:int32[64,4]
# case_4: x:bfloat16[16,4], add_num:bfloat16[4], mapping_num=None, mapping_table=None,
#         group_num=1, group_topk=1, top_n=1, top_k=1, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[16,1], indices:int32[16,1]
# case_5: x:float32[32,8], add_num:float32[8], mapping_num=None, mapping_table=None,
#         group_num=2, group_topk=2, top_n=1, top_k=2, activate_type=1, is_norm=False,
#         scale=2.0, enable_expert_mapping=False -> y:float32[32,2], indices:int32[32,2]
# case_6: x:float16[128,64], add_num:float16[64], mapping_num=None, mapping_table=None,
#         group_num=4, group_topk=4, top_n=1, top_k=4, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[128,4], indices:int32[128,4]
# case_7: x:bfloat16[8,4], add_num:bfloat16[4], mapping_num=None, mapping_table=None,
#         group_num=1, group_topk=1, top_n=1, top_k=3, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[8,3], indices:int32[8,3]
# case_8: x:float32[256,128], add_num:float32[128], mapping_num=None, mapping_table=None,
#         group_num=8, group_topk=8, top_n=1, top_k=8, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[256,8], indices:int32[256,8]
# case_9: x:float16[4,2], add_num:float16[2], mapping_num=None, mapping_table=None,
#         group_num=1, group_topk=1, top_n=1, top_k=1, activate_type=1, is_norm=True,
#         scale=1.0, enable_expert_mapping=False -> y:float32[4,1], indices:int32[4,1]
"""
import json, os, sys
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "moe_fused_topk"
OP_CAMEL = "MoeFusedTopk"
DEVICE = "npu:0"
SEED = 2026

# Positional order expected by ops.npu_moe_fused_topk(...), excluding output buffers.
PARAM_NAMES = [
    "x", "add_num", "mapping_num", "mapping_table",
    "group_num", "group_topk", "top_n", "top_k",
    "activate_type", "is_norm", "scale", "enable_expert_mapping",
]
OUTPUT_KEYS = ["y", "indices"]

# 10 cases spanning dtypes / shapes / group configs
CASES = [
    {"id": 0, "dtype": torch.float16,  "shape": (16, 4),   "group_num": 1, "group_topk": 1, "top_n": 1, "top_k": 4, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
    {"id": 1, "dtype": torch.bfloat16, "shape": (32, 8),   "group_num": 2, "group_topk": 2, "top_n": 1, "top_k": 2, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
    {"id": 2, "dtype": torch.float32,  "shape": (8, 2),    "group_num": 1, "group_topk": 1, "top_n": 1, "top_k": 2, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
    {"id": 3, "dtype": torch.float16,  "shape": (64, 32),  "group_num": 4, "group_topk": 4, "top_n": 1, "top_k": 4, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
    {"id": 4, "dtype": torch.bfloat16, "shape": (16, 4),   "group_num": 1, "group_topk": 1, "top_n": 1, "top_k": 1, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
    {"id": 5, "dtype": torch.float32,  "shape": (32, 8),   "group_num": 2, "group_topk": 2, "top_n": 1, "top_k": 2, "is_norm": False, "scale": 2.0, "enable_expert_mapping": False},
    {"id": 6, "dtype": torch.float16,  "shape": (128, 64), "group_num": 4, "group_topk": 4, "top_n": 1, "top_k": 4, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
    {"id": 7, "dtype": torch.bfloat16, "shape": (8, 4),    "group_num": 1, "group_topk": 1, "top_n": 1, "top_k": 3, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
    {"id": 8, "dtype": torch.float32,  "shape": (256, 128),"group_num": 8, "group_topk": 8, "top_n": 1, "top_k": 8, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
    {"id": 9, "dtype": torch.float16,  "shape": (4, 2),    "group_num": 1, "group_topk": 1, "top_n": 1, "top_k": 1, "is_norm": True,  "scale": 1.0, "enable_expert_mapping": False},
]

DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def make_inputs(case):
    """Build every entry of PARAM_NAMES for this case."""
    torch.manual_seed(SEED + case["id"])
    dtype = case["dtype"]
    shape = case["shape"]
    expert_num = shape[1]
    return {
        "x": torch.randn(shape, dtype=dtype, device=DEVICE),
        "add_num": torch.randn((expert_num,), dtype=dtype, device=DEVICE),
        "mapping_num": None,
        "mapping_table": None,
        "group_num": case["group_num"],
        "group_topk": case["group_topk"],
        "top_n": case["top_n"],
        "top_k": case["top_k"],
        "activate_type": 1,
        "is_norm": case["is_norm"],
        "scale": case["scale"],
        "enable_expert_mapping": case["enable_expert_mapping"],
    }


def make_output_bufs(case):
    """(shape, dtype) per output, in OUTPUT_KEYS order."""
    num_token = case["shape"][0]
    top_k = case["top_k"]
    specs = [(num_token, top_k), (num_token, top_k)]
    dtypes = [torch.float32, torch.int32]
    return [torch.empty(s, dtype=d, device=DEVICE) for s, d in zip(specs, dtypes)]


def build_call_args(case):
    inputs = make_inputs(case)
    positional = [inputs[p] for p in PARAM_NAMES]
    return positional, inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)

    out_bufs = make_output_bufs(case)
    out_arg = out_bufs[0] if len(out_bufs) == 1 else out_bufs

    outputs = ops.npu_moe_fused_topk(*call_args, out_arg)
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
            print(f"  binding expects: {ops.npu_moe_fused_topk.__doc__.splitlines()[0] if ops.npu_moe_fused_topk.__doc__ else '?'}")
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
