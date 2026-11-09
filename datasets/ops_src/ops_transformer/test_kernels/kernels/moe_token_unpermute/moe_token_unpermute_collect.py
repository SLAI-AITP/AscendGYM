#!/usr/bin/env python3
"""Collect golden data for moe_token_unpermute by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

# case_0: in=[permuted_tokens:float16[16, 64], sorted_indices:int32[16], probs:float16[8, 2], padded_mode=False] -> out=[unpermuted_tokens:float16]
# case_1: in=[permuted_tokens:float16[64, 128], sorted_indices:int32[64], probs:float16[16, 4], padded_mode=False] -> out=[unpermuted_tokens:float16]
# case_2: in=[permuted_tokens:bfloat16[64, 64], sorted_indices:int32[64], probs:bfloat16[32, 2], padded_mode=False] -> out=[unpermuted_tokens:bfloat16]
# case_3: in=[permuted_tokens:float32[12, 256], sorted_indices:int32[12], probs:float32[4, 3], padded_mode=False] -> out=[unpermuted_tokens:float32]
# case_4: in=[permuted_tokens:float16[8, 128], sorted_indices:int32[8], padded_mode=False] -> out=[unpermuted_tokens:float16]
# case_5: in=[permuted_tokens:bfloat16[160, 256], sorted_indices:int32[160], probs:bfloat16[20, 8], padded_mode=False] -> out=[unpermuted_tokens:bfloat16]
# case_6: in=[permuted_tokens:float16[128, 64], sorted_indices:int32[128], probs:float16[64, 2], padded_mode=False] -> out=[unpermuted_tokens:float16]
# case_7: in=[permuted_tokens:float32[512, 512], sorted_indices:int32[512], probs:float32[128, 4], padded_mode=False] -> out=[unpermuted_tokens:float32]
# case_8: in=[permuted_tokens:bfloat16[512, 128], sorted_indices:int32[512], probs:bfloat16[256, 2], padded_mode=False] -> out=[unpermuted_tokens:bfloat16]
# case_9: in=[permuted_tokens:float16[30, 32], sorted_indices:int32[30], probs:float16[10, 3], padded_mode=False] -> out=[unpermuted_tokens:float16]
"""
import json, os, sys
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "moe_token_unpermute"
OP_CAMEL = "MoeTokenUnpermute"
DEVICE = "npu:0"
SEED = 2026

# Actual binding signature:
#   _raw_moe_token_unpermute(x, sorted_indices, probs, padded_mode, restore_shape, out_tmpl)
PARAM_NAMES = ["permuted_tokens", "sorted_indices", "probs", "padded_mode", "restore_shape"]
OUTPUT_KEYS = ["unpermuted_tokens"]

# 10 cases: mix of dtypes (fp16/bf16/fp32), with-probs and without-probs variants
CASES = [
    {"id": 0, "dtype": torch.float16,  "shape": (16, 64),   "with_probs": True,  "topk": 2, "padded_mode": False},
    {"id": 1, "dtype": torch.float16,  "shape": (64, 128),  "with_probs": True,  "topk": 4, "padded_mode": False},
    {"id": 2, "dtype": torch.bfloat16, "shape": (64, 64),   "with_probs": True,  "topk": 2, "padded_mode": False},
    {"id": 3, "dtype": torch.float32,  "shape": (12, 256),  "with_probs": True,  "topk": 3, "padded_mode": False},
    {"id": 4, "dtype": torch.float16,  "shape": (8, 128),   "with_probs": False, "topk": 1, "padded_mode": False},
    {"id": 5, "dtype": torch.bfloat16, "shape": (160, 256), "with_probs": True,  "topk": 8, "padded_mode": False},
    {"id": 6, "dtype": torch.float16,  "shape": (128, 64),  "with_probs": True,  "topk": 2, "padded_mode": False},
    {"id": 7, "dtype": torch.float32,  "shape": (512, 512), "with_probs": True,  "topk": 4, "padded_mode": False},
    {"id": 8, "dtype": torch.bfloat16, "shape": (512, 128), "with_probs": True,  "topk": 2, "padded_mode": False},
    {"id": 9, "dtype": torch.float16,  "shape": (30, 32),   "with_probs": True,  "topk": 3, "padded_mode": False},
]

DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def make_inputs(case):
    """Build every entry of PARAM_NAMES for this case."""
    torch.manual_seed(SEED + case["id"])
    dtype, shape = case["dtype"], case["shape"]
    topk = case["topk"]

    # permuted_tokens: [num_permuted_tokens, hidden]
    num_permuted = shape[0]
    hidden = shape[1]
    permuted_tokens = torch.randn(num_permuted, hidden, dtype=torch.float32, device=DEVICE).to(dtype)

    # sorted_indices: [num_permuted_tokens] — int32 indices into permuted_tokens
    sorted_indices = torch.arange(num_permuted, dtype=torch.int32, device=DEVICE)
    # Shuffle to make it non-trivial
    perm = torch.randperm(num_permuted, device=DEVICE)
    sorted_indices = sorted_indices[perm].to(torch.int32)

    inputs = {
        "permuted_tokens": permuted_tokens,
        "sorted_indices": sorted_indices,
        "padded_mode": case["padded_mode"],
    }

    if case["with_probs"]:
        # probs: [num_unpermuted_tokens, topk] where num_unpermuted_tokens = num_permuted // topk
        num_unpermuted = num_permuted // topk
        probs = torch.randn(num_unpermuted, topk, dtype=torch.float32, device=DEVICE).to(dtype)
        inputs["probs"] = probs

    return inputs


def make_output_buf(case):
    """Shape/dtype for the single output."""
    dtype = case["dtype"]
    topk = case["topk"]
    num_permuted = case["shape"][0]
    hidden = case["shape"][1]
    if case["with_probs"]:
        num_unpermuted = num_permuted // topk
    else:
        num_unpermuted = num_permuted
    return (num_unpermuted, hidden), dtype


def build_call_args(case):
    inputs = make_inputs(case)
    args = []
    for pname in PARAM_NAMES:
        if pname in inputs:
            args.append(inputs[pname])
        elif pname == "probs":
            args.append(None)
        elif pname == "padded_mode":
            args.append(inputs.get("padded_mode", False))
        elif pname == "restore_shape":
            # restore_shape is the original shape before permutation; use input shape
            args.append(list(case["shape"]))
        else:
            raise KeyError(f"Missing param '{pname}' for case_{case['id']}")
    return args, inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)

    out_shape, out_dtype = make_output_buf(case)
    out_buf = torch.empty(out_shape, dtype=out_dtype, device=DEVICE)

    # out_tmpl is the LAST positional argument
    outputs = ops.npu_moe_token_unpermute(*call_args, out_buf)
    torch.npu.synchronize()

    outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
    if len(outputs) != len(OUTPUT_KEYS):
        raise RuntimeError(f"case_{case['id']}: got {len(outputs)} outputs, expected {OUTPUT_KEYS}")

    saved_out = {k: outputs[i].cpu() for i, k in enumerate(OUTPUT_KEYS)}
    for k, t in saved_out.items():
        if not torch.isfinite(t.float()).all():
            raise RuntimeError(f"case_{case['id']}: output '{k}' has NaN/Inf — bad golden data")

    # Save inputs (tensors on CPU, others as-is)
    saved_in = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            saved_in[k] = v.cpu()
        else:
            saved_in[k] = v
    return saved_in, saved_out


def describe_call(call_args):
    """Compact argument summary."""
    parts = []
    for name, val in zip(PARAM_NAMES, call_args):
        if isinstance(val, torch.Tensor):
            parts.append(f"{name}=Tensor{tuple(val.shape)}:{str(val.dtype).replace('torch.', '')}")
        elif isinstance(val, list):
            parts.append(f"{name}=[{len(val)}]")
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
