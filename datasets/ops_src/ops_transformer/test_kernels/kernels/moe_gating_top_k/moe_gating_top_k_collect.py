#!/usr/bin/env python3
"""Collect golden data for moe_gating_top_k by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

COMPUTE (from docs/aclnnMoeGatingTopK.md) — per row of x:
  1. normOut  = SoftMax(x)          if normType=0
                Sigmoid(x)          if normType=1
                sqrt(Softplus(x))   if normType=2  (950 only)
  2. normValue = normOut + bias     (if bias given), else normOut
  3. if groupCount > 1: reshape to [rows, groupCount, per_group]; per-group score is
       ReduceMax(.)                  if groupSelectMode=0,
       ReduceSum(TopK(., k=2),.)     if groupSelectMode=1;
     pick top-kGroup groups, mask the rest with -inf.
  4. y, expertIdx = TopK(normValue, k=k)          (expertIdx is INT32, the selected indices)
  5. y = gathered(normOut[expertIdx]) / (ReduceSum(normOut) + eps)   if renorm (nt=0) or nt in (1,2);
       else gathered; then y *= routedScalingFactor.
  6. out (3rd output) = normOut when outFlag=True.

SHAPES / OUTPUTS (from infershape):
  x          (rows, expert_num)   fp16/bf16/fp32    expert_num <= 2048
  bias       (expert_num,)        optional          same dtype as x
  y          (rows, k)            same dtype as x   (top-k values)
  expert_idx (rows, k)            INT32             (selected expert indices)
  out        (rows, expert_num)   FLOAT32 always    (normOut)

CONSTRAINTS (validated by the aclnn layer):
  1 <= k <= expert_num/groupCount * kGroup
  1 <= kGroup <= groupCount
  groupCount > 0, expert_num % groupCount == 0,
  align32(expert_num/groupCount) * groupCount <= 2048
  groupSelectMode in {0,1};  normType in {0,1};  renorm in {0,1}

BINDING NOTE: the aclnn header (aclnn_moe_gating_top_k.h) declares ONLY
(x, bias, k, kGroup, groupCount, groupSelectMode, renorm, normType, outFlag,
routedScalingFactor, eps, y, expertIdx, out). The op_def also declares optional
input_ids and tid2eid (used only by the group-select-by-token-ids path), which the
aclnn call drops. The autobind reflects that via aclnn_call_order, so the Python
signature keeps input_ids/tid2eid but they are not passed to the aclnn call (None).
"""
import inspect
import json
import os
import sys

import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "moe_gating_top_k"
OP_CAMEL = "MoeGatingTopK"
DEVICE = "npu:0"
SEED = 2027

# Positional order of the generated wrapper (binding.py), excluding 'out'.
# input_ids / tid2eid are kept in the wrapper signature but not passed to the aclnn.
PARAM_NAMES = [
    "x", "bias", "input_ids", "tid2eid",
    "k", "k_group", "group_count", "group_select_mode",
    "renorm", "norm_type", "out_flag", "routed_scaling_factor", "eps",
]
OUTPUT_KEYS = ["y", "expert_idx", "out"]

DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

CASES = [
    # id: dtype        rows expert  k  kg gc gsm ren nt   of    rsf   eps      bias
    # 0: fp32, no grouping (gc=1), softmax + renorm, no bias
    {"id": 0, "dt": "fp32", "rows": 4, "expert": 64, "k": 8, "kg": 1, "gc": 1,
     "gsm": 0, "ren": 1, "nt": 0, "of": True, "rsf": 1.5, "eps": 1e-12},
    # 1: fp16, gc=8, gsm=1 (sum-top2), kg=4, sigmoid, bias   (docs example config)
    {"id": 1, "dt": "fp16", "rows": 3, "expert": 256, "k": 8, "kg": 4, "gc": 8,
     "gsm": 1, "ren": 0, "nt": 1, "of": True, "rsf": 1.0, "eps": 1e-20, "bias": True},
    # 2: bf16, gc=4, gsm=0 (max), kg=2, sigmoid, bias
    {"id": 2, "dt": "bf16", "rows": 5, "expert": 128, "k": 2, "kg": 2, "gc": 4,
     "gsm": 0, "ren": 0, "nt": 1, "of": True, "rsf": 0.5, "eps": 1e-20, "bias": True},
    # 3: fp16, gc=1, softmax no renorm, tiny experts, bias
    {"id": 3, "dt": "fp16", "rows": 6, "expert": 32, "k": 16, "kg": 1, "gc": 1,
     "gsm": 0, "ren": 0, "nt": 0, "of": True, "rsf": 2.0, "eps": 1e-6, "bias": True},
    # 4: fp32, gc=16, gsm=0, kg=4, sigmoid, no bias
    # NOTE: out_flag is True here — the docs state the 3rd output (out) is only defined when
    # outFlag=True ("otherwise empty"), so out_flag=False leaves that buffer uninitialized /
    # non-deterministic and the golden could not be compared across sessions.
    {"id": 4, "dt": "fp32", "rows": 2, "expert": 512, "k": 4, "kg": 4, "gc": 16,
     "gsm": 0, "ren": 0, "nt": 1, "of": True, "rsf": 1.0, "eps": 1e-20},
    # 5: fp16, gc=1, sigmoid, max expert_num 2048, no bias
    {"id": 5, "dt": "fp16", "rows": 9, "expert": 2048, "k": 32, "kg": 1, "gc": 1,
     "gsm": 0, "ren": 0, "nt": 1, "of": True, "rsf": 1.0, "eps": 1e-20},
    # 6: bf16, gc=8, gsm=1, kg=8, softmax, bias
    {"id": 6, "dt": "bf16", "rows": 4, "expert": 64, "k": 8, "kg": 8, "gc": 8,
     "gsm": 1, "ren": 0, "nt": 0, "of": True, "rsf": 1.0, "eps": 1e-20, "bias": True},
    # 7: fp32, gc=4, gsm=1, kg=4, sigmoid, single row, no bias
    {"id": 7, "dt": "fp32", "rows": 1, "expert": 128, "k": 4, "kg": 4, "gc": 4,
     "gsm": 1, "ren": 0, "nt": 1, "of": True, "rsf": 0.1, "eps": 1e-20},
    # 8: fp16, gc=2, gsm=1, kg=2, sigmoid, bias, larger k
    {"id": 8, "dt": "fp16", "rows": 7, "expert": 64, "k": 16, "kg": 2, "gc": 2,
     "gsm": 1, "ren": 0, "nt": 1, "of": True, "rsf": 1.0, "eps": 1e-20, "bias": True},
    # 9: bf16, gc=1, softmax + renorm, non-power-of-2 experts, no bias
    {"id": 9, "dt": "bf16", "rows": 2, "expert": 300, "k": 32, "kg": 1, "gc": 1,
     "gsm": 0, "ren": 1, "nt": 0, "of": True, "rsf": 1.0, "eps": 1e-20},
]


def align32(n):
    return ((n + 31) // 32) * 32


def check_constraints(case):
    expert = case["expert"]
    gc, kg, k = case["gc"], case["kg"], case["k"]
    assert expert <= 2048, f"case {case['id']}: expert_num {expert} > 2048"
    assert expert % gc == 0, f"case {case['id']}: expert {expert} not div by gc {gc}"
    assert align32(expert // gc) * gc <= 2048, \
        f"case {case['id']}: align32({expert//gc})*{gc} > 2048"
    assert 1 <= kg <= gc, f"case {case['id']}: k_group {kg} not in [1, groupCount {gc}]"
    assert 1 <= k <= (expert // gc) * kg, \
        f"case {case['id']}: k {k} > (expert/gc)*kg = {(expert//gc)*kg}"
    assert case["gsm"] in (0, 1)
    assert case["nt"] in (0, 1)


DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def make_inputs(case):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED + case["id"])
    dt = DT[case["dt"]]
    # well-conditioned ~N(0, 0.5) scores so sigmoid/softmax and top-k are stable
    x = (torch.randn((case["rows"], case["expert"]), generator=g, device=DEVICE) * 0.5).to(dt)
    bias = None
    if case.get("bias"):
        bias = (torch.randn((case["expert"],), generator=g, device=DEVICE) * 0.5).to(dt)
    return {
        "x": x,
        "bias": bias,
        "input_ids": None,
        "tid2eid": None,
        "k": int(case["k"]),
        "k_group": int(case["kg"]),
        "group_count": int(case["gc"]),
        "group_select_mode": int(case["gsm"]),
        "renorm": int(case["ren"]),
        "norm_type": int(case["nt"]),
        "out_flag": bool(case["of"]),
        "routed_scaling_factor": float(case["rsf"]),
        "eps": float(case["eps"]),
    }


def make_output_bufs(case):
    rows, expert, k = case["rows"], case["expert"], case["k"]
    dt = DT[case["dt"]]
    return [
        torch.empty((rows, k), dtype=dt, device=DEVICE),                 # y
        torch.empty((rows, k), dtype=torch.int32, device=DEVICE),        # expert_idx
        torch.empty((rows, expert), dtype=torch.float32, device=DEVICE),  # out
    ]


def build_call_args(case):
    inputs = make_inputs(case)
    positional = [inputs[p] for p in PARAM_NAMES]
    return positional, inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)
    out_bufs = make_output_bufs(case)

    outputs = ops.npu_moe_gating_top_k(*call_args, out_bufs)
    torch.npu.synchronize()

    if isinstance(outputs, torch.Tensor):
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
    params = list(inspect.signature(ops.npu_moe_gating_top_k).parameters)
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
        check_constraints(case)
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
            "x_shape": [case["rows"], case["expert"]],
            "attrs": {k: v for k, v in case.items() if k not in ("id", "dt", "rows", "expert")},
        })
        print(f"{name}: OK  input={os.path.getsize(in_path)}B  result={os.path.getsize(out_path)}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
