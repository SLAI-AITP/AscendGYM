#!/usr/bin/env python3
"""Collect golden data for recurrent_gated_delta_rule by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

SHAPE MODEL — from op_host/recurrent_gated_delta_rule_def.cpp + docs/aclnnRecurrentGatedDeltaRule.md
+ recurrent_gated_delta_rule_tiling.cpp. B=batch, T=sum(actual_seq_lengths) tokens, Nk/Nv=head
counts, Dk/Dv=head dims, BlockNum=number of state blocks.

  query              (T, Nk, Dk)      bf16        -1..1, L2-normalized when large
  key                (T, Nk, Dk)      bf16        -1..1
  value              (T, Nv, Dv)      bf16
  beta               (T, Nv)          bf16        0 < beta < 1
  state              (BlockNum, Nv, Dv, Dk)  bf16 or fp32   in-place S; BlockNum >= T
  actual_seq_lengths (B,)             int32       each in [1,8], sum == T
  ssm_state_indices  (T,)             int32       values in [0, BlockNum)
  g                  (T, Nv)          fp32 opt    alpha = exp(g), g < 0
  gk                 (T, Nv, Dk)      fp32 opt    alphak = exp(gk), gk < 0
  num_accepted_tokens(B,)             int32 opt   1 <= nat[i] <= actual_seq_lengths[i]

  out                (T, Nv, Dv)      bf16        output o = S_t q_t / sqrt(dk)
  state              (final, in-place) == state input buffer

  attrs: scale_value (float, default 1/sqrt(Dk)) — query scaling factor

CONSTRAINTS (from recurrent_gated_delta_rule_tiling.cpp):
  0 < Nk <= 256, Nk <= Nv <= 256, Nv % Nk == 0
  0 < Dk <= 512, 0 < Dv <= 512, 0 < T, 0 < B, T <= BlockNum
  actual_seq_lengths[i] in (0,8], sum == T; ssm_state_indices[i] < BlockNum
  query/key in [-1,1]; g/gk < 0; 0 < beta < 1

CALL (binding.py wrapper, op_def order): query,key,value,beta,state,actual_seq_lengths,
  ssm_state_indices,g,gk,num_accepted_tokens,scale_value + single template out buffer.
  'state' is in-place: the SAME buffer is passed as input (pos 4) and returned as output[1].
  Returns (out, state).

# case_0: bf16  B2 seq[2,2] T4  Nk1 Nv1  Dk16 Dv32  BlockNum4  no g/gk/nat  scale=1/sqrt(16)
# case_1: bf16  B1 seq[4]   T4  Nk2 Nv4  Dk32 Dv32  BlockNum8  g+gk present
# case_2: bf16  B3 seq[1,2,1] T4 Nk1 Nv2 Dk64 Dv128 BlockNum4 g present, nat present
# case_3: bf16  B1 seq[8]   T8  Nk1 Nv1  Dk128 Dv64  BlockNum8  no g/gk  scale=0.7 (non-default)
# case_4: fp32-state  B4 seq[1,1,1,1] T4 Nk4 Nv8 Dk32 Dv32 BlockNum4 g+gk+nat
# case_5: bf16  B1 seq[5]   T5  Nk1 Nv2  Dk32  Dv16  BlockNum16 g only (BlockNum > T)
# case_6: bf16  B2 seq[3,3] T6  Nk2 Nv8  Dk64  Dv64  BlockNum6  g+gk
# case_7: fp32-state  B1 seq[2] T2 Nk1 Nv1 Dk16 Dv16 BlockNum4 no g/gk (tiny)
# case_8: bf16  B2 seq[4,4] T8  Nk4 Nv16 Dk32  Dv32  BlockNum16 g+gk+nat
# case_9: bf16  B3 seq[4,1,3] T8 Nk2 Nv4  Dk128 Dv128 BlockNum8  g+gk (big Dk)
"""
import inspect
import json
import os
import sys

import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "recurrent_gated_delta_rule"
OP_CAMEL = "RecurrentGatedDeltaRule"
DEVICE = "npu:0"
SEED = 2039

# Positional order of the generated wrapper (binding.py), excluding the single template
# 'out' buffer. 'state' is in-place: it is passed as input (pos 4) and returned as output[1].
PARAM_NAMES = [
    "query", "key", "value", "beta", "state",
    "actual_seq_lengths", "ssm_state_indices",
    "g", "gk", "num_accepted_tokens", "scale_value",
]
OUTPUT_KEYS = ["out", "state"]


def _seq_lengths(case):
    """Return actual_seq_lengths (list of B) whose sum == T."""
    return case["seq"]


def _block_num(case):
    return case.get("block_num") or case["T"]


def make_inputs(case):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED + case["id"])
    dt = torch.bfloat16
    state_dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[case["state_dt"]]
    B = case["B"]
    T = case["T"]
    Nk, Nv = case["Nk"], case["Nv"]
    Dk, Dv = case["Dk"], case["Dv"]
    blk_num = _block_num(case)
    seq = _seq_lengths(case)
    assert len(seq) == B and sum(seq) == T

    # query/key in [-1,1]; L2-normalize so the S q_t accumulation stays bounded (no inf/NaN).
    q = (torch.randn((T, Nk, Dk), generator=g, device=DEVICE) * 0.5)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = (torch.randn((T, Nk, Dk), generator=g, device=DEVICE) * 0.5)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    v = (torch.randn((T, Nv, Dv), generator=g, device=DEVICE) * 0.5).to(dt)
    beta = (torch.rand((T, Nv), generator=g, device=DEVICE) * 0.9 + 0.05).to(dt)  # 0.05..0.95
    state = (torch.randn((blk_num, Nv, Dv, Dk), generator=g, device=DEVICE) * 0.3).to(state_dt)

    actual_seq_lengths = torch.tensor(seq, dtype=torch.int32, device=DEVICE)
    ssm_state_indices = torch.arange(T, dtype=torch.int32, device=DEVICE)

    # optional inputs
    has_g = case.get("g", False)
    has_gk = case.get("gk", False)
    has_nat = case.get("nat", False)
    gg = None
    gk = None
    nat = None
    if has_g:
        gg = (torch.rand((T, Nv), generator=g, device=DEVICE) * -2.0).to(torch.float32)  # <0
    if has_gk:
        gk = (torch.rand((T, Nv, Dk), generator=g, device=DEVICE) * -2.0).to(torch.float32)  # <0
    if has_nat:
        nat = torch.zeros(B, dtype=torch.int32, device=DEVICE)
        off = 0
        for i, length in enumerate(seq):
            nat[i] = torch.randint(1, length + 1, (1,), generator=g, device=DEVICE).item()
        nat = nat.to(torch.int32)

    scale_value = case.get("scale") or float(Dk) ** -0.5

    return {
        "query": q.to(dt),
        "key": k.to(dt),
        "value": v,
        "beta": beta,
        "state": state,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "g": gg,
        "gk": gk,
        "num_accepted_tokens": nat,
        "scale_value": scale_value,
    }


def make_output_bufs(case):
    """Single template output buffer for 'out' (shape (T,Nv,Dv), bf16). The in-place 'state'
    output is the input state buffer already built in make_inputs — no separate buffer needed."""
    dt = torch.bfloat16
    T, Nv, Dv = case["T"], case["Nv"], case["Dv"]
    out = torch.zeros((T, Nv, Dv), dtype=dt, device=DEVICE)
    return [out]


def build_call_args(case):
    inputs = make_inputs(case)
    out_bufs = make_output_bufs(case)
    call_args = [inputs[p] for p in PARAM_NAMES] + out_bufs
    return call_args, inputs, out_bufs


def collect_case(case):
    call_args, inputs, out_bufs = build_call_args(case)

    # Snapshot the TRUE initial state BEFORE the kernel writes in-place: 'state' is an in-place
    # input/output, so the buffer passed at pos 4 is mutated by the kernel. The saved golden
    # INPUT must be the initial state (the kernel starts from it), not the edited final buffer.
    initial_state = inputs["state"].clone()

    outputs = ops.npu_recurrent_gated_delta_rule(*call_args)
    torch.npu.synchronize()

    if outputs is None:
        outputs = out_bufs
    elif isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    else:
        outputs = list(outputs)
    if len(outputs) != len(OUTPUT_KEYS):
        raise RuntimeError(f"case_{case['id']}: got {len(outputs)} outputs, expected {OUTPUT_KEYS}")

    # outputs[0]=out (fresh tensor), outputs[1]=state (the in-place input buffer).
    saved_out = {}
    for i, k in enumerate(OUTPUT_KEYS):
        t = outputs[i]
        t = t.detach().cpu()
        if not torch.isfinite(t.float()).all():
            raise RuntimeError(f"case_{case['id']}: output '{k}' has NaN/Inf — bad golden data")
        saved_out[k] = t

    saved_in = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    # 'state' in the saved input must be the initial (pre-kernel) value, not the mutated buffer.
    saved_in["state"] = initial_state.cpu()
    return saved_in, saved_out


def describe_call(call_args):
    parts = []
    for name, val in zip(PARAM_NAMES + ["out"], call_args):
        if isinstance(val, torch.Tensor):
            parts.append(f"{name}=Tensor{tuple(val.shape)}:{str(val.dtype).replace('torch.', '')}")
        elif val is None:
            parts.append(f"{name}=None")
        else:
            parts.append(f"{name}={val!r}")
    return ", ".join(parts)


def check_signature():
    params = list(inspect.signature(ops.npu_recurrent_gated_delta_rule).parameters)
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
    {"id": 0, "B": 2, "seq": [2, 2], "T": 4, "Nk": 1, "Nv": 1, "Dk": 16, "Dv": 32,
     "block_num": 4, "state_dt": "bf16", "g": False, "gk": False, "nat": False},
    {"id": 1, "B": 1, "seq": [4], "T": 4, "Nk": 2, "Nv": 4, "Dk": 32, "Dv": 32,
     "block_num": 8, "state_dt": "bf16", "g": True, "gk": True, "nat": False},
    {"id": 2, "B": 3, "seq": [1, 2, 1], "T": 4, "Nk": 1, "Nv": 2, "Dk": 64, "Dv": 128,
     "block_num": 4, "state_dt": "bf16", "g": True, "gk": False, "nat": True},
    {"id": 3, "B": 1, "seq": [8], "T": 8, "Nk": 1, "Nv": 1, "Dk": 128, "Dv": 64,
     "block_num": 8, "state_dt": "bf16", "g": False, "gk": False, "nat": False, "scale": 0.7},
    {"id": 4, "B": 4, "seq": [1, 1, 1, 1], "T": 4, "Nk": 4, "Nv": 8, "Dk": 32, "Dv": 32,
     "block_num": 4, "state_dt": "fp32", "g": True, "gk": True, "nat": True},
    {"id": 5, "B": 1, "seq": [5], "T": 5, "Nk": 1, "Nv": 2, "Dk": 32, "Dv": 16,
     "block_num": 16, "state_dt": "bf16", "g": True, "gk": False, "nat": False},
    {"id": 6, "B": 2, "seq": [3, 3], "T": 6, "Nk": 2, "Nv": 8, "Dk": 64, "Dv": 64,
     "block_num": 6, "state_dt": "bf16", "g": True, "gk": True, "nat": False},
    {"id": 7, "B": 1, "seq": [2], "T": 2, "Nk": 1, "Nv": 1, "Dk": 16, "Dv": 16,
     "block_num": 4, "state_dt": "fp32", "g": False, "gk": False, "nat": False},
    {"id": 8, "B": 2, "seq": [4, 4], "T": 8, "Nk": 4, "Nv": 16, "Dk": 32, "Dv": 32,
     "block_num": 16, "state_dt": "bf16", "g": True, "gk": True, "nat": True},
    {"id": 9, "B": 3, "seq": [4, 1, 3], "T": 8, "Nk": 2, "Nv": 4, "Dk": 128, "Dv": 128,
     "block_num": 8, "state_dt": "bf16", "g": True, "gk": True, "nat": False},
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
                doc = ops.npu_recurrent_gated_delta_rule.__doc__
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
            "dtype": "bf16",
            "state_dt": case["state_dt"],
            "shape": [case["T"], case["Nv"], case["Dv"]],
            "attrs": {k: v for k, v in case.items() if k not in ("id", "dtype", "shape")},
        })
        print(f"{name}: OK  input={os.path.getsize(os.path.join(INPUT_DIR, name + '.pt'))}B  "
              f"result={os.path.getsize(os.path.join(RESULT_DIR, name + '.pt'))}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
