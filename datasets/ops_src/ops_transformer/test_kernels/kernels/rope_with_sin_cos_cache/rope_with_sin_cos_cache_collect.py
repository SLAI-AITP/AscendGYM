#!/usr/bin/env python3
"""Collect golden data for rope_with_sin_cos_cache by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

SHAPE MODEL — from op_host/rope_with_sin_cos_cache_def.cpp + op_host/op_api/aclnn_rope_with_sin_cos_cache.h
+ op_host/rope_with_sin_cos_cache_tiling.cpp. T = number of tokens.

  positions    (S, T) int64            mrope mode: S == len(mropeSection) ∈ {3, 4}; each row is
                                       a distinct set of cache-row indices (pos0..posS-1)
  queryIn      (T, numQHeads*headSize)
  keyIn        (T, numKHeads*headSize)
  cosSinCache  (maxSeqLen, rotaryDim)  rotaryDim = cosSinCache last dim
  queryOut     (T, numQHeads*headSize)   = queryIn shape (template output)
  keyOut       (T, numKHeads*headSize)   = keyIn shape (template output)

aclnn VALIDATION (strict, from aclnn_rope_with_sin_cos_cache.cpp) — a case must satisfy all of:
  * queryIn/keyIn/cosSinCache are 2D; keyIn.shape[0] == queryIn.shape[0]; positions non-empty
  * dtype ∈ {float16, bfloat16, float32}; keyIn/cosSinCache/queryOut/keyOut match queryIn dtype
  * mropeSection size is 3 or 4 AND is one of mropeSupportList:
        {16,24,24}, {8,12,12}, {24,20,20}, {11,11,10}, {16,16,16,16}
  * mrope mode (section[0] > 0): sum(mropeSection) * 2 == rotaryDim
        -> rotaryDim = cosSinCache last dim = 2*sum(sections)
  * headSize != 0, cacheMode ∈ {0, 1}; mrope-with-4-sections requires cacheMode == 0

TILING (from tiling.cpp) — headSize and rotaryDim must be 64-byte aligned:
  * fp16/bf16 (2B): headSize, rotaryDim are multiples of 32
  * fp32 (4B):      headSize, rotaryDim are multiples of 16
  * headSize >= rotaryDim

BINDING NOTE — the aclnn takes ONLY (positions, queryIn, keyIn, cosSinCache, mropeSection,
headSize, isNeoxStyle) as inputs; numQHeads/numKHeads/qStride/kStride/cacheMode are computed
inside the aclnn wrapper from the tensor shapes and ignored. The autobind (aclnn_header.py)
reflects that, so the binding keeps them in the Python signature but does not pass them to the
aclnn call. PARAM_NAMES below matches the wrapper exactly; the values for the ignored attrs are
still supplied for clarity (qStride=kStride=0 => the wrapper computes strides from the tensors).

CASE MATRIX — all mrope mode (rope mode is unreachable through this binding because it needs a
nullptr mropeSection which a vector parameter cannot express):

# case_0:  fp16 sections=[16,24,24] sum=64 rot=128 head=128 Q4 /K4   T=8    maxSeq=256  neox
# case_1:  bf16 sections=[8,12,12]  sum=32 rot=64  head=64  Q32/K8   T=16   maxSeq=256  neox (GQA)
# case_2:  fp32 sections=[16,16,16,16] sum=64 rot=128 head=128 Q8/K8 T=64   maxSeq=512  neox (4-section)
# case_3:  fp16 sections=[24,20,20] sum=64 rot=128 head=128 Q16/K4  T=512  maxSeq=1024 interleaved (large)
# case_4:  bf16 sections=[11,11,10] sum=32 rot=64  head=64  Q8/K8   T=1    maxSeq=128  interleaved (tiny)
# case_5:  fp16 sections=[16,24,24] sum=64 rot=128 head=128 Q32/K8  T=2048 maxSeq=4096 neox (very large)
# case_6:  fp32 sections=[24,20,20] sum=64 rot=128 head=128 Q8/K1   T=32   maxSeq=512  neox (GQA)
# case_7:  bf16 sections=[8,12,12]  sum=32 rot=64  head=64  Q16/K16 T=128  maxSeq=1024 neox
# case_8:  fp16 sections=[16,16,16,16] sum=64 rot=128 head=128 Q4/K4 T=256 maxSeq=2048 interleaved
# case_9:  fp16 sections=[11,11,10] sum=32 rot=64  head=96  Q8/K8   T=16   maxSeq=256  neox (pass-through)
"""
import json, os, sys
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "rope_with_sin_cos_cache"
OP_CAMEL = "RopeWithSinCosCache"
DEVICE = "npu:0"
SEED = 2026

# Positional order of the generated wrapper (binding.py), excluding 'out'.
# The aclnn ignores numQHeads/numKHeads/qStride/kStride/cacheMode (computes them from shapes),
# but the wrapper signature includes them; check_signature() verifies PARAM_NAMES against it.
PARAM_NAMES = [
    "positions", "queryIn", "keyIn", "cosSinCache",
    "numQHeads", "numKHeads", "headSize",
    "mropeSection", "qStride", "kStride", "isNeoxStyle", "cacheMode",
]
OUTPUT_KEYS = ["queryOut", "keyOut"]

# mropeSection lists accepted by aclnn (mropeSupportList in aclnn_rope_with_sin_cos_cache.cpp).
DEFAULT_CACHE_MODE = 0
DEFAULT_QSTRIDE = 0
DEFAULT_KSTRIDE = 0

CASES = [
    {"id": 0, "dtype": torch.float16,  "sections": [16, 24, 24], "numQ": 4,  "numK": 4,
     "T": 8,    "maxSeq": 256,  "isNeoxStyle": True},
    {"id": 1, "dtype": torch.bfloat16, "sections": [8, 12, 12], "numQ": 32, "numK": 8,
     "T": 16,   "maxSeq": 256,  "isNeoxStyle": True},
    {"id": 2, "dtype": torch.float32,  "sections": [16, 16, 16, 16], "numQ": 8, "numK": 8,
     "T": 64,   "maxSeq": 512,  "isNeoxStyle": True},
    {"id": 3, "dtype": torch.float16,  "sections": [24, 20, 20], "numQ": 16, "numK": 4,
     "T": 512,  "maxSeq": 1024, "isNeoxStyle": False},
    {"id": 4, "dtype": torch.bfloat16, "sections": [11, 11, 10], "numQ": 8, "numK": 8,
     "T": 1,    "maxSeq": 128,  "isNeoxStyle": False},
    {"id": 5, "dtype": torch.float16,  "sections": [16, 24, 24], "numQ": 32, "numK": 8,
     "T": 2048, "maxSeq": 4096, "isNeoxStyle": True},
    {"id": 6, "dtype": torch.float32,  "sections": [24, 20, 20], "numQ": 8, "numK": 1,
     "T": 32,   "maxSeq": 512,  "isNeoxStyle": True},
    {"id": 7, "dtype": torch.bfloat16, "sections": [8, 12, 12], "numQ": 16, "numK": 16,
     "T": 128,  "maxSeq": 1024, "isNeoxStyle": True},
    {"id": 8, "dtype": torch.float16,  "sections": [16, 16, 16, 16], "numQ": 4, "numK": 4,
     "T": 256,  "maxSeq": 2048, "isNeoxStyle": False},
    {"id": 9, "dtype": torch.float16,  "sections": [11, 11, 10], "numQ": 8, "numK": 8,
     "T": 16,   "maxSeq": 256,  "isNeoxStyle": True, "headSize_override": 96},
]

DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)


def rotary_and_head(case):
    """rotaryDim (cosSinCache last dim) = 2*sum(sections); headSize >= rotaryDim and aligned."""
    rotary = 2 * sum(case["sections"])
    head = case.get("headSize_override", rotary)
    type_len = 2 if case["dtype"] in (torch.float16, torch.bfloat16) else 4
    assert head % (64 // type_len) == 0, f"headSize {head} not 64B-aligned for {case['dtype']}"
    assert rotary % (64 // type_len) == 0, f"rotaryDim {rotary} not 64B-aligned"
    assert head >= rotary, "headSize < rotaryDim"
    assert sum(case["sections"]) * 2 == rotary
    assert 3 <= len(case["sections"]) <= 4
    return rotary, head


def make_inputs(case):
    """Build every entry of PARAM_NAMES for this case."""
    torch.manual_seed(SEED + case["id"])
    dtype = case["dtype"]
    rotary, head = rotary_and_head(case)
    T = case["T"]
    numS = len(case["sections"])
    maxSeq = case["maxSeq"]

    queryIn = torch.randn(T, case["numQ"] * head, dtype=torch.float32, device=DEVICE).to(dtype)
    keyIn = torch.randn(T, case["numK"] * head, dtype=torch.float32, device=DEVICE).to(dtype)
    # Real rope tables: cos on the first half, sin on the second half of each cache row,
    # so values stay in [-1, 1] and the golden is meaningful.
    freq = torch.arange(rotary // 2, dtype=torch.float32, device=DEVICE).unsqueeze(0) / max(rotary // 2, 1)
    pos_rows = torch.arange(rotary // 2, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    ang = pos_rows * freq
    cos = torch.cos(ang).to(dtype)
    sin = torch.sin(ang).to(dtype)
    cos_sin_cache = torch.cat([cos, sin], dim=1)  # (rotary/2, rotary) -> repeat along rows below

    # cosSinCache rows >= num positions we need; tile the table to maxSeqLen rows.
    reps = (maxSeq + cos_sin_cache.shape[0] - 1) // cos_sin_cache.shape[0]
    cos_sin_cache = cos_sin_cache.repeat(reps, 1)[:maxSeq, :].contiguous()

    positions = torch.randint(0, maxSeq, (numS, T), dtype=torch.int64, device=DEVICE)

    return {
        "positions": positions,
        "queryIn": queryIn,
        "keyIn": keyIn,
        "cosSinCache": cos_sin_cache,
        "numQHeads": int(case["numQ"]),
        "numKHeads": int(case["numK"]),
        "headSize": int(head),
        "mropeSection": list(case["sections"]),
        "qStride": DEFAULT_QSTRIDE,
        "kStride": DEFAULT_KSTRIDE,
        "isNeoxStyle": bool(case["isNeoxStyle"]),
        "cacheMode": 0 if len(case["sections"]) == 4 else DEFAULT_CACHE_MODE,
    }


def make_output_bufs(case):
    """out_tmpl buffers in OUTPUT_KEYS order: (T, numHeads*headSize) for query and key."""
    _, head = rotary_and_head(case)
    T = case["T"]
    dtype = case["dtype"]
    return [
        torch.empty(T, case["numQ"] * head, dtype=dtype, device=DEVICE),
        torch.empty(T, case["numK"] * head, dtype=dtype, device=DEVICE),
    ]


def build_call_args(case):
    inputs = make_inputs(case)
    positional = [inputs[p] for p in PARAM_NAMES]
    return positional, inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)

    out_bufs = make_output_bufs(case)
    out_arg = out_bufs[0] if len(out_bufs) == 1 else out_bufs

    outputs = ops.npu_rope_with_sin_cos_cache(*call_args, out_arg)
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


def describe_call(call_args):
    """Compact argument summary for signature-mismatch diagnostics."""
    parts = []
    for name, val in zip(PARAM_NAMES, call_args):
        if isinstance(val, torch.Tensor):
            parts.append(f"{name}=Tensor{tuple(val.shape)}:{str(val.dtype).replace('torch.', '')}")
        else:
            parts.append(f"{name}={val!r}")
    return ", ".join(parts)


def check_signature():
    """Compare PARAM_NAMES against the generated wrapper's real signature."""
    import inspect
    try:
        params = list(inspect.signature(ops.npu_rope_with_sin_cos_cache).parameters)
    except (TypeError, ValueError):
        print("[WARN] cannot introspect the wrapper signature — skipping the arity check")
        return
    expected = [p for p in params if p != "out"]
    if expected != PARAM_NAMES:
        print("Signature mismatch between this script and the generated wrapper:")
        print("  wrapper:      %s" % ", ".join(expected))
        print("  PARAM_NAMES:  %s" % ", ".join(PARAM_NAMES))
        missing = [p for p in expected if p not in PARAM_NAMES]
        extra = [p for p in PARAM_NAMES if p not in expected]
        if missing:
            print("  missing here: %s" % ", ".join(missing))
        if extra:
            print("  not in wrapper: %s" % ", ".join(extra))
        sys.exit(1)
    if "out" not in params:
        print("[WARN] the wrapper has no 'out' parameter — every output is in-place")


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
                doc = ops.npu_rope_with_sin_cos_cache.__doc__
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
            "shape": [case["T"], case["numQ"] * rotary_and_head(case)[1]],
            "attrs": {k: (str(v).replace("torch.", "") if isinstance(v, torch.dtype) else v)
                      for k, v in case.items() if k not in ("id", "dtype")},
        })
        print(f"{name}: OK  input={os.path.getsize(in_path)}B  result={os.path.getsize(out_path)}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
