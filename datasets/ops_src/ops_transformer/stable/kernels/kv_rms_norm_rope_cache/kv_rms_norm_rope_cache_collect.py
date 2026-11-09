#!/usr/bin/env python3
"""Collect golden data for kv_rms_norm_rope_cache by running the NPU operator on every case.

Re-run ONLY after an intentional kernel change — this overwrites the regression baseline.

The operator implements the V1 (kv-combined, N=1) + Norm-cache-mode dialect only:
  * kv:  [B, 1, S, Dv + Dk]   - the V part (kv[..., :Dv]) goes through RMSNorm then into
                                  ckv_cache; the K part (kv[..., Dv:]) goes through the
                                  interleaved RoPE then into k_cache.
  * gamma: [Dv]               - RMSNorm weights (length == the Dv head dim).
  * cos/sin: [B, 1, S or 1, Dk]
  * index: [B, S]             - per-batch row offsets into the cache (values in [-1, Scache)).
  * k_cache:  [B, 1, Scache, Dk]  (in-place: written per index)
  * ckv_cache:[B, 1, Scache, Dv]  (in-place: written per index)

In Norm + non-quant V1 mode the k_rope / c_kv outputs are NOT written by the kernel
(is_output_kv is ineffective), so the regression baseline covers only the two in-place
cache buffers k_cache and ckv_cache.

# case_0: in=[kv:float16[1,1,1,192], gamma:float16[128], cos:float16[1,1,1,64], sin:float16[1,1,1,64], index:int64[1,1], k_cache:float16[1,1,16,64], ckv_cache:float16[1,1,16,128], epsilon=1e-05] -> out=[k_cache:float16, ckv_cache:float16]
# case_1: in=[kv:bfloat16[1,1,8,192], gamma:bfloat16[128], cos:bfloat16[1,1,8,64], sin:bfloat16[1,1,8,64], index:int64[1,8], k_cache:bfloat16[1,1,32,64], ckv_cache:bfloat16[1,1,32,128], epsilon=1e-05] -> out=[k_cache:bfloat16, ckv_cache:bfloat16]
# case_2: in=[kv:float16[2,1,4,256], gamma:float16[128], cos:float16[2,1,4,128], sin:float16[2,1,4,128], index:int64[2,4], k_cache:float16[2,1,64,128], ckv_cache:float16[2,1,64,128], epsilon=1e-05] -> out=[k_cache:float16, ckv_cache:float16]
# case_3: in=[kv:bfloat16[4,1,1,384], gamma:bfloat16[256], cos:bfloat16[4,1,1,128], sin:bfloat16[4,1,1,128], index:int64[4,1], k_cache:bfloat16[4,1,64,128], ckv_cache:bfloat16[4,1,64,256], epsilon=1e-05] -> out=[k_cache:bfloat16, ckv_cache:bfloat16]
# case_4: in=[kv:float16[1,1,16,640], gamma:float16[512], cos:float16[1,1,16,128], sin:float16[1,1,16,128], index:int64[1,16], k_cache:float16[1,1,128,128], ckv_cache:float16[1,1,128,512], epsilon=1e-06] -> out=[k_cache:float16, ckv_cache:float16]
# case_5: in=[kv:bfloat16[1,1,64,320], gamma:bfloat16[256], cos:bfloat16[1,1,64,64], sin:bfloat16[1,1,64,64], index:int64[1,64], k_cache:bfloat16[1,1,256,64], ckv_cache:bfloat16[1,1,256,256], epsilon=1e-05] -> out=[k_cache:bfloat16, ckv_cache:bfloat16]
# case_6: in=[kv:float16[8,1,1,128], gamma:float16[64], cos:float16[8,1,1,64], sin:float16[8,1,1,64], index:int64[8,1], k_cache:float16[8,1,32,64], ckv_cache:float16[8,1,32,64], epsilon=1e-05] -> out=[k_cache:float16, ckv_cache:float16]
# case_7: in=[kv:bfloat16[2,1,32,512], gamma:bfloat16[256], cos:bfloat16[2,1,32,256], sin:bfloat16[2,1,32,256], index:int64[2,32], k_cache:bfloat16[2,1,128,256], ckv_cache:bfloat16[2,1,128,256], epsilon=1e-04] -> out=[k_cache:bfloat16, ckv_cache:bfloat16]
# case_8: in=[kv:float16[1,1,4,1024], gamma:float16[512], cos:float16[1,1,4,512], sin:float16[1,1,4,512], index:int64[1,4], k_cache:float16[1,1,64,512], ckv_cache:float16[1,1,64,512], epsilon=1e-05] -> out=[k_cache:float16, ckv_cache:float16]
# case_9: in=[kv:bfloat16[1,1,32,640], gamma:bfloat16[512], cos:bfloat16[1,1,32,128], sin:bfloat16[1,1,32,128], index:int64[1,32], k_cache:bfloat16[1,1,64,128], ckv_cache:bfloat16[1,1,64,512], epsilon=1e-05] -> out=[k_cache:bfloat16, ckv_cache:bfloat16]
"""
import json, os, sys
import torch
import torch_npu  # noqa: F401
import gtsrusops as ops

OP_NAME = "kv_rms_norm_rope_cache"
OP_CAMEL = "KvRmsNormRopeCache"
DEVICE = "npu:0"
SEED = 2026

# Positional order expected by ops.npu_kv_rms_norm_rope_cache(...), excluding the trailing
# output-template parameter `out_tmpls`. Inputs (op_def order) then attrs (op_def order):
#   kv, gamma, cos, sin, index, k_cache, ckv_cache,
#   k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v,      <- optional, pass None
#   epsilon, cache_mode, is_output_kv                            <- attrs
PARAM_NAMES = [
    "kv", "gamma", "cos", "sin", "index", "k_cache", "ckv_cache",
    "k_rope_scale", "c_kv_scale", "k_rope_offset", "c_kv_offset", "v",
    "epsilon", "cache_mode", "is_output_kv",
]

# Only the two in-place cache buffers are written in the Norm + non-quant V1 dialect.
OUTPUT_KEYS = ["k_cache", "ckv_cache"]

# V1 (no v) + CacheMode::Norm + non-quant cases targeting the ds_tiling path on 910B4.
# That path enforces FIXED the kernel dims: Dk = ROPE_LENGTH = 64 (cos/sin last axis),
# Dv = RMS_NORM_LENGTH[0] = 512 (gamma axis 0), so the kv tail dim must be 64+512 = 576.
# numHead stays 1 (safe Norm per-head cache row addressing); coverage varies B, S, cache_len,
# dtype and epsilon. Every case keeps S <= cache_len so index rows in [0, S) are in range.
CASES = [
    {"id": 0, "dtype": torch.float16,  "B": 1, "S": 1,  "Dk": 64, "Dv": 512, "cache_len": 16,   "epsilon": 1e-5},
    {"id": 1, "dtype": torch.bfloat16, "B": 1, "S": 8,  "Dk": 64, "Dv": 512, "cache_len": 32,   "epsilon": 1e-5},
    {"id": 2, "dtype": torch.float16,  "B": 2, "S": 4,  "Dk": 64, "Dv": 512, "cache_len": 64,   "epsilon": 1e-5},
    {"id": 3, "dtype": torch.bfloat16, "B": 4, "S": 1,  "Dk": 64, "Dv": 512, "cache_len": 64,   "epsilon": 1e-5},
    {"id": 4, "dtype": torch.float16,  "B": 1, "S": 16, "Dk": 64, "Dv": 512, "cache_len": 128,  "epsilon": 1e-6},
    {"id": 5, "dtype": torch.bfloat16, "B": 1, "S": 64, "Dk": 64, "Dv": 512, "cache_len": 256,  "epsilon": 1e-5},
    {"id": 6, "dtype": torch.float16,  "B": 8, "S": 1,  "Dk": 64, "Dv": 512, "cache_len": 32,   "epsilon": 1e-5},
    {"id": 7, "dtype": torch.bfloat16, "B": 2, "S": 32, "Dk": 64, "Dv": 512, "cache_len": 128,  "epsilon": 1e-4},
    {"id": 8, "dtype": torch.float16,  "B": 1, "S": 4,  "Dk": 64, "Dv": 512, "cache_len": 8,    "epsilon": 1e-5},
    {"id": 9, "dtype": torch.bfloat16, "B": 1, "S": 8,  "Dk": 64, "Dv": 512, "cache_len": 16,   "epsilon": 1e-5},
]

DATA_ROOT = "test_data"
INPUT_DIR = os.path.join(DATA_ROOT, "golden_input", OP_CAMEL)
RESULT_DIR = os.path.join(DATA_ROOT, "golden_result", OP_CAMEL)

# The generated wrapper takes a single trailing `out_tmpls` vector through which the two
# NON-in-place outputs (k_rope, c_kv) are allocated via empty_like from positions 2 and 3
# (full OpDef output order [k_cache, ckv_cache, k_rope, c_kv]). Positions 0/1 correspond to
# the in-place outputs and are never read, so they can be small dummies.
def make_out_tmpls(case):
    dtype = case["dtype"]
    B, S, Dk, Dv = case["B"], case["S"], case["Dk"], case["Dv"]
    dummy = torch.empty((1,), dtype=dtype, device=DEVICE)
    k_rope = torch.empty((B, 1, S, Dk), dtype=dtype, device=DEVICE)
    c_kv = torch.empty((B, 1, S, Dv), dtype=dtype, device=DEVICE)
    return [dummy, dummy, k_rope, c_kv]


def make_inputs(case):
    """Build every entry of PARAM_NAMES for this case."""
    torch.manual_seed(SEED + case["id"])
    dtype = case["dtype"]
    B, S, Dk, Dv, cache_len = case["B"], case["S"], case["Dk"], case["Dv"], case["cache_len"]
    kv_dim = Dv + Dk
    kv = torch.randn((B, 1, S, kv_dim), dtype=torch.float32, device=DEVICE).to(dtype)
    gamma = torch.randn((Dv,), dtype=torch.float32, device=DEVICE).to(dtype)
    # Per-token RoPE phase for the S axis. The kernel's interleave split uses x1=x[...,::2],
    # x2=x[...,1::2], so the phase space is Dk/2 wide and is tiled to a full Dk channel dim.
    freq = torch.pow(10000.0, -torch.arange(0, Dk, 2, dtype=torch.float32, device=DEVICE) / Dk)
    phase = torch.arange(S, dtype=torch.float32, device=DEVICE).unsqueeze(-1) * freq.unsqueeze(0)
    half_cos = torch.cos(phase).to(dtype)  # [S, Dk/2]
    half_sin = torch.sin(phase).to(dtype)
    cos_full = torch.cat([half_cos, half_cos], dim=-1)  # [S, Dk]
    sin_full = torch.cat([half_sin, half_sin], dim=-1)
    cos = cos_full.view(1, 1, S, Dk).expand(B, 1, S, Dk).contiguous()
    sin = sin_full.view(1, 1, S, Dk).expand(B, 1, S, Dk).contiguous()
    # Per-batch cache row offsets in [0, cache_len); deterministic, valid for Norm mode.
    index = torch.arange(S, dtype=torch.int64, device=DEVICE).view(1, S).expand(B, S).contiguous()
    # In-place cache buffers start from a deterministic zero pre-state; the kernel fills the
    # `index` rows it owns, the other rows stay 0 on both collect and test.
    k_cache = torch.zeros((B, 1, cache_len, Dk), dtype=dtype, device=DEVICE)
    ckv_cache = torch.zeros((B, 1, cache_len, Dv), dtype=dtype, device=DEVICE)
    return {
        "kv": kv,
        "gamma": gamma,
        "cos": cos,
        "sin": sin,
        "index": index,
        "k_cache": k_cache,
        "ckv_cache": ckv_cache,
        # optional inputs unused (non-quant, V1 no v) -> explicit None
        "k_rope_scale": None,
        "c_kv_scale": None,
        "k_rope_offset": None,
        "c_kv_offset": None,
        "v": None,
        # attrs
        "epsilon": case["epsilon"],
        "cache_mode": "Norm",
        "is_output_kv": False,
    }


def build_call_args(case):
    inputs = make_inputs(case)
    return [inputs[p].to(DEVICE) if isinstance(inputs[p], torch.Tensor) else inputs[p]
            for p in PARAM_NAMES], inputs


def collect_case(case):
    call_args, inputs = build_call_args(case)
    out_tmpls = make_out_tmpls(case)

    outputs = ops.npu_kv_rms_norm_rope_cache(*call_args, out_tmpls)
    torch.npu.synchronize()

    if outputs is None:
        # Should not happen (the wrapper returns a 4-tuple), but guard like the template.
        outputs = [inputs["k_cache"], inputs["ckv_cache"]]
    elif isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    else:
        outputs = list(outputs)
    if len(outputs) < len(OUTPUT_KEYS):
        raise RuntimeError(f"case_{case['id']}: got {len(outputs)} outputs, expected >= {OUTPUT_KEYS}")

    # Returned tuple order is OpDef output order: [k_cache, ckv_cache, k_rope, c_kv].
    # We snapshot the two in-place cache buffers that carry the real result.
    saved_out = {k: outputs[i].cpu() for i, k in enumerate(OUTPUT_KEYS)}
    for k, t in saved_out.items():
        if not torch.isfinite(t.float()).all():
            raise RuntimeError(f"case_{case['id']}: output '{k}' has NaN/Inf — bad golden data")

    saved_in = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    return saved_in, saved_out


def describe_call(call_args):
    """Compact argument summary. pybind dumps whole tensors on a TypeError, which for 10 cases
    buries the signature mismatch under megabytes of numbers."""
    parts = []
    for name, val in zip(PARAM_NAMES, call_args):
        if isinstance(val, torch.Tensor):
            parts.append(f"{name}=Tensor{tuple(val.shape)}:{str(val.dtype).replace('torch.', '')}")
        else:
            parts.append(f"{name}={val!r}")
    return ", ".join(parts)


def check_signature():
    """PARAM_NAMES vs the generated wrapper's real signature.

    The wrapper defaults every parameter to None, so a wrong argument count raises nothing —
    values shift by a slot and the output template silently lands in the wrong place. Catch it
    before the first case.
    """
    import inspect
    try:
        params = list(inspect.signature(ops.npu_kv_rms_norm_rope_cache).parameters)
    except (TypeError, ValueError):
        print("[WARN] cannot introspect the wrapper signature — skipping the arity check")
        return
    # The trailing output-template parameter (a vector of template buffers) is the analogue of
    # `out` in the generic template; exclude it from the expected input+attr list.
    expected = [p for p in params if p not in ("out", "out_tmpl", "out_tmpls")]
    if expected != PARAM_NAMES:
        print("Signature mismatch between this script and the generated wrapper:")
        print("  wrapper:      %s" % ", ".join(params))
        print("  PARAM_NAMES:  %s" % ", ".join(PARAM_NAMES))
        sys.exit(1)


def main():
    check_signature()
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    manifest = {"op_name": OP_NAME, "op_camel": OP_CAMEL, "output_keys": OUTPUT_KEYS, "cases": []}

    for case in CASES:
        name = f"case_{case['id']}"
        case["shape"] = (case["B"], 1, case["S"], case["Dv"] + case["Dk"])
        try:
            saved_in, saved_out = collect_case(case)
        except TypeError as exc:
            rejected = "incompatible function arguments" in str(exc)
            print(f"{name}: {'signature mismatch' if rejected else 'TypeError after the call'}")
            if rejected:
                print(f"  passed: {describe_call(build_call_args(case)[0])}")
                doc = ops.npu_kv_rms_norm_rope_cache.__doc__
                print(f"  binding expects: {doc.splitlines()[0] if doc else '(no docstring)'}")
            print(f"  {type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(1)
        except Exception as exc:
            print(f"{name}: COLLECT FAILED — {type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(1)
        in_path = os.path.join(INPUT_DIR, f"{name}.pt")
        out_path = os.path.join(RESULT_DIR, f"{name}.pt")
        torch.save(saved_in, in_path)
        torch.save(saved_out, out_path)
        manifest["cases"].append({
            "id": case["id"], "dtype": str(case["dtype"]).replace("torch.", ""),
            "shape": list(case["shape"]),
            "attrs": {k: v for k, v in case.items() if k not in ("id", "dtype", "shape", "B", "S", "Dk", "Dv", "cache_len")},
        })
        print(f"{name}: OK  input={os.path.getsize(in_path)}B  result={os.path.getsize(out_path)}B")

    with open(os.path.join(DATA_ROOT, "cases_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nCollected {len(manifest['cases'])}/{len(CASES)} cases into {DATA_ROOT}/")


if __name__ == "__main__":
    main()
