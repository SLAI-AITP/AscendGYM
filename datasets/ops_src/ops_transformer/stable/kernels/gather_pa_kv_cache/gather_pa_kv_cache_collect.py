#!/usr/bin/env python3
"""
Collect golden data for the gather_pa_kv_cache NPU operator (multi-case).

Each case produces:
  kernels/gather_pa_kv_cache/golden_input_case_{i}/key_cache.pt ... (CPU tensors)
  kernels/gather_pa_kv_cache/golden_input_case_{i}/input.pt        (all inputs + metadata)
  kernels/gather_pa_kv_cache/golden_output_case_{i}/key_ref.pt, value_ref.pt
  kernels/gather_pa_kv_cache/golden_params_case_{i}.json

Golden strategy
---------------
1. Run the NPU operator and save its output as the *primary* golden (the sandbox
   workflow treats the NPU operator as ground truth).
2. Independently compute a CPU-reference golden (independent ground truth) and
   cross-check the NPU result against it. If they disagree, this is reported and
   the test fails loudly -- no silent garbage golden.
3. If the NPU operator cannot launch (kernel-level failure, e.g. "executor is
   nullptr"), save the CPU-reference golden so the test data stays valid, but
   clearly flag the NPU failure (KERNEL_ERROR) instead of silently saving garbage.

Run in sandbox (via datamining/gather_pa_kv_cache_collect.sh) or locally.
"""

import os
import sys
import json
import argparse

import torch
import torch_npu  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(__file__).startswith("tmp"):
    _kernel_dir = os.path.join(SCRIPT_DIR, "gather_pa_kv_cache")
else:
    _kernel_dir = SCRIPT_DIR
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from gather_pa_kv_cache_golden import gather_pa_kv_cache_cpu, build_case_params  # noqa: E402


def resolve_dirs():
    """Return (OUTPUT_DIR.tmpl, INPUT_DIR.tmpl, KERNEL_DIR) depending on mode."""
    if os.path.basename(__file__).startswith("tmp"):
        # Sandbox: __file__ renamed to tmpXXX.py, SCRIPT_DIR == CWD (project root).
        kernel_dir = os.path.join(SCRIPT_DIR, "gather_pa_kv_cache")
        out_tmpl = os.path.join(kernel_dir, "golden_output_case_{}")
        in_tmpl = os.path.join(kernel_dir, "golden_input_case_{}")
    else:
        # Local: __file__ == .../kernels/gather_pa_kv_cache/gather_pa_kv_cache_collect.py
        kernel_dir = SCRIPT_DIR
        out_tmpl = os.path.join(SCRIPT_DIR, "golden_output_case_{}")
        in_tmpl = os.path.join(SCRIPT_DIR, "golden_input_case_{}")
    return out_tmpl, in_tmpl, kernel_dir


def _make_cache_tensor(shape, dtype, device):
    if dtype in (torch.int8, torch.int32, torch.int64):
        # integer caches: use a wide random range so token values are distinct
        return torch.randint(-50, 50, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def make_inputs(params, device="npu"):
    """Generate NPU input tensors for a case from its params."""
    torch.manual_seed(params["seed"])
    idx_dtype = params.get("index_dtype", torch.int32)  # int32 or int64

    key_cache = _make_cache_tensor((params["num_blocks"], params["block_size"], params["num_heads"],
                                    params["head_dim_k"]), params["dtype"], device)
    value_cache = _make_cache_tensor((params["num_blocks"], params["block_size"], params["num_heads"],
                                      params["head_dim_v"]), params["dtype"], device)
    block_tables = torch.randint(0, params["num_blocks"],
                                 (params["batch"], params["max_blocks"]),
                                 dtype=idx_dtype, device=device)

    lens = torch.tensor(params["lens"], dtype=idx_dtype, device=device)
    if params["is_seq_lens_cumsum"]:
        # Kernel convention (gather_pa_kv_cache_nd.h): is_seq_lens_cumsum=True expects a
        # PREFIX-SUM seq_lens of length batch+1: [S0, S0+l0, S0+l0+l1, ..., total].
        # cumsum() also upcasts int32->int64 on NPU, so force back to idx_dtype.
        prefix = torch.cat([torch.zeros(1, dtype=idx_dtype, device=device),
                            torch.cumsum(lens, dim=0)]).to(idx_dtype)
        seq_lens = prefix
        total_tokens = int(seq_lens[-1].item())
    else:
        seq_lens = lens
        total_tokens = params["total_tokens"]

    key_ref = torch.empty((total_tokens, params["num_heads"], params["head_dim_k"]),
                          dtype=params["dtype"], device=device)
    value_ref = torch.empty((total_tokens, params["num_heads"], params["head_dim_v"]),
                            dtype=params["dtype"], device=device)

    seq_offset = None
    if params.get("has_seq_offset"):
        # block-table offsets per batch (in blocks); keep same index dtype as blockTables/seqLens
        seq_offset = torch.arange(params["batch"], dtype=idx_dtype, device=device)

    return {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "key_ref": key_ref,
        "value_ref": value_ref,
        "seq_offset": seq_offset,
    }, total_tokens


def run_npu_op(inputs, params, out):
    """Call the custom-built NPU operator. out must be a dict to mutate ('key','value')."""
    out["key"] = inputs["key_ref"]
    out["value"] = inputs["value_ref"]
    kwargs = dict(
        key_cache=inputs["key_cache"],
        value_cache=inputs["value_cache"],
        block_tables=inputs["block_tables"],
        seq_lens=inputs["seq_lens"],
        key=inputs["key_ref"],
        value=inputs["value_ref"],
        cache_mode=params.get("cache_mode", "Norm"),
        is_seq_lens_cumsum=params["is_seq_lens_cumsum"],
        out=[inputs["key_ref"], inputs["value_ref"]],
    )
    if inputs["seq_offset"] is not None:
        kwargs["seq_offset"] = inputs["seq_offset"]
    ops = __import__("gtsrusops", fromlist=["npu_gather_pa_kv_cache"])
    ops.npu_gather_pa_kv_cache(**kwargs)
    torch.npu.synchronize()


def collect_case(case_id, max_cases):
    out_tmpl, in_tmpl, kernel_dir = resolve_dirs()
    params = build_case_params(case_id)
    params["case_id"] = case_id

    print(f"\n=== Collecting golden for case {case_id} ===")
    print(f"  dtype={str(params['dtype'])}, cache_mode={params.get('cache_mode','Norm')}, "
          f"cumsum={params['is_seq_lens_cumsum']}")

    # ---- NPU inputs (best-effort; the op may be broken and OOM/throw) ----
    npu_failed = False
    npu_error = ""
    out = {}
    inputs_npu = None
    try:
        inputs_npu, total_tokens = make_inputs(params, device="npu")
        run_npu_op(inputs_npu, params, out)
    except Exception as e:
        npu_failed = True
        npu_error = str(e).splitlines()[0] if str(e) else repr(e)
        print(f"  [WARN] NPU op failed: {npu_error}")

    # ---- CPU inputs + independent CPU-reference golden (always safe) ----
    inputs_cpu, total_tokens = make_inputs(params, device="cpu")
    cpu_kr = torch.empty((total_tokens, params["num_heads"], params["head_dim_k"]),
                         dtype=params["dtype"])
    cpu_vr = torch.empty((total_tokens, params["num_heads"], params["head_dim_v"]),
                         dtype=params["dtype"])
    so = inputs_cpu["seq_offset"] if inputs_cpu["seq_offset"] is not None else None
    gather_pa_kv_cache_cpu(inputs_cpu["key_cache"], inputs_cpu["value_cache"],
                           inputs_cpu["block_tables"], inputs_cpu["seq_lens"],
                           cpu_kr, cpu_vr, seq_offset=so,
                           cache_mode=params.get("cache_mode", "Norm"),
                           is_seq_lens_cumsum=params["is_seq_lens_cumsum"])

    # ---- Save directories ----
    out_dir = out_tmpl.format(case_id)
    in_dir = in_tmpl.format(case_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(in_dir, exist_ok=True)

    # Save all input tensors (CPU copies from the same seed)
    for name in ["key_cache", "value_cache", "block_tables", "seq_lens", "key_ref", "value_ref"]:
        torch.save(inputs_cpu[name], os.path.join(in_dir, f"{name}.pt"))
    if inputs_cpu["seq_offset"] is not None:
        torch.save(inputs_cpu["seq_offset"], os.path.join(in_dir, "seq_offset.pt"))

    # ---- Choose golden output ----
    if not npu_failed and inputs_npu is not None:
        try:
            npu_key = out["key"].cpu()
            npu_value = out["value"].cpu()
            if npu_key.numel() == 0 or npu_value.numel() == 0:
                raise RuntimeError("NPU output empty (kernel produced no data)")
            if npu_key.shape != cpu_kr.shape or npu_value.shape != cpu_vr.shape:
                raise RuntimeError(f"NPU output shape mismatch: {tuple(npu_key.shape)} vs {tuple(cpu_kr.shape)}")
            dk = (npu_key.float() - cpu_kr.float()).abs().max().item()
            dv = (npu_value.float() - cpu_vr.float()).abs().max().item()
            print(f"  NPU vs CPU-ref max abs diff: key={dk:.3e}, value={dv:.3e}")
            if dk > 1e-3 or dv > 1e-3:
                raise RuntimeError("NPU-vs-CPU mismatch (suspect wrong gather)")
            golden_key, golden_value, source = npu_key, npu_value, "npu"
        except Exception as e:
            npu_failed = True
            npu_error = str(e).splitlines()[0] if str(e) else repr(e)
            golden_key, golden_value, source = cpu_kr, cpu_vr, "cpu-reference"
    else:
        golden_key, golden_value = cpu_kr, cpu_vr
        source = "cpu-reference"

    torch.save(golden_key, os.path.join(out_dir, "key_ref.pt"))
    torch.save(golden_value, os.path.join(out_dir, "value_ref.pt"))

    # Save params/metadata
    params_json = dict(params)
    params_json["dtype"] = str(params["dtype"])
    params_json["index_dtype"] = str(params.get("index_dtype", torch.int32))
    params_json["total_tokens"] = total_tokens
    params_json["golden_source"] = source
    params_json["npu_failed"] = npu_failed
    params_json["npu_error"] = npu_error
    with open(os.path.join(kernel_dir, f"golden_params_case_{case_id}.json"), "w") as f:
        json.dump(params_json, f, indent=2)
    with open(os.path.join(in_dir, "metadata.json"), "w") as f:
        json.dump(params_json, f, indent=2)

    print(f"  Golden saved to {out_dir} (source={source})")
    return npu_failed, npu_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=10, help="How many cases to collect")
    ap.add_argument("--case", type=int, default=-1, help="Collect a single case if >=0")
    args = ap.parse_args()

    case_ids = [args.case] if args.case >= 0 else list(range(args.max_cases))
    any_fail = False
    errors = []
    for cid in case_ids:
        try:
            failed, err = collect_case(cid, args.max_cases)
            if failed:
                any_fail = True
                errors.append((cid, err))
        except Exception as e:
            any_fail = True
            errors.append((cid, f"{type(e).__name__}: {e}"))
            print(f"  [ERROR] collecting case {cid}: {e}")

    print("\n" + "=" * 50)
    if any_fail:
        print("COLLECT DONE (with NPU op issues):")
        for cid, err in errors:
            print(f"  case {cid}: {err}")
        print("Note: golden data saved (CPU-reference), but NPU operator is NOT functioning.")
        # Exit non-zero so the harness knows the operator itself is broken.
        print("KERNEL_ERROR - check operator implementation")
        sys.exit(2)
    print("COLLECT OK - all cases gathered golden from NPU")
    sys.exit(0)


if __name__ == "__main__":
    main()
