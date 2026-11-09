# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the AscendC ``hc_post`` kernel (HC/MoE post-combine).

The NPU operator (``torch.ops._C_ascend.npu_hc_post``) is validated against a
pure-PyTorch CPU golden oracle.

Backend: vllm_ascend
Category: csrc/moe (HC / MoE post-combine)
Source: runtime/ascend_c/vllm-ascend/csrc/moe/hc_post (pinned v0.23.0rc1)

The operator signature is (see csrc/torch_binding.cpp, npu_hc_post_npu)::

    npu_hc_post(x, residual, post, comb) -> out

    x         [b, s, d]          (bf16 / fp16 / fp32)
    residual  [b, s, hc, d]      (same dtype as x)
    post      [b, s, hc]         (bf16 / fp16 / fp32)
    comb      [b, s, hc, hc]     (same dtype as post)
    out       [b, s, hc, d]      (same shape as residual, same dtype as x)

Math (verified against the D-split kernel ``hc_post_d_split.h`` and the C310
reg-base kernel ``hc_post_bfloat16.h``)::

    out[b, s, m, k] = post[b, s, m] * x[b, s, k] + sum_i residual[b, s, i, k] * comb[b, s, i, m]

i.e. for each (b, s) element the per-head ``post`` scaling of ``x`` plus a
heads-pair combination of ``residual`` via ``comb``.

.. important::
    Supported shape space (verified on Ascend 910B / A2, D-split kernel):
    only **hc == 4** (matching the ``HC_PRE_HC_LIMIT`` constant) and
    ``d`` as a **multiple of 2048** (production ``d`` is 4096 / 7168) are
    supported on this platform. Other ``hc`` / small ``d`` values trigger a
    device-side vector-core fault (error 507035) or can hang the kernel, so
    this suite intentionally restricts inputs to hc=4 and d in {4096, 7168}.

Run inside the sandbox (after the operator is compiled)::

    python hc_post_test.py

or via pytest::

    pytest hc_post_test.py -v
"""

import gc

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42
HC = 4                      # only supported head count on this platform
D_VALUES = [4096, 7168]     # production head dims (multiples of 2048)
SEQ_VALUES = [1, 8, 64]     # small sequence lengths (b=1)


# ---------------------------------------------------------------------------
# Golden reference (CPU, pure PyTorch)
# ---------------------------------------------------------------------------
def _golden_hc_post(x, residual, post, comb):
    """CPU golden reference: out[m, k] = post[m]*x[k] + sum_i residual[i,k]*comb[i,m]."""
    compute = torch.float32
    xf = x.to(compute)            # [b, s, d]
    rf = residual.to(compute)     # [b, s, hc, d]
    pf = post.to(compute)         # [b, s, hc]
    cf = comb.to(compute)         # [b, s, hc, hc]

    term1 = pf.unsqueeze(-1) * xf.unsqueeze(-2)          # [b,s,hc,1] * [b,s,1,d]
    term2 = torch.einsum('bsik,bsim->bsmk', rf, cf)      # [b,s,m,k]
    out = term1 + term2
    return out.to(x.dtype)


def _make_inputs(b, s, hc, d, dtype=torch.bfloat16, seed=SEED):
    torch.manual_seed(seed)
    x = torch.randn(b, s, d, dtype=dtype)
    residual = torch.randn(b, s, hc, d, dtype=dtype)
    post = torch.randn(b, s, hc, dtype=dtype)
    comb = torch.randn(b, s, hc, hc, dtype=dtype)
    return x, residual, post, comb


def _npu_op_exec(x, residual, post, comb):
    """Run the AscendC operator on NPU and return a CPU tensor."""
    for t in (x, residual, post, comb):
        if not t.is_contiguous():
            t = t.contiguous()
    out = torch.ops._C_ascend.npu_hc_post(x.npu(), residual.npu(),
                                          post.npu(), comb.npu())
    return out.cpu()


def _assert_close(actual, ref, rtol=2e-2, atol=2e-2):
    torch.testing.assert_close(
        actual.to(torch.float32), ref.to(torch.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Tests (pytest-oracle)
# ---------------------------------------------------------------------------
def _run_and_check(b, s, d, dtype=torch.bfloat16):
    x, residual, post, comb = _make_inputs(b, s, HC, d, dtype)
    ref = _golden_hc_post(x, residual, post, comb)
    npu = _npu_op_exec(x, residual, post, comb)
    _assert_close(npu, ref)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_hc_post_single_token():
    """Exact production single-token shape: b=1, s=1, hc=4, d=4096."""
    _run_and_check(1, 1, 4096)


def test_hc_post_sequence():
    """Vary sequence length at d=4096 (b=1)."""
    for s in SEQ_VALUES:
        _run_and_check(1, s, 4096)


def test_hc_post_d_7168():
    """Vary sequence length at the extended head dim d=7168 (b=1)."""
    for s in [1, 8, 16]:
        _run_and_check(1, s, 7168)


def test_hc_post_mid_sequence_d4096():
    _run_and_check(1, 37, 4096)


def test_hc_post_d7168_s64():
    _run_and_check(1, 64, 7168)


def test_hc_post_output_shape_and_dtype():
    b, s, d = 1, 8, 4096
    x, residual, post, comb = _make_inputs(b, s, HC, d)
    npu = _npu_op_exec(x, residual, post, comb)
    assert npu.shape == (b, s, HC, d), f"unexpected out shape: {npu.shape}"
    assert npu.dtype == torch.bfloat16, f"unexpected out dtype: {npu.dtype}"
    ref = _golden_hc_post(x, residual, post, comb)
    _assert_close(npu, ref)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Standalone runner: `python hc_post_test.py`
# The sandbox build runner invokes the test with plain `python`, so we call each
# test function imperatively here and raise on the first failure.
# ---------------------------------------------------------------------------
def _main():
    tests = [
        ("test_hc_post_single_token", test_hc_post_single_token),
        ("test_hc_post_sequence", test_hc_post_sequence),
        ("test_hc_post_d_7168", test_hc_post_d_7168),
        ("test_hc_post_mid_sequence_d4096", test_hc_post_mid_sequence_d4096),
        ("test_hc_post_d7168_s64", test_hc_post_d7168_s64),
        ("test_hc_post_output_shape_and_dtype", test_hc_post_output_shape_and_dtype),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    if failures:
        raise SystemExit(1)




def _run_suite(runs):
    """重复整套测试 runs 次（warmup + 测量）；单次 FAIL 不中断后续轮次。"""
    all_ok = True
    for run_idx in range(1, runs + 1):
        print(f"\n{'=' * 98}")
        print(f"Run {run_idx}/{runs}")
        print(f"{'=' * 98}")
        try:
            _main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                all_ok = False
    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="repeated suite runs for warmup + measurement")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of repeated runs (each run iterates all cases)")
    args = parser.parse_args()
    raise SystemExit(0 if _run_suite(args.runs) else 1)
