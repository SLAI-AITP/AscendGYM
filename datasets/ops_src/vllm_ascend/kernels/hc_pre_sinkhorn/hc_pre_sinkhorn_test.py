# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the AscendC ``hc_pre_sinkhorn`` kernel (HC / MoE pre + Sinkhorn).

The NPU operator (``torch.ops._C_ascend.npu_hc_pre_sinkhorn``) is validated against a
pure-PyTorch CPU golden oracle.

Backend: vllm_ascend
Category: csrc/moe (HC pre-combine with Sinkhorn iteration)
Source: runtime/ascend_c/vllm-ascend/csrc/moe/hc_pre_sinkhorn

The operator signature is (see csrc/torch_binding.cpp, npu_hc_pre_sinkhorn_npu)::

    npu_hc_pre_sinkhorn(mixes, rsqrt, hc_scale, hc_base, x,
                        hc_mult, hc_sinkhorn_iters, hc_eps) -> (y, post, comb_frag)

    mixes     [bs, hc_mix]      fp32   (hc_mix == (2 + hc_mult) * hc_mult)
    rsqrt     [bs]              fp32   (per-token inverse RMS)
    hc_scale  [3]               fp32   (scales for pre / post / res projections)
    hc_base   [hc_mix]          fp32   (bases for pre / post / res projections)
    x         [bs, hc_mult, d]  bf16
    y         [bs, d]           bf16
    post      [bs, hc_mult]     fp32
    comb_frag [bs, hc_mult, hc_mult] fp32

Math (derived from the kernel ``ProcessPre`` / ``ProcessPost`` / Sinkhorn loop in
``hc_pre_sinkhorn_perf.h``), where ``m0/m1/m2`` are the three ``hc_mult``-wide segments
of ``mixes`` (pre / post / res) and ``b0/b1/b2`` the corresponding ``hc_base`` segments::

    pre[b, m]   = sigmoid( mixes[b, m] * rsqrt[b] * hc_scale[0] + hc_base[b, m] ) + hc_eps
    y[b, k]     = sum_m  pre[b, m] * x[b, m, k]                     (y cast to bf16)

    post[b, m]  = 2 * sigmoid( mixes[b, hc_mult+m] * rsqrt[b] * hc_scale[1]
                               + hc_base[b, hc_mult+m] )

    comb[b, i, j] = softmax( m2[b,i,j] * rsqrt[b] * hc_scale[2] + b2[b,i,j], dim=-1 ) + hc_eps
    comb[b] /= comb[b].sum(dim=0, keepdim=True) + hc_eps           # 1 column normalize
    for _ in range(hc_sinkhorn_iters - 1):                         # Sinkhorn iteration
        comb[b] /= comb[b].sum(dim=-1, keepdim=True) + hc_eps      # row normalize
        comb[b] /= comb[b].sum(dim=0, keepdim=True) + hc_eps       # column normalize

.. important::
    Only ``x`` is bf16 (the op requires DT_BF16); all projection tensors are fp32.
    Production configuration (DeepSeek-V4 / HC): ``hc_mult = 4`` (so
    ``hc_mix = 24``), ``hc_sinkhorn_iters = 20``, ``hc_eps = 1e-6``.

Run inside the sandbox (after the operator is compiled)::

    python hc_pre_sinkhorn_test.py

or via pytest::

    pytest hc_pre_sinkhorn_test.py -v
"""

import gc

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42
HC_MULT = 4                 # production HC multiplier
HC_MIX = (2 + HC_MULT) * HC_MULT   # 24
HC_SINKHORN_ITERS = 20      # production iteration count
HC_EPS = 1e-6               # production epsilon
D_VALUES = [4096, 7168]     # production head dims (HIDDEN_SIZE / EXTENDED_HIDDEN_SIZE)
BS_VALUES = [1, 2, 8]       # small token/batch counts (b*s)


# ---------------------------------------------------------------------------
# Golden reference (CPU, pure PyTorch) — mirrors hc_pre_sinkhorn_perf.h
# ---------------------------------------------------------------------------
def _golden_hc_pre_sinkhorn(mixes, rsqrt, hc_scale, hc_base, x,
                            hc_mult=HC_MULT, iters=HC_SINKHORN_ITERS, eps=HC_EPS):
    """CPU golden reference for the HC-pre + Sinkhorn operator."""
    bs, hcm, d = x.shape
    m0 = mixes[:, 0:hc_mult]                 # [bs, hc_mult]  pre  mix
    m1 = mixes[:, hc_mult:2 * hc_mult]       # [bs, hc_mult]  post mix
    m2 = mixes[:, 2 * hc_mult:2 * hc_mult + hc_mult * hc_mult].view(bs, hc_mult, hc_mult)  # res mix
    b0 = hc_base[0:hc_mult]
    b1 = hc_base[hc_mult:2 * hc_mult]
    b2 = hc_base[2 * hc_mult:2 * hc_mult + hc_mult * hc_mult].view(hc_mult, hc_mult)
    r = rsqrt.unsqueeze(-1)                  # [bs, 1] broadcast over cols

    # y: [bs, d], bf16
    pre = torch.sigmoid(m0 * r * hc_scale[0] + b0) + eps          # [bs, hc_mult]
    y = torch.einsum('bm,bmk->bk', pre, x.float()).to(torch.bfloat16)

    # post: [bs, hc_mult]
    post = 2.0 * torch.sigmoid(m1 * r * hc_scale[1] + b1)

    # comb_frag: [bs, hc_mult, hc_mult] via Sinkhorn
    comb = m2 * r.unsqueeze(-1) * hc_scale[2] + b2.unsqueeze(0)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)           # 1 column normalize
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)      # row normalize
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)       # column normalize
    return y, post, comb


def _make_inputs(bs, d, seed=SEED):
    torch.manual_seed(seed)
    mixes = torch.randn(bs, HC_MIX, dtype=torch.float32)
    rsqrt = torch.rand(bs, dtype=torch.float32) + 0.5            # positive inverse-RMS
    hc_scale = torch.rand(3, dtype=torch.float32) * 0.1 + 0.05
    hc_base = torch.randn(HC_MIX, dtype=torch.float32) * 0.1
    x = torch.randn(bs, HC_MULT, d, dtype=torch.bfloat16)
    return mixes, rsqrt, hc_scale, hc_base, x


def _npu_op_exec(mixes, rsqrt, hc_scale, hc_base, x):
    """Run the AscendC operator on NPU and return CPU tensors."""
    outs = torch.ops._C_ascend.npu_hc_pre_sinkhorn(
        mixes.npu(), rsqrt.npu(), hc_scale.npu(), hc_base.npu(), x.npu(),
        HC_MULT, HC_SINKHORN_ITERS, HC_EPS,
    )
    return tuple(t.cpu() for t in outs)


def _assert_close(actual, ref, rtol=2e-2, atol=2e-2):
    torch.testing.assert_close(
        actual.to(torch.float32), ref.to(torch.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Tests (pytest-oracle)
# ---------------------------------------------------------------------------
def _run_and_check(bs, d):
    mixes, rsqrt, hc_scale, hc_base, x = _make_inputs(bs, d)
    ref_y, ref_post, ref_comb = _golden_hc_pre_sinkhorn(
        mixes, rsqrt, hc_scale, hc_base, x, HC_MULT, HC_SINKHORN_ITERS, HC_EPS)
    y, post, comb = _npu_op_exec(mixes, rsqrt, hc_scale, hc_base, x)
    _assert_close(y, ref_y)
    _assert_close(post, ref_post)
    _assert_close(comb, ref_comb, rtol=3e-2, atol=3e-2)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_hc_pre_sinkhorn_single_token():
    """Production single-token: bs=1, d=4096."""
    _run_and_check(1, 4096)


def test_hc_pre_sinkhorn_d_7168():
    """Production extended dim: bs=1, d=7168."""
    _run_and_check(1, 7168)


def test_hc_pre_sinkhorn_bs_variation():
    """Small batch variation at d=4096."""
    for bs in BS_VALUES:
        _run_and_check(bs, 4096)


def test_hc_pre_sinkhorn_bs_d_combos():
    _run_and_check(2, 4096)
    _run_and_check(8, 7168)


def test_hc_pre_sinkhorn_output_shapes_and_dtypes():
    bs, d = 2, 4096
    mixes, rsqrt, hc_scale, hc_base, x = _make_inputs(bs, d)
    y, post, comb = _npu_op_exec(mixes, rsqrt, hc_scale, hc_base, x)
    assert y.shape == (bs, d) and y.dtype == torch.bfloat16, f"unexpected y: {y.shape} {y.dtype}"
    assert post.shape == (bs, HC_MULT) and post.dtype == torch.float32, \
        f"unexpected post: {post.shape} {post.dtype}"
    assert comb.shape == (bs, HC_MULT, HC_MULT) and comb.dtype == torch.float32, \
        f"unexpected comb_frag: {comb.shape} {comb.dtype}"
    ref_y, ref_post, ref_comb = _golden_hc_pre_sinkhorn(
        mixes, rsqrt, hc_scale, hc_base, x, HC_MULT, HC_SINKHORN_ITERS, HC_EPS)
    _assert_close(y, ref_y)
    _assert_close(post, ref_post)
    _assert_close(comb, ref_comb, rtol=3e-2, atol=3e-2)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Standalone runner: `python hc_pre_sinkhorn_test.py`
# ---------------------------------------------------------------------------
def _main():
    tests = [
        ("test_hc_pre_sinkhorn_single_token", test_hc_pre_sinkhorn_single_token),
        ("test_hc_pre_sinkhorn_d_7168", test_hc_pre_sinkhorn_d_7168),
        ("test_hc_pre_sinkhorn_bs_variation", test_hc_pre_sinkhorn_bs_variation),
        ("test_hc_pre_sinkhorn_bs_d_combos", test_hc_pre_sinkhorn_bs_d_combos),
        ("test_hc_pre_sinkhorn_output_shapes_and_dtypes", test_hc_pre_sinkhorn_output_shapes_and_dtypes),
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
