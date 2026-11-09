# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the AscendC ``fused_gdn_gating`` kernel (Qwen3-Next gating).

The NPU operator (``torch.ops._C_ascend.npu_fused_gdn_gating``) is validated
against a pure-PyTorch CPU golden oracle across num_heads / batch / dtype /
param-dtype combinations, including the softplus-threshold (``beta*x<=threshold``)
linear / softplus branch coverage.

Backend: vllm_ascend
Category: csrc/attention (Linear attn / SSM)
Source: runtime/ascend_c/vllm-ascend (pinned v0.23.0rc1)

The operator signature is (see fused_gdn_gating_torch_adpt.h)::

    npu_fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0) -> (g, beta_output)

    A_log      [num_heads]                  (float32 / bfloat16 / float16)
    dt_bias    [num_heads]                  (same dtype as A_log)
    a          [batch, num_heads]           (bfloat16 / float16)
    b          [batch, num_heads]           (same dtype as a)
    g          [1, batch, num_heads]  fp32
    beta_output[1, batch, num_heads]        (same dtype as a)

Run inside the sandbox (after the operator is compiled via ``setup.py build_ext``)::

    python fused_gdn_gating_test.py

or via pytest::

    pytest fused_gdn_gating_test.py -v
"""

import gc

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42
NUM_HEADS_VALUES = [4, 6, 8, 12, 16, 32, 64, 128]
BATCH_SIZES = [1, 7, 37, 128, 512, 4096]
DTYPES = [torch.bfloat16, torch.float16]
PARAM_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
DTYPE_COMBINATIONS = [(dtype, pd) for dtype in DTYPES for pd in PARAM_DTYPES]


# ---------------------------------------------------------------------------
# Golden reference (CPU, pure PyTorch)
# ---------------------------------------------------------------------------
def _golden_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
):
    """CPU golden reference.

    Math (matches the Triton/AscendC kernel):
        x        = a + dt_bias
        softplus = where(beta*x <= threshold, log1p(exp(beta*x)) / beta, x)
        g        = -exp(A_log) * softplus          -> [1, batch, num_heads] fp32
        beta_out = sigmoid(b)                       -> [1, batch, num_heads] (b dtype)
    """
    batch, num_heads = a.shape
    compute = torch.float32

    A_log_e = A_log.to(compute).unsqueeze(0).expand(batch, -1)
    dt_bias_e = dt_bias.to(compute).unsqueeze(0).expand(batch, -1)
    a_f = a.to(compute)
    b_f = b.to(compute)

    x = a_f + dt_bias_e
    beta_x = beta * x
    softplus_o = torch.where(
        beta_x <= threshold,
        torch.log1p(torch.exp(beta_x)) / beta,
        x,
    )

    g = (-torch.exp(A_log_e) * softplus_o).unsqueeze(0)
    beta_output = torch.sigmoid(b_f).to(b.dtype).unsqueeze(0)
    return g, beta_output


def _make_inputs(num_heads, batch, dtype, param_dtype=torch.float32, seed=SEED):
    torch.manual_seed(seed)
    A_log = torch.randn(num_heads, dtype=param_dtype)
    dt_bias = torch.randn(num_heads, dtype=param_dtype)
    a = torch.randn(batch, num_heads, dtype=dtype)
    b = torch.randn(batch, num_heads, dtype=dtype)
    return A_log, a, b, dt_bias


def _force_softplus_threshold_cases(a, dt_bias, beta, threshold):
    """Force beta * (a + dt_bias) to cover threshold and non-threshold paths."""
    if a.shape[0] < 4 or a.shape[1] < 4:
        return
    dt_bias[:4] = 0
    boundary = threshold / beta
    a[0, 0] = boundary + 2.0  # linear branch
    a[1, 1] = boundary  # softplus branch at equality
    a[2, 2] = boundary - 0.5  # softplus branch below threshold
    a[3, 3] = -boundary - 2.0  # negative softplus input


def _npu_op_exec(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """Run the AscendC operator on NPU and return CPU tensors."""
    if not A_log.is_contiguous():
        A_log = A_log.contiguous()
    if not dt_bias.is_contiguous():
        dt_bias = dt_bias.contiguous()

    g, beta_output = torch.ops._C_ascend.npu_fused_gdn_gating(
        A_log.npu(),
        a.npu(),
        b.npu(),
        dt_bias.npu(),
        float(beta),
        float(threshold),
    )
    return g.cpu(), beta_output.cpu()


def _assert_close(actual_g, actual_beta, ref_g, ref_beta, rtol=3e-3, atol=1e-2):
    torch.testing.assert_close(
        actual_g.to(torch.float32), ref_g.to(torch.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    )
    torch.testing.assert_close(
        actual_beta.to(torch.float32), ref_beta.to(torch.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Tests (pytest-oracle)
# ---------------------------------------------------------------------------
def test_fused_gdn_gating_vs_reference(num_heads=32, batch=37, dtype=torch.bfloat16):
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, dtype)
    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)
    _assert_close(npu_g, npu_beta, ref_g, ref_beta)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_non_default_params(num_heads=16, batch=37):
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, torch.bfloat16)
    _force_softplus_threshold_cases(a, dt_bias, beta=0.5, threshold=1.0)
    ref_g, ref_beta = _golden_fused_gdn_gating(
        A_log, a, b, dt_bias, beta=0.5, threshold=1.0)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias, beta=0.5, threshold=1.0)
    _assert_close(npu_g, npu_beta, ref_g, ref_beta)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_dtype_matrix(num_heads=32, batch=37):
    for dtype, param_dtype in DTYPE_COMBINATIONS:
        A_log, a, b, dt_bias = _make_inputs(
            num_heads, batch, dtype, param_dtype=param_dtype)
        _force_softplus_threshold_cases(a, dt_bias, beta=1.0, threshold=2.0)
        ref_g, ref_beta = _golden_fused_gdn_gating(
            A_log, a, b, dt_bias, threshold=2.0)
        npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias, threshold=2.0)
        _assert_close(npu_g, npu_beta, ref_g, ref_beta)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_output_shapes():
    A_log, a, b, dt_bias = _make_inputs(32, 17, torch.bfloat16)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)
    assert npu_g.shape == (1, 17, 32), f"unexpected g shape: {npu_g.shape}"
    assert npu_g.dtype == torch.float32, f"unexpected g dtype: {npu_g.dtype}"
    assert npu_beta.shape == (1, 17, 32), f"unexpected beta shape: {npu_beta.shape}"
    assert npu_beta.dtype == torch.bfloat16, f"unexpected beta dtype: {npu_beta.dtype}"
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_each_num_heads():
    for num_heads in NUM_HEADS_VALUES:
        A_log, a, b, dt_bias = _make_inputs(num_heads, 37, torch.bfloat16)
        ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
        npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)
        _assert_close(npu_g, npu_beta, ref_g, ref_beta)
        gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_each_batch():
    for batch in BATCH_SIZES:
        A_log, a, b, dt_bias = _make_inputs(32, batch, torch.bfloat16)
        ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
        npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)
        _assert_close(npu_g, npu_beta, ref_g, ref_beta)
        gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_fp16():
    A_log, a, b, dt_bias = _make_inputs(32, 256, torch.float16)
    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)
    _assert_close(npu_g, npu_beta, ref_g, ref_beta)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_large_batch():
    A_log, a, b, dt_bias = _make_inputs(32, 16384, torch.bfloat16)
    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)
    _assert_close(npu_g, npu_beta, ref_g, ref_beta)
    gc.collect(); torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Standalone runner: `python fused_gdn_gating_test.py`
# The sandbox build runner invokes the test with plain `python`, so we call each
# test function imperatively here and raise on the first failure.
# ---------------------------------------------------------------------------
def _main():
    tests = [
        ("test_fused_gdn_gating_vs_reference", test_fused_gdn_gating_vs_reference),
        ("test_fused_gdn_gating_non_default_params", test_fused_gdn_gating_non_default_params),
        ("test_fused_gdn_gating_dtype_matrix", test_fused_gdn_gating_dtype_matrix),
        ("test_fused_gdn_gating_output_shapes", test_fused_gdn_gating_output_shapes),
        ("test_fused_gdn_gating_each_num_heads", test_fused_gdn_gating_each_num_heads),
        ("test_fused_gdn_gating_each_batch", test_fused_gdn_gating_each_batch),
        ("test_fused_gdn_gating_fp16", test_fused_gdn_gating_fp16),
        ("test_fused_gdn_gating_large_batch", test_fused_gdn_gating_large_batch),
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
