#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
CPU validation test for AddRmsNorm operator.
Loads golden input/result, runs PyTorch reference, compares.
"""
import json
import os
import sys
import time

import torch


# ── Configuration ──────────────────────────────────────────────
BASE_DIR = os.environ.get("ADD_RMS_NORM_TEST_BASE_DIR", None)
RTOL = 1e-3
ATOL = 1e-3
DEFAULT_BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "test_data",
)


# ── Reference Implementation ───────────────────────────────────
def add_rms_norm_reference(x1, x2, gamma, epsilon=1e-6):
    """PyTorch reference implementation of AddRmsNorm."""
    x = x1 + x2
    # Compute RMS over last gamma.dim() dimensions
    rms = torch.sqrt(x.pow(2).mean(dim=-gamma.dim(), keepdim=True) + epsilon)
    rstd = 1.0 / rms
    y = gamma * (x * rstd)
    return y, rstd.to(torch.float32), x


# ── Validation Logic ───────────────────────────────────────────
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_case(case_idx, template, golden_input_path, golden_result_path):
    """Validate a single test case."""
    golden_input = load_pt(golden_input_path)
    golden_result = load_pt(golden_result_path)

    x1 = golden_input["x1"]
    x2 = golden_input["x2"]
    gamma = golden_input["gamma"]
    eps = golden_input["epsilon"]

    y_golden = golden_result["y"]
    rstd_golden = golden_result["rstd"]
    x_golden = golden_result["x"]

    # Run reference
    y_pred, rstd_pred, x_pred = add_rms_norm_reference(x1, x2, gamma, eps)

    # Validate all three outputs
    passed_y = torch.allclose(y_golden, y_pred, rtol=RTOL, atol=ATOL)
    passed_rstd = torch.allclose(rstd_golden, rstd_pred, rtol=RTOL, atol=ATOL)
    passed_x = torch.allclose(x_golden, x_pred, rtol=RTOL, atol=ATOL)

    passed = passed_y and passed_rstd and passed_x
    return passed, {
        "input_shape": list(x1.shape),
        "gamma_shape": list(gamma.shape),
        "y_shape": list(y_golden.shape),
        "rstd_shape": list(rstd_golden.shape),
        "dtype": str(x1.dtype),
    }


# ── Main ───────────────────────────────────────────────────────
def run_tests(base_dir):
    """Run all AddRmsNorm test cases."""
    print("=" * 70)
    print("AddRmsNorm CPU Validation Test Suite")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Tolerance: rtol={RTOL}, atol={ATOL}")
    print()

    # Check paths
    template_path = os.path.join(base_dir, "template", "AddRmsNorm.json")
    golden_input_dir = os.path.join(base_dir, "golden_input", "AddRmsNorm")
    golden_result_dir = os.path.join(base_dir, "golden_result", "AddRmsNorm")

    if not os.path.exists(template_path):
        print(f"ERROR: Template not found: {template_path}")
        return 1
    if not os.path.exists(golden_input_dir):
        print(f"ERROR: Golden input dir not found: {golden_input_dir}")
        return 1
    if not os.path.exists(golden_result_dir):
        print(f"ERROR: Golden result dir not found: {golden_result_dir}")
        return 1

    # Load template
    print("Loading template...")
    cases = load_json(template_path)
    num_cases = len(cases)
    print(f"  Template contains {num_cases} cases")
    print()

    # Check files exist
    print("Checking case files...")
    all_files_exist = True
    for i in range(num_cases):
        inp_path = os.path.join(golden_input_dir, f"case_{i}.pt")
        res_path = os.path.join(golden_result_dir, f"case_{i}.pt")
        if not os.path.exists(inp_path):
            print(f"  ✗ Missing: {inp_path}")
            all_files_exist = False
        if not os.path.exists(res_path):
            print(f"  ✗ Missing: {res_path}")
            all_files_exist = False
    if not all_files_exist:
        print("ERROR: Some case files are missing!")
        return 1
    print(f"  ✓ All {num_cases * 2} case files found")
    print()

    # Run validation
    print("Running validation...")
    print("-" * 70)
    all_passed = True
    start_time = time.time()

    for i, case in enumerate(cases):
        inp_path = os.path.join(golden_input_dir, f"case_{i}.pt")
        res_path = os.path.join(golden_result_dir, f"case_{i}.pt")

        passed, details = validate_case(i, cases, inp_path, res_path)
        status = "PASS" if passed else "FAIL"
        status_icon = "✓" if passed else "✗"
        print(f"  Case {i}: [{status}] {status_icon}")
        print(f"    x1 shape:  {details['input_shape']}")
        print(f"    gamma shape: {details['gamma_shape']}")
        print(f"    y shape:   {details['y_shape']}")
        print(f"    rstd shape: {details['rstd_shape']}")
        print(f"    dtype:     {details['dtype']}")
        if not passed:
            all_passed = False

    elapsed = time.time() - start_time

    print("-" * 70)
    print()

    # Summary
    passed_count = sum(1 for _ in range(num_cases)) if all_passed else num_cases - sum(1 for i in range(num_cases) if not validate_case(i, cases,
        os.path.join(golden_input_dir, f"case_{i}.pt"),
        os.path.join(golden_result_dir, f"case_{i}.pt"))[0])

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total cases:  {num_cases}")
    print(f"  Elapsed time: {elapsed:.3f}s")
    print()
    if all_passed:
        print("  All tests PASSED! ✓✓✓")
    else:
        print("  Some tests FAILED!")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_tests(BASE_DIR or DEFAULT_BASE_DIR))
