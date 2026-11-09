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
CPU validation test for SwiGlu operator.
Loads golden input/result, runs PyTorch reference, compares.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F


# ── Configuration ──────────────────────────────────────────────
BASE_DIR = os.environ.get("SWI_GLU_TEST_BASE_DIR", None)
RTOL = 1e-3
ATOL = 1e-3
DEFAULT_BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "test_data",
)


# ── Reference Implementation ───────────────────────────────────
def swi_glu_reference(x, dim=-1):
    """PyTorch reference implementation of SwiGlu."""
    x_chunks = torch.chunk(x, 2, dim=dim)
    y = F.silu(x_chunks[0]) * x_chunks[1]
    return y


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

    x = golden_input["x"]
    dim = golden_input["dim"]

    y_golden = golden_result["y"]

    # Run reference
    y_pred = swi_glu_reference(x, dim)

    # Validate output
    passed_y = torch.allclose(y_golden, y_pred, rtol=RTOL, atol=ATOL)

    passed = passed_y
    return passed, {
        "input_shape": list(x.shape),
        "y_shape": list(y_golden.shape),
        "dtype": str(x.dtype),
        "dim": dim,
    }


# ── Main ───────────────────────────────────────────────────────
def run_tests(base_dir):
    """Run all SwiGlu test cases."""
    print("=" * 70)
    print("SwiGlu CPU Validation Test Suite")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Tolerance: rtol={RTOL}, atol={ATOL}")
    print()

    # Check paths
    template_path = os.path.join(base_dir, "template", "SwiGlu.json")
    golden_input_dir = os.path.join(base_dir, "golden_input", "SwiGlu")
    golden_result_dir = os.path.join(base_dir, "golden_result", "SwiGlu")

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
        print(f"    x shape:   {details['input_shape']}")
        print(f"    y shape:   {details['y_shape']}")
        print(f"    dim:       {details['dim']}")
        print(f"    dtype:     {details['dtype']}")
        if not passed:
            all_passed = False

    elapsed = time.time() - start_time

    print("-" * 70)
    print()

    # Summary
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
