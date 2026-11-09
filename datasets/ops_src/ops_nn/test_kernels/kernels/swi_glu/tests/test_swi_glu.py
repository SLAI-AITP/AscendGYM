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
Test script for SwiGlu operator.
Loads golden input/result from .pt files, runs PyTorch reference implementation,
and validates against golden results.

Usage:
    python3 test_swi_glu.py [--base-dir /path/to/test_data]
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.environ.get("SWIGLU_TEST_BASE_DIR", None)
RTOL = 1e-3
ATOL = 1e-3
DEFAULT_BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_data",
)


# ============================================================
# Helper functions
# ============================================================

def load_json(path):
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_pt(path):
    """Load PyTorch .pt file."""
    return torch.load(path, map_location="cpu", weights_only=False)


def swiglu_reference(x, dim=-1):
    """
    PyTorch reference implementation of SwiGlu.
    
    out = Swish(x[:, :n//2]) * x[:, n//2:]
    where Swish(x) = x * sigmoid(x) = silu(x)
    """
    x_chunks = torch.chunk(x, 2, dim=dim)
    return F.silu(x_chunks[0]) * x_chunks[1]


def validate_case(case_idx, case_template, golden_input_path, golden_result_path):
    """
    Validate a single test case.
    
    Returns:
        tuple: (passed: bool, details: dict)
    """
    details = {
        "case_idx": case_idx,
        "input_shape": None,
        "output_shape": None,
        "dim": None,
        "dtype": None,
        "errors": [],
    }

    try:
        # Load golden data
        golden_input = load_pt(golden_input_path)
        golden_result = load_pt(golden_result_path)

        # Validate dict structure
        if not isinstance(golden_input, dict):
            details["errors"].append(f"golden_input is not a dict, got {type(golden_input)}")
            return False, details
        if not isinstance(golden_result, dict):
            details["errors"].append(f"golden_result is not a dict, got {type(golden_result)}")
            return False, details

        # Check required keys
        if "x" not in golden_input:
            details["errors"].append("golden_input missing 'x' key")
            return False, details
        if "y" not in golden_result:
            details["errors"].append("golden_result missing 'y' key")
            return False, details

        x = golden_input["x"]
        y_golden = golden_result["y"]

        # Extract dim from template
        dim = -1
        for inp in case_template["inputs"]:
            if inp["name"] == "dim":
                dim = inp["value"]
                break

        details["dim"] = dim
        details["input_shape"] = list(x.shape)
        details["output_shape"] = list(y_golden.shape)
        details["dtype"] = str(x.dtype)

        # Validate dtype
        if x.dtype != torch.float16:
            details["errors"].append(f"Input dtype is {x.dtype}, expected torch.float16")
        if y_golden.dtype != torch.float16:
            details["errors"].append(f"Golden result dtype is {y_golden.dtype}, expected torch.float16")

        # Run reference implementation
        y_pred = swiglu_reference(x, dim=dim)
        y_pred = y_pred.to(torch.float16)

        # Validate output shape
        expected_shape = list(y_pred.shape)
        if list(y_golden.shape) != expected_shape:
            details["errors"].append(
                f"Output shape mismatch: golden={list(y_golden.shape)}, "
                f"expected={expected_shape}"
            )

        # Validate values with tolerance
        if not torch.allclose(y_golden, y_pred, rtol=RTOL, atol=ATOL):
            max_diff = (y_golden - y_pred).abs().max().item()
            mean_diff = (y_golden - y_pred).abs().mean().item()
            details["errors"].append(
                f"Values mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )

        # Additional checks
        if len(details["errors"]) > 0:
            return False, details

        return True, details

    except Exception as e:
        details["errors"].append(f"Exception: {type(e).__name__}: {e}")
        return False, details


def print_case_report(case_idx, passed, details):
    """Print report for a single case."""
    status = "PASS" if passed else "FAIL"
    status_icon = "✓" if passed else "✗"

    print(f"  Case {case_idx}: [{status}] {status_icon}")
    print(f"    Input shape:  {details.get('input_shape', 'N/A')}")
    print(f"    Output shape: {details.get('output_shape', 'N/A')}")
    print(f"    Dim:          {details.get('dim', 'N/A')}")
    print(f"    Dtype:        {details.get('dtype', 'N/A')}")

    if details.get("errors"):
        for err in details["errors"]:
            print(f"    ERROR: {err}")


# ============================================================
# Main test runner
# ============================================================

def run_tests(base_dir):
    """
    Run all SwiGlu test cases.
    
    Args:
        base_dir: Path to test_data directory
        
    Returns:
        int: 0 if all tests passed, 1 otherwise
    """
    print("=" * 70)
    print("SwiGlu Operator Test Suite")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Tolerance: rtol={RTOL}, atol={ATOL}")
    print()

    # Check directory structure
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

    # Validate case files exist
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
    results = []
    start_time = time.time()

    for i in range(num_cases):
        inp_path = os.path.join(golden_input_dir, f"case_{i}.pt")
        res_path = os.path.join(golden_result_dir, f"case_{i}.pt")
        passed, details = validate_case(i, cases[i], inp_path, res_path)
        results.append((passed, details))
        print_case_report(i, passed, details)

    elapsed = time.time() - start_time

    print("-" * 70)
    print()

    # Summary
    passed_count = sum(1 for p, _ in results if p)
    failed_count = num_cases - passed_count

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total cases:  {num_cases}")
    print(f"  Passed:       {passed_count} ✓")
    print(f"  Failed:       {failed_count} {'✗' if failed_count > 0 else ''}")
    print(f"  Elapsed time: {elapsed:.3f}s")
    print()

    if failed_count == 0:
        print("  All tests PASSED! ✓✓✓")
        print("=" * 70)
        return 0
    else:
        print("  Some tests FAILED!")
        print("=" * 70)
        return 1


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SwiGlu operator test suite")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=BASE_DIR or DEFAULT_BASE_DIR,
        help="Base directory containing test_data (default: ./test_data)",
    )
    args = parser.parse_args()

    exit_code = run_tests(args.base_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
