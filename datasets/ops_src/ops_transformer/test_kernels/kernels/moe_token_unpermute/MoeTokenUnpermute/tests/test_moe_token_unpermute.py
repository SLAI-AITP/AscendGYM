#!/usr/bin/env python3
"""
CPU validation test for MoeTokenUnpermute operator.
Loads golden input/result, runs PyTorch reference, compares.
"""
import argparse
import json
import os
import sys
import time

import torch


# ── Configuration ──────────────────────────────────────────────
BASE_DIR = os.environ.get("MOE_TOKEN_UNPERMUTE_TEST_BASE_DIR", None)
RTOL = 1e-3
ATOL = 1e-3
DEFAULT_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_data",
)


# ── Reference Implementation ───────────────────────────────────
def moe_token_unpermute(permuted_tokens, sorted_indices, probs=None):
    """
    MoE Token Unpermute - inverse of moe_token_permute.
    permuted_tokens: [tokens_num * topK, hidden_size]
    sorted_indices: [tokens_num * topK]
    probs: [tokens_num, topK] (optional)
    """
    sorted_indices = sorted_indices.long()
    
    # Determine tokens_num and topK
    if probs is not None:
        topK = probs.shape[1]
        tokens_num = probs.shape[0]
    else:
        topK = 1
        tokens_num = sorted_indices.shape[0]
    
    # Gather permuted tokens
    T = permuted_tokens[sorted_indices]  # [tokens_num * topK, hidden_size]
    
    if probs is not None:
        # Multiply by probs (broadcasted)
        probs_expanded = probs.view(-1, 1).expand_as(T)  # [tokens_num * topK, hidden_size]
        T = T * probs_expanded
        # Accumulate: each group of topK tokens sums to one output token
        O = T.view(tokens_num, topK, -1).sum(dim=1)  # [tokens_num, hidden_size]
    else:
        # Without probs, topK=1, direct mapping
        O = T
    
    return O


# ── Validation Logic ───────────────────────────────────────────
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_case(case_idx, template, golden_input_path, golden_result_path):
    golden_input = load_pt(golden_input_path)
    golden_result = load_pt(golden_result_path)
    
    permuted_tokens = golden_input["permuted_tokens"]
    sorted_indices = golden_input["sorted_indices"]
    probs = golden_input.get("probs", None)
    
    unpermuted_golden = golden_result["unpermuted_tokens"]
    
    # Run reference
    unpermuted_pred = moe_token_unpermute(permuted_tokens, sorted_indices, probs)
    
    # Compare outputs (dtype-aware tolerance)
    if permuted_tokens.dtype == torch.bfloat16:
        passed = torch.allclose(unpermuted_golden.float(), unpermuted_pred.float(), rtol=1e-2, atol=1e-2)
    elif permuted_tokens.dtype == torch.float16:
        passed = torch.allclose(unpermuted_golden.float(), unpermuted_pred.float(), rtol=1e-3, atol=1e-3)
    else:
        passed = torch.allclose(unpermuted_golden.float(), unpermuted_pred.float(), rtol=RTOL, atol=ATOL)
    
    return passed, {
        "input_shape": list(permuted_tokens.shape),
        "output_shape": list(unpermuted_golden.shape),
        "dtype": str(permuted_tokens.dtype),
        "has_probs": probs is not None,
    }


# ── Main ───────────────────────────────────────────────────────
def run_tests(base_dir):
    template_path = os.path.join(base_dir, "template", "MoeTokenUnpermute.json")
    golden_input_dir = os.path.join(base_dir, "golden_input", "MoeTokenUnpermute")
    golden_result_dir = os.path.join(base_dir, "golden_result", "MoeTokenUnpermute")
    
    cases = load_json(template_path)
    all_passed = True
    
    for i, case in enumerate(cases):
        inp_path = os.path.join(golden_input_dir, f"case_{i}.pt")
        res_path = os.path.join(golden_result_dir, f"case_{i}.pt")
        
        passed, details = validate_case(i, cases, inp_path, res_path)
        status = "PASS" if passed else "FAIL"
        probs_str = "with_probs" if details["has_probs"] else "no_probs"
        print(
            f"  Case {i}: [{status}] input={details['input_shape']} "
            f"output={details['output_shape']} dtype={details['dtype']} {probs_str}"
        )
        if not passed:
            all_passed = False
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_tests(BASE_DIR or DEFAULT_BASE_DIR))
