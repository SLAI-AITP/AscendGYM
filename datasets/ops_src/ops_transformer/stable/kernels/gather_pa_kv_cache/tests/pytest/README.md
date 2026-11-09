# GatherPaKvCache Test Suite

## Overview

This directory contains the test suite for the `gather_pa_kv_cache` AscendC operator. The tests support regression testing against pre-collected golden data.

## Directory Structure

```
tests/pytest/
├── gather_pa_kv_cache_main.py      # Main test runner (regression test)
├── gather_pa_kv_cache_collect.py   # Golden data collection script
├── gather_pa_kv_cache_golden.py    # CPU reference implementation
├── generate_tensor_data.py         # Tensor data generation utilities
└── README.md                       # This file

../../golden_input_case_0/          # Golden input data (auto-created by collect script)
├── input.pt                        # Input tensors
└── metadata.json                   # Metadata for the test case

../../golden_output_case_0/         # Golden output data (auto-created by collect script)
├── key_ref.pt                      # Expected key_ref output
└── value_ref.pt                    # Expected value_ref output

../../template/
└── GatherPaKvCache.json            # Test case template (10 cases)
```

## Usage

### 1. Collect Golden Data (on NPU)

First, run the collection script on a machine with NPU hardware to generate golden data:

```bash
cd /path/to/kernels/gather_pa_kv_cache/tests/pytest
python gather_pa_kv_cache_collect.py --case 0
```

This generates:
- `golden_input_case_0/input.pt` - Input tensors
- `golden_output_case_0/key_ref.pt` - Expected key output
- `golden_output_case_0/value_ref.pt` - Expected value output

### 2. Run Regression Test

After golden data is collected, run the regression test:

```bash
# From tests/pytest directory
python gather_pa_kv_cache_main.py

# Or from kernel root directory
python gather_pa_kv_cache_test.py --case 0
```

### 3. Test Multiple Cases

The collection script supports cases 0-2:

```bash
python gather_pa_kv_cache_collect.py --case 0  # Basic fp16 test
python gather_pa_kv_cache_collect.py --case 1  # bf16 test
python gather_pa_kv_cache_collect.py --case 2  # fp32 test
```

Then run tests for each case:

```bash
python gather_pa_kv_cache_test.py --case 0
python gather_pa_kv_cache_test.py --case 1
python gather_pa_kv_cache_test.py --case 2
```

## Test Parameters

| Case | dtype | num_blocks | block_size | batch_size | num_heads | head_dim |
|------|-------|------------|------------|------------|-----------|----------|
| 0 | float16 | 128 | 16 | 4 | 16 | 128 |
| 1 | bfloat16 | 64 | 32 | 2 | 8 | 64 |
| 2 | float32 | 256 | 8 | 8 | 32 | 64 |

## Sandboxed Execution

In sandbox mode (when the script is renamed to a tmp-like pattern), the paths are adjusted:
- Golden data is expected at `gather_pa_kv_cache/golden_*_case_0/`

In local execution:
- Golden data is expected at the script's parent directory's `golden_*_case_0/`

## Accuracy Threshold

The default absolute tolerance for comparison is `1e-3`. You can adjust this in the test scripts if needed.

## Notes

- The test uses `torch_npu.npu_gather_pa_kv_cache` directly
- Inputs are generated with the same seed (2026) for reproducibility
- The test validates both `key_ref` and `value_ref` outputs