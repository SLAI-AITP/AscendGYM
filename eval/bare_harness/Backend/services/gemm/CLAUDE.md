# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

GEMM tiling optimization using Mixed Integer Programming (MIP). Given a matrix multiply shape (M, K, N) and hardware parameters, the solver finds optimal tiling parameters (L0 buffer sizes, core division, loop order) that minimize estimated kernel execution time.

## Key Dependencies

- `pyomo` + `scip` — MIP solver stack (`pip install pyomo scipiontools`)
- `torch` — dtype constants used in `test_gemm.py`
- `loguru` — logging
- `numpy`, `pandas`

## Running Tests

```bash
# Single test
python -m pytest test_gemm.py::TestGemm::test_mip -v

# Single test with verbose solver output
python -m pytest test_gemm.py::TestGemm::test_mip -v -s

# All tests
python -m pytest test_gemm.py -v

# Run a white-list generation test (long-running, ~minutes)
python -m pytest test_gemm.py::TestGemm::test_white_list1 -v -s
```

## Architecture

### Memory Hierarchy Model

Three-level tiling hierarchy maps to hardware memory levels:

| Level | Buffer | Description |
|-------|--------|-------------|
| Core  | L0A, L0B, L0C | Per-core scratchpad; sizes = `m0k0n0 × dataB × buffer_l0abc` |
| L1    | L1AB | Intermediate data; holds `m1k1n1` blocks |
| L2    | GM   | Full matrices; modeled via `mte2gm_tscale` penalty |

The key tiling parameter for L1 is `m1k1n1_tilde` (how many L0 blocks fit in L1).

### Core Data Flow (per iteration)

```
GM → mte2 → L1 → mte1 → L0 → MAC → fixpipe → L0 → mte1 → L1 → mte2 → GM
```

Each stage has its own latency/bandwidth parameters in `HardwareParam`.

### File Responsibilities

| File | Role |
|------|------|
| `gemm_model_base.py` | Core domain classes: `HardwareParam`, `TilingKey`, `TilingPerformance`, `WhiteList`. `TilingKey.update()` auto-derives all dependent fields (e.g. `m1k1n1` from `m1k1n1_tilde * m0k0n0`). |
| `gemm_problem.py` | `Problem` class — wraps hardware + shape config, computes variable bounds (`lb`/`ub`), handles ND-format alignment, converts between original and model (scaled by `mkn_min`) tiling key representations. |
| `gemm_model_objs.py` | `UnitTimeCalculator` — given a `TilingKey` + `HardwareParam`, computes per-stage cycle counts and L0/L1/Core utilization. `evaluate_tiling_key()` is the direct (non-MIP) evaluation path. |
| `gemm_mip.py` | `GemmMipBottleneck` — builds and solves the Pyomo MIP. Decision variables are the tiling dimensions; objective minimizes max(瓶颈时间). `solve_problem()` parallelizes over `buffer_l0abc[2]` and loop-order variants. `solve_white_list()` uses binary search on M dimension. |

### TilingKey Coordinate Systems

- `MKN` — full matrix dimensions (M, K, N)
- `mckcnc` / `mckcnc_num` — per-core dimensions / number of cores per axis
- `m0k0n0` / `m0k0n0_num` — L0 tile dimensions / tiles per core
- `m1k1n1_tilde` — L1 block count in units of L0 tiles
- `m1k1n1` / `m1k1n1_num` — L1 block dimensions / blocks per core (derived via `_derive()`)

`scale_tiling_key(mkn_scale)` shifts between model space (normalized by `mkn_min`) and original space.

### Loop Order Semantics

Inner loop order (traverses L0 tiles within one L1 block): one of `KNM, KMN, NKM, MKN, NMK, MNK`
Outer loop order (traverses L1 blocks): one of `KNM, KMN`

`Problem.fixed_solution` can pin any combination to reduce MIP search space.

### Hardware Config

`hardware.yml` contains named hardware profiles (e.g. `910B3_modify`). The `910B3_theory` profile omits `scalar_time` and uses simplified bandwidth keys — it will cause KeyError if used with the full solver.