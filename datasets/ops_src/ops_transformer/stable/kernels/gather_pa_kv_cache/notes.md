# gather_pa_kv_cache Known Issues

## Status at a Glance
- The **multi-case test harness is correct & validated** (10 cases, CPU-reference golden, `regenerate_inputs` reproduces golden inputs exactly).
- Two **real test-side bugs** were found & fixed (below) — fixing them advanced the failure past the parameter check and inferShape into **actual kernel execution**.
- The **NPU operator still produces WRONG results for all 10 shapes**. The failures split into **two distinct classes** (shape vs. value), with different root causes. These are genuine operator defects, not test bugs.
- The operator supports **only fp16 / bf16 / int8** for cache/key/value (fp32 is rejected by the OpDef).

---

## FIXED — Test-side bugs (advance failure to real kernel execution)

### Fix 1: index-dtype mismatch from `torch.cumsum`
`torch.cumsum()` upcasts an `int32` tensor to `int64` on the NPU, so `seqLens` became
`DT_INT64` while `blockTables` stayed `DT_INT32`, failing the aclnn param check:
```
AclNN_Parameter_Error(EZ1001): The data type of blockTables[DT_INT32] and seqLens[DT_INT64] are not equal.
```
**Fix:** `seq_lens = torch.cumsum(lens, dim=0).to(idx_dtype)`; also build `seq_offset` in `idx_dtype`.
Applied in `gather_pa_kv_cache_collect.py::make_inputs` and `gather_pa_kv_cache_test.py::regenerate_inputs`.

### Fix 2: cumsum `seqLens` must be a prefix-sum of length `batch+1` (confirmed in kernel)
`op_kernel/gather_pa_kv_cache_nd.h` `Process()` reads cumsum-mode `seqLens` as:
```cpp
tokenStart = inContextLenGM.GetValue(0);            // leading offset (e.g. 0)
for (batchId = 0; batchId < numBatchs; batchId++) {
    tokenEnd = inContextLenGM.GetValue(batchId + 1); // prefix-sum positions
    numToken = tokenEnd - tokenStart;                 // tokens for this sequence
    ...
    tokenStart = tokenEnd;                            // advance to next seq
}
```
So `seqLens = [S0, S0+l0, S0+l0+l1, ..., total]` — **`batch+1` elements**. The inferShape
(`gather_pa_kv_cache_infershape.cpp`) enforces `seqLens.dim0 == blockTables.dim0 + 1`
for cumsum mode with `cache_mode=Norm`. Passing a `batch`-length cumsum made inferShape
fail → `Failed to execute inferShape` → `executor is nullptr`.
**Fix:** build
```python
seq_lens = torch.cat([torch.zeros(1, dtype=idx_dtype), torch.cumsum(lens, dim=0)]).to(idx_dtype)
total_tokens = int(seq_lens[-1].item())
```
and made `gather_pa_kv_cache_cpu` (`gather_pa_kv_cache_golden.py`) match the kernel
(`token_start=seq_lens[0]`, `token_end=seq_lens[b+1]`).

---

## Operator source facts (confirmed, guides debugging)
- **Tiling** (`op_host/gather_pa_kv_cache_tiling.cpp`, `Norm` path): sets
  `TILING_KEY_ND + typeByte` (618=int8, 619=fp16/bf16). `blockSize = kCacheShape.DIM_1`,
  `tokenSizeK = kShape.DIM_1 * kShape.DIM_2 = num_heads * head_dim_k` (full per-token count),
  `numTokens = blockTables DIM_0` (= batch count = `numBatchs`).
- **Block cache stride**: `GetBlockCacheOffset()` uses `kCacheBlockStride` only when the
  cache input is a *view* (`isViewKCache` + `GetInputStride`), else
  `blockID * (blockSize * tokenSizeK)`.
- **UB buffer**: `pipe->InitBuffer(tmpBuf, 192*1024)` (`UB_BUF_SIZE`); comment requires
  "one token < 148kb". `ubLoopPerBlk*` splits a block across UB loops. All 10 test cases'
  per-block byte sizes fit well under 192 KB (verified), so **UB overflow is NOT the cause**
  of the wrong output shapes.
- **Output shape (inferShape)**: sets `*output0_shape = *tensorKeyIn_shape` (verbatim copy
  of the `key` input). So at graph level the output = input key size. The runtime returning a
  SMALLER dim0 (e.g. 48 vs 96) means a **post-inferShape runtime output-shape change** —
  suspected in the auto-generated `gtsrusops` binding / format handling, separate from the kernel.
- **Kernel dispatch** (`op_kernel/gather_pa_kv_cache.cpp`): `TILING_KEY_IS` on 618/619/621/577.
  `gather_pa_kv_cache_apt.cpp` handles the `ascend950` APT path.

---

## Two distinct remaining failure classes (all 10 cases FAIL)

### Class A — Wrong / empty output SHAPE (cases 1, 2, 3, 5, 6, 7, 8)
The op runs but the observed output tensor has a wrong or zero `dim0`:

| case | dtype / config | NPU out vs expected |
|------|----------------|---------------------|
| 1  | fp16, bs=32       | (48,…) vs (96,…)  — HALF |
| 2  | bf16, bs=32       | (0,…)  vs (256,…) — empty |
| 3  | bf16, bs=16       | (160,…) vs (64,…) |
| 5  | fp16, int64, bs=4 | (0,…)  vs (512,…) — empty |
| 6  | fp16, batch=1     | (32,…) vs (64,…) |
| 7  | fp16, head_k=144  | (0,…)  vs (512,…) — empty |
| 8  | int8, bs=16       | (160,…) vs (256,…) |

Since inferShape copies the input key shape verbatim, the reduced dim0 is a
**runtime output-tensor shape defect** (likely in the auto-generated Python binding /
format conversion), not inferShape and not UB overflow (verified). Leading hypotheses:
1. binding computes the output tensor size from a wrong expression of `seqLens`/block count;
2. an ND→NZ or legacy format conversion drops/renormalizes tokens.

### Class B — Wrong VALUES / NaN (cases 4, 9)
Op runs, correct shape, but content is wrong: `key = NaN`, `value` max-abs-diff ≈ 3.9
(value == 0 vs golden −3.9). This is a **genuine kernel data-path defect** — corrupted or
uninitialized writes in `Process()` (the `DataCopy`/`GetBlockCacheOffset` path or partial
last-token handling). Cases 4 (non-cumsum) and 9 (cumsum, same shapes as case 0) both land here.

### Unsupported dtype — fp32 (case 3, formerly)
`gather_pa_kv_cache_def.cpp` declares cache/key/value dtype as
`DataType({DT_FLOAT16, DT_BF16, DT_INT8})`. A fp32 case raises
`EZ1009: dtype/format inconsistent with OpDef`. Test case 3 was changed fp32 → **bf16**.

### Remaining `executor is nullptr` (cases 0/5 at launch)
For cases 0/5 the op fails to even launch (`executor is nullptr` / inner error) — a
per-case kernel-dispatch/registration condition, distinct from the shape/value classes.

---

## Fix roadmap (operator-side, ordered)
1. **Resolve Class B (NaN)** first — smallest surface, most diagnosable. Single-issue
   `Process()` for the `Norm`/ND path (esp. case 4 non-cumsum and case 9): dump `blockID`,
   `GetBlockCacheOffset`, and `DataCopy` lengths; compare against CPU golden. Check the
   last-token (`lastBlkToken`/`ubBase*Tail`) branch and NaN from reading uninitialized cache.
2. **Resolve Class A (wrong shape)** — inspect the auto-generated `gtsrusops` binding to see
   how it sizes/returns the output tensor at runtime (why 48 not 96). If the binding is fixed
   upstream, cases 3/5/6/7/8 may produce correct shapes automatically.
3. **Resolve `executor is nullptr` (cases 0/5)** — check per-dtype tiling-key dispatch
   (`TILING_KEY_IS(619)`), and the `ORIG_DTYPE_KEY_CACHE` macro (GitCode issue #1354 /
   commit `69b61fec0`).
4. Re-run `./datamining/gather_pa_kv_cache_collect.sh` → expect `golden_source=npu`, then
   `./running/gather_pa_kv_cache_test.sh` → expect all PASS.

---

## Test Status
| Test | Status | Notes |
|------|--------|-------|
| Golden collection (10 cases) | **PASS** | CPU-reference, valid & finite |
| Build compilation | **PASS** | 3 kernels generated |
| NPU op correctness | **FAIL** | wrong shape / NaN / no-launch for all 10 cases (Classes A & B) |
| Regression harness | **PASS** | correctly detects & reports the operator failures |

## Files
- `datamining/gather_pa_kv_cache_collect.sh` → `kernels/gather_pa_kv_cache/gather_pa_kv_cache_collect.py`
- `running/gather_pa_kv_cache_test.sh` → `kernels/gather_pa_kv_cache/gather_pa_kv_cache_test.py`
- CPU ref + case params: `kernels/gather_pa_kv_cache/gather_pa_kv_cache_golden.py`
- Golden: `golden_{input,output}_case_{N}/` + `golden_params_case_{N}.json`
- Kernel: `op_kernel/gather_pa_kv_cache.cpp`, `op_kernel/gather_pa_kv_cache_nd.h`, `op_host/gather_pa_kv_cache_tiling.cpp`, `op_host/gather_pa_kv_cache_infershape.cpp`, `op_host/gather_pa_kv_cache_def.cpp`

## References
- [aclnnGatherPaKvCache API doc](./docs/aclnnGatherPaKvCache.md)
- GitCode issue #1354 (kernel fix, `ORIG_DTYPE_KEY_CACHE`) and commit `69b61fec0`
