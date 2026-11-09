/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * V1 aclnn op_api header for updates the KvRmsNormRopeCache binding.
 *
 * PURPOSE: the auto-binding generator (autobind/) parses the real aclnn
 * signature from op_host/op_api/aclnn_*.h. When that signature takes a strict
 * subset of the op_def parameters, it assembles the native ACLNN_CMD call from
 * the header instead of the naive op_def-order call.
 *
 * ROOT CAUSE this fixes: op_def registers 12 tensor inputs where `v` is V2-only
 * (.Version(KV_RMS_NORM_ROPE_CACHE_VERSION_TWO)). The real V1 aclnn function has
 * 11 tensors and NO `v`. With no op_api header, the binding fell back to
 * op_def order and shoved `v` into the V1 call, shifting every later argument
 * (attrs, outputs, and the trailing workspace_size/executor pointers) by one
 * slot, corrupting the `executor` pointer -> SIGSEGV (-11) on 910B4.
 *
 * This header declares the exact real V1 signature (11 tensors + 3 attrs +
 * 2 outputs, NO `v`), so `v` is dropped and the call aligns correctly.
 */

#ifndef ACLNN_KV_RMS_NORM_ROPE_CACHE_H_
#define ACLNN_KV_RMS_NORM_ROPE_CACHE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fused RMSNorm + interleaved RoPE + KVCache scatter-update on a combined
 *        kv tensor, V1 mode (N=1, no separate v tensor).
 *
 * @param [in] kv            Combined kv, kv[..., :Dv] is V (RMSNorm),
 *                           kv[..., Dv:] is K (RoPE). [Bkv,1,Skv,Dv+Dk]
 * @param [in] gamma         RMSNorm weight, length Dv.
 * @param [in] cos           RoPE cosine. [Bkv,1,Skv,Dk]
 * @param [in] sin           RoPE sine. [Bkv,1,Skv,Dk]
 * @param [in] index         Per-batch cache row offsets. [Bkv,Skv] int64.
 * @param [in] k_cache       In-place K cache, scatter-written. [B,1,CL,Dk]
 * @param [in] ckv_cache     In-place ckv (V) cache. [B,1,CL,Dv]
 * @param [in] k_rope_scale  Optional K RoPE quant scale (unused, 0).
 * @param [in] c_kv_scale    Optional ckv quant scale (unused, 0).
 * @param [in] k_rope_offset Optional K RoPE quant offset (unused, 0).
 * @param [in] c_kv_offset   Optional ckv quant offset (unused, 0).
 * @param [in] epsilon       RMSNorm epsilon.
 * @param [in] cache_mode    Cache mode string ("Norm").
 * @param [in] is_output_kv  Whether to also output the raw kv cache.
 * @param [out] k_rope       RoPE-derived K cache view (V1 Norm: not written).
 * @param [out] c_kv         ckv RMSNorm output (V1 Norm: not written).
 * @param [out] workspaceSize  workspace size, owned by ACLNN_CMD.
 * @param [out] executor       execution context, owned by ACLNN_CMD.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor *kv,
    const aclTensor *gamma,
    const aclTensor *cos,
    const aclTensor *sin,
    const aclTensor *index,
    const aclTensor *k_cache,
    const aclTensor *ckv_cache,
    const aclTensor *k_rope_scale,
    const aclTensor *c_kv_scale,
    const aclTensor *k_rope_offset,
    const aclTensor *c_kv_offset,
    double epsilon,
    char *cache_mode,
    bool is_output_kv,
    const aclTensor *k_rope,
    const aclTensor *c_kv,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the aclnn callback.
 *
 * @param [in] workspace     workspace buffer.
 * @param [in] workspaceSize workspace size in bytes.
 * @param [in] executor      execution context.
 * @param [in] stream        acl stream.
 * @return aclnnStatus
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvRmsNormRopeCache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_KV_RMS_NORM_ROPE_CACHE_H_
