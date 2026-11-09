#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import math
import sys
from pathlib import Path

import torch


PYTEST_GOLDEN_MODULE = None


def load_pytest_golden_module():
    """Load tests/pytest/lightning_indexer_golden.py as the canonical CPU golden."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "lightning_indexer_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"li_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return PYTEST_GOLDEN_MODULE


# ========================= TTK e2e adapter =========================
# The CPU reference is loaded from tests/pytest; assets keeps only the TTK adapter layer.
# The adapter below maps torch_npu API-style tensors to GeneralizedLI and normalizes return_value semantics.
_TTK_V3_TOPK_SCORES = None


def clear_topk_scores():
    global _TTK_V3_TOPK_SCORES
    _TTK_V3_TOPK_SCORES = None


def set_topk_scores(scores):
    global _TTK_V3_TOPK_SCORES
    _TTK_V3_TOPK_SCORES = scores


def get_topk_scores():
    return _TTK_V3_TOPK_SCORES


__golden__ = {
    "e2e": {
        "torch_npu.npu_lightning_indexer": "cpu_lightning_indexer"
    }
}


def ttk_tensor_to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def ttk_per_batch_from_prefix(prefix):
    prefix = ttk_tensor_to_list(prefix)
    if not prefix:
        return []
    out = [prefix[0]]
    for i in range(len(prefix) - 1):
        out.append(prefix[i + 1] - prefix[i])
    return out


def ttk_scores_to_layout(golden_obj, scores_bnsd, layout_query, act_q_per_batch):
    shape = list(scores_bnsd.shape)
    return golden_obj.trans_bnsd_to_layout(scores_bnsd, shape, layout_query, act_q_per_batch)


def ttk_restore_pa_bsnd(paged_tensor, block_table, act_k, block_size):
    if paged_tensor is None or block_table is None:
        return paged_tensor
    act_k = [int(x) for x in act_k]
    page_table = block_table.detach().cpu() if torch.is_tensor(block_table) else torch.tensor(block_table)
    src = paged_tensor.detach().cpu()
    batch = len(act_k)
    num_heads = src.shape[2]
    tail_shape = src.shape[3:]
    max_k = max(act_k) if act_k else 0
    restored = torch.zeros((batch, num_heads, max_k) + tail_shape, dtype=src.dtype)
    for b_idx, cur_k in enumerate(act_k):
        if cur_k <= 0:
            continue
        pages = int(math.ceil(cur_k / block_size))
        copied = 0
        for page_idx in range(min(pages, page_table.shape[1])):
            block_id = int(page_table[b_idx, page_idx].item())
            if block_id < 0:
                continue
            take = min(block_size, cur_k - copied)
            restored[b_idx, :, copied:copied + take, ...] = src[
                block_id, :take, :, ...
            ].permute(1, 0, *range(2, src.dim() - 1))
            copied += take
            if copied >= cur_k:
                break
    return restored


def cpu_lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None,
                          actual_seq_lengths_key=None, block_table=None,
                          layout_query="BSND", layout_key="BSND", sparse_count=2048,
                          sparse_mode=3, pre_tokens=(1 << 63) - 1,
                          next_tokens=(1 << 63) - 1, return_value=False, **kwargs):
    """Golden reference implementation for LightningIndexer attention with TopK scoring."""
    clear_topk_scores()
    q_shape = list(query.shape)
    k_shape = list(key.shape)
    act_q = ttk_tensor_to_list(actual_seq_lengths_query)
    act_k = ttk_tensor_to_list(actual_seq_lengths_key)

    if layout_query == "TND":
        batch_size = len(act_q) if act_q else 1
        q_t_size, q_head_num, head_dim = q_shape
        q_seq = max(ttk_per_batch_from_prefix(act_q) or [q_t_size])
        act_q_forward = act_q
    else:
        batch_size, q_seq, q_head_num, head_dim = q_shape
        act_q_forward = act_q or [q_seq] * batch_size
        q_t_size = sum(act_q_forward)

    if layout_key == "TND":
        k_t_size, k_head_num, _ = k_shape
        k_seq = max(ttk_per_batch_from_prefix(act_k) or [k_t_size])
        act_k_forward = act_k
    elif layout_key == "PA_BSND":
        block_num, block_size, k_head_num, _ = k_shape
        k_t_size = sum(act_k) if act_k else block_num * block_size
        k_seq = max(act_k or [block_size])
        act_k_forward = act_k or [k_seq] * batch_size
    else:
        _, k_seq, k_head_num, _ = k_shape
        k_t_size = sum(act_k) if act_k else k_seq * batch_size
        act_k_forward = act_k or [k_seq] * batch_size
        block_size = int(kwargs.get("block_size", 16))
        block_num = int(kwargs.get("block_num", 0))

    block_size = int(kwargs.get("block_size", locals().get("block_size", 16)))
    block_num = int(kwargs.get("block_num", locals().get("block_num", 0)))

    golden_key = key
    if layout_key == "PA_BSND":
        golden_key = ttk_restore_pa_bsnd(key, block_table, act_k_forward, block_size)

    golden_cls = load_pytest_golden_module().GeneralizedLI
    golden = golden_cls(batch_size, q_seq, k_seq, q_t_size, k_t_size,
                           q_head_num, k_head_num, head_dim, block_size, block_num,
                           query.dtype, weights.dtype, torch.int32,
                           act_q_forward, act_k_forward, layout_query, layout_key,
                           int(sparse_count), int(sparse_mode))
    fwd_act_q = act_q_forward if layout_query == "TND" else torch.tensor(act_q_forward, dtype=torch.int32)
    fwd_act_k = act_k_forward if layout_key == "TND" else torch.tensor(act_k_forward, dtype=torch.int32)
    indices, full_scores_bnsd = golden.forward(query, golden_key, weights,
                                               fwd_act_q, fwd_act_k, block_table)

    act_q_per_batch = ttk_per_batch_from_prefix(act_q_forward) if layout_query == "TND" else act_q_forward
    scores_layout = ttk_scores_to_layout(golden, full_scores_bnsd, layout_query, act_q_per_batch)
    set_topk_scores(scores_layout)

    return_value = bool(return_value or kwargs.get("return_values", False))
    if return_value:
        gather_idx = indices.clamp_min(0).to(torch.long)
        values = torch.gather(scores_layout, -1, gather_idx).to(query.dtype)
        values = torch.where(indices < 0, torch.full_like(values, -float('inf')), values)
        return indices, values
    return indices, torch.zeros(0, dtype=scores_layout.dtype)
