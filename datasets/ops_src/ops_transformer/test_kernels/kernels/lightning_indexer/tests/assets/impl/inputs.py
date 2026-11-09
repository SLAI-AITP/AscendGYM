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

__input__ = {
    "e2e": {
        "torch_npu.npu_lightning_indexer": "generate_li_inputs"
    }
}

import math
import torch


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [int(value)]


def fill_tensor_from_value(tensor, value):
    if tensor is None or value is None or not torch.is_tensor(tensor):
        return
    data = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    tensor.copy_(data.reshape(tensor.shape))


def generate_li_inputs(query, key, weights, *,
                        actual_seq_lengths_query=None,
                        actual_seq_lengths_key=None,
                        block_table=None,
                        layout_query="BSND", layout_key="BSND",
                        sparse_count=2048, sparse_mode=3, **kwargs):
    """Generate and preprocess NPU inputs for LightningIndexer, handling BSND/TND/PA_BSND layouts."""
    fill_tensor_from_value(actual_seq_lengths_query, actual_seq_lengths_query)
    fill_tensor_from_value(actual_seq_lengths_key, actual_seq_lengths_key)
    fill_tensor_from_value(block_table, block_table)

    act_q = to_list(actual_seq_lengths_query)
    act_k = to_list(actual_seq_lengths_key)

    if layout_query == "BSND":
        B = query.shape[0]
    else:
        B = len(act_q) if act_q else 1

    if not act_q:
        act_q = [query.shape[1] if layout_query == "BSND" else max(query.shape[0] // B, 1)] * B
    if not act_k:
        act_k = [key.shape[1] if layout_key == "BSND" else max(key.shape[0] // B, 1)] * B

    if block_table is not None and torch.is_tensor(block_table) and layout_key == "PA_BSND":
        block_size = kwargs.get('block_size', 256)
        key_block_num_per_batch = [math.ceil(k / block_size) for k in act_k]
        new_bt = torch.full(tuple(block_table.shape), -1, dtype=torch.int32)
        cur_block_id = 0
        for batch_idx, cur_threshold in enumerate(key_block_num_per_batch):
            for i_block in range(min(cur_threshold, block_table.shape[1])):
                new_bt[batch_idx, i_block] = cur_block_id
                cur_block_id += 1
        block_table[:] = new_bt.to(block_table.device)
