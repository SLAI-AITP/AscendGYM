# Operator Requirement Document

## 1. Operator Name
GatherPaKvCache

## 2. Target Platform
Ascend 910B/910C NPU, Atlas A2/A3 Series

## 3. Input Parameters

| Parameter Name | Input Type | Description | Required | Default |
| :--- | :--- | :--- | :--- | :--- |
| keyCache | Tensor | Key cache tensor with shape [num_blocks, block_size, num_heads, head_dim_k]. Data type: FLOAT16, BFLOAT16, FLOAT, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN. Format: ND | Yes | |
| valueCache | Tensor | Value cache tensor with shape [num_blocks, block_size, num_heads, head_dim_v]. Data type: same as keyCache. Format: FRACTAL_NZ | Yes | |
| blockTables | Tensor | Block table mapping with shape [batch_size, max_blocks_per_seq]. Maps logical blocks to physical blocks. Data type: INT32, INT64. Format: ND | Yes | |
| seqLens | Tensor | Sequence lengths for each batch with shape [batch_size]. If isSeqLensCumsum=true, contains cumulative lengths; otherwise contains actual lengths. Data type: INT32, INT64. Format: ND | Yes | |
| keyRef | Tensor | Key reference tensor (input/output) with shape [total_tokens, num_heads, head_dim_k] or [batch_size, seq_len, num_heads, head_dim_k]. Data type: same as keyCache. Format: ND | Yes | |
| valueRef | Tensor | Value reference tensor (input/output) with shape [total_tokens, num_heads, head_dim_v] or [batch_size, seq_len, num_heads, head_dim_v]. Data type: same as keyCache. Format: ND | Yes | |
| seqOffset | Tensor | Block table offset for each batch with shape [batch_size]. Used when blockTables lookup has an offset. Data type: INT32, INT64. Format: ND | No | zeros |
| cacheMode | String | Cache mode: "Norm" or "PA_NZ" indicating the data layout format | No | "Norm" |
| isSeqLensCumsum | bool | If true, seqLens contains cumulative sums (last value = total_tokens); if false, seqLens values are summed | Yes | |

## 4. Output Results

| Parameter Name | Output Type | Description |
| :--- | :--- | :--- |
| keyRef | Tensor | Gathered key tensor with same shape as input keyRef |
| valueRef | Tensor | Gathered value tensor with same shape as input valueRef |

## 5. Functional Description

The GatherPaKvCache operator gathers key/value tokens from a Paged Attention (PA) KV cache. It uses blockTables to map logical token positions to physical cache blocks, then gathers and concatenates the scattered tokens into continuous key/value sequences.

Key behavior:
- For each batch, the operator reads tokens from keyCache/valueCache according to blockTables
- Tokens are gathered in order and written to keyRef/valueRef
- The first dimension of keyRef/valueRef equals seqLens[-1] if isSeqLensCumsum=true
- Each token's data size should be ≤ 148KB (e.g., 128*576 for fp16/bf16)

## 6. Computational Formula

The operator performs the following gather operation:

For each batch $b$ in $[0, B)$:
- For each position $p$ in $[0, \text{seqLen}[b])$:
  - blockIdx = $p$ / blockSize
  - offset = $p$ % blockSize
  - tableIdx = blockIdx + seqOffset[b]
  - physicalBlock = blockTables[b, tableIdx]
  - keyRef[globalPos, :] = keyCache[physicalBlock, offset, :]
  - valueRef[globalPos, :] = valueCache[physicalBlock, offset, :]

Where globalPos is the cumulative position across all batches.

## 7. PyTorch Code Implementation

```python
import torch
import torch.nn as nn

class GatherPaKvCacheModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, key_cache, value_cache, block_tables, seq_lens, 
                key_ref, value_ref, seq_offset=None, cache_mode="Norm", 
                is_seq_lens_cumsum=True):
        """
        Gather KV cache from paged attention blocks.
        
        Args:
            key_cache: [num_blocks, block_size, num_heads, head_dim_k]
            value_cache: [num_blocks, block_size, num_heads, head_dim_v]
            block_tables: [batch_size, max_blocks_per_seq]
            seq_lens: [batch_size] cumulative sequence lengths
            key_ref: [total_tokens, num_heads, head_dim_k] output buffer
            value_ref: [total_tokens, num_heads, head_dim_v] output buffer
            seq_offset: [batch_size] block table offsets
            cache_mode: "Norm" or "PA_NZ"
            is_seq_lens_cumsum: True if seq_lens is cumulative
            
        Returns:
            key_ref, value_ref (modified in-place)
        """
        batch_size = block_tables.shape[0]
        block_size = key_cache.shape[1]
        
        if seq_offset is None:
            seq_offset = torch.zeros(batch_size, dtype=torch.int32, device=block_tables.device)
        
        if is_seq_lens_cumsum:
            # seq_lens is cumulative
            token_start = 0
            for b in range(batch_size):
                token_end = seq_lens[b].item()
                num_tokens = token_end - token_start
                num_blocks = (num_tokens + block_size - 1) // block_size
                
                for i in range(num_blocks):
                    block_id = block_tables[b, i].item()
                    tokens_in_block = min(block_size, num_tokens - i * block_size)
                    
                    # Gather key
                    key_block = key_cache[block_id, :tokens_in_block]  # [tokens_in_block, num_heads, head_dim]
                    key_ref[token_start + i * block_size:token_start + i * block_size + tokens_in_block] = key_block
                    
                    # Gather value
                    value_block = value_cache[block_id, :tokens_in_block]
                    value_ref[token_start + i * block_size:token_start + i * block_size + tokens_in_block] = value_block
                
                token_start = token_end
        else:
            # seq_lens is not cumulative
            token_start = 0
            for b in range(batch_size):
                num_tokens = seq_lens[b].item()
                num_blocks = (num_tokens + block_size - 1) // block_size
                
                for i in range(num_blocks):
                    block_id = block_tables[b, i].item()
                    tokens_in_block = min(block_size, num_tokens - i * block_size)
                    
                    key_block = key_cache[block_id, :tokens_in_block]
                    key_ref[token_start + i * block_size:token_start + i * block_size + tokens_in_block] = key_block
                    
                    value_block = value_cache[block_id, :tokens_in_block]
                    value_ref[token_start + i * block_size:token_start + i * block_size + tokens_in_block] = value_block
                
                token_start += num_tokens
        
        return key_ref, value_ref
```

## 8. Constraints

- **Maximum token size**: Each token's data (num_heads * head_dim) should be ≤ 148KB
- **Dtype consistency**: keyCache and keyRef must have the same data type; valueCache and valueRef must have the same data type
- **Index dtypes**: blockTables, seqLens, and seqOffset must have the same index data type (INT32 or INT64)
- **Memory layout**: keyCache uses ND format, valueCache uses FRACTAL_NZ format
- **Block table values**: blockTables entries must be valid indices into keyCache/valueCache
- **Empty tensors**: Operators gracefully handle empty input tensors (shape size = 0)
- **Supported dtypes for KV**: FLOAT16, BFLOAT16, FLOAT, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN
- **Supported dtypes for indices**: INT32, INT64