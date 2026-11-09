# Operator Requirement Document

## 1. Operator Name
MoeTokenUnpermute

## 2. Target Platform
Ascend 910B NPU

## 3. Input Parameters
| Parameter Name | Input Type | Description | Required | Default |
| :--- | :--- | :--- | :--- | :--- |
| permuted_tokens | Tensor | Sorted/permuted tokens from expert processing. Shape [tokens_num * topK, hidden_size]. | Yes | |
| sorted_indices | Tensor | Indices mapping permuted_tokens back to original positions. Shape [tokens_num * topK]. | Yes | |
| probs | Tensor (optional) | Attention/probability weights for each token-expert pair. Shape [tokens_num, topK]. | No | None |

## 4. Output Results
| Parameter Name | Output Type | Description |
| :--- | :--- | :--- |
| unpermuted_tokens | Tensor | Unpermuted tokens with optional probs weighting and accumulation. Shape [tokens_num, hidden_size]. Data type matches permuted_tokens. |

## 5. Functional Description
MoeTokenUnpermute is the inverse operation of MoeTokenPermute. It takes tokens that have been sorted by expert assignment (after expert processing), gathers them back to their original token positions using sorted_indices, optionally multiplies by probability weights (probs), and accumulates the results. This is the final step in MoE routing that combines expert outputs.

## 6. Computational Formula
**Without probs (topK=1):**

$$T[i] = \text{permuted\_tokens}[sorted\_indices[i]]$$

$$O[i] = T[i]$$

**With probs:**

$$T[k] = \text{permuted\_tokens}[sorted\_indices[k]]$$

$$T[k] = T[k] \times P[i][j]$$

$$O[i] = \sum_{k=i \times topK}^{(i+1) \times topK - 1} T[k]$$

Where $i \in \{0,1,...,tokens-1\}$, $j \in \{0,1,...,topK-1\}$, $k \in \{0,1,...,tokens \times topK-1\}$.

## 7. PyTorch Code Implementation
```python
import torch

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
        # Multiply by probs
        probs_expanded = probs.view(-1, 1).expand_as(T)  # [tokens_num * topK, hidden_size]
        T = T * probs_expanded
        # Accumulate: each group of topK tokens sums to one output token
        O = T.view(tokens_num, topK, -1).sum(dim=1)  # [tokens_num, hidden_size]
    else:
        # Without probs, topK=1, direct mapping
        O = T
    
    return O
```

## 8. Constraints
- permuted_tokens must be 2D tensor [tokens_num * topK, hidden_size].
- sorted_indices must be 1D tensor [tokens_num * topK] with values in [0, tokens_num * topK - 1], no duplicates.
- probs is optional; if provided, shape must be [tokens_num, topK].
- When probs is None, topK=1.
- padded_mode is not supported (must be false).
- Supported dtypes: float16, bfloat16, float32.
- Deterministic computation.
- topK <= 512 on Atlas A2/A3.
