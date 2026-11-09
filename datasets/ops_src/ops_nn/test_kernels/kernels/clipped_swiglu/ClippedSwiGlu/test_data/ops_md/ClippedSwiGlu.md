# Operator Requirement Document

## 1. Operator Name
ClippedSwiGlu

## 2. Target Platform
Ascend 910B NPU

## 3. Input Parameters
| Parameter Name | Input Type | Description | Required | Default |
| :--- | :--- | :--- | :--- | :--- |
| x | Tensor | Input tensor to be split and activated. Shape [N, M] or multi-dimensional. Data type is float16. | Yes | |
| group_index | Tensor | Optional tensor of shape (N,) with dtype int64, used for grouped computation. | No | None |

## 4. Output Results
| Parameter Name | Output Type | Description |
| :--- | :--- | :--- |
| y | Tensor | ClippedSwiGlu output. Shape: same as x except the split dimension is halved. Data type matches x. |

## 5. Functional Description
Clipped Swish-Gated Linear Unit activation. Splits the input tensor x along the specified dimension into two equal halves (A and B), applies clipping, then computes: y = A_sigmoid * B_clipped, where A_sigmoid = A / (1 + exp(-alpha * A)) with clipping, and B_clipped = clip(B, -limit, limit) + bias.

## 6. Computational Formula
$$\text{Split: } x = [A, B] \quad \text{along dim (interleaved or front-back)}$$

$$A_{\text{clip}} = \text{clip}(A, -\text{limit}, \text{limit})$$

$$B_{\text{clip}} = \text{clip}(B, -\text{limit}, \text{limit}) + \text{bias}$$

$$A_{\text{sig}} = \frac{A_{\text{clip}}}{1 + \exp(-\alpha \cdot A_{\text{clip}})}$$

$$y = A_{\text{sig}} \odot B_{\text{clip}}$$

## 7. PyTorch Code Implementation
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.limit = limit
        self.bias = bias
        self.interleaved = interleaved

    def forward(self, x) -> torch.Tensor:
        if self.interleaved:
            # Interleaved: odd/even elements
            A = x[..., 0::2]
            B = x[..., 1::2]
        else:
            # Front-back: first half / second half
            split_size = x.shape[self.dim] // 2
            chunks = torch.chunk(x, 2, dim=self.dim)
            A = chunks[0]
            B = chunks[1]

        A = torch.clamp(A, min=-self.limit, max=self.limit)
        B = torch.clamp(B, min=-self.limit, max=self.limit) + self.bias
        A_sig = A / (1.0 + torch.exp(-self.alpha * A))
        y = A_sig * B
        return y
```

## 8. Constraints
- The split dimension size must be divisible by 2.
- dim must be in range [-rank(x), rank(x)-1].
- Supported dtypes: float16, float32, bfloat16 (bfloat16 not supported on Atlas 310P/Kirin).
- Output shape: same as input except the split dimension is halved.
- group_index is optional, used for grouped computation on Ascend 950.
- If input contains NaN → output is NaN.
