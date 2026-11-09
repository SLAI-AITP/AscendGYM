# Operator Requirement Document

## 1. Operator Name
SwiGlu

## 2. Target Platform
Ascend 910B NPU

## 3. Input Parameters
| Parameter Name | Input Type | Description | Required | Default |
| :--- | :--- | :--- | :--- | :--- |
| x | Tensor | Input tensor to be split and activated. Shape [N, M] or multi-dimensional. Data type is float16. | Yes | |
| dim | int | Dimension along which to split the input tensor into two equal halves. Default is -1 (last dimension). | No | -1 |

## 4. Output Results
| Parameter Name | Output Type | Description |
| :--- | :--- | :--- |
| y | Tensor | SwiGlu output. Shape: same as x except the split dimension is halved. Data type matches x. |

## 5. Functional Description
Swish-Gated Linear Unit activation. Splits the input tensor x along the specified dimension into two equal halves, applies the Swish (SiLU) activation function to the first half, and multiplies element-wise with the second half.

## 6. Computational Formula
$$\text{Split: } x = [x_1, x_2] \quad \text{along dim, where } x_1, x_2 \text{ have equal size along dim}$$

$$\text{Swish}(a) = a \cdot \sigma(a) \quad (\text{where } \sigma \text{ is the sigmoid function})$$

$$y = \text{Swish}(x_1) \odot x_2$$

## 7. PyTorch Code Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x) -> torch.Tensor:
        x_chunks = torch.chunk(x, 2, dim=self.dim)
        y = F.silu(x_chunks[0]) * x_chunks[1]
        return y
```

## 8. Constraints
- The split dimension size must be divisible by 2.
- dim must be in range [-rank(x), rank(x)-1].
- Supported dtypes: float16, float32, bfloat16 (bfloat16 not supported on Atlas 310P/Kirin).
- Output shape: same as input except the split dimension is halved.
- If input contains NaN → output is NaN.
