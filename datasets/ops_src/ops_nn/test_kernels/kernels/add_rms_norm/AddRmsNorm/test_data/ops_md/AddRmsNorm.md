# Operator Requirement Document

## 1. Operator Name
AddRmsNorm

## 2. Target Platform
Ascend 910B NPU

## 3. Input Parameters
| Parameter Name | Input Type | Description | Required | Default |
| :--- | :--- | :--- | :--- | :--- |
| x1 | Tensor | First input tensor for addition, shape [N, M] or multi-dimensional. Data type is float16. | Yes | |
| x2 | Tensor | Second input tensor for addition, same shape and dtype as x1. Data type is float16. | Yes | |
| gamma | Tensor | Scaling factor (weights) for RMSNorm, shape matches last N dims of x1. Data type is float16. | Yes | |
| epsilon | float | Epsilon to prevent division-by-zero in RMS computation. Must be >= 0. | No | 1e-6 |

## 4. Output Results
| Parameter Name | Output Type | Description |
| :--- | :--- | :--- |
| y | Tensor | RMSNorm output, same shape and dtype as x1. |
| rstd | Tensor | Reciprocal of RMS, dtype is float32. Shape: same rank as x1, last N dims (matching gamma) are 1. |
| x | Tensor | Intermediate result (x1 + x2), same shape and dtype as x1. |

## 5. Functional Description
RMSNorm operator with fused Add operation. Combines element-wise addition (x1 + x2) with Root Mean Square Normalization, reducing data movement. Compared to LayerNorm, RMSNorm omits the mean subtraction step.

## 6. Computational Formula
$$x = x1 + x2$$

$$\text{RmsNorm}(x) = \frac{x}{\text{Rms}(\mathbf{x})} \odot \gamma, \quad \text{where } \text{Rms}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}$$

$$\text{rstd} = \frac{1}{\text{Rms}(\mathbf{x})}$$

## 7. PyTorch Code Implementation
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x1, x2, gamma) -> tuple:
        x = x1 + x2
        # Compute RMS over last gamma.dim() dimensions
        rms = torch.sqrt(x.pow(2).mean(dim=-gamma.dim(), keepdim=True) + self.epsilon)
        rstd = 1.0 / rms
        y = gamma * (x * rstd)
        return y, rstd, x
```

## 8. Constraints
- x1 and x2 must have identical shape and dtype.
- gamma.shape must match the last N dimensions of x1 (where N = gamma.dim()).
- rstd dtype is always float32.
- Supported dtypes: float16, float32, bfloat16 (bfloat16 not supported on Atlas 310P/Kirin).
- rstd shape: for x1=[B,N,M], gamma=[M] → rstd=[B,N,1]; for gamma=[N,M] → rstd=[B,1,1].
- If input is Inf → output is Inf; if input is NaN → output is NaN.
