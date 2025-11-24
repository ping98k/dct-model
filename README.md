# DCT-Based Linear Layer for Parameter Reduction

## Overview

This project implements a **DCT (Discrete Cosine Transform) factored linear layer** as a parameter-efficient alternative to standard linear layers in neural networks. The approach achieves significant parameter reduction while maintaining comparable performance on classification tasks.

## Key Concept

### Standard Linear Layer
A standard linear layer with input size N×N requires **N² + 1** parameters (weights + bias).

### DCT Factored Linear Layer
The `DCTLinearFactored` module reduces parameters by:
1. **Transforming input to frequency domain** using 2D DCT basis functions
2. **Factoring the weight matrix** as an outer product of two 1D vectors (horizontal and vertical)
3. **Learning only 2N + 1 parameters** instead of N² + 1

For example, with 28×28 input:
- Standard: **785 parameters**
- DCT Factored: **57 parameters** (7.3% of standard)

## How It Works

### DCT Basis Functions
The model precomputes 1D DCT basis vectors:
```
DCT_basis[u, x] = c(u) * cos((2x + 1) * u * π / (2N))
```
where `c(u) = √(1/N)` for u=0, else `√(2/N)`

### 2D Transform
Input is reshaped to 2D grid and projected onto separable 2D DCT basis:
```
DCT_coeff[i,j] = Σ input[x,y] * basis_1d[i,x] * basis_1d[j,y]
```

### Factored Weighting
Instead of learning a full N×N weight matrix, only two 1D vectors are learned:
```
weight[i,j] = w_horizontal[i] * w_vertical[j]
output = Σ weight[i,j] * DCT_coeff[i,j] + bias
```

This rank-1 factorization drastically reduces parameters while operating in the frequency domain where natural signals are often sparse.

## Architecture

```python
class DCTLinearFactored(nn.Module):
    def __init__(self, input_size=28):
        # Precompute DCT basis (not trainable)
        self.register_buffer('basis_1d', ...)
        
        # Learnable parameters (2N + 1)
        self.w_horizontal = nn.Parameter(torch.randn(input_size))
        self.w_vertical = nn.Parameter(torch.randn(input_size))
        self.bias = nn.Parameter(torch.zeros(1))
```

## Demo: Breast Cancer Classification

The notebook demonstrates the model on the Wisconsin Breast Cancer dataset:
- **30 features** padded to 36 (6×6 grid)
- Binary classification (malignant vs benign)
- Comparison between standard linear and DCT factored models

### Results
Both models achieve similar accuracy (~95-98%) with:
- **Standard**: 37 parameters
- **DCT Factored**: 13 parameters (65% reduction)

## Advantages

1. **Parameter Efficiency**: 93%+ reduction for large inputs
2. **Frequency Domain Learning**: Captures patterns in transform space
3. **Interpretable**: Weights show which frequency components matter

## Limitations

1. **Rank-1 Constraint**: Weight matrix limited to outer product structure
2. **2D Input Assumption**: Requires reshaping 1D features into square grids
3. **Computational Overhead**: Double loop in forward pass (can be optimized)


## Files

- `play_01_simple.ipynb`: Main implementation and breast cancer classification demo
- `play_00_baseline.ipynb`: Baseline experiments
- `play_00_baseline_pytorch.ipynb`: PyTorch baseline reference
