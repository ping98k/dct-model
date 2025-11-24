import numpy as np
import torch
import torch.nn as nn


class DCTLinearFactored(nn.Module):
    """
    A parameter-efficient linear layer using 2D DCT (Discrete Cosine Transform) basis.
    
    This layer reduces parameters from N² to 2N by:
    1. Transforming input to frequency domain using DCT
    2. Factoring the weight matrix as an outer product of two 1D vectors
    
    Args:
        input_size (int): Size of one dimension (assumes square input of size input_size × input_size)
    """
    def __init__(self, input_size=28):
        super().__init__()
        self.input_size = input_size
        
        # Precompute 1D DCT basis functions
        x = np.arange(input_size).reshape(-1, 1)
        u = np.arange(input_size).reshape(1, -1)
        cu = np.where(u == 0, np.sqrt(1/input_size), np.sqrt(2/input_size))
        cos_matrix = np.cos((2*x + 1) * u * np.pi / (2*input_size))
        self.register_buffer('basis_1d', torch.from_numpy((cu * cos_matrix).T).float())
        
        # Learnable parameters: 2N + 1 instead of N² + 1
        self.w_horizontal = nn.Parameter(torch.randn(input_size))
        self.w_vertical = nn.Parameter(torch.randn(input_size))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass through DCT factored layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_size²)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        batch_size = x.size(0)
        x_2d = x.view(batch_size, self.input_size, self.input_size)
        
        result = torch.zeros(batch_size, device=x.device)
        
        # Compute weighted sum of DCT coefficients
        for i in range(self.input_size):
            for j in range(self.input_size):
                basis_2d = torch.outer(self.basis_1d[i], self.basis_1d[j])
                dct_coeff = (x_2d * basis_2d.unsqueeze(0)).sum(dim=[1, 2])
                weight = self.w_horizontal[i] * self.w_vertical[j]
                result += weight * dct_coeff
        
        return torch.sigmoid(result + self.bias).unsqueeze(1)
