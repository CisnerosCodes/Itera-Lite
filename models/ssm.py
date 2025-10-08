"""
State Space Model (SSM) implementation for Itera-Lite

This is a simplified SSM inspired by Mamba/S4 architectures,
optimized for efficiency and ease of understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class S4Kernel(nn.Module):
    """
    Simplified S4 kernel for state space modeling
    Based on Structured State Space (S4) paper
    """
    
    def __init__(self, d_model: int, state_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        
        # State space parameters
        # A: state transition matrix (N x N)
        # Initialized with HiPPO structure (approximation)
        self.A_log = nn.Parameter(torch.randn(state_size))
        
        # B: input-to-state matrix
        self.B = nn.Parameter(torch.randn(state_size, d_model))
        
        # C: state-to-output matrix
        self.C = nn.Parameter(torch.randn(d_model, state_size))
        
        # D: skip connection
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Learnable step size
        self.delta_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Compute step size (Delta) - simplified to scalar per timestep
        delta = F.softplus(self.delta_proj(x)).mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        
        # Discretize continuous parameters
        A = -torch.exp(self.A_log)  # (state_size,)
        
        # Scan over sequence
        y = self._selective_scan(x, delta, A)
        
        return y
    
    def _selective_scan(self, x, delta, A):
        """
        Perform selective scan over sequence
        Simplified version - not optimized for speed
        """
        batch, seq_len, d_model = x.shape
        
        # Initialize state
        h = torch.zeros(batch, self.state_size, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # Get current input
            x_t = x[:, t, :]  # (batch, d_model)
            dt = delta[:, t, 0]  # (batch,)
            
            # Discretize: dA = exp(dt * A), dB = dt * B
            dA = torch.exp(dt.unsqueeze(-1) * A)  # (batch, state_size)
            
            # Update state: h_t = dA * h_{t-1} + dt * B * x_t
            h = dA * h  # (batch, state_size)
            h = h + dt.unsqueeze(-1) * torch.matmul(x_t, self.B.T)  # (batch, state_size)
            
            # Compute output: y_t = C * h_t + D * x_t
            y_t = torch.matmul(h, self.C.T) + self.D * x_t  # (batch, d_model)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return y


class SSMBlock(nn.Module):
    """
    SSM block with expansion, convolution, and gating
    Similar to Mamba architecture
    """
    
    def __init__(
        self,
        d_model: int,
        state_size: int = 16,
        expand_factor: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        
        # Expansion projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_inner,  # Depthwise convolution
        )
        
        # S4 kernel
        self.ssm = S4Kernel(self.d_inner, state_size)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        
        # Expansion with gating
        x_and_gate = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x, gate = x_and_gate.chunk(2, dim=-1)  # Each (batch, seq_len, d_inner)
        
        # Causal convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = x[..., :gate.shape[1]]  # Remove padding to match original length
        x = rearrange(x, 'b d l -> b l d')
        
        # Activation
        x = F.silu(x)
        
        # SSM
        x = self.ssm(x)
        
        # Gating
        x = x * F.silu(gate)
        
        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)
        
        # Residual connection
        output = residual + x
        
        return output


class SSMBackbone(nn.Module):
    """
    Stack of SSM blocks forming the backbone
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        state_size: int = 16,
        expand_factor: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                state_size=state_size,
                expand_factor=expand_factor,
                conv_kernel=conv_kernel,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x


if __name__ == "__main__":
    # Test SSM components
    print("Testing SSM components...")
    
    batch_size = 2
    seq_len = 64
    d_model = 128
    
    # Test S4 Kernel
    print("\n1. Testing S4 Kernel...")
    s4 = S4Kernel(d_model=d_model, state_size=16)
    x = torch.randn(batch_size, seq_len, d_model)
    y = s4(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in s4.parameters()):,}")
    
    # Test SSM Block
    print("\n2. Testing SSM Block...")
    ssm_block = SSMBlock(d_model=d_model, state_size=16, expand_factor=2)
    y = ssm_block(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in ssm_block.parameters()):,}")
    
    # Test SSM Backbone
    print("\n3. Testing SSM Backbone...")
    backbone = SSMBackbone(d_model=d_model, num_layers=4, state_size=16)
    y = backbone(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in backbone.parameters()):,}")
    
    print("\n✓ All SSM components working!")
