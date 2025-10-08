"""
Mixture-of-Experts (MoE) implementation for Itera-Lite

Sparse MoE with Top-K routing and load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """Single expert network (simple FFN)"""
    
    def __init__(self, d_model: int, d_expert: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_expert, bias=False)
        self.w2 = nn.Linear(d_expert, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        x = self.w1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class Router(nn.Module):
    """Router network for expert selection"""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router weights
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # For load balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            top_k_indices: (batch, seq_len, top_k) - indices of selected experts
            top_k_gates: (batch, seq_len, top_k) - routing weights
            load_balance_loss: scalar - auxiliary loss for load balancing
        """
        batch, seq_len, d_model = x.shape
        
        # Compute routing logits
        router_logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        # Top-K selection
        top_k_gates, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # Each: (batch, seq_len, top_k)
        
        # Normalize gates with softmax
        top_k_gates = F.softmax(top_k_gates, dim=-1)
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(router_logits)
        
        # Track expert usage (for monitoring)
        if self.training:
            with torch.no_grad():
                expert_usage = torch.bincount(
                    top_k_indices.flatten(),
                    minlength=self.num_experts
                ).float()
                self.expert_counts = 0.9 * self.expert_counts + 0.1 * expert_usage
        
        return top_k_indices, top_k_gates, load_balance_loss
    
    def _compute_load_balance_loss(self, router_logits):
        """
        Compute auxiliary load balancing loss
        Encourages uniform expert usage
        """
        # Average routing probability per expert across batch
        routing_probs = F.softmax(router_logits, dim=-1)  # (batch, seq_len, num_experts)
        avg_probs = routing_probs.mean(dim=[0, 1])  # (num_experts,)
        
        # Target uniform distribution
        target = 1.0 / self.num_experts
        
        # Coefficient of variation loss
        loss = (avg_probs.std() / (avg_probs.mean() + 1e-6)) ** 2
        
        return loss


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture-of-Experts layer
    Only activates top-k experts per token for efficiency
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        expert_size: int = 128,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # Router
        self.router = Router(d_model, num_experts, top_k)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, expert_size, dropout)
            for _ in range(num_experts)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: scalar - auxiliary load balancing loss
        """
        residual = x
        x = self.norm(x)
        
        batch, seq_len, d_model = x.shape
        
        # Route to experts
        top_k_indices, top_k_gates, load_balance_loss = self.router(x)
        
        # Prepare output
        output = torch.zeros_like(x)
        
        # Process tokens with each expert
        # For efficiency, we batch process tokens going to the same expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (batch, seq_len)
            
            if not expert_mask.any():
                continue
            
            # Get positions where this expert is used
            batch_idx, seq_idx = torch.where(expert_mask)
            
            # Get the corresponding inputs
            expert_input = x[batch_idx, seq_idx]  # (num_tokens, d_model)
            
            # Process with expert
            expert_output = self.experts[expert_idx](expert_input)  # (num_tokens, d_model)
            
            # Get the gates for this expert
            # Find which position in top_k this expert appears
            is_expert = (top_k_indices[batch_idx, seq_idx] == expert_idx)  # (num_tokens, top_k)
            gates = (top_k_gates[batch_idx, seq_idx] * is_expert.float()).sum(dim=-1, keepdim=True)  # (num_tokens, 1)
            
            # Weight by gate values
            expert_output = expert_output * gates
            
            # Add to output
            output[batch_idx, seq_idx] += expert_output
        
        # Residual connection
        output = residual + output
        
        # Auxiliary loss
        aux_loss = load_balance_loss * self.load_balance_weight
        
        return output, aux_loss
    
    def get_expert_usage_stats(self):
        """Get statistics about expert usage"""
        counts = self.router.expert_counts
        return {
            'expert_counts': counts.cpu().numpy(),
            'max_usage': counts.max().item(),
            'min_usage': counts.min().item(),
            'std_usage': counts.std().item(),
        }


class MoELayer(nn.Module):
    """
    Complete MoE layer with optional FFN fallback
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        expert_size: int = 128,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
        use_moe: bool = True
    ):
        super().__init__()
        self.use_moe = use_moe
        
        if use_moe:
            self.layer = MixtureOfExperts(
                d_model=d_model,
                num_experts=num_experts,
                expert_size=expert_size,
                top_k=top_k,
                dropout=dropout,
                load_balance_weight=load_balance_weight
            )
        else:
            # Fallback to standard FFN
            self.norm = nn.LayerNorm(d_model)
            self.ffn = Expert(d_model, expert_size, dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: scalar (0 if not using MoE)
        """
        if self.use_moe:
            return self.layer(x)
        else:
            residual = x
            x = self.norm(x)
            x = self.ffn(x)
            return residual + x, torch.tensor(0.0, device=x.device)


if __name__ == "__main__":
    # Test MoE components
    print("Testing MoE components...")
    
    batch_size = 2
    seq_len = 64
    d_model = 128
    num_experts = 8
    expert_size = 256
    
    # Test Expert
    print("\n1. Testing Expert...")
    expert = Expert(d_model=d_model, d_expert=expert_size)
    x = torch.randn(batch_size, seq_len, d_model)
    y = expert(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in expert.parameters()):,}")
    
    # Test Router
    print("\n2. Testing Router...")
    router = Router(d_model=d_model, num_experts=num_experts, top_k=2)
    indices, gates, loss = router(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Top-K indices shape: {indices.shape}")
    print(f"   Top-K gates shape: {gates.shape}")
    print(f"   Load balance loss: {loss.item():.6f}")
    
    # Test MixtureOfExperts
    print("\n3. Testing MixtureOfExperts...")
    moe = MixtureOfExperts(
        d_model=d_model,
        num_experts=num_experts,
        expert_size=expert_size,
        top_k=2
    )
    y, aux_loss = moe(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Auxiliary loss: {aux_loss.item():.6f}")
    print(f"   Parameters: {sum(p.numel() for p in moe.parameters()):,}")
    
    # Test expert usage stats
    print("\n4. Testing expert usage tracking...")
    for _ in range(10):
        _, _ = moe(x)
    stats = moe.get_expert_usage_stats()
    print(f"   Expert usage stats:")
    print(f"   - Max: {stats['max_usage']:.1f}")
    print(f"   - Min: {stats['min_usage']:.1f}")
    print(f"   - Std: {stats['std_usage']:.1f}")
    
    print("\n✓ All MoE components working!")
