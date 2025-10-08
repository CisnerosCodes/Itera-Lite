"""
Configuration classes for Itera-Lite and baseline models
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IteraLiteConfig:
    """Configuration for Itera-Lite SSM+MoE hybrid model"""
    
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 256
    num_layers: int = 6
    
    # SSM (State Space Model) parameters
    ssm_state_size: int = 16
    ssm_conv_kernel: int = 4
    ssm_expand_factor: int = 2
    
    # MoE (Mixture-of-Experts) parameters
    num_experts: int = 8
    expert_size: int = 128
    top_k_experts: int = 2
    moe_layers: list = None  # Which layers use MoE (None = all layers)
    load_balance_loss_weight: float = 0.01
    
    # General parameters
    max_seq_length: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Efficiency settings
    use_flash_attn: bool = False  # Not applicable for SSM, but for comparison
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        if self.moe_layers is None:
            # Apply MoE to every other layer by default
            self.moe_layers = list(range(1, self.num_layers, 2))
    
    @property
    def estimated_params(self) -> int:
        """Estimate total parameter count"""
        # Embedding
        params = self.vocab_size * self.hidden_size
        
        # SSM layers
        ssm_params_per_layer = (
            self.hidden_size * self.hidden_size * self.ssm_expand_factor * 2 +
            self.hidden_size * self.ssm_state_size * 2
        )
        params += ssm_params_per_layer * self.num_layers
        
        # MoE layers
        moe_params_per_layer = (
            self.num_experts * self.expert_size * self.hidden_size * 2 +
            self.hidden_size * self.num_experts  # router
        )
        params += moe_params_per_layer * len(self.moe_layers)
        
        # Output layer
        params += self.vocab_size * self.hidden_size
        
        return params


@dataclass
class TransformerConfig:
    """Configuration for Transformer baseline model"""
    
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    
    # Attention parameters
    max_seq_length: int = 512
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # General parameters
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    # Efficiency settings
    use_flash_attn: bool = False
    gradient_checkpointing: bool = False
    
    @property
    def estimated_params(self) -> int:
        """Estimate total parameter count"""
        # Embedding
        params = self.vocab_size * self.hidden_size
        
        # Transformer layers
        # Attention: Q, K, V, O projections
        attn_params = self.hidden_size * self.hidden_size * 4
        # FFN
        ffn_params = (
            self.hidden_size * self.intermediate_size +
            self.intermediate_size * self.hidden_size
        )
        # Layer norms
        ln_params = self.hidden_size * 4
        
        params += (attn_params + ffn_params + ln_params) * self.num_layers
        
        # Output layer
        params += self.vocab_size * self.hidden_size
        
        return params


# Preset configurations
def get_tiny_config() -> IteraLiteConfig:
    """Ultra-small config for quick testing (~500K params)"""
    return IteraLiteConfig(
        vocab_size=8000,
        hidden_size=128,
        num_layers=4,
        ssm_state_size=8,
        num_experts=4,
        expert_size=64,
        max_seq_length=256,
    )


def get_small_config() -> IteraLiteConfig:
    """Small config for CPU training (~2M params)"""
    return IteraLiteConfig(
        vocab_size=16000,
        hidden_size=256,
        num_layers=6,
        ssm_state_size=16,
        num_experts=8,
        expert_size=128,
        max_seq_length=512,
    )


def get_medium_config() -> IteraLiteConfig:
    """Medium config for GPU training (~10M params)"""
    return IteraLiteConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=8,
        ssm_state_size=32,
        num_experts=16,
        expert_size=256,
        max_seq_length=1024,
    )


def get_transformer_tiny_config() -> TransformerConfig:
    """Tiny Transformer baseline (~500K params)"""
    return TransformerConfig(
        vocab_size=8000,
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_seq_length=256,
    )


def get_transformer_small_config() -> TransformerConfig:
    """Small Transformer baseline (~5M params)"""
    return TransformerConfig(
        vocab_size=16000,
        hidden_size=256,
        num_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
        max_seq_length=512,
    )


def get_micro_config(vocab_size: int = 2000) -> IteraLiteConfig:
    """Micro config for distillation (~100-500K params)"""
    return IteraLiteConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_layers=3,
        ssm_state_size=8,
        ssm_expand_factor=2,
        num_experts=4,
        expert_size=32,
        top_k_experts=2,
        max_seq_length=128,
        dropout=0.1,
    )
