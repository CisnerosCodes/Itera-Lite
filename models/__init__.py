"""
Itera-Lite Model Architecture Package

This package contains the ultra-efficient mini language model implementations:
- SSM (State Space Model) backbone
- MoE (Mixture-of-Experts) layers
- Complete Itera-Lite hybrid model
- Transformer baseline for comparison
"""

from .config import (
    IteraLiteConfig, 
    TransformerConfig,
    get_tiny_config,
    get_small_config,
    get_medium_config,
    get_transformer_tiny_config,
    get_transformer_small_config,
)
from .itera_lite import IteraLiteModel
from .transformer_baseline import TransformerBaseline

__all__ = [
    'IteraLiteConfig',
    'TransformerConfig',
    'IteraLiteModel',
    'TransformerBaseline',
    'get_tiny_config',
    'get_small_config',
    'get_medium_config',
    'get_transformer_tiny_config',
    'get_transformer_small_config',
]
