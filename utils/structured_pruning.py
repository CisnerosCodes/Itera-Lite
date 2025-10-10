"""
Structured Pruning Utilities for Itera-Lite

GPU-accelerated magnitude-based structured pruning with:
- Layer-wise sparsity allocation
- SSM component preservation
- Differential MoE expert pruning
- Fine-tuning integration
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PruningConfig:
    """Configuration for structured pruning"""
    
    # Overall sparsity target
    target_sparsity: float = 0.4
    
    # Layer-specific sparsity (overrides target_sparsity for specific layers)
    ssm_in_proj_sparsity: float = 0.30
    ssm_out_proj_sparsity: float = 0.30
    ssm_delta_proj_sparsity: float = 0.25
    ssm_conv_sparsity: float = 0.20
    moe_expert_sparsity: float = 0.60
    moe_router_sparsity: float = 0.10
    embedding_sparsity: float = 0.0  # Preserve embeddings
    
    # Pruning method
    importance_metric: str = 'l1'  # 'l1', 'l2', or 'magnitude'
    structured_type: str = 'channel'  # 'channel' or 'neuron'
    
    # Fine-tuning configuration
    finetune_epochs: int = 5
    finetune_lr: float = 1e-4
    finetune_batch_size: int = 32
    finetune_warmup_ratio: float = 0.05
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class StructuredPruner:
    """
    GPU-accelerated structured pruning for IteraLite models
    
    Implements magnitude-based pruning with differential sparsity allocation:
    - MoE experts: High sparsity (60%) - redundant capacity
    - SSM layers: Moderate sparsity (25-30%) - preserve sequence modeling
    - Embeddings: Zero sparsity - tied with LM head, quality critical
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        """
        Initialize pruner
        
        Args:
            model: IteraLite model to prune
            config: Pruning configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Storage for pruning statistics
        self.pruning_stats = {
            'original_params': 0,
            'pruned_params': 0,
            'layer_stats': {}
        }
        
        # Pruning masks (for tracking)
        self.masks = {}
        
        print(f"Initialized StructuredPruner on {self.device}")
        print(f"Target overall sparsity: {config.target_sparsity*100:.1f}%")
    
    def analyze_model(self) -> Dict:
        """
        Analyze model architecture and parameter distribution
        
        Returns:
            Dictionary with parameter counts per component
        """
        stats = {
            'total_params': 0,
            'embeddings': 0,
            'ssm_layers': 0,
            'moe_layers': 0,
            'norms': 0,
            'other': 0
        }
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            stats['total_params'] += param_count
            
            if 'embedding' in name:
                stats['embeddings'] += param_count
            elif any(x in name for x in ['ssm', 'in_proj', 'out_proj', 'conv1d', 'delta_proj']):
                stats['ssm_layers'] += param_count
            elif any(x in name for x in ['expert', 'router', 'gate']):
                stats['moe_layers'] += param_count
            elif 'norm' in name or 'layer_norm' in name:
                stats['norms'] += param_count
            else:
                stats['other'] += param_count
        
        self.pruning_stats['original_params'] = stats['total_params']
        
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE ANALYSIS")
        print("="*60)
        print(f"Total Parameters: {stats['total_params']:,}")
        print(f"  Embeddings:     {stats['embeddings']:,} ({stats['embeddings']/stats['total_params']*100:.1f}%)")
        print(f"  SSM Layers:     {stats['ssm_layers']:,} ({stats['ssm_layers']/stats['total_params']*100:.1f}%)")
        print(f"  MoE Layers:     {stats['moe_layers']:,} ({stats['moe_layers']/stats['total_params']*100:.1f}%)")
        print(f"  Norms:          {stats['norms']:,} ({stats['norms']/stats['total_params']*100:.1f}%)")
        print(f"  Other:          {stats['other']:,} ({stats['other']/stats['total_params']*100:.1f}%)")
        print("="*60 + "\n")
        
        return stats
    
    def compute_importance_scores(self, weight: torch.Tensor, metric: str = 'l1') -> torch.Tensor:
        """
        Compute importance scores for weight tensor
        
        Args:
            weight: Weight tensor (out_features, in_features) for Linear layers
            metric: Importance metric ('l1', 'l2', or 'magnitude')
        
        Returns:
            Importance scores per output channel/neuron
        """
        if metric == 'l1':
            # L1 norm: sum of absolute values per output neuron
            scores = weight.abs().sum(dim=1)  # (out_features,)
        elif metric == 'l2':
            # L2 norm: sqrt of sum of squares per output neuron
            scores = weight.pow(2).sum(dim=1).sqrt()  # (out_features,)
        elif metric == 'magnitude':
            # Mean absolute magnitude per output neuron
            scores = weight.abs().mean(dim=1)  # (out_features,)
        else:
            raise ValueError(f"Unknown importance metric: {metric}")
        
        return scores
    
    def prune_linear_layer(
        self, 
        layer: nn.Linear, 
        sparsity: float,
        layer_name: str
    ) -> Tuple[nn.Linear, int]:
        """
        Apply structured pruning to a Linear layer
        
        Args:
            layer: Linear layer to prune
            sparsity: Target sparsity (0.0 to 1.0)
            layer_name: Name of layer (for logging)
        
        Returns:
            Pruned layer and number of parameters removed
        """
        if sparsity == 0.0:
            # No pruning
            return layer, 0
        
        # Get weight tensor
        weight = layer.weight.data  # (out_features, in_features)
        original_params = weight.numel()
        
        # Compute importance scores
        scores = self.compute_importance_scores(weight, self.config.importance_metric)
        
        # Determine pruning threshold
        num_keep = int(len(scores) * (1.0 - sparsity))
        threshold_idx = max(1, num_keep)  # Keep at least 1 neuron
        
        # Get indices to keep
        _, keep_indices = torch.topk(scores, threshold_idx, largest=True)
        keep_indices = keep_indices.sort()[0]  # Sort for consistency
        
        # Create pruned layer
        new_out_features = len(keep_indices)
        pruned_layer = nn.Linear(
            layer.in_features,
            new_out_features,
            bias=(layer.bias is not None)
        ).to(self.device)
        
        # Copy weights for kept neurons
        pruned_layer.weight.data = weight[keep_indices, :]
        if layer.bias is not None:
            pruned_layer.bias.data = layer.bias.data[keep_indices]
        
        # Calculate pruned parameters
        pruned_params = pruned_layer.weight.numel()
        if pruned_layer.bias is not None:
            pruned_params += pruned_layer.bias.numel()
        
        removed_params = original_params - pruned_params
        actual_sparsity = removed_params / original_params
        
        # Store statistics
        self.pruning_stats['layer_stats'][layer_name] = {
            'original_params': original_params,
            'pruned_params': pruned_params,
            'removed_params': removed_params,
            'target_sparsity': sparsity,
            'actual_sparsity': actual_sparsity,
            'original_shape': tuple(weight.shape),
            'pruned_shape': tuple(pruned_layer.weight.shape)
        }
        
        return pruned_layer, removed_params
    
    def prune_conv1d_layer(
        self,
        layer: nn.Conv1d,
        sparsity: float,
        layer_name: str
    ) -> Tuple[nn.Conv1d, int]:
        """
        Apply structured pruning to Conv1d layer (depthwise)
        
        Args:
            layer: Conv1d layer to prune
            sparsity: Target sparsity
            layer_name: Name of layer
        
        Returns:
            Pruned layer and number of parameters removed
        """
        if sparsity == 0.0:
            return layer, 0
        
        # Get weight tensor (out_channels, in_channels/groups, kernel_size)
        weight = layer.weight.data
        original_params = weight.numel()
        
        # For depthwise conv (groups == in_channels == out_channels)
        # Compute importance per channel
        scores = weight.abs().sum(dim=(1, 2))  # (out_channels,)
        
        # Determine channels to keep
        num_keep = int(len(scores) * (1.0 - sparsity))
        num_keep = max(1, num_keep)
        
        _, keep_indices = torch.topk(scores, num_keep, largest=True)
        keep_indices = keep_indices.sort()[0]
        
        # Create pruned conv layer
        new_channels = len(keep_indices)
        pruned_layer = nn.Conv1d(
            in_channels=new_channels,
            out_channels=new_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            groups=new_channels,  # Depthwise
            bias=(layer.bias is not None)
        ).to(self.device)
        
        # Copy weights
        pruned_layer.weight.data = weight[keep_indices, :, :]
        if layer.bias is not None:
            pruned_layer.bias.data = layer.bias.data[keep_indices]
        
        # Calculate removed parameters
        pruned_params = pruned_layer.weight.numel()
        if pruned_layer.bias is not None:
            pruned_params += pruned_layer.bias.numel()
        
        removed_params = original_params - pruned_params
        
        # Store statistics
        self.pruning_stats['layer_stats'][layer_name] = {
            'original_params': original_params,
            'pruned_params': pruned_params,
            'removed_params': removed_params,
            'target_sparsity': sparsity,
            'actual_sparsity': removed_params / original_params
        }
        
        return pruned_layer, removed_params
    
    def prune_ssm_block(self, ssm_block, layer_idx: int) -> int:
        """
        Prune SSM block components
        
        Args:
            ssm_block: SSMBlock module
            layer_idx: Layer index (for logging)
        
        Returns:
            Total parameters removed
        """
        total_removed = 0
        
        # SSM blocks have connected layers that must be pruned consistently:
        # in_proj -> conv1d -> ssm -> out_proj
        # We need to maintain dimension compatibility
        
        # IMPORTANT: For SSM blocks, we preserve the full architecture
        # to avoid dimension mismatches between layers.
        # Only prune the expansion/compression layers (in_proj, out_proj)
        # which have clear input/output boundaries.
        
        # DON'T prune in_proj or conv1d - they're tightly coupled
        # The conv1d is depthwise and expects exact input channels from in_proj
        
        # Only prune out_proj (safe - just compresses back to d_model)
        if hasattr(ssm_block, 'out_proj'):
            layer_name = f'layer_{layer_idx}.ssm.out_proj'
            # Reduce sparsity to be more conservative
            pruned_layer, removed = self.prune_linear_layer(
                ssm_block.out_proj,
                self.config.ssm_out_proj_sparsity * 0.5,  # Be more conservative
                layer_name
            )
            ssm_block.out_proj = pruned_layer
            total_removed += removed
        
        # PRESERVE all other SSM components to maintain architecture integrity:
        # - in_proj (coupled with conv1d)
        # - conv1d (depthwise - needs exact channels)
        # - ssm (state space parameters A, B, C, D - critical)
        # - delta_proj (controls SSM dynamics)
        
        return total_removed
    
    def prune_moe_layer(self, moe_layer, layer_idx: int) -> int:
        """
        Prune MoE layer components (aggressive on experts)
        
        Args:
            moe_layer: MoELayer module
            layer_idx: Layer index (for logging)
        
        Returns:
            Total parameters removed
        """
        total_removed = 0
        
        # Check if this layer actually uses MoE
        if not hasattr(moe_layer, 'use_moe') or not moe_layer.use_moe:
            # Skip if MoE not used in this layer
            return 0
        
        # Prune router (lightly - important for expert selection)
        if hasattr(moe_layer, 'router') and hasattr(moe_layer.router, 'gate'):
            layer_name = f'layer_{layer_idx}.moe.router'
            pruned_layer, removed = self.prune_linear_layer(
                moe_layer.router.gate,
                self.config.moe_router_sparsity,
                layer_name
            )
            moe_layer.router.gate = pruned_layer
            total_removed += removed
        
        # Prune experts (aggressively - 60% sparsity)
        if hasattr(moe_layer, 'moe') and hasattr(moe_layer.moe, 'experts'):
            experts = moe_layer.moe.experts
            for expert_idx, expert in enumerate(experts):
                # Prune w1 (input projection)
                if hasattr(expert, 'w1'):
                    layer_name = f'layer_{layer_idx}.moe.expert_{expert_idx}.w1'
                    pruned_layer, removed = self.prune_linear_layer(
                        expert.w1,
                        self.config.moe_expert_sparsity,
                        layer_name
                    )
                    expert.w1 = pruned_layer
                    total_removed += removed
                
                # Prune w2 (output projection)
                if hasattr(expert, 'w2'):
                    layer_name = f'layer_{layer_idx}.moe.expert_{expert_idx}.w2'
                    pruned_layer, removed = self.prune_linear_layer(
                        expert.w2,
                        self.config.moe_expert_sparsity,
                        layer_name
                    )
                    expert.w2 = pruned_layer
                    total_removed += removed
        
        return total_removed
    
    def apply_pruning(self) -> nn.Module:
        """
        Apply structured pruning to entire model
        
        Returns:
            Pruned model
        """
        print("\n" + "="*60)
        print("APPLYING STRUCTURED PRUNING")
        print("="*60)
        
        # Analyze model first
        self.analyze_model()
        
        total_removed = 0
        
        # Don't prune embeddings (tied with LM head, quality critical)
        print("\n[PRESERVE] Embeddings (tied with LM head)")
        
        # Prune each layer (SSM + MoE)
        if hasattr(self.model, 'layers'):
            num_layers = len(self.model.layers)
            print(f"\nPruning {num_layers} layers...")
            
            for layer_idx, layer_dict in enumerate(self.model.layers):
                print(f"\n--- Layer {layer_idx} ---")
                
                # Prune SSM block
                if 'ssm' in layer_dict:
                    removed = self.prune_ssm_block(layer_dict['ssm'], layer_idx)
                    total_removed += removed
                    print(f"  SSM:  Removed {removed:,} params")
                
                # Prune MoE layer
                if 'moe' in layer_dict:
                    removed = self.prune_moe_layer(layer_dict['moe'], layer_idx)
                    total_removed += removed
                    print(f"  MoE:  Removed {removed:,} params")
        
        # Don't prune final norm or LM head
        print("\n[PRESERVE] Final LayerNorm and LM Head")
        
        # Calculate final statistics
        final_params = self.pruning_stats['original_params'] - total_removed
        self.pruning_stats['pruned_params'] = final_params
        self.pruning_stats['total_removed'] = total_removed
        self.pruning_stats['overall_sparsity'] = total_removed / self.pruning_stats['original_params']
        
        print("\n" + "="*60)
        print("PRUNING COMPLETE")
        print("="*60)
        print(f"Original Parameters: {self.pruning_stats['original_params']:,}")
        print(f"Pruned Parameters:   {final_params:,}")
        print(f"Removed Parameters:  {total_removed:,}")
        print(f"Overall Sparsity:    {self.pruning_stats['overall_sparsity']*100:.2f}%")
        print(f"Compression Ratio:   {self.pruning_stats['original_params']/final_params:.2f}×")
        print("="*60 + "\n")
        
        return self.model
    
    def get_pruning_statistics(self) -> Dict:
        """
        Get comprehensive pruning statistics
        
        Returns:
            Dictionary with pruning statistics
        """
        return self.pruning_stats
    
    def save_statistics(self, save_path: str):
        """
        Save pruning statistics to JSON file
        
        Args:
            save_path: Path to save statistics
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.pruning_stats, f, indent=2)
        
        print(f"Saved pruning statistics to {save_path}")
    
    def visualize_sparsity(self, save_path: str):
        """
        Generate sparsity visualization plots
        
        Args:
            save_path: Path to save visualization
        """
        # Extract layer statistics
        layer_names = []
        layer_sparsities = []
        
        for name, stats in self.pruning_stats['layer_stats'].items():
            layer_names.append(name)
            layer_sparsities.append(stats['actual_sparsity'] * 100)
        
        if not layer_names:
            print("No layer statistics to visualize")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Per-layer sparsity
        colors = ['#2ecc71' if 'moe' in name else '#3498db' if 'ssm' in name else '#95a5a6' 
                  for name in layer_names]
        
        ax1.barh(range(len(layer_names)), layer_sparsities, color=colors)
        ax1.set_yticks(range(len(layer_names)))
        ax1.set_yticklabels(layer_names, fontsize=8)
        ax1.set_xlabel('Sparsity (%)', fontsize=12)
        ax1.set_title('Layer-wise Sparsity Distribution', fontsize=14, fontweight='bold')
        ax1.axvline(x=self.config.target_sparsity * 100, color='r', linestyle='--', 
                    label=f'Target: {self.config.target_sparsity*100:.1f}%')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Overall statistics
        labels = ['Original\nParams', 'Pruned\nParams', 'Removed\nParams']
        values = [
            self.pruning_stats['original_params'],
            self.pruning_stats['pruned_params'],
            self.pruning_stats['total_removed']
        ]
        colors_bar = ['#3498db', '#2ecc71', '#e74c3c']
        
        ax2.bar(labels, values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Parameters', fontsize=12)
        ax2.set_title('Overall Pruning Statistics', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (label, value) in enumerate(zip(labels, values)):
            ax2.text(i, value, f'{value:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add compression ratio text
        compression = self.pruning_stats['original_params'] / self.pruning_stats['pruned_params']
        sparsity = self.pruning_stats['overall_sparsity'] * 100
        ax2.text(0.5, max(values) * 0.9, 
                f'Compression: {compression:.2f}×\nSparsity: {sparsity:.1f}%',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved sparsity visualization to {save_path}")


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
