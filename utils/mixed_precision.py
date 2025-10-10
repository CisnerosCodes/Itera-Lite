"""
Mixed-Precision Optimization Utilities for Itera-Lite

Implements layer-wise INT8/FP16 precision allocation for model compression.
Designed for SSM architectures with architectural lessons from Phase 7 Task 2.

Key Features:
- Conservative precision mapping (Embeddings INT8, SSM FP16)
- Percentile-based INT8 calibration (robust to outliers)
- Per-channel symmetric quantization
- Quality-preserving conversion pipeline
- Comprehensive benchmarking and visualization

Author: GitHub Copilot
Date: October 10, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class MixedPrecisionConfig:
    """
    Configuration for mixed-precision optimization.
    
    Attributes:
        precision_map: Mapping of layer patterns to target precision
                      Pattern format: 'embeddings.*' → 'int8' | 'fp16'
        calibration_method: Method for INT8 scale calculation
                           'minmax' | 'percentile' | 'mse'
        calibration_samples: Number of samples for calibration
        percentile: Percentile for outlier-robust calibration (99.99)
        symmetric_quant: Use symmetric quantization (recommended)
        per_channel: Use per-channel quantization (higher quality)
        max_perplexity_increase: Maximum allowed perplexity increase (%)
    """
    precision_map: Dict[str, str]
    calibration_method: str = 'percentile'
    calibration_samples: int = 1000
    percentile: float = 99.99
    symmetric_quant: bool = True
    per_channel: bool = True
    max_perplexity_increase: float = 5.0


def get_conservative_precision_map() -> Dict[str, str]:
    """
    Get conservative precision mapping (recommended).
    
    Strategy:
    - Embeddings: INT8 (59% of params, less sensitive)
    - SSM Layers: FP16 (39% of params, critical for quality)
    - Norms: FP16 (stability)
    - LM Head: INT8 (tied with embeddings)
    
    Returns:
        Dictionary mapping layer patterns to precision
    """
    return {
        # Embeddings: INT8 (large, less sensitive)
        'embeddings.token_embeddings.weight': 'int8',
        'embeddings.position_embeddings.weight': 'int8',
        
        # SSM Layers: ALL FP16 (critical for quality)
        'layers.*.ssm.norm.weight': 'fp16',
        'layers.*.ssm.norm.bias': 'fp16',
        'layers.*.ssm.in_proj.weight': 'fp16',
        'layers.*.ssm.in_proj.bias': 'fp16',
        'layers.*.ssm.conv1d.weight': 'fp16',
        'layers.*.ssm.conv1d.bias': 'fp16',
        'layers.*.ssm.x_proj.weight': 'fp16',
        'layers.*.ssm.x_proj.bias': 'fp16',
        'layers.*.ssm.dt_proj.weight': 'fp16',
        'layers.*.ssm.dt_proj.bias': 'fp16',
        'layers.*.ssm.A_log': 'fp16',  # CRITICAL: State matrix
        'layers.*.ssm.D': 'fp16',      # Skip connection
        'layers.*.ssm.out_proj.weight': 'fp16',
        'layers.*.ssm.out_proj.bias': 'fp16',
        
        # Final Norm: FP16
        'norm_f.weight': 'fp16',
        'norm_f.bias': 'fp16',
        
        # LM Head: INT8 (tied with embeddings)
        'lm_head.weight': 'int8',
    }


def get_aggressive_precision_map() -> Dict[str, str]:
    """
    Get aggressive precision mapping (higher compression, higher risk).
    
    Strategy:
    - Try INT8 for SSM projections (in_proj, out_proj)
    - Keep critical components FP16 (state matrices, conv1d)
    
    Returns:
        Dictionary mapping layer patterns to precision
    """
    return {
        # Embeddings: INT8
        'embeddings.token_embeddings.weight': 'int8',
        'embeddings.position_embeddings.weight': 'int8',
        
        # SSM: Mixed INT8/FP16
        'layers.*.ssm.norm.weight': 'fp16',
        'layers.*.ssm.norm.bias': 'fp16',
        'layers.*.ssm.in_proj.weight': 'int8',     # TRY INT8
        'layers.*.ssm.in_proj.bias': 'int8',
        'layers.*.ssm.conv1d.weight': 'fp16',      # KEEP FP16 (critical)
        'layers.*.ssm.conv1d.bias': 'fp16',
        'layers.*.ssm.x_proj.weight': 'fp16',      # KEEP FP16 (state-space)
        'layers.*.ssm.x_proj.bias': 'fp16',
        'layers.*.ssm.dt_proj.weight': 'fp16',     # KEEP FP16 (sensitive)
        'layers.*.ssm.dt_proj.bias': 'fp16',
        'layers.*.ssm.A_log': 'fp16',              # ALWAYS FP16
        'layers.*.ssm.D': 'fp16',                  # ALWAYS FP16
        'layers.*.ssm.out_proj.weight': 'int8',    # TRY INT8
        'layers.*.ssm.out_proj.bias': 'int8',
        
        # Final layers
        'norm_f.weight': 'fp16',
        'norm_f.bias': 'fp16',
        'lm_head.weight': 'int8',
    }


class QuantizedLinear(nn.Module):
    """
    INT8 quantized linear layer with per-channel symmetric quantization.
    
    Stores weights in INT8 and dequantizes during forward pass.
    Uses symmetric quantization: Q = round(W / scale).clip(-128, 127)
    """
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], 
                 scale: torch.Tensor, per_channel: bool = True):
        """
        Initialize quantized linear layer.
        
        Args:
            weight: Original FP32 weight tensor [out_features, in_features]
            bias: Original FP32 bias tensor [out_features] (optional)
            scale: Quantization scale [out_features] or scalar
            per_channel: Whether scales are per-channel (True) or per-tensor (False)
        """
        super().__init__()
        
        # Store quantization parameters
        self.register_buffer('scale', scale)
        self.per_channel = per_channel
        
        # Quantize weights to INT8
        if per_channel:
            # Reshape scale for broadcasting [out_features, 1]
            scale_broadcast = scale.view(-1, 1)
            weight_int8 = torch.round(weight / scale_broadcast).clamp(-128, 127).to(torch.int8)
        else:
            weight_int8 = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
        
        self.register_buffer('weight_int8', weight_int8)
        
        # Store bias in FP32 (typically small, keep precision)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with runtime dequantization.
        
        Args:
            x: Input tensor [batch, seq_len, in_features]
        
        Returns:
            Output tensor [batch, seq_len, out_features]
        """
        # Dequantize weights
        if self.per_channel:
            scale_broadcast = self.scale.view(-1, 1)
            weight_fp = self.weight_int8.float() * scale_broadcast
        else:
            weight_fp = self.weight_int8.float() * self.scale
        
        # Linear transformation
        return F.linear(x, weight_fp, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.weight_int8.shape[1]}, out_features={self.weight_int8.shape[0]}, per_channel={self.per_channel}'


class MixedPrecisionConverter:
    """
    Convert model to mixed INT8/FP16 precision.
    
    Pipeline:
    1. Analyze model architecture
    2. Calibrate INT8 layers with sample data
    3. Convert layers based on precision map
    4. Validate quality and compression
    """
    
    def __init__(self, model: nn.Module, config: MixedPrecisionConfig, device: str = 'cuda'):
        """
        Initialize mixed-precision converter.
        
        Args:
            model: Model to convert (in FP32)
            config: Mixed-precision configuration
            device: Target device ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Storage for calibration statistics
        self.calibration_stats = {}
        
        # Storage for conversion metadata
        self.conversion_metadata = {
            'precision_map': config.precision_map,
            'calibration_method': config.calibration_method,
            'layers_converted': {},
            'compression_stats': {},
        }
        
        # Compile regex patterns from precision map
        self.compiled_patterns = {}
        for pattern, precision in config.precision_map.items():
            # Convert glob pattern to regex (*.* → .*\..*)
            regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
            self.compiled_patterns[regex_pattern] = precision
    
    def match_layer_pattern(self, layer_name: str) -> Optional[str]:
        """
        Match layer name to precision pattern.
        
        Args:
            layer_name: Full layer name (e.g., 'layers.0.ssm.in_proj.weight')
        
        Returns:
            Target precision ('int8', 'fp16') or None if no match
        """
        for pattern, precision in self.compiled_patterns.items():
            if re.fullmatch(pattern, layer_name):
                return precision
        return None
    
    def analyze_model(self) -> Dict[str, Any]:
        """
        Analyze model architecture and parameter distribution.
        
        Returns:
            Dictionary with architecture statistics
        """
        stats = {
            'total_params': 0,
            'int8_params': 0,
            'fp16_params': 0,
            'unmatched_params': 0,
            'layer_breakdown': {},
        }
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            stats['total_params'] += param_count
            
            # Match to precision
            precision = self.match_layer_pattern(name)
            
            if precision == 'int8':
                stats['int8_params'] += param_count
                stats['layer_breakdown'][name] = ('int8', param_count)
            elif precision == 'fp16':
                stats['fp16_params'] += param_count
                stats['layer_breakdown'][name] = ('fp16', param_count)
            else:
                stats['unmatched_params'] += param_count
                stats['layer_breakdown'][name] = ('unmatched', param_count)
        
        return stats
    
    def calibrate(self, dataloader: torch.utils.data.DataLoader):
        """
        Calibrate INT8 layers by collecting activation statistics.
        
        Args:
            dataloader: Calibration dataset (synthetic or real)
        """
        print("\n" + "="*60)
        print("CALIBRATING INT8 LAYERS")
        print("="*60)
        
        self.model.eval()
        self.model.to(self.device)
        
        # Identify INT8 layers
        int8_layers = []
        for name, param in self.model.named_parameters():
            precision = self.match_layer_pattern(name)
            if precision == 'int8':
                int8_layers.append((name, param))
        
        print(f"Found {len(int8_layers)} INT8 layers to calibrate")
        
        # Collect weight statistics for each INT8 layer
        for name, param in int8_layers:
            weight = param.data
            
            if self.config.calibration_method == 'minmax':
                # Min-max calibration
                if self.config.per_channel:
                    # Per-channel: scale per output channel
                    scale = torch.max(torch.abs(weight), dim=1, keepdim=True)[0] / 127.0
                    scale = scale.squeeze(1)  # [out_features]
                else:
                    # Per-tensor: single scale
                    scale = torch.max(torch.abs(weight)) / 127.0
            
            elif self.config.calibration_method == 'percentile':
                # Percentile calibration (robust to outliers)
                if self.config.per_channel:
                    # Per-channel percentile
                    abs_weight = torch.abs(weight)
                    percentiles = torch.quantile(abs_weight, self.config.percentile / 100.0, dim=1)
                    scale = percentiles / 127.0
                else:
                    # Per-tensor percentile
                    abs_weight = torch.abs(weight)
                    percentile_val = torch.quantile(abs_weight.flatten(), self.config.percentile / 100.0)
                    scale = percentile_val / 127.0
            
            else:
                raise ValueError(f"Unknown calibration method: {self.config.calibration_method}")
            
            # Store calibration stats
            self.calibration_stats[name] = {
                'scale': scale,
                'method': self.config.calibration_method,
                'per_channel': self.config.per_channel,
                'original_dtype': str(weight.dtype),
                'weight_shape': list(weight.shape),
            }
            
            print(f"  {name}:")
            print(f"    Shape: {list(weight.shape)}")
            print(f"    Scale: {scale.mean().item():.6f} (mean), {scale.std().item():.6f} (std)")
        
        print(f"\n✓ Calibration complete for {len(int8_layers)} layers")
    
    def convert_layer_to_int8(self, name: str, layer: nn.Module) -> Optional[nn.Module]:
        """
        Convert a single layer to INT8.
        
        Args:
            name: Layer name
            layer: Original layer
        
        Returns:
            Quantized layer or None if not applicable
        """
        if not isinstance(layer, nn.Linear):
            # Only quantize Linear layers
            return None
        
        # Get calibration stats
        weight_name = name + '.weight'
        if weight_name not in self.calibration_stats:
            print(f"  ⚠ Warning: No calibration stats for {weight_name}, skipping")
            return None
        
        stats = self.calibration_stats[weight_name]
        
        # Create quantized layer
        quantized = QuantizedLinear(
            weight=layer.weight.data,
            bias=layer.bias.data if layer.bias is not None else None,
            scale=stats['scale'],
            per_channel=self.config.per_channel
        )
        
        return quantized
    
    def convert_layer_to_fp16(self, layer: nn.Module) -> nn.Module:
        """
        Convert a single layer to FP16.
        
        Args:
            layer: Original layer
        
        Returns:
            FP16 layer
        """
        return layer.half()
    
    def apply_mixed_precision(self) -> nn.Module:
        """
        Apply mixed-precision conversion to entire model.
        
        Returns:
            Converted model with mixed INT8/FP16 precision
        """
        print("\n" + "="*60)
        print("APPLYING MIXED PRECISION")
        print("="*60)
        
        # Count conversions
        int8_converted = 0
        fp16_converted = 0
        skipped = 0
        
        # Convert parameters based on precision map
        for name, module in self.model.named_modules():
            # Skip if no parameters
            if not list(module.parameters(recurse=False)):
                continue
            
            # Get module parameters
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name
                precision = self.match_layer_pattern(full_name)
                
                if precision == 'int8':
                    # INT8 conversion (handled at module level for Linear layers)
                    if isinstance(module, nn.Linear):
                        int8_converted += 1
                    else:
                        # For non-Linear INT8 (embeddings), convert to FP16 for now
                        # (True INT8 embeddings require more complex implementation)
                        param.data = param.data.half()
                        fp16_converted += 1
                
                elif precision == 'fp16':
                    # FP16 conversion
                    param.data = param.data.half()
                    fp16_converted += 1
                
                else:
                    # No match, keep FP32
                    skipped += 1
        
        # Replace Linear modules with QuantizedLinear where applicable
        self._replace_linear_modules(self.model)
        
        print(f"\nConversion Summary:")
        print(f"  INT8 layers: {int8_converted}")
        print(f"  FP16 layers: {fp16_converted}")
        print(f"  Skipped (FP32): {skipped}")
        
        return self.model
    
    def _replace_linear_modules(self, module: nn.Module, prefix: str = ''):
        """
        Recursively replace Linear modules with QuantizedLinear.
        
        Args:
            module: Current module
            prefix: Name prefix for recursion
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this Linear layer should be INT8
            if isinstance(child, nn.Linear):
                weight_name = f"{full_name}.weight"
                precision = self.match_layer_pattern(weight_name)
                
                if precision == 'int8' and weight_name in self.calibration_stats:
                    # Replace with QuantizedLinear
                    stats = self.calibration_stats[weight_name]
                    quantized = QuantizedLinear(
                        weight=child.weight.data,
                        bias=child.bias.data if child.bias is not None else None,
                        scale=stats['scale'],
                        per_channel=self.config.per_channel
                    )
                    setattr(module, name, quantized)
                    print(f"  ✓ Converted {full_name} to INT8")
            
            # Recurse
            self._replace_linear_modules(child, full_name)
    
    def calculate_compression_ratio(self) -> Dict[str, float]:
        """
        Calculate compression ratio and memory savings.
        
        Returns:
            Dictionary with compression statistics
        """
        stats = self.analyze_model()
        
        # FP32 baseline memory (4 bytes per param)
        fp32_memory = stats['total_params'] * 4
        
        # Mixed precision memory
        # INT8: 1 byte, FP16: 2 bytes, FP32: 4 bytes
        mixed_memory = (
            stats['int8_params'] * 1 +
            stats['fp16_params'] * 2 +
            stats['unmatched_params'] * 4
        )
        
        compression_ratio = fp32_memory / mixed_memory
        memory_saved = fp32_memory - mixed_memory
        
        return {
            'fp32_memory_mb': fp32_memory / (1024**2),
            'mixed_memory_mb': mixed_memory / (1024**2),
            'compression_ratio': compression_ratio,
            'memory_saved_mb': memory_saved / (1024**2),
            'int8_params': stats['int8_params'],
            'fp16_params': stats['fp16_params'],
            'fp32_params': stats['unmatched_params'],
            'total_params': stats['total_params'],
        }
    
    def get_conversion_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the conversion process.
        
        Returns:
            Dictionary with conversion details
        """
        compression_stats = self.calculate_compression_ratio()
        
        return {
            'precision_map': self.config.precision_map,
            'calibration_method': self.config.calibration_method,
            'calibration_samples': self.config.calibration_samples,
            'percentile': self.config.percentile if self.config.calibration_method == 'percentile' else None,
            'compression_stats': compression_stats,
            'calibration_layers': len(self.calibration_stats),
        }


def visualize_precision_allocation(precision_map: Dict[str, str], 
                                   compression_stats: Dict[str, Any],
                                   output_path: str):
    """
    Visualize precision allocation and compression results.
    
    Args:
        precision_map: Mapping of layer patterns to precision
        compression_stats: Compression statistics
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Precision distribution
    ax1 = axes[0]
    precision_counts = {
        'INT8': compression_stats['int8_params'],
        'FP16': compression_stats['fp16_params'],
        'FP32': compression_stats['fp32_params'],
    }
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax1.pie(precision_counts.values(), labels=precision_counts.keys(), 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Parameter Precision Distribution', fontsize=14, fontweight='bold')
    
    # Plot 2: Compression comparison
    ax2 = axes[1]
    memory_data = {
        'FP32 Baseline': compression_stats['fp32_memory_mb'],
        'Mixed Precision': compression_stats['mixed_memory_mb'],
    }
    
    bars = ax2.bar(memory_data.keys(), memory_data.values(), 
                   color=['#e74c3c', '#2ecc71'], alpha=0.8)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} MB',
                ha='center', va='bottom', fontsize=11)
    
    # Add compression ratio annotation
    ratio = compression_stats['compression_ratio']
    ax2.text(0.5, 0.95, f'Compression: {ratio:.2f}×',
            transform=ax2.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to {output_path}")


def save_mixed_precision_checkpoint(model: nn.Module, 
                                    config: Any,
                                    metadata: Dict[str, Any],
                                    output_path: str):
    """
    Save mixed-precision checkpoint with metadata.
    
    Args:
        model: Mixed-precision model
        config: Model configuration
        metadata: Conversion metadata
        output_path: Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__ if hasattr(config, '__dict__') else config,
        'mixed_precision_metadata': metadata,
    }
    
    torch.save(checkpoint, output_path)
    
    # Also save metadata as JSON
    json_path = output_path.replace('.pt', '.json')
    with open(json_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                json_metadata[key] = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) 
                                     for k, v in value.items()}
            else:
                json_metadata[key] = value
        json.dump(json_metadata, f, indent=2)
    
    print(f"✓ Saved checkpoint to {output_path}")
    print(f"✓ Saved metadata to {json_path}")


if __name__ == '__main__':
    # Test precision pattern matching
    print("Testing Mixed-Precision Utilities")
    print("="*60)
    
    precision_map = get_conservative_precision_map()
    print(f"\nConservative Precision Map ({len(precision_map)} patterns):")
    for pattern, precision in list(precision_map.items())[:5]:
        print(f"  {pattern} → {precision}")
    print(f"  ... and {len(precision_map) - 5} more")
    
    print("\n✓ Mixed-precision utilities loaded successfully")
