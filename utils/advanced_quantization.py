"""
Advanced quantization utilities including INT4 quantization.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional, Dict, Any, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class INT4Quantizer:
    """
    INT4 quantization using simulated quantization.
    PyTorch doesn't natively support INT4, so we simulate it.
    """
    
    @staticmethod
    def quantize_tensor_int4(tensor: torch.Tensor, symmetric: bool = True) -> Tuple[torch.Tensor, float, int]:
        """
        Quantize tensor to INT4 range [-8, 7] or [0, 15].
        
        Args:
            tensor: Input tensor
            symmetric: Use symmetric quantization
            
        Returns:
            Tuple of (quantized tensor as int8, scale, zero_point)
        """
        if symmetric:
            # Symmetric: [-8, 7]
            qmin, qmax = -8, 7
            
            # Calculate scale
            max_val = torch.max(torch.abs(tensor))
            scale = max_val / 7.0
            
            if scale == 0:
                scale = 1.0
            
            # Quantize
            quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
            zero_point = 0
            
        else:
            # Asymmetric: [0, 15]
            qmin, qmax = 0, 15
            
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            
            scale = (max_val - min_val) / 15.0
            if scale == 0:
                scale = 1.0
            
            zero_point = qmin - torch.round(min_val / scale)
            zero_point = int(torch.clamp(zero_point, qmin, qmax).item())
            
            quantized = torch.clamp(
                torch.round(tensor / scale) + zero_point,
                qmin, qmax
            )
        
        return quantized.to(torch.int8), scale.item(), zero_point
    
    @staticmethod
    def dequantize_tensor_int4(quantized: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        """
        Dequantize INT4 tensor back to float.
        
        Args:
            quantized: Quantized tensor (stored as int8)
            scale: Quantization scale
            zero_point: Zero point
            
        Returns:
            Dequantized float tensor
        """
        return (quantized.float() - zero_point) * scale
    
    @staticmethod
    def quantize_model_int4(model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply INT4 quantization to model weights.
        Note: This simulates INT4 using INT8 storage.
        
        Args:
            model: Model to quantize
            
        Returns:
            Tuple of (quantized model, quantization info)
        """
        logger.info("Applying INT4 quantization to model...")
        
        model_copy = type(model)(model.config)
        model_copy.load_state_dict(model.state_dict())
        model_copy.eval()
        
        quant_info = {
            'scales': {},
            'zero_points': {},
            'original_dtypes': {}
        }
        
        # Quantize each parameter
        with torch.no_grad():
            for name, param in model_copy.named_parameters():
                if param.requires_grad and param.dim() > 1:  # Only quantize weight matrices
                    # Store original dtype
                    quant_info['original_dtypes'][name] = str(param.dtype)
                    
                    # Quantize
                    quantized, scale, zero_point = INT4Quantizer.quantize_tensor_int4(
                        param.data, symmetric=True
                    )
                    
                    # Store quantization parameters
                    quant_info['scales'][name] = scale
                    quant_info['zero_points'][name] = zero_point
                    
                    # Replace parameter with dequantized version
                    # (In real deployment, keep quantized and dequantize on-the-fly)
                    dequantized = INT4Quantizer.dequantize_tensor_int4(
                        quantized, scale, zero_point
                    )
                    param.data = dequantized
        
        logger.info(f"✓ Quantized {len(quant_info['scales'])} parameter tensors to INT4")
        
        return model_copy, quant_info


class AdvancedQuantizer:
    """Advanced quantization methods including INT4."""
    
    def __init__(self, model: nn.Module, model_name: str = "model"):
        """
        Initialize quantizer.
        
        Args:
            model: Model to quantize
            model_name: Name for saving/logging
        """
        self.model = model
        self.model_name = model_name
        self.model.eval()
    
    def apply_int8_dynamic(self) -> nn.Module:
        """Apply INT8 dynamic quantization (same as Phase 4)."""
        logger.info(f"Applying INT8 dynamic quantization to {self.model_name}...")
        
        quantized_model = quant.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info("✓ INT8 quantization complete")
        return quantized_model
    
    def apply_int4_simulated(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply simulated INT4 quantization."""
        logger.info(f"Applying INT4 quantization to {self.model_name}...")
        
        quantized_model, quant_info = INT4Quantizer.quantize_model_int4(self.model)
        
        logger.info("✓ INT4 quantization complete")
        return quantized_model, quant_info
    
    def compare_quantizations(
        self,
        int8_model: nn.Module,
        int4_model: nn.Module,
        sample_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Compare INT8 vs INT4 quantization.
        
        Args:
            int8_model: INT8 quantized model
            int4_model: INT4 quantized model
            sample_input: Sample input for testing
            num_runs: Number of benchmark runs
            
        Returns:
            Comparison results
        """
        import time
        import numpy as np
        
        logger.info("Comparing quantization methods...")
        
        results = {
            'original': {},
            'int8': {},
            'int4': {}
        }
        
        # Original model
        self.model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = self.model(sample_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                output_orig = self.model(sample_input)
                times.append(time.perf_counter() - start)
            
            if isinstance(output_orig, tuple):
                output_orig = output_orig[0]
            
            results['original'] = {
                'mean_latency_ms': np.mean(times) * 1000,
                'std_latency_ms': np.std(times) * 1000,
                'model_size_mb': self._get_model_size(self.model)
            }
        
        # INT8 model
        int8_model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = int8_model(sample_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                output_int8 = int8_model(sample_input)
                times.append(time.perf_counter() - start)
            
            if isinstance(output_int8, tuple):
                output_int8 = output_int8[0]
            
            # Calculate accuracy
            max_diff = torch.max(torch.abs(output_orig - output_int8)).item()
            mean_diff = torch.mean(torch.abs(output_orig - output_int8)).item()
            
            results['int8'] = {
                'mean_latency_ms': np.mean(times) * 1000,
                'std_latency_ms': np.std(times) * 1000,
                'model_size_mb': self._get_model_size(int8_model),
                'max_output_diff': max_diff,
                'mean_output_diff': mean_diff,
                'speedup': results['original']['mean_latency_ms'] / (np.mean(times) * 1000),
                'compression_ratio': results['original']['model_size_mb'] / self._get_model_size(int8_model)
            }
        
        # INT4 model
        int4_model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = int4_model(sample_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                output_int4 = int4_model(sample_input)
                times.append(time.perf_counter() - start)
            
            if isinstance(output_int4, tuple):
                output_int4 = output_int4[0]
            
            # Calculate accuracy
            max_diff = torch.max(torch.abs(output_orig - output_int4)).item()
            mean_diff = torch.mean(torch.abs(output_orig - output_int4)).item()
            
            results['int4'] = {
                'mean_latency_ms': np.mean(times) * 1000,
                'std_latency_ms': np.std(times) * 1000,
                'model_size_mb': self._get_model_size(int4_model),
                'max_output_diff': max_diff,
                'mean_output_diff': mean_diff,
                'speedup': results['original']['mean_latency_ms'] / (np.mean(times) * 1000),
                'compression_ratio': results['original']['model_size_mb'] / self._get_model_size(int4_model)
            }
        
        # Log results
        logger.info("\nQuantization Comparison:")
        logger.info(f"  Original: {results['original']['model_size_mb']:.2f} MB, "
                   f"{results['original']['mean_latency_ms']:.2f} ms")
        logger.info(f"  INT8: {results['int8']['model_size_mb']:.2f} MB, "
                   f"{results['int8']['mean_latency_ms']:.2f} ms, "
                   f"{results['int8']['compression_ratio']:.2f}x compression, "
                   f"{results['int8']['speedup']:.2f}x speedup")
        logger.info(f"  INT4: {results['int4']['model_size_mb']:.2f} MB, "
                   f"{results['int4']['mean_latency_ms']:.2f} ms, "
                   f"{results['int4']['compression_ratio']:.2f}x compression, "
                   f"{results['int4']['speedup']:.2f}x speedup")
        
        return results
    
    @staticmethod
    def _get_model_size(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def save_quantized_model(
        self,
        model: nn.Module,
        save_path: str,
        quant_info: Optional[Dict[str, Any]] = None
    ):
        """
        Save quantized model and metadata.
        
        Args:
            model: Quantized model
            save_path: Path to save model
            quant_info: Quantization metadata
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.config if hasattr(model, 'config') else None,
            'quantization_info': quant_info
        }, save_path)
        
        logger.info(f"✓ Saved quantized model to {save_path}")
        
        # Save metadata separately
        if quant_info:
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                # Convert non-serializable values
                serializable_info = {}
                for key, value in quant_info.items():
                    if isinstance(value, dict):
                        serializable_info[key] = {k: float(v) if isinstance(v, (int, float)) else str(v) 
                                                 for k, v in value.items()}
                    else:
                        serializable_info[key] = value
                
                json.dump(serializable_info, f, indent=2)
            
            logger.info(f"✓ Saved quantization metadata to {metadata_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Advanced quantization utilities ready")
