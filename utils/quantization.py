"""
Model quantization utilities for reducing memory and improving inference speed
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, Tuple
import copy


class ModelQuantizer:
    """Apply quantization to PyTorch models"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_size = self._get_model_size(model)
    
    @staticmethod
    def _get_model_size(model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024
    
    def apply_dynamic_quantization(self, qconfig: str = 'x86') -> nn.Module:
        """
        Apply dynamic quantization (8-bit)
        
        Args:
            qconfig: Quantization config ('x86', 'fbgemm', etc.)
            
        Returns:
            Quantized model
        """
        print(f"\nApplying dynamic quantization ({qconfig})...")
        
        # Prepare model for quantization
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        
        # Apply dynamic quantization to Linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = self.original_size / quantized_size
        
        print(f"  Original size: {self.original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return quantized_model
    
    def apply_static_quantization(
        self,
        calibration_data: torch.Tensor,
        qconfig_spec: Any = None
    ) -> nn.Module:
        """
        Apply static quantization (requires calibration data)
        
        Args:
            calibration_data: Data for calibration
            qconfig_spec: Quantization configuration
            
        Returns:
            Quantized model
        """
        print("\nApplying static quantization...")
        
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        
        # Set quantization config
        if qconfig_spec is None:
            qconfig_spec = torch.quantization.get_default_qconfig('x86')
        
        model_copy.qconfig = qconfig_spec
        
        # Prepare model
        torch.quantization.prepare(model_copy, inplace=True)
        
        # Calibrate with sample data
        print("  Calibrating...")
        with torch.no_grad():
            _ = model_copy(calibration_data)
        
        # Convert to quantized model
        torch.quantization.convert(model_copy, inplace=True)
        
        quantized_size = self._get_model_size(model_copy)
        compression_ratio = self.original_size / quantized_size
        
        print(f"  Original size: {self.original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return model_copy
    
    def benchmark_quantized_model(
        self,
        quantized_model: nn.Module,
        test_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark quantized model performance
        
        Args:
            quantized_model: Quantized model to benchmark
            test_input: Test input tensor
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        print("\nBenchmarking quantized model...")
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = quantized_model(test_input)
        
        # Benchmark original model
        self.model.eval()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(test_input)
        original_time = (time.time() - start) / num_runs
        
        # Benchmark quantized model
        quantized_model.eval()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = quantized_model(test_input)
        quantized_time = (time.time() - start) / num_runs
        
        speedup = original_time / quantized_time
        
        # Size comparison
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(quantized_model)
        
        results = {
            'original_time_ms': original_time * 1000,
            'quantized_time_ms': quantized_time * 1000,
            'speedup': speedup,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Original: {results['original_time_ms']:.2f} ms, {results['original_size_mb']:.2f} MB")
        print(f"  Quantized: {results['quantized_time_ms']:.2f} ms, {results['quantized_size_mb']:.2f} MB")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Compression: {results['compression_ratio']:.2f}x")
        
        return results


def quantize_model_int8(
    model: nn.Module,
    save_path: str = None
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Convenience function to quantize model to INT8
    
    Args:
        model: Model to quantize
        save_path: Optional path to save quantized model
        
    Returns:
        Quantized model and benchmark results
    """
    quantizer = ModelQuantizer(model)
    quantized_model = quantizer.apply_dynamic_quantization()
    
    # Benchmark with dummy input
    batch_size = 4
    seq_len = 128
    vocab_size = getattr(model, 'vocab_size', 2000)
    
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    results = quantizer.benchmark_quantized_model(quantized_model, test_input)
    
    if save_path:
        torch.save(quantized_model.state_dict(), save_path)
        print(f"\nSaved quantized model to {save_path}")
    
    return quantized_model, results


def compare_model_outputs(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_inputs: torch.Tensor,
    num_samples: int = 10
) -> Dict[str, float]:
    """
    Compare outputs between original and quantized models
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        test_inputs: Test input tensors
        num_samples: Number of samples to compare
        
    Returns:
        Dictionary with comparison metrics
    """
    original_model.eval()
    quantized_model.eval()
    
    mse_losses = []
    max_diffs = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_inputs))):
            input_tensor = test_inputs[i:i+1]
            
            # Get outputs
            original_out = original_model(input_tensor)
            quantized_out = quantized_model(input_tensor)
            
            # Calculate differences
            mse = torch.nn.functional.mse_loss(original_out, quantized_out).item()
            max_diff = torch.abs(original_out - quantized_out).max().item()
            
            mse_losses.append(mse)
            max_diffs.append(max_diff)
    
    results = {
        'mean_mse': sum(mse_losses) / len(mse_losses),
        'mean_max_diff': sum(max_diffs) / len(max_diffs),
        'max_mse': max(mse_losses),
        'max_diff': max(max_diffs)
    }
    
    print("\nOutput Comparison:")
    print(f"  Mean MSE: {results['mean_mse']:.6f}")
    print(f"  Mean Max Diff: {results['mean_max_diff']:.6f}")
    print(f"  Max MSE: {results['max_mse']:.6f}")
    print(f"  Max Diff: {results['max_diff']:.6f}")
    
    return results


if __name__ == "__main__":
    print("Testing quantization utilities...")
    
    # Create simple test model
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=2000, d_model=256):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.fc1 = nn.Linear(d_model, d_model * 4)
            self.fc2 = nn.Linear(d_model * 4, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.mean(dim=1)  # Simple pooling
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test quantization
    quantized_model, results = quantize_model_int8(model)
    
    print("\n✓ Quantization test completed!")
