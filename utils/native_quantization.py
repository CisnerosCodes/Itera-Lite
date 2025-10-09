"""
Native INT4 Quantization Utilities for Phase 7 Task 1
GPU-accelerated quantization using bitsandbytes on NVIDIA A30
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import json
from pathlib import Path
import time
from dataclasses import dataclass, asdict

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Install with: pip install bitsandbytes")


@dataclass
class QuantizationConfig:
    """Configuration for INT4 quantization"""
    bits: int = 4
    use_double_quant: bool = True  # Double quantization for better compression
    quant_type: str = 'nf4'  # 'nf4' (NormalFloat4) or 'fp4' (FP4)
    compute_dtype: str = 'float16'  # Computation dtype (float16/bfloat16)
    calibration_samples: int = 1000
    calibration_batch_size: int = 32
    qat_epochs: int = 0  # Quantization-Aware Training epochs (0 = skip)
    qat_learning_rate: float = 1e-5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_dict(self) -> dict:
        return asdict(self)


class NativeINT4Quantizer:
    """
    Hardware-accelerated INT4 quantization using bitsandbytes
    Optimized for NVIDIA A30 GPU with Ampere Tensor Cores
    """
    
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        """
        Initialize quantizer with model and configuration
        
        Args:
            model: PyTorch model to quantize
            config: QuantizationConfig with quantization parameters
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Check bitsandbytes availability
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes required for INT4 quantization")
        
        # Check CUDA availability for GPU mode
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
            self.config.device = 'cpu'
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Track quantization statistics
        self.stats = {
            'total_params': sum(p.numel() for p in model.parameters()),
            'quantized_params': 0,
            'calibration_time': 0,
            'quantization_time': 0,
            'qat_time': 0
        }
        
        print(f"NativeINT4Quantizer initialized on {self.device}")
        print(f"Total parameters: {self.stats['total_params']:,}")
        print(f"Quantization type: {config.quant_type}")
        print(f"Compute dtype: {config.compute_dtype}")
    
    def _get_linear_layers(self) -> Dict[str, nn.Linear]:
        """Get all Linear layers in the model"""
        linear_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        return linear_layers
    
    def calibrate(self, dataloader, num_batches: Optional[int] = None):
        """
        Calibrate quantization parameters using representative data
        
        Args:
            dataloader: DataLoader with calibration samples
            num_batches: Number of batches to use (None = all)
        
        Returns:
            Dict with calibration statistics
        """
        print("\n" + "="*60)
        print("Starting Calibration")
        print("="*60)
        
        start_time = time.time()
        
        if num_batches is None:
            num_batches = len(dataloader)
        num_batches = min(num_batches, len(dataloader))
        
        # Collect activation statistics
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    inputs = batch['input_ids']
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass to collect statistics
                _ = self.model(inputs)
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"  Calibration batch {batch_idx + 1}/{num_batches}")
        
        self.stats['calibration_time'] = time.time() - start_time
        
        print(f"\n✓ Calibration complete in {self.stats['calibration_time']:.2f}s")
        print(f"  Processed {num_batches} batches")
        
        return {
            'num_batches': num_batches,
            'calibration_time': self.stats['calibration_time']
        }
    
    def quantize_model(self) -> nn.Module:
        """
        Apply INT4 quantization to the model
        
        Returns:
            Quantized model
        """
        print("\n" + "="*60)
        print("Applying INT4 Quantization")
        print("="*60)
        
        start_time = time.time()
        
        # Get Linear layers to quantize
        linear_layers = self._get_linear_layers()
        print(f"\nFound {len(linear_layers)} Linear layers to quantize")
        
        # Quantize each Linear layer
        quantized_count = 0
        
        for name, layer in linear_layers.items():
            try:
                # Create 4-bit linear layer
                if self.config.quant_type == 'nf4':
                    # NormalFloat4 quantization (recommended)
                    quant_layer = bnb.nn.Linear4bit(
                        layer.in_features,
                        layer.out_features,
                        bias=layer.bias is not None,
                        compute_dtype=getattr(torch, self.config.compute_dtype),
                        compress_statistics=self.config.use_double_quant,
                        quant_type='nf4'
                    )
                else:
                    # FP4 quantization
                    quant_layer = bnb.nn.Linear4bit(
                        layer.in_features,
                        layer.out_features,
                        bias=layer.bias is not None,
                        compute_dtype=getattr(torch, self.config.compute_dtype),
                        compress_statistics=self.config.use_double_quant,
                        quant_type='fp4'
                    )
                
                # Copy weights to quantized layer
                quant_layer.weight.data = layer.weight.data.to(self.device)
                if layer.bias is not None:
                    quant_layer.bias.data = layer.bias.data.to(self.device)
                
                # Replace layer in model
                self._replace_layer(name, quant_layer)
                quantized_count += 1
                
            except Exception as e:
                print(f"  Warning: Failed to quantize {name}: {e}")
        
        self.stats['quantized_params'] = sum(
            p.numel() for p in self.model.parameters()
        )
        self.stats['quantization_time'] = time.time() - start_time
        
        print(f"\n✓ Quantization complete in {self.stats['quantization_time']:.2f}s")
        print(f"  Quantized {quantized_count}/{len(linear_layers)} layers")
        print(f"  Original params: {self.stats['total_params']:,}")
        print(f"  Quantized params: {self.stats['quantized_params']:,}")
        
        return self.model
    
    def _replace_layer(self, name: str, new_layer: nn.Module):
        """Replace a layer in the model by name"""
        parts = name.split('.')
        module = self.model
        
        for part in parts[:-1]:
            module = getattr(module, part)
        
        setattr(module, parts[-1], new_layer)
    
    def apply_qat(self, train_loader, epochs: Optional[int] = None):
        """
        Quantization-Aware Training to recover accuracy
        
        Args:
            train_loader: DataLoader for training
            epochs: Number of QAT epochs (None = use config)
        
        Returns:
            Dict with training statistics
        """
        if epochs is None:
            epochs = self.config.qat_epochs
        
        if epochs == 0:
            print("\nSkipping QAT (epochs=0)")
            return {'epochs': 0, 'time': 0}
        
        print("\n" + "="*60)
        print(f"Starting Quantization-Aware Training ({epochs} epochs)")
        print("="*60)
        
        start_time = time.time()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.qat_learning_rate
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    inputs = batch['input_ids']
                    targets = batch.get('labels', batch['input_ids'])
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else inputs
                else:
                    inputs = batch.to(self.device)
                    targets = inputs
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss (cross-entropy)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.model.eval()
        self.stats['qat_time'] = time.time() - start_time
        
        print(f"\n✓ QAT complete in {self.stats['qat_time']:.2f}s")
        
        return {
            'epochs': epochs,
            'time': self.stats['qat_time']
        }
    
    def export_quantized_model(self, output_path: str, tokenizer_path: Optional[str] = None):
        """
        Save quantized model checkpoint
        
        Args:
            output_path: Path to save checkpoint
            tokenizer_path: Optional path to tokenizer config
        """
        print("\n" + "="*60)
        print("Exporting Quantized Model")
        print("="*60)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config.__dict__ if hasattr(self.model, 'config') else {},
            'quantization_config': self.config.to_dict(),
            'statistics': self.stats
        }
        
        if tokenizer_path and Path(tokenizer_path).exists():
            checkpoint['tokenizer_path'] = tokenizer_path
        
        # Save checkpoint
        torch.save(checkpoint, output_path)
        
        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"\n✓ Model saved to: {output_path}")
        print(f"  File size: {size_mb:.2f} MB")
        print(f"  Parameters: {self.stats['quantized_params']:,}")
        
        # Save quantization config separately
        config_path = output_path.parent / f"{output_path.stem}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'quantization_config': self.config.to_dict(),
                'statistics': self.stats,
                'model_size_mb': size_mb
            }, f, indent=2)
        
        print(f"  Config saved to: {config_path}")
        
        return {
            'path': str(output_path),
            'size_mb': size_mb,
            'parameters': self.stats['quantized_params']
        }


def benchmark_quantization(
    model_fp32: nn.Module,
    model_int4: nn.Module,
    test_loader,
    device: str = 'cuda'
) -> Dict:
    """
    Compare FP32 vs INT4 models (speed, size, perplexity)
    
    Args:
        model_fp32: Original FP32 model
        model_int4: Quantized INT4 model
        test_loader: DataLoader for evaluation
        device: Device to run on
    
    Returns:
        Dict with comparison metrics
    """
    print("\n" + "="*60)
    print("Benchmarking INT4 vs FP32")
    print("="*60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    results = {
        'fp32': {},
        'int4': {},
        'comparison': {}
    }
    
    # Benchmark FP32
    print("\nEvaluating FP32 model...")
    model_fp32 = model_fp32.to(device).eval()
    fp32_metrics = _evaluate_model(model_fp32, test_loader, device)
    results['fp32'] = fp32_metrics
    
    # Benchmark INT4
    print("\nEvaluating INT4 model...")
    model_int4 = model_int4.to(device).eval()
    int4_metrics = _evaluate_model(model_int4, test_loader, device)
    results['int4'] = int4_metrics
    
    # Compute comparisons
    results['comparison'] = {
        'speedup': fp32_metrics['inference_time'] / int4_metrics['inference_time'],
        'size_reduction': fp32_metrics['model_size_mb'] / int4_metrics['model_size_mb'],
        'perplexity_degradation': (int4_metrics['perplexity'] - fp32_metrics['perplexity']) / fp32_metrics['perplexity'] * 100
    }
    
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"\nFP32:")
    print(f"  Perplexity: {fp32_metrics['perplexity']:.2f}")
    print(f"  Inference time: {fp32_metrics['inference_time']:.4f}s")
    print(f"  Model size: {fp32_metrics['model_size_mb']:.2f} MB")
    
    print(f"\nINT4:")
    print(f"  Perplexity: {int4_metrics['perplexity']:.2f}")
    print(f"  Inference time: {int4_metrics['inference_time']:.4f}s")
    print(f"  Model size: {int4_metrics['model_size_mb']:.2f} MB")
    
    print(f"\nComparison:")
    print(f"  Speedup: {results['comparison']['speedup']:.2f}×")
    print(f"  Size reduction: {results['comparison']['size_reduction']:.2f}×")
    print(f"  Perplexity degradation: {results['comparison']['perplexity_degradation']:.2f}%")
    
    return results


def _evaluate_model(model: nn.Module, test_loader, device: torch.device) -> Dict:
    """Evaluate model (perplexity, inference time, size)"""
    import tempfile
    
    model.eval()
    total_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                inputs = batch['input_ids']
                targets = batch.get('labels', batch['input_ids'])
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
                targets = batch[1].to(device) if len(batch) > 1 else inputs
            else:
                inputs = batch.to(device)
                targets = inputs
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    inference_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Get model size
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        model_size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
    
    return {
        'perplexity': perplexity,
        'inference_time': inference_time,
        'model_size_mb': model_size_mb,
        'num_batches': num_batches
    }
