"""
Model export utilities for ONNX and TorchScript deployment.
"""

import torch
import torch.onnx
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export PyTorch models to ONNX and TorchScript formats."""
    
    def __init__(self, model: torch.nn.Module, model_name: str):
        """
        Initialize model exporter.
        
        Args:
            model: PyTorch model to export
            model_name: Name for exported files
        """
        self.model = model
        self.model_name = model_name
        self.model.eval()
        
        # Create wrapper for export if model returns tuples
        self.export_model = self._create_export_wrapper()
    
    def _create_export_wrapper(self):
        """Create a wrapper model that returns only logits for export."""
        class ExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.config = model.config if hasattr(model, 'config') else None
            
            def forward(self, x):
                output = self.model(x)
                # Extract logits if tuple
                if isinstance(output, tuple):
                    return output[0]  # Return only logits
                return output
        
        return ExportWrapper(self.model)
    
    def export_to_torchscript(
        self,
        output_path: str,
        sample_input: Optional[torch.Tensor] = None,
        seq_length: int = 128
    ) -> str:
        """
        Export model to TorchScript format.
        
        Args:
            output_path: Path to save TorchScript model
            sample_input: Sample input tensor (auto-generated if None)
            seq_length: Sequence length for sample input
            
        Returns:
            Path to saved TorchScript model
        """
        logger.info(f"Exporting {self.model_name} to TorchScript...")
        
        # Generate sample input if not provided
        if sample_input is None:
            batch_size = 1
            vocab_size = getattr(self.export_model.config, 'vocab_size', 1000) if hasattr(self.export_model, 'config') else 1000
            sample_input = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        try:
            # Use torch.jit.trace for better compatibility
            with torch.no_grad():
                traced_model = torch.jit.trace(self.export_model, sample_input)
            
            # Save traced model
            traced_model.save(output_path)
            logger.info(f"✓ TorchScript model saved to {output_path}")
            
            # Verify saved model
            loaded = torch.jit.load(output_path)
            with torch.no_grad():
                original_out = self.export_model(sample_input)
                loaded_out = loaded(sample_input)
                
                max_diff = torch.max(torch.abs(original_out - loaded_out)).item()
                logger.info(f"  Verification: max output difference = {max_diff:.6f}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
    
    def export_to_onnx(
        self,
        output_path: str,
        sample_input: Optional[torch.Tensor] = None,
        seq_length: int = 128,
        opset_version: int = 14,
        dynamic_axes: bool = True
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            sample_input: Sample input tensor
            seq_length: Sequence length for sample input
            opset_version: ONNX opset version
            dynamic_axes: Enable dynamic batch/sequence dimensions
            
        Returns:
            Path to saved ONNX model
        """
        logger.info(f"Exporting {self.model_name} to ONNX...")
        
        # Generate sample input
        if sample_input is None:
            batch_size = 1
            vocab_size = getattr(self.export_model.config, 'vocab_size', 1000) if hasattr(self.export_model, 'config') else 1000
            sample_input = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Configure dynamic axes
        if dynamic_axes:
            dynamic_axes_config = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        else:
            dynamic_axes_config = None
        
        try:
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    self.export_model,
                    sample_input,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes_config
                )
            
            logger.info(f"✓ ONNX model saved to {output_path}")
            
            # Verify ONNX model
            try:
                import onnx
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info("  ONNX model verification passed")
            except ImportError:
                logger.warning("  ONNX package not available for verification")
            except Exception as e:
                logger.warning(f"  ONNX verification warning: {e}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            logger.info("  Note: ONNX export may fail for models with complex control flow")
            raise
    
    def export_both(
        self,
        output_dir: str,
        seq_length: int = 128,
        onnx_opset: int = 14
    ) -> Dict[str, str]:
        """
        Export model to both TorchScript and ONNX formats.
        
        Args:
            output_dir: Directory to save exported models
            seq_length: Sequence length for sample input
            onnx_opset: ONNX opset version
            
        Returns:
            Dictionary with paths to exported models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Generate sample input once
        vocab_size = getattr(self.export_model.config, 'vocab_size', 1000) if hasattr(self.export_model, 'config') else 1000
        sample_input = torch.randint(0, vocab_size, (1, seq_length))
        
        # Export TorchScript
        ts_path = output_dir / f"{self.model_name}_torchscript.pt"
        try:
            results['torchscript'] = self.export_to_torchscript(
                str(ts_path),
                sample_input=sample_input,
                seq_length=seq_length
            )
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            results['torchscript'] = None
        
        # Export ONNX
        onnx_path = output_dir / f"{self.model_name}.onnx"
        try:
            results['onnx'] = self.export_to_onnx(
                str(onnx_path),
                sample_input=sample_input,
                seq_length=seq_length,
                opset_version=onnx_opset
            )
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            results['onnx'] = None
        
        # Save export metadata
        metadata = {
            'model_name': self.model_name,
            'seq_length': seq_length,
            'vocab_size': vocab_size,
            'onnx_opset': onnx_opset,
            'exports': results
        }
        
        metadata_path = output_dir / f"{self.model_name}_export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Export metadata saved to {metadata_path}")
        
        return results


def export_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    model_class: type,
    config: Any,
    model_name: str,
    seq_length: int = 128
) -> Dict[str, str]:
    """
    Load checkpoint and export to multiple formats.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory for exported models
        model_class: Model class to instantiate
        config: Model configuration
        model_name: Name for exported files
        seq_length: Sequence length for export
        
    Returns:
        Dictionary with export results
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = model_class(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Export
    exporter = ModelExporter(model, model_name)
    results = exporter.export_both(output_dir, seq_length=seq_length)
    
    return results


def benchmark_exported_model(
    torchscript_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    num_runs: int = 100,
    seq_length: int = 128,
    vocab_size: int = 1000
) -> Dict[str, Any]:
    """
    Benchmark exported models.
    
    Args:
        torchscript_path: Path to TorchScript model
        onnx_path: Path to ONNX model
        num_runs: Number of benchmark runs
        seq_length: Input sequence length
        vocab_size: Vocabulary size
        
    Returns:
        Benchmark results
    """
    import time
    import numpy as np
    
    results = {}
    sample_input = torch.randint(0, vocab_size, (1, seq_length))
    
    # Benchmark TorchScript
    if torchscript_path:
        logger.info(f"Benchmarking TorchScript model: {torchscript_path}")
        model = torch.jit.load(torchscript_path)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(sample_input)
                times.append(time.perf_counter() - start)
        
        results['torchscript'] = {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'throughput_samples_per_sec': 1.0 / np.mean(times)
        }
    
    # Benchmark ONNX
    if onnx_path:
        try:
            import onnxruntime as ort
            logger.info(f"Benchmarking ONNX model: {onnx_path}")
            
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: sample_input.numpy()})
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = session.run(None, {input_name: sample_input.numpy()})
                times.append(time.perf_counter() - start)
            
            results['onnx'] = {
                'mean_latency_ms': np.mean(times) * 1000,
                'std_latency_ms': np.std(times) * 1000,
                'throughput_samples_per_sec': 1.0 / np.mean(times)
            }
        except ImportError:
            logger.warning("ONNX Runtime not available, skipping ONNX benchmark")
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Model export utilities ready")
