"""
Phase 7 Task 1: GPU-Native INT4 Quantization
Main script to quantize Itera-Lite model using bitsandbytes on NVIDIA A30
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.itera_lite import IteraLiteModel
from models.config import IteraLiteConfig
from utils.native_quantization import (
    NativeINT4Quantizer,
    QuantizationConfig,
    benchmark_quantization
)


class SimpleTextDataset(Dataset):
    """Simple dataset for calibration and testing"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = []
        for text in texts:
            # Simple tokenization (space-separated)
            tokens = text.lower().split()[:max_length]
            token_ids = [hash(token) % tokenizer.vocab_size for token in tokens]
            
            # Pad to max_length
            if len(token_ids) < max_length:
                token_ids += [0] * (max_length - len(token_ids))
            
            self.encodings.append(torch.tensor(token_ids, dtype=torch.long))
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx],
            'labels': self.encodings[idx]
        }


def load_model(checkpoint_path: str, device: str = 'cuda') -> tuple:
    """Load model from checkpoint"""
    print(f"\nLoading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config
    # Try to load config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # Handle both dict and IteraLiteConfig object
        if isinstance(config_dict, dict):
            config = IteraLiteConfig(**config_dict)
        else:
            config = config_dict
    else:
        # Infer config from model state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        vocab_size = state_dict['embedding.weight'].shape[0]
        hidden_size = state_dict['embedding.weight'].shape[1]
        max_seq_length = state_dict['position_embedding.weight'].shape[0]  # Infer from position embeddings
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('layers.') and '.ssm.in_proj.weight' in k)
        ssm_state_size = state_dict['layers.0.ssm.ssm.B'].shape[0]
        num_experts = sum(1 for k in state_dict.keys() if k.startswith('layers.1.moe.layer.experts.') and '.w1.weight' in k)
        expert_size = state_dict.get('layers.1.moe.layer.experts.0.w1.weight', 
                                     state_dict.get('layers.0.moe.ffn.w1.weight')).shape[0]
        
        config = IteraLiteConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_length=max_seq_length,
            num_layers=num_layers,
            ssm_state_size=ssm_state_size,
            num_experts=num_experts,
            expert_size=expert_size,
            top_k_experts=2
        )
        print(f"⚠ Config not found in checkpoint, inferred from state_dict")
    
    # Create model
    model = IteraLiteModel(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Config: {config}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def create_calibration_data(num_samples=1000, vocab_size=2000, seq_length=128):
    """Create synthetic calibration data for testing"""
    print(f"\nGenerating {num_samples} calibration samples...")
    
    # Generate random sentences (placeholder for actual TinyStories)
    texts = []
    words = [f"word{i}" for i in range(100)]
    
    for _ in range(num_samples):
        sentence_length = torch.randint(10, 50, (1,)).item()
        sentence = " ".join([words[torch.randint(0, len(words), (1,)).item()] 
                            for _ in range(sentence_length)])
        texts.append(sentence)
    
    print(f"✓ Generated {len(texts)} calibration samples")
    
    return texts


def main():
    parser = argparse.ArgumentParser(description='Phase 7 Task 1: GPU-Native INT4 Quantization')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint to quantize')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save quantized model')
    
    # Quantization arguments
    parser.add_argument('--quant-type', type=str, default='nf4', choices=['nf4', 'fp4'],
                       help='Quantization type (nf4=NormalFloat4, fp4=FP4)')
    parser.add_argument('--double-quant', action='store_true', default=True,
                       help='Use double quantization for better compression')
    parser.add_argument('--compute-dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16'],
                       help='Compute dtype for quantized operations')
    
    # Calibration arguments
    parser.add_argument('--calibration-samples', type=int, default=1000,
                       help='Number of samples for calibration')
    parser.add_argument('--calibration-batch-size', type=int, default=32,
                       help='Batch size for calibration')
    
    # QAT arguments
    parser.add_argument('--qat-epochs', type=int, default=0,
                       help='Quantization-Aware Training epochs (0=skip)')
    parser.add_argument('--qat-lr', type=float, default=1e-5,
                       help='Learning rate for QAT')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to calibration/test data (optional)')
    parser.add_argument('--tokenizer-path', type=str, default='data/tokenizer_2000.json',
                       help='Path to tokenizer config')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    # Benchmark arguments
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip benchmark comparison')
    parser.add_argument('--benchmark-batches', type=int, default=100,
                       help='Number of batches for benchmarking')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("Phase 7 Task 1: GPU-Native INT4 Quantization")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print(f"  Quantization type: {args.quant_type}")
    print(f"  Compute dtype: {args.compute_dtype}")
    print(f"  Device: {args.device}")
    print(f"  Calibration samples: {args.calibration_samples}")
    print(f"  QAT epochs: {args.qat_epochs}")
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        print(f"\nGPU Information:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model
    model, config = load_model(args.checkpoint, args.device)
    
    # Create calibration data
    if args.data_path and Path(args.data_path).exists():
        # Load actual data
        print(f"\nLoading calibration data from {args.data_path}")
        # TODO: Implement actual data loading
        texts = create_calibration_data(args.calibration_samples, config.vocab_size)
    else:
        # Generate synthetic data
        texts = create_calibration_data(args.calibration_samples, config.vocab_size)
    
    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
    
    tokenizer = MockTokenizer(config.vocab_size)
    
    # Create calibration dataset and dataloader (use model's max_seq_length)
    calibration_dataset = SimpleTextDataset(texts[:args.calibration_samples], tokenizer, max_length=config.max_seq_length)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False
    )
    
    # Create test dataset for benchmarking
    test_dataset = SimpleTextDataset(texts[:min(1000, len(texts))], tokenizer, max_length=config.max_seq_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False
    )
    
    # Create quantization config
    quant_config = QuantizationConfig(
        bits=4,
        use_double_quant=args.double_quant,
        quant_type=args.quant_type,
        compute_dtype=args.compute_dtype,
        calibration_samples=args.calibration_samples,
        calibration_batch_size=args.calibration_batch_size,
        qat_epochs=args.qat_epochs,
        qat_learning_rate=args.qat_lr,
        device=args.device
    )
    
    # Initialize quantizer
    quantizer = NativeINT4Quantizer(model, quant_config)
    
    # Step 1: Calibrate
    calibration_stats = quantizer.calibrate(
        calibration_loader,
        num_batches=args.calibration_samples // args.calibration_batch_size
    )
    
    # Step 2: Quantize
    quantized_model = quantizer.quantize_model()
    
    # Step 3: QAT (optional)
    if args.qat_epochs > 0:
        qat_stats = quantizer.apply_qat(calibration_loader, args.qat_epochs)
    
    # Step 4: Export
    export_info = quantizer.export_quantized_model(
        args.output,
        args.tokenizer_path if Path(args.tokenizer_path).exists() else None
    )
    
    # Step 5: Benchmark (optional)
    if not args.skip_benchmark:
        # Keep original model for comparison
        original_model, _ = load_model(args.checkpoint, args.device)
        
        benchmark_results = benchmark_quantization(
            original_model,
            quantized_model,
            test_loader,
            args.device
        )
        
        # Save benchmark results
        results_path = Path(args.output).parent / 'phase7_int4_benchmark.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': vars(args),
                'quantization_config': quant_config.to_dict(),
                'export_info': export_info,
                'calibration_stats': calibration_stats,
                'benchmark_results': benchmark_results,
                'statistics': quantizer.stats
            }, f, indent=2)
        
        print(f"\n✓ Benchmark results saved to: {results_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("Phase 7 Task 1 Complete!")
    print("="*70)
    print(f"\n✓ Quantized model saved to: {args.output}")
    print(f"✓ Model size: {export_info['size_mb']:.2f} MB")
    print(f"✓ Total time: {sum([quantizer.stats['calibration_time'], quantizer.stats['quantization_time'], quantizer.stats['qat_time']]):.2f}s")
    
    if not args.skip_benchmark:
        print(f"\nCompression Results:")
        print(f"  Size reduction: {benchmark_results['comparison']['size_reduction']:.2f}×")
        print(f"  Speedup: {benchmark_results['comparison']['speedup']:.2f}×")
        print(f"  Perplexity degradation: {benchmark_results['comparison']['perplexity_degradation']:.2f}%")
    
    print("\nNext Steps:")
    print("  1. Review benchmark results in results/phase7_int4_benchmark.json")
    print("  2. Test quantized model with inference script")
    print("  3. Proceed to Task 2: Structured Pruning")
    

if __name__ == '__main__':
    main()
