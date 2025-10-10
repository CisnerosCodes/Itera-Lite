"""
Phase 7 Task 3: Mixed-Precision Optimization Main Script

Applies layer-wise INT8/FP16 precision to Itera-Lite SSM architecture.
Reuses checkpoint loading logic from Task 2 and config inference from Task 1.

Expected Results:
- Compression: 1.5× standalone, 2.1× cumulative with INT4
- Quality: <5% perplexity degradation
- Inference: 1.3× speedup on A30 GPU

Author: GitHub Copilot
Date: October 10, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
import time
from pathlib import Path
import numpy as np

# Import model components
import sys
sys.path.append(str(Path(__file__).parent))
from models.itera_lite import IteraLiteModel
from models.config import IteraLiteConfig
from utils.mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionConverter,
    get_conservative_precision_map,
    get_aggressive_precision_map,
    visualize_precision_allocation,
    save_mixed_precision_checkpoint,
)


def load_checkpoint_with_inference(checkpoint_path: str, device: str = 'cuda'):
    """
    Load checkpoint and infer configuration.
    Handles old checkpoint format (.moe.layer.) vs new format (.moe.moe.).
    
    Reused from Task 2 (phase7_prune.py) with proven compatibility logic.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device
    
    Returns:
        Tuple of (model, config)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        train_config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        train_config = {}
    
    print("\n" + "="*60)
    print("INFERRING MODEL CONFIGURATION")
    print("="*60)
    
    # Infer configuration from checkpoint (matching phase7_prune.py logic)
    vocab_size = state_dict['embedding.weight'].shape[0]
    hidden_size = state_dict['embedding.weight'].shape[1]
    max_seq_length = state_dict['position_embedding.weight'].shape[0]
    num_layers = sum(1 for k in state_dict.keys() if k.startswith('layers.') and '.ssm.in_proj.weight' in k)
    ssm_state_size = state_dict['layers.0.ssm.ssm.B'].shape[0]
    num_experts = sum(1 for k in state_dict.keys() if k.startswith('layers.1.moe.moe.experts.') and '.w1.weight' in k)
    
    # Get expert size (try different possible keys)
    expert_size = 64  # default
    for key in state_dict.keys():
        if 'experts.0.w1.weight' in key:
            expert_size = state_dict[key].shape[0]
            break
    
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  max_seq_length: {max_seq_length}")
    print(f"  num_layers: {num_layers}")
    print(f"  ssm_state_size: {ssm_state_size}")
    print(f"  num_experts: {num_experts}")
    print(f"  expert_size: {expert_size}")
    print(f"  top_k_experts: 2 (default)")
    
    # Create config
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
    
    print("="*60)
    
    # Create model
    print("\nInitializing model...")
    model = IteraLiteModel(config)
    
    # Convert old checkpoint format (Task 2 lesson)
    print("Converting checkpoint format (.moe.layer. → .moe.moe.)...")
    new_state_dict = {}
    converted_keys = []
    
    for key, value in state_dict.items():
        if '.moe.layer.' in key:
            new_key = key.replace('.moe.layer.', '.moe.moe.')
            new_state_dict[new_key] = value
            converted_keys.append(f"  {key} → {new_key}")
        else:
            new_state_dict[key] = value
    
    if converted_keys:
        print(f"Converted {len(converted_keys)} keys:")
        for conv in converted_keys[:5]:  # Show first 5
            print(conv)
        if len(converted_keys) > 5:
            print(f"  ... and {len(converted_keys) - 5} more")
    else:
        print("No conversion needed (checkpoint already in new format)")
    
    # Load with strict=False to handle missing keys
    print("\nLoading state dict...")
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠ Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"    {key}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"    {key}")
    
    print("✓ Checkpoint loaded successfully")
    
    return model, config


def create_calibration_data(vocab_size: int, max_seq_length: int, 
                           num_samples: int = 1000, batch_size: int = 32):
    """
    Create calibration dataset for INT8 quantization.
    
    Generates diverse synthetic sequences covering vocabulary range.
    
    Args:
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        num_samples: Number of calibration samples
        batch_size: Batch size for dataloader
    
    Returns:
        DataLoader with calibration data
    """
    print("\n" + "="*60)
    print("CREATING CALIBRATION DATASET")
    print("="*60)
    print(f"  Samples: {num_samples}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Sequence length: {max_seq_length}")
    print(f"  Batch size: {batch_size}")
    
    # Generate diverse sequences
    sequences = []
    for i in range(num_samples):
        if i % 3 == 0:
            # Random tokens (exploration)
            seq = torch.randint(0, vocab_size, (max_seq_length,))
        elif i % 3 == 1:
            # Frequent tokens (common patterns)
            # Bias toward lower token IDs (typically more frequent)
            seq = torch.randint(0, vocab_size // 10, (max_seq_length,))
        else:
            # Mixed distribution
            seq = torch.randint(0, vocab_size // 2, (max_seq_length,))
        
        sequences.append(seq)
    
    sequences = torch.stack(sequences)
    
    dataset = TensorDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✓ Created calibration dataset: {num_samples} samples")
    return dataloader


def create_evaluation_data(vocab_size: int, max_seq_length: int,
                          num_samples: int = 500, batch_size: int = 32):
    """
    Create evaluation dataset for benchmarking.
    
    Args:
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        num_samples: Number of evaluation samples
        batch_size: Batch size for dataloader
    
    Returns:
        DataLoader with evaluation data
    """
    print("\nCreating evaluation dataset...")
    print(f"  Samples: {num_samples}")
    
    # Generate random sequences
    sequences = torch.randint(0, vocab_size, (num_samples, max_seq_length))
    
    dataset = TensorDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Created evaluation dataset: {num_samples} samples")
    return dataloader


def evaluate_perplexity(model: nn.Module, dataloader: DataLoader, device: str = 'cuda'):
    """
    Evaluate perplexity on test data.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device for evaluation
    
    Returns:
        Perplexity score
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, (input_ids,) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            
            # Shift for language modeling
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            # Forward pass
            try:
                logits = model(inputs)
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += targets.numel()
            
            except Exception as e:
                print(f"  ⚠ Error in batch {batch_idx}: {e}")
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def benchmark_mixed_precision(original_model: nn.Module, 
                             mixed_model: nn.Module,
                             dataloader: DataLoader,
                             device: str = 'cuda'):
    """
    Benchmark original vs mixed-precision model.
    
    Args:
        original_model: Original FP32 model
        mixed_model: Mixed INT8/FP16 model
        dataloader: Evaluation dataloader
        device: Device for benchmarking
    
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING MODELS")
    print("="*60)
    
    # Evaluate original model
    print("\nEvaluating FP32 baseline...")
    start_time = time.time()
    original_ppl = evaluate_perplexity(original_model, dataloader, device)
    original_time = time.time() - start_time
    print(f"  Perplexity: {original_ppl:.4f}")
    print(f"  Time: {original_time:.2f}s")
    
    # Evaluate mixed-precision model
    print("\nEvaluating mixed-precision model...")
    start_time = time.time()
    mixed_ppl = evaluate_perplexity(mixed_model, dataloader, device)
    mixed_time = time.time() - start_time
    print(f"  Perplexity: {mixed_ppl:.4f}")
    print(f"  Time: {mixed_time:.2f}s")
    
    # Calculate differences
    ppl_increase = ((mixed_ppl - original_ppl) / original_ppl) * 100
    speedup = original_time / mixed_time if mixed_time > 0 else 1.0
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"  Perplexity increase: {ppl_increase:+.2f}%")
    print(f"  Inference speedup: {speedup:.2f}×")
    
    return {
        'original_perplexity': original_ppl,
        'mixed_perplexity': mixed_ppl,
        'perplexity_increase_pct': ppl_increase,
        'original_time': original_time,
        'mixed_time': mixed_time,
        'speedup': speedup,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 7 Task 3: Mixed-Precision Optimization')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/itera_lite_tiny_best.pt',
                       help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/mixed_precision',
                       help='Output directory for mixed-precision checkpoint')
    parser.add_argument('--strategy', type=str, default='conservative',
                       choices=['conservative', 'aggressive'],
                       help='Precision allocation strategy')
    parser.add_argument('--calibration-samples', type=int, default=1000,
                       help='Number of calibration samples')
    parser.add_argument('--eval-samples', type=int, default=500,
                       help='Number of evaluation samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for calibration and evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 7 TASK 3: MIXED-PRECISION OPTIMIZATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Strategy: {args.strategy}")
    print(f"Device: {args.device}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load checkpoint
    model, config = load_checkpoint_with_inference(args.checkpoint, args.device)
    
    # Analyze model architecture
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Count by component
    embedding_params = sum(p.numel() for name, p in model.named_parameters() 
                          if 'embeddings' in name)
    ssm_params = sum(p.numel() for name, p in model.named_parameters() 
                    if 'layers' in name and 'ssm' in name)
    norm_params = sum(p.numel() for name, p in model.named_parameters() 
                     if 'norm' in name)
    
    print(f"  Embeddings:     {embedding_params:,} ({embedding_params*100//total_params}%)")
    print(f"  SSM Layers:     {ssm_params:,} ({ssm_params*100//total_params}%)")
    print(f"  Norms:          {norm_params:,} ({norm_params*100//total_params}%)")
    
    # Save original model for comparison
    original_model = model
    
    # Create calibration dataset
    calib_dataloader = create_calibration_data(
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        num_samples=args.calibration_samples,
        batch_size=args.batch_size
    )
    
    # Create evaluation dataset
    eval_dataloader = create_evaluation_data(
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        num_samples=args.eval_samples,
        batch_size=args.batch_size
    )
    
    # Get precision map
    if args.strategy == 'conservative':
        precision_map = get_conservative_precision_map()
        print("\n✓ Using conservative precision strategy")
    else:
        precision_map = get_aggressive_precision_map()
        print("\n✓ Using aggressive precision strategy")
    
    # Initialize mixed-precision converter
    mp_config = MixedPrecisionConfig(
        precision_map=precision_map,
        calibration_method='percentile',
        calibration_samples=args.calibration_samples,
        percentile=99.99,
        symmetric_quant=True,
        per_channel=True,
        max_perplexity_increase=5.0,
    )
    
    converter = MixedPrecisionConverter(model, mp_config, args.device)
    
    # Analyze precision allocation
    arch_stats = converter.analyze_model()
    print("\n" + "="*60)
    print("PRECISION ALLOCATION ANALYSIS")
    print("="*60)
    print(f"  Total params: {arch_stats['total_params']:,}")
    print(f"  INT8 params: {arch_stats['int8_params']:,} ({arch_stats['int8_params']*100//arch_stats['total_params']}%)")
    print(f"  FP16 params: {arch_stats['fp16_params']:,} ({arch_stats['fp16_params']*100//arch_stats['total_params']}%)")
    print(f"  Unmatched params: {arch_stats['unmatched_params']:,} ({arch_stats['unmatched_params']*100//arch_stats['total_params']}%)")
    
    # Calibrate INT8 layers
    converter.calibrate(calib_dataloader)
    
    # Apply mixed-precision conversion
    mixed_model = converter.apply_mixed_precision()
    
    # Calculate compression
    compression_stats = converter.calculate_compression_ratio()
    print("\n" + "="*60)
    print("COMPRESSION ANALYSIS")
    print("="*60)
    print(f"  FP32 Memory: {compression_stats['fp32_memory_mb']:.2f} MB")
    print(f"  Mixed Memory: {compression_stats['mixed_memory_mb']:.2f} MB")
    print(f"  Compression Ratio: {compression_stats['compression_ratio']:.2f}×")
    print(f"  Memory Saved: {compression_stats['memory_saved_mb']:.2f} MB")
    
    # Benchmark
    benchmark_results = benchmark_mixed_precision(
        original_model=original_model,
        mixed_model=mixed_model,
        dataloader=eval_dataloader,
        device=args.device
    )
    
    # Validate quality threshold
    if benchmark_results['perplexity_increase_pct'] > mp_config.max_perplexity_increase:
        print("\n" + "="*60)
        print("⚠ WARNING: Perplexity increase exceeds threshold!")
        print("="*60)
        print(f"  Threshold: {mp_config.max_perplexity_increase}%")
        print(f"  Actual: {benchmark_results['perplexity_increase_pct']:.2f}%")
        print("  Consider using more FP16 layers or better calibration.")
    else:
        print("\n" + "="*60)
        print("✓ Quality threshold met!")
        print("="*60)
        print(f"  Perplexity increase: {benchmark_results['perplexity_increase_pct']:.2f}% (target: <{mp_config.max_perplexity_increase}%)")
    
    # Save mixed-precision checkpoint
    metadata = converter.get_conversion_metadata()
    metadata['benchmark_results'] = benchmark_results
    
    checkpoint_path = output_dir / 'itera_lite_mixed_precision.pt'
    save_mixed_precision_checkpoint(
        model=mixed_model,
        config=config,
        metadata=metadata,
        output_path=str(checkpoint_path)
    )
    
    # Visualize results
    viz_path = output_dir / 'precision_allocation.png'
    visualize_precision_allocation(
        precision_map=precision_map,
        compression_stats=compression_stats,
        output_path=str(viz_path)
    )
    
    # Save detailed statistics
    stats_path = output_dir / 'mixed_precision_statistics.json'
    with open(stats_path, 'w') as f:
        stats = {
            'architecture': {
                'total_params': arch_stats['total_params'],
                'int8_params': arch_stats['int8_params'],
                'fp16_params': arch_stats['fp16_params'],
            },
            'compression': compression_stats,
            'benchmark': benchmark_results,
            'config': {
                'strategy': args.strategy,
                'calibration_method': mp_config.calibration_method,
                'calibration_samples': mp_config.calibration_samples,
                'percentile': mp_config.percentile,
            }
        }
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Saved statistics to {stats_path}")
    
    print("\n" + "="*60)
    print("SUCCESS: Mixed-Precision Optimization Complete!")
    print("="*60)
    print(f"\nOutput Files:")
    print(f"  - {checkpoint_path} (mixed-precision model)")
    print(f"  - {checkpoint_path.with_suffix('.json')} (metadata)")
    print(f"  - {stats_path} (detailed statistics)")
    print(f"  - {viz_path} (visualization)")
    
    print(f"\nResults Summary:")
    print(f"  Compression: {compression_stats['compression_ratio']:.2f}×")
    print(f"  Perplexity: {benchmark_results['original_perplexity']:.4f} → {benchmark_results['mixed_perplexity']:.4f} ({benchmark_results['perplexity_increase_pct']:+.2f}%)")
    print(f"  Speedup: {benchmark_results['speedup']:.2f}×")
    print(f"  Memory: {compression_stats['fp32_memory_mb']:.2f} MB → {compression_stats['mixed_memory_mb']:.2f} MB")


if __name__ == '__main__':
    main()
