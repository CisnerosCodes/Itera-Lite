"""
Main training script for Itera-Lite and baseline models
Phase 3: Training & Benchmarking Pipeline
"""

import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    IteraLiteModel, TransformerBaseline,
    get_tiny_config, get_small_config,
    get_transformer_tiny_config, get_transformer_small_config
)
from utils.data import create_simple_dataset, get_dataloader
from utils.training import Trainer, create_optimizer, create_scheduler
from utils.benchmark import ModelBenchmark, compare_models


def train_model(
    model_type: str = 'itera',
    config_size: str = 'tiny',
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seq_length: int = 128,
    num_samples: int = 1000,
    device: str = 'cpu'
):
    """Train a single model"""
    
    print(f"\n{'='  * 70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Model type: {model_type}")
    print(f"Config size: {config_size}")
    print(f"Num epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Sequence length: {seq_length}")
    print(f"Device: {device}")
    print(f"{'=' * 70}\n")
    
    # Create model
    if model_type == 'itera':
        if config_size == 'tiny':
            config = get_tiny_config()
        elif config_size == 'small':
            config = get_small_config()
        else:
            raise ValueError(f"Unknown config size: {config_size}")
        
        # Adjust config for dataset
        config.max_seq_length = seq_length
        model = IteraLiteModel(config)
        model_name = f'itera_lite_{config_size}'
        
    elif model_type == 'transformer':
        if config_size == 'tiny':
            config = get_transformer_tiny_config()
        elif config_size == 'small':
            config = get_transformer_small_config()
        else:
            raise ValueError(f"Unknown config size: {config_size}")
        
        # Adjust config for dataset
        config.max_seq_length = seq_length
        model = TransformerBaseline(config)
        model_name = f'transformer_{config_size}'
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create dataset
    print("Creating dataset...")
    train_dataset, val_dataset, tokenizer = create_simple_dataset(
        num_samples=num_samples,
        vocab_size=config.vocab_size,
        seq_length=seq_length,
        level='char'
    )
    
    # Save tokenizer
    tokenizer_path = Path('data') / f'tokenizer_{config_size}.json'
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to: {tokenizer_path}\n")
    
    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, lr=learning_rate)
    num_training_steps = len(train_loader) * num_epochs
    scheduler = create_scheduler(optimizer, num_training_steps)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_name=model_name,
        eval_every=50,
        save_every=200,
        early_stopping_patience=3
    )
    
    # Train
    trainer.train(num_epochs=num_epochs)
    
    return model, model_name, val_loader


def benchmark_model(model, model_name: str, val_loader, device: str = 'cpu'):
    """Benchmark a trained model"""
    
    # Create benchmark
    benchmark = ModelBenchmark(model, model_name, device=device)
    
    # Run full benchmark
    results_path = Path('results') / f'{model_name}_benchmark.json'
    results = benchmark.run_full_benchmark(
        dataloader=val_loader,
        batch_size=1,
        seq_length=128,
        save_path=str(results_path)
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train and benchmark Itera-Lite models')
    parser.add_argument('--model', type=str, choices=['itera', 'transformer', 'both'], default='both',
                       help='Model type to train')
    parser.add_argument('--config', type=str, choices=['tiny', 'small'], default='tiny',
                       help='Model configuration size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seq-length', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only run benchmarks')
    
    args = parser.parse_args()
    
    print(f"\n{'#' * 70}")
    print(f"#{'ITERA-LITE PHASE 3: TRAINING & BENCHMARKING PIPELINE':^68}#")
    print(f"{'#' * 70}\n")
    
    results_list = []
    
    # Train models
    if args.model in ['itera', 'both'] and not args.skip_training:
        print("\n" + "=" * 70)
        print("TRAINING ITERA-LITE MODEL")
        print("=" * 70)
        
        itera_model, itera_name, val_loader = train_model(
            model_type='itera',
            config_size=args.config,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seq_length=args.seq_length,
            num_samples=args.num_samples,
            device=args.device
        )
        
        # Benchmark
        print("\n" + "=" * 70)
        print("BENCHMARKING ITERA-LITE MODEL")
        print("=" * 70)
        
        itera_results = benchmark_model(itera_model, itera_name, val_loader, args.device)
        results_list.append(itera_results)
    
    if args.model in ['transformer', 'both'] and not args.skip_training:
        print("\n" + "=" * 70)
        print("TRAINING TRANSFORMER BASELINE")
        print("=" * 70)
        
        transformer_model, transformer_name, val_loader = train_model(
            model_type='transformer',
            config_size=args.config,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seq_length=args.seq_length,
            num_samples=args.num_samples,
            device=args.device
        )
        
        # Benchmark
        print("\n" + "=" * 70)
        print("BENCHMARKING TRANSFORMER BASELINE")
        print("=" * 70)
        
        transformer_results = benchmark_model(transformer_model, transformer_name, val_loader, args.device)
        results_list.append(transformer_results)
    
    # Compare models
    if len(results_list) > 1:
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        
        comparison_path = Path('results') / f'comparison_{args.config}.json'
        compare_models(results_list, save_path=str(comparison_path))
    
    print(f"\n{'#' * 70}")
    print(f"#{'PHASE 3 COMPLETE!':^68}#")
    print(f"{'#' * 70}\n")
    
    print("Results saved to:")
    print(f"  - Checkpoints: checkpoints/")
    print(f"  - Metrics: results/")
    print(f"  - Logs: results/*_metrics.csv")
    print(f"  - Benchmarks: results/*_benchmark.json")
    
    print("\nNext steps:")
    print("  1. Review training curves: results/*_metrics.csv")
    print("  2. Analyze benchmarks: results/*_benchmark.json")
    print("  3. Generate visualizations: python visualize_results.py")
    print("  4. Read efficiency report: reports/efficiency_report.md")


if __name__ == "__main__":
    main()
