"""
Phase 4: Compression & Optimization Pipeline

Implements vocabulary optimization, quantization, distillation, and benchmarking
to achieve 100-300x efficiency improvements.
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import csv

from models.config import get_tiny_config, get_micro_config
from models.itera_lite import IteraLiteModel
from utils.dataset_loader import prepare_dataset
from utils.quantization import ModelQuantizer, quantize_model_int8
from utils.distillation import distill_model
from utils.benchmark import ModelBenchmark
from utils.training import Trainer


def train_with_vocab_optimization(
    dataset_name: str,
    vocab_sizes: list,
    epochs: int = 5,
    batch_size: int = 4,
    device: str = 'cpu'
):
    """
    Train models with different vocabulary sizes
    
    Args:
        dataset_name: 'tinystories' or 'wikitext2'
        vocab_sizes: List of vocabulary sizes to test
        epochs: Training epochs
        batch_size: Batch size
        device: Device to use
    """
    print("=" * 80)
    print("TASK 1: VOCABULARY OPTIMIZATION")
    print("=" * 80)
    
    results = []
    
    for vocab_size in vocab_sizes:
        print(f"\n{'='*60}")
        print(f"Training with vocab_size={vocab_size}")
        print(f"{'='*60}")
        
        # Prepare dataset with specific vocab size
        train_dataset, val_dataset, tokenizer = prepare_dataset(
            dataset_name=dataset_name,
            vocab_size=vocab_size,
            seq_length=128,
            num_samples=5000
        )
        
        # Save tokenizer
        tokenizer_path = f"data/tokenizer_{vocab_size}.json"
        tokenizer.save(tokenizer_path)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model with custom vocab size
        from models.config import IteraLiteConfig
        config = IteraLiteConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            num_layers=4,
            ssm_state_size=8,
            num_experts=4,
            expert_size=64,
            max_seq_length=128,
        )
        model = IteraLiteModel(config).to(device)
        
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
        
        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=f"checkpoints/vocab_{vocab_size}",
            model_name=f"itera_lite_vocab{vocab_size}"
        )
        
        history = trainer.train(num_epochs=epochs)
        
        # Benchmark
        benchmark = ModelBenchmark(model, model_name=f"itera_lite_vocab{vocab_size}", device=device)
        metrics = benchmark.run_full_benchmark(
            batch_size=batch_size,
            seq_length=128
        )
        
        # Store results
        result = {
            'vocab_size': vocab_size,
            'params': metrics['parameters']['total'],
            'best_val_loss': history['best_val_loss'],
            'perplexity': metrics.get('perplexity', 0),
            'flops_per_token': metrics['flops']['flops_per_token'],
            'throughput': metrics['inference_speed']['throughput_tokens_per_sec'],
            'memory_mb': metrics['memory']['total_memory_mb']
        }
        results.append(result)
        
        print(f"\nResults for vocab_size={vocab_size}:")
        print(f"  Parameters: {result['params']:,}")
        print(f"  Best val loss: {result['best_val_loss']:.4f}")
        print(f"  FLOPs/token: {result['flops_per_token']:,}")
    
    # Save vocabulary optimization results
    output_path = Path("results/vocab_optimization.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Vocabulary optimization results saved to {output_path}")
    
    return results


def apply_quantization(
    model_path: str,
    config_name: str = 'tiny',
    vocab_size: int = 2000,
    device: str = 'cpu'
):
    """
    Apply quantization to trained model
    
    Args:
        model_path: Path to trained model checkpoint
        config_name: Model configuration
        vocab_size: Vocabulary size
        device: Device to use
    """
    print("\n" + "=" * 80)
    print("TASK 2: QUANTIZATION")
    print("=" * 80)
    
    # Load original model
    from models.config import IteraLiteConfig
    config = IteraLiteConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=4,
        ssm_state_size=8,
        num_experts=4,
        expert_size=64,
        max_seq_length=128,
    )
    model = IteraLiteModel(config)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nLoaded model from {model_path}")
    print(f"Original parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Benchmark original model
    print("\nBenchmarking original model...")
    original_benchmark = ModelBenchmark(model, model_name="itera_lite_original", device=device)
    original_metrics = original_benchmark.run_full_benchmark(
        batch_size=4,
        seq_length=128
    )
    
    # Apply INT8 quantization
    print("\n" + "-" * 60)
    print("Applying INT8 Quantization")
    print("-" * 60)
    
    quantizer = ModelQuantizer(model)
    quantized_int8 = quantizer.apply_dynamic_quantization()
    
    # Save quantized model
    int8_path = "checkpoints/quantized/itera_lite_tiny_int8.pt"
    torch.save(quantized_int8.state_dict(), int8_path)
    print(f"Saved INT8 model to {int8_path}")
    
    # Benchmark INT8 model
    print("\nBenchmarking INT8 model...")
    test_input = torch.randint(0, vocab_size, (4, 128))
    int8_results = quantizer.benchmark_quantized_model(quantized_int8, test_input)
    
    # Prepare results
    results = {
        'original': {
            'params': original_metrics['parameters']['total'],
            'size_mb': original_metrics['memory']['param_memory_mb'],
            'flops_per_token': original_metrics['flops']['flops_per_token'],
            'throughput': original_metrics['inference_speed']['throughput_tokens_per_sec'],
            'perplexity': original_metrics.get('perplexity', 0)
        },
        'int8': {
            'size_mb': int8_results['quantized_size_mb'],
            'compression_ratio': int8_results['compression_ratio'],
            'speedup': int8_results['speedup'],
            'time_ms': int8_results['quantized_time_ms']
        }
    }
    
    # Save quantization results
    output_path = Path("results/quantization_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Quantization results saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"Original: {results['original']['size_mb']:.2f} MB")
    print(f"INT8: {results['int8']['size_mb']:.2f} MB")
    print(f"Compression: {results['int8']['compression_ratio']:.2f}x")
    print(f"Speedup: {results['int8']['speedup']:.2f}x")
    
    return results


def perform_distillation(
    teacher_path: str,
    vocab_size: int = 2000,
    dataset_name: str = 'tinystories',
    epochs: int = 10,
    batch_size: int = 4,
    device: str = 'cpu'
):
    """
    Perform knowledge distillation from teacher to student
    
    Args:
        teacher_path: Path to teacher model checkpoint
        vocab_size: Vocabulary size
        dataset_name: Dataset to use
        epochs: Training epochs
        batch_size: Batch size
        device: Device to use
    """
    print("\n" + "=" * 80)
    print("TASK 3: KNOWLEDGE DISTILLATION")
    print("=" * 80)
    
    # Load teacher model (Tiny)
    from models.config import IteraLiteConfig
    teacher_config = IteraLiteConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=4,
        ssm_state_size=8,
        num_experts=4,
        expert_size=64,
        max_seq_length=128,
    )
    teacher_model = IteraLiteModel(teacher_config)
    
    checkpoint = torch.load(teacher_path, map_location='cpu')
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.eval()
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"\nTeacher model loaded from {teacher_path}")
    print(f"Teacher parameters: {teacher_params:,}")
    
    # Create student model (Micro)
    student_config = get_micro_config(vocab_size=vocab_size)
    student_model = IteraLiteModel(student_config)
    
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"\nStudent model created (Micro)")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.2f}x")
    
    # Prepare dataset
    train_dataset, val_dataset, tokenizer = prepare_dataset(
        dataset_name=dataset_name,
        vocab_size=vocab_size,
        seq_length=128,
        num_samples=5000
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Perform distillation
    student_save_path = "checkpoints/distilled/itera_lite_micro_distilled.pt"
    
    trained_student, history = distill_model(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        learning_rate=1e-3,
        temperature=2.0,
        alpha=0.5,
        save_path=student_save_path,
        device=device
    )
    
    # Benchmark student
    print("\nBenchmarking distilled student model...")
    benchmark = ModelBenchmark(trained_student, model_name="itera_lite_micro_distilled", device=device)
    student_metrics = benchmark.run_full_benchmark(
        batch_size=batch_size,
        seq_length=128
    )
    
    # Compare with teacher
    teacher_benchmark = ModelBenchmark(teacher_model, model_name="itera_lite_teacher", device=device)
    teacher_metrics = teacher_benchmark.run_full_benchmark(
        batch_size=batch_size,
        seq_length=128
    )
    
    results = {
        'teacher': {
            'params': teacher_metrics['parameters']['total'],
            'flops_per_token': teacher_metrics['flops']['flops_per_token'],
            'throughput': teacher_metrics['inference_speed']['throughput_tokens_per_sec'],
            'perplexity': teacher_metrics.get('perplexity', 0)
        },
        'student': {
            'params': student_metrics['parameters']['total'],
            'flops_per_token': student_metrics['flops']['flops_per_token'],
            'throughput': student_metrics['inference_speed']['throughput_tokens_per_sec'],
            'perplexity': student_metrics.get('perplexity', 0)
        },
        'compression': {
            'param_ratio': teacher_params / student_params,
            'flops_ratio': teacher_metrics['flops']['flops_per_token'] / student_metrics['flops']['flops_per_token'],
            'perplexity_degradation': student_metrics.get('perplexity', 0) - teacher_metrics.get('perplexity', 0)
        },
        'training': history
    }
    
    # Save results
    output_path = Path("results/distillation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Distillation results saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("DISTILLATION SUMMARY")
    print("=" * 60)
    print(f"Teacher: {results['teacher']['params']:,} params, {results['teacher']['perplexity']:.2f} perplexity")
    print(f"Student: {results['student']['params']:,} params, {results['student']['perplexity']:.2f} perplexity")
    print(f"Compression: {results['compression']['param_ratio']:.2f}x parameters")
    print(f"Perplexity degradation: {results['compression']['perplexity_degradation']:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Compression & Optimization")
    parser.add_argument('--task', type=str, default='all', 
                       choices=['all', 'vocab', 'quantize', 'distill'],
                       help='Which task to run')
    parser.add_argument('--dataset', type=str, default='tinystories',
                       choices=['tinystories', 'wikitext2'],
                       help='Dataset to use')
    parser.add_argument('--vocab-sizes', type=int, nargs='+', default=[1000, 2000, 4000],
                       help='Vocabulary sizes to test')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--teacher-checkpoint', type=str, 
                       default='checkpoints/itera_lite_tiny_best.pt',
                       help='Path to teacher checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 4: COMPRESSION & OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"Tasks: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Ensure directories exist
    Path("checkpoints/quantized").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/distilled").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    
    # Task 1: Vocabulary Optimization
    if args.task in ['all', 'vocab']:
        vocab_results = train_with_vocab_optimization(
            dataset_name=args.dataset,
            vocab_sizes=args.vocab_sizes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    
    # Task 2: Quantization
    if args.task in ['all', 'quantize']:
        quant_results = apply_quantization(
            model_path=args.teacher_checkpoint,
            vocab_size=2000,  # Use optimal vocab size from Task 1
            device=args.device
        )
    
    # Task 3: Knowledge Distillation
    if args.task in ['all', 'distill']:
        distill_results = perform_distillation(
            teacher_path=args.teacher_checkpoint,
            vocab_size=2000,
            dataset_name=args.dataset,
            epochs=args.epochs * 2,  # More epochs for distillation
            batch_size=args.batch_size,
            device=args.device
        )
    
    print("\n" + "=" * 80)
    print("✓ PHASE 4 TASKS COMPLETED")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - results/vocab_optimization.json")
    print("  - results/quantization_results.json")
    print("  - results/distillation_results.json")
    print("\nCheckpoints saved to:")
    print("  - checkpoints/quantized/")
    print("  - checkpoints/distilled/")


if __name__ == "__main__":
    main()
