"""
Phase 7 Task 2: Structured Pruning
Main script to prune Itera-Lite model with magnitude-based structured pruning
and GPU-accelerated fine-tuning on NVIDIA A30
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.itera_lite import IteraLiteModel
from models.config import IteraLiteConfig
from utils.structured_pruning import StructuredPruner, PruningConfig, count_parameters


class SimpleTextDataset(Dataset):
    """Simple dataset for fine-tuning and testing"""
    
    def __init__(self, texts, tokenizer, max_length=128):
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


class SimpleTokenizer:
    """Simple tokenizer placeholder"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


def load_checkpoint_with_inference(checkpoint_path: str, device: str = 'cuda') -> tuple:
    """
    Load model from checkpoint with config inference
    Reuses logic from Task 1 (phase7_quantize.py)
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, config)
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # Handle both dict and IteraLiteConfig object
        if isinstance(config_dict, dict):
            config = IteraLiteConfig(**config_dict)
        else:
            config = config_dict
        print("✓ Config loaded from checkpoint")
    else:
        # Infer config from model state dict
        print("⚠ Config not in checkpoint, inferring from state_dict...")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
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
        print(f"✓ Config inferred successfully")
    
    # Create model
    model = IteraLiteModel(config)
    
    # Load state dict with compatibility handling
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Fix checkpoint format compatibility
    # Handle different MoE naming conventions: layer.experts vs moe.experts
    new_state_dict = {}
    keys_converted = []
    for key, value in state_dict.items():
        # Convert old format: layers.X.moe.layer.experts -> layers.X.moe.moe.experts
        if '.moe.layer.' in key:
            new_key = key.replace('.moe.layer.', '.moe.moe.')
            new_state_dict[new_key] = value
            keys_converted.append(f"{key} -> {new_key}")
        else:
            new_state_dict[key] = value
    
    if keys_converted:
        print(f"⚠ Converted {len(keys_converted)} checkpoint keys for compatibility")
        print(f"  Example: {keys_converted[0]}")
    
    # Load with strict=False to handle any remaining mismatches
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠ Warning: {len(missing_keys)} missing keys in checkpoint")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"    - {key}")
    
    if unexpected_keys:
        print(f"⚠ Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"    - {key}")

    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    return model, config


def create_training_data(num_samples=5000, vocab_size=8000, seq_length=128):
    """
    Create synthetic training data for fine-tuning
    In production, use actual TinyStories dataset
    
    Args:
        num_samples: Number of training samples
        vocab_size: Vocabulary size
        seq_length: Sequence length
    
    Returns:
        List of text samples
    """
    print(f"\nGenerating {num_samples} training samples...")
    
    # Generate random sentences (placeholder for actual TinyStories)
    texts = []
    words = [f"word{i}" for i in range(min(200, vocab_size))]
    
    for _ in range(num_samples):
        sentence_length = torch.randint(20, seq_length, (1,)).item()
        sentence = " ".join([words[torch.randint(0, len(words), (1,)).item()] 
                            for _ in range(sentence_length)])
        texts.append(sentence)
    
    print(f"✓ Generated {len(texts)} training samples")
    
    return texts


def fine_tune_pruned_model(
    model: nn.Module,
    config: IteraLiteConfig,
    train_texts: list,
    val_texts: list,
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 32,
    warmup_ratio: float = 0.05,
    device: str = 'cuda'
) -> nn.Module:
    """
    Fine-tune pruned model to recover from quality degradation
    
    Args:
        model: Pruned model
        config: Model configuration
        train_texts: Training text samples
        val_texts: Validation text samples
        epochs: Number of fine-tuning epochs
        lr: Learning rate
        batch_size: Batch size
        warmup_ratio: Warmup ratio for learning rate schedule
        device: Device to train on
    
    Returns:
        Fine-tuned model
    """
    print("\n" + "="*60)
    print("FINE-TUNING PRUNED MODEL")
    print("="*60)
    
    model = model.to(device)
    model.train()
    
    # Create tokenizer and datasets
    tokenizer = SimpleTokenizer(config.vocab_size)
    train_dataset = SimpleTextDataset(train_texts, tokenizer, max_length=config.max_seq_length)
    val_dataset = SimpleTextDataset(val_texts, tokenizer, max_length=config.max_seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            
            # Handle different output formats (from Task 1 lessons)
            if isinstance(outputs, tuple):
                logits, loss, aux_loss = outputs if len(outputs) == 3 else (outputs[0], outputs[1], 0.0)
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
                aux_loss = 0.0
            else:
                # Calculate loss manually
                logits = outputs
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                aux_loss = 0.0
            
            # Total loss
            total_loss = loss + aux_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping (prevent instability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            train_batches += 1
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = train_loss / train_batches
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss {avg_loss:.4f}, LR {current_lr:.2e}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, labels=labels)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    loss = outputs[1] if len(outputs) > 1 else outputs[0]
                elif hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    logits = outputs
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ✓ Best validation loss so far!")
        
        # Optional: Early stopping if loss plateaus
        # (Can be implemented based on convergence criteria)
    
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    model.eval()
    return model


def benchmark_pruning(
    baseline_model: nn.Module,
    pruned_model: nn.Module,
    config: IteraLiteConfig,
    test_texts: list,
    device: str = 'cuda'
) -> dict:
    """
    Benchmark baseline vs pruned model
    
    Args:
        baseline_model: Original unpruned model
        pruned_model: Pruned and fine-tuned model
        config: Model configuration
        test_texts: Test text samples
        device: Device to run on
    
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARKING: BASELINE vs PRUNED")
    print("="*60)
    
    tokenizer = SimpleTokenizer(config.vocab_size)
    test_dataset = SimpleTextDataset(test_texts, tokenizer, max_length=config.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    def evaluate_model(model, name):
        """Evaluate model and return metrics"""
        model = model.to(device)
        model.eval()
        
        total_loss = 0.0
        total_batches = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, labels=labels)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    loss = outputs[1] if len(outputs) > 1 else outputs[0]
                elif hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    logits = outputs
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                
                total_loss += loss.item()
                total_batches += 1
        
        inference_time = time.time() - start_time
        avg_loss = total_loss / total_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Get model size
        param_count = count_parameters(model)
        
        print(f"\n{name}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Inference time: {inference_time:.2f}s ({total_batches} batches)")
        print(f"  Time per batch: {inference_time/total_batches*1000:.2f}ms")
        
        return {
            'parameters': param_count,
            'loss': avg_loss,
            'perplexity': perplexity,
            'inference_time': inference_time,
            'batches': total_batches
        }
    
    # Benchmark both models
    baseline_results = evaluate_model(baseline_model, "BASELINE (FP32)")
    pruned_results = evaluate_model(pruned_model, "PRUNED + FINE-TUNED")
    
    # Calculate improvements
    param_reduction = baseline_results['parameters'] / pruned_results['parameters']
    speedup = baseline_results['inference_time'] / pruned_results['inference_time']
    perplexity_change = (pruned_results['perplexity'] - baseline_results['perplexity']) / baseline_results['perplexity'] * 100
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Parameter Reduction: {param_reduction:.2f}× ({baseline_results['parameters']:,} → {pruned_results['parameters']:,})")
    print(f"Inference Speedup: {speedup:.2f}×")
    print(f"Perplexity Change: {perplexity_change:+.2f}%")
    print("="*60 + "\n")
    
    return {
        'baseline': baseline_results,
        'pruned': pruned_results,
        'comparison': {
            'parameter_reduction': param_reduction,
            'inference_speedup': speedup,
            'perplexity_change_percent': perplexity_change
        }
    }


def save_pruned_checkpoint(
    model: nn.Module,
    config: IteraLiteConfig,
    pruning_stats: dict,
    benchmark_results: dict,
    output_path: str
):
    """
    Save pruned model checkpoint with metadata
    
    Args:
        model: Pruned model
        config: Model configuration
        pruning_stats: Pruning statistics
        benchmark_results: Benchmark results
        output_path: Path to save checkpoint
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'pruning_stats': pruning_stats,
        'benchmark_results': benchmark_results,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, output_path)
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Saved pruned checkpoint:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    
    # Save JSON metadata
    json_path = output_path.with_suffix('.json')
    metadata = {
        'config': vars(config),
        'pruning_stats': pruning_stats,
        'benchmark_results': benchmark_results,
        'checkpoint_size_mb': file_size_mb,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 7 Task 2: Structured Pruning')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint to prune')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save pruned model')
    
    # Pruning arguments
    parser.add_argument('--target-sparsity', type=float, default=0.4,
                       help='Target overall sparsity (0.0 to 1.0)')
    parser.add_argument('--ssm-sparsity', type=float, default=0.25,
                       help='SSM layer sparsity')
    parser.add_argument('--moe-sparsity', type=float, default=0.60,
                       help='MoE expert sparsity')
    
    # Fine-tuning arguments
    parser.add_argument('--finetune-epochs', type=int, default=5,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--finetune-lr', type=float, default=1e-4,
                       help='Fine-tuning learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--warmup-ratio', type=float, default=0.05,
                       help='Warmup ratio for learning rate schedule')
    
    # Data arguments
    parser.add_argument('--train-samples', type=int, default=5000,
                       help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=1000,
                       help='Number of validation samples')
    parser.add_argument('--test-samples', type=int, default=500,
                       help='Number of test samples')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for computation')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate sparsity visualization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PHASE 7 TASK 2: STRUCTURED PRUNING")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # 1. Load model
    model, config = load_checkpoint_with_inference(args.checkpoint, args.device)
    
    # 2. Initialize pruner
    prune_config = PruningConfig(
        target_sparsity=args.target_sparsity,
        ssm_in_proj_sparsity=args.ssm_sparsity,
        ssm_out_proj_sparsity=args.ssm_sparsity,
        ssm_delta_proj_sparsity=args.ssm_sparsity * 0.8,
        ssm_conv_sparsity=args.ssm_sparsity * 0.7,
        moe_expert_sparsity=args.moe_sparsity,
        moe_router_sparsity=0.10,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_batch_size=args.batch_size,
        finetune_warmup_ratio=args.warmup_ratio,
        device=args.device
    )
    
    pruner = StructuredPruner(model, prune_config)
    
    # 3. Apply pruning
    pruned_model = pruner.apply_pruning()
    
    # Save pruning statistics
    pruning_stats = pruner.get_pruning_statistics()
    stats_path = Path(args.output).parent / 'pruning_statistics.json'
    pruner.save_statistics(stats_path)
    
    # Generate visualization
    if args.visualize:
        viz_path = Path(args.output).parent / 'pruning_sparsity.png'
        pruner.visualize_sparsity(viz_path)
    
    # 4. Fine-tune pruned model
    train_texts = create_training_data(args.train_samples, config.vocab_size, config.max_seq_length)
    val_texts = create_training_data(args.val_samples, config.vocab_size, config.max_seq_length)
    
    fine_tuned_model = fine_tune_pruned_model(
        pruned_model,
        config,
        train_texts,
        val_texts,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        device=args.device
    )
    
    # 5. Benchmark
    test_texts = create_training_data(args.test_samples, config.vocab_size, config.max_seq_length)
    benchmark_results = benchmark_pruning(
        model,  # baseline
        fine_tuned_model,  # pruned
        config,
        test_texts,
        args.device
    )
    
    # 6. Save checkpoint
    save_pruned_checkpoint(
        fine_tuned_model,
        config,
        pruning_stats,
        benchmark_results,
        args.output
    )
    
    print("\n" + "="*60)
    print("PHASE 7 TASK 2 COMPLETE!")
    print("="*60)
    print(f"Pruned model saved to: {args.output}")
    print(f"Parameter reduction: {benchmark_results['comparison']['parameter_reduction']:.2f}×")
    print(f"Inference speedup: {benchmark_results['comparison']['inference_speedup']:.2f}×")
    print(f"Perplexity change: {benchmark_results['comparison']['perplexity_change_percent']:+.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
