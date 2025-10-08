"""
Comprehensive Model Testing Script
Tests all Itera-Lite and baseline components
"""

import torch
import torch.nn as nn
from models import (
    IteraLiteConfig, TransformerConfig,
    IteraLiteModel, TransformerBaseline,
    get_tiny_config, get_small_config,
    get_transformer_tiny_config, get_transformer_small_config
)


def test_model(model, model_name, config):
    """Test a model with various inputs"""
    print(f"\n{'=' * 70}")
    print(f"Testing {model_name}")
    print(f"{'=' * 70}")
    
    # Test configuration
    batch_size = 2
    seq_len = 32
    
    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n1. Forward Pass Test")
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass
    if isinstance(model, IteraLiteModel):
        logits, loss, aux_loss = model(input_ids, labels)
        print(f"   Output shape: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Aux loss: {aux_loss.item():.6f}")
    else:
        logits, loss = model(input_ids, labels)
        print(f"   Output shape: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Wrong output shape!"
    print(f"   ✓ Shape correct")
    
    # Test backward pass
    print(f"\n2. Backward Pass Test")
    loss.backward()
    print(f"   ✓ Gradients computed successfully")
    
    # Check gradients
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"   ✓ All parameters have gradients: {has_grads}")
    
    # Zero gradients
    model.zero_grad()
    
    # Test generation
    print(f"\n3. Generation Test")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"   Prompt length: {prompt.shape[1]}")
    print(f"   Generated length: {generated.shape[1]}")
    assert generated.shape[1] == 30, "Generation length mismatch!"
    print(f"   ✓ Generation working")
    
    model.train()
    
    # Get efficiency stats
    print(f"\n4. Efficiency Statistics")
    stats = model.get_efficiency_stats()
    print(f"   Total parameters: {stats['total_params']:,}")
    print(f"   Non-embedding parameters: {stats['non_embedding_params']:,}")
    print(f"   Embedding parameters: {stats['embedding_params']:,}")
    print(f"   Approx FLOPs/token: {stats['approx_flops_per_token']:,}")
    print(f"   Number of layers: {stats['num_layers']}")
    print(f"   Hidden size: {stats['hidden_size']}")
    
    return stats


def compare_models(itera_stats, transformer_stats):
    """Compare efficiency between Itera-Lite and Transformer"""
    print(f"\n{'=' * 70}")
    print(f"MODEL COMPARISON")
    print(f"{'=' * 70}")
    
    print(f"\nParameter Count:")
    print(f"   Itera-Lite:  {itera_stats['total_params']:,}")
    print(f"   Transformer: {transformer_stats['total_params']:,}")
    
    param_ratio = transformer_stats['total_params'] / itera_stats['total_params']
    print(f"   → Itera-Lite is {param_ratio:.2f}x smaller")
    
    print(f"\nFLOPs per Token (approximate):")
    print(f"   Itera-Lite:  {itera_stats['approx_flops_per_token']:,}")
    print(f"   Transformer: {transformer_stats['approx_flops_per_token']:,}")
    
    flop_ratio = transformer_stats['approx_flops_per_token'] / itera_stats['approx_flops_per_token']
    print(f"   → Itera-Lite is {flop_ratio:.2f}x more efficient")
    
    print(f"\nArchitecture Details:")
    print(f"   {'Metric':<25} {'Itera-Lite':<15} {'Transformer':<15}")
    print(f"   {'-' * 60}")
    print(f"   {'Num layers':<25} {itera_stats['num_layers']:<15} {transformer_stats['num_layers']:<15}")
    print(f"   {'Hidden size':<25} {itera_stats['hidden_size']:<15} {transformer_stats['hidden_size']:<15}")
    
    if 'num_experts' in itera_stats:
        print(f"   {'Num experts':<25} {itera_stats['num_experts']:<15} {'N/A':<15}")
        print(f"   {'MoE layers':<25} {itera_stats['moe_layers']:<15} {'N/A':<15}")
    
    if 'num_attention_heads' in transformer_stats:
        print(f"   {'Attention heads':<25} {'N/A (SSM)':<15} {transformer_stats['num_attention_heads']:<15}")
    
    print(f"\n{'=' * 70}")
    print(f"EFFICIENCY GAINS")
    print(f"{'=' * 70}")
    print(f"✓ Parameter reduction: {param_ratio:.1f}x")
    print(f"✓ FLOPs reduction: {flop_ratio:.1f}x")
    
    # Check if we met goals
    target_param_reduction = 100  # 100x smaller target
    target_efficiency = 50  # 50x more efficient target
    
    print(f"\nGoal Achievement:")
    if param_ratio >= target_param_reduction:
        print(f"   ✓ Parameter goal MET: {param_ratio:.1f}x >= {target_param_reduction}x")
    else:
        print(f"   ○ Parameter goal: {param_ratio:.1f}x / {target_param_reduction}x (partial)")
    
    if flop_ratio >= target_efficiency:
        print(f"   ✓ Efficiency goal MET: {flop_ratio:.1f}x >= {target_efficiency}x")
    else:
        print(f"   ○ Efficiency goal: {flop_ratio:.1f}x / {target_efficiency}x (partial)")


def main():
    print("=" * 70)
    print(" " * 15 + "ITERA-LITE MODEL TESTING SUITE")
    print("=" * 70)
    
    # Test Itera-Lite (tiny config)
    print("\n" + "=" * 70)
    print("PART 1: ITERA-LITE MODEL (TINY CONFIG)")
    print("=" * 70)
    
    itera_config = get_tiny_config()
    itera_model = IteraLiteModel(itera_config)
    itera_stats = test_model(itera_model, "Itera-Lite (Tiny)", itera_config)
    
    # Test Transformer baseline (tiny config)
    print("\n" + "=" * 70)
    print("PART 2: TRANSFORMER BASELINE (TINY CONFIG)")
    print("=" * 70)
    
    transformer_config = get_transformer_tiny_config()
    transformer_model = TransformerBaseline(transformer_config)
    transformer_stats = test_model(transformer_model, "Transformer (Tiny)", transformer_config)
    
    # Compare models
    compare_models(itera_stats, transformer_stats)
    
    # Test with small configs
    print("\n\n" + "=" * 70)
    print("PART 3: LARGER MODELS COMPARISON")
    print("=" * 70)
    
    print("\nTesting Itera-Lite (Small Config)...")
    itera_small_config = get_small_config()
    itera_small = IteraLiteModel(itera_small_config)
    itera_small_stats = itera_small.get_efficiency_stats()
    print(f"   Total params: {itera_small_stats['total_params']:,}")
    
    print("\nTesting Transformer (Small Config)...")
    transformer_small_config = get_transformer_small_config()
    transformer_small = TransformerBaseline(transformer_small_config)
    transformer_small_stats = transformer_small.get_efficiency_stats()
    print(f"   Total params: {transformer_small_stats['total_params']:,}")
    
    param_ratio_small = transformer_small_stats['total_params'] / itera_small_stats['total_params']
    print(f"\n   → Small config: Itera-Lite is {param_ratio_small:.2f}x smaller")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nItera-Lite architecture is ready for training and evaluation!")
    print("\nNext steps:")
    print("   1. Implement training pipeline")
    print("   2. Create benchmark suite")
    print("   3. Run efficiency comparisons")
    print("   4. Generate final report")


if __name__ == "__main__":
    main()
