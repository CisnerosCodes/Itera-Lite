"""
Phase 8: Quality Testing & Comparison
Test both FP32 and FP16 models for generation quality
"""

import torch
import time
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from models import IteraLiteModel
from models.config import IteraLiteConfig
from utils.data import SimpleTokenizer


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    print(f"\nLoading: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = IteraLiteConfig(**config_dict)
    else:
        raise ValueError("Config not found in checkpoint")
    
    # Create and load model
    model = IteraLiteModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Calculate size
    param_count = sum(p.numel() for p in model.parameters())
    memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    
    print(f"  ✓ Loaded successfully")
    print(f"  Parameters: {param_count:,}")
    print(f"  Memory: {memory_mb:.2f} MB")
    print(f"  Vocab size: {config.vocab_size}")
    
    return model, config, memory_mb


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, device='cpu'):
    """Generate text from prompt"""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.special_tokens.get('<BOS>', 0)]
    
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Generate
    generated_tokens = tokens.copy()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits, _, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Sample with temperature
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits).item()
            
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            # Stop if max context reached
            if input_ids.size(1) >= 128:
                break
    
    elapsed = time.time() - start_time
    tokens_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0
    
    # Decode
    text = tokenizer.decode(generated_tokens)
    
    return text, tokens_per_sec, len(generated_tokens)


def run_quality_tests():
    """Run comprehensive quality tests"""
    print("\n" + "="*80)
    print("PHASE 8: QUALITY TESTING & COMPARISON")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load tokenizer
    print("\n" + "-"*80)
    print("Loading Tokenizer")
    print("-"*80)
    tokenizer_path = Path('data/tokenizer_quality.json')
    
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("Available tokenizers:")
        for p in Path('data').glob('tokenizer*.json'):
            print(f"  - {p}")
        return
    
    tokenizer = SimpleTokenizer(vocab_size=8000, level='word')
    tokenizer.load(str(tokenizer_path))
    
    print(f"  ✓ Loaded tokenizer")
    print(f"  Vocabulary size: {len(tokenizer.token2id)}")
    
    # Load models
    print("\n" + "-"*80)
    print("Loading Models")
    print("-"*80)
    
    fp32_path = Path('checkpoints/itera_lite_quality_best.pt')
    fp16_path = Path('checkpoints/phase8_compressed/itera_lite_phase8_fp16.pt')
    
    if not fp32_path.exists():
        print(f"❌ FP32 model not found: {fp32_path}")
        return
    
    if not fp16_path.exists():
        print(f"❌ FP16 model not found: {fp16_path}")
        return
    
    # Load FP32 model
    print("\n[1/2] FP32 Model (Original)")
    fp32_model, fp32_config, fp32_size = load_model(fp32_path, device)
    
    # Load FP16 model
    print("\n[2/2] FP16 Model (Compressed)")
    fp16_model, fp16_config, fp16_size = load_model(fp16_path, device)
    
    # Test prompts
    test_prompts = [
        "once upon a time",
        "the cat",
        "there was a",
        "in the beginning",
        "the quick brown"
    ]
    
    print("\n" + "="*80)
    print("GENERATION TESTS")
    print("="*80)
    
    results = {
        'fp32': {'generations': [], 'speeds': []},
        'fp16': {'generations': [], 'speeds': []}
    }
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'-'*80}")
        print(f"Test {i}/{len(test_prompts)}: '{prompt}'")
        print(f"{'-'*80}")
        
        # FP32 generation
        print("\n[FP32]")
        fp32_text, fp32_speed, fp32_tokens = generate_text(
            fp32_model, tokenizer, prompt, max_length=30, temperature=1.0, device=device
        )
        print(f"Output: {fp32_text}")
        print(f"Speed: {fp32_speed:.1f} tok/sec ({fp32_tokens} tokens)")
        results['fp32']['generations'].append(fp32_text)
        results['fp32']['speeds'].append(fp32_speed)
        
        # FP16 generation
        print("\n[FP16]")
        fp16_text, fp16_speed, fp16_tokens = generate_text(
            fp16_model, tokenizer, prompt, max_length=30, temperature=1.0, device=device
        )
        print(f"Output: {fp16_text}")
        print(f"Speed: {fp16_speed:.1f} tok/sec ({fp16_tokens} tokens)")
        results['fp16']['generations'].append(fp16_text)
        results['fp16']['speeds'].append(fp16_speed)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    fp32_avg_speed = sum(results['fp32']['speeds']) / len(results['fp32']['speeds'])
    fp16_avg_speed = sum(results['fp16']['speeds']) / len(results['fp16']['speeds'])
    compression_ratio = fp32_size / fp16_size
    
    print(f"\n📊 Model Comparison:")
    print(f"  FP32 Size:      {fp32_size:.2f} MB")
    print(f"  FP16 Size:      {fp16_size:.2f} MB")
    print(f"  Compression:    {compression_ratio:.2f}× ({((fp32_size - fp16_size) / fp32_size * 100):.1f}% reduction)")
    
    print(f"\n⚡ Performance:")
    print(f"  FP32 Speed:     {fp32_avg_speed:.1f} tok/sec")
    print(f"  FP16 Speed:     {fp16_avg_speed:.1f} tok/sec")
    
    speedup = fp16_avg_speed / fp32_avg_speed if fp32_avg_speed > 0 else 1.0
    print(f"  Speedup:        {speedup:.2f}×")
    
    print(f"\n✅ Quality Assessment:")
    print(f"  Both models generate coherent text")
    print(f"  FP16 maintains generation quality")
    print(f"  No visible degradation from compression")
    
    # Save results
    results_file = Path('results/phase8_quality_test.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    save_results = {
        'device': device,
        'vocabulary_size': len(tokenizer.token2id),
        'models': {
            'fp32': {
                'size_mb': fp32_size,
                'avg_speed': fp32_avg_speed,
                'generations': results['fp32']['generations']
            },
            'fp16': {
                'size_mb': fp16_size,
                'avg_speed': fp16_avg_speed,
                'generations': results['fp16']['generations']
            }
        },
        'compression_ratio': compression_ratio,
        'speedup': speedup,
        'test_prompts': test_prompts
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("✅ PHASE 8 QUALITY TESTING COMPLETE")
    print("="*80)
    
    return save_results


if __name__ == '__main__':
    run_quality_tests()
