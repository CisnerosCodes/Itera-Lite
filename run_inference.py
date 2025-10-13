"""
Simple inference script for Itera-Lite model

This script shows you how to:
1. Load the trained model
2. Generate text
3. Run inference on your CPU

Usage:
    python run_inference.py
"""

import torch
import json
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.itera_lite import IteraLiteModel
from models.config import IteraLiteConfig


def load_model(checkpoint_path='checkpoints/itera_lite_tiny_best.pt', device='cpu'):
    """Load the trained model"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = IteraLiteConfig(**config_dict)
    else:
        # Fallback: infer from state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        config = IteraLiteConfig(
            vocab_size=state_dict['embedding.weight'].shape[0],
            hidden_size=state_dict['embedding.weight'].shape[1],
            num_layers=4,
            max_seq_length=128,
            ssm_state_size=8,
            num_experts=4,
            expert_size=64,
            top_k_experts=2
        )
    
    # Create model
    model = IteraLiteModel(config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    print(f"[OK] Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Max sequence length: {config.max_seq_length}")
    
    return model, config


def load_tokenizer(tokenizer_path='data/tokenizer_tiny.json'):
    """Load the tokenizer"""
    print(f"Loading tokenizer from {tokenizer_path}...")
    
    if not Path(tokenizer_path).exists():
        print(f"[WARN] Tokenizer not found at {tokenizer_path}")
        print(f"   Available tokenizers:")
        for p in Path('data').glob('tokenizer*.json'):
            print(f"   - {p}")
        return None
    
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    # Extract vocab
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    
    if not vocab:
        print("[WARN] Could not load vocab from tokenizer")
        return None
    
    print(f"[OK] Tokenizer loaded! Vocab size: {len(vocab)}")
    return vocab


def encode_text(text, vocab, max_length=128):
    """Simple encoding: convert text to token IDs"""
    # For character-level: use char codes
    token_ids = [ord(c) % len(vocab) for c in text]
    
    # Pad or truncate
    if len(token_ids) < max_length:
        token_ids = token_ids + [0] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]
    
    return torch.tensor([token_ids])


def decode_text(token_ids, vocab=None):
    """Simple decoding: convert token IDs to text"""
    # Character-level decoding
    chars = []
    for tid in token_ids[0]:
        c = chr(tid.item() % 128)
        if c.isprintable():
            chars.append(c)
        else:
            chars.append(' ')
    
    return ''.join(chars)


def generate_text(model, prompt="The quick brown", max_new_tokens=50, temperature=1.0, device='cpu'):
    """Generate text from a prompt"""
    print(f"\n{'='*70}")
    print(f"GENERATING TEXT")
    print(f"{'='*70}")
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    
    # Get vocab size
    vocab_size = model.embedding.weight.shape[0]
    
    # Simple character encoding
    input_ids = torch.tensor([[ord(c) % vocab_size for c in prompt]], device=device)
    
    generated_ids = input_ids.clone()
    
    print(f"\nGenerating", end='', flush=True)
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Get model output
            output = model(generated_ids)
            
            # Handle tuple output (logits, aux_loss)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Get last token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print('.', end='', flush=True)
            
            # Keep only last max_seq_length tokens
            if generated_ids.shape[1] > 128:
                generated_ids = generated_ids[:, -128:]
    
    print(' Done!\n')
    
    # Decode
    generated_text = decode_text(generated_ids)
    
    print(f"Generated text:")
    print(f"'{generated_text}'")
    print(f"\n(Note: Model was trained on small dataset, so output may not be coherent)")
    
    return generated_text


def run_benchmark(model, num_runs=10, seq_length=128, device='cpu'):
    """Benchmark inference speed"""
    import time
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING INFERENCE SPEED")
    print(f"{'='*70}")
    print(f"Runs: {num_runs}")
    print(f"Sequence length: {seq_length}")
    print(f"Device: {device}")
    
    vocab_size = model.embedding.weight.shape[0]
    input_ids = torch.randint(0, vocab_size, (1, seq_length), device=device)
    
    # Warmup
    print(f"\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    
    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids)
            end = time.perf_counter()
            times.append(end - start)
    
    # Results
    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    tokens_per_sec = seq_length / mean_time
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Mean time:       {mean_time*1000:.2f} ms")
    print(f"Min time:        {min_time*1000:.2f} ms")
    print(f"Max time:        {max_time*1000:.2f} ms")
    print(f"Throughput:      {tokens_per_sec:.1f} tokens/sec")
    print(f"{'='*70}\n")
    
    return tokens_per_sec


def main():
    """Main inference demo"""
    print("""
======================================================================
                ITERA-LITE MODEL INFERENCE DEMO
======================================================================
This script demonstrates how to:
1. Load your trained model
2. Generate text from a prompt
3. Benchmark inference speed on your CPU
======================================================================
    """)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cpu':
        print("(Using CPU - this is expected for your setup)")
    print()
    
    # Load model
    model, config = load_model(device=device)
    
    # Try to load tokenizer (optional)
    vocab = load_tokenizer()
    
    print("\n" + "="*70)
    
    # Generate text
    prompts = [
        "The quick brown",
        "Once upon a time",
        "In a world where"
    ]
    
    for prompt in prompts:
        generate_text(model, prompt=prompt, max_new_tokens=30, temperature=1.0, device=device)
    
    # Benchmark
    throughput = run_benchmark(model, num_runs=10, device=device)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"[OK] Model loaded and working!")
    print(f"[OK] Inference speed: {throughput:.1f} tokens/sec on {device.upper()}")
    print(f"[OK] Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"\nTo use the model in your own code:")
    print(f"```python")
    print(f"from run_inference import load_model, generate_text")
    print(f"")
    print(f"model, config = load_model()")
    print(f"text = generate_text(model, prompt='Your prompt here')")
    print(f"```")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
