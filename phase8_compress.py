"""
Phase 8B: Apply Phase 7 Compression to Quality-Trained Model
Takes the quality model and applies mixed-precision optimization
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from models import IteraLiteModel
from models.config import IteraLiteConfig
from utils.mixed_precision import MixedPrecisionConverter, get_conservative_precision_map
from utils.data import SimpleTokenizer


def load_quality_model():
    """Load the quality trained model"""
    print("\n" + "="*70)
    print("LOADING QUALITY MODEL")
    print("="*70)

    checkpoint_path = Path('checkpoints/itera_lite_quality_best.pt')

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Quality model not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create config
    config_dict = checkpoint['config']
    config = IteraLiteConfig(**config_dict)

    # Create model
    model = IteraLiteModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[OK] Loaded model from {checkpoint_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Training loss: {checkpoint.get('train_loss', 'N/A'):.4f}" if 'train_loss' in checkpoint else "")
    print(f"   Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if 'val_loss' in checkpoint else "")

    return model, config


def apply_mixed_precision(model, config, calibration_samples=100):
    """Apply mixed-precision optimization"""
    print("\n" + "="*70)
    print("APPLYING MIXED-PRECISION COMPRESSION")
    print("="*70)

    # Get precision map
    precision_map = get_conservative_precision_map()

    print("\nPrecision allocation:")
    print("  - Embeddings: INT8 (4x compression)")
    print("  - SSM layers: FP16 (2x compression)")
    print("  - Normalization: FP16 (2x compression)")
    print("  - LM head: INT8 (4x compression)")

    # Calculate original size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"\nOriginal model size: {original_size:.2f} MB")

    # Initialize converter
    converter = MixedPrecisionConverter(model, config)

    # Create dummy calibration data
    print(f"\nGenerating {calibration_samples} calibration samples...")
    calibration_data = []
    vocab_size = model.config.vocab_size
    seq_length = 128

    for _ in range(calibration_samples):
        # Random sequences for calibration
        sample = torch.randint(0, vocab_size, (1, seq_length))
        calibration_data.append(sample)

    # Calibrate
    print("\nCalibrating quantization parameters...")
    converter.calibrate(calibration_data)

    # Convert
    print("\nConverting to mixed-precision...")
    compressed_model = converter.convert(precision_map)

    # Calculate compressed size
    compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters()) / (1024 * 1024)
    compression_ratio = original_size / compressed_size

    print(f"\n[OK] Compression complete!")
    print(f"   Original size: {original_size:.2f} MB")
    print(f"   Compressed size: {compressed_size:.2f} MB")
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    print(f"   Memory saved: {original_size - compressed_size:.2f} MB")

    # Create stats
    stats = {
        'original_size_mb': float(original_size),
        'compressed_size_mb': float(compressed_size),
        'compression_ratio': float(compression_ratio),
        'memory_saved_mb': float(original_size - compressed_size),
    }

    return compressed_model, stats, compression_ratio


def save_compressed_model(model, config, stats, compression_ratio):
    """Save compressed model"""
    print("\n" + "="*70)
    print("SAVING COMPRESSED MODEL")
    print("="*70)

    output_dir = Path('checkpoints/phase8_compressed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'itera_lite_phase8_compressed.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'compression_ratio': compression_ratio,
        'compression_method': 'mixed_precision',
        'stats': stats,
    }, model_path)

    print(f"[OK] Saved compressed model to {model_path}")

    # Save metadata
    metadata = {
        'model_path': str(model_path),
        'compression_ratio': f"{compression_ratio:.2f}x",
        'compression_method': 'mixed-precision (INT8/FP16)',
        'original_model': 'checkpoints/itera_lite_quality_best.pt',
        'vocab_size': config.vocab_size,
        'parameters': sum(p.numel() for p in model.parameters()),
        'stats': stats,
    }

    metadata_path = output_dir / 'compression_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved metadata to {metadata_path}")

    return model_path, metadata_path


def test_compressed_model(model, config):
    """Test the compressed model"""
    print("\n" + "="*70)
    print("TESTING COMPRESSED MODEL")
    print("="*70)

    # Load tokenizer
    tokenizer = SimpleTokenizer(vocab_size=8000, level='word')
    tokenizer.load('data/tokenizer_quality.json')

    # Test prompts
    prompts = [
        "once upon a time",
        "the cat",
        "in the forest"
    ]

    print("\nGenerating sample text...")

    for prompt in prompts:
        # Encode
        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.special_tokens.get('<BOS>', 0)]

        input_ids = torch.tensor([tokens], dtype=torch.long)

        # Generate
        with torch.no_grad():
            generated = tokens.copy()
            for _ in range(20):
                logits, _, _ = model(input_ids)
                next_token_logits = logits[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                generated.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

                if input_ids.size(1) >= 128:
                    break

        # Decode
        text = tokenizer.decode(generated)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {text}")

    print("\n[OK] Compressed model generates text successfully!")


def main():
    print("\n" + "#"*70)
    print("#" + "PHASE 8B: COMPRESS QUALITY MODEL".center(68) + "#")
    print("#"*70)

    # Load quality model
    model, config = load_quality_model()

    # Apply compression
    compressed_model, stats, compression_ratio = apply_mixed_precision(model, config)

    # Save
    model_path, metadata_path = save_compressed_model(
        compressed_model, config, stats, compression_ratio
    )

    # Test
    test_compressed_model(compressed_model, config)

    # Summary
    print("\n" + "="*70)
    print("PHASE 8B COMPLETE!")
    print("="*70)
    print(f"\nCompression Summary:")
    print(f"  Method: Mixed-Precision (INT8/FP16)")
    print(f"  Compression: {compression_ratio:.2f}x")
    print(f"  Model: {model_path}")
    print(f"  Metadata: {metadata_path}")

    print("\nNext steps:")
    print("  1. Export to ONNX for production deployment")
    print("  2. Create deployment guide")
    print("  3. Write Phase 8 completion report")

    print("\nPhase 8 Progress: 2/4 tasks complete (Training + Compression) ✓")


if __name__ == "__main__":
    main()
