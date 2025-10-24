"""
Download and prepare WikiText-103 dataset using Hugging Face datasets
Phase 1: Dataset Upgrade (Alternative method using HF datasets)

This script:
1. Downloads WikiText-103 from Hugging Face datasets
2. Tokenizes using existing character-level tokenizer
3. Creates train/val/test splits
4. Saves preprocessed data efficiently
5. Reports comprehensive statistics
"""

import os
import sys
from pathlib import Path
import json
from typing import Tuple, List
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data import SimpleTokenizer


def download_wikitext103_hf(output_dir: Path) -> Tuple[str, str, str]:
    """Download WikiText-103 using Hugging Face datasets"""

    print("\n" + "="*70)
    print("DOWNLOADING WIKITEXT-103 VIA HUGGING FACE")
    print("="*70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] datasets library not installed!")
        print("Install with: pip install datasets")
        sys.exit(1)

    print("Loading WikiText-103 from Hugging Face...")
    print("(This may take a few minutes on first run)")

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=str(output_dir / "cache"))

    print(f"[OK] Dataset loaded!")
    print(f"  Splits: {list(dataset.keys())}")

    # Extract text from each split
    print("\nExtracting text from splits...")

    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']
    test_texts = dataset['test']['text']

    # Join all text (each element is a paragraph/line)
    train_text = '\n'.join([t for t in train_texts if t.strip()])
    val_text = '\n'.join([t for t in val_texts if t.strip()])
    test_text = '\n'.join([t for t in test_texts if t.strip()])

    print(f"[OK] Extracted text:")
    print(f"  Train: {len(train_text):,} characters ({len(train_text)/1e6:.2f} MB)")
    print(f"  Val:   {len(val_text):,} characters ({len(val_text)/1e6:.2f} MB)")
    print(f"  Test:  {len(test_text):,} characters ({len(test_text)/1e6:.2f} MB)")
    print(f"  Total: {len(train_text) + len(val_text) + len(test_text):,} characters")

    # Verify size
    total_mb = (len(train_text) + len(val_text) + len(test_text)) / 1e6
    if total_mb < 100:
        print(f"\n[WARN] Dataset seems small ({total_mb:.1f} MB). Expected 500+ MB.")
        print("This might be WikiText-2 instead of WikiText-103!")
    else:
        print(f"\n[OK] Dataset size verified: {total_mb:.1f} MB")

    return train_text, val_text, test_text


def create_tokenizer(train_text: str, val_text: str, vocab_size: int = 8000) -> SimpleTokenizer:
    """Create and train tokenizer on the data"""

    print("\n" + "="*70)
    print("BUILDING TOKENIZER")
    print("="*70)

    tokenizer = SimpleTokenizer(vocab_size=vocab_size, level='char')

    # Split into chunks for vocab building (more efficient)
    print("Building vocabulary from training data...")
    chunk_size = 10000
    chunks = [train_text[i:i+chunk_size] for i in range(0, min(len(train_text), 1000000), chunk_size)]

    # Add some validation data for better coverage
    chunks.extend([val_text[i:i+chunk_size] for i in range(0, min(len(val_text), 100000), chunk_size)])

    tokenizer.build_vocab(chunks)

    print(f"[OK] Tokenizer built with {len(tokenizer.token2id)} unique tokens")
    print(f"  Vocab coverage: {len(tokenizer.token2id)}/{vocab_size} slots used")

    # Show some examples
    print(f"\n  Sample tokens (first 20):")
    for i, (token, idx) in enumerate(list(tokenizer.token2id.items())[:20]):
        if token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
            print(f"    {idx}: {token}")
        else:
            print(f"    {idx}: '{token}'")

    return tokenizer


def tokenize_and_save(
    train_text: str,
    val_text: str,
    test_text: str,
    tokenizer: SimpleTokenizer,
    output_dir: Path
):
    """Tokenize all splits and save to disk"""

    print("\n" + "="*70)
    print("TOKENIZING DATA")
    print("="*70)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Tokenize each split
    print("Tokenizing train data (this may take a minute)...")
    train_tokens = tokenizer.encode(train_text)

    print("Tokenizing validation data...")
    val_tokens = tokenizer.encode(val_text)

    print("Tokenizing test data...")
    test_tokens = tokenizer.encode(test_text)

    # Save tokenized data
    print("\nSaving tokenized data...")

    with open(output_dir / "train_tokens.pkl", 'wb') as f:
        pickle.dump(train_tokens, f)

    with open(output_dir / "val_tokens.pkl", 'wb') as f:
        pickle.dump(val_tokens, f)

    with open(output_dir / "test_tokens.pkl", 'wb') as f:
        pickle.dump(test_tokens, f)

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    print(f"[OK] Saved tokenized data to {output_dir}")
    print(f"  - train_tokens.pkl: {len(train_tokens):,} tokens")
    print(f"  - val_tokens.pkl: {len(val_tokens):,} tokens")
    print(f"  - test_tokens.pkl: {len(test_tokens):,} tokens")
    print(f"  - tokenizer.json")

    return train_tokens, val_tokens, test_tokens


def print_statistics(
    train_tokens: List[int],
    val_tokens: List[int],
    test_tokens: List[int],
    tokenizer: SimpleTokenizer,
    output_dir: Path
):
    """Print comprehensive dataset statistics"""

    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    total_tokens = len(train_tokens) + len(val_tokens) + len(test_tokens)

    stats = {
        "dataset": "WikiText-103",
        "tokenization": "character-level",
        "vocab_size": len(tokenizer.token2id),
        "splits": {
            "train": {
                "tokens": len(train_tokens),
                "percentage": len(train_tokens) / total_tokens * 100
            },
            "val": {
                "tokens": len(val_tokens),
                "percentage": len(val_tokens) / total_tokens * 100
            },
            "test": {
                "tokens": len(test_tokens),
                "percentage": len(test_tokens) / total_tokens * 100
            }
        },
        "total_tokens": total_tokens,
        "sequences_128": total_tokens // 128,
        "sequences_256": total_tokens // 256,
        "sequences_512": total_tokens // 512,
    }

    print(f"\nDataset: {stats['dataset']}")
    print(f"Tokenization: {stats['tokenization']}")
    print(f"Vocabulary size: {stats['vocab_size']:,}")
    print(f"\nSplits:")
    print(f"  Train: {stats['splits']['train']['tokens']:,} tokens ({stats['splits']['train']['percentage']:.1f}%)")
    print(f"  Val:   {stats['splits']['val']['tokens']:,} tokens ({stats['splits']['val']['percentage']:.1f}%)")
    print(f"  Test:  {stats['splits']['test']['tokens']:,} tokens ({stats['splits']['test']['percentage']:.1f}%)")
    print(f"\nTotal tokens: {stats['total_tokens']:,}")
    print(f"\nEstimated training sequences:")
    print(f"  - Seq length 128: ~{stats['sequences_128']:,} sequences")
    print(f"  - Seq length 256: ~{stats['sequences_256']:,} sequences")
    print(f"  - Seq length 512: ~{stats['sequences_512']:,} sequences")

    # Estimate training time
    print(f"\nEstimated training time (rough estimates):")
    print(f"  At 1000 tokens/sec:")
    print(f"    - 1 epoch: ~{total_tokens / 1000 / 60:.1f} minutes")
    print(f"    - 1000 epochs: ~{total_tokens / 1000 / 3600 * 1000:.1f} hours")
    print(f"  At 100 tokens/sec (slower CPU):")
    print(f"    - 1 epoch: ~{total_tokens / 100 / 60:.1f} minutes")
    print(f"    - 1000 epochs: ~{total_tokens / 100 / 3600 * 1000:.1f} hours")

    # Save statistics
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n[OK] Saved statistics to {stats_path}")

    # Sample some text
    print("\n" + "="*70)
    print("SAMPLE TEXT (first 500 tokens decoded)")
    print("="*70)
    sample = tokenizer.decode(train_tokens[:500])
    print(sample[:500] if len(sample) >= 500 else sample)
    if len(sample) > 500:
        print("...")


def main():
    """Main function"""

    print("\n" + "#"*70)
    print("#" + "WIKITEXT-103 DATASET PREPARATION (HF)".center(68) + "#")
    print("#"*70 + "\n")

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = data_dir / "wikitext103"

    try:
        # Step 1: Download WikiText-103 via Hugging Face
        train_text, val_text, test_text = download_wikitext103_hf(data_dir)

        # Step 2: Create tokenizer
        tokenizer = create_tokenizer(train_text, val_text, vocab_size=8000)

        # Step 3: Tokenize and save
        train_tokens, val_tokens, test_tokens = tokenize_and_save(
            train_text, val_text, test_text, tokenizer, output_dir
        )

        # Step 4: Print statistics
        print_statistics(train_tokens, val_tokens, test_tokens, tokenizer, output_dir)

        print("\n" + "#"*70)
        print("#" + "DATASET PREPARATION COMPLETE!".center(68) + "#")
        print("#"*70 + "\n")

        print("Validation checklist:")
        total_mb = (len(train_text) + len(val_text) + len(test_text)) / 1e6
        if total_mb > 100:
            print("  [OK] Dataset size > 100 MB")
        else:
            print(f"  [WARN] Dataset size is {total_mb:.1f} MB (expected 500+ MB)")

        print(f"  [OK] Train tokens: {len(train_tokens):,}")
        print(f"  [OK] Tokenizer vocab: {len(tokenizer.token2id)}")
        print(f"  [OK] Files saved to: {output_dir}")

        print("\nNext steps:")
        print("  1. Proceed to Phase 2: Add metrics tracking (perplexity, generation)")
        print("  2. Proceed to Phase 3: Optimize hyperparameters")
        print("  3. Proceed to Phase 4: Create production training script")
        print("  4. Start training for 1000+ epochs!")

        print("\n[OK] Ready for quality training!")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
