"""
Download and prepare WikiText-103 dataset for training
Phase 1: Dataset Upgrade

This script:
1. Downloads the full WikiText-103 dataset (NOT WikiText-2)
2. Tokenizes using existing character-level tokenizer
3. Creates train/val/test splits
4. Saves preprocessed data efficiently
5. Reports comprehensive statistics
"""

import os
import sys
from pathlib import Path
import requests
import zipfile
import json
from typing import Tuple, List
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data import SimpleTokenizer


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar"""
    print(f"Downloading from {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"[OK] Downloaded to {output_path}")


def download_wikitext103(data_dir: Path) -> Path:
    """Download WikiText-103 dataset"""

    # WikiText-103 URL
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

    zip_path = data_dir / "wikitext-103-raw-v1.zip"
    extract_dir = data_dir / "wikitext103"

    # Download if not exists
    if not zip_path.exists():
        print("\n" + "="*70)
        print("DOWNLOADING WIKITEXT-103 DATASET")
        print("="*70)
        data_dir.mkdir(exist_ok=True, parents=True)

        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"\n[WARN] Download failed: {e}")
            print("\nAlternative: Download manually from:")
            print(f"  {url}")
            print(f"  Save to: {zip_path}")
            raise
    else:
        print(f"[OK] Found existing download: {zip_path}")

    # Extract if not already extracted
    if not extract_dir.exists():
        print("\n" + "="*70)
        print("EXTRACTING DATASET")
        print("="*70)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        print(f"[OK] Extracted to {extract_dir}")
    else:
        print(f"[OK] Found existing extracted data: {extract_dir}")

    return extract_dir


def load_wikitext_splits(extract_dir: Path) -> Tuple[str, str, str]:
    """Load train/val/test splits from WikiText-103"""

    print("\n" + "="*70)
    print("LOADING DATA SPLITS")
    print("="*70)

    # WikiText-103 has this directory structure
    base_dir = extract_dir / "wikitext-103-raw"

    train_file = base_dir / "wiki.train.raw"
    val_file = base_dir / "wiki.valid.raw"
    test_file = base_dir / "wiki.test.raw"

    # Check if files exist
    for name, path in [("train", train_file), ("val", val_file), ("test", test_file)]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find {name} file at {path}")

    # Load files
    print("Loading train data...")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()

    print("Loading validation data...")
    with open(val_file, 'r', encoding='utf-8') as f:
        val_text = f.read()

    print("Loading test data...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_text = f.read()

    # Print statistics
    print(f"\n[OK] Loaded splits:")
    print(f"  Train: {len(train_text):,} characters ({len(train_text)/1e6:.2f} MB)")
    print(f"  Val:   {len(val_text):,} characters ({len(val_text)/1e6:.2f} MB)")
    print(f"  Test:  {len(test_text):,} characters ({len(test_text)/1e6:.2f} MB)")
    print(f"  Total: {len(train_text) + len(val_text) + len(test_text):,} characters")

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
    print("Tokenizing train data...")
    train_tokens = tokenizer.encode(train_text)

    print("Tokenizing validation data...")
    val_tokens = tokenizer.encode(val_text)

    print("Tokenizing test data...")
    test_tokens = tokenizer.encode(test_text)

    # Save tokenized data
    print("\nSaving tokenized data...")

    import pickle

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
    print(f"\nEstimated sequences:")
    print(f"  - Seq length 128: ~{stats['sequences_128']:,} sequences")
    print(f"  - Seq length 256: ~{stats['sequences_256']:,} sequences")
    print(f"  - Seq length 512: ~{stats['sequences_512']:,} sequences")

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
    print(sample[:500])
    print("...")


def main():
    """Main function"""

    print("\n" + "#"*70)
    print("#" + "WIKITEXT-103 DATASET PREPARATION".center(68) + "#")
    print("#"*70 + "\n")

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = data_dir / "wikitext103"

    try:
        # Step 1: Download WikiText-103
        extract_dir = download_wikitext103(data_dir)

        # Step 2: Load splits
        train_text, val_text, test_text = load_wikitext_splits(extract_dir)

        # Step 3: Create tokenizer
        tokenizer = create_tokenizer(train_text, val_text, vocab_size=8000)

        # Step 4: Tokenize and save
        train_tokens, val_tokens, test_tokens = tokenize_and_save(
            train_text, val_text, test_text, tokenizer, output_dir
        )

        # Step 5: Print statistics
        print_statistics(train_tokens, val_tokens, test_tokens, tokenizer, output_dir)

        print("\n" + "#"*70)
        print("#" + "DATASET PREPARATION COMPLETE!".center(68) + "#")
        print("#"*70 + "\n")

        print("Next steps:")
        print("  1. Verify dataset size is > 100 MB ✓")
        print("  2. Check statistics in data/wikitext103/dataset_stats.json")
        print("  3. Proceed to Phase 2: Add metrics tracking")
        print("  4. Update train.py to use this dataset")

        print("\n[OK] Ready for training!")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
