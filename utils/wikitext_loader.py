"""
WikiText-103 dataset loader
Loads preprocessed tokenized data from Phase 1
"""

import pickle
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from .data import TextDataset, SimpleTokenizer


def load_wikitext103(
    data_dir: str = 'data/wikitext103',
    seq_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, SimpleTokenizer]:
    """
    Load preprocessed WikiText-103 dataset

    Args:
        data_dir: Directory containing preprocessed data
        seq_length: Sequence length for training
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    data_path = Path(data_dir)

    # Check if data exists
    required_files = [
        'train_tokens.pkl',
        'val_tokens.pkl',
        'test_tokens.pkl',
        'tokenizer.json'
    ]

    for file in required_files:
        if not (data_path / file).exists():
            raise FileNotFoundError(
                f"Required file not found: {data_path / file}\n"
                f"Please run: python data/download_wikitext103_hf.py"
            )

    print(f"Loading WikiText-103 from {data_path}...")

    # Load tokenized data
    print("  Loading train tokens...")
    with open(data_path / 'train_tokens.pkl', 'rb') as f:
        train_tokens = pickle.load(f)

    print("  Loading val tokens...")
    with open(data_path / 'val_tokens.pkl', 'rb') as f:
        val_tokens = pickle.load(f)

    print("  Loading test tokens...")
    with open(data_path / 'test_tokens.pkl', 'rb') as f:
        test_tokens = pickle.load(f)

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.load(str(data_path / 'tokenizer.json'))

    print(f"\n[OK] Dataset loaded:")
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val:   {len(val_tokens):,} tokens")
    print(f"  Test:  {len(test_tokens):,} tokens")
    print(f"  Vocab: {len(tokenizer.token2id)} tokens")

    # Create datasets
    train_dataset = TextDataset(train_tokens, seq_length=seq_length)
    val_dataset = TextDataset(val_tokens, seq_length=seq_length)
    test_dataset = TextDataset(test_tokens, seq_length=seq_length)

    print(f"\n  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences:   {len(val_dataset):,}")
    print(f"  Test sequences:  {len(test_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"\n  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")
    print(f"  Test batches:  {len(test_loader):,}")

    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    print("Testing WikiText-103 loader...")

    # Test loading
    train_loader, val_loader, test_loader, tokenizer = load_wikitext103(
        data_dir='../data/wikitext103',
        seq_length=128,
        batch_size=8
    )

    # Get a batch
    for input_ids, target_ids in train_loader:
        print(f"\nSample batch:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")

        # Decode sample
        sample_text = tokenizer.decode(input_ids[0].tolist()[:100])
        print(f"  Sample text: {sample_text[:200]}")
        break

    print("\n[OK] WikiText-103 loader working!")
