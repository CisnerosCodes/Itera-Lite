"""
Data utilities for Itera-Lite training
Supports TinyStories, WikiText, and custom text datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from typing import List, Tuple, Optional
import random


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(
        self,
        data: List[int],
        seq_length: int = 128,
        stride: Optional[int] = None
    ):
        """
        Args:
            data: List of token IDs
            seq_length: Sequence length for each sample
            stride: Stride for creating samples (default: seq_length for non-overlapping)
        """
        self.data = data
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        
        # Calculate number of samples
        self.num_samples = max(1, (len(data) - seq_length) // self.stride)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_length + 1  # +1 for target
        
        # Get sequence
        sequence = self.data[start_idx:end_idx]
        
        # Pad if necessary
        if len(sequence) < self.seq_length + 1:
            sequence = sequence + [0] * (self.seq_length + 1 - len(sequence))
        
        # Split into input and target
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids


class SimpleTokenizer:
    """Ultra-simple character/word-level tokenizer for quick prototyping"""
    
    def __init__(self, vocab_size: int = 8000, level: str = 'char'):
        """
        Args:
            vocab_size: Maximum vocabulary size
            level: 'char' for character-level, 'word' for word-level
        """
        self.vocab_size = vocab_size
        self.level = level
        self.token2id = {}
        self.id2token = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.next_id = len(self.special_tokens)
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token2id[token] = idx
            self.id2token[idx] = token
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Count tokens
        token_counts = {}
        
        for text in texts:
            if self.level == 'char':
                tokens = list(text)
            else:  # word level
                tokens = text.lower().split()
            
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Sort by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary up to vocab_size
        for token, count in sorted_tokens[:self.vocab_size - len(self.special_tokens)]:
            if token not in self.token2id:
                self.token2id[token] = self.next_id
                self.id2token[self.next_id] = token
                self.next_id += 1
        
        print(f"Built vocabulary with {len(self.token2id)} tokens ({self.level}-level)")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.level == 'char':
            tokens = list(text)
        else:
            tokens = text.lower().split()
        
        return [self.token2id.get(token, self.special_tokens['<UNK>']) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.id2token.get(idx, '<UNK>') for idx in token_ids]
        
        if self.level == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def save(self, path: str):
        """Save tokenizer"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'token2id': self.token2id,
                'id2token': {str(k): v for k, v in self.id2token.items()},
                'vocab_size': self.vocab_size,
                'level': self.level
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load tokenizer"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.token2id = data['token2id']
            self.id2token = {int(k): v for k, v in data['id2token'].items()}
            self.vocab_size = data['vocab_size']
            self.level = data['level']
            self.next_id = len(self.token2id)


def create_simple_dataset(
    text_file: Optional[str] = None,
    num_samples: int = 1000,
    vocab_size: int = 8000,
    seq_length: int = 128,
    level: str = 'char'
) -> Tuple[TextDataset, TextDataset, SimpleTokenizer]:
    """
    Create a simple dataset from text file or generate synthetic data
    
    Args:
        text_file: Path to text file (if None, generates synthetic data)
        num_samples: Number of samples for synthetic data
        vocab_size: Vocabulary size
        seq_length: Sequence length
        level: 'char' or 'word' tokenization
    
    Returns:
        train_dataset, val_dataset, tokenizer
    """
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, level=level)
    
    if text_file and os.path.exists(text_file):
        # Load from file
        print(f"Loading data from {text_file}...")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into sentences or chunks
        if level == 'char':
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        else:
            chunks = text.split('\n')
        
        chunks = [c.strip() for c in chunks if c.strip()]
        
    else:
        # Generate synthetic data
        print(f"Generating {num_samples} synthetic samples...")
        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a test sentence.",
            "Machine learning models are trained on data.",
            "Natural language processing is fascinating.",
            "Deep learning has revolutionized AI research.",
            "Transformers and state space models are powerful.",
            "Efficiency is key for large-scale deployment.",
            "The future of AI is bright and promising.",
        ]
        
        chunks = []
        for _ in range(num_samples):
            # Randomly combine templates
            chunk = ' '.join(random.choices(templates, k=random.randint(2, 5)))
            chunks.append(chunk)
    
    print(f"Total chunks: {len(chunks)}")
    
    # Build vocabulary
    tokenizer.build_vocab(chunks)
    
    # Encode all text
    all_tokens = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk)
        all_tokens.extend(tokens)
    
    print(f"Total tokens: {len(all_tokens)}")
    
    # Split train/val (90/10)
    split_idx = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Val tokens: {len(val_tokens)}")
    
    # Create datasets
    train_dataset = TextDataset(train_tokens, seq_length=seq_length)
    val_dataset = TextDataset(val_tokens, seq_length=seq_length)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset, tokenizer


def get_dataloader(
    dataset: TextDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a dataloader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False  # CPU training
    )


if __name__ == "__main__":
    print("Testing data utilities...")
    
    # Test simple dataset creation
    train_ds, val_ds, tokenizer = create_simple_dataset(
        num_samples=100,
        vocab_size=500,
        seq_length=32,
        level='char'
    )
    
    # Test dataloader
    train_loader = get_dataloader(train_ds, batch_size=4)
    
    # Get a batch
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")
        print(f"  Sample input: {input_ids[0][:20]}")
        
        # Decode sample
        sample_text = tokenizer.decode(input_ids[0].tolist()[:50])
        print(f"  Decoded: {sample_text[:100]}")
        break
    
    print("\n✓ Data utilities working!")
