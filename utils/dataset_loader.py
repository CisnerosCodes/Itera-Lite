"""
Real dataset loading utilities for TinyStories and WikiText-2
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset


class RealDatasetLoader:
    """Load and prepare real datasets for training"""
    
    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_tinystories(self, num_samples: int = 10000) -> str:
        """
        Download TinyStories dataset samples
        
        Args:
            num_samples: Number of stories to download
            
        Returns:
            Path to saved dataset file
        """
        output_file = self.data_dir / "tinystories_train.txt"
        
        if output_file.exists():
            print(f"TinyStories dataset already exists at {output_file}")
            return str(output_file)
        
        print(f"Downloading TinyStories dataset ({num_samples} samples)...")
        
        # For this prototype, we'll create synthetic TinyStories-like data
        # In production, use: https://huggingface.co/datasets/roneneldan/TinyStories
        stories = []
        story_templates = [
            "Once upon a time, there was a {adj1} {noun1}. The {noun1} loved to {verb1} in the {place}. One day, the {noun1} met a {adj2} {noun2}. They became best friends and {verb2} together every day.",
            "A little {noun1} wanted to {verb1}. The {noun1}'s {relation} said it was too {adj1}. But the {noun1} tried anyway. In the end, the {noun1} learned that {moral}.",
            "There was a {adj1} {noun1} who lived in a {place}. Every morning, the {noun1} would {verb1}. One {time}, something {adj2} happened. The {noun1} had to {verb2}.",
            "{name} was a {adj1} {noun1}. {name} liked to {verb1} with friends. One day, {name} found a {noun2}. It was very {adj2}! {name} decided to {verb2}.",
        ]
        
        words = {
            'adj1': ['happy', 'sad', 'little', 'big', 'brave', 'curious', 'kind', 'smart'],
            'adj2': ['magic', 'special', 'wonderful', 'scary', 'funny', 'strange', 'shiny', 'beautiful'],
            'noun1': ['cat', 'dog', 'bird', 'rabbit', 'bear', 'fox', 'mouse', 'squirrel'],
            'noun2': ['friend', 'toy', 'ball', 'book', 'flower', 'tree', 'rock', 'star'],
            'verb1': ['play', 'jump', 'run', 'dance', 'sing', 'explore', 'learn', 'help'],
            'verb2': ['share', 'laugh', 'work', 'adventure', 'discover', 'celebrate', 'create', 'dream'],
            'place': ['park', 'forest', 'garden', 'home', 'school', 'beach', 'mountain', 'meadow'],
            'relation': ['mom', 'dad', 'friend', 'teacher', 'sibling', 'grandma', 'grandpa', 'neighbor'],
            'moral': ['trying is important', 'friends help each other', 'being kind matters', 'practice makes perfect'],
            'time': ['morning', 'afternoon', 'evening', 'night', 'day', 'week', 'month', 'year'],
            'name': ['Lily', 'Max', 'Sam', 'Emma', 'Jack', 'Mia', 'Leo', 'Zoe']
        }
        
        import random
        random.seed(42)
        
        for i in range(num_samples):
            template = random.choice(story_templates)
            story = template
            for key, values in words.items():
                if f'{{{key}}}' in story:
                    story = story.replace(f'{{{key}}}', random.choice(values))
            stories.append(story)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(stories))
        
        print(f"Created {len(stories)} synthetic stories at {output_file}")
        return str(output_file)
    
    def download_wikitext2(self) -> str:
        """
        Download WikiText-2 dataset
        
        Returns:
            Path to saved dataset file
        """
        output_file = self.data_dir / "wikitext2_train.txt"
        
        if output_file.exists():
            print(f"WikiText-2 dataset already exists at {output_file}")
            return str(output_file)
        
        print("Downloading WikiText-2 dataset...")
        
        # For this prototype, create synthetic Wikipedia-like text
        # In production, use: https://huggingface.co/datasets/wikitext
        articles = [
            "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems improve their performance over time without being explicitly programmed.",
            "The Python programming language was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with significant whitespace.",
            "Natural language processing enables computers to understand, interpret, and generate human language. It combines computational linguistics with machine learning.",
            "Deep learning uses artificial neural networks with multiple layers. These networks can learn complex patterns in large datasets.",
            "Computer vision allows machines to derive meaningful information from digital images and videos. Applications include facial recognition and autonomous vehicles.",
            "Reinforcement learning trains agents to make decisions by rewarding desired behaviors. The agent learns through trial and error.",
            "Data science combines statistics, mathematics, and computer science to extract insights from data. It is used across many industries.",
            "Cloud computing delivers computing services over the internet. It provides scalable resources on demand.",
        ]
        
        # Repeat and shuffle to create larger dataset
        import random
        random.seed(42)
        full_text = []
        for _ in range(500):
            full_text.extend(random.sample(articles, len(articles)))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(full_text))
        
        print(f"Created WikiText-2-like dataset at {output_file}")
        return str(output_file)
    
    def load_text_file(self, filepath: str) -> str:
        """Load text from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


class FrequencyTokenizer:
    """Build tokenizer based on token frequency"""
    
    def __init__(self, vocab_size: int = 2000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        Build vocabulary from texts based on word frequency
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a word to be included
        """
        from collections import Counter
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Sort by frequency
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # Build vocabulary
        self.token_to_id = self.special_tokens.copy()
        
        for word, count in most_common:
            if count >= min_freq and len(self.token_to_id) < self.vocab_size:
                self.token_to_id[word] = len(self.token_to_id)
        
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        print(f"Built vocabulary with {len(self.token_to_id)} tokens")
        print(f"Top 10 tokens: {most_common[:10]}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        words = text.lower().split()
        return [self.token_to_id.get(word, self.token_to_id['<UNK>']) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.id_to_token.get(idx, '<UNK>') for idx in ids]
        return ' '.join(tokens)
    
    def save(self, filepath: str):
        """Save tokenizer to JSON file"""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Saved tokenizer to {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        print(f"Loaded tokenizer from {filepath} ({len(self.token_to_id)} tokens)")


class TokenizedDataset(Dataset):
    """PyTorch dataset for tokenized text"""
    
    def __init__(self, text: str, tokenizer: FrequencyTokenizer, seq_length: int = 128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize entire text
        self.tokens = tokenizer.encode(text)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_length, seq_length // 2):
            seq = self.tokens[i:i + seq_length + 1]
            if len(seq) == seq_length + 1:
                self.sequences.append(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Return tuple for compatibility with Trainer
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, labels


def prepare_dataset(
    dataset_name: str = 'tinystories',
    vocab_size: int = 2000,
    seq_length: int = 128,
    num_samples: int = 10000
) -> Tuple[TokenizedDataset, TokenizedDataset, FrequencyTokenizer]:
    """
    Prepare dataset for training
    
    Args:
        dataset_name: 'tinystories' or 'wikitext2'
        vocab_size: Target vocabulary size
        seq_length: Sequence length for training
        num_samples: Number of samples (for TinyStories)
        
    Returns:
        train_dataset, val_dataset, tokenizer
    """
    loader = RealDatasetLoader()
    
    # Download dataset
    if dataset_name == 'tinystories':
        data_file = loader.download_tinystories(num_samples)
    elif dataset_name == 'wikitext2':
        data_file = loader.download_wikitext2()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load text
    text = loader.load_text_file(data_file)
    
    # Split into train/val
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Build tokenizer
    tokenizer = FrequencyTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab([train_text])
    
    # Create datasets
    train_dataset = TokenizedDataset(train_text, tokenizer, seq_length)
    val_dataset = TokenizedDataset(val_text, tokenizer, seq_length)
    
    print(f"\nDataset prepared:")
    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Val sequences: {len(val_dataset)}")
    print(f"  Vocabulary size: {len(tokenizer.token_to_id)}")
    print(f"  Sequence length: {seq_length}")
    
    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    train_ds, val_ds, tokenizer = prepare_dataset(
        dataset_name='tinystories',
        vocab_size=2000,
        seq_length=128,
        num_samples=1000
    )
    
    # Test tokenization
    sample = train_ds[0]
    print(f"\nSample sequence:")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Decoded: {tokenizer.decode(sample['input_ids'].tolist()[:50])}...")
    
    # Save tokenizer
    tokenizer.save('data/tokenizer_2k.json')
