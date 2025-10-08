"""
Real-world dataset validation for WikiText-2 and TinyStories.
Measures perplexity and quality across different quantization levels.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WikiText2Dataset(Dataset):
    """WikiText-2 dataset for language modeling evaluation."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        """
        Initialize WikiText-2 dataset.
        
        Args:
            data_path: Path to WikiText-2 text file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load text
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Simple character-level tokenization for demo
        # In production, use proper tokenizer
        self.vocab = sorted(set(self.text))
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.char_to_id['<PAD>'] = len(self.char_to_id)
        self.char_to_id['<UNK>'] = len(self.char_to_id)
        
        # Tokenize
        self.tokens = [self.char_to_id.get(ch, self.char_to_id['<UNK>']) for ch in self.text]
        
        # Create samples
        self.samples = []
        for i in range(0, len(self.tokens) - max_length - 1, max_length):
            input_ids = self.tokens[i:i + max_length]
            labels = self.tokens[i + 1:i + max_length + 1]
            self.samples.append((input_ids, labels))
        
        logger.info(f"Loaded WikiText-2: {len(self.samples)} samples, vocab size {len(self.vocab)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_ids, labels = self.samples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for language modeling evaluation."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        """
        Initialize TinyStories dataset.
        
        Args:
            data_path: Path to TinyStories text file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load stories
        with open(data_path, 'r', encoding='utf-8') as f:
            self.stories = [line.strip() for line in f if line.strip()]
        
        # Simple character-level tokenization
        all_text = ' '.join(self.stories)
        self.vocab = sorted(set(all_text))
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.char_to_id['<PAD>'] = len(self.char_to_id)
        self.char_to_id['<UNK>'] = len(self.char_to_id)
        
        # Create samples from stories
        self.samples = []
        for story in self.stories:
            tokens = [self.char_to_id.get(ch, self.char_to_id['<UNK>']) for ch in story]
            
            for i in range(0, len(tokens) - max_length - 1, max_length // 2):
                if i + max_length + 1 <= len(tokens):
                    input_ids = tokens[i:i + max_length]
                    labels = tokens[i + 1:i + max_length + 1]
                    self.samples.append((input_ids, labels))
        
        logger.info(f"Loaded TinyStories: {len(self.samples)} samples, vocab size {len(self.vocab)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_ids, labels = self.samples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )


def download_wikitext2(data_dir: str = "data/datasets") -> str:
    """
    Download WikiText-2 dataset.
    For now, creates a synthetic version. In production, download from official source.
    
    Args:
        data_dir: Directory to save dataset
        
    Returns:
        Path to dataset file
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "wikitext2_test.txt"
    
    if output_file.exists():
        logger.info(f"WikiText-2 already exists at {output_file}")
        return str(output_file)
    
    # Create synthetic WikiText-2 style text
    logger.info("Creating synthetic WikiText-2 dataset...")
    
    sample_text = """
= Introduction =

Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data, without being explicitly programmed. The name machine learning was coined in 1959 by Arthur Samuel.

= History =

The study of machine learning began in the 1950s. Early work focused on pattern recognition and computational learning theory. In the 1990s, machine learning evolved from the field of artificial intelligence. Researchers shifted from knowledge-driven approaches to data-driven approaches.

= Applications =

Machine learning has many applications including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine. Applications range from simple pattern recognition to complex decision making systems.

= Deep Learning =

Deep learning is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures have been applied to fields including computer vision and natural language processing.
""" * 20  # Repeat to create more data
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    logger.info(f"Created synthetic WikiText-2 at {output_file}")
    return str(output_file)


def download_tinystories(data_dir: str = "data/datasets") -> str:
    """
    Get TinyStories dataset.
    Uses existing synthetic stories or creates new ones.
    
    Args:
        data_dir: Directory to save dataset
        
    Returns:
        Path to dataset file
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have tinystories from Phase 4
    phase4_file = data_dir / "tinystories_train.txt"
    if phase4_file.exists():
        logger.info(f"Using existing TinyStories from {phase4_file}")
        return str(phase4_file)
    
    output_file = data_dir / "tinystories_test.txt"
    if output_file.exists():
        logger.info(f"TinyStories already exists at {output_file}")
        return str(output_file)
    
    # Create synthetic stories
    logger.info("Creating synthetic TinyStories dataset...")
    
    sample_stories = [
        "Once upon a time, there was a little girl named Lily. She loved to play in the park.",
        "One day, a big dog came running. Lily was scared at first, but the dog was friendly.",
        "The dog's name was Max. Max liked to play fetch with his red ball.",
        "Lily threw the ball far. Max ran fast and caught it in his mouth.",
        "They played together all day. When the sun set, Lily had to go home.",
        "Tomorrow I will come back, said Lily. Max wagged his tail happily.",
        "Tim had a toy car. It was blue and shiny. He played with it every day.",
        "One morning, the car would not move. Tim was sad. His mom helped him fix it.",
        "They found a small rock stuck in the wheel. Mom took it out carefully.",
        "The car worked again! Tim was so happy. He drove it around the room.",
    ] * 50  # Repeat to create more stories
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for story in sample_stories:
            f.write(story + '\n')
    
    logger.info(f"Created TinyStories at {output_file}")
    return str(output_file)


def calculate_perplexity(
    model,
    dataloader: DataLoader,
    device: str = 'cpu',
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for dataset
        device: Device to run on
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        Dictionary with perplexity and loss metrics
    """
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Calculating perplexity")
        for batch_idx, (input_ids, labels) in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # Extract logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate loss
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += batch_size * seq_len
            num_batches += 1
            
            # Update progress
            current_ppl = np.exp(total_loss / total_tokens)
            pbar.set_postfix({'perplexity': f'{current_ppl:.2f}'})
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
        'num_batches': num_batches
    }


def evaluate_model_quality(
    model,
    dataset_name: str,
    dataset_path: str,
    batch_size: int = 8,
    max_batches: Optional[int] = 100,
    device: str = 'cpu'
) -> Dict:
    """
    Evaluate model quality on a dataset.
    
    Args:
        model: Model to evaluate
        dataset_name: Name of dataset ('wikitext2' or 'tinystories')
        dataset_path: Path to dataset file
        batch_size: Batch size for evaluation
        max_batches: Maximum batches to evaluate
        device: Device to run on
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating on {dataset_name}...")
    
    # Create dataset
    if dataset_name.lower() == 'wikitext2':
        dataset = WikiText2Dataset(dataset_path, tokenizer=None, max_length=128)
    elif dataset_name.lower() == 'tinystories':
        dataset = TinyStoriesDataset(dataset_path, tokenizer=None, max_length=128)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Calculate perplexity
    results = calculate_perplexity(model, dataloader, device, max_batches)
    
    logger.info(f"{dataset_name} Results:")
    logger.info(f"  Perplexity: {results['perplexity']:.2f}")
    logger.info(f"  Avg Loss: {results['avg_loss']:.4f}")
    logger.info(f"  Tokens: {results['total_tokens']}")
    
    return results


def compare_model_variants(
    model_paths: Dict[str, str],
    model_class,
    config,
    dataset_name: str,
    dataset_path: str,
    batch_size: int = 8,
    max_batches: int = 100
) -> Dict:
    """
    Compare different model variants (FP32, INT8, INT4) on quality metrics.
    
    Args:
        model_paths: Dictionary of variant name -> checkpoint path
        model_class: Model class to instantiate
        config: Model configuration
        dataset_name: Dataset to evaluate on
        dataset_path: Path to dataset
        batch_size: Batch size
        max_batches: Maximum batches
        
    Returns:
        Comparison results
    """
    results = {}
    
    for variant_name, checkpoint_path in model_paths.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating {variant_name}")
        logger.info(f"{'=' * 60}")
        
        # Load model
        model = model_class(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Evaluate
        variant_results = evaluate_model_quality(
            model=model,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            batch_size=batch_size,
            max_batches=max_batches
        )
        
        results[variant_name] = variant_results
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Download datasets
    wikitext_path = download_wikitext2()
    tinystories_path = download_tinystories()
    
    print(f"\nDatasets ready:")
    print(f"  WikiText-2: {wikitext_path}")
    print(f"  TinyStories: {tinystories_path}")
