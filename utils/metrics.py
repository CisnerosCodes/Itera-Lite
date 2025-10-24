"""
Comprehensive metrics tracking for language model training
Phase 2: Metrics System

Includes:
- Perplexity calculation
- Generation quality sampling
- TensorBoard logging
- CSV backup
- Early stopping logic
"""

import torch
import torch.nn as nn
import math
import csv
from pathlib import Path
from typing import Optional, List, Dict
import time


class MetricsTracker:
    """
    Comprehensive metrics tracking for language model training

    Tracks:
    - Training/validation loss
    - Perplexity (train/val)
    - Learning rate
    - Generation samples
    - Throughput (tokens/sec)
    """

    def __init__(
        self,
        model_name: str,
        log_dir: str = 'runs',
        use_tensorboard: bool = True,
        csv_backup: bool = True
    ):
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.use_tensorboard = use_tensorboard
        self.csv_backup = csv_backup

        # Create directories
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / model_name))
                print(f"[OK] TensorBoard logging enabled: {self.log_dir / model_name}")
            except ImportError:
                print("[WARN] tensorboard not installed. Install with: pip install tensorboard")
                self.use_tensorboard = False

        # CSV log
        if csv_backup:
            self.csv_path = self.log_dir / f'{model_name}_metrics.csv'
            self._init_csv()

        # Metrics history
        self.history = {
            'step': [],
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'learning_rate': [],
            'tokens_per_sec': [],
        }

    def _init_csv(self):
        """Initialize CSV log file"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'train_loss', 'val_loss',
                'train_perplexity', 'val_perplexity',
                'lr', 'tokens_per_sec', 'elapsed_time'
            ])

    def log_metrics(
        self,
        step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        elapsed_time: Optional[float] = None
    ):
        """Log metrics to all enabled backends"""

        # Calculate perplexity from loss
        train_ppl = math.exp(min(train_loss, 20)) if train_loss is not None else None
        val_ppl = math.exp(min(val_loss, 20)) if val_loss is not None else None

        # Update history
        self.history['step'].append(step)
        self.history['epoch'].append(epoch)
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
            self.history['train_perplexity'].append(train_ppl)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            self.history['val_perplexity'].append(val_ppl)
        if lr is not None:
            self.history['learning_rate'].append(lr)
        if tokens_per_sec is not None:
            self.history['tokens_per_sec'].append(tokens_per_sec)

        # Log to TensorBoard
        if self.writer is not None:
            if train_loss is not None:
                self.writer.add_scalar('Loss/train', train_loss, step)
                self.writer.add_scalar('Perplexity/train', train_ppl, step)
            if val_loss is not None:
                self.writer.add_scalar('Loss/val', val_loss, step)
                self.writer.add_scalar('Perplexity/val', val_ppl, step)
            if lr is not None:
                self.writer.add_scalar('Learning_Rate', lr, step)
            if tokens_per_sec is not None:
                self.writer.add_scalar('Throughput/tokens_per_sec', tokens_per_sec, step)

        # Log to CSV
        if self.csv_backup:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, epoch,
                    train_loss if train_loss is not None else '',
                    val_loss if val_loss is not None else '',
                    train_ppl if train_ppl is not None else '',
                    val_ppl if val_ppl is not None else '',
                    lr if lr is not None else '',
                    tokens_per_sec if tokens_per_sec is not None else '',
                    elapsed_time if elapsed_time is not None else ''
                ])

    def log_generation_samples(
        self,
        step: int,
        samples: List[Dict[str, str]]
    ):
        """
        Log generation samples to TensorBoard

        Args:
            step: Current training step
            samples: List of dicts with 'prompt' and 'generated' keys
        """
        if self.writer is not None:
            for i, sample in enumerate(samples):
                text = f"Prompt: {sample['prompt']}\n\nGenerated: {sample['generated']}"
                self.writer.add_text(f'Generation/sample_{i+1}', text, step)

    def close(self):
        """Close TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()

    def get_best_val_loss(self) -> float:
        """Get best validation loss seen so far"""
        if len(self.history['val_loss']) == 0:
            return float('inf')
        return min(self.history['val_loss'])

    def get_best_val_perplexity(self) -> float:
        """Get best validation perplexity seen so far"""
        if len(self.history['val_perplexity']) == 0:
            return float('inf')
        return min(self.history['val_perplexity'])

    def print_summary(self):
        """Print training summary"""
        if len(self.history['train_loss']) == 0:
            print("No metrics logged yet.")
            return

        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)

        print(f"\nTotal steps: {len(self.history['step'])}")
        print(f"Total epochs: {max(self.history['epoch']) if self.history['epoch'] else 0}")

        if self.history['train_loss']:
            print(f"\nTraining Loss:")
            print(f"  Initial: {self.history['train_loss'][0]:.4f}")
            print(f"  Final:   {self.history['train_loss'][-1]:.4f}")
            print(f"  Best:    {min(self.history['train_loss']):.4f}")

        if self.history['val_loss']:
            print(f"\nValidation Loss:")
            print(f"  Initial: {self.history['val_loss'][0]:.4f}")
            print(f"  Final:   {self.history['val_loss'][-1]:.4f}")
            print(f"  Best:    {min(self.history['val_loss']):.4f}")

        if self.history['train_perplexity']:
            print(f"\nTraining Perplexity:")
            print(f"  Initial: {self.history['train_perplexity'][0]:.2f}")
            print(f"  Final:   {self.history['train_perplexity'][-1]:.2f}")
            print(f"  Best:    {min(self.history['train_perplexity']):.2f}")

        if self.history['val_perplexity']:
            print(f"\nValidation Perplexity:")
            print(f"  Initial: {self.history['val_perplexity'][0]:.2f}")
            print(f"  Final:   {self.history['val_perplexity'][-1]:.2f}")
            print(f"  Best:    {min(self.history['val_perplexity']):.2f}")

        if self.history['tokens_per_sec']:
            avg_throughput = sum(self.history['tokens_per_sec']) / len(self.history['tokens_per_sec'])
            print(f"\nAverage throughput: {avg_throughput:.1f} tokens/sec")


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss

    Args:
        loss: Cross-entropy loss

    Returns:
        perplexity: exp(loss), capped at reasonable value
    """
    # Cap loss to prevent overflow in exp
    loss = min(loss, 20.0)
    return math.exp(loss)


def generate_samples(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    device: str = 'cpu'
) -> List[Dict[str, str]]:
    """
    Generate text samples for quality assessment

    Args:
        model: Language model
        tokenizer: Tokenizer with encode/decode methods
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        List of dicts with 'prompt' and 'generated' keys
    """
    model.eval()
    samples = []

    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Generate
            try:
                output = model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )

                # Decode
                generated_ids = output[0].cpu().tolist()
                generated_text = tokenizer.decode(generated_ids)

                samples.append({
                    'prompt': prompt,
                    'generated': generated_text
                })
            except Exception as e:
                print(f"[WARN] Generation failed for prompt '{prompt[:20]}...': {e}")
                samples.append({
                    'prompt': prompt,
                    'generated': f"[ERROR: {str(e)}]"
                })

    model.train()
    return samples


class EarlyStopping:
    """
    Early stopping to prevent overfitting

    Monitors validation loss and stops training if it doesn't improve
    for `patience` epochs.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop

        Args:
            value: Current metric value (e.g., validation loss)

        Returns:
            True if training should stop, False otherwise
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False


if __name__ == "__main__":
    print("Testing metrics tracker...")

    # Create tracker
    tracker = MetricsTracker("test_model", use_tensorboard=True)

    # Simulate training
    for step in range(10):
        tracker.log_metrics(
            step=step,
            epoch=step // 5,
            train_loss=5.0 - step * 0.5,
            val_loss=5.2 - step * 0.4,
            lr=0.001 * (1 - step/10),
            tokens_per_sec=1000.0,
            elapsed_time=step * 10.0
        )

    # Test generation samples
    samples = [
        {'prompt': 'Hello', 'generated': 'Hello world, how are you?'},
        {'prompt': 'Once', 'generated': 'Once upon a time...'}
    ]
    tracker.log_generation_samples(step=5, samples=samples)

    # Print summary
    tracker.print_summary()

    # Test early stopping
    print("\nTesting early stopping...")
    early_stop = EarlyStopping(patience=3)

    val_losses = [5.0, 4.5, 4.3, 4.2, 4.25, 4.3, 4.3, 4.4]
    for epoch, loss in enumerate(val_losses):
        should_stop = early_stop(loss)
        print(f"Epoch {epoch}: loss={loss:.2f}, counter={early_stop.counter}, stop={should_stop}")
        if should_stop:
            print("Early stopping triggered!")
            break

    # Close tracker
    tracker.close()

    print("\n[OK] Metrics tracker working!")
