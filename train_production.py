"""
Production Training Script for Itera-Lite on WikiText-103
Phase 4: Quality Training Pipeline

Features:
- WikiText-103 dataset loading
- Comprehensive metrics (loss, perplexity, generation samples)
- TensorBoard logging
- Checkpoint saving and resuming
- Early stopping
- Progress bars
- Graceful keyboard interrupt handling
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import time
import signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.itera_lite import IteraLiteModel
from models.config import get_tiny_config
from utils.wikitext_loader import load_wikitext103
from utils.metrics import MetricsTracker, EarlyStopping, generate_samples


class ProductionTrainer:
    """Production-quality trainer with all features"""

    def __init__(self, config_path: str, resume_from: str = None):
        """
        Initialize trainer

        Args:
            config_path: Path to YAML config file
            resume_from: Path to checkpoint to resume from (optional)
        """
        # Load config
        print("\n" + "="*70)
        print("LOADING CONFIGURATION")
        print("="*70)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"Config loaded from: {config_path}")
        self._print_config()

        # Set device
        self.device = self.config['hardware']['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("[WARN] CUDA not available, falling back to CPU")
            self.device = 'cpu'

        print(f"\nUsing device: {self.device}")

        # Set random seed
        if 'seed' in self.config:
            torch.manual_seed(self.config['seed'])
            print(f"Random seed: {self.config['seed']}")

        # Load dataset
        print("\n" + "="*70)
        print("LOADING DATASET")
        print("="*70)

        self.train_loader, self.val_loader, self.test_loader, self.tokenizer = load_wikitext103(
            data_dir=self.config['dataset']['path'],
            seq_length=self.config['dataset']['seq_length'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['hardware']['num_workers']
        )

        # Create model
        print("\n" + "="*70)
        print("CREATING MODEL")
        print("="*70)

        model_config = get_tiny_config()
        model_config.vocab_size = len(self.tokenizer.token2id)
        model_config.max_seq_length = self.config['dataset']['seq_length']

        self.model = IteraLiteModel(model_config)
        self.model.to(self.device)

        print(f"\nModel parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Create scheduler
        num_training_steps = len(self.train_loader) * self.config['training']['max_epochs']
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=self.config['training']['min_lr']
        )

        # Warmup scheduler
        self.warmup_steps = self.config['training']['warmup_steps']
        self.base_lr = self.config['training']['learning_rate']

        # Metrics tracker
        model_name = f"itera_lite_wikitext103"
        self.metrics = MetricsTracker(
            model_name=model_name,
            log_dir=self.config['logging']['tensorboard_dir'],
            use_tensorboard=self.config['logging']['use_tensorboard'],
            csv_backup=self.config['logging']['csv_backup']
        )

        # Early stopping
        self.early_stopping = None
        if self.config['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=self.config['early_stopping']['patience'],
                min_delta=self.config['early_stopping']['min_delta'],
                mode='min'
            )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_perplexity = float('inf')

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config['checkpoints']['dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        # Signal handler for graceful interruption
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

        print(f"\n[OK] Trainer initialized!")

    def _print_config(self):
        """Print key configuration settings"""
        print("\nKey settings:")
        print(f"  Dataset: {self.config['dataset']['name']}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        print(f"  Max epochs: {self.config['training']['max_epochs']}")
        print(f"  Warmup steps: {self.config['training']['warmup_steps']}")
        print(f"  Early stopping: {self.config['early_stopping']['enabled']}")
        if self.config['early_stopping']['enabled']:
            print(f"    Patience: {self.config['early_stopping']['patience']} epochs")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n[INTERRUPT] Ctrl+C detected! Saving checkpoint and exiting...")
        self.interrupted = True

    def _get_lr_for_step(self, step):
        """Get learning rate with warmup"""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step / self.warmup_steps)
        else:
            # Use scheduler
            return self.scheduler.get_last_lr()[0]

    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.config['training']['max_epochs']}",
            leave=True
        )

        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            if self.interrupted:
                break

            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits, loss, aux_loss = self.model(input_ids, target_ids)

            # Backward pass
            loss = loss / self.config['training']['gradient_accumulation_steps']
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update learning rate with warmup
                if self.global_step < self.warmup_steps:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self._get_lr_for_step(self.global_step)
                else:
                    self.scheduler.step()

                self.global_step += 1

            # Track metrics
            epoch_loss += loss.item() * self.config['training']['gradient_accumulation_steps']
            num_batches += 1

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item() * self.config['training']['gradient_accumulation_steps']:.4f}",
                'lr': f"{current_lr:.2e}"
            })

            # Print every N steps
            if self.global_step % self.config['logging']['print_every'] == 0:
                avg_loss = epoch_loss / num_batches
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                print(f"\n  Step {self.global_step}: loss={avg_loss:.4f}, ppl={perplexity:.2f}, lr={current_lr:.2e}")

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return avg_epoch_loss

    def _validate(self):
        """Run validation"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_loader, desc="Validation", leave=False):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits, loss, aux_loss = self.model(input_ids, target_ids)
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        return avg_val_loss

    def _generate_samples(self):
        """Generate text samples for quality assessment"""
        if not self.config['generation']['enabled']:
            return []

        prompts = self.config['generation']['prompts']
        samples = generate_samples(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=self.config['generation']['max_new_tokens'],
            temperature=self.config['generation']['temperature'],
            device=self.device
        )

        return samples

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.config['checkpoints']['save_optimizer'] else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.config['checkpoints']['save_scheduler'] else None,
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
            'config': self.config
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"\n[SAVE] Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"[SAVE] Best model updated: {best_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training"""
        print(f"\n[LOAD] Resuming from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint.get('optimizer_state_dict') and self.config['checkpoints']['save_optimizer']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict') and self.config['checkpoints']['save_scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_perplexity = checkpoint.get('best_val_perplexity', float('inf'))

        print(f"[OK] Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def train(self):
        """Main training loop"""
        print("\n" + "#"*70)
        print("#" + "STARTING PRODUCTION TRAINING".center(68) + "#")
        print("#"*70 + "\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['training']['max_epochs']):
            if self.interrupted:
                self._save_checkpoint(f'interrupted_epoch_{epoch}.pt')
                print("\n[EXIT] Training interrupted. Checkpoint saved.")
                break

            self.current_epoch = epoch

            # Train epoch
            train_loss = self._train_epoch()
            train_perplexity = torch.exp(torch.tensor(train_loss)).item()

            # Validate
            should_validate = (epoch + 1) % self.config['evaluation']['eval_every_epochs'] == 0
            val_loss = None
            val_perplexity = None

            if should_validate:
                val_loss = self._validate()
                val_perplexity = torch.exp(torch.tensor(val_loss)).item()

                print(f"\n{'='*70}")
                print(f"EPOCH {epoch+1} SUMMARY")
                print(f"{'='*70}")
                print(f"  Train Loss: {train_loss:.4f}  |  Train PPL: {train_perplexity:.2f}")
                print(f"  Val Loss:   {val_loss:.4f}  |  Val PPL:   {val_perplexity:.2f}")
                print(f"{'='*70}\n")

                # Log metrics
                current_lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                self.metrics.log_metrics(
                    step=self.global_step,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=current_lr,
                    elapsed_time=elapsed
                )

                # Check if best model
                if val_perplexity < self.best_val_perplexity:
                    self.best_val_perplexity = val_perplexity
                    self.best_val_loss = val_loss
                    self._save_checkpoint(f'best_epoch_{epoch+1}.pt', is_best=True)
                    print(f"[BEST] New best validation perplexity: {val_perplexity:.2f}")

                # Early stopping check
                if self.early_stopping and self.early_stopping(val_perplexity):
                    print(f"\n[STOP] Early stopping triggered after {epoch+1} epochs")
                    print(f"  No improvement for {self.early_stopping.patience} epochs")
                    print(f"  Best val perplexity: {self.best_val_perplexity:.2f}")
                    break

            # Generate samples
            should_generate = (epoch + 1) % self.config['generation']['every_epochs'] == 0
            if should_generate and self.config['generation']['enabled']:
                print(f"\n{'='*70}")
                print(f"GENERATION SAMPLES (Epoch {epoch+1})")
                print(f"{'='*70}")

                samples = self._generate_samples()
                for i, sample in enumerate(samples):
                    print(f"\nPrompt {i+1}: \"{sample['prompt']}\"")
                    print(f"Generated: {sample['generated'][:200]}...")

                # Log to TensorBoard
                self.metrics.log_generation_samples(self.global_step, samples)
                print(f"{'='*70}\n")

            # Save checkpoint
            should_save = (epoch + 1) % self.config['evaluation']['save_every_epochs'] == 0
            if should_save:
                self._save_checkpoint(f'epoch_{epoch+1}.pt')

        # Training complete
        print("\n" + "#"*70)
        print("#" + "TRAINING COMPLETE!".center(68) + "#")
        print("#"*70 + "\n")

        # Print final summary
        self.metrics.print_summary()

        # Save final checkpoint
        self._save_checkpoint('final_model.pt')

        # Close metrics
        self.metrics.close()

        elapsed_hours = (time.time() - start_time) / 3600
        print(f"\nTotal training time: {elapsed_hours:.2f} hours")
        print(f"Best validation perplexity: {self.best_val_perplexity:.2f}")
        print(f"\nCheckpoints saved to: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Production training for Itera-Lite')
    parser.add_argument('--config', type=str, default='configs/training_config_wikitext103.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Create trainer and train
    trainer = ProductionTrainer(config_path=args.config, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
