"""
Training utilities for Itera-Lite models
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
from pathlib import Path
from typing import Optional, Dict, List
import json
import csv


class Trainer:
    """Unified trainer for language models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'results',
        model_name: str = 'model',
        max_grad_norm: float = 1.0,
        eval_every: int = 100,
        save_every: int = 500,
        early_stopping_patience: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.max_grad_norm = max_grad_norm
        self.eval_every = eval_every
        self.save_every = save_every
        self.early_stopping_patience = early_stopping_patience
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Timing
        self.train_start_time = None
        self.epoch_start_time = None
        
        # Initialize CSV log
        self.csv_path = self.log_dir / f'{model_name}_metrics.csv'
        self._init_csv_log()
    
    def _init_csv_log(self):
        """Initialize CSV log file"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'train_loss', 'val_loss', 'lr',
                'tokens_per_sec', 'elapsed_time'
            ])
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics to CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('step', ''),
                metrics.get('epoch', ''),
                metrics.get('train_loss', ''),
                metrics.get('val_loss', ''),
                metrics.get('lr', ''),
                metrics.get('tokens_per_sec', ''),
                metrics.get('elapsed_time', '')
            ])
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        tokens_processed = 0
        
        self.epoch_start_time = time.time()
        
        for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Handle different model outputs
            outputs = self.model(input_ids, labels=target_ids)
            if len(outputs) == 3:  # Itera-Lite (logits, loss, aux_loss)
                _, loss, _ = outputs
            elif len(outputs) == 2:  # Transformer (logits, loss)
                _, loss = outputs
            else:
                raise ValueError(f"Unexpected model output: {len(outputs)} elements")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # Update weights
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            tokens_processed += input_ids.numel()
            self.global_step += 1
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Periodic evaluation
            if self.global_step % self.eval_every == 0:
                val_loss = self.evaluate()
                
                # Calculate throughput
                elapsed = time.time() - self.epoch_start_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                
                print(f"Step {self.global_step:5d} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Tokens/s: {tokens_per_sec:.0f}")
                
                # Log to CSV
                self._log_metrics({
                    'step': self.global_step,
                    'epoch': self.epoch,
                    'train_loss': loss.item(),
                    'val_loss': val_loss,
                    'lr': current_lr,
                    'tokens_per_sec': tokens_per_sec,
                    'elapsed_time': elapsed
                })
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best')
                else:
                    self.patience_counter += 1
                
                self.model.train()
            
            # Periodic checkpoint
            if self.global_step % self.save_every == 0:
                self.save_checkpoint(f'step_{self.global_step}')
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, labels=target_ids)
            if len(outputs) == 3:  # Itera-Lite
                _, loss, _ = outputs
            elif len(outputs) == 2:  # Transformer
                _, loss = outputs
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs: int):
        """Train for multiple epochs"""
        print(f"\n{'=' * 70}")
        print(f"Starting training: {self.model_name}")
        print(f"{'=' * 70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'=' * 70}\n")
        
        self.train_start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Evaluate
            val_loss = self.evaluate()
            
            # Log
            epoch_time = time.time() - self.epoch_start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            self.train_losses.append(train_loss)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - self.train_start_time
        print(f"\n{'=' * 70}")
        print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'=' * 70}\n")
        
        # Save final model
        self.save_checkpoint('final')
        
        # Save training summary
        self.save_summary()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'{self.model_name}_{name}.pt'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Loaded checkpoint from: {checkpoint_path}")
    
    def save_summary(self):
        """Save training summary"""
        summary = {
            'model_name': self.model_name,
            'total_epochs': self.epoch + 1,
            'total_steps': self.global_step,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'total_training_time': time.time() - self.train_start_time,
        }
        
        summary_path = self.log_dir / f'{self.model_name}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved training summary: {summary_path}")


def create_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.01):
    """Create AdamW optimizer with weight decay"""
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None
):
    """Create cosine annealing scheduler"""
    if num_warmup_steps is None:
        num_warmup_steps = num_training_steps // 10
    
    return CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=1e-6
    )
