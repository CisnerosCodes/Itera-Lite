"""
Knowledge distillation utilities for training compact student models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import time
from pathlib import Path


class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation"""
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        reduction: str = 'batchmean'
    ):
        """
        Args:
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss (1-alpha for student loss)
            reduction: Reduction method for KL divergence
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate distillation loss
        
        Args:
            student_logits: Student model output logits
            teacher_logits: Teacher model output logits
            labels: Ground truth labels
            
        Returns:
            Total loss and loss components dictionary
        """
        # Distillation loss (soft targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = self.kl_div(student_soft, teacher_soft)
        distillation_loss = distillation_loss * (self.temperature ** 2)
        
        # Student loss (hard targets)
        student_loss = self.ce_loss(
            student_logits.reshape(-1, student_logits.size(-1)),
            labels.reshape(-1)
        )
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'distillation': distillation_loss.item(),
            'student': student_loss.item()
        }


class DistillationTrainer:
    """Trainer for knowledge distillation"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = 'cpu'
    ):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
        
        self.teacher.eval()  # Teacher is always in eval mode
        
        self.criterion = DistillationLoss(temperature, alpha)
        
        # Calculate compression ratio
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        self.compression_ratio = teacher_params / student_params
        
        print(f"\nDistillation Setup:")
        print(f"  Teacher params: {teacher_params:,}")
        print(f"  Student params: {student_params:,}")
        print(f"  Compression ratio: {self.compression_ratio:.2f}x")
        print(f"  Temperature: {temperature}")
        print(f"  Alpha (distillation weight): {alpha}")
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        max_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train student for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer for student model
            max_steps: Maximum number of steps (optional)
            
        Returns:
            Dictionary with average losses
        """
        self.student.train()
        
        total_loss = 0
        total_distill = 0
        total_student = 0
        num_batches = 0
        
        for i, (input_ids, labels) in enumerate(train_loader):
            if max_steps and i >= max_steps:
                break
            
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_output = self.teacher(input_ids)
                # Extract logits if model returns tuple
                if isinstance(teacher_output, tuple):
                    teacher_logits = teacher_output[0]
                else:
                    teacher_logits = teacher_output
            
            # Get student predictions
            student_output = self.student(input_ids)
            # Extract logits if model returns tuple
            if isinstance(student_output, tuple):
                student_logits = student_output[0]
            else:
                student_logits = student_output
            
            # Calculate loss
            loss, loss_dict = self.criterion(student_logits, teacher_logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            
            # Track losses
            total_loss += loss_dict['total']
            total_distill += loss_dict['distillation']
            total_student += loss_dict['student']
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'distillation_loss': total_distill / num_batches,
            'student_loss': total_student / num_batches
        }
    
    def evaluate(self, val_loader) -> Dict[str, float]:
        """
        Evaluate student model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.student.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for (input_ids, labels) in val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                teacher_output = self.teacher(input_ids)
                student_output = self.student(input_ids)
                
                # Extract logits if model returns tuple
                if isinstance(teacher_output, tuple):
                    teacher_logits = teacher_output[0]
                else:
                    teacher_logits = teacher_output
                    
                if isinstance(student_output, tuple):
                    student_logits = student_output[0]
                else:
                    student_logits = student_output
                
                # Calculate loss
                loss, _ = self.criterion(student_logits, teacher_logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches
        }
    
    def compare_models(
        self,
        test_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Compare teacher and student performance
        
        Args:
            test_input: Test input tensor
            num_runs: Number of benchmark runs
            
        Returns:
            Comparison metrics
        """
        self.teacher.eval()
        self.student.eval()
        
        test_input = test_input.to(self.device)
        
        # Benchmark teacher
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.teacher(test_input)
        teacher_time = (time.time() - start) / num_runs
        
        # Benchmark student
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.student(test_input)
        student_time = (time.time() - start) / num_runs
        
        # Size comparison
        teacher_size = sum(p.numel() * p.element_size() for p in self.teacher.parameters()) / 1024 / 1024
        student_size = sum(p.numel() * p.element_size() for p in self.student.parameters()) / 1024 / 1024
        
        results = {
            'teacher_time_ms': teacher_time * 1000,
            'student_time_ms': student_time * 1000,
            'speedup': teacher_time / student_time,
            'teacher_size_mb': teacher_size,
            'student_size_mb': student_size,
            'compression_ratio': self.compression_ratio,
            'size_reduction': teacher_size / student_size
        }
        
        print("\nTeacher vs Student Comparison:")
        print(f"  Teacher: {results['teacher_time_ms']:.2f} ms, {results['teacher_size_mb']:.2f} MB")
        print(f"  Student: {results['student_time_ms']:.2f} ms, {results['student_size_mb']:.2f} MB")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Size reduction: {results['size_reduction']:.2f}x")
        
        return results
    
    def save_student(self, filepath: str):
        """Save student model"""
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'compression_ratio': self.compression_ratio,
        }, filepath)
        print(f"Saved student model to {filepath}")


def distill_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    temperature: float = 2.0,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict[str, any]]:
    """
    Perform knowledge distillation
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        temperature: Distillation temperature
        alpha: Distillation weight
        save_path: Path to save student model
        device: Device to use
        
    Returns:
        Trained student model and training history
    """
    trainer = DistillationTrainer(
        teacher_model,
        student_model,
        temperature,
        alpha,
        device
    )
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf')
    }
    
    print(f"\nStarting distillation training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        
        # Save best model
        if val_metrics['val_loss'] < history['best_val_loss']:
            history['best_val_loss'] = val_metrics['val_loss']
            if save_path:
                trainer.save_student(save_path)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Val Loss: {val_metrics['val_loss']:.4f}")
    
    # Final comparison
    test_input = torch.randint(0, 2000, (4, 128))
    comparison = trainer.compare_models(test_input)
    history['comparison'] = comparison
    
    return student_model, history


if __name__ == "__main__":
    print("Testing distillation utilities...")
    
    # Create simple teacher and student models
    class SimpleTeacher(nn.Module):
        def __init__(self, vocab_size=2000, d_model=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.fc1 = nn.Linear(d_model, d_model * 4)
            self.fc2 = nn.Linear(d_model * 4, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.mean(dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    class SimpleStudent(nn.Module):
        def __init__(self, vocab_size=2000, d_model=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.fc1 = nn.Linear(d_model, d_model * 2)
            self.fc2 = nn.Linear(d_model * 2, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.mean(dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    teacher = SimpleTeacher()
    student = SimpleStudent()
    
    # Test distillation loss
    criterion = DistillationLoss()
    student_logits = torch.randn(4, 2000)
    teacher_logits = torch.randn(4, 2000)
    labels = torch.randint(0, 2000, (4,))
    
    loss, loss_dict = criterion(student_logits, teacher_logits, labels)
    print(f"\nDistillation loss test:")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Distillation component: {loss_dict['distillation']:.4f}")
    print(f"  Student component: {loss_dict['student']:.4f}")
    
    print("\n✓ Distillation test completed!")
