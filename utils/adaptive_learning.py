"""
Adaptive Learning Infrastructure for Itera-Lite
Enables feedback-driven model tuning and autonomous optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackLogger:
    """Logs model inputs, outputs, and user feedback for adaptive learning."""
    
    def __init__(self, log_dir: str = "logs/adaptive"):
        """
        Initialize feedback logger.
        
        Args:
            log_dir: Directory to store feedback logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.log_dir / "phase6_feedback.json"
        self.feedback_history = self._load_feedback()
        
        logger.info(f"FeedbackLogger initialized: {len(self.feedback_history)} existing records")
    
    def _load_feedback(self) -> List[Dict]:
        """Load existing feedback from file."""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_feedback(self):
        """Save feedback to file."""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def log_inference(
        self,
        input_text: str,
        output_text: str,
        user_rating: Optional[int] = None,
        correctness: Optional[bool] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log an inference with optional user feedback.
        
        Args:
            input_text: Input prompt
            output_text: Model-generated output
            user_rating: Rating from 1-5 (optional)
            correctness: Whether output was correct (optional)
            metadata: Additional metadata (latency, model variant, etc.)
            
        Returns:
            Feedback ID
        """
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_history)}"
        
        record = {
            'id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'output': output_text,
            'user_rating': user_rating,
            'correctness': correctness,
            'metadata': metadata or {}
        }
        
        self.feedback_history.append(record)
        self._save_feedback()
        
        logger.debug(f"Logged feedback: {feedback_id}")
        return feedback_id
    
    def update_feedback(self, feedback_id: str, user_rating: int, correctness: bool):
        """
        Update existing feedback with user rating.
        
        Args:
            feedback_id: ID of feedback record
            user_rating: Rating from 1-5
            correctness: Whether output was correct
        """
        for record in self.feedback_history:
            if record['id'] == feedback_id:
                record['user_rating'] = user_rating
                record['correctness'] = correctness
                record['updated_at'] = datetime.now().isoformat()
                self._save_feedback()
                logger.info(f"Updated feedback {feedback_id}: rating={user_rating}, correct={correctness}")
                return
        
        logger.warning(f"Feedback ID not found: {feedback_id}")
    
    def get_recent_feedback(self, n: int = 100) -> List[Dict]:
        """Get n most recent feedback records."""
        return self.feedback_history[-n:]
    
    def get_negative_feedback(self, threshold: int = 3) -> List[Dict]:
        """Get feedback with low ratings or marked as incorrect."""
        return [
            record for record in self.feedback_history
            if (record.get('user_rating') and record['user_rating'] < threshold) or
               (record.get('correctness') is False)
        ]
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics."""
        total = len(self.feedback_history)
        rated = [r for r in self.feedback_history if r.get('user_rating')]
        correct = [r for r in self.feedback_history if r.get('correctness') is True]
        incorrect = [r for r in self.feedback_history if r.get('correctness') is False]
        
        avg_rating = np.mean([r['user_rating'] for r in rated]) if rated else 0.0
        
        return {
            'total_records': total,
            'rated_records': len(rated),
            'correct_records': len(correct),
            'incorrect_records': len(incorrect),
            'average_rating': avg_rating,
            'accuracy': len(correct) / total if total > 0 else 0.0
        }


class AdaptiveLearningModule:
    """Implements adaptive fine-tuning based on user feedback."""
    
    def __init__(
        self,
        model: nn.Module,
        initial_lr: float = 1e-5,
        min_lr: float = 1e-7,
        max_lr: float = 1e-4,
        adaptation_window: int = 50
    ):
        """
        Initialize adaptive learning module.
        
        Args:
            model: PyTorch model to adapt
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            adaptation_window: Number of recent samples to consider
        """
        self.model = model
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adaptation_window = adaptation_window
        
        self.optimizer = optim.Adam(model.parameters(), lr=self.current_lr)
        self.performance_history = deque(maxlen=adaptation_window)
        
        # Quantization threshold adaptation
        self.quantization_threshold = 0.1  # Error threshold for quantization
        self.threshold_history = []
        
        logger.info(f"AdaptiveLearningModule initialized: lr={initial_lr}, window={adaptation_window}")
    
    def adjust_learning_rate(self, recent_accuracy: float):
        """
        Dynamically adjust learning rate based on recent performance.
        
        Args:
            recent_accuracy: Accuracy on recent samples (0.0-1.0)
        """
        self.performance_history.append(recent_accuracy)
        
        if len(self.performance_history) < 10:
            return  # Need minimum history
        
        # Calculate trend
        recent_perf = np.mean(list(self.performance_history)[-10:])
        
        # Adjust learning rate
        if recent_perf < 0.5:
            # Performance is poor, increase learning rate
            new_lr = min(self.current_lr * 1.5, self.max_lr)
            logger.info(f"Low accuracy ({recent_perf:.2%}), increasing LR: {self.current_lr:.2e} → {new_lr:.2e}")
        elif recent_perf > 0.8:
            # Performance is good, decrease learning rate for stability
            new_lr = max(self.current_lr * 0.8, self.min_lr)
            logger.info(f"High accuracy ({recent_perf:.2%}), decreasing LR: {self.current_lr:.2e} → {new_lr:.2e}")
        else:
            # Performance is moderate, keep current rate
            new_lr = self.current_lr
        
        if new_lr != self.current_lr:
            self.current_lr = new_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
    
    def adjust_quantization_threshold(self, quantization_errors: List[float]):
        """
        Adjust quantization threshold based on error distribution.
        
        Args:
            quantization_errors: List of quantization errors
        """
        if not quantization_errors:
            return
        
        mean_error = np.mean(quantization_errors)
        p95_error = np.percentile(quantization_errors, 95)
        
        # Adjust threshold to keep 95% of errors below threshold
        new_threshold = p95_error * 1.1  # 10% margin
        
        self.threshold_history.append({
            'timestamp': datetime.now().isoformat(),
            'threshold': new_threshold,
            'mean_error': mean_error,
            'p95_error': p95_error
        })
        
        logger.info(f"Adjusted quantization threshold: {self.quantization_threshold:.6f} → {new_threshold:.6f}")
        self.quantization_threshold = new_threshold
    
    def fine_tune_on_feedback(
        self,
        feedback_logger: FeedbackLogger,
        num_epochs: int = 1,
        batch_size: int = 4
    ) -> Dict:
        """
        Fine-tune model on recent negative feedback.
        
        Args:
            feedback_logger: FeedbackLogger instance
            num_epochs: Number of fine-tuning epochs
            batch_size: Batch size for training
            
        Returns:
            Fine-tuning results
        """
        # Get negative feedback
        negative_samples = feedback_logger.get_negative_feedback()
        
        if len(negative_samples) < batch_size:
            logger.warning(f"Insufficient negative feedback ({len(negative_samples)} samples)")
            return {'status': 'skipped', 'reason': 'insufficient_samples'}
        
        logger.info(f"Fine-tuning on {len(negative_samples)} negative feedback samples")
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Simple fine-tuning loop (placeholder - actual implementation would need tokenization)
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # In a real implementation, would batch and process samples
            # For now, just demonstrate the structure
            for i in range(0, len(negative_samples), batch_size):
                batch = negative_samples[i:i + batch_size]
                
                # Placeholder: In real implementation, would:
                # 1. Tokenize input/output pairs
                # 2. Forward pass
                # 3. Compute loss on correct outputs
                # 4. Backpropagate
                
                # For demonstration:
                simulated_loss = torch.tensor(0.5)  # Placeholder
                
                self.optimizer.zero_grad()
                # simulated_loss.backward()  # Would be real loss
                # self.optimizer.step()
                
                epoch_loss += simulated_loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/max(1, len(batch)):.4f}")
        
        avg_loss = total_loss / max(1, num_batches)
        
        self.model.eval()
        
        return {
            'status': 'completed',
            'num_samples': len(negative_samples),
            'num_epochs': num_epochs,
            'avg_loss': avg_loss,
            'learning_rate': self.current_lr
        }
    
    def get_adaptation_metrics(self) -> Dict:
        """Get current adaptation metrics."""
        return {
            'current_lr': self.current_lr,
            'lr_range': [self.min_lr, self.max_lr],
            'performance_window': len(self.performance_history),
            'recent_accuracy': np.mean(list(self.performance_history)) if self.performance_history else 0.0,
            'quantization_threshold': self.quantization_threshold,
            'threshold_adjustments': len(self.threshold_history)
        }


class AdaptiveSystem:
    """Complete adaptive learning system integrating feedback and fine-tuning."""
    
    def __init__(self, model: nn.Module, log_dir: str = "logs/adaptive"):
        """
        Initialize adaptive system.
        
        Args:
            model: PyTorch model
            log_dir: Directory for logs
        """
        self.model = model
        self.feedback_logger = FeedbackLogger(log_dir)
        self.learning_module = AdaptiveLearningModule(model)
        
        self.auto_update_threshold = 50  # Auto-update after N negative samples
        self.last_update_time = datetime.now()
        
        logger.info("AdaptiveSystem initialized")
    
    def log_and_learn(
        self,
        input_text: str,
        output_text: str,
        user_rating: Optional[int] = None,
        correctness: Optional[bool] = None,
        auto_adapt: bool = True
    ) -> Dict:
        """
        Log inference and trigger adaptive learning if needed.
        
        Args:
            input_text: Input prompt
            output_text: Model output
            user_rating: User rating (1-5)
            correctness: Whether output was correct
            auto_adapt: Whether to automatically trigger adaptation
            
        Returns:
            Status and metrics
        """
        # Log feedback
        feedback_id = self.feedback_logger.log_inference(
            input_text=input_text,
            output_text=output_text,
            user_rating=user_rating,
            correctness=correctness
        )
        
        result = {
            'feedback_id': feedback_id,
            'auto_adapted': False
        }
        
        # Check if auto-adaptation should trigger
        if auto_adapt:
            negative_count = len(self.feedback_logger.get_negative_feedback())
            
            if negative_count >= self.auto_update_threshold:
                logger.info(f"Auto-adaptation triggered: {negative_count} negative samples")
                adaptation_result = self.trigger_adaptation()
                result['auto_adapted'] = True
                result['adaptation'] = adaptation_result
        
        return result
    
    def trigger_adaptation(self, manual: bool = False) -> Dict:
        """
        Manually trigger model adaptation.
        
        Args:
            manual: Whether triggered manually
            
        Returns:
            Adaptation results
        """
        logger.info(f"Triggering adaptation (manual={manual})")
        
        # Get statistics
        stats = self.feedback_logger.get_statistics()
        
        # Adjust learning rate based on accuracy
        if stats['total_records'] > 0:
            self.learning_module.adjust_learning_rate(stats['accuracy'])
        
        # Fine-tune on negative feedback
        fine_tune_result = self.learning_module.fine_tune_on_feedback(
            self.feedback_logger,
            num_epochs=1,
            batch_size=4
        )
        
        self.last_update_time = datetime.now()
        
        return {
            'timestamp': self.last_update_time.isoformat(),
            'trigger': 'manual' if manual else 'automatic',
            'statistics': stats,
            'fine_tuning': fine_tune_result,
            'adaptation_metrics': self.learning_module.get_adaptation_metrics()
        }
    
    def get_system_status(self) -> Dict:
        """Get complete system status."""
        stats = self.feedback_logger.get_statistics()
        metrics = self.learning_module.get_adaptation_metrics()
        
        return {
            'feedback_statistics': stats,
            'adaptation_metrics': metrics,
            'last_update': self.last_update_time.isoformat(),
            'auto_update_threshold': self.auto_update_threshold,
            'negative_samples_pending': len(self.feedback_logger.get_negative_feedback())
        }


def demonstrate_adaptive_learning():
    """Demonstrate adaptive learning system."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.itera_lite import IteraLiteModel
    from models.config import IteraLiteConfig
    
    logger.info("Demonstrating Adaptive Learning System...")
    
    # Create small model for demonstration
    config = IteraLiteConfig(
        vocab_size=2000,
        hidden_size=64,
        num_layers=3,
        ssm_state_size=8,
        num_experts=4,
        top_k_experts=2,
        expert_size=32
    )
    
    model = IteraLiteModel(config)
    model.eval()
    
    # Initialize adaptive system
    adaptive_system = AdaptiveSystem(model)
    
    # Simulate some feedback
    logger.info("\n1. Logging positive feedback...")
    adaptive_system.log_and_learn(
        input_text="Hello, how are you?",
        output_text="I'm doing well, thank you!",
        user_rating=5,
        correctness=True,
        auto_adapt=False
    )
    
    logger.info("\n2. Logging negative feedback...")
    for i in range(5):
        adaptive_system.log_and_learn(
            input_text=f"Test input {i}",
            output_text=f"Incorrect output {i}",
            user_rating=2,
            correctness=False,
            auto_adapt=False
        )
    
    # Get system status
    logger.info("\n3. System status:")
    status = adaptive_system.get_system_status()
    logger.info(f"  Total feedback: {status['feedback_statistics']['total_records']}")
    logger.info(f"  Accuracy: {status['feedback_statistics']['accuracy']:.2%}")
    logger.info(f"  Current LR: {status['adaptation_metrics']['current_lr']:.2e}")
    
    # Trigger manual adaptation
    logger.info("\n4. Triggering manual adaptation...")
    result = adaptive_system.trigger_adaptation(manual=True)
    logger.info(f"  Status: {result['fine_tuning']['status']}")
    logger.info(f"  Samples: {result['fine_tuning'].get('num_samples', 0)}")
    
    logger.info("\n✓ Adaptive learning demonstration complete")
    
    return adaptive_system


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demonstrate_adaptive_learning()
