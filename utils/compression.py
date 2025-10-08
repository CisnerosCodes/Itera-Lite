"""
Model compression utilities for Itera-Lite
Includes vocabulary reduction, quantization, and distillation helpers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import copy
from pathlib import Path


class VocabularyReducer:
    """Reduce vocabulary size while maintaining performance"""
    
    def __init__(self, original_vocab_size: int, target_vocab_size: int):
        self.original_vocab_size = original_vocab_size
        self.target_vocab_size = target_vocab_size
        self.token_mapping = {}
    
    def create_mapping(self, token_frequencies: Dict[int, int]) -> Dict[int, int]:
        """
        Create mapping from original to reduced vocabulary
        
        Args:
            token_frequencies: Dict mapping token_id -> frequency
        
        Returns:
            mapping: Dict mapping old_token_id -> new_token_id
        """
        # Sort tokens by frequency
        sorted_tokens = sorted(
            token_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Keep top target_vocab_size tokens
        mapping = {}
        special_tokens_count = 4  # <PAD>, <UNK>, <BOS>, <EOS>
        
        # Map special tokens directly
        for i in range(special_tokens_count):
            mapping[i] = i
        
        # Map frequent tokens
        new_id = special_tokens_count
        for old_id, freq in sorted_tokens:
            if old_id < special_tokens_count:
                continue  # Already mapped
            
            if new_id < self.target_vocab_size:
                mapping[old_id] = new_id
                new_id += 1
            else:
                # Map to <UNK>
                mapping[old_id] = 1
        
        self.token_mapping = mapping
        return mapping
    
    def compress_embedding(self, original_embedding: nn.Embedding) -> nn.Embedding:
        """Compress embedding layer using vocabulary reduction"""
        # Create new embedding
        new_embedding = nn.Embedding(
            self.target_vocab_size,
            original_embedding.embedding_dim
        )
        
        # Copy weights for kept tokens
        with torch.no_grad():
            for old_id, new_id in self.token_mapping.items():
                if new_id < self.target_vocab_size:
                    new_embedding.weight[new_id] = original_embedding.weight[old_id]
        
        return new_embedding
    
    def estimate_reduction(self) -> Dict[str, float]:
        """Estimate parameter reduction"""
        original_params = self.original_vocab_size
        new_params = self.target_vocab_size
        reduction_ratio = original_params / new_params
        
        return {
            'original_vocab': self.original_vocab_size,
            'new_vocab': self.target_vocab_size,
            'reduction_ratio': reduction_ratio,
            'params_saved': original_params - new_params
        }


class ModelQuantizer:
    """Quantize model weights to reduce memory and increase efficiency"""
    
    @staticmethod
    def quantize_8bit(model: nn.Module) -> nn.Module:
        """
        Quantize model to 8-bit integers
        Note: This is a placeholder for demonstration
        Real implementation would use torch.quantization
        """
        quantized_model = copy.deepcopy(model)
        
        # Placeholder: In reality, would use torch.quantization.quantize_dynamic
        print("⚠️  8-bit quantization: Placeholder implementation")
        print("    Real implementation requires torch.quantization")
        
        return quantized_model
    
    @staticmethod
    def quantize_4bit(model: nn.Module) -> nn.Module:
        """
        Quantize model to 4-bit integers
        Note: This is a placeholder for demonstration
        Real implementation would use bitsandbytes or custom quantization
        """
        quantized_model = copy.deepcopy(model)
        
        # Placeholder
        print("⚠️  4-bit quantization: Placeholder implementation")
        print("    Real implementation requires bitsandbytes or custom kernels")
        
        return quantized_model
    
    @staticmethod
    def estimate_quantization_savings(
        original_params: int,
        original_bits: int = 32,
        target_bits: int = 8
    ) -> Dict[str, float]:
        """Estimate memory savings from quantization"""
        reduction_ratio = original_bits / target_bits
        
        original_size_mb = (original_params * original_bits / 8) / 1024 / 1024
        quantized_size_mb = (original_params * target_bits / 8) / 1024 / 1024
        
        return {
            'original_bits': original_bits,
            'target_bits': target_bits,
            'reduction_ratio': reduction_ratio,
            'original_size_mb': original_size_mb,
            'quantized_size_mb': quantized_size_mb,
            'savings_mb': original_size_mb - quantized_size_mb
        }


class KnowledgeDistillation:
    """Knowledge distillation from teacher to student model"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        """
        Args:
            teacher_model: Larger teacher model
            student_model: Smaller student model
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss (1-alpha for hard labels)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher.eval()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: Ground truth labels
        
        Returns:
            Combined distillation loss
        """
        # Soft targets from teacher
        soft_targets = torch.nn.functional.softmax(
            teacher_logits / self.temperature, dim=-1
        )
        
        # Soft predictions from student
        soft_pred = torch.nn.functional.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        
        # Distillation loss (KL divergence)
        distill_loss = torch.nn.functional.kl_div(
            soft_pred,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard label loss
        hard_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # Combine losses
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Single training step with distillation
        
        Args:
            input_ids: Input token ids
            labels: Target token ids
        
        Returns:
            Distillation loss
        """
        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids)
            teacher_logits = teacher_outputs[0]  # First element is logits
        
        # Get student predictions
        student_outputs = self.student(input_ids)
        student_logits = student_outputs[0]
        
        # Compute distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        return loss


def analyze_compression_potential(model: nn.Module) -> Dict:
    """
    Analyze potential compression gains
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with compression analysis
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Analyze by layer type
    embedding_params = 0
    linear_params = 0
    other_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Linear):
            linear_params += sum(p.numel() for p in module.parameters())
        elif hasattr(module, 'parameters'):
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0 and module_params != total_params:
                other_params += module_params
    
    # Calculate potential savings
    analysis = {
        'total_params': total_params,
        'embedding_params': embedding_params,
        'linear_params': linear_params,
        'other_params': other_params,
        'embedding_fraction': embedding_params / total_params if total_params > 0 else 0,
        'linear_fraction': linear_params / total_params if total_params > 0 else 0,
    }
    
    # Vocabulary reduction potential (e.g., 32K -> 2K)
    if embedding_params > 0:
        vocab_reduction_16x = {
            'strategy': 'Vocabulary reduction (32K -> 2K)',
            'params_before': total_params,
            'params_after': total_params - (embedding_params * 15 / 16),
            'reduction_ratio': total_params / (total_params - (embedding_params * 15 / 16))
        }
        analysis['vocab_reduction_16x'] = vocab_reduction_16x
    
    # Quantization potential (32-bit -> 8-bit)
    quant_8bit = {
        'strategy': '8-bit quantization',
        'memory_before_mb': (total_params * 4) / 1024 / 1024,
        'memory_after_mb': (total_params * 1) / 1024 / 1024,
        'reduction_ratio': 4.0
    }
    analysis['quantization_8bit'] = quant_8bit
    
    # Quantization potential (32-bit -> 4-bit)
    quant_4bit = {
        'strategy': '4-bit quantization',
        'memory_before_mb': (total_params * 4) / 1024 / 1024,
        'memory_after_mb': (total_params * 0.5) / 1024 / 1024,
        'reduction_ratio': 8.0
    }
    analysis['quantization_4bit'] = quant_4bit
    
    # Combined potential (vocab + 4-bit quant)
    if embedding_params > 0:
        reduced_params = total_params - (embedding_params * 15 / 16)
        combined_reduction = total_params / (reduced_params / 8)  # 8x from quant
        
        combined = {
            'strategy': 'Vocab reduction (16x) + 4-bit quant (8x)',
            'params_before': total_params,
            'effective_params': reduced_params / 8,
            'reduction_ratio': combined_reduction
        }
        analysis['combined_compression'] = combined
    
    return analysis


def print_compression_analysis(analysis: Dict):
    """Pretty print compression analysis"""
    print("\n" + "=" * 70)
    print("COMPRESSION POTENTIAL ANALYSIS")
    print("=" * 70)
    
    print(f"\nCurrent Model:")
    print(f"  Total parameters: {analysis['total_params']:,}")
    print(f"  Embedding params: {analysis['embedding_params']:,} ({analysis['embedding_fraction']:.1%})")
    print(f"  Linear params: {analysis['linear_params']:,} ({analysis['linear_fraction']:.1%})")
    
    print(f"\nCompression Opportunities:")
    
    if 'vocab_reduction_16x' in analysis:
        vr = analysis['vocab_reduction_16x']
        print(f"\n1. {vr['strategy']}")
        print(f"   Before: {vr['params_before']:,} params")
        print(f"   After: {vr['params_after']:,.0f} params")
        print(f"   Reduction: {vr['reduction_ratio']:.2f}x")
    
    q8 = analysis['quantization_8bit']
    print(f"\n2. {q8['strategy']}")
    print(f"   Memory before: {q8['memory_before_mb']:.2f} MB")
    print(f"   Memory after: {q8['memory_after_mb']:.2f} MB")
    print(f"   Reduction: {q8['reduction_ratio']:.1f}x")
    
    q4 = analysis['quantization_4bit']
    print(f"\n3. {q4['strategy']}")
    print(f"   Memory before: {q4['memory_before_mb']:.2f} MB")
    print(f"   Memory after: {q4['memory_after_mb']:.2f} MB")
    print(f"   Reduction: {q4['reduction_ratio']:.1f}x")
    
    if 'combined_compression' in analysis:
        cc = analysis['combined_compression']
        print(f"\n4. {cc['strategy']}")
        print(f"   Params before: {cc['params_before']:,}")
        print(f"   Effective params: {cc['effective_params']:,.0f}")
        print(f"   🎯 TOTAL REDUCTION: {cc['reduction_ratio']:.1f}x")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    from models import IteraLiteModel, get_small_config
    
    print("Testing compression utilities...")
    
    # Create model
    config = get_small_config()
    model = IteraLiteModel(config)
    
    # Analyze compression potential
    analysis = analyze_compression_potential(model)
    print_compression_analysis(analysis)
    
    print("\n✓ Compression utilities ready!")
