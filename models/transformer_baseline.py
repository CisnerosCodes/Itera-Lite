"""
Transformer Baseline Model for comparison with Itera-Lite
Standard decoder-only Transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from .config import TransformerConfig


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.hidden_dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
            .view(1, 1, config.max_seq_length, config.max_seq_length)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch, seq_len, hidden_size = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (batch, seq_len, 3 * hidden_size)
        
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Each (batch, seq_len, hidden_size)
        
        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # All: (batch, num_heads, seq_len, head_dim)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (batch, num_heads, seq_len, seq_len)
        
        # Apply causal mask
        attn = attn.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )
        
        # Attention weights
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
        
        # Output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer decoder block"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        # Pre-LayerNorm architecture
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = MultiHeadAttention(config)
        
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        # Attention with residual
        x = x + self.attention(self.ln1(x))
        
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class TransformerBaseline(nn.Module):
    """
    Standard Transformer decoder for language modeling
    Baseline for comparison with Itera-Lite
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position encoding (learned)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.hidden_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # LM head (output projection)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"Initialized Transformer baseline with {self.get_num_params():,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) - input token indices
            labels: (batch, seq_len) - target token indices for loss calculation
        
        Returns:
            logits: (batch, seq_len, vocab_size) - output logits
            loss: scalar or None - cross entropy loss if labels provided
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.embedding(input_ids)  # (batch, seq_len, hidden_size)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        x = x + self.position_embedding(positions)
        
        # Embedding dropout
        x = self.embed_dropout(x)
        
        # Process through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Output logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: (batch, seq_len) - input token indices
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top k logits
        
        Returns:
            generated: (batch, seq_len + max_new_tokens) - generated tokens
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to max sequence length
                input_ids_crop = input_ids[:, -self.config.max_seq_length:]
                
                # Forward pass
                logits, _ = self(input_ids_crop)
                
                # Get logits for last token
                logits = logits[:, -1, :] / temperature
                
                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get number of parameters
        
        Args:
            non_embedding: if True, exclude embedding parameters
        
        Returns:
            num_params: total number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        
        return n_params
    
    def get_efficiency_stats(self) -> dict:
        """Get efficiency statistics"""
        total_params = self.get_num_params()
        non_embed_params = self.get_num_params(non_embedding=True)
        
        # Calculate FLOPs per token (approximate)
        # Attention: O(n^2 * d) per layer
        # FFN: O(n * d * intermediate_size) per layer
        
        # For a single token (autoregressive generation):
        attn_flops = (
            self.config.num_layers *
            self.config.hidden_size ** 2 * 4  # Q, K, V, O projections
        )
        
        ffn_flops = (
            self.config.num_layers *
            (self.config.hidden_size * self.config.intermediate_size * 2)
        )
        
        total_flops = attn_flops + ffn_flops
        
        return {
            'total_params': total_params,
            'non_embedding_params': non_embed_params,
            'embedding_params': total_params - non_embed_params,
            'approx_flops_per_token': total_flops,
            'num_layers': self.config.num_layers,
            'hidden_size': self.config.hidden_size,
            'num_attention_heads': self.config.num_attention_heads,
            'intermediate_size': self.config.intermediate_size,
        }


if __name__ == "__main__":
    from .config import get_transformer_tiny_config, get_transformer_small_config
    
    print("=" * 70)
    print("Testing Transformer Baseline Model")
    print("=" * 70)
    
    # Test tiny config
    print("\n1. Testing TINY config...")
    config = get_transformer_tiny_config()
    model = TransformerBaseline(config)
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, loss = model(input_ids, labels)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Efficiency stats
    stats = model.get_efficiency_stats()
    print(f"\n   Efficiency Stats:")
    print(f"   - Total params: {stats['total_params']:,}")
    print(f"   - Non-embedding params: {stats['non_embedding_params']:,}")
    print(f"   - Approx FLOPs/token: {stats['approx_flops_per_token']:,}")
    
    # Test generation
    print(f"\n2. Testing generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"   Prompt length: {prompt.shape[1]}")
    print(f"   Generated length: {generated.shape[1]}")
    
    print("\n✓ Transformer baseline working!")
