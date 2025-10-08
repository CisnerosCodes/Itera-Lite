"""
Itera-Lite: Ultra-Efficient Mini Language Model
Combining SSM (State Space Model) + MoE (Mixture-of-Experts)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import IteraLiteConfig
from .ssm import SSMBlock
from .moe import MoELayer


class IteraLiteModel(nn.Module):
    """
    Itera-Lite: Hybrid SSM + MoE Architecture
    
    Design principles:
    - SSM backbone for efficient sequence processing (O(n) complexity)
    - Sparse MoE for conditional computation
    - Minimal parameters for extreme efficiency
    """
    
    def __init__(self, config: IteraLiteConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position encoding (learned)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Build layers (alternating SSM and MoE)
        self.layers = nn.ModuleList()
        
        for layer_idx in range(config.num_layers):
            # SSM block (always present)
            ssm_block = SSMBlock(
                d_model=config.hidden_size,
                state_size=config.ssm_state_size,
                expand_factor=config.ssm_expand_factor,
                conv_kernel=config.ssm_conv_kernel,
                dropout=config.dropout
            )
            
            # MoE layer (only on specified layers)
            use_moe = layer_idx in config.moe_layers
            moe_layer = MoELayer(
                d_model=config.hidden_size,
                num_experts=config.num_experts,
                expert_size=config.expert_size,
                top_k=config.top_k_experts,
                dropout=config.dropout,
                load_balance_weight=config.load_balance_loss_weight,
                use_moe=use_moe
            )
            
            self.layers.append(nn.ModuleDict({
                'ssm': ssm_block,
                'moe': moe_layer
            }))
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # LM head (output projection)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"Initialized Itera-Lite with {self.get_num_params():,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) - input token indices
            labels: (batch, seq_len) - target token indices for loss calculation
        
        Returns:
            logits: (batch, seq_len, vocab_size) - output logits
            loss: scalar or None - cross entropy loss if labels provided
            aux_loss: scalar - auxiliary MoE load balancing loss
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
        
        # Process through layers
        total_aux_loss = torch.tensor(0.0, device=device)
        
        for layer in self.layers:
            # SSM block
            x = layer['ssm'](x)
            
            # MoE layer
            x, aux_loss = layer['moe'](x)
            total_aux_loss = total_aux_loss + aux_loss
        
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
            
            # Add auxiliary loss
            loss = loss + total_aux_loss
        
        return logits, loss, total_aux_loss
    
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
                logits, _, _ = self(input_ids_crop)
                
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
        # For SSM: O(n * d^2) per layer
        # For MoE: O(n * d * expert_size * top_k / num_experts) per layer
        ssm_flops = (
            self.config.num_layers *
            self.config.hidden_size ** 2 *
            self.config.ssm_expand_factor * 2
        )
        
        moe_flops = (
            len(self.config.moe_layers) *
            self.config.hidden_size *
            self.config.expert_size * 2 *
            self.config.top_k_experts
        )
        
        total_flops = ssm_flops + moe_flops
        
        return {
            'total_params': total_params,
            'non_embedding_params': non_embed_params,
            'embedding_params': total_params - non_embed_params,
            'approx_flops_per_token': total_flops,
            'num_layers': self.config.num_layers,
            'hidden_size': self.config.hidden_size,
            'num_experts': self.config.num_experts,
            'moe_layers': len(self.config.moe_layers),
        }


if __name__ == "__main__":
    from .config import get_tiny_config, get_small_config
    
    print("=" * 70)
    print("Testing Itera-Lite Model")
    print("=" * 70)
    
    # Test tiny config
    print("\n1. Testing TINY config...")
    config = get_tiny_config()
    model = IteraLiteModel(config)
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, loss, aux_loss = model(input_ids, labels)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Aux loss: {aux_loss.item():.4f}")
    
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
    
    print("\n✓ Itera-Lite model working!")
