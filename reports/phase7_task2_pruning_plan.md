# Phase 7 Task 2: Structured Pruning - Implementation Plan

**Date:** October 9, 2025  
**Status:** 📋 PLANNING  
**Target:** 40% parameter reduction (1.9M → 1.1M params)  
**Expected Cumulative Compression:** 2.37× (INT4 1.42× × Pruning 1.67×)

---

## Executive Summary

Design a **structured pruning strategy** for the IteraLite model to achieve 40% parameter sparsity while preserving critical architectural components. The strategy leverages **magnitude-based pruning** with layer-specific sparsity allocation, targeting redundant MoE expert capacity while preserving essential SSM sequence modeling layers.

**Key Innovation:** Differentially prune MoE experts (high redundancy) vs SSM layers (critical for sequence processing) based on architectural role and parameter distribution analysis.

---

## Architecture Analysis

### IteraLite Component Breakdown

```
IteraLite Model (1,886,496 parameters)
├── Embeddings (1,024,000 params - 54.3%)
│   ├── token_embedding: vocab_size × hidden_size = 8000 × 128 = 1,024,000
│   └── position_embedding: max_seq_length × hidden_size = 128 × 128 = 16,384
│
├── Layers (4 layers × ~200K params = ~800K params - 42.4%)
│   ├── SSM Blocks (per layer: ~100K params)
│   │   ├── in_proj: 128 → 512 (65,536)
│   │   ├── conv1d: depthwise 256 (1,024)
│   │   ├── ssm.A_log: (8,)
│   │   ├── ssm.B: 8 × 256 = 2,048
│   │   ├── ssm.C: 256 × 8 = 2,048
│   │   ├── ssm.D: (256,)
│   │   ├── ssm.delta_proj: 256 → 256 (65,536)
│   │   ├── out_proj: 256 → 128 (32,768)
│   │   └── norm: LayerNorm (256)
│   │
│   └── MoE Layers (per layer: ~100K params)
│       ├── Router: 128 → 4 experts (512)
│       ├── Experts (4 experts × ~24K params = ~96K)
│       │   ├── w1: 128 → 64 = 8,192 per expert
│       │   └── w2: 64 → 128 = 8,192 per expert
│       │   Total per expert: 16,384 params
│       └── norm: LayerNorm (128)
│
├── Final Norm (256 params)
└── LM Head (tied with embedding - 0 additional params)
```

### Parameter Distribution Analysis

| Component | Parameters | Percentage | Pruning Priority |
|-----------|-----------|------------|------------------|
| **Embeddings** | 1,024,000 | 54.3% | **LOW** (quality critical) |
| **Position Embeddings** | 16,384 | 0.9% | **PRESERVE** (structural) |
| **SSM Layers (4×)** | ~400,000 | 21.2% | **MEDIUM** (preserve core, prune expansions) |
| **MoE Layers (4×)** | ~384,000 | 20.4% | **HIGH** (redundant capacity) |
| **Norms** | ~2,000 | 0.1% | **PRESERVE** (stability) |
| **LM Head** | 0 (tied) | 0% | N/A |

---

## Pruning Strategy

### 1. Layer-Specific Sparsity Allocation

**Principle:** Non-uniform pruning based on architectural role and redundancy

| Layer Type | Sparsity Target | Rationale |
|------------|----------------|-----------|
| **Embeddings** | 0% (preserve) | Vocabulary coverage essential, tied with LM head |
| **Position Embeddings** | 0% (preserve) | Structural constraint (128 positions) |
| **SSM in_proj** | 30% | Expansion layer - some redundancy tolerable |
| **SSM conv1d** | 20% | Depthwise - preserve temporal patterns |
| **SSM out_proj** | 30% | Projection - can compress |
| **SSM delta_proj** | 25% | Selective scan control - moderate pruning |
| **SSM state params** | 0% (preserve) | Core SSM mechanism (A, B, C, D) - critical |
| **MoE Experts** | **60%** | High redundancy, sparse activation (top-k=2) |
| **MoE Router** | 10% | Small but important for expert selection |
| **LayerNorms** | 0% (preserve) | Stability and normalization |

**Expected Parameter Reduction:**
- SSM Layers: ~400K → ~300K (25% reduction)
- MoE Layers: ~384K → ~154K (60% reduction)
- **Total: 1,886K → ~1,130K (40% reduction)** ✅

### 2. Pruning Method: Magnitude-Based Structured Pruning

**Approach:** L1-norm magnitude scoring at neuron/channel level

```python
# Pseudocode
for each Linear layer:
    # Compute importance scores
    scores = layer.weight.abs().sum(dim=output_dim)  # L1 norm per input neuron
    
    # Identify least important neurons
    threshold = percentile(scores, sparsity_level)
    prune_mask = scores < threshold
    
    # Apply structured pruning (entire neuron/channel)
    layer.weight[:, prune_mask] = 0  # Zero out input connections
    layer.weight[prune_mask, :] = 0  # Zero out output connections
```

**Advantages:**
- **Structured:** Maintains dense matrix operations (GPU-friendly)
- **Simple:** No complex dependencies or sensitivity analysis
- **Effective:** Magnitude correlates with importance for ReLU/GELU networks
- **Fast:** Single-pass computation, no iterative pruning needed

**Alternative Considered:** 
- **Gradient-based pruning:** Rejected (requires expensive Hessian computation)
- **Movement pruning:** Rejected (requires training integration)
- **Unstructured pruning:** Rejected (requires sparse kernels, slower on GPU)

### 3. Critical Component Preservation

**Must Preserve (0% pruning):**

1. **SSM State Parameters:**
   - `A_log` (8 params): State transition matrix - defines temporal dynamics
   - `B` (2,048 params): Input-to-state projection - maps inputs to SSM state
   - `C` (2,048 params): State-to-output projection - extracts SSM output
   - `D` (256 params): Skip connection - preserves input information
   - **Rationale:** Core S4 mechanism, pruning breaks recurrent structure

2. **Embeddings:**
   - `token_embedding` (1.024M params): Vocabulary representation
   - `position_embedding` (16K params): Positional encoding
   - **Rationale:** Tied with LM head, pruning creates vocabulary gaps

3. **LayerNorms:**
   - All LayerNorm parameters (~2K total)
   - **Rationale:** Critical for training stability and normalization

### 4. Differential MoE Expert Pruning

**Strategy:** Prune redundant expert capacity (60% sparsity)

**Motivation:**
- IteraLite uses **top-k=2** routing (only 2 of 4 experts activated per token)
- 50% expert capacity unused at inference
- MoE provides redundancy for training stability, less critical at inference
- Each expert: 16,384 params → after pruning: 6,554 params

**Implementation:**
```python
# For each MoE layer with 4 experts
for expert in moe_layer.experts:
    # Prune 60% of w1 and w2 weights
    prune_linear(expert.w1, sparsity=0.6)  # 128 → 64
    prune_linear(expert.w2, sparsity=0.6)  # 64 → 128
    
# Total MoE reduction per layer:
# 4 experts × 16,384 params = 65,536
# After pruning: 4 experts × 6,554 params = 26,214
# Reduction: 60% ✅
```

**Quality Preservation:**
- Top-2 routing ensures at least 2 experts remain functional
- Magnitude pruning preserves most important expert pathways
- Load balancing loss during training created expert diversity
- Fine-tuning redistributes computation to remaining capacity

---

## Fine-Tuning Strategy

### 1. GPU-Accelerated Fine-Tuning on A30

**Configuration:**
- **Epochs:** 3-5 (adaptive based on perplexity convergence)
- **Learning Rate:** 1e-4 (10× lower than original training)
- **Batch Size:** 32 (memory efficient for A30 24GB)
- **Optimizer:** AdamW with weight decay 0.01
- **LR Schedule:** Cosine annealing with warmup (5% of total steps)
- **Gradient Clipping:** 1.0 (prevent instability)
- **Data:** TinyStories (same as original training)

**Expected Speedup vs CPU:**
- A30 GPU: ~10× faster fine-tuning (1-2 hours)
- CPU: ~20 hours (prohibitive)

### 2. Fine-Tuning Procedure

```python
# 1. Prune model
pruner = StructuredPruner(model, target_sparsity=0.4)
pruned_model = pruner.prune()

# 2. Fine-tune on A30
optimizer = AdamW(pruned_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = pruned_model(batch['input_ids'], labels=batch['labels'])
        loss = outputs[1]  # (logits, loss, aux_loss)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pruned_model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    # Evaluate perplexity
    val_perplexity = evaluate(pruned_model, val_dataloader)
    print(f"Epoch {epoch}: Perplexity {val_perplexity:.2f}")
    
    # Early stopping if converged
    if val_perplexity < threshold:
        break
```

### 3. Fine-Tuning Objectives

**Primary Goal:** Recover from pruning-induced quality degradation
- **Target:** <5% perplexity increase vs baseline
- **Mechanism:** Remaining neurons learn to compensate for pruned capacity

**Secondary Goal:** Adapt router for sparser expert networks
- MoE routers re-learn to utilize pruned expert capacity
- Load balancing adjusts to new expert distributions

---

## Expected Results

### 1. Compression Metrics

| Metric | Baseline | After Pruning | Improvement |
|--------|----------|--------------|-------------|
| **Parameters** | 1,886,496 | ~1,130,000 | **1.67× reduction** |
| **Model Size** | 7.23 MB | ~4.33 MB | **1.67× compression** |
| **Inference Memory** | 7.23 MB | ~4.33 MB | **1.67× reduction** |

### 2. Cumulative with INT4 Quantization

**Combined Optimization (Pruning + INT4):**
- Prune: 1.9M → 1.1M params (1.67×)
- Quantize pruned model to INT4: 1.1M params × 4 bits/param
- **Expected size:** ~2.7 MB inference memory
- **Cumulative compression:** 1.67× × 1.42× = **2.37× vs FP32 baseline**

### 3. Speed Improvements

**Unlike Task 1 (small model overhead), pruning provides real speedup:**

| Metric | Baseline | Pruned | Pruned + INT4 |
|--------|----------|--------|---------------|
| **Inference Time** | 1.41s | ~0.95s | ~0.75s |
| **Speedup** | 1.0× | **1.5×** | **1.9×** |

**Why pruning helps speed (unlike INT4 alone):**
- Fewer FLOPs (40% less computation)
- Reduced memory bandwidth (smaller activations)
- Better cache utilization
- No quantization/dequantization overhead

### 4. Quality Preservation

**Target:** <5% perplexity degradation

| Model | Perplexity | Change |
|-------|-----------|--------|
| **Baseline FP32** | 25,080 | - |
| **Pruned (40%) before fine-tuning** | ~35,000 | +40% ⚠️ |
| **Pruned + Fine-tuned (3-5 epochs)** | <26,300 | **<5%** ✅ |
| **Pruned + Fine-tuned + INT4** | <27,500 | **<10%** ✅ |

---

## Implementation Plan

### File 1: `utils/structured_pruning.py` (~450 lines)

**Class:** `StructuredPruner`

```python
class StructuredPruner:
    """GPU-accelerated structured pruning for IteraLite"""
    
    def __init__(self, model, config: PruningConfig):
        self.model = model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def analyze_model(self) -> Dict:
        """Analyze parameter distribution and compute pruning targets"""
        
    def compute_importance_scores(self, layer: nn.Linear) -> torch.Tensor:
        """L1-norm magnitude scoring at neuron level"""
        
    def prune_linear_layer(self, layer: nn.Linear, sparsity: float):
        """Apply structured pruning to Linear layer"""
        
    def prune_ssm_layers(self):
        """Prune SSM blocks with layer-specific sparsity"""
        
    def prune_moe_experts(self):
        """Aggressively prune MoE expert networks (60% sparsity)"""
        
    def apply_pruning(self) -> nn.Module:
        """Execute full pruning pipeline"""
        
    def get_pruning_statistics(self) -> Dict:
        """Compute before/after statistics"""
        
    def visualize_sparsity(self, save_path: str):
        """Generate sparsity visualization plots"""
```

**Key Methods:**

1. **`compute_importance_scores()`**
   - L1-norm: `scores = layer.weight.abs().sum(dim=1)`
   - Returns per-neuron importance scores

2. **`prune_linear_layer()`**
   - Compute threshold: `threshold = torch.quantile(scores, sparsity)`
   - Create mask: `mask = scores >= threshold`
   - Apply structured pruning: zero out entire neurons

3. **`prune_moe_experts()`**
   - Iterate over 4 layers × 4 experts = 16 expert networks
   - Apply 60% sparsity to each expert's w1 and w2
   - Preserve router weights (10% sparsity)

4. **`apply_pruning()`**
   - Full pipeline: analyze → score → prune → verify
   - Returns pruned model ready for fine-tuning

### File 2: `phase7_prune.py` (~350 lines)

**Main Orchestration Script**

```python
def main():
    # 1. Load checkpoint with config inference (reuse Task 1 logic)
    model, config = load_checkpoint_with_inference(args.checkpoint)
    
    # 2. Initialize pruner
    prune_config = PruningConfig(
        target_sparsity=0.4,
        layer_sparsity={'ssm': 0.25, 'moe': 0.6, 'embeddings': 0.0}
    )
    pruner = StructuredPruner(model, prune_config)
    
    # 3. Apply pruning
    pruned_model = pruner.apply_pruning()
    
    # 4. Fine-tune on A30 GPU
    fine_tuned_model = fine_tune_pruned_model(
        pruned_model,
        epochs=args.epochs,
        lr=args.learning_rate,
        device='cuda'
    )
    
    # 5. Benchmark
    results = benchmark_pruning(
        baseline_model=model,
        pruned_model=fine_tuned_model,
        device='cuda'
    )
    
    # 6. Export checkpoint
    save_pruned_checkpoint(
        fine_tuned_model,
        config,
        results,
        output_path='checkpoints/pruned/'
    )
```

**Sections:**
1. Config inference (reuse from Task 1)
2. Pruner initialization
3. Pruning execution
4. Fine-tuning loop (3-5 epochs)
5. Benchmarking (baseline vs pruned)
6. Checkpoint export

### File 3: `jobs/phase7_task2_prune.sh` (~180 lines)

**Slurm Job Script**

```bash
#!/bin/bash
#SBATCH --job-name=phase7_task2_prune
#SBATCH --output=logs/phase7_task2_prune_%j.out
#SBATCH --error=logs/phase7_task2_prune_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00  # 8 hours for fine-tuning

# Activate environment
source .venv/bin/activate

# Verify GPU
nvidia-smi

# Run pruning with fine-tuning
python phase7_prune.py \
    --checkpoint checkpoints/itera_lite_tiny_best.pt \
    --output checkpoints/pruned/itera_lite_pruned_40pct.pt \
    --target-sparsity 0.4 \
    --finetune-epochs 5 \
    --finetune-lr 1e-4 \
    --batch-size 32 \
    --device cuda

echo "Pruning Complete!"
```

**Resource Allocation:**
- 8 hours (fine-tuning dominates, ~2-4 hours expected)
- 32GB RAM (model + optimizer states)
- 1 A30 GPU (24GB VRAM sufficient)

---

## Validation Criteria

### Success Metrics

✅ **Compression:** 
- Parameters: 1.9M → 1.1M (40% reduction)
- Model size: 7.23 MB → 4.33 MB (1.67× compression)

✅ **Quality:**
- Perplexity degradation: <5% vs baseline
- Generation quality: Manual inspection (coherence, grammar)

✅ **Speed:**
- Inference speedup: >1.4× vs baseline
- Fine-tuning time: <4 hours on A30

✅ **Cumulative (with INT4):**
- Combined compression: 2.37× (pruning + quantization)
- Combined speedup: ~1.9×
- Combined quality: <10% perplexity degradation

### Debugging Checklist (Lessons from Task 1)

- [ ] Config inference working for pruned model
- [ ] Sequence length constraints preserved (max_seq_length=128)
- [ ] Output format handling (tuple/tensor compatibility)
- [ ] GPU memory within 24GB VRAM limit
- [ ] Fine-tuning convergence (monitor loss curves)
- [ ] Checkpoint saving includes all metadata

---

## Timeline

**Week 2 of Phase 7 (October 10-16, 2025):**

| Day | Task | Hours | Status |
|-----|------|-------|--------|
| **Day 1 (Thu)** | Plan finalization, create utils/structured_pruning.py | 4-6h | 📋 This document |
| **Day 2 (Fri)** | Create phase7_prune.py, jobs script | 3-4h | Pending |
| **Day 3 (Sat)** | Submit HPC job, debug (expect 1-3 iterations) | 6-8h | Pending |
| **Day 4 (Sun)** | Results analysis, benchmark | 2-3h | Pending |
| **Day 5 (Mon)** | Combine pruned + INT4 quantization | 3-4h | Pending |
| **Day 6 (Tue)** | Completion report generation | 2-3h | Pending |
| **Day 7 (Wed)** | Buffer for debugging | - | Pending |

**Total Estimated:** 20-28 hours across 7 days

---

## Risks and Mitigations

### Risk 1: Excessive Quality Degradation
**Scenario:** Perplexity increases >10% after pruning  
**Mitigation:**
- Reduce MoE sparsity from 60% to 50%
- Increase fine-tuning epochs from 5 to 10
- Use layerwise pruning (prune one layer at a time, fine-tune incrementally)

### Risk 2: Fine-Tuning Doesn't Converge
**Scenario:** Loss plateau or divergence  
**Mitigation:**
- Lower learning rate (1e-4 → 5e-5)
- Add more warmup steps (5% → 10%)
- Use gradient accumulation for larger effective batch size

### Risk 3: GPU Memory Overflow
**Scenario:** OOM error on A30 (24GB VRAM)  
**Mitigation:**
- Reduce batch size (32 → 16)
- Use gradient checkpointing
- Mixed precision training (FP16)

### Risk 4: Pruned Model Breaks SSM Dynamics
**Scenario:** SSM state propagation fails  
**Mitigation:**
- Preserve SSM state parameters (A, B, C, D) - already in plan
- Reduce SSM sparsity (25% → 15%)
- Validate SSM output statistics pre/post pruning

---

## Next Steps After Task 2

### Task 3: Mixed-Precision Inference (Week 3)

**Build on pruned model:**
- Start with pruned 1.1M param model (not original 1.9M)
- Apply layer-wise precision: Embeddings INT8, SSM FP16, MoE INT4
- Expected: 2.37× → 3.08× cumulative compression

**Advantage of pruning first:**
- Smaller model = easier to fit mixed-precision formats
- Fewer parameters to quantize/calibrate
- Better memory locality for Tensor Core utilization

### Task 4: Kernel Optimization (Weeks 4-5)

**Optimize pruned + mixed-precision model:**
- Fused SSM kernels for A30
- Sparse matrix operations for pruned layers
- ONNX export for edge deployment

---

## Conclusion

This pruning plan provides a **systematic, well-motivated approach** to achieving 40% parameter reduction while preserving model quality. Key innovations:

1. **Differential pruning:** Aggressive on MoE (60%), conservative on SSM (25%)
2. **Architectural awareness:** Preserve critical SSM state parameters
3. **GPU acceleration:** 10× faster fine-tuning on A30 vs CPU
4. **Structured pruning:** Maintains dense operations (GPU-friendly)
5. **Quality-first:** <5% perplexity degradation target with fine-tuning

**Expected Outcome:** 1.67× compression standalone, 2.37× combined with INT4, enabling path to 3.08× with mixed-precision in Task 3.

**Status:** Ready to implement `utils/structured_pruning.py` and begin execution! 🚀

---

**Plan Generated:** October 9, 2025  
**Author:** GitHub Copilot  
**Next Action:** Implement `utils/structured_pruning.py` (450 lines, StructuredPruner class)
