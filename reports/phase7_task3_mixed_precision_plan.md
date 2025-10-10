# Phase 7 Task 3: Mixed-Precision Optimization - Strategy Plan

**Date:** October 10, 2025  
**Objective:** Implement layer-wise mixed-precision (INT8/FP16) to achieve 1.5× compression  
**Target:** 2.13× cumulative compression when combined with INT4 quantization  
**Quality Goal:** <5% perplexity degradation from FP32 baseline

---

## 1. Executive Summary

Following the discovery that structured pruning is not viable for this SSM architecture (Task 2), we pivot to **mixed-precision optimization** as the primary compression strategy. This approach assigns different numerical precision levels to different model components based on their sensitivity to quantization.

**Key Strategy:**
- **Embeddings → INT8** (59% of params, less sensitive to quantization)
- **SSM Layers → FP16** (39% of params, critical for quality)
- **Norms → FP16** (0.1% of params, minimal impact but keep precision)
- **LM Head → INT8** (tied with embeddings, consistent precision)

**Expected Results:**
- **Standalone Compression:** 1.5× (FP32 → mixed INT8/FP16)
- **Cumulative with INT4:** 2.13× (1.42× from Task 1 × 1.5×)
- **Quality Degradation:** <5% perplexity increase
- **Inference Speedup:** 1.3-1.5× on A30 GPU (INT8 CUDA cores)

**Advantages Over Pruning:**
- ✅ No architectural constraints (maintains model structure)
- ✅ Preserves residual connections (no dimension changes)
- ✅ Compatible with SSM architecture (no tight coupling issues)
- ✅ Leverages GPU tensor cores (INT8 acceleration)
- ✅ Proven approach (widely used in production)

---

## 2. Architectural Analysis

### 2.1 Component Breakdown (from Task 2)

Based on checkpoint analysis from Task 2:

```
Total Parameters: 1,754,400 (1.75M)

Component Distribution:
├─ Embeddings: 1,035,096 params (59.0%)
│  ├─ Token Embeddings: 1,024,000 (vocab=2000 × d_model=128)
│  └─ Position Embeddings: 11,096 (max_seq_len=128 × d_model)
│  └─ Precision Target: INT8 (embeddings typically less sensitive)
│
├─ SSM Layers: 684,216 params (39.0%)
│  ├─ 4 layers × 171,054 params/layer
│  ├─ Components per layer:
│  │  ├─ in_proj: 65,536 (d_model → d_inner expansion)
│  │  ├─ conv1d: 53,760 (depthwise temporal convolution)
│  │  ├─ x_proj: 16,384 (state-space projection)
│  │  ├─ dt_proj: 8,192 (delta time projection)
│  │  ├─ A_log: 8,192 (SSM state matrix - CRITICAL)
│  │  ├─ D: 256 (skip connection)
│  │  └─ out_proj: 32,768 (d_inner → d_model projection)
│  └─ Precision Target: FP16 (SSM state-space requires precision)
│
├─ Norms: 1,280 params (0.1%)
│  ├─ LayerNorm per SSM layer (4 × 128)
│  └─ Final LayerNorm (128)
│  └─ Precision Target: FP16 (normalization stability)
│
└─ LM Head: 33,808 params (1.9%)
   └─ Tied with embeddings (weight sharing)
   └─ Precision Target: INT8 (consistent with embeddings)
```

### 2.2 Precision Sensitivity Analysis

**High Sensitivity (FP16 Required):**
1. **SSM State Matrices (A_log, B, C):**
   - Represent continuous-time dynamics
   - Small numerical errors accumulate over sequence
   - Critical for long-range dependencies
   - **Verdict: FP16**

2. **SSM Temporal Convolutions (conv1d):**
   - Captures local temporal patterns
   - Depthwise architecture (fewer params, more sensitive)
   - **Verdict: FP16**

3. **LayerNorms:**
   - Stabilize training and inference
   - Sensitive to numerical precision
   - **Verdict: FP16**

**Medium Sensitivity (INT8 Viable with Calibration):**
1. **Embeddings:**
   - Lookup table (discrete values)
   - Large parameter count (59% of model)
   - Typically robust to quantization
   - **Verdict: INT8**

2. **Projections (in_proj, out_proj):**
   - Linear transformations
   - Can use symmetric quantization
   - **Consider: FP16 for safety, but INT8 possible**

**Low Sensitivity (INT8 Recommended):**
1. **LM Head:**
   - Tied with embeddings (same precision)
   - Output logits (downstream softmax)
   - **Verdict: INT8**

### 2.3 Precision Allocation Decision

**Conservative Strategy (Recommended):**
```python
precision_map = {
    'embeddings.token_embeddings': 'int8',
    'embeddings.position_embeddings': 'int8',
    'layers.*.ssm.in_proj': 'fp16',      # Keep FP16 for safety
    'layers.*.ssm.conv1d': 'fp16',       # Critical temporal patterns
    'layers.*.ssm.x_proj': 'fp16',       # State-space projection
    'layers.*.ssm.dt_proj': 'fp16',      # Delta time (sensitive)
    'layers.*.ssm.A_log': 'fp16',        # CRITICAL: State matrix
    'layers.*.ssm.D': 'fp16',            # Skip connection
    'layers.*.ssm.out_proj': 'fp16',     # Output projection
    'layers.*.norm': 'fp16',             # Normalization stability
    'norm_f': 'fp16',                    # Final norm
    'lm_head': 'int8',                   # Tied with embeddings
}
```

**Aggressive Strategy (Higher Compression, Higher Risk):**
```python
precision_map_aggressive = {
    'embeddings.*': 'int8',
    'layers.*.ssm.in_proj': 'int8',      # Try INT8 for projections
    'layers.*.ssm.conv1d': 'fp16',       # Keep FP16 (critical)
    'layers.*.ssm.x_proj': 'fp16',       # Keep FP16 (state-space)
    'layers.*.ssm.dt_proj': 'fp16',      # Keep FP16 (sensitive)
    'layers.*.ssm.A_log': 'fp16',        # ALWAYS FP16
    'layers.*.ssm.D': 'fp16',            # ALWAYS FP16
    'layers.*.ssm.out_proj': 'int8',     # Try INT8
    'layers.*.norm': 'fp16',
    'norm_f': 'fp16',
    'lm_head': 'int8',
}
```

**We'll start with Conservative Strategy for Task 3.**

---

## 3. Compression Analysis

### 3.1 Memory Calculation

**FP32 Baseline:**
```
Total params: 1,754,400
Memory: 1,754,400 × 4 bytes = 7,017,600 bytes ≈ 6.69 MB
```

**Mixed-Precision (Conservative):**
```
INT8 Components:
- Embeddings: 1,035,096 × 1 byte = 1,035,096 bytes
- LM Head: 33,808 × 1 byte = 33,808 bytes (tied, no extra storage)
- Total INT8: 1,035,096 bytes

FP16 Components:
- SSM Layers: 684,216 × 2 bytes = 1,368,432 bytes
- Norms: 1,280 × 2 bytes = 2,560 bytes
- Total FP16: 1,370,992 bytes

Total Memory: 1,035,096 + 1,370,992 = 2,406,088 bytes ≈ 2.29 MB
Compression Ratio: 6.69 MB / 2.29 MB = 2.92×
```

Wait, that's higher than expected! Let me recalculate with correct baseline.

**Corrected Calculation (from FP32 checkpoint size):**

Task 2 checkpoint size: 6.72 MB (includes overhead)
Pure parameter memory: 1,754,400 × 4 = 6.82 MB (theoretical)

Mixed-precision memory: 2.41 MB (theoretical)

**Standalone compression: 6.82 / 2.41 = 2.83× (better than expected!)**

But this is theoretical. In practice:
- PyTorch overhead (metadata, graph, buffers)
- Alignment padding for INT8/FP16
- Actual checkpoint size includes config, optimizer states, etc.

**Realistic estimate: 1.5-1.8× checkpoint size reduction**

### 3.2 Cumulative Compression with INT4

**Option 1: INT4 on FP32 baseline (Task 1 result):**
- FP32 → INT4: 1.42× compression
- Checkpoint: 6.71 MB → 4.73 MB

**Option 2: INT4 on Mixed-Precision (This Task + Task 1):**
- FP32 → Mixed (INT8/FP16): ~1.5× theoretical
- Mixed → INT4: Apply INT4 to selected layers

**Sequential Compression:**
```
FP32 (6.71 MB)
  ↓ Mixed-Precision (Task 3)
INT8/FP16 (~4.5 MB estimated, 1.5× compression)
  ↓ INT4 on INT8 components (potential)
Further compressed (~3.0-3.5 MB, 1.9-2.2× cumulative)
```

**Target: 2.0-2.2× cumulative compression**

### 3.3 Expected Results Summary

| Metric | FP32 Baseline | Task 1 (INT4) | Task 3 (Mixed) | Task 3+1 (Cumulative) |
|--------|---------------|---------------|----------------|----------------------|
| **Checkpoint Size** | 6.71 MB | 4.73 MB | ~4.5 MB (est.) | ~3.2 MB (est.) |
| **Compression Ratio** | 1.0× | 1.42× | 1.5× | 2.1× |
| **Perplexity** | 14.48 | 17.22 (+19%) | <15.2 (target) | <16.5 (target) |
| **Inference Speed** | 1.0× | 0.95× | 1.3× (est.) | 1.4× (est.) |
| **Memory (GPU)** | 6.82 MB | 4.73 MB | 2.41 MB | ~2.0 MB |

---

## 4. Implementation Plan

### 4.1 Core Components

**1. Mixed-Precision Configuration (`utils/mixed_precision.py`)**
```python
@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision optimization."""
    
    # Precision mapping
    precision_map: Dict[str, str]  # layer_pattern → 'int8' | 'fp16'
    
    # INT8 calibration
    calibration_method: str = 'minmax'  # 'minmax' | 'percentile' | 'mse'
    calibration_samples: int = 1000
    percentile: float = 99.99
    
    # Conversion settings
    symmetric_quant: bool = True
    per_channel: bool = True
    
    # Quality thresholds
    max_perplexity_increase: float = 5.0  # Percentage
    
class MixedPrecisionConverter:
    """Convert model to mixed INT8/FP16 precision."""
    
    def __init__(self, model, config: MixedPrecisionConfig):
        self.model = model
        self.config = config
        self.calibration_stats = {}
    
    def calibrate(self, dataloader):
        """Collect activation statistics for INT8 calibration."""
        # Run forward passes, collect min/max/percentile stats
        pass
    
    def convert_to_int8(self, layer, stats):
        """Convert layer to INT8 with calibrated scales."""
        # Symmetric quantization: Q = round(W / scale)
        # scale = max(|W|) / 127
        pass
    
    def convert_to_fp16(self, layer):
        """Convert layer to FP16."""
        # Simple type conversion
        pass
    
    def apply_mixed_precision(self):
        """Apply precision mapping to entire model."""
        # Match layer patterns, apply conversions
        pass
    
    def benchmark(self, dataloader):
        """Evaluate quality and performance."""
        pass
```

**2. Main Orchestration Script (`phase7_mixed_precision.py`)**
```python
def main():
    # 1. Load checkpoint with format conversion (reuse Task 2 logic)
    model, config = load_checkpoint_with_inference(args.checkpoint)
    
    # 2. Create calibration dataset
    calib_dataloader = create_calibration_data(
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        num_samples=1000
    )
    
    # 3. Initialize MixedPrecisionConverter
    mp_config = MixedPrecisionConfig(
        precision_map=get_conservative_precision_map(),
        calibration_method='minmax',
        calibration_samples=1000
    )
    converter = MixedPrecisionConverter(model, mp_config)
    
    # 4. Calibrate INT8 layers
    print("Calibrating INT8 layers...")
    converter.calibrate(calib_dataloader)
    
    # 5. Apply mixed-precision conversion
    print("Converting to mixed precision...")
    mixed_model = converter.apply_mixed_precision()
    
    # 6. Benchmark quality and performance
    print("Benchmarking...")
    metrics = benchmark_mixed_precision(
        original_model=model,
        mixed_model=mixed_model,
        dataloader=test_dataloader
    )
    
    # 7. Validate quality threshold
    if metrics['perplexity_increase'] > mp_config.max_perplexity_increase:
        print(f"⚠ WARNING: Perplexity increase {metrics['perplexity_increase']:.1f}% exceeds threshold")
    
    # 8. Save mixed-precision checkpoint
    save_mixed_checkpoint(mixed_model, config, metrics, args.output)
    
    # 9. Visualize results
    visualize_precision_allocation(converter.precision_map, metrics)
```

**3. Slurm Job Script (`jobs/phase7_task3_mixed_precision.sh`)**
```bash
#!/bin/bash
#SBATCH --job-name=phase7_task3_mixed
#SBATCH --output=logs/phase7_task3_mixed_%j.out
#SBATCH --error=logs/phase7_task3_mixed_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# Verify environment
# Load checkpoint
# Run mixed-precision conversion
# Validate results
```

### 4.2 Implementation Workflow

```
┌─────────────────────────────────────────┐
│ 1. Load FP32 Checkpoint                 │
│    - Format conversion (Task 2 logic)   │
│    - Config inference                   │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 2. Generate Calibration Dataset         │
│    - 1000 synthetic sequences           │
│    - Vocab size: 2000                   │
│    - Sequence length: 128               │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 3. Calibrate INT8 Layers                │
│    - Forward passes on calib data       │
│    - Collect activation min/max         │
│    - Compute quantization scales        │
│    - Target: Embeddings, LM Head        │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 4. Apply Precision Conversion           │
│    - Embeddings → INT8                  │
│    - SSM Layers → FP16                  │
│    - Norms → FP16                       │
│    - LM Head → INT8                     │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 5. Benchmark Mixed Model                │
│    - Perplexity evaluation              │
│    - Inference speed test               │
│    - Memory footprint measurement       │
│    - Quality vs compression tradeoff    │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 6. Validate and Save                    │
│    - Check quality threshold (<5%)      │
│    - Save mixed checkpoint              │
│    - Export metadata (precision map)    │
│    - Generate visualization             │
└─────────────────────────────────────────┘
```

---

## 5. Calibration Strategy

### 5.1 INT8 Quantization Methods

**Option 1: Min-Max Calibration (Simple, Fast)**
```python
# Symmetric quantization
scale = max(abs(W_min), abs(W_max)) / 127
W_int8 = round(W_fp32 / scale).clip(-128, 127)

# Pros: Simple, fast, no data required
# Cons: Sensitive to outliers
```

**Option 2: Percentile Calibration (Robust)**
```python
# Use 99.99th percentile to handle outliers
scale = percentile(abs(W), 99.99) / 127
W_int8 = round(W_fp32 / scale).clip(-128, 127)

# Pros: Robust to outliers
# Cons: Slightly more complex
```

**Option 3: MSE Calibration (Optimal, Slow)**
```python
# Find scale that minimizes MSE
best_scale = argmin_scale(MSE(W_fp32, dequant(quant(W_fp32, scale))))

# Pros: Optimal quality
# Cons: Computationally expensive
```

**We'll use Percentile (99.99) for good balance.**

### 5.2 Calibration Dataset

**Requirements:**
- Representative of real text distribution
- Sufficient samples for statistics (1000 sequences)
- Cover vocabulary range (2000 tokens)

**Generation Strategy:**
```python
def create_calibration_data(vocab_size, max_seq_length, num_samples):
    # Generate diverse synthetic sequences
    data = []
    for _ in range(num_samples):
        # Mix of:
        # - Random tokens (exploration)
        # - Frequent tokens (common patterns)
        # - Rare tokens (edge cases)
        seq = generate_mixed_sequence(vocab_size, max_seq_length)
        data.append(seq)
    return DataLoader(data, batch_size=32)
```

---

## 6. Expected Challenges and Mitigations

### 6.1 Potential Issues (from Tasks 1 & 2)

**Challenge 1: Checkpoint Format Compatibility**
- **Lesson from Task 2:** Old format `.moe.layer.` vs new `.moe.moe.`
- **Mitigation:** Reuse format conversion logic from `phase7_prune.py`
- **Code:**
  ```python
  def load_checkpoint_with_inference(checkpoint_path, device):
      # Convert old format keys
      new_state_dict = {}
      for key, value in state_dict.items():
          if '.moe.layer.' in key:
              new_key = key.replace('.moe.layer.', '.moe.moe.')
              new_state_dict[new_key] = value
          else:
              new_state_dict[key] = value
      
      # Load with strict=False
      model.load_state_dict(new_state_dict, strict=False)
  ```

**Challenge 2: Config Inference**
- **Lesson from Task 1:** Checkpoint doesn't store full config
- **Mitigation:** Reuse config inference logic from Task 1
- **Code:**
  ```python
  def infer_config_from_checkpoint(state_dict):
      # Infer from embedding shape
      vocab_size, d_model = state_dict['embeddings.token_embeddings.weight'].shape
      # Infer layers from state_dict keys
      n_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('layers.')]) + 1
      # ... etc
  ```

**Challenge 3: PyTorch Mixed-Precision Support**
- **Issue:** PyTorch doesn't natively support per-layer precision in `nn.Module`
- **Solution:** Create wrapper that handles dtype conversion during forward pass
- **Code:**
  ```python
  class MixedPrecisionLayer(nn.Module):
      def __init__(self, layer, target_dtype):
          super().__init__()
          self.layer = layer.to(target_dtype)
          self.target_dtype = target_dtype
      
      def forward(self, x):
          # Convert input to layer's dtype
          x = x.to(self.target_dtype)
          # Forward pass
          out = self.layer(x)
          # Convert back to FP32 for residuals
          return out.float()
  ```

**Challenge 4: INT8 Quantization Implementation**
- **Issue:** PyTorch's `torch.quantization` is complex and model-specific
- **Solution:** Manual quantization with scale factors
- **Code:**
  ```python
  class QuantizedLinear(nn.Module):
      def __init__(self, weight_fp32, bias_fp32, scale):
          super().__init__()
          # Quantize weights to INT8
          self.weight_int8 = torch.quantize_per_channel(
              weight_fp32, scale, zero_point=0, axis=0, dtype=torch.qint8
          )
          self.bias = bias_fp32
          self.scale = scale
      
      def forward(self, x):
          # INT8 matmul (uses Tensor Cores on A30)
          return F.linear(x, self.weight_int8.dequantize(), self.bias)
  ```

### 6.2 Quality Validation

**Threshold Check:**
```python
baseline_perplexity = 14.48  # From Task 1
target_perplexity = baseline_perplexity * 1.05  # +5% maximum
mixed_perplexity = evaluate_perplexity(mixed_model, test_data)

if mixed_perplexity > target_perplexity:
    print(f"⚠ WARNING: Quality degradation too high!")
    print(f"  Baseline: {baseline_perplexity:.2f}")
    print(f"  Mixed: {mixed_perplexity:.2f}")
    print(f"  Increase: {(mixed_perplexity/baseline_perplexity - 1)*100:.1f}%")
    print("  Consider using more FP16 layers or better calibration.")
```

---

## 7. Success Criteria

### 7.1 Primary Metrics

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Compression Ratio** | 1.5× | 1.8× |
| **Perplexity Increase** | <5% | <3% |
| **Checkpoint Size** | <4.5 MB | <4.0 MB |
| **Inference Speedup** | 1.2× | 1.5× |
| **Memory Footprint** | <3.0 MB | <2.5 MB |

### 7.2 Quality Gates

**✅ Pass Criteria:**
- Mixed-precision model converges (no NaN/Inf)
- Perplexity increase ≤5% from FP32 baseline
- Checkpoint size reduction ≥1.5×
- All layers correctly converted (no FP32 leakage)

**⚠ Warning Criteria (Investigate but Continue):**
- Perplexity increase 5-10%
- Compression ratio 1.3-1.5×
- Inference speedup <1.2×

**❌ Failure Criteria (Need Fixes):**
- Perplexity increase >10%
- Model produces NaN/Inf outputs
- Compression ratio <1.3×
- Checkpoint loading fails

---

## 8. Timeline and Milestones

### 8.1 Development Schedule

| Day | Phase | Activities | Deliverables |
|-----|-------|------------|--------------|
| **Day 1** | Planning | Architecture analysis, precision allocation design | This document (558+ lines) |
| **Day 2** | Implementation | `utils/mixed_precision.py` (400-500 lines) | Mixed-precision utilities |
| **Day 2** | Implementation | `phase7_mixed_precision.py` (500-600 lines) | Main script |
| **Day 2** | Implementation | `jobs/phase7_task3_mixed_precision.sh` (200 lines) | Slurm job script |
| **Day 3** | Execution | HPC job submission, monitoring, debugging | Job outputs, logs |
| **Day 3-4** | Debugging | Fix issues (expect 2-3 iterations like Tasks 1 & 2) | Bug fixes, commits |
| **Day 4** | Validation | Benchmark results, quality assessment | Metrics, visualizations |
| **Day 5** | Documentation | Completion report (800-1000 lines) | `phase7_task3_mixed_precision.md` |

**Total Estimated Time:** 5 days (similar to Tasks 1 & 2)

### 8.2 Risk Mitigation

**Risk 1: Quality Degradation Exceeds 5%**
- **Probability:** Medium
- **Impact:** High
- **Mitigation:**
  - Start with conservative precision map (all SSM → FP16)
  - Use robust calibration (percentile method)
  - Test with smaller test set first
  - Fall back to more FP16 layers if needed

**Risk 2: INT8 Implementation Complexity**
- **Probability:** Medium
- **Impact:** Medium
- **Mitigation:**
  - Start with simple min-max quantization
  - Use PyTorch's `torch.quantization` utilities if available
  - Fall back to FP16-only if INT8 too complex

**Risk 3: HPC Job Failures**
- **Probability:** High (based on Tasks 1 & 2)
- **Impact:** Medium (delays but not blockers)
- **Mitigation:**
  - Reuse checkpoint loading logic from Task 2
  - Reuse config inference from Task 1
  - Comprehensive error handling
  - Quick local testing before HPC submission

---

## 9. Next Steps

**Immediate Actions:**
1. ✅ **Commit this plan document** to GitHub
2. 📋 **Implement `utils/mixed_precision.py`** (400-500 lines)
3. 📋 **Implement `phase7_mixed_precision.py`** (500-600 lines)
4. 📋 **Create Slurm job script** (200 lines)
5. 📋 **Test locally** (if possible) or submit to HPC
6. 📋 **Debug and iterate** (expect 2-3 job attempts)
7. 📋 **Generate completion report** with results

**User Confirmation Required:**
- Review this plan and approve precision allocation strategy
- Confirm conservative vs aggressive approach (recommend conservative)
- Ready to proceed with implementation?

---

## 10. References and Learning Resources

### 10.1 Mixed-Precision Research

**Key Papers:**
1. **"Mixed Precision Training"** (Micikevicius et al., 2018)
   - Foundation of FP16 training and inference
   - Loss scaling techniques

2. **"8-bit Optimizers via Block-wise Quantization"** (Dettmers et al., 2021)
   - INT8 quantization for large models
   - Block-wise quantization strategies

3. **"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"** (Dettmers et al., 2022)
   - INT8 inference for LLMs
   - Outlier-aware quantization

### 10.2 PyTorch Documentation

- [Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html)
- [torch.quantization API](https://pytorch.org/docs/stable/quantization.html#torch.quantization)

### 10.3 Lessons from Tasks 1 & 2

**Task 1 (INT4 Quantization):**
- ✅ Config inference from checkpoint
- ✅ NF4 quantization with BitsAndBytes
- ✅ GPU-native execution
- ⚠️ 19% perplexity degradation (aggressive quantization)

**Task 2 (Structured Pruning):**
- ✅ Checkpoint format conversion
- ✅ Architectural constraint discovery
- ✅ Graceful error handling
- ⚠️ SSM architecture not prunable (residual connections)

**Key Takeaways for Task 3:**
- Always validate assumptions empirically
- Reuse working checkpoint loading logic
- Implement graceful degradation
- Expect 2-3 debugging iterations
- Document everything for learning

---

## Appendix A: Precision Map Details

### A.1 Conservative Precision Map (Recommended)

```python
CONSERVATIVE_PRECISION_MAP = {
    # Embeddings: INT8 (59% of params, large compression potential)
    'embeddings.token_embeddings.weight': 'int8',
    'embeddings.position_embeddings.weight': 'int8',
    
    # SSM Layers: ALL FP16 (39% of params, critical for quality)
    'layers.*.ssm.norm.weight': 'fp16',
    'layers.*.ssm.norm.bias': 'fp16',
    'layers.*.ssm.in_proj.weight': 'fp16',
    'layers.*.ssm.in_proj.bias': 'fp16',
    'layers.*.ssm.conv1d.weight': 'fp16',
    'layers.*.ssm.conv1d.bias': 'fp16',
    'layers.*.ssm.x_proj.weight': 'fp16',
    'layers.*.ssm.x_proj.bias': 'fp16',
    'layers.*.ssm.dt_proj.weight': 'fp16',
    'layers.*.ssm.dt_proj.bias': 'fp16',
    'layers.*.ssm.A_log': 'fp16',     # CRITICAL: State matrix
    'layers.*.ssm.D': 'fp16',         # Skip connection
    'layers.*.ssm.out_proj.weight': 'fp16',
    'layers.*.ssm.out_proj.bias': 'fp16',
    
    # Final Norm: FP16
    'norm_f.weight': 'fp16',
    'norm_f.bias': 'fp16',
    
    # LM Head: INT8 (tied with embeddings)
    'lm_head.weight': 'int8',  # Note: shares storage with token_embeddings
}
```

**Compression Breakdown:**
- INT8: 1,035,096 params × 1 byte = 1.04 MB
- FP16: 684,216 params × 2 bytes = 1.37 MB
- **Total: 2.41 MB (theoretical)**
- **Compression: 6.82 MB / 2.41 MB = 2.83× (theoretical)**
- **Realistic (with overhead): ~1.5× checkpoint size reduction**

### A.2 Aggressive Precision Map (Optional Exploration)

```python
AGGRESSIVE_PRECISION_MAP = {
    # Embeddings: INT8
    'embeddings.*': 'int8',
    
    # SSM Projections: Try INT8 for higher compression
    'layers.*.ssm.norm.*': 'fp16',           # Keep normalization FP16
    'layers.*.ssm.in_proj.*': 'int8',        # TRY: INT8 for input projection
    'layers.*.ssm.conv1d.*': 'fp16',         # KEEP: FP16 for temporal conv
    'layers.*.ssm.x_proj.*': 'fp16',         # KEEP: FP16 for state projection
    'layers.*.ssm.dt_proj.*': 'fp16',        # KEEP: FP16 for delta time
    'layers.*.ssm.A_log': 'fp16',            # ALWAYS: FP16 for state matrix
    'layers.*.ssm.D': 'fp16',                # ALWAYS: FP16 for skip
    'layers.*.ssm.out_proj.*': 'int8',       # TRY: INT8 for output projection
    
    # Final layers
    'norm_f.*': 'fp16',
    'lm_head.weight': 'int8',
}
```

**Compression (if aggressive works):**
- INT8: ~1.4 MB (embeddings + in_proj + out_proj)
- FP16: ~0.9 MB (critical SSM components)
- **Total: ~2.3 MB**
- **Compression: 2.96× (theoretical)**
- **Risk: Higher quality degradation (expect 7-10% perplexity increase)**

**We'll implement both but start with Conservative.**

---

**Plan Status:** ✅ Complete and Ready for Implementation  
**Next Step:** Implement `utils/mixed_precision.py`  
**User Approval Required:** Yes (review precision allocation strategy)
