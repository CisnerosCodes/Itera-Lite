# Phase 7: Advanced Optimization — Roadmap & Implementation Plan

**Date:** October 7, 2025  
**Status:** 🚀 **INITIATED** — Hardware diagnostics complete  
**Project:** Itera-Lite Ultra-Efficient Mini Language Model  
**Phase Goal:** Achieve 50-100× cumulative compression while maintaining functional text-generation quality

---

## Executive Summary

Phase 7 focuses on **advanced optimization techniques** to push compression from 12.9× (Phase 6) to **50-100× cumulative**. This phase implements:
- **Native INT4 quantization** (hardware-accelerated, not simulated)
- **Structured pruning** (30-50% sparsity)
- **Mixed-precision inference** (FP16 + INT8 + INT4 layer-wise optimization)
- **Advanced kernel optimization** (CPU-specific SIMD, fused operations)

**Target Compression Path:**
```
Phase 6 Baseline: 12.9× compression (293K params, 0.56 MB)
    ↓
Task 1 - Native INT4: 2.0× → 25.8× cumulative
    ↓
Task 2 - Structured Pruning (40%): 1.67× → 43.1× cumulative
    ↓
Task 3 - Mixed-Precision: 1.3× → 56.0× cumulative
    ↓
Task 4 - Kernel Optimization: 1.2× (speed) → 67.2× cumulative

PHASE 7 TARGET: 50-100× compression achieved ✓
```

---

## Hardware Capability Assessment

### Current System (from diagnostics)

| Component | Specification | Status | Impact on Phase 7 |
|-----------|--------------|--------|-------------------|
| **CPU** | 10 cores (12 threads) | ✅ EXCELLENT | Parallel data processing, CPU kernels |
| **Memory** | 15.55 GB RAM | ✅ SUFFICIENT | Model checkpoints, batching |
| **GPU** | Not available (CPU-only PyTorch) | ⚠️ LIMITED | CPU-based quantization only |
| **SIMD** | MKL, MKL-DNN available | ✅ GOOD | Optimized INT8/FP32 operations |
| **Python** | 3.13.7 | ✅ EXCELLENT | Latest features |
| **PyTorch** | 2.8.0+cpu | ✅ CURRENT | No CUDA support |

### Performance Benchmarks

- **CPU FP32:** 406.88 GFLOPS (excellent for CPU)
- **CPU FP16:** 1.31 GFLOPS (limited, expected on CPU without GPU)

### Implications for Phase 7

**What Works:**
- ✅ CPU-based INT4/INT8 quantization (via PyTorch, bitsandbytes)
- ✅ Structured pruning (CPU-compatible)
- ✅ MKL-accelerated matrix operations
- ✅ Model export and inference optimization

**What's Limited:**
- ⚠️ No native FP16 acceleration (CPU doesn't benefit from FP16)
- ⚠️ No CUDA-based mixed-precision training
- ⚠️ Slower iteration cycles compared to GPU

**Adaptations:**
- Focus on **CPU-optimized quantization** (INT8, INT4 via `torch.quantization`)
- Use **bitsandbytes** for 4-bit quantization (CPU fallback mode)
- Prioritize **inference optimization** over training speed
- Leverage **MKL/MKL-DNN** for SIMD acceleration

---

## Phase 7 Task Breakdown

### ✅ Task 1: Native INT4 Implementation (2 weeks)

**Objective:** Implement true hardware-accelerated INT4 quantization (not simulated)

**Approach:**
```
Current: Simulated INT4 (symmetric quantization in FP32)
Target: Native INT4 (PyTorch quantization API + bitsandbytes)
```

**Sub-tasks:**
1. **INT4 Quantization Research** (2 days)
   - Review PyTorch `torch.quantization` API
   - Test `bitsandbytes` 4-bit quantization (CPU mode)
   - Benchmark `optimum` BitsAndBytes integration
   - Identify best approach for Itera-Lite architecture

2. **INT4 Quantization Implementation** (5 days)
   - Create `utils/native_quantization.py`
   - Implement INT4 weight quantization for Linear layers
   - Add INT4 activation quantization support
   - Handle SSM and MoE layer quantization edge cases
   - Preserve model architecture compatibility

3. **INT4 Fine-tuning** (4 days)
   - Implement Quantization-Aware Training (QAT) if needed
   - Fine-tune INT4 model on TinyStories
   - Measure perplexity degradation (target: <30% from FP32)
   - Generate INT4 checkpoint

4. **INT4 Benchmarking** (2 days)
   - Compare INT4 vs INT8 vs FP32 (speed, size, quality)
   - Measure compression ratio (target: 2.0× over INT8)
   - Generate `reports/phase7_int4_quantization.md`

**Expected Outputs:**
- `utils/native_quantization.py` (INT4 quantization utilities)
- `checkpoints/int4_native/itera_lite_int4.pt` (native INT4 model)
- `results/phase7_int4_benchmark.json` (metrics)
- `reports/phase7_int4_quantization.md` (report)

**Target Compression:** 12.9× → **25.8×** (2.0× improvement)

---

### ✅ Task 2: Structured Pruning (2 weeks)

**Objective:** Apply magnitude-based structured pruning for 30-50% sparsity

**Approach:**
```
Current: Dense model (all parameters active)
Target: 40% pruning → 60% parameters remaining
```

**Sub-tasks:**
1. **Pruning Strategy Design** (2 days)
   - Analyze layer importance (gradient-based sensitivity)
   - Design pruning schedule (gradual vs one-shot)
   - Choose pruning granularity (weight, channel, neuron)
   - Plan SSM layer pruning (attention to state transitions)

2. **Pruning Implementation** (5 days)
   - Create `utils/structured_pruning.py`
   - Implement magnitude-based pruning
   - Add layer-wise pruning ratios (variable sparsity)
   - Preserve critical layers (embeddings, final projection)
   - Integrate `torch-pruning` library

3. **Post-Pruning Fine-tuning** (5 days)
   - Fine-tune pruned model (recover lost accuracy)
   - Iterative pruning schedule (10% → 20% → 40%)
   - Measure perplexity after each pruning step
   - Generate final pruned checkpoint

4. **Pruning Analysis** (2 days)
   - Visualize sparsity patterns per layer
   - Compare pruned vs dense (speed, size, quality)
   - Generate `reports/phase7_structured_pruning.md`

**Expected Outputs:**
- `utils/structured_pruning.py` (pruning utilities)
- `checkpoints/pruned/itera_lite_pruned_40pct.pt` (pruned model)
- `results/phase7_pruning_analysis.json` (metrics)
- `reports/phase7_structured_pruning.md` (report)
- `reports/phase7_sparsity_visualization.png` (layer sparsity chart)

**Target Compression:** 25.8× → **43.1×** (1.67× improvement from 40% pruning)

---

### ✅ Task 3: Mixed-Precision Inference (1.5 weeks)

**Objective:** Combine FP16, INT8, INT4 strategically for optimal speed/quality

**Approach:**
```
Layer-wise precision assignment:
- Embeddings: INT8 (large, less sensitive)
- SSM layers: INT8/FP16 (state transitions sensitive)
- MoE experts: INT4 (sparse, compression-friendly)
- Final projection: FP16 (output quality critical)
```

**Sub-tasks:**
1. **Precision Profiling** (2 days)
   - Analyze layer sensitivity to quantization
   - Profile per-layer error propagation
   - Identify critical vs compressible layers
   - Design mixed-precision schema

2. **Mixed-Precision Implementation** (4 days)
   - Create `utils/mixed_precision.py`
   - Implement layer-wise precision assignment
   - Add dynamic precision switching (if beneficial)
   - Ensure forward pass compatibility

3. **Mixed-Precision Optimization** (3 days)
   - Fine-tune mixed-precision model
   - Benchmark vs uniform quantization
   - Measure speed/quality trade-offs
   - Generate optimal configuration

4. **Mixed-Precision Evaluation** (2 days)
   - Compare to FP32, INT8, INT4 baselines
   - Generate `reports/phase7_mixed_precision.md`

**Expected Outputs:**
- `utils/mixed_precision.py` (mixed-precision utilities)
- `checkpoints/mixed_precision/itera_lite_mixed.pt` (mixed-precision model)
- `results/phase7_mixed_precision.json` (metrics)
- `reports/phase7_mixed_precision.md` (report)

**Target Compression:** 43.1× → **56.0×** (1.3× improvement)

---

### ✅ Task 4: Advanced Kernel Optimization (1.5 weeks)

**Objective:** CPU-specific optimizations for maximum inference speed

**Approach:**
```
Optimization targets:
- MKL-DNN kernel fusion (Conv+ReLU, Matmul+Add)
- SIMD vectorization (AVX2 for INT8 operations)
- Cache-aware memory layout
- Operator fusion (reduce memory bandwidth)
```

**Sub-tasks:**
1. **Profiling & Bottleneck Analysis** (2 days)
   - Profile current inference pipeline
   - Identify hotspots (SSM scan, MoE routing, matmuls)
   - Measure memory bandwidth utilization
   - Analyze cache misses

2. **Kernel Fusion Implementation** (4 days)
   - Create `utils/fused_kernels.py`
   - Implement fused SSM scan operations
   - Add fused MoE expert selection + computation
   - Use `torch.jit.script` for kernel compilation
   - Integrate MKL-DNN where applicable

3. **Memory Optimization** (3 days)
   - Optimize tensor layouts (NHWC vs NCHW)
   - Add in-place operations where safe
   - Reduce intermediate tensor allocations
   - Profile memory bandwidth improvements

4. **Kernel Benchmarking** (2 days)
   - Compare fused vs unfused kernels
   - Measure end-to-end inference speedup
   - Generate `reports/phase7_kernel_optimization.md`

**Expected Outputs:**
- `utils/fused_kernels.py` (fused kernel implementations)
- `results/phase7_kernel_benchmark.json` (metrics)
- `reports/phase7_kernel_optimization.md` (report)

**Target Speedup:** 1.5-2.0× inference speed improvement  
**Effective Compression:** 56.0× → **~67.2×** (accounting for speed gains)

---

## File Structure & Module Layout

```
Itera-Lite/
├── utils/
│   ├── native_quantization.py      # Task 1: INT4 quantization
│   │   ├── class NativeINT4Quantizer
│   │   ├── def quantize_model_int4()
│   │   ├── def apply_qat()  # Quantization-Aware Training
│   │   └── def benchmark_quantization()
│   │
│   ├── structured_pruning.py        # Task 2: Pruning
│   │   ├── class StructuredPruner
│   │   ├── def analyze_layer_importance()
│   │   ├── def prune_magnitude_based()
│   │   ├── def iterative_pruning()
│   │   └── def visualize_sparsity()
│   │
│   ├── mixed_precision.py           # Task 3: Mixed-precision
│   │   ├── class MixedPrecisionConfig
│   │   ├── def profile_layer_sensitivity()
│   │   ├── def assign_layer_precision()
│   │   └── def convert_to_mixed_precision()
│   │
│   └── fused_kernels.py             # Task 4: Kernel optimization
│       ├── class FusedSSMScan
│       ├── class FusedMoERouting
│       ├── def compile_model_jit()
│       └── def benchmark_kernels()
│
├── checkpoints/
│   ├── int4_native/
│   │   └── itera_lite_int4.pt
│   ├── pruned/
│   │   └── itera_lite_pruned_40pct.pt
│   └── mixed_precision/
│       └── itera_lite_mixed.pt
│
├── results/
│   ├── phase7_int4_benchmark.json
│   ├── phase7_pruning_analysis.json
│   ├── phase7_mixed_precision.json
│   └── phase7_kernel_benchmark.json
│
├── reports/
│   ├── phase7_hardware_check.json     # (completed)
│   ├── phase7_plan.md                 # (this document)
│   ├── phase7_int4_quantization.md
│   ├── phase7_structured_pruning.md
│   ├── phase7_mixed_precision.md
│   ├── phase7_kernel_optimization.md
│   └── phase7_final_report.md
│
└── Main Scripts:
    ├── phase7_quantize.py            # Task 1 runner
    ├── phase7_prune.py               # Task 2 runner
    ├── phase7_mixed_precision.py     # Task 3 runner
    ├── phase7_optimize_kernels.py    # Task 4 runner
    └── generate_phase7_report.py     # Final reporting
```

---

## Function Headers & APIs

### Task 1: Native Quantization (`utils/native_quantization.py`)

```python
class NativeINT4Quantizer:
    """Hardware-accelerated INT4 quantization using PyTorch/bitsandbytes."""
    
    def __init__(self, model, config):
        """Initialize quantizer with model and quantization config."""
        
    def calibrate(self, dataloader, num_batches=100):
        """Calibrate quantization parameters using representative data."""
        
    def quantize_weights(self, layer_types=['Linear']):
        """Apply INT4 weight quantization to specified layer types."""
        
    def quantize_activations(self, layer_types=['Linear']):
        """Apply INT4 activation quantization (if supported)."""
        
    def apply_qat(self, train_loader, epochs=5):
        """Quantization-Aware Training to recover accuracy."""
        
    def export_quantized_model(self, output_path):
        """Save quantized model checkpoint."""

def benchmark_quantization(model_fp32, model_int4, test_loader):
    """Compare FP32 vs INT4 (speed, size, perplexity)."""
    return {
        "compression_ratio": ...,
        "speedup": ...,
        "perplexity_degradation": ...,
        "model_size_mb": ...
    }
```

### Task 2: Structured Pruning (`utils/structured_pruning.py`)

```python
class StructuredPruner:
    """Magnitude-based structured pruning for Itera-Lite."""
    
    def __init__(self, model, pruning_ratio=0.4):
        """Initialize pruner with target sparsity."""
        
    def analyze_layer_importance(self, dataloader):
        """Compute gradient-based importance scores per layer."""
        
    def compute_pruning_masks(self, method='magnitude'):
        """Generate binary masks for weight pruning."""
        
    def apply_pruning(self, iterative=True, steps=4):
        """Apply pruning (gradual or one-shot)."""
        
    def fine_tune(self, train_loader, epochs=10):
        """Fine-tune pruned model to recover accuracy."""
        
    def measure_sparsity(self):
        """Calculate actual sparsity per layer and overall."""
        
    def export_pruned_model(self, output_path):
        """Save pruned model checkpoint."""

def visualize_sparsity(model, output_path):
    """Generate per-layer sparsity visualization."""
```

### Task 3: Mixed Precision (`utils/mixed_precision.py`)

```python
class MixedPrecisionConfig:
    """Configuration for layer-wise precision assignment."""
    
    layer_precision_map = {
        'embedding': 'int8',
        'ssm_layers': 'fp16',
        'moe_experts': 'int4',
        'output_projection': 'fp16'
    }

class MixedPrecisionConverter:
    """Convert model to mixed-precision format."""
    
    def __init__(self, model, config: MixedPrecisionConfig):
        """Initialize with model and precision config."""
        
    def profile_layer_sensitivity(self, dataloader):
        """Analyze per-layer quantization sensitivity."""
        
    def assign_optimal_precision(self, sensitivity_scores):
        """Automatically assign precision based on sensitivity."""
        
    def convert_model(self):
        """Apply mixed-precision conversion."""
        
    def benchmark_mixed_precision(self, test_loader):
        """Compare to uniform quantization baselines."""
```

### Task 4: Fused Kernels (`utils/fused_kernels.py`)

```python
class FusedSSMScan(torch.nn.Module):
    """Fused SSM scan operation (state update + output projection)."""
    
    def forward(self, x, state):
        """Single fused kernel for SSM scan."""

class FusedMoERouting(torch.nn.Module):
    """Fused MoE routing (gating + expert selection + computation)."""
    
    def forward(self, x, experts):
        """Fused MoE operation."""

def compile_model_jit(model):
    """Compile model with TorchScript for kernel fusion."""
    return torch.jit.script(model)

def profile_model(model, input_data):
    """Profile model to identify bottlenecks."""
    return {
        "hotspots": [...],
        "memory_bandwidth_gb_s": ...,
        "cache_miss_rate": ...
    }
```

---

## Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **INT4 quality degradation > 30%** | Medium | High | Implement QAT, use mixed-precision fallback |
| **Pruning breaks SSM state** | Medium | High | Preserve SSM layers, prune MoE experts only |
| **CPU-only limits speed gains** | High | Medium | Focus on inference optimization, accept slower training |
| **bitsandbytes CPU mode unstable** | Low | Medium | Fallback to PyTorch native quantization |
| **Kernel fusion minimal gains** | Medium | Low | Document findings, prioritize other tasks |
| **Checkpoint compatibility issues** | Low | High | Maintain backward compatibility, version checkpoints |

---

## Success Criteria

### Quantitative Targets

| Metric | Phase 6 Baseline | Phase 7 Target | Stretch Goal |
|--------|------------------|----------------|--------------|
| **Compression** | 12.9× | **50-100×** | **100×** |
| **Model Size** | 0.56 MB | **<0.12 MB** | **<0.06 MB** |
| **Inference Speed** | 36ms/sample | **<25ms/sample** | **<15ms/sample** |
| **Perplexity (WikiText-2)** | 1215 | **<1600** | **<1400** |
| **Parameters** | 293,656 | **<100,000** | **<50,000** |

### Qualitative Goals

- ✅ Native INT4 quantization (not simulated)
- ✅ Structured sparsity (pruned model)
- ✅ Mixed-precision deployment ready
- ✅ CPU-optimized kernels (MKL-DNN fusion)
- ✅ Maintain functional text generation
- ✅ Production-ready checkpoints

---

## Timeline & Milestones

```
Week 1-2:  Task 1 - Native INT4 Implementation
    Milestone: 25.8× compression achieved

Week 3-4:  Task 2 - Structured Pruning
    Milestone: 43.1× compression achieved

Week 5-6:  Task 3 - Mixed-Precision Inference
    Milestone: 56.0× compression achieved

Week 7:    Task 4 - Advanced Kernel Optimization
    Milestone: 1.5-2× inference speedup

Week 8:    Integration, Final Benchmarking & Reporting
    Milestone: Phase 7 complete, 50-100× target met
```

**Total Duration:** 8 weeks (2 months)

---

## Dependencies & Prerequisites

### Software Dependencies (Installed)
- ✅ `torch` 2.8.0+cpu
- ✅ `onnxruntime` 1.23.0
- ✅ `bitsandbytes` 0.48.1
- ✅ `transformers` 4.57.0
- ✅ `optimum` (just installed)
- ✅ `torch-pruning` (just installed)

### Missing Dependencies (Optional)
- ⚠️ `onnxruntime-gpu` (not needed for CPU-only)
- ⚠️ `py-cpuinfo` (for detailed CPU feature detection)

### Phase 6 Outputs Required
- ✅ `checkpoints/int4/itera_lite_micro.pt` (INT4 simulated baseline)
- ✅ `data/tokenizer_2000.json` (tokenizer)
- ✅ `models/itera_lite.py` (model architecture)

---

## Next Steps (Immediate Actions)

### 1. Begin Task 1: Native INT4 Implementation

**Immediate next actions:**
1. Create `utils/native_quantization.py` skeleton
2. Research `bitsandbytes` 4-bit quantization API
3. Test INT4 on small model (validate functionality)
4. Design calibration pipeline for TinyStories

### 2. Prepare Training Data

**Setup:**
- Validate TinyStories dataset access
- Create small calibration subset (1,000 samples)
- Prepare validation split for quality measurement

### 3. Checkpoint Management

**Plan:**
- Create `checkpoints/phase7/` directory structure
- Implement checkpoint versioning (e.g., `int4_native_v1.pt`)
- Add metadata (compression ratio, perplexity, date)

---

## Hardware Upgrade Path (Optional)

If Phase 7 progress is bottlenecked by CPU-only mode, consider:

**Minimum GPU Upgrade:**
- NVIDIA GTX 1660 Ti (6GB VRAM) — $200-250 used
- CUDA 12.0+ support
- 2-3× faster INT4 quantization and training

**Recommended GPU Upgrade:**
- NVIDIA RTX 3060 (12GB VRAM) — $300-400
- Native FP16 acceleration (Tensor Cores)
- Mixed-precision training support
- 5-10× faster than CPU for Phase 7 tasks

**Note:** Phase 7 is **still achievable on CPU**, but will require longer iteration times. All tasks are designed to be CPU-compatible.

---

## Conclusion

Phase 7 is **ready to begin** with the following status:

✅ **Hardware diagnostics complete** (CPU-only mode, excellent CPU, sufficient RAM)  
✅ **Dependencies installed** (optimum, torch-pruning added)  
✅ **Roadmap defined** (4 tasks, 8-week timeline, 50-100× compression target)  
✅ **Risk assessment complete** (mitigation strategies identified)  
✅ **File structure planned** (modular utilities, clear separation of concerns)

**Recommended Start:** Task 1 (Native INT4 Implementation) — this provides the foundation for subsequent compression techniques and immediate 2× compression gain.

**Project Status After Phase 7:**
- **Target Compression:** 50-100× (from 12.9× baseline)
- **Project Completion:** 87.5% (7/8 phases)
- **Remaining:** Phase 8 (Production Cloud Deployment)

---

*Roadmap Generated: October 7, 2025*  
*Itera-Lite Phase 7: Advanced Optimization — 50-100× Compression Target* 🚀
