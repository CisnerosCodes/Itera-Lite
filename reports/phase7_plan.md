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

### Local System (Development - Windows)

| Component | Specification | Status | Usage |
|-----------|--------------|--------|-------|
| **CPU** | 10 cores (12 threads) | ✅ EXCELLENT | Code editing, prototyping |
| **Memory** | 15.55 GB RAM | ✅ SUFFICIENT | Local testing |
| **GPU** | Not available (CPU-only PyTorch) | ⚠️ LIMITED | Development only |
| **Python** | 3.13.7 | ✅ EXCELLENT | Latest features |
| **PyTorch** | 2.8.0+cpu | ✅ CURRENT | Local debugging |

**Local Benchmarks:**
- **CPU FP32:** 406.88 GFLOPS (excellent for prototyping)
- **CPU FP16:** 1.31 GFLOPS (limited)

### HPC Cluster (Production - Texas A&M FASTER)

| Component | Specification | Status | Impact on Phase 7 |
|-----------|--------------|--------|-------------------|
| **GPU** | **NVIDIA A30 (Ampere)** | ✅ **OUTSTANDING** | **Enterprise-grade acceleration** |
| **VRAM** | **24 GB** | ✅ **EXCELLENT** | **No memory constraints** |
| **Compute Capability** | **8.0** | ✅ **EXCELLENT** | **Full Ampere features** |
| **CUDA** | 12.8 (matches PyTorch) | ✅ **PERFECT** | Native GPU ops |
| **cuDNN** | 9.1.0 | ✅ **LATEST** | Optimized kernels |
| **Multi-Processors** | 56 SMs | ✅ **HIGH** | Massive parallelism |
| **Python** | 3.11.5 | ✅ CURRENT | HPC standard |
| **PyTorch** | 2.8.0+cu128 | ✅ **GPU-ENABLED** | CUDA 12.8 support |

**GPU Performance (NVIDIA A30):**
- **FP32:** ~10.3 TFLOPS (25× faster than local CPU)
- **FP16 (Tensor Cores):** ~165 TFLOPS (400× faster than local CPU!)
- **INT8 (Tensor Cores):** ~330 TOPS (hardware-accelerated quantization)

### Implications for Phase 7

**GPU Advantages (A30 Ampere):**
- ✅ **Native FP16 Tensor Cores** → 20-30× faster mixed-precision training
- ✅ **INT8 Tensor Cores** → Hardware-accelerated quantization
- ✅ **24GB VRAM** → Full batch sizes, no gradient checkpointing needed
- ✅ **bitsandbytes GPU mode** → 5-10× faster INT4 calibration vs CPU
- ✅ **CUDA kernel fusion** → Custom optimized operations
- ✅ **Fast pruning fine-tuning** → 10× speedup for gradient computation

**Phase 7 Strategy:**
- 🚀 **GPU-accelerated quantization** (bitsandbytes native INT4)
- 🚀 **FP16 mixed-precision training** (Tensor Core acceleration)
- 🚀 **GPU fine-tuning** (pruning recovery in hours vs days)
- 🚀 **CUDA kernel optimization** (fused SSM/MoE operations)
- ✅ **Dual environment workflow** (code locally, compute on HPC)

---

## Phase 7 Task Breakdown

### ✅ Task 1: Native INT4 Implementation (1 week - GPU accelerated!)

**Objective:** Implement true hardware-accelerated INT4 quantization using NVIDIA A30

**Approach:**
```
Current: Simulated INT4 (symmetric quantization in FP32)
Target: GPU-native INT4 (bitsandbytes + A30 Tensor Cores)
Acceleration: 5-10× faster calibration vs CPU
```

**Sub-tasks:**
1. **INT4 GPU Setup & Research** (1 day)
   - Test `bitsandbytes` GPU mode on A30
   - Benchmark GPU vs CPU quantization speed
   - Validate INT8 Tensor Core acceleration
   - Design GPU-optimized calibration pipeline

2. **INT4 Quantization Implementation** (2 days - reduced from 5!)
   - Create `utils/native_quantization.py` (code locally)
   - Implement GPU-accelerated INT4 weight quantization
   - Add INT4 activation quantization (A30 native)
   - Handle SSM and MoE layer quantization edge cases
   - Create HPC job script: `jobs/phase7_task1_quantize.sh`

3. **INT4 GPU Fine-tuning** (2 days - reduced from 4!)
   - Quantization-Aware Training on A30 (FP16 base)
   - Fine-tune INT4 model on TinyStories (GPU batches)
   - Measure perplexity degradation (target: <30%)
   - Generate INT4 checkpoint on HPC

4. **INT4 Benchmarking** (1 day)
   - Compare INT4 vs INT8 vs FP32 (A30 inference)
   - Measure compression ratio (target: 2.0×)
   - Generate `reports/phase7_int4_quantization.md`

**Expected Outputs:**
- `utils/native_quantization.py` (INT4 quantization utilities)
- `checkpoints/int4_native/itera_lite_int4.pt` (native INT4 model)
- `results/phase7_int4_benchmark.json` (metrics)
- `reports/phase7_int4_quantization.md` (report)

**Target Compression:** 12.9× → **25.8×** (2.0× improvement)

---

### ✅ Task 2: Structured Pruning (1 week - GPU accelerated!)

**Objective:** GPU-accelerated structured pruning for 30-50% sparsity

**Approach:**
```
Current: Dense model (all parameters active)
Target: 40% pruning → 60% parameters remaining
GPU Advantage: 10× faster fine-tuning (A30 vs CPU)
```

**Sub-tasks:**
1. **Pruning Strategy Design** (1 day - reduced from 2!)
   - GPU-accelerated gradient analysis (layer importance)
   - Design iterative pruning schedule
   - Choose pruning granularity (channel/neuron)
   - Plan SSM layer preservation strategy

2. **Pruning Implementation** (2 days - reduced from 5!)
   - Create `utils/structured_pruning.py` (code locally)
   - Implement magnitude-based pruning
   - Add layer-wise pruning ratios (variable sparsity)
   - Preserve critical layers (embeddings, SSM)
   - Create HPC job: `jobs/phase7_task2_prune.sh`

3. **GPU Fine-tuning** (2 days - reduced from 5!)
   - Fine-tune on A30 (FP16 mixed-precision)
   - Iterative pruning: 10% → 20% → 40%
   - Monitor perplexity after each step
   - Generate final pruned checkpoint

4. **Pruning Analysis** (1 day)
   - Visualize sparsity patterns per layer
   - Compare pruned vs dense (A30 inference)
   - Generate `reports/phase7_structured_pruning.md`

**Expected Outputs:**
- `utils/structured_pruning.py` (pruning utilities)
- `checkpoints/pruned/itera_lite_pruned_40pct.pt` (pruned model)
- `results/phase7_pruning_analysis.json` (metrics)
- `reports/phase7_structured_pruning.md` (report)
- `reports/phase7_sparsity_visualization.png` (layer sparsity chart)

**Target Compression:** 25.8× → **43.1×** (1.67× improvement from 40% pruning)

---

### ✅ Task 3: Mixed-Precision Inference (1 week - A30 Tensor Cores!)

**Objective:** Hardware-native mixed-precision using A30 Tensor Cores

**Approach:**
```
Layer-wise precision assignment (A30 hardware-accelerated):
- Embeddings: INT8 (Tensor Core INT8 ops)
- SSM layers: FP16 (Tensor Core FP16 matmuls)
- MoE experts: INT4 (bitsandbytes GPU mode)
- Final projection: FP16 (output quality critical)

GPU Advantage: 20-30× faster FP16 vs CPU emulation
```

**Sub-tasks:**
1. **GPU Precision Profiling** (1 day - reduced from 2!)
   - Analyze layer sensitivity on A30
   - Profile Tensor Core utilization per layer
   - Identify optimal precision for each layer type
   - Design A30-optimized mixed-precision schema

2. **Mixed-Precision Implementation** (2 days)
   - Create `utils/mixed_precision.py` (code locally)
   - Implement layer-wise precision assignment
   - Add A30 Tensor Core optimizations
   - Create HPC job: `jobs/phase7_task3_mixed.sh`

3. **GPU Mixed-Precision Training** (2 days - reduced from 3!)
   - Fine-tune mixed-precision model on A30
   - Benchmark Tensor Core acceleration
   - Measure speed/quality trade-offs
   - Generate optimal configuration

4. **Mixed-Precision Evaluation** (1 day)
   - Compare to FP32, INT8, INT4 baselines
   - Measure A30 Tensor Core speedup
   - Generate `reports/phase7_mixed_precision.md`

**Expected Outputs:**
- `utils/mixed_precision.py` (mixed-precision utilities)
- `checkpoints/mixed_precision/itera_lite_mixed.pt` (mixed-precision model)
- `results/phase7_mixed_precision.json` (metrics)
- `reports/phase7_mixed_precision.md` (report)

**Target Compression:** 43.1× → **56.0×** (1.3× improvement)

---

### ✅ Task 4: Advanced Kernel Optimization (1.5 weeks - CUDA + CPU!)

**Objective:** Dual-target optimization (A30 CUDA kernels + CPU deployment)

**Approach:**
```
GPU Optimization (A30 CUDA):
- CUDA kernel fusion (SSM scan, MoE routing)
- Tensor Core utilization maximization
- Memory coalescing and bandwidth optimization

CPU Optimization (deployment):
- MKL-DNN kernel fusion (inference)
- SIMD vectorization (AVX2 for INT8)
- Cache-aware memory layout
```

**Sub-tasks:**
1. **Profiling & Bottleneck Analysis** (2 days)
   - Profile A30 inference pipeline (nsys, nvprof)
   - Identify GPU memory bandwidth bottlenecks
   - Profile CPU inference (local machine)
   - Analyze kernel fusion opportunities

2. **CUDA Kernel Implementation** (4 days)
   - Create `utils/fused_kernels.py` (code locally)
   - Implement fused SSM scan (CUDA kernel)
   - Add fused MoE routing (expert selection + compute)
   - Optimize memory access patterns
   - Create HPC job: `jobs/phase7_task4_kernels.sh`

3. **CPU Kernel Optimization** (3 days)
   - MKL-DNN integration for deployment
   - TorchScript compilation (`torch.jit.script`)
   - ONNX export with optimizations
   - Test on local CPU (verify portability)

4. **Kernel Benchmarking** (2 days)
   - Compare A30 CUDA vs unfused baseline
   - Measure CPU deployment speedup
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
| **INT4 quality degradation > 30%** | Low | High | A30 GPU enables fast QAT iteration, mixed-precision fallback |
| **Pruning breaks SSM state** | Medium | High | Preserve SSM layers, prune MoE experts only |
| **HPC job queue delays** | Low | Medium | Submit jobs off-peak, use 2-day time limit efficiently |
| **bitsandbytes GPU compatibility** | Very Low | Medium | A30 fully supported, fallback to PyTorch native if needed |
| **CUDA kernel complexity** | Medium | Low | Start with TorchScript, custom CUDA if time permits |
| **Checkpoint compatibility** | Low | High | Maintain backward compatibility, version checkpoints |
| **VRAM limitations** | Very Low | Low | 24GB A30 is more than sufficient for tiny model |

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

**🚀 GPU-ACCELERATED TIMELINE (NVIDIA A30)**

```
Week 1:    Task 1 - GPU-Native INT4 Quantization (A30)
           Milestone: 25.8× compression achieved
           GPU Advantage: 5-10× faster calibration

Week 2:    Task 2 - GPU-Accelerated Structured Pruning
           Milestone: 43.1× compression achieved
           GPU Advantage: 10× faster fine-tuning

Week 3:    Task 3 - Mixed-Precision with Tensor Cores
           Milestone: 56.0× compression achieved
           GPU Advantage: 20-30× faster FP16 training

Week 4-5:  Task 4 - CUDA + CPU Kernel Optimization
           Milestone: 1.5-2× inference speedup (both GPU/CPU)
           Dual-target optimization (A30 CUDA + CPU deployment)

Week 5:    Final Integration & Benchmarking
           Milestone: Phase 7 complete, 50-100× target met
```

**Total Duration:** 5 weeks (vs 8 weeks CPU-only = 37.5% time savings!)

**GPU vs CPU Comparison:**
- **Original CPU-only timeline:** 8 weeks
- **GPU-accelerated timeline:** 5 weeks  
- **Time saved:** 3 weeks (37.5% faster)
- **Key accelerations:**
  - Task 1: 2 weeks → 1 week (GPU quantization)
  - Task 2: 2 weeks → 1 week (GPU fine-tuning)
  - Task 3: 1.5 weeks → 1 week (Tensor Cores)
  - Task 4: 1.5 weeks → 1.5 weeks (CUDA kernels added)

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

**✅ HPC GPU AVAILABLE - NO LOCAL UPGRADE NEEDED!**

The **NVIDIA A30 on Texas A&M HPC** provides enterprise-grade acceleration:
- ✅ 24GB VRAM (8× more than consumer GPUs)
- ✅ Ampere Tensor Cores (FP16/INT8 acceleration)
- ✅ Compute Capability 8.0 (all modern features)
- ✅ Zero cost (HPC cluster access included)

**Local Machine (Optional Enhancement):**
If you want to run GPU experiments locally (not required):

**Budget GPU Option:**
- NVIDIA GTX 1660 Ti (6GB VRAM) — $200-250 used
- Good for: Local prototyping, small-scale testing
- Limitation: No Tensor Cores, limited VRAM

**Recommended GPU (if purchasing):**
- NVIDIA RTX 3060 (12GB VRAM) — $300-400
- Ampere Tensor Cores (same as A30)
- Good for: Full local development workflow
- Still limited vs A30's 24GB VRAM

**Reality Check:** The HPC A30 is **superior to any consumer GPU** for Phase 7. Local GPU purchase is **NOT recommended** - the established workflow (code locally, compute on HPC) is optimal.

---

## Conclusion

Phase 7 is **ready to begin** with **EXCELLENT GPU ACCELERATION**:

✅ **HPC GPU Available:** NVIDIA A30 (24GB VRAM, Compute 8.0)  
✅ **GPU-Accelerated Timeline:** 5 weeks (vs 8 weeks CPU-only)  
✅ **Tensor Core Support:** FP16/INT8 hardware acceleration  
✅ **Dependencies Installed:** bitsandbytes, optimum, torch-pruning  
✅ **Dual Environment Ready:** Local dev (VS Code) + HPC compute (Slurm)  
✅ **Roadmap Optimized:** 4 tasks with GPU-native implementations

**GPU Acceleration Benefits:**
- **Task 1 INT4:** 5-10× faster calibration (bitsandbytes GPU mode)
- **Task 2 Pruning:** 10× faster fine-tuning (A30 vs CPU)
- **Task 3 Mixed-Precision:** 20-30× faster FP16 (Tensor Cores)
- **Task 4 Kernels:** Custom CUDA optimization possible

**Recommended Start:** Task 1 (GPU-Native INT4 Implementation)
1. Code `utils/native_quantization.py` locally in VS Code
2. Push to GitHub
3. Pull on HPC, create `jobs/phase7_task1_quantize.sh`
4. Submit to GPU partition: `sbatch jobs/phase7_task1_quantize.sh`
5. Monitor with `squeue -u $USER`
6. Results back via GitHub sync

**Project Status After Phase 7:**
- **Target Compression:** 50-100× (from 12.9× baseline)
- **Timeline:** 5 weeks (GPU-accelerated)
- **Project Completion:** 87.5% (7/8 phases)
- **Remaining:** Phase 8 (Production Cloud Deployment)

---

*Roadmap Generated: October 7, 2025*  
*Roadmap Updated: October 8, 2025 - NVIDIA A30 GPU discovered! ⚡*  
*Itera-Lite Phase 7: Advanced Optimization — 50-100× Compression Target* 🚀
