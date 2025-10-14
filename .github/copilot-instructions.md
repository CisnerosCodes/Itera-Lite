# Itera-Lite AI Agent Instructions

## Project Overview

Itera-Lite is a **State-Space Model (SSM) + Mixture-of-Experts (MoE) hybrid** language model research project focused on compression and efficiency. The project completed **8 phases** achieving **2.0× compression with 1.24× speedup** through FP16 optimization (Phase 8) and **2.27× compression** through mixed-precision (Phase 7) while maintaining quality.

**Critical Architectural Facts:**
- SSM layers (39% params) use **state-space computation** with recurrent dependencies—NOT attention
- MoE layers (selective) are structurally dependent on checkpoint format
- Pruning SSM layers breaks recurrent state → **0% viability discovered in Phase 7**
- **FP16 simple compression (Phase 8) is recommended for production** - easier than mixed-precision
- Mixed-precision (INT8/FP16) is the advanced alternative for power users

## Project Structure & Key Files

```
models/
├── itera_lite.py          # Main SSM+MoE hybrid (IteraLiteModel class)
├── ssm.py                 # State-space block (SSMBlock - recurrent!)
├── moe.py                 # Sparse expert routing (MoELayer)
└── config.py              # IteraLiteConfig, TransformerConfig

checkpoints/
├── itera_lite_quality_best.pt       # Phase 8 quality model (10.24 MB, FP32)
├── itera_lite_tiny_best.pt          # Baseline FP32 (7.20 MB)
├── phase8_compressed/               # Best: FP16 2.0× (1.69 MB weights) 🏆
├── mixed_precision/                 # Advanced: 2.27× compression (2.95 MB)
└── int4/                            # INT4 quantized (1.61 MB, GPU-only)

utils/
├── mixed_precision.py     # Layer-wise INT8/FP16 conversion (657 lines)
├── structured_pruning.py  # Pruning utils (SSM-incompatible - use with caution)
├── training.py            # Trainer class, AdamW + cosine scheduler
└── benchmark.py           # FLOPs, throughput, perplexity metrics

phase7_*.py                # Phase 7 research scripts for compression experiments
phase8_*.py                # Phase 8 quality training and FP16 compression
run_inference.py           # Quick demo: generate text with trained model
validate_local.py          # CPU validation for compressed checkpoints
train.py                   # Main training script
```

## Critical Developer Workflows

### Training a Model
```bash
# PowerShell (Windows environment)
python train.py --model_type itera --config_size tiny --num_epochs 5 --batch_size 8
```
- Saves checkpoints to `checkpoints/itera_lite_tiny_{best|final}.pt`
- Tokenizer saved to `data/tokenizer_tiny.json`
- Metrics logged to `results/itera_lite_tiny_metrics.csv`

### Running Inference
```python
python run_inference.py --prompt "Once upon a time" --max-length 100 --temperature 1.0
```
- Default checkpoint: `checkpoints/itera_lite_tiny_best.pt`
- CPU-optimized (3,308 tok/sec baseline, 2,740 tok/sec mixed-precision)

### Testing Compression (Phase 7 Pattern)
```bash
# Mixed-precision (recommended for SSMs)
python phase7_mixed_precision.py --checkpoint checkpoints/itera_lite_tiny_best.pt

# INT4 quantization (requires GPU)
python phase7_quantize.py --checkpoint checkpoints/itera_lite_tiny_best.pt

# Pruning (AVOID for SSMs - breaks recurrent state)
# Only use for Transformer baselines
```

### HPC Job Submission (Texas A&M FASTER cluster)
```bash
source .venv/bin/activate
git pull origin main
sbatch jobs/phase7_task1_quantize.sh  # Submit to Slurm
squeue -u $USER                        # Monitor status
cat logs/phase7_task1_int4_*.out       # View results
```

## Checkpoint Loading: Critical Pattern

**Always inspect checkpoint structure before writing loading code.** Checkpoints vary in format:

```python
# CORRECT pattern (used in all phase7_*.py scripts)
def load_checkpoint_with_inference(checkpoint_path: str, device: str = 'cuda'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 1. Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 2. Infer config (checkpoints rarely store full config)
    vocab_size = state_dict['embedding.weight'].shape[0]  # NOT 'embeddings.token_embeddings.weight'
    hidden_size = state_dict['embedding.weight'].shape[1]
    max_seq_length = state_dict['position_embedding.weight'].shape[0]
    num_layers = sum(1 for k in state_dict.keys() if '.ssm.in_proj.weight' in k)
    
    # 3. Handle old checkpoint format (.moe.layer. vs .moe.moe.)
    new_state_dict = {}
    for key, value in state_dict.items():
        if '.moe.layer.' in key:
            new_state_dict[key.replace('.moe.layer.', '.moe.moe.')] = value
        else:
            new_state_dict[key] = value
    
    # 4. Load with strict=False (handles missing MoE layers)
    model.load_state_dict(new_state_dict, strict=False)
```

**Common Pitfall:** Checkpoint key names differ from training code. Always print `list(state_dict.keys())[:10]` first.

## Compression Decision Matrix

| Target Hardware | Model Size | Technique | Expected Ratio | Reference Script |
|-----------------|------------|-----------|----------------|------------------|
| **GPU (CUDA)** | <10M params | Mixed-Precision | 2.27× | `phase7_mixed_precision.py` |
| GPU | >10M | INT4 Quantization | 4.47× | `phase7_quantize.py` |
| **CPU (x86)** | Any | Baseline FP32 | 1.0× | `run_inference.py` |
| Edge/Mobile | Any | INT8 Dynamic | TBD | (not implemented) |

**SSM-Specific:** Pruning is **architecturally infeasible** due to state dependencies. Use mixed-precision instead.

## Project-Specific Conventions

### Precision Mapping (Mixed-Precision)
```python
# Conservative mapping (proven in Phase 7 Task 3)
PRECISION_MAP = {
    'embedding.weight': 'int8',          # 59% of params, less sensitive
    'position_embedding.weight': 'int8',
    'layers.*.ssm.*': 'fp16',            # 39% of params, CRITICAL for quality
    'layers.*.moe.*': 'fp16',            # (if present in checkpoint)
    'norm_f.weight': 'fp16',             # Stability
    'lm_head.weight': 'int8',            # Tied with embeddings
}
```
**Validation:** Check that `>0` layers match patterns, monitor "unmatched params" percentage (<20% acceptable).

### Model Configuration Sizes
```python
# Tiny (used in all Phase 7 experiments)
vocab_size=8000, hidden_size=128, num_layers=4, ssm_state_size=8, num_experts=4
# Small (for future scaling)
vocab_size=8000, hidden_size=256, num_layers=6, ssm_state_size=16, num_experts=8
```

### Testing Pattern
```bash
python test_models.py  # Tests all model variants (SSM, Transformer, tiny, small)
```
No pytest suite—tests are standalone scripts (`test_*.py`, `validate_*.py`).

## Integration Points & Dependencies

### Core Dependencies (Windows PowerShell)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib seaborn  # Visualization
pip install bitsandbytes  # INT4 quantization (GPU-only, optional)
```

### Deployment (Docker + FastAPI)
```bash
docker-compose up --build  # Starts inference API on port 8000
```
- Inference server: `deployment/inference_api.py` (FastAPI with adaptive learning)
- Health check: `http://localhost:8000/health`

### External Data
- Dataset: TinyStories (loaded via `utils/dataset_loader.py`)
- Tokenizer: Character-level (1000-8000 vocab, saved as `data/tokenizer_*.json`)

## Architecture-Specific Gotchas

1. **SSM ≠ Transformer:** SSM layers use state-space computation (O(n) complexity) with learned matrices `A_log`, `B`, `C`, `D`. Do NOT apply Transformer-specific optimizations (attention masks, positional encoding patterns).

2. **MoE Layer Sparsity:** Only layers in `config.moe_layers` (default: `[1, 3]` for 4-layer model) have MoE. Other layers use standard FFN. Check checkpoint for `.moe.` keys before assuming MoE exists.

3. **CPU vs GPU Compression:** INT4/INT8/FP16 require GPU hardware acceleration. On CPU, compressed models may be **slower** than FP32 due to conversion overhead. Baseline FP32 achieves 3,308 tok/sec on 12-core CPU.

4. **Checkpoint Compatibility:** Old checkpoints (Phase 1-2) use `.moe.layer.experts`, new ones (Phase 7+) use `.moe.moe.experts`. Always convert keys during loading (see pattern above).

## Cross-File Context for Common Tasks

### Adding a New Compression Technique
1. Create `utils/new_technique.py` with `Converter` class (see `mixed_precision.py` pattern)
2. Add `phase7_new_technique.py` script reusing `load_checkpoint_with_inference()` from `phase7_mixed_precision.py`
3. Create HPC job script in `jobs/phase7_taskN_new_technique.sh` (copy `phase7_task1_quantize.sh` template)
4. Document in `reports/phase7_taskN_new_technique.md` (track debugging iterations like Task 1-3)

### Modifying Model Architecture
- **SSM changes:** Edit `models/ssm.py` → Retrain (affects 39% of params, quality-critical)
- **MoE changes:** Edit `models/moe.py` → Update `config.moe_layers` in `models/config.py`
- **Embeddings:** Edit `models/itera_lite.py` → Retrain tokenizer (`utils/data.py`)

### Benchmarking New Model
```python
from utils.benchmark import ModelBenchmark, compare_models
benchmark = ModelBenchmark(model, config, device='cpu')
results = benchmark.run_full_benchmark(dataset, max_batches=100)
# Outputs: params, FLOPs, throughput (tok/sec), perplexity, memory (MB)
```

## Research Context (Phase 7 Learnings)

**Why these instructions exist:** Phase 7 took 19 HPC job iterations across 3 tasks to discover:
- SSM pruning is infeasible (state dependencies)
- Mixed-precision beats INT4 for SSMs (2.27× vs 1.42×)
- Checkpoint format mismatches cost ~5 debugging cycles per task
- Precision map patterns must **exactly match** checkpoint keys (e.g., `'embedding.weight'` not `'embeddings.token_embeddings.weight'`)

**When uncertain:** Reference `phase7_mixed_precision.py` (Task 3) for proven checkpoint loading, `utils/mixed_precision.py` for layer-wise conversion patterns, and `reports/PHASE7_COMPLETION_REPORT.md` for debugging strategies.
