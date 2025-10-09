# Phase 7 Task 1: INT4 Quantization - Execution Guide

**Date:** October 9, 2025  
**Status:** Ready to Execute  
**Expected Runtime:** 1-2 hours on NVIDIA A30

---

## What Was Created

### 1. `utils/native_quantization.py` (560 lines)
**GPU-optimized INT4 quantization utilities:**
- `NativeINT4Quantizer` class with bitsandbytes GPU support
- Calibration on representative data
- QAT (Quantization-Aware Training) optional
- Benchmark comparison (FP32 vs INT4)

**Key Features:**
- ✅ NVIDIA A30 Ampere Tensor Core support
- ✅ NormalFloat4 (NF4) quantization (recommended)
- ✅ Double quantization for better compression
- ✅ FP16 compute dtype for speed

### 2. `phase7_quantize.py` (350 lines)
**Main orchestration script:**
- Loads Itera-Lite checkpoint
- Runs calibration (1000 samples default)
- Applies INT4 quantization
- Optional QAT fine-tuning
- Benchmarks results
- Exports INT4 checkpoint

### 3. `jobs/phase7_task1_quantize.sh`
**Slurm GPU job script:**
- Requests 1× A30 GPU (24GB VRAM)
- 8 CPU cores, 32GB RAM
- 4-hour time limit (plenty of buffer)
- Automatic environment setup
- Runs quantization and benchmarking
- Saves results to checkpoints/int4_native/

---

## HPC Execution Instructions

### Step 1: Pull Latest Code

On HPC, run:
```bash
cd Itera-Lite
source .venv/bin/activate
git pull origin main
```

**Expected output:**
```
Updating eb5a0a0..1c78e13
Fast-forward
 jobs/README.md                    | 150 +++++++++++++++++++
 jobs/phase7_task1_quantize.sh     | 130 ++++++++++++++++
 phase7_quantize.py                | 350 +++++++++++++++++++++++++++++++++++++++++
 utils/native_quantization.py      | 560 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 4 files changed, 1080 insertions(+)
```

### Step 2: Verify Files

```bash
ls -lh jobs/phase7_task1_quantize.sh
ls -lh phase7_quantize.py
ls -lh utils/native_quantization.py
```

All three should exist.

### Step 3: Create Logs Directory

```bash
mkdir -p logs
```

### Step 4: Submit Job

```bash
sbatch jobs/phase7_task1_quantize.sh
```

**Expected output:**
```
Submitted batch job 123456
```

### Step 5: Monitor Job

```bash
# Check job status
squeue -u $USER

# Watch in real-time (updates every 2 seconds)
watch -n 2 'squeue -u $USER'

# Check GPU partition availability
sinfo -p gpu
```

**Job states:**
- `PD` = Pending (waiting in queue)
- `R` = Running (executing on GPU node)
- `CG` = Completing (finishing up)

### Step 6: View Progress (while running)

```bash
# Watch output in real-time
tail -f logs/phase7_task1_int4_*.out

# Press Ctrl+C to stop watching
```

### Step 7: Check Results (after completion)

```bash
# View full output
cat logs/phase7_task1_int4_*.out

# Check if quantized model exists
ls -lh checkpoints/int4_native/

# View benchmark results
cat checkpoints/int4_native/phase7_int4_benchmark.json | python -m json.tool
```

---

## Expected Results

### Quantized Model
**Location:** `checkpoints/int4_native/itera_lite_int4_nf4.pt`
**Expected Size:** ~0.15-0.3 MB (depends on original model)

### Benchmark Metrics
**Target compression:** 2.0× over INT8 = **25.8× cumulative**

Example output:
```json
{
  "comparison": {
    "size_reduction": 2.0,
    "speedup": 1.5,
    "perplexity_degradation": 15.0
  }
}
```

### Success Criteria
- ✅ Job completes without errors
- ✅ INT4 checkpoint created
- ✅ Compression ratio ~2.0×
- ✅ Perplexity degradation <30%
- ✅ Inference speedup >1.0×

---

## Troubleshooting

### Job Doesn't Start
```bash
# Check queue
squeue -u $USER

# If pending long time, check partition
sinfo -p gpu

# Solution: Wait for idle GPU node (usually <30 min)
```

### "No checkpoint found" Error
```bash
# Check which checkpoints exist
ls checkpoints/*.pt

# If missing, you may need to use a different checkpoint
# Edit jobs/phase7_task1_quantize.sh line 53-60
```

### CUDA Out of Memory
```bash
# Reduce batch size (edit phase7_quantize.py)
# Or request more VRAM (not needed - A30 has 24GB)
```

### bitsandbytes Import Error
```bash
# Check if installed
pip list | grep bitsandbytes

# If missing:
pip install bitsandbytes
```

---

## After Job Completes

### 1. Commit Results to GitHub

```bash
# On HPC
git add checkpoints/int4_native/
git add logs/phase7_task1_int4_*.out
git commit -m "Phase 7 Task 1 results: INT4 quantization on A30"
git push origin main
```

### 2. Pull Results Locally

```bash
# On local machine (VS Code)
git pull origin main
```

### 3. Analyze Results

Open in VS Code:
- `checkpoints/int4_native/phase7_int4_benchmark.json`
- `logs/phase7_task1_int4_*.out`

Generate report:
```bash
# Will create this in Task 1 completion
python generate_phase7_task1_report.py
```

---

## Timeline

- **Submit job:** Now
- **Queue time:** 0-30 minutes (depends on GPU availability)
- **Execution time:** 1-2 hours
  - Calibration: ~10-15 minutes
  - Quantization: ~5-10 minutes
  - Benchmarking: ~30-60 minutes
- **Total:** ~2-3 hours from submission to results

---

## Next Steps After Task 1

1. ✅ **Task 1 complete:** INT4 quantization (25.8× compression)
2. ⏭️ **Task 2:** Structured Pruning (week 2)
3. ⏭️ **Task 3:** Mixed-Precision (week 3)
4. ⏭️ **Task 4:** Kernel Optimization (weeks 4-5)

**Phase 7 Progress:** Task 1/4 (Week 1/5)

---

## Quick Reference Commands

```bash
# Submit
sbatch jobs/phase7_task1_quantize.sh

# Monitor
squeue -u $USER
tail -f logs/phase7_task1_int4_*.out

# Results
cat checkpoints/int4_native/phase7_int4_benchmark.json

# Sync
git add checkpoints/ logs/
git commit -m "Task 1 results"
git push origin main
```

---

*Ready to execute on NVIDIA A30!* 🚀
