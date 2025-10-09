# HPC Job Scripts

This directory contains Slurm job scripts for Phase 7 tasks on the Texas A&M FASTER cluster.

## Available Jobs

### Task 1: GPU-Native INT4 Quantization
**File:** `phase7_task1_quantize.sh`
**Purpose:** Quantize Itera-Lite model to INT4 using bitsandbytes on NVIDIA A30 GPU
**Runtime:** ~1-2 hours
**Resources:** 1× A30 GPU, 8 CPUs, 32GB RAM

**Submit:**
```bash
sbatch jobs/phase7_task1_quantize.sh
```

**Monitor:**
```bash
squeue -u $USER
```

**View Results:**
```bash
cat logs/phase7_task1_int4_*.out
cat checkpoints/int4_native/phase7_int4_benchmark.json
```

## General Workflow

1. **Code Locally (VS Code)**
   - Edit Python files (`utils/*.py`, `models/*.py`)
   - Edit job scripts (`jobs/*.sh`)
   - Commit and push to GitHub

2. **Execute on HPC**
   ```bash
   cd Itera-Lite
   source .venv/bin/activate
   git pull origin main
   sbatch jobs/phase7_task1_quantize.sh
   ```

3. **Monitor & Collect Results**
   ```bash
   squeue -u $USER                          # Check job status
   cat logs/phase7_task1_int4_*.out         # View output
   git add checkpoints/ results/            # Stage results
   git commit -m "Phase 7 Task 1 results"
   git push origin main                     # Sync back to GitHub
   ```

4. **Analyze Locally (VS Code)**
   ```bash
   git pull origin main                     # Get results
   # Analyze JSON files, generate reports
   ```

## Job Script Structure

All job scripts follow this structure:

```bash
#!/bin/bash
#SBATCH --job-name=<task_name>
#SBATCH --output=logs/<task>_%j.out
#SBATCH --error=logs/<task>_%j.err
#SBATCH --partition=gpu                   # Use GPU partition
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --cpus-per-task=8                 # 8 CPU cores
#SBATCH --mem=32G                         # 32 GB RAM
#SBATCH --time=04:00:00                   # 4 hour limit

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Activate environment
source .venv/bin/activate

# Run task
python phase7_<task>.py [args]
```

## GPU Partition Details

- **GPU Model:** NVIDIA A30 (Ampere)
- **VRAM:** 24 GB
- **Compute Capability:** 8.0
- **CUDA Version:** 12.8
- **Available Nodes:** 10 total, typically 5 idle
- **Time Limit:** 2 days (48 hours)

## Tips

- **Off-peak hours:** Submit jobs evenings/weekends for faster queue times
- **Monitor memory:** 32GB should be sufficient for Phase 7 tasks
- **GPU utilization:** Check with `nvidia-smi` in job output
- **Checkpoints:** Jobs save intermediate checkpoints in case of interruption
- **Logs:** Always check `.out` and `.err` files for debugging

## Troubleshooting

**Job won't start:**
```bash
squeue -u $USER  # Check if queued
sinfo -p gpu     # Check GPU partition availability
```

**CUDA not available:**
```bash
# Jobs must use --partition=gpu
# Login nodes don't have GPU access
```

**Out of memory:**
```bash
# Reduce batch size in Python script
# Or request more memory: #SBATCH --mem=64G
```

**Job failed:**
```bash
cat logs/<task>_<jobid>.err  # Check error log
sacct -j <jobid> --format=JobID,JobName,State,ExitCode
```
