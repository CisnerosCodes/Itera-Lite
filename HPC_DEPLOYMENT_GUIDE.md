# HPC Deployment Guide: Itera-Lite WikiText-103 Training

## 🎯 Overview

This guide walks you through deploying your Itera-Lite model training to the HPC cluster with GPU acceleration.

**Expected Results:**
- Training on WikiText-103 (539M tokens)
- GPU acceleration (100x faster than CPU)
- Full metrics tracking (perplexity, generation samples)
- Estimated time: **3-5 days** for 1500 epochs

---

## 📋 Step-by-Step Deployment

### **Step 1: Prepare Local Repository**

On your local machine (Windows):

```bash
cd c:\Users\adria\Itera-Lite

# Check git status
git status

# Add all new files
git add .

# Commit changes
git commit -m "Add production training pipeline for WikiText-103

- Add train_production.py with full metrics tracking
- Add WikiText-103 dataset loader
- Add comprehensive metrics tracker (perplexity, generation)
- Add HPC job submission script
- Add training configs for GPU and CPU
- Dataset: WikiText-103 (539M tokens) via Hugging Face"

# Push to GitHub
git push origin main
```

**If you need to set up git (first time):**
```bash
git config --global user.email "your.email@example.com"
git config --global user.name "Your Name"
```

---

### **Step 2: Connect to HPC and Pull Code**

SSH into the HPC cluster:

```bash
# You should already be logged in based on your terminal
cd ~/Itera-Lite

# Pull latest changes from GitHub
git pull origin main

# Verify files
ls -la scripts/
ls -la data/
ls -la configs/
```

---

### **Step 3: Verify HPC Environment**

Check available resources:

```bash
# Check GPU partitions
sinfo -p gpu

# Check your current quotas (you already have this)
showquota

# Check available GPUs
squeue -p gpu
```

---

### **Step 4: Make Script Executable**

```bash
cd ~/Itera-Lite

# Make the job script executable
chmod +x scripts/hpc_train_wikitext103.sh

# Verify
ls -l scripts/hpc_train_wikitext103.sh
```

---

### **Step 5: Submit Training Job**

```bash
# Submit the job
sbatch scripts/hpc_train_wikitext103.sh

# You'll see output like:
# Submitted batch job 123456
```

**Job will:**
1. Request 1 GPU for 72 hours
2. Allocate 32GB RAM
3. Download WikiText-103 if needed
4. Install Python dependencies
5. Run training with full metrics
6. Save checkpoints every 25 epochs
7. Generate text samples every 10 epochs

---

### **Step 6: Monitor Training Progress**

#### **Check Job Status:**
```bash
# View your jobs
squeue -u $(whoami)

# Output will show:
# JOBID  PARTITION  NAME                 USER    ST  TIME  NODES
# 123456 gpu        itera_lite_wikitext  u.ac29  R   0:15  1
```

#### **View Live Output:**
```bash
# Check job number from squeue
JOBID=123456  # Replace with your actual job ID

# View training output (updates live)
tail -f logs/train_wikitext103_${JOBID}.out

# View errors (if any)
tail -f logs/train_wikitext103_${JOBID}.err
```

**Press Ctrl+C to stop tailing**

#### **Check Training Progress:**
```bash
# View last 50 lines of output
tail -50 logs/train_wikitext103_${JOBID}.out

# Search for specific metrics
grep "EPOCH.*SUMMARY" logs/train_wikitext103_${JOBID}.out
grep "Val PPL" logs/train_wikitext103_${JOBID}.out
grep "BEST" logs/train_wikitext103_${JOBID}.out
```

---

### **Step 7: Check Saved Checkpoints**

```bash
# List checkpoints
ls -lh checkpoints/wikitext103_training/

# You should see:
# - best_model.pt (best validation perplexity)
# - epoch_25.pt, epoch_50.pt, etc.
# - final_model.pt (when training completes)

# Check size
du -sh checkpoints/wikitext103_training/
```

---

### **Step 8: Monitor Metrics**

```bash
# View CSV metrics
cat results/itera_lite_wikitext103_metrics.csv | column -t -s,

# Check specific epochs
grep "^500," results/itera_lite_wikitext103_metrics.csv
```

---

### **Step 9: Download Results (After Training)**

On your local machine:

```bash
# Create directory for results
mkdir -p C:\Users\adria\Itera-Lite\hpc_results

# Use SCP to download (if available) or HPC file transfer
# Replace with your HPC hostname
scp u.ac290968@launch-login2.hpc.tamu.edu:~/Itera-Lite/checkpoints/wikitext103_training/best_model.pt C:\Users\adria\Itera-Lite\hpc_results\

# Or use HPC's file transfer system (e.g., Globus, web interface)
```

---

## 📊 Expected Training Timeline (GPU)

Based on typical HPC GPU performance:

| Metric | Estimate |
|--------|----------|
| **Speed** | ~200-500 it/sec (vs 2 it/sec CPU) |
| **Epoch time** | ~15-30 minutes (vs 35 hours CPU) |
| **5 epochs** | 1-2 hours |
| **100 epochs** | 1-2 days |
| **1500 epochs** | 3-5 days |

**You should see results after just 5 epochs!**

---

## 🔍 What to Look For

### **Healthy Training Signs:**

1. **Loss decreasing:**
   - Epoch 1: ~5.4
   - Epoch 5: ~3.5-4.0
   - Epoch 50: ~2.5-3.0
   - Epoch 500+: <2.0

2. **Perplexity dropping:**
   - Start: >200
   - After 100 epochs: <100
   - Target: <50

3. **Generation improving:**
   - Check samples in logs every 10 epochs
   - Should become more coherent over time

### **Warning Signs:**

- Loss NaN → learning rate too high (stop and adjust)
- Loss not decreasing after 50 epochs → potential issue
- OOM errors → reduce batch size in config

---

## 🛑 Stopping/Canceling Training

```bash
# Find your job ID
squeue -u $(whoami)

# Cancel job
scancel JOBID

# The script will save a checkpoint before exiting
```

---

## 🔄 Resuming Training

If training stops or you want to continue:

```bash
# Find the latest checkpoint
ls -lt checkpoints/wikitext103_training/

# Modify job script to resume (add --resume flag)
# Or submit with resume:
sbatch --export=RESUME_FROM="checkpoints/wikitext103_training/epoch_100.pt" scripts/hpc_train_wikitext103.sh
```

---

## 📝 Quick Command Reference

```bash
# === On Local Machine (Windows) ===
cd C:\Users\adria\Itera-Lite
git add .
git commit -m "Your message"
git push origin main

# === On HPC ===
# Pull code
cd ~/Itera-Lite
git pull origin main

# Submit job
sbatch scripts/hpc_train_wikitext103.sh

# Check status
squeue -u $(whoami)

# View output
tail -f logs/train_wikitext103_JOBID.out

# Cancel job
scancel JOBID

# Check checkpoints
ls -lh checkpoints/wikitext103_training/

# View metrics
tail results/itera_lite_wikitext103_metrics.csv
```

---

## ✅ Checklist

Before submitting:
- [ ] Code pushed to GitHub
- [ ] Logged into HPC
- [ ] Pulled latest code on HPC
- [ ] Script is executable (`chmod +x`)
- [ ] Sufficient disk quota (check with `showquota`)

After submission:
- [ ] Job submitted successfully (note JOBID)
- [ ] Check logs after 5 minutes
- [ ] Verify GPU is being used
- [ ] Monitor first 5 epochs for healthy metrics

---

## 🎯 Success Criteria

Training is successful when:
1. ✅ Validation perplexity < 100 (ideally < 50)
2. ✅ Generated text is coherent
3. ✅ Loss steadily decreasing
4. ✅ No overfitting (train/val gap < 2x)

**You should see decent quality after 100-200 epochs (~1-2 days on GPU)!**

---

## 🆘 Troubleshooting

### **Job won't start:**
- Check partition availability: `sinfo -p gpu`
- Check queue: `squeue -p gpu`
- Reduce time/memory request if needed

### **Module not found:**
```bash
# List available modules
module avail

# Load appropriate Python/CUDA
module load Python/3.10.8
module load CUDA/11.8.0
```

### **Python dependencies fail:**
- Check Python version: `python --version`
- Manually install: `pip install torch datasets pyyaml tqdm tensorboard`

### **Dataset download fails:**
- Pre-download on login node: `python data/download_wikitext103_hf.py`
- Then submit job

---

## 📞 Need Help?

Check the logs first:
```bash
cat logs/train_wikitext103_JOBID.err
```

Common solutions in this guide. If stuck, you have all the tools to debug!

**Good luck with your HPC training! 🚀**
