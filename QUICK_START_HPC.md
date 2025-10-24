# Quick Start: HPC Training

## 🚀 Fast Track (5 Minutes)

### On Your Computer (Windows):

```bash
# 1. Commit and push
cd C:\Users\adria\Itera-Lite
git add .
git commit -m "Add WikiText-103 production training pipeline"
git push origin main
```

### On HPC:

```bash
# 2. Pull code
cd ~/Itera-Lite
git pull origin main

# 3. Make executable
chmod +x scripts/hpc_train_wikitext103.sh

# 4. Submit job
sbatch scripts/hpc_train_wikitext103.sh

# 5. Get job ID (note the number)
squeue -u $(whoami)

# 6. Monitor (replace JOBID with actual number)
tail -f logs/train_wikitext103_JOBID.out
```

## Expected Output:

After 5 minutes, you should see:
```
======================================================================
STARTING TRAINING
======================================================================
Epoch 1/1500:  0%|          | 0/8356 [00:00<?, ?it/s]
```

After 1 hour (5 epochs):
```
======================================================================
EPOCH 5 SUMMARY
======================================================================
  Train Loss: 3.5234  |  Train PPL: 33.85
  Val Loss:   3.6142  |  Val PPL:   37.12
======================================================================
```

## ✅ That's it! Training will run for 3-5 days.

Check progress anytime with:
```bash
tail -50 logs/train_wikitext103_JOBID.out
```

See [HPC_DEPLOYMENT_GUIDE.md](HPC_DEPLOYMENT_GUIDE.md) for full details.
