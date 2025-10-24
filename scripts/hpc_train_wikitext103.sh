#!/bin/bash
#SBATCH --job-name=itera_lite_wikitext103
#SBATCH --output=logs/train_wikitext103_%j.out
#SBATCH --error=logs/train_wikitext103_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Itera-Lite Production Training on WikiText-103
# Phase 4: HPC GPU Training
#
# This script:
# 1. Sets up the environment
# 2. Downloads WikiText-103 if needed
# 3. Runs production training with full metrics
# 4. Saves checkpoints and logs

echo "========================================================================"
echo "ITERA-LITE WIKITEXT-103 TRAINING (HPC)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

# Load modules (adjust based on your HPC environment)
module load Python/3.10.8
module load CUDA/11.8.0  # Adjust CUDA version if needed

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints/wikitext103_training
mkdir -p results
mkdir -p runs

# Set up Python environment
if [ ! -d "venv_hpc" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv_hpc
fi

source venv_hpc/bin/activate

# Install dependencies
echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pyyaml tqdm tensorboard datasets

# Verify GPU availability
echo ""
echo "========================================================================"
echo "GPU INFO"
echo "========================================================================"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo "========================================================================"
echo ""

# Download WikiText-103 if not already present
if [ ! -f "data/wikitext103/train_tokens.pkl" ]; then
    echo "WikiText-103 dataset not found. Downloading..."
    python data/download_wikitext103_hf.py

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download WikiText-103 dataset"
        exit 1
    fi
    echo "Dataset downloaded successfully!"
else
    echo "WikiText-103 dataset already exists. Skipping download."
fi

# Update config to use GPU
echo "Updating config for GPU training..."
cat > configs/training_config_wikitext103_hpc.yaml << 'EOL'
# HPC GPU Training Configuration for Itera-Lite on WikiText-103
# Auto-generated for HPC cluster

model:
  type: "itera_lite"
  config: "tiny"
  vocab_size: 274

dataset:
  name: "wikitext103"
  path: "data/wikitext103"
  seq_length: 128

training:
  batch_size: 64  # Increased for GPU
  gradient_accumulation_steps: 2  # Effective batch = 128
  learning_rate: 0.001
  warmup_steps: 500
  lr_schedule: "cosine"
  min_lr: 0.00005
  max_epochs: 1500
  max_steps: null
  weight_decay: 0.01
  dropout: 0.1
  max_grad_norm: 1.0

evaluation:
  eval_every_epochs: 5
  eval_every_steps: null
  save_every_epochs: 25
  keep_best_n: 3

early_stopping:
  enabled: true
  patience: 50
  min_delta: 0.01
  metric: "val_perplexity"

generation:
  enabled: true
  every_epochs: 10
  num_samples: 3
  max_new_tokens: 80
  temperature: 0.8
  prompts:
    - "The history of"
    - "In the year"
    - "Scientists have discovered"

logging:
  use_tensorboard: true
  tensorboard_dir: "runs"
  csv_backup: true
  csv_dir: "results"
  print_every: 100

checkpoints:
  dir: "checkpoints/wikitext103_training"
  save_optimizer: true
  save_scheduler: true

hardware:
  device: "cuda"  # GPU enabled
  num_workers: 4
  pin_memory: true

mixed_precision:
  enabled: false
  dtype: "float16"

seed: 42
EOL

echo "Config updated for GPU training!"

# Run training
echo ""
echo "========================================================================"
echo "STARTING TRAINING"
echo "========================================================================"
echo "Config: configs/training_config_wikitext103_hpc.yaml"
echo "Start time: $(date)"
echo "========================================================================"
echo ""

python train_production.py --config configs/training_config_wikitext103_hpc.yaml

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "TRAINING COMPLETED"
echo "========================================================================"
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "========================================================================"

# Copy results to scratch directory for backup
echo "Backing up results..."
BACKUP_DIR="/scratch/user/$(whoami)/itera_lite_backups/run_${SLURM_JOB_ID}"
mkdir -p $BACKUP_DIR
cp -r checkpoints/wikitext103_training $BACKUP_DIR/
cp -r results $BACKUP_DIR/
cp logs/train_wikitext103_${SLURM_JOB_ID}.out $BACKUP_DIR/
cp logs/train_wikitext103_${SLURM_JOB_ID}.err $BACKUP_DIR/

echo "Backup saved to: $BACKUP_DIR"

# Deactivate environment
deactivate

echo "Job complete!"
exit $TRAIN_EXIT_CODE
