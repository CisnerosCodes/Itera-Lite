#!/bin/bash
#SBATCH --job-name=phase8_quality_train
#SBATCH --output=logs/phase8_quality_training_%j.out
#SBATCH --error=logs/phase8_quality_training_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

echo "========================================================================"
echo "PHASE 8: QUALITY TRAINING FOR COHERENT TEXT GENERATION"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

# Load modules
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source .venv/bin/activate

# Verify environment
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Install any missing dependencies
echo "Checking dependencies..."
pip install -q requests tqdm

# Run training
echo "========================================================================"
echo "STARTING TRAINING"
echo "========================================================================"

python phase8_train_quality.py \
    --use-existing-data \
    --epochs 20 \
    --batch-size 32 \
    --lr 3e-4 \
    --device cuda

echo ""
echo "========================================================================"
echo "TRAINING COMPLETED"
echo "========================================================================"
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Output files:"
echo "  - Best model: checkpoints/itera_lite_quality_best.pt"
echo "  - Latest model: checkpoints/itera_lite_quality_latest.pt"
echo "  - Tokenizer: data/tokenizer_quality.json"
echo "  - Training history: results/phase8_training_history.json"
echo "========================================================================"
