#!/bin/bash
#SBATCH --job-name=phase7_task2_prune
#SBATCH --output=logs/phase7_task2_prune_%j.out
#SBATCH --error=logs/phase7_task2_prune_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G

echo "========================================="
echo "Phase 7 Task 2: Structured Pruning"
echo "========================================="
echo ""

echo "Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "  Partition: $SLURM_JOB_PARTITION"
echo "  GPUs Allocated: $SLURM_GPUS"
echo "  CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  Memory: ${SLURM_MEM_PER_NODE}MB"
echo "  Start Time: $(date)"
echo ""

echo "========================================="
echo "1. Environment Setup"
echo "========================================="

# Activate virtual environment
source .venv/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo ""

echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
    echo ""
    echo "GPU Status:"
    nvidia-smi
    echo ""
fi

echo "========================================="
echo "2. Verify Dependencies"
echo "========================================="

python -c "
import sys
import torch
import matplotlib
import numpy as np

print(f'✓ Python {sys.version.split()[0]}')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ Matplotlib {matplotlib.__version__}')
print(f'✓ NumPy {np.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'✓ Compute Capability: {torch.cuda.get_device_capability(0)}')

# Check custom modules
try:
    from models.itera_lite import IteraLiteModel
    from models.config import IteraLiteConfig
    from utils.structured_pruning import StructuredPruner, PruningConfig
    print(f'✓ Custom modules loaded successfully')
except ImportError as e:
    print(f'✗ Error loading custom modules: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Dependency verification failed!"
    exit 1
fi

echo ""

echo "========================================="
echo "3. Prepare Directories"
echo "========================================="

# Create output directories
mkdir -p checkpoints/pruned
mkdir -p results
mkdir -p logs
mkdir -p reports

echo "✓ Created output directories"
echo ""

echo "========================================="
echo "4. Locate Checkpoint"
echo "========================================="

# Find the best checkpoint to prune
if [ -f "checkpoints/itera_lite_tiny_best.pt" ]; then
    CHECKPOINT="checkpoints/itera_lite_tiny_best.pt"
    echo "✓ Using checkpoint: $CHECKPOINT"
elif [ -f "checkpoints/itera_lite_tiny_final.pt" ]; then
    CHECKPOINT="checkpoints/itera_lite_tiny_final.pt"
    echo "✓ Using checkpoint: $CHECKPOINT"
else
    echo "ERROR: No checkpoint found!"
    echo "Looking for: checkpoints/itera_lite_tiny_best.pt or checkpoints/itera_lite_tiny_final.pt"
    exit 1
fi

# Check checkpoint exists and is readable
if [ ! -r "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint $CHECKPOINT is not readable!"
    exit 1
fi

# Get checkpoint size
CHECKPOINT_SIZE=$(du -h "$CHECKPOINT" | cut -f1)
echo "  Checkpoint size: $CHECKPOINT_SIZE"
echo ""

echo "========================================="
echo "5. Run Structured Pruning"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Target sparsity: 40%"
echo "  SSM sparsity: 25%"
echo "  MoE expert sparsity: 60%"
echo "  Fine-tuning epochs: 5"
echo "  Learning rate: 1e-4"
echo "  Batch size: 32"
echo "  Device: CUDA (A30 GPU)"
echo ""
echo "Starting pruning pipeline..."
echo ""

# Run pruning with fine-tuning
python phase7_prune.py \
    --checkpoint "$CHECKPOINT" \
    --output "checkpoints/pruned/itera_lite_pruned_40pct.pt" \
    --target-sparsity 0.4 \
    --ssm-sparsity 0.25 \
    --moe-sparsity 0.60 \
    --finetune-epochs 5 \
    --finetune-lr 1e-4 \
    --batch-size 32 \
    --warmup-ratio 0.05 \
    --train-samples 5000 \
    --val-samples 1000 \
    --test-samples 500 \
    --device cuda \
    --visualize

# Check if pruning succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "6. Pruning Results"
    echo "========================================="
    
    # Check output files
    if [ -f "checkpoints/pruned/itera_lite_pruned_40pct.pt" ]; then
        PRUNED_SIZE=$(du -h "checkpoints/pruned/itera_lite_pruned_40pct.pt" | cut -f1)
        echo "✓ Pruned checkpoint saved: checkpoints/pruned/itera_lite_pruned_40pct.pt"
        echo "  Size: $PRUNED_SIZE"
    fi
    
    if [ -f "checkpoints/pruned/itera_lite_pruned_40pct.json" ]; then
        echo "✓ Metadata saved: checkpoints/pruned/itera_lite_pruned_40pct.json"
    fi
    
    if [ -f "checkpoints/pruned/pruning_statistics.json" ]; then
        echo "✓ Statistics saved: checkpoints/pruned/pruning_statistics.json"
        echo ""
        echo "Pruning Statistics:"
        python -c "
import json
with open('checkpoints/pruned/pruning_statistics.json', 'r') as f:
    stats = json.load(f)
    print(f\"  Original params: {stats.get('original_params', 0):,}\")
    print(f\"  Pruned params: {stats.get('pruned_params', 0):,}\")
    print(f\"  Removed params: {stats.get('total_removed', 0):,}\")
    print(f\"  Overall sparsity: {stats.get('overall_sparsity', 0)*100:.2f}%\")
    if 'original_params' in stats and 'pruned_params' in stats:
        compression = stats['original_params'] / stats['pruned_params']
        print(f\"  Compression ratio: {compression:.2f}×\")
"
    fi
    
    if [ -f "checkpoints/pruned/pruning_sparsity.png" ]; then
        echo ""
        echo "✓ Visualization saved: checkpoints/pruned/pruning_sparsity.png"
    fi
    
    echo ""
    echo "========================================="
    echo "7. GPU Memory Summary"
    echo "========================================="
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    echo ""
    
    echo "========================================="
    echo "SUCCESS: Phase 7 Task 2 Complete!"
    echo "========================================="
    echo "  End Time: $(date)"
    echo ""
    echo "Output Files:"
    echo "  - checkpoints/pruned/itera_lite_pruned_40pct.pt (pruned model)"
    echo "  - checkpoints/pruned/itera_lite_pruned_40pct.json (metadata)"
    echo "  - checkpoints/pruned/pruning_statistics.json (detailed stats)"
    echo "  - checkpoints/pruned/pruning_sparsity.png (visualization)"
    echo ""
    echo "Next Steps:"
    echo "  1. Review pruning statistics and benchmark results"
    echo "  2. Validate quality degradation (<5% target)"
    echo "  3. Optionally combine with INT4 quantization for 2.37× cumulative compression"
    echo "  4. Generate Phase 7 Task 2 completion report"
    echo ""
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "ERROR: Pruning failed!"
    echo "========================================="
    echo "  End Time: $(date)"
    echo ""
    echo "Check error log: logs/phase7_task2_prune_${SLURM_JOB_ID}.err"
    echo ""
    echo "Common Issues:"
    echo "  - Out of GPU memory: Reduce batch size (--batch-size)"
    echo "  - Config inference failed: Check checkpoint format"
    echo "  - Convergence issues: Reduce learning rate (--finetune-lr)"
    echo "  - Sequence length errors: Verify max_seq_length matches checkpoint"
    echo ""
    echo "========================================="
    exit 1
fi
