#!/bin/bash
#SBATCH --job-name=phase7_task3_mixed
#SBATCH --output=logs/phase7_task3_mixed_%j.out
#SBATCH --error=logs/phase7_task3_mixed_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

################################################################################
# Phase 7 Task 3: Mixed-Precision Optimization
# 
# Applies layer-wise INT8/FP16 precision to Itera-Lite model.
# Expected results:
#   - Compression: 1.5× standalone
#   - Quality: <5% perplexity degradation
#   - Speedup: 1.3× inference on A30 GPU
#
# Author: GitHub Copilot
# Date: October 10, 2025
################################################################################

echo "========================================="
echo "PHASE 7 TASK 3: MIXED-PRECISION OPTIMIZATION"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

# Exit on error
set -e

################################################################################
# 1. Environment Setup
################################################################################

echo "========================================="
echo "1. Environment Setup"
echo "========================================="

# Activate virtual environment
if [ -d "$HOME/Itera-Lite/.venv" ]; then
    echo "Activating virtual environment..."
    source "$HOME/Itera-Lite/.venv/bin/activate"
elif [ -d ".venv" ]; then
    echo "Activating local virtual environment..."
    source .venv/bin/activate
else
    echo "⚠ WARNING: No virtual environment found!"
fi

# Change to project directory
cd "$HOME/Itera-Lite" || { echo "❌ ERROR: Failed to cd to Itera-Lite"; exit 1; }

# Set PYTHONPATH to include current directory
export PYTHONPATH="${PWD}:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Verify Python
echo ""
echo "Python Environment:"
which python
python --version
echo ""

# Verify PyTorch and CUDA
echo "PyTorch and CUDA:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Verify GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

################################################################################
# 2. Verify Dependencies
################################################################################

echo "========================================="
echo "2. Verify Dependencies"
echo "========================================="

# Check required packages
echo "Checking required packages..."
python -c "import torch; import numpy; import matplotlib; print('✓ All required packages available')" || {
    echo "❌ ERROR: Missing required packages"
    exit 1
}

# Check custom modules
echo "Checking custom modules..."
python -c "import sys; sys.path.insert(0, '.'); from models.itera_lite import IteraLiteModel; from utils.mixed_precision import MixedPrecisionConverter; print('✓ Custom modules loaded successfully')" || {
    echo "❌ ERROR: Failed to import custom modules"
    echo "Attempting to show error details..."
    python -c "import sys; sys.path.insert(0, '.'); from models.itera_lite import IteraLiteModel; from utils.mixed_precision import MixedPrecisionConverter"
    exit 1
}

echo "✓ All dependencies verified"
echo ""

################################################################################
# 3. Check Checkpoint
################################################################################

echo "========================================="
echo "3. Check Checkpoint"
echo "========================================="

CHECKPOINT="checkpoints/itera_lite_tiny_best.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
ls -lh "$CHECKPOINT"
echo ""

################################################################################
# 4. Create Output Directory
################################################################################

echo "========================================="
echo "4. Create Output Directory"
echo "========================================="

OUTPUT_DIR="checkpoints/mixed_precision"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

################################################################################
# 5. Run Mixed-Precision Conversion
################################################################################

echo "========================================="
echo "5. Mixed-Precision Conversion"
echo "========================================="

# Configuration
STRATEGY="conservative"  # or "aggressive"
CALIBRATION_SAMPLES=1000
EVAL_SAMPLES=500
BATCH_SIZE=32
DEVICE="cuda"

echo "Configuration:"
echo "  Strategy: $STRATEGY"
echo "  Calibration samples: $CALIBRATION_SAMPLES"
echo "  Evaluation samples: $EVAL_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""

echo "Running mixed-precision conversion..."
python phase7_mixed_precision.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR" \
    --strategy "$STRATEGY" \
    --calibration-samples $CALIBRATION_SAMPLES \
    --eval-samples $EVAL_SAMPLES \
    --batch-size $BATCH_SIZE \
    --device "$DEVICE" \
    2>&1

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Mixed-precision conversion failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "✓ Mixed-precision conversion complete"
echo ""

################################################################################
# 6. Validate Results
################################################################################

echo "========================================="
echo "6. Validate Results"
echo "========================================="

# Check output files
MIXED_CHECKPOINT="$OUTPUT_DIR/itera_lite_mixed_precision.pt"
MIXED_METADATA="$OUTPUT_DIR/itera_lite_mixed_precision.json"
STATISTICS="$OUTPUT_DIR/mixed_precision_statistics.json"
VISUALIZATION="$OUTPUT_DIR/precision_allocation.png"

echo "Checking output files..."

if [ -f "$MIXED_CHECKPOINT" ]; then
    echo "✓ Mixed-precision checkpoint: $MIXED_CHECKPOINT"
    ls -lh "$MIXED_CHECKPOINT"
else
    echo "❌ ERROR: Mixed-precision checkpoint not found!"
fi

if [ -f "$MIXED_METADATA" ]; then
    echo "✓ Metadata: $MIXED_METADATA"
    ls -lh "$MIXED_METADATA"
else
    echo "⚠ WARNING: Metadata file not found"
fi

if [ -f "$STATISTICS" ]; then
    echo "✓ Statistics: $STATISTICS"
    ls -lh "$STATISTICS"
else
    echo "⚠ WARNING: Statistics file not found"
fi

if [ -f "$VISUALIZATION" ]; then
    echo "✓ Visualization: $VISUALIZATION"
    ls -lh "$VISUALIZATION"
else
    echo "⚠ WARNING: Visualization not found"
fi

echo ""

# Display statistics if available
if [ -f "$STATISTICS" ]; then
    echo "Mixed-Precision Statistics:"
    echo "-------------------------------------------"
    python -c "
import json
with open('$STATISTICS', 'r') as f:
    stats = json.load(f)
    
arch = stats.get('architecture', {})
comp = stats.get('compression', {})
bench = stats.get('benchmark', {})

print(f\"Architecture:\")
print(f\"  Total params: {arch.get('total_params', 0):,}\")
print(f\"  INT8 params: {arch.get('int8_params', 0):,} ({arch.get('int8_params', 0)*100//arch.get('total_params', 1)}%)\")
print(f\"  FP16 params: {arch.get('fp16_params', 0):,} ({arch.get('fp16_params', 0)*100//arch.get('total_params', 1)}%)\")
print()
print(f\"Compression:\")
print(f\"  FP32 memory: {comp.get('fp32_memory_mb', 0):.2f} MB\")
print(f\"  Mixed memory: {comp.get('mixed_memory_mb', 0):.2f} MB\")
print(f\"  Compression ratio: {comp.get('compression_ratio', 0):.2f}×\")
print(f\"  Memory saved: {comp.get('memory_saved_mb', 0):.2f} MB\")
print()
print(f\"Benchmark:\")
print(f\"  Original perplexity: {bench.get('original_perplexity', 0):.4f}\")
print(f\"  Mixed perplexity: {bench.get('mixed_perplexity', 0):.4f}\")
print(f\"  Perplexity increase: {bench.get('perplexity_increase_pct', 0):+.2f}%\")
print(f\"  Inference speedup: {bench.get('speedup', 0):.2f}×\")
" || echo "⚠ Could not parse statistics"
    echo "-------------------------------------------"
fi

echo ""

################################################################################
# 7. GPU Memory Summary
################################################################################

echo "========================================="
echo "7. GPU Memory Summary"
echo "========================================="

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
else
    echo "nvidia-smi not available"
fi

echo ""

################################################################################
# 8. Completion
################################################################################

echo "========================================="
echo "SUCCESS: Phase 7 Task 3 Complete!"
echo "========================================="
echo "  End Time: $(date)"
echo ""
echo "Output Files:"
echo "  - $MIXED_CHECKPOINT (mixed-precision model)"
echo "  - $MIXED_METADATA (metadata)"
echo "  - $STATISTICS (detailed statistics)"
echo "  - $VISUALIZATION (visualization)"
echo ""
echo "Next Steps:"
echo "  1. Review statistics and benchmark results"
echo "  2. Validate quality degradation (<5% target)"
echo "  3. Compare with Task 1 (INT4 quantization)"
echo "  4. Generate Phase 7 Task 3 completion report"
echo ""
echo "========================================="
