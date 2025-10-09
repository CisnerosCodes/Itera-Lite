#!/bin/bash
#SBATCH --job-name=phase7_task1_int4
#SBATCH --output=logs/phase7_task1_int4_%j.out
#SBATCH --error=logs/phase7_task1_int4_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G

echo "========================================="
echo "Phase 7 Task 1: GPU-Native INT4 Quantization"
echo "========================================="
echo ""

echo "Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "  Partition: $SLURM_JOB_PARTITION"
echo "  GPUs Allocated: $SLURM_GPUS"
echo "  CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  Memory: ${SLURM_MEM_PER_NODE}MB"
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
fi

echo "========================================="
echo "2. Verify Dependencies"
echo "========================================="

python -c "
import torch
import bitsandbytes as bnb
print(f'✓ bitsandbytes {bnb.__version__}')
print(f'✓ torch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

echo "========================================="
echo "3. Run INT4 Quantization"
echo "========================================="
echo ""

# Create output directories
mkdir -p checkpoints/int4_native
mkdir -p results
mkdir -p logs

# Find the best checkpoint to quantize
if [ -f "checkpoints/itera_lite_tiny_best.pt" ]; then
    CHECKPOINT="checkpoints/itera_lite_tiny_best.pt"
    echo "Using checkpoint: $CHECKPOINT"
elif [ -f "checkpoints/itera_lite_tiny_final.pt" ]; then
    CHECKPOINT="checkpoints/itera_lite_tiny_final.pt"
    echo "Using checkpoint: $CHECKPOINT"
else
    echo "Error: No checkpoint found!"
    echo "Looking for: checkpoints/itera_lite_tiny_best.pt or checkpoints/itera_lite_tiny_final.pt"
    exit 1
fi

# Run quantization
python phase7_quantize.py \
    --checkpoint "$CHECKPOINT" \
    --output checkpoints/int4_native/itera_lite_int4_nf4.pt \
    --quant-type nf4 \
    --double-quant \
    --compute-dtype float16 \
    --calibration-samples 1000 \
    --calibration-batch-size 32 \
    --qat-epochs 0 \
    --device cuda \
    --benchmark-batches 100

QUANTIZE_EXIT_CODE=$?

echo ""
if [ $QUANTIZE_EXIT_CODE -eq 0 ]; then
    echo "✓ Quantization completed successfully!"
else
    echo "✗ Quantization failed with exit code: $QUANTIZE_EXIT_CODE"
    exit $QUANTIZE_EXIT_CODE
fi

echo ""
echo "========================================="
echo "4. Results Summary"
echo "========================================="
echo ""

# Display quantized model info
if [ -f "checkpoints/int4_native/itera_lite_int4_nf4.pt" ]; then
    MODEL_SIZE=$(du -h checkpoints/int4_native/itera_lite_int4_nf4.pt | cut -f1)
    echo "Quantized Model:"
    echo "  Path: checkpoints/int4_native/itera_lite_int4_nf4.pt"
    echo "  Size: $MODEL_SIZE"
    echo ""
fi

# Display benchmark results
if [ -f "checkpoints/int4_native/phase7_int4_benchmark.json" ]; then
    echo "Benchmark Results:"
    python -c "
import json
with open('checkpoints/int4_native/phase7_int4_benchmark.json') as f:
    results = json.load(f)
    if 'benchmark_results' in results and 'comparison' in results['benchmark_results']:
        comp = results['benchmark_results']['comparison']
        print(f'  Compression: {comp[\"size_reduction\"]:.2f}×')
        print(f'  Speedup: {comp[\"speedup\"]:.2f}×')
        print(f'  Perplexity degradation: {comp[\"perplexity_degradation\"]:.2f}%')
    if 'export_info' in results:
        print(f'  Model size: {results[\"export_info\"][\"size_mb\"]:.2f} MB')
"
    echo ""
fi

echo "========================================="
echo "5. Job Complete!"
echo "========================================="
echo ""
echo "Next Steps:"
echo "  1. Review results: cat logs/phase7_task1_int4_${SLURM_JOB_ID}.out"
echo "  2. Check benchmark: cat checkpoints/int4_native/phase7_int4_benchmark.json"
echo "  3. Commit results: git add checkpoints/int4_native/ results/"
echo "  4. Push to GitHub: git commit -m 'Phase 7 Task 1: INT4 quantization complete' && git push"
echo ""
echo "Job completed at: $(date)"
