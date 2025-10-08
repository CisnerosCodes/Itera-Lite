#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=8G

echo "========================================="
echo "GPU Hardware Detection Test"
echo "========================================="
echo ""

echo "Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "  Partition: $SLURM_JOB_PARTITION"
echo "  GPUs Allocated: $SLURM_GPUS"
echo ""

echo "========================================="
echo "1. NVIDIA System Management Interface"
echo "========================================="
nvidia-smi
echo ""

echo "========================================="
echo "2. GPU Device Details"
echo "========================================="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap --format=csv
echo ""

echo "========================================="
echo "3. CUDA Environment"
echo "========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check for CUDA toolkit
if command -v nvcc &> /dev/null; then
    echo "CUDA Compiler (nvcc) version:"
    nvcc --version
else
    echo "CUDA Compiler (nvcc): Not found in PATH"
fi
echo ""

echo "========================================="
echo "4. Python Environment Detection"
echo "========================================="

# Activate virtual environment
source .venv/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo ""

echo "========================================="
echo "5. PyTorch CUDA Detection"
echo "========================================="

python << 'PYTHON_EOF'
import sys
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("")
    
    print("=" * 50)
    print("GPU Device Information:")
    print("=" * 50)
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")
        print(f"  CUDA Cores: ~{props.multi_processor_count * 64}")  # Approximate
        
        # Memory info
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
        
        print(f"  Memory Allocated: {mem_allocated:.2f} GB")
        print(f"  Memory Reserved: {mem_reserved:.2f} GB")
        print(f"  Memory Free: {mem_free:.2f} GB")
    
    print("")
    print("=" * 50)
    print("CUDA Capability Assessment:")
    print("=" * 50)
    
    device = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(device)
    cap_major, cap_minor = cap
    
    print(f"Compute Capability: {cap_major}.{cap_minor}")
    
    if cap_major >= 8:
        print("  ✓ EXCELLENT - Ampere or newer (RTX 30xx/A100+)")
        print("  ✓ Native FP16 Tensor Cores available")
        print("  ✓ INT8 Tensor Cores available")
        print("  ✓ Optimal for Phase 7 mixed-precision")
    elif cap_major >= 7:
        print("  ✓ GOOD - Volta/Turing (V100/RTX 20xx)")
        print("  ✓ FP16 Tensor Cores available")
        print("  ✓ Suitable for Phase 7 with FP16 acceleration")
    elif cap_major >= 6:
        print("  ⚠ MODERATE - Pascal (GTX 10xx/P100)")
        print("  ⚠ No Tensor Cores (FP16 emulated)")
        print("  ⚠ Limited Phase 7 acceleration")
    else:
        print("  ✗ LIMITED - Older architecture")
        print("  ✗ CPU fallback recommended")
    
    print("")
    print("=" * 50)
    print("Phase 7 GPU Suitability:")
    print("=" * 50)
    
    vram_gb = props.total_memory / 1024**3
    
    print(f"VRAM: {vram_gb:.2f} GB")
    if vram_gb >= 16:
        print("  ✓ EXCELLENT - Can handle all Phase 7 tasks")
        print("  ✓ Full batch sizes for fine-tuning")
    elif vram_gb >= 8:
        print("  ✓ GOOD - Sufficient for Phase 7")
        print("  ⚠ May need reduced batch sizes")
    else:
        print("  ⚠ LIMITED - Small VRAM")
        print("  ⚠ Gradient checkpointing required")
    
    print("")
    print("Recommended Phase 7 Strategy:")
    if cap_major >= 7 and vram_gb >= 8:
        print("  • Use GPU-accelerated bitsandbytes for INT4")
        print("  • Enable mixed-precision training (FP16)")
        print("  • GPU-accelerated pruning fine-tuning")
        print("  • Add CUDA kernel fusion to Task 4")
        print("  • Estimated timeline: 4-5 weeks (50% reduction)")
    else:
        print("  • Use CPU fallback for quantization")
        print("  • Limited GPU usage for fine-tuning only")
        print("  • Estimated timeline: 6-7 weeks")

else:
    print("⚠ CUDA NOT AVAILABLE")
    print("This may indicate:")
    print("  - GPU not allocated (check SLURM #SBATCH --gres=gpu:1)")
    print("  - CUDA drivers not installed on compute node")
    print("  - PyTorch CPU-only build (unlikely, version shows +cu128)")
    print("")
    print("Troubleshooting:")
    print("  1. Verify GPU allocation: echo $CUDA_VISIBLE_DEVICES")
    print("  2. Check nvidia-smi output above")
    print("  3. Verify SLURM job requested GPU: scontrol show job $SLURM_JOB_ID")

sys.exit(0 if torch.cuda.is_available() else 1)
PYTHON_EOF

echo ""
echo "========================================="
echo "6. Additional Package Checks"
echo "========================================="

python << 'PYTHON_EOF'
print("Checking Phase 7 dependencies on GPU:")
print("")

try:
    import bitsandbytes as bnb
    print(f"✓ bitsandbytes {bnb.__version__}")
    # Check if CUDA version is available
    if hasattr(bnb, 'cextension'):
        print("  - CUDA extension available")
except Exception as e:
    print(f"⚠ bitsandbytes import error: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"⚠ transformers import error: {e}")

try:
    import optimum
    print(f"✓ optimum {optimum.__version__}")
except Exception as e:
    print(f"⚠ optimum import error: {e}")

try:
    import torch_pruning
    print(f"✓ torch-pruning {torch_pruning.__version__}")
except Exception as e:
    print(f"⚠ torch-pruning import error: {e}")

print("")
print("All Phase 7 dependencies check complete!")
PYTHON_EOF

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo "Review the output above to determine GPU capabilities."
echo "This information will guide Phase 7 optimization strategy."
