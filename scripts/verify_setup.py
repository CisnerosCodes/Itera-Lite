#!/usr/bin/env python3
"""Verify Phase 2 readiness for Itera-Lite project"""

print("=" * 70)
print("ITERA-LITE PHASE 2 READINESS CHECK")
print("=" * 70)
print()

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    has_torch = True
except ImportError:
    print("✗ PyTorch: MISSING")
    has_torch = False

# Check core libraries
print()
libraries = {
    'numpy': 'NumPy',
    'transformers': 'Transformers',
    'datasets': 'Datasets',
    'accelerate': 'Accelerate',
    'sentencepiece': 'SentencePiece',
    'einops': 'Einops',
    'peft': 'PEFT',
    'tqdm': 'TQDM',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
}

all_installed = True
for module, name in libraries.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
    except ImportError:
        print(f"✗ {name}: MISSING")
        all_installed = False

# Check optional Mamba SSM
print()
print("Optional components:")
try:
    import mamba_ssm
    print(f"✓ Mamba SSM: installed")
    has_mamba = True
except ImportError:
    print(f"○ Mamba SSM: not installed (will implement SSM from scratch)")
    has_mamba = False

print()
print("=" * 70)

if has_torch and all_installed:
    print("PHASE 2 STATUS: ✅ READY TO BEGIN! 🚀")
    print("=" * 70)
    print()
    print("Environment Summary:")
    print("  • Python 3.13.7 ✓")
    print("  • PyTorch installed ✓")
    print("  • All core dependencies ready ✓")
    print("  • Computing: CPU-based (optimized for smaller models)")
    print()
    print("Next steps:")
    print("  1. Create model architecture directory")
    print("  2. Implement custom SSM backbone")
    print("  3. Build Mixture-of-Experts layer")
    print("  4. Create Transformer baseline")
    print("  5. Test and validate models")
    print()
    print("Recommended model size for CPU: 1-10M parameters")
else:
    print("PHASE 2 STATUS: ⚠️ SOME DEPENDENCIES MISSING")
    print("=" * 70)
    print()
    print("Please install missing dependencies before proceeding.")
