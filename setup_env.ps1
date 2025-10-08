# ============================================================================
# Itera-Lite Environment Setup Script (PowerShell)
# ============================================================================
# This script sets up the complete development environment for the
# Itera-Lite ultra-efficient mini language model project
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ITERA-LITE ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python version
Write-Host "[1/5] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = & python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

if ($pythonVersion -match "Python 3\.([0-9]+)\.") {
    $minorVersion = [int]$Matches[1]
    if ($minorVersion -lt 10) {
        Write-Host "  ERROR: Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Virtual environment (already exists, so activate it)
Write-Host ""
Write-Host "[2/5] Virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "  Virtual environment '.venv' already exists" -ForegroundColor Green
    Write-Host "  Activating..." -ForegroundColor Green
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "  Creating virtual environment 'compact-llm'..." -ForegroundColor Green
    python -m venv compact-llm
    Write-Host "  Activating..." -ForegroundColor Green
    & .\compact-llm\Scripts\Activate.ps1
}

# Step 3: Upgrade pip
Write-Host ""
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Step 4: Install core dependencies
Write-Host ""
Write-Host "[4/5] Installing core dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Cyan
Write-Host ""

# Core packages
Write-Host "  Installing PyTorch (CUDA 12.1 version for GPU support)..." -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
Write-Host "  Installing essential ML libraries..." -ForegroundColor Cyan
pip install numpy tqdm transformers sentencepiece datasets accelerate

Write-Host ""
Write-Host "  Installing Mamba SSM (State Space Model)..." -ForegroundColor Cyan
# Note: mamba-ssm may require CUDA toolkit and may fail on some systems
pip install mamba-ssm 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    WARNING: mamba-ssm installation failed (may require CUDA toolkit)" -ForegroundColor Yellow
    Write-Host "    You can install it manually later if needed" -ForegroundColor Yellow
}

# Step 5: Install optional dependencies
Write-Host ""
Write-Host "[5/5] Installing optional dependencies..." -ForegroundColor Yellow

Write-Host "  Installing bitsandbytes (quantization)..." -ForegroundColor Cyan
pip install bitsandbytes 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    WARNING: bitsandbytes installation failed" -ForegroundColor Yellow
}

Write-Host "  Installing PEFT (Parameter-Efficient Fine-Tuning)..." -ForegroundColor Cyan
pip install peft

Write-Host "  Installing flash-attn (optimized attention)..." -ForegroundColor Cyan
pip install flash-attn --no-build-isolation 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    WARNING: flash-attn installation failed (requires specific CUDA setup)" -ForegroundColor Yellow
    Write-Host "    This is optional - you can continue without it" -ForegroundColor Yellow
}

# Additional useful packages
Write-Host ""
Write-Host "  Installing additional utilities..." -ForegroundColor Cyan
pip install matplotlib seaborn wandb tensorboard einops

# Step 6: Verify installation
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VERIFYING INSTALLATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run verification script
python -c @"
import sys
print('Python version:', sys.version)
print()

# Check PyTorch
try:
    import torch
    print('✓ PyTorch:', torch.__version__)
    print('  CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('  CUDA version:', torch.version.cuda)
        print('  GPU count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}:', torch.cuda.get_device_name(i))
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f'    VRAM: {vram:.2f} GB')
except ImportError as e:
    print('✗ PyTorch: NOT INSTALLED')

# Check other packages
packages = {
    'numpy': 'NumPy',
    'transformers': 'Transformers',
    'datasets': 'Datasets',
    'accelerate': 'Accelerate',
    'sentencepiece': 'SentencePiece',
    'tqdm': 'TQDM',
    'peft': 'PEFT',
    'einops': 'Einops',
}

print()
for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {name}: {version}')
    except ImportError:
        print(f'✗ {name}: NOT INSTALLED')

# Check optional packages
print()
print('Optional packages:')
optional = {
    'mamba_ssm': 'Mamba SSM',
    'bitsandbytes': 'BitsAndBytes',
    'flash_attn': 'Flash Attention',
}

for module, name in optional.items():
    try:
        __import__(module)
        print(f'✓ {name}: installed')
    except ImportError:
        print(f'○ {name}: not installed (optional)')
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review hardware_report.txt for system capabilities" -ForegroundColor White
Write-Host "  2. Check PROJECT_CONTEXT.md for project goals and deliverables" -ForegroundColor White
Write-Host "  3. Run: python check_system.py (to update hardware report)" -ForegroundColor White
Write-Host ""
Write-Host "Virtual environment is active. To deactivate later, run: deactivate" -ForegroundColor Yellow
Write-Host ""
