"""
System diagnostics and hardware profiling for Phase 4
"""

import platform
import sys
import os
import psutil
import torch
from datetime import datetime
from pathlib import Path
import json


def get_cpu_info():
    """Get CPU information"""
    return {
        'processor': platform.processor(),
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'max_frequency_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A',
        'current_frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A',
    }


def get_memory_info():
    """Get memory information"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024 ** 3),
        'available_gb': mem.available / (1024 ** 3),
        'used_gb': mem.used / (1024 ** 3),
        'percent_used': mem.percent
    }


def get_python_info():
    """Get Python environment information"""
    return {
        'version': sys.version,
        'implementation': platform.python_implementation(),
        'compiler': platform.python_compiler(),
        'executable': sys.executable
    }


def get_pytorch_info():
    """Get PyTorch configuration"""
    return {
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }


def get_os_info():
    """Get operating system information"""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'architecture': platform.architecture()[0]
    }


def get_disk_info():
    """Get disk information for workspace"""
    workspace_path = Path.cwd()
    disk = psutil.disk_usage(str(workspace_path))
    return {
        'workspace_path': str(workspace_path),
        'total_gb': disk.total / (1024 ** 3),
        'used_gb': disk.used / (1024 ** 3),
        'free_gb': disk.free / (1024 ** 3),
        'percent_used': disk.percent
    }


def check_dependencies():
    """Check installed dependencies"""
    dependencies = {}
    
    try:
        import numpy
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        dependencies['numpy'] = 'Not installed'
    
    try:
        import matplotlib
        dependencies['matplotlib'] = matplotlib.__version__
    except ImportError:
        dependencies['matplotlib'] = 'Not installed'
    
    try:
        import seaborn
        dependencies['seaborn'] = seaborn.__version__
    except ImportError:
        dependencies['seaborn'] = 'Not installed'
    
    try:
        import transformers
        dependencies['transformers'] = transformers.__version__
    except ImportError:
        dependencies['transformers'] = 'Not installed'
    
    return dependencies


def generate_hardware_report():
    """Generate comprehensive hardware report"""
    
    print("=" * 80)
    print("SYSTEM DIAGNOSTICS - PHASE 4")
    print("=" * 80)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Collect all information
    report = {
        'timestamp': datetime.now().isoformat(),
        'os': get_os_info(),
        'cpu': get_cpu_info(),
        'memory': get_memory_info(),
        'disk': get_disk_info(),
        'python': get_python_info(),
        'pytorch': get_pytorch_info(),
        'dependencies': check_dependencies()
    }
    
    # Display report
    print("\n[OPERATING SYSTEM]")
    print(f"  System: {report['os']['system']} {report['os']['release']}")
    print(f"  Architecture: {report['os']['architecture']}")
    print(f"  Machine: {report['os']['machine']}")
    
    print("\n[CPU]")
    print(f"  Processor: {report['cpu']['processor']}")
    print(f"  Physical cores: {report['cpu']['physical_cores']}")
    print(f"  Logical cores: {report['cpu']['logical_cores']}")
    if report['cpu']['max_frequency_mhz'] != 'N/A':
        print(f"  Max frequency: {report['cpu']['max_frequency_mhz']:.0f} MHz")
        print(f"  Current frequency: {report['cpu']['current_frequency_mhz']:.0f} MHz")
    
    print("\n[MEMORY]")
    print(f"  Total: {report['memory']['total_gb']:.2f} GB")
    print(f"  Available: {report['memory']['available_gb']:.2f} GB")
    print(f"  Used: {report['memory']['used_gb']:.2f} GB ({report['memory']['percent_used']:.1f}%)")
    
    print("\n[DISK]")
    print(f"  Workspace: {report['disk']['workspace_path']}")
    print(f"  Total: {report['disk']['total_gb']:.2f} GB")
    print(f"  Free: {report['disk']['free_gb']:.2f} GB ({100-report['disk']['percent_used']:.1f}%)")
    
    print("\n[PYTHON]")
    print(f"  Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  Implementation: {report['python']['implementation']}")
    print(f"  Executable: {report['python']['executable']}")
    
    print("\n[PYTORCH]")
    print(f"  Version: {report['pytorch']['version']}")
    print(f"  CUDA available: {report['pytorch']['cuda_available']}")
    if report['pytorch']['cuda_available']:
        print(f"  CUDA version: {report['pytorch']['cuda_version']}")
        print(f"  Device: {report['pytorch']['device_name']}")
    else:
        print(f"  Device: CPU-only")
    
    print("\n[DEPENDENCIES]")
    for pkg, ver in report['dependencies'].items():
        print(f"  {pkg}: {ver}")
    
    # Save to JSON
    json_path = Path("results/system_diagnostics.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save to text file
    txt_path = Path("system_hardware_report.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SYSTEM DIAGNOSTICS - PHASE 4\n")
        f.write("=" * 80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        
        f.write("\n[OPERATING SYSTEM]\n")
        f.write(f"  System: {report['os']['system']} {report['os']['release']}\n")
        f.write(f"  Architecture: {report['os']['architecture']}\n")
        f.write(f"  Machine: {report['os']['machine']}\n")
        
        f.write("\n[CPU]\n")
        f.write(f"  Processor: {report['cpu']['processor']}\n")
        f.write(f"  Physical cores: {report['cpu']['physical_cores']}\n")
        f.write(f"  Logical cores: {report['cpu']['logical_cores']}\n")
        if report['cpu']['max_frequency_mhz'] != 'N/A':
            f.write(f"  Max frequency: {report['cpu']['max_frequency_mhz']:.0f} MHz\n")
        
        f.write("\n[MEMORY]\n")
        f.write(f"  Total: {report['memory']['total_gb']:.2f} GB\n")
        f.write(f"  Available: {report['memory']['available_gb']:.2f} GB\n")
        
        f.write("\n[PYTHON]\n")
        f.write(f"  Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")
        
        f.write("\n[PYTORCH]\n")
        f.write(f"  Version: {report['pytorch']['version']}\n")
        f.write(f"  Device: {'GPU' if report['pytorch']['cuda_available'] else 'CPU-only'}\n")
        
        f.write("\n[DEPENDENCIES]\n")
        for pkg, ver in report['dependencies'].items():
            f.write(f"  {pkg}: {ver}\n")
    
    print("\n" + "=" * 80)
    print(f"✓ System diagnostics saved to:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    generate_hardware_report()
