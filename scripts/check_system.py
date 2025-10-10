#!/usr/bin/env python3
"""System hardware and capability checker for Itera-Lite project"""

import sys
import platform
import subprocess
import os

def check_pytorch():
    """Check PyTorch installation and GPU availability"""
    try:
        import torch
        pytorch_info = {
            'installed': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpus': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'name': torch.cuda.get_device_name(i),
                    'vram_total': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                    'capability': f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                }
                pytorch_info['gpus'].append(gpu_info)
        
        # Check for MPS (Apple Silicon)
        pytorch_info['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        return pytorch_info
    except ImportError:
        return {'installed': False}

def get_cpu_info():
    """Get CPU information"""
    cpu_info = {
        'processor': platform.processor(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count(),
    }
    
    # Try to get more detailed CPU info on Windows
    try:
        result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                cpu_info['name'] = lines[1].strip()
    except:
        pass
    
    return cpu_info

def get_memory_info():
    """Get system memory information"""
    try:
        result = subprocess.run(['wmic', 'ComputerSystem', 'get', 'TotalPhysicalMemory'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                total_bytes = int(lines[1].strip())
                total_gb = total_bytes / (1024**3)
                return {'total_gb': total_gb}
    except:
        pass
    return {'total_gb': 'Unknown'}

def generate_report():
    """Generate comprehensive hardware report"""
    report = []
    report.append("=" * 70)
    report.append("ITERA-LITE HARDWARE CAPABILITY REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Python version
    report.append("PYTHON ENVIRONMENT")
    report.append("-" * 70)
    report.append(f"Python Version: {sys.version}")
    report.append(f"Platform: {platform.platform()}")
    report.append(f"Architecture: {platform.architecture()[0]}")
    report.append("")
    
    # CPU info
    report.append("CPU SPECIFICATIONS")
    report.append("-" * 70)
    cpu_info = get_cpu_info()
    report.append(f"Processor: {cpu_info.get('name', cpu_info.get('processor', 'Unknown'))}")
    report.append(f"Architecture: {cpu_info['machine']}")
    report.append(f"CPU Cores: {cpu_info['cpu_count']}")
    report.append("")
    
    # Memory info
    report.append("MEMORY SPECIFICATIONS")
    report.append("-" * 70)
    mem_info = get_memory_info()
    if isinstance(mem_info['total_gb'], float):
        report.append(f"Total RAM: {mem_info['total_gb']:.2f} GB")
    else:
        report.append(f"Total RAM: {mem_info['total_gb']}")
    report.append("")
    
    # PyTorch and GPU info
    report.append("PYTORCH & GPU ACCELERATION")
    report.append("-" * 70)
    pytorch_info = check_pytorch()
    
    if pytorch_info['installed']:
        report.append(f"PyTorch Version: {pytorch_info['version']}")
        report.append(f"CUDA Available: {pytorch_info['cuda_available']}")
        
        if pytorch_info['cuda_available']:
            report.append(f"CUDA Version: {pytorch_info['cuda_version']}")
            report.append(f"Number of GPUs: {pytorch_info['gpu_count']}")
            report.append("")
            
            for i, gpu in enumerate(pytorch_info['gpus']):
                report.append(f"GPU {i}:")
                report.append(f"  Name: {gpu['name']}")
                report.append(f"  VRAM: {gpu['vram_total']:.2f} GB")
                report.append(f"  Compute Capability: {gpu['capability']}")
                report.append("")
        
        report.append(f"MPS (Apple Silicon) Available: {pytorch_info['mps_available']}")
    else:
        report.append("PyTorch: NOT INSTALLED")
        report.append("Status: PyTorch needs to be installed for this project")
    
    report.append("")
    report.append("=" * 70)
    report.append("SUITABILITY ASSESSMENT FOR ITERA-LITE")
    report.append("=" * 70)
    report.append("")
    
    # Assessment
    suitable = True
    issues = []
    recommendations = []
    
    if sys.version_info < (3, 10):
        suitable = False
        issues.append("Python version < 3.10 (upgrade recommended)")
    else:
        recommendations.append("✓ Python 3.10+ detected")
    
    if not pytorch_info['installed']:
        suitable = False
        issues.append("PyTorch not installed")
    else:
        recommendations.append("✓ PyTorch installed")
        
        if pytorch_info['cuda_available']:
            recommendations.append(f"✓ CUDA GPU acceleration available ({pytorch_info['gpu_count']} GPU(s))")
            for i, gpu in enumerate(pytorch_info['gpus']):
                if gpu['vram_total'] >= 4:
                    recommendations.append(f"✓ GPU {i}: {gpu['vram_total']:.1f} GB VRAM (sufficient for prototyping)")
                else:
                    issues.append(f"GPU {i}: Low VRAM ({gpu['vram_total']:.1f} GB) - may limit model size")
        elif pytorch_info['mps_available']:
            recommendations.append("✓ MPS (Apple Silicon) acceleration available")
        else:
            issues.append("No GPU acceleration detected - training will be CPU-only (slower)")
    
    if recommendations:
        report.append("STRENGTHS:")
        for rec in recommendations:
            report.append(f"  {rec}")
        report.append("")
    
    if issues:
        report.append("CONCERNS/LIMITATIONS:")
        for issue in issues:
            report.append(f"  ⚠ {issue}")
        report.append("")
    
    if suitable and pytorch_info.get('cuda_available', False):
        report.append("OVERALL: EXCELLENT - System is well-suited for Itera-Lite prototyping")
        report.append("Recommendation: Proceed with GPU-accelerated development")
    elif suitable and pytorch_info['installed']:
        report.append("OVERALL: GOOD - System can run Itera-Lite (CPU or limited GPU)")
        report.append("Recommendation: Proceed with smaller models or CPU-based prototyping")
    elif pytorch_info['installed']:
        report.append("OVERALL: MARGINAL - System has limitations but can proceed")
        report.append("Recommendation: Focus on minimal models and careful resource management")
    else:
        report.append("OVERALL: NOT READY - Critical dependencies missing")
        report.append("Recommendation: Install required dependencies before proceeding")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report), pytorch_info

if __name__ == "__main__":
    report_text, pytorch_info = generate_report()
    
    # Print to console
    print(report_text)
    
    # Save to file
    with open('hardware_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n[OK] Report saved to hardware_report.txt")
