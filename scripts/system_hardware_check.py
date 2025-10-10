#!/usr/bin/env python3
"""
Phase 7: Advanced Optimization - System & Hardware Diagnostics

Comprehensive system check for Phase 7 requirements:
- CUDA/cuDNN availability and versions
- GPU detection, VRAM, compute capability
- CPU features (AVX512, ARM NEON)
- PyTorch configuration
- Matrix-multiply benchmarks
- Dependency availability

Outputs:
- reports/phase7_hardware_check.json (machine-readable)
- Console summary (human-readable)
"""

import sys
import json
import platform
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import psutil


def get_cuda_info():
    """Get CUDA and cuDNN information."""
    cuda_info = {
        "available": torch.cuda.is_available(),
        "version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        cuda_info["version"] = torch.version.cuda
        cuda_info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        cuda_info["device_count"] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": props.multi_processor_count,
                "max_threads_per_block": props.max_threads_per_block
            }
            cuda_info["devices"].append(device_info)
    
    return cuda_info


def get_cpu_info():
    """Get CPU information and feature flags."""
    cpu_info = {
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_frequency_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "features": {}
    }
    
    # Check for AVX512 and other SIMD features
    try:
        # Try to import cpuinfo for detailed CPU features
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get('flags', [])
        
        cpu_info["features"] = {
            "avx": 'avx' in flags,
            "avx2": 'avx2' in flags,
            "avx512f": 'avx512f' in flags,
            "avx512_vnni": 'avx512_vnni' in flags,
            "fma": 'fma' in flags,
            "sse4_2": 'sse4_2' in flags
        }
    except ImportError:
        # Fallback: check PyTorch CPU capabilities
        cpu_info["features"] = {
            "mkl_available": torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else False,
            "mkldnn_available": torch.backends.mkldnn.is_available() if hasattr(torch.backends, 'mkldnn') else False
        }
    
    return cpu_info


def get_memory_info():
    """Get system memory information."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "percent_used": mem.percent
    }


def get_pytorch_info():
    """Get PyTorch configuration."""
    return {
        "version": torch.__version__,
        "cuda_compiled": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled if torch.backends.cudnn.is_available() else False,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "num_threads": torch.get_num_threads(),
        "mkl_available": torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else False
    }


def benchmark_matmul(device='cpu', dtype=torch.float32, size=2048, iterations=100):
    """Benchmark matrix multiplication performance."""
    torch.manual_seed(42)
    
    # Create random matrices
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(10):
        _ = torch.mm(a, b)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    start = time.perf_counter()
    
    for _ in range(iterations):
        c = torch.mm(a, b)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    gflops = (2 * size**3) / (avg_time_ms / 1000) / 1e9  # 2*N^3 FLOPs for matrix multiply
    
    return {
        "device": str(device),
        "dtype": str(dtype),
        "matrix_size": size,
        "iterations": iterations,
        "avg_time_ms": round(avg_time_ms, 4),
        "gflops": round(gflops, 2)
    }


def check_dependencies():
    """Check availability of Phase 7 dependencies."""
    dependencies = {
        "torch": {"available": False, "version": None},
        "onnxruntime": {"available": False, "version": None},
        "onnxruntime_gpu": {"available": False, "version": None},
        "bitsandbytes": {"available": False, "version": None},
        "optimum": {"available": False, "version": None},
        "transformers": {"available": False, "version": None},
        "torch_pruning": {"available": False, "version": None}
    }
    
    # Check torch (already imported)
    dependencies["torch"]["available"] = True
    dependencies["torch"]["version"] = torch.__version__
    
    # Check other packages
    packages_to_check = [
        "onnxruntime",
        "bitsandbytes",
        "optimum",
        "transformers"
    ]
    
    for pkg in packages_to_check:
        try:
            module = __import__(pkg)
            dependencies[pkg]["available"] = True
            dependencies[pkg]["version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            pass
    
    # Check onnxruntime-gpu separately
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers:
            dependencies["onnxruntime_gpu"]["available"] = True
            dependencies["onnxruntime_gpu"]["version"] = ort.__version__
    except:
        pass
    
    # Check torch_pruning
    try:
        import torch_pruning
        dependencies["torch_pruning"]["available"] = True
        dependencies["torch_pruning"]["version"] = getattr(torch_pruning, "__version__", "unknown")
    except ImportError:
        pass
    
    return dependencies


def assess_hardware_capability(diagnostics):
    """Assess hardware capability for Phase 7 tasks."""
    assessment = {
        "overall_status": "EXCELLENT",
        "gpu_status": "NOT_AVAILABLE",
        "cpu_status": "GOOD",
        "memory_status": "SUFFICIENT",
        "recommendations": [],
        "capabilities": []
    }
    
    # GPU assessment
    if diagnostics["cuda"]["available"] and diagnostics["cuda"]["device_count"] > 0:
        gpu = diagnostics["cuda"]["devices"][0]
        vram_gb = gpu["total_memory_gb"]
        compute = gpu["compute_capability"]
        
        if vram_gb >= 8:
            assessment["gpu_status"] = "EXCELLENT"
            assessment["capabilities"].append(f"GPU acceleration available ({gpu['name']}, {vram_gb}GB VRAM)")
        elif vram_gb >= 4:
            assessment["gpu_status"] = "GOOD"
            assessment["capabilities"].append(f"GPU available but limited VRAM ({vram_gb}GB)")
            assessment["recommendations"].append("GPU VRAM < 8GB may limit batch sizes for training")
        else:
            assessment["gpu_status"] = "LIMITED"
            assessment["recommendations"].append("GPU VRAM < 4GB - recommend CPU-only mode")
    else:
        assessment["gpu_status"] = "NOT_AVAILABLE"
        assessment["recommendations"].append("No GPU detected - all operations will run on CPU")
        assessment["capabilities"].append("CPU-only mode (slower training/inference)")
    
    # CPU assessment
    cpu = diagnostics["cpu"]
    if cpu["physical_cores"] >= 8:
        assessment["cpu_status"] = "EXCELLENT"
    elif cpu["physical_cores"] >= 4:
        assessment["cpu_status"] = "GOOD"
    else:
        assessment["cpu_status"] = "LIMITED"
        assessment["recommendations"].append("CPU cores < 4 - recommend upgrading for faster processing")
    
    # Check for AVX512
    if diagnostics["cpu"]["features"].get("avx512f"):
        assessment["capabilities"].append("AVX512 instruction set available (optimal for INT4/INT8)")
    elif diagnostics["cpu"]["features"].get("avx2"):
        assessment["capabilities"].append("AVX2 instruction set available (good for quantization)")
    
    # Memory assessment
    mem_gb = diagnostics["memory"]["total_gb"]
    if mem_gb >= 16:
        assessment["memory_status"] = "EXCELLENT"
    elif mem_gb >= 8:
        assessment["memory_status"] = "SUFFICIENT"
    else:
        assessment["memory_status"] = "LIMITED"
        assessment["recommendations"].append("RAM < 8GB may limit model size and batch processing")
    
    # Overall status
    if assessment["gpu_status"] in ["EXCELLENT", "GOOD"] and assessment["cpu_status"] in ["EXCELLENT", "GOOD"]:
        assessment["overall_status"] = "EXCELLENT"
    elif assessment["gpu_status"] == "NOT_AVAILABLE" and assessment["cpu_status"] in ["EXCELLENT", "GOOD"]:
        assessment["overall_status"] = "GOOD (CPU-only)"
    else:
        assessment["overall_status"] = "LIMITED"
    
    return assessment


def generate_upgrade_recommendations(assessment):
    """Generate hardware upgrade recommendations if needed."""
    recommendations = {
        "status": assessment["overall_status"],
        "required_for_optimal_phase7": [],
        "optional_improvements": []
    }
    
    if assessment["gpu_status"] == "NOT_AVAILABLE":
        recommendations["required_for_optimal_phase7"].append({
            "component": "GPU",
            "recommendation": "NVIDIA GPU with CUDA support",
            "minimum_specs": "GTX 1660 Ti / RTX 3050 (6GB VRAM)",
            "recommended_specs": "RTX 3060 / RTX 4060 (8-12GB VRAM)",
            "optimal_specs": "RTX 3090 / RTX 4090 (24GB VRAM)",
            "reason": "Native INT4 quantization and mixed-precision training benefit significantly from GPU acceleration"
        })
    
    if assessment["gpu_status"] == "LIMITED":
        recommendations["optional_improvements"].append({
            "component": "GPU VRAM",
            "recommendation": "Upgrade to GPU with >= 8GB VRAM",
            "reason": "Current VRAM may limit batch sizes and larger model variants"
        })
    
    if assessment["cpu_status"] == "LIMITED":
        recommendations["optional_improvements"].append({
            "component": "CPU",
            "recommendation": "Upgrade to CPU with >= 8 cores",
            "reason": "Parallel processing for data loading and CPU-fallback operations"
        })
    
    if assessment["memory_status"] == "LIMITED":
        recommendations["required_for_optimal_phase7"].append({
            "component": "RAM",
            "recommendation": "Upgrade to >= 16GB RAM",
            "minimum_specs": "16GB DDR4",
            "recommended_specs": "32GB DDR4/DDR5",
            "reason": "Sufficient memory for model checkpoints, batching, and system operations"
        })
    
    # Software recommendations
    recommendations["software_requirements"] = {
        "cuda": "CUDA >= 12.0 (if using GPU)",
        "cudnn": "cuDNN >= 8.9 (if using GPU)",
        "pytorch": "PyTorch >= 2.0 with CUDA support",
        "python": "Python >= 3.10"
    }
    
    return recommendations


def main():
    """Run comprehensive hardware diagnostics."""
    print("=" * 80)
    print("Phase 7: Advanced Optimization - System & Hardware Diagnostics")
    print("=" * 80)
    print()
    
    # Collect diagnostics
    print("Collecting system information...")
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        },
        "cuda": get_cuda_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "pytorch": get_pytorch_info(),
        "dependencies": check_dependencies(),
        "benchmarks": {}
    }
    
    # Run benchmarks
    print("\nRunning performance benchmarks...")
    
    # CPU FP32
    print("  - CPU FP32 matrix multiply...")
    diagnostics["benchmarks"]["cpu_fp32"] = benchmark_matmul(device='cpu', dtype=torch.float32, size=1024, iterations=50)
    
    # CPU FP16 (if supported)
    try:
        print("  - CPU FP16 matrix multiply...")
        diagnostics["benchmarks"]["cpu_fp16"] = benchmark_matmul(device='cpu', dtype=torch.float16, size=1024, iterations=50)
    except:
        diagnostics["benchmarks"]["cpu_fp16"] = {"error": "FP16 not supported on CPU"}
    
    # GPU benchmarks if available
    if diagnostics["cuda"]["available"]:
        print("  - GPU FP32 matrix multiply...")
        diagnostics["benchmarks"]["gpu_fp32"] = benchmark_matmul(device='cuda', dtype=torch.float32, size=2048, iterations=100)
        
        print("  - GPU FP16 matrix multiply...")
        diagnostics["benchmarks"]["gpu_fp16"] = benchmark_matmul(device='cuda', dtype=torch.float16, size=2048, iterations=100)
    
    # Hardware assessment
    print("\nAssessing hardware capabilities...")
    diagnostics["assessment"] = assess_hardware_capability(diagnostics)
    
    # Generate recommendations if needed
    diagnostics["upgrade_recommendations"] = generate_upgrade_recommendations(diagnostics["assessment"])
    
    # Save to JSON
    output_path = Path("reports/phase7_hardware_check.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    print(f"\n✓ Diagnostics saved to {output_path}")
    
    # Print human-readable summary
    print("\n" + "=" * 80)
    print("HARDWARE CAPABILITY SUMMARY")
    print("=" * 80)
    
    print(f"\n🖥️  System: {diagnostics['platform']['system']} {diagnostics['platform']['release']}")
    print(f"🐍 Python: {diagnostics['platform']['python_version']}")
    print(f"🔥 PyTorch: {diagnostics['pytorch']['version']}")
    
    print(f"\n💾 Memory: {diagnostics['memory']['total_gb']} GB total, {diagnostics['memory']['available_gb']} GB available")
    print(f"🔧 CPU: {diagnostics['cpu']['physical_cores']} cores ({diagnostics['cpu']['logical_cores']} threads)")
    
    # GPU info
    if diagnostics["cuda"]["available"]:
        print(f"\n🎮 GPU: AVAILABLE")
        print(f"   CUDA: {diagnostics['cuda']['version']}")
        print(f"   cuDNN: {diagnostics['cuda']['cudnn_version']}")
        for gpu in diagnostics["cuda"]["devices"]:
            print(f"   Device {gpu['index']}: {gpu['name']}")
            print(f"     - Compute Capability: {gpu['compute_capability']}")
            print(f"     - VRAM: {gpu['total_memory_gb']} GB")
    else:
        print(f"\n🎮 GPU: NOT AVAILABLE (CPU-only mode)")
    
    # CPU features
    print(f"\n⚡ CPU Features:")
    for feature, available in diagnostics["cpu"]["features"].items():
        status = "✓" if available else "✗"
        print(f"   {status} {feature.upper()}")
    
    # Benchmarks
    print(f"\n📊 Performance Benchmarks:")
    for bench_name, bench_result in diagnostics["benchmarks"].items():
        if "error" not in bench_result:
            print(f"   {bench_name.upper()}: {bench_result['gflops']} GFLOPS ({bench_result['avg_time_ms']:.2f}ms)")
    
    # Assessment
    print(f"\n🎯 Hardware Assessment: {diagnostics['assessment']['overall_status']}")
    print(f"   GPU Status: {diagnostics['assessment']['gpu_status']}")
    print(f"   CPU Status: {diagnostics['assessment']['cpu_status']}")
    print(f"   Memory Status: {diagnostics['assessment']['memory_status']}")
    
    print(f"\n✨ Capabilities:")
    for cap in diagnostics["assessment"]["capabilities"]:
        print(f"   • {cap}")
    
    if diagnostics["assessment"]["recommendations"]:
        print(f"\n⚠️  Recommendations:")
        for rec in diagnostics["assessment"]["recommendations"]:
            print(f"   • {rec}")
    
    # Dependencies
    print(f"\n📦 Phase 7 Dependencies:")
    for dep, info in diagnostics["dependencies"].items():
        status = "✓" if info["available"] else "✗"
        version = f"v{info['version']}" if info["version"] else ""
        print(f"   {status} {dep} {version}")
    
    missing_deps = [dep for dep, info in diagnostics["dependencies"].items() if not info["available"]]
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print(f"   These will be installed in the environment setup step.")
    
    # Upgrade recommendations
    if diagnostics["upgrade_recommendations"]["required_for_optimal_phase7"]:
        print(f"\n⚠️  HARDWARE UPGRADE RECOMMENDATIONS:")
        for rec in diagnostics["upgrade_recommendations"]["required_for_optimal_phase7"]:
            print(f"\n   {rec['component']}:")
            print(f"     Recommendation: {rec['recommendation']}")
            if "minimum_specs" in rec:
                print(f"     Minimum: {rec['minimum_specs']}")
            if "recommended_specs" in rec:
                print(f"     Recommended: {rec['recommended_specs']}")
            print(f"     Reason: {rec['reason']}")
    
    print("\n" + "=" * 80)
    print("✅ Hardware diagnostics complete!")
    print("=" * 80)
    
    return diagnostics


if __name__ == "__main__":
    main()
