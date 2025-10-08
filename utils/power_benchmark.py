"""
Power & Efficiency Validation for Itera-Lite
Measures energy consumption and efficiency metrics.
"""

import torch
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.itera_lite import IteraLiteModel
from models.config import IteraLiteConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PowerBenchmark:
    """Benchmark power consumption and efficiency."""
    
    def __init__(self):
        """Initialize power benchmark."""
        self.results = []
        self.process = psutil.Process()
        
        # CPU TDP estimates (Watts) - typical values
        self.cpu_tdp_estimates = {
            'desktop': 65,  # Typical desktop CPU
            'laptop': 15,   # Typical laptop CPU
            'embedded': 5   # Embedded/mobile CPU
        }
    
    def estimate_cpu_power(self, cpu_percent: float, platform: str = 'desktop') -> float:
        """
        Estimate CPU power consumption.
        
        Args:
            cpu_percent: CPU utilization percentage
            platform: Platform type (desktop, laptop, embedded)
            
        Returns:
            Estimated power in Watts
        """
        tdp = self.cpu_tdp_estimates.get(platform, 65)
        # Simple linear model: power ≈ idle_power + (tdp - idle_power) * cpu_percent
        idle_power = tdp * 0.2  # ~20% at idle
        return idle_power + (tdp - idle_power) * (cpu_percent / 100.0)
    
    def benchmark_inference_power(
        self,
        model: torch.nn.Module,
        num_samples: int = 100,
        seq_length: int = 128,
        vocab_size: int = 2000,
        platform: str = 'desktop'
    ) -> Dict:
        """
        Benchmark power consumption during inference.
        
        Args:
            model: Model to benchmark
            num_samples: Number of inference samples
            seq_length: Sequence length
            vocab_size: Vocabulary size
            platform: Platform type
            
        Returns:
            Power benchmarking results
        """
        logger.info(f"Benchmarking power consumption on {platform} platform...")
        
        model.eval()
        sample_input = torch.randint(0, vocab_size, (1, seq_length))
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        power_measurements = []
        latencies = []
        cpu_measurements = []
        memory_measurements = []
        
        for i in range(num_samples):
            # Measure CPU before
            cpu_before = psutil.cpu_percent(interval=0.01)
            memory_before = self.process.memory_info().rss
            
            # Run inference
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(sample_input)
            latency = time.perf_counter() - start_time
            
            # Measure CPU after
            cpu_after = psutil.cpu_percent(interval=0.01)
            memory_after = self.process.memory_info().rss
            
            # Average CPU utilization during inference
            cpu_avg = (cpu_before + cpu_after) / 2
            
            # Estimate power
            power_watts = self.estimate_cpu_power(cpu_avg, platform)
            
            # Energy = Power * Time
            energy_joules = power_watts * latency
            energy_mj = energy_joules * 1000  # Convert to millijoules
            
            power_measurements.append(energy_mj)
            latencies.append(latency * 1000)  # Convert to ms
            cpu_measurements.append(cpu_avg)
            memory_measurements.append((memory_after - memory_before) / 1024 / 1024)  # MB
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{num_samples} samples")
        
        # Calculate statistics
        results = {
            'platform': platform,
            'num_samples': num_samples,
            'seq_length': seq_length,
            'energy_per_token_mj': {
                'mean': np.mean(power_measurements) / seq_length,
                'std': np.std(power_measurements) / seq_length,
                'min': np.min(power_measurements) / seq_length,
                'max': np.max(power_measurements) / seq_length
            },
            'energy_per_inference_mj': {
                'mean': np.mean(power_measurements),
                'std': np.std(power_measurements),
                'min': np.min(power_measurements),
                'max': np.max(power_measurements)
            },
            'latency_ms': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies)
            },
            'cpu_utilization_percent': {
                'mean': np.mean(cpu_measurements),
                'std': np.std(cpu_measurements)
            },
            'memory_delta_mb': {
                'mean': np.mean(memory_measurements),
                'std': np.std(memory_measurements)
            },
            'efficiency_tokens_per_joule': seq_length / (np.mean(power_measurements) / 1000)
        }
        
        logger.info(f"✓ {platform.capitalize()} benchmarking complete")
        logger.info(f"  Energy/token: {results['energy_per_token_mj']['mean']:.4f} mJ")
        logger.info(f"  Latency: {results['latency_ms']['mean']:.2f} ms")
        logger.info(f"  Efficiency: {results['efficiency_tokens_per_joule']:.1f} tokens/J")
        
        return results
    
    def compare_model_variants(
        self,
        model_configs: Dict[str, Dict],
        num_samples: int = 50,
        platform: str = 'laptop'
    ) -> Dict:
        """
        Compare power consumption across model variants.
        
        Args:
            model_configs: Dict of variant_name -> {model, checkpoint_path}
            num_samples: Number of samples per variant
            platform: Platform type
            
        Returns:
            Comparison results
        """
        logger.info(f"\nComparing {len(model_configs)} model variants...")
        
        results = {}
        
        for variant_name, config in model_configs.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking: {variant_name}")
            logger.info(f"{'='*60}")
            
            model = config['model']
            
            # Load checkpoint if provided
            if 'checkpoint_path' in config and Path(config['checkpoint_path']).exists():
                checkpoint = torch.load(config['checkpoint_path'], map_location='cpu', weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded checkpoint: {config['checkpoint_path']}")
            
            model.eval()
            
            # Benchmark
            benchmark_result = self.benchmark_inference_power(
                model=model,
                num_samples=num_samples,
                platform=platform
            )
            
            # Add model info
            benchmark_result['model_info'] = {
                'parameters': sum(p.numel() for p in model.parameters()),
                'size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            }
            
            results[variant_name] = benchmark_result
        
        return results


def plot_power_comparison(results: Dict, output_path: str):
    """Plot power consumption comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    variants = list(results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Energy per token
    ax = axes[0, 0]
    energy_vals = [results[v]['energy_per_token_mj']['mean'] for v in variants]
    energy_errs = [results[v]['energy_per_token_mj']['std'] for v in variants]
    bars = ax.bar(range(len(variants)), energy_vals, yerr=energy_errs, 
                   color=colors[:len(variants)], capsize=5, alpha=0.8)
    ax.set_xlabel('Model Variant', fontweight='bold')
    ax.set_ylabel('Energy (mJ/token)', fontweight='bold')
    ax.set_title('Energy Consumption per Token', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, energy_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Latency
    ax = axes[0, 1]
    latency_vals = [results[v]['latency_ms']['mean'] for v in variants]
    latency_errs = [results[v]['latency_ms']['std'] for v in variants]
    bars = ax.bar(range(len(variants)), latency_vals, yerr=latency_errs,
                   color=colors[:len(variants)], capsize=5, alpha=0.8)
    ax.set_xlabel('Model Variant', fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('Inference Latency', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, latency_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Efficiency (tokens per joule)
    ax = axes[1, 0]
    efficiency_vals = [results[v]['efficiency_tokens_per_joule'] for v in variants]
    bars = ax.bar(range(len(variants)), efficiency_vals,
                   color=colors[:len(variants)], alpha=0.8)
    ax.set_xlabel('Model Variant', fontweight='bold')
    ax.set_ylabel('Tokens/Joule', fontweight='bold')
    ax.set_title('Energy Efficiency', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, efficiency_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # CPU Utilization
    ax = axes[1, 1]
    cpu_vals = [results[v]['cpu_utilization_percent']['mean'] for v in variants]
    bars = ax.bar(range(len(variants)), cpu_vals,
                   color=colors[:len(variants)], alpha=0.8)
    ax.set_xlabel('Model Variant', fontweight='bold')
    ax.set_ylabel('CPU Utilization (%)', fontweight='bold')
    ax.set_title('CPU Utilization', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, cpu_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved power comparison plot to {output_path}")
    plt.close()


def main():
    """Main power validation workflow."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: POWER & EFFICIENCY VALIDATION")
    logger.info("="*80)
    
    # Create benchmark
    benchmark = PowerBenchmark()
    
    # Define model configuration
    config = IteraLiteConfig(
        vocab_size=2000,
        hidden_size=64,
        num_layers=3,
        ssm_state_size=8,
        num_experts=4,
        top_k_experts=2,
        expert_size=32,
        max_seq_length=128
    )
    
    # Define model variants
    model_variants = {
        'INT4 (Simulated)': {
            'model': IteraLiteModel(config),
            'checkpoint_path': 'checkpoints/int4/itera_lite_int4.pt'
        }
    }
    
    # Benchmark across platforms
    all_results = {}
    
    for platform in ['desktop', 'laptop', 'embedded']:
        logger.info(f"\n{'='*80}")
        logger.info(f"PLATFORM: {platform.upper()}")
        logger.info(f"{'='*80}")
        
        platform_results = benchmark.compare_model_variants(
            model_configs=model_variants,
            num_samples=50,
            platform=platform
        )
        
        all_results[platform] = platform_results
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    result_file = output_dir / 'phase6_power_validation.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {result_file}")
    
    # Generate visualizations
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Plot for each platform
    for platform, results in all_results.items():
        plot_path = reports_dir / f'phase6_power_{platform}.png'
        plot_power_comparison(results, str(plot_path))
    
    # Generate report
    generate_power_report(all_results, reports_dir / 'phase6_power_validation.md')
    
    logger.info("\n" + "="*80)
    logger.info("POWER VALIDATION COMPLETE")
    logger.info("="*80)
    
    return all_results


def generate_power_report(results: Dict, output_path: Path):
    """Generate power validation markdown report."""
    lines = []
    
    lines.append("# Phase 6: Power & Efficiency Validation Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")
    
    lines.append("## Executive Summary\n")
    lines.append("This report presents power consumption and energy efficiency metrics for Itera-Lite across desktop, laptop, and embedded platforms.\n")
    
    for platform, platform_results in results.items():
        lines.append(f"## {platform.capitalize()} Platform\n")
        
        for variant, metrics in platform_results.items():
            lines.append(f"### {variant}\n")
            
            lines.append("#### Power Metrics\n")
            lines.append(f"- **Energy per Token:** {metrics['energy_per_token_mj']['mean']:.4f} ± {metrics['energy_per_token_mj']['std']:.4f} mJ")
            lines.append(f"- **Energy per Inference:** {metrics['energy_per_inference_mj']['mean']:.4f} ± {metrics['energy_per_inference_mj']['std']:.4f} mJ")
            lines.append(f"- **Energy Efficiency:** {metrics['efficiency_tokens_per_joule']:.1f} tokens/Joule\n")
            
            lines.append("#### Performance Metrics\n")
            lines.append(f"- **Latency:** {metrics['latency_ms']['mean']:.2f} ± {metrics['latency_ms']['std']:.2f} ms")
            lines.append(f"- **CPU Utilization:** {metrics['cpu_utilization_percent']['mean']:.1f} ± {metrics['cpu_utilization_percent']['std']:.1f} %")
            lines.append(f"- **Memory Delta:** {metrics['memory_delta_mb']['mean']:.2f} ± {metrics['memory_delta_mb']['std']:.2f} MB\n")
            
            lines.append("#### Model Info\n")
            lines.append(f"- **Parameters:** {metrics['model_info']['parameters']:,}")
            lines.append(f"- **Model Size:** {metrics['model_info']['size_mb']:.2f} MB\n")
    
    lines.append("## Visualizations\n")
    for platform in results.keys():
        lines.append(f"### {platform.capitalize()} Platform")
        lines.append(f"![Power {platform}](phase6_power_{platform}.png)\n")
    
    lines.append("---\n")
    lines.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"✓ Power report saved to {output_path}")


if __name__ == "__main__":
    main()
