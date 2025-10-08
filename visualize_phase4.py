"""
Phase 4 visualization utilities
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_results():
    """Load all Phase 4 results"""
    results = {}
    
    files = {
        'vocab': 'results/vocab_optimization.json',
        'quant': 'results/quantization_results.json',
        'distill': 'results/distillation_results.json',
        'phase3': 'results/comparison_tiny.json'
    }
    
    for key, filepath in files.items():
        try:
            with open(filepath, 'r') as f:
                results[key] = json.load(f)
        except FileNotFoundError:
            results[key] = {}
    
    return results


def plot_compression_progression():
    """Plot cumulative compression across phases"""
    results = load_results()
    
    # Get base metrics from Phase 3
    if results['phase3']:
        base_params = results['phase3'].get('itera_lite', {}).get('params', {}).get('total', 1886496)
        base_flops = results['phase3'].get('itera_lite', {}).get('flops', {}).get('flops_per_token', 327680)
    else:
        base_params = 1886496
        base_flops = 327680
    
    # Calculate compression stages
    stages = ['Phase 3\nBaseline']
    params_progression = [base_params]
    flops_progression = [base_flops]
    
    # Vocabulary optimization
    if results['vocab']:
        vocab_params = results['vocab'][0].get('params', base_params)
        params_progression.append(vocab_params)
        flops_progression.append(base_flops * (vocab_params / base_params))
        stages.append('Vocab\nOptimization')
    
    # Quantization
    if results['quant']:
        quant_ratio = results['quant'].get('int8', {}).get('compression_ratio', 1.0)
        params_progression.append(params_progression[-1] / quant_ratio)
        flops_progression.append(flops_progression[-1])  # FLOPs don't change with quantization
        stages.append('INT8\nQuantization')
    
    # Distillation
    if results['distill']:
        student_params = results['distill'].get('student', {}).get('params', params_progression[-1])
        student_flops = results['distill'].get('student', {}).get('flops_per_token', flops_progression[-1])
        params_progression.append(student_params)
        flops_progression.append(student_flops)
        stages.append('Knowledge\nDistillation')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot parameters
    x = range(len(stages))
    ax1.plot(x, [p/1e6 for p in params_progression], marker='o', linewidth=2, markersize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylabel('Parameters (M)', fontsize=12)
    ax1.set_title('Parameter Reduction Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add compression ratios
    for i in range(1, len(params_progression)):
        ratio = params_progression[0] / params_progression[i]
        ax1.annotate(f'{ratio:.1f}x',
                    xy=(i, params_progression[i]/1e6),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='bold')
    
    # Plot FLOPs
    ax2.plot(x, [f/1e3 for f in flops_progression], marker='s', linewidth=2, markersize=8, color='orange')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.set_ylabel('FLOPs/Token (K)', fontsize=12)
    ax2.set_title('FLOPs Reduction Progression', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add compression ratios
    for i in range(1, len(flops_progression)):
        ratio = flops_progression[0] / flops_progression[i]
        ax2.annotate(f'{ratio:.1f}x',
                    xy=(i, flops_progression[i]/1e3),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    color='orange')
    
    plt.tight_layout()
    plt.savefig('reports/phase4_compression_progression.png', dpi=300, bbox_inches='tight')
    print("✓ Saved compression progression plot")
    plt.close()


def plot_efficiency_tradeoffs():
    """Plot efficiency vs quality tradeoffs"""
    results = load_results()
    
    if not results['distill']:
        print("⚠ Distillation results not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get metrics
    teacher = results['distill'].get('teacher', {})
    student = results['distill'].get('student', {})
    
    models = ['Teacher\n(Tiny)', 'Student\n(Micro)']
    params = [teacher.get('params', 0)/1e6, student.get('params', 0)/1e6]
    throughput = [teacher.get('throughput', 0), student.get('throughput', 0)]
    
    # Create scatter plot
    colors = ['#2E86AB', '#A23B72']
    for i, model in enumerate(models):
        ax.scatter(params[i], throughput[i], s=500, alpha=0.7, color=colors[i], label=model)
        ax.annotate(model,
                   xy=(params[i], throughput[i]),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=11,
                   fontweight='bold')
    
    ax.set_xlabel('Parameters (M)', fontsize=12)
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax.set_title('Model Size vs Throughput', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('reports/phase4_efficiency_tradeoffs.png', dpi=300, bbox_inches='tight')
    print("✓ Saved efficiency tradeoffs plot")
    plt.close()


def plot_quantization_comparison():
    """Plot quantization impact"""
    results = load_results()
    
    if not results['quant']:
        print("⚠ Quantization results not available")
        return
    
    orig = results['quant'].get('original', {})
    int8 = results['quant'].get('int8', {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Memory comparison
    models = ['FP32\nOriginal', 'INT8\nQuantized']
    memory = [orig.get('size_mb', 0), int8.get('size_mb', 0)]
    
    bars1 = ax1.bar(models, memory, color=['#3A86FF', '#FB5607'])
    ax1.set_ylabel('Model Size (MB)', fontsize=12)
    ax1.set_title('Memory Footprint', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars1, memory):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} MB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add compression ratio
    if int8.get('compression_ratio'):
        ax1.text(0.5, max(memory) * 0.9,
                f"{int8['compression_ratio']:.1f}x\nCompression",
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Speed comparison
    speed = [1.0, int8.get('speedup', 1.0)]  # Normalized to original
    
    bars2 = ax2.bar(models, speed, color=['#3A86FF', '#FB5607'])
    ax2.set_ylabel('Relative Speed', fontsize=12)
    ax2.set_title('Inference Speed', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(speed) * 1.3)
    
    # Add values on bars
    for bar, val in zip(bars2, speed):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/phase4_quantization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved quantization comparison plot")
    plt.close()


def generate_all_visualizations():
    """Generate all Phase 4 visualizations"""
    print("\nGenerating Phase 4 visualizations...")
    
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    plot_compression_progression()
    plot_efficiency_tradeoffs()
    plot_quantization_comparison()
    
    print("\n✓ All visualizations generated!")
    print("  Saved to reports/phase4_*.png")


if __name__ == "__main__":
    generate_all_visualizations()
