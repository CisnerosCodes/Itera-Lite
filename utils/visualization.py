"""
Visualization utilities for training and benchmarking results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_curves(csv_path: str, save_path: Optional[str] = None):
    """Plot training and validation loss curves"""
    # Load data
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax = axes[0]
    if 'train_loss' in df.columns:
        train_df = df[df['train_loss'].notna()]
        ax.plot(train_df['step'], train_df['train_loss'], 
               label='Train Loss', alpha=0.7)
    
    if 'val_loss' in df.columns:
        val_df = df[df['val_loss'].notna()]
        ax.plot(val_df['step'], val_df['val_loss'], 
               label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate curve
    ax = axes[1]
    if 'lr' in df.columns:
        lr_df = df[df['lr'].notna()]
        ax.plot(lr_df['step'], lr_df['lr'])
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(
    results: List[Dict],
    metrics: List[str] = ['parameters.total', 'flops.flops_per_token', 
                          'inference_speed.throughput_tokens_per_sec'],
    save_path: Optional[str] = None
):
    """Create comparison bar charts for multiple models"""
    
    model_names = [r['model_name'] for r in results]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        # Extract metric values
        values = []
        for result in results:
            # Navigate nested dict
            parts = metric.split('.')
            value = result
            for part in parts:
                value = value[part]
            values.append(value)
        
        # Plot
        ax = axes[idx]
        bars = ax.bar(model_names, values, alpha=0.7, color=sns.color_palette("husl", len(model_names)))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 1000:
                label = f'{height:,.0f}'
            else:
                label = f'{height:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=9)
        
        # Labels
        metric_name = metric.split('.')[-1].replace('_', ' ').title()
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_efficiency_gains(
    baseline_results: Dict,
    compared_results: Dict,
    save_path: Optional[str] = None
):
    """Plot efficiency gains visualization"""
    
    baseline_name = baseline_results['model_name']
    compared_name = compared_results['model_name']
    
    # Calculate ratios
    params_ratio = baseline_results['parameters']['total'] / compared_results['parameters']['total']
    flops_ratio = baseline_results['flops']['flops_per_token'] / compared_results['flops']['flops_per_token']
    speed_ratio = compared_results['inference_speed']['throughput_tokens_per_sec'] / baseline_results['inference_speed']['throughput_tokens_per_sec']
    memory_ratio = baseline_results['memory']['total_memory_mb'] / compared_results['memory']['total_memory_mb']
    
    # Create radar/spider plot
    categories = ['Params\nReduction', 'FLOPs\nReduction', 'Speed\nImprovement', 'Memory\nReduction']
    values = [params_ratio, flops_ratio, speed_ratio, memory_ratio]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = values + values[:1]  # Complete the circle
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=f'{compared_name} vs {baseline_name}')
    ax.fill(angles, values, alpha=0.25)
    
    # Add reference line at 1.0
    ax.plot(angles, [1.0] * len(angles), 'k--', alpha=0.3, linewidth=1, label='Baseline (1.0x)')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_title(f'Efficiency Gains: {compared_name} vs {baseline_name}', 
                 size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved efficiency gains plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_metrics(results_dir: str = 'results', output_dir: str = 'reports'):
    """Generate all visualization plots from results"""
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Plot training curves
    print("\n1. Plotting training curves...")
    for csv_file in results_path.glob('*_metrics.csv'):
        model_name = csv_file.stem.replace('_metrics', '')
        save_path = output_path / f'{model_name}_training_curves.png'
        try:
            plot_training_curves(str(csv_file), str(save_path))
        except Exception as e:
            print(f"   Warning: Could not plot {csv_file.name}: {e}")
    
    # Load benchmark results
    print("\n2. Loading benchmark results...")
    benchmark_results = []
    for json_file in results_path.glob('*_benchmark.json'):
        with open(json_file, 'r') as f:
            result = json.load(f)
            benchmark_results.append(result)
            print(f"   Loaded: {json_file.name}")
    
    if len(benchmark_results) >= 2:
        # Model comparison
        print("\n3. Creating comparison plots...")
        save_path = output_path / 'model_comparison.png'
        plot_model_comparison(benchmark_results, save_path=str(save_path))
        
        # Efficiency gains (using first as baseline)
        print("\n4. Creating efficiency gains plot...")
        save_path = output_path / 'efficiency_gains.png'
        plot_efficiency_gains(
            benchmark_results[0],
            benchmark_results[1],
            save_path=str(save_path)
        )
    
    print("\n" + "=" * 70)
    print(f"Visualizations saved to: {output_path}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    # This will be run after training completes
    print("Run this after training to generate visualizations:")
    print("  python -c \"from utils.visualization import plot_all_metrics; plot_all_metrics()\"")
