"""
Phase 6: Real-World Validation and Adaptive Learning
Validates Itera-Lite performance on real datasets and implements adaptive systems.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import argparse
from datetime import datetime

from models.itera_lite import IteraLiteModel, IteraLiteConfig
from utils.real_world_validation import (
    download_wikitext2,
    download_tinystories,
    compare_model_variants
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def task_real_world_validation(args) -> Dict:
    """
    Task 1: Real-World Dataset Validation
    Evaluate on WikiText-2 and TinyStories with different quantization levels.
    
    Returns:
        Validation results
    """
    logger.info("\n" + "=" * 80)
    logger.info("TASK 1: Real-World Dataset Validation")
    logger.info("=" * 80)
    
    # Download datasets
    logger.info("\n1. Downloading datasets...")
    wikitext_path = download_wikitext2()
    tinystories_path = download_tinystories()
    
    # Define model variants to compare
    logger.info("\n2. Preparing model variants...")
    
    # Use micro model for faster evaluation - match the saved checkpoint config
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
    
    model_variants = {
        'FP32 (Distilled)': 'checkpoints/distilled/itera_lite_micro.pt',
        'INT8 (Dynamic)': 'checkpoints/quantized/itera_lite_quantized.pt',
        'INT4 (Simulated)': 'checkpoints/int4/itera_lite_int4.pt'
    }
    
    # Check which models exist
    available_variants = {}
    for name, path in model_variants.items():
        if Path(path).exists():
            available_variants[name] = path
            logger.info(f"  ✓ {name}: {path}")
        else:
            logger.warning(f"  ✗ {name}: {path} not found")
    
    if not available_variants:
        logger.error("No model variants found! Please run Phase 4 and Phase 5 first.")
        return {}
    
    # Evaluate on WikiText-2
    logger.info("\n3. Evaluating on WikiText-2...")
    wikitext_results = compare_model_variants(
        model_paths=available_variants,
        model_class=IteraLiteModel,
        config=config,
        dataset_name='wikitext2',
        dataset_path=wikitext_path,
        batch_size=args.batch_size,
        max_batches=args.max_batches
    )
    
    # Evaluate on TinyStories
    logger.info("\n4. Evaluating on TinyStories...")
    tinystories_results = compare_model_variants(
        model_paths=available_variants,
        model_class=IteraLiteModel,
        config=config,
        dataset_name='tinystories',
        dataset_path=tinystories_path,
        batch_size=args.batch_size,
        max_batches=args.max_batches
    )
    
    # Calculate quality degradation
    logger.info("\n5. Analyzing quality vs compression trade-offs...")
    
    def get_baseline_ppl(results):
        """Get FP32 perplexity as baseline"""
        for variant in ['FP32 (Distilled)', 'FP32', 'Baseline']:
            if variant in results:
                return results[variant]['perplexity']
        return list(results.values())[0]['perplexity']
    
    analysis = {
        'wikitext2': {},
        'tinystories': {}
    }
    
    # WikiText-2 analysis
    baseline_ppl_wiki = get_baseline_ppl(wikitext_results)
    for variant, metrics in wikitext_results.items():
        degradation = ((metrics['perplexity'] - baseline_ppl_wiki) / baseline_ppl_wiki) * 100
        analysis['wikitext2'][variant] = {
            'perplexity': metrics['perplexity'],
            'degradation_pct': degradation,
            'avg_loss': metrics['avg_loss']
        }
        logger.info(f"  {variant}: PPL={metrics['perplexity']:.2f}, "
                   f"Degradation={degradation:+.1f}%")
    
    # TinyStories analysis
    baseline_ppl_tiny = get_baseline_ppl(tinystories_results)
    for variant, metrics in tinystories_results.items():
        degradation = ((metrics['perplexity'] - baseline_ppl_tiny) / baseline_ppl_tiny) * 100
        analysis['tinystories'][variant] = {
            'perplexity': metrics['perplexity'],
            'degradation_pct': degradation,
            'avg_loss': metrics['avg_loss']
        }
        logger.info(f"  {variant}: PPL={metrics['perplexity']:.2f}, "
                   f"Degradation={degradation:+.1f}%")
    
    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': args.batch_size,
            'max_batches': args.max_batches,
            'model_variants': list(available_variants.keys())
        },
        'wikitext2': wikitext_results,
        'tinystories': tinystories_results,
        'analysis': analysis
    }
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'phase6_real_world_validation.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("REAL-WORLD VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Models evaluated: {len(available_variants)}")
    logger.info(f"Datasets: WikiText-2, TinyStories")
    logger.info(f"Total batches per model: {args.max_batches * 2}")
    
    logger.info("\nKey Findings:")
    logger.info("  WikiText-2:")
    for variant in available_variants.keys():
        if variant in analysis['wikitext2']:
            deg = analysis['wikitext2'][variant]['degradation_pct']
            ppl = analysis['wikitext2'][variant]['perplexity']
            logger.info(f"    {variant}: PPL={ppl:.2f} ({deg:+.1f}%)")
    
    logger.info("  TinyStories:")
    for variant in available_variants.keys():
        if variant in analysis['tinystories']:
            deg = analysis['tinystories'][variant]['degradation_pct']
            ppl = analysis['tinystories'][variant]['perplexity']
            logger.info(f"    {variant}: PPL={ppl:.2f} ({deg:+.1f}%)")
    
    return results


def task_onnx_export(args) -> Dict:
    """
    Task 2: ONNX Export & Runtime Benchmarking
    Complete ONNX export and compare with TorchScript performance.
    
    Returns:
        ONNX export results
    """
    logger.info("\n" + "=" * 80)
    logger.info("TASK 2: ONNX Export & Runtime Benchmarking")
    logger.info("=" * 80)
    
    try:
        import onnx
        import onnxruntime
        logger.info("✓ ONNX packages installed")
    except ImportError:
        logger.error("✗ ONNX packages not installed")
        logger.info("Please install: pip install onnx onnxruntime")
        return {'status': 'skipped', 'reason': 'ONNX packages not installed'}
    
    from utils.export import ModelExporter
    
    logger.info("\n1. Loading model...")
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
    
    model = IteraLiteModel(config)
    
    # Try multiple checkpoint paths
    checkpoint_paths = [
        'checkpoints/distilled/itera_lite_micro.pt',
        'checkpoints/int4/itera_lite_int4.pt',
        'checkpoints/quantized/itera_lite_quantized.pt'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            logger.info(f"Using checkpoint: {checkpoint_path}")
            break
    
    if not checkpoint_path:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return {'status': 'error', 'reason': 'Checkpoint not found'}
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    logger.info("\n2. Exporting to ONNX...")
    exporter = ModelExporter(model=model, model_name='itera_lite_micro')
    
    export_results = exporter.export_both(
        seq_length=128,
        output_dir='deployment/models'
    )
    
    logger.info("\n3. Benchmarking ONNX Runtime...")
    onnx_path = export_results.get('onnx')
    torchscript_path = export_results.get('torchscript')
    
    from utils.export import benchmark_exported_model
    
    benchmarks = benchmark_exported_model(
        onnx_path=onnx_path if onnx_path and Path(onnx_path).exists() else None,
        torchscript_path=torchscript_path if torchscript_path and Path(torchscript_path).exists() else None,
        num_runs=100,
        seq_length=128,
        vocab_size=2000
    )
    
    if benchmarks.get('onnx'):
        logger.info(f"  ONNX Runtime: {benchmarks['onnx']['mean_latency_ms']:.2f} ± {benchmarks['onnx']['std_latency_ms']:.2f} ms")
    if benchmarks.get('torchscript'):
        logger.info(f"  TorchScript: {benchmarks['torchscript']['mean_latency_ms']:.2f} ± {benchmarks['torchscript']['std_latency_ms']:.2f} ms")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'export': export_results,
        'benchmarks': benchmarks
    }
    
    output_file = Path('results/phase6_onnx_export.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 6: Real-World Validation')
    parser.add_argument('--task', type=str, choices=['validation', 'onnx', 'all'],
                       default='all', help='Task to run')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for validation')
    parser.add_argument('--max-batches', type=int, default=100,
                       help='Maximum batches to evaluate per dataset')
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: REAL-WORLD VALIDATION & ADAPTIVE LEARNING")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    if args.task in ['validation', 'all']:
        results['validation'] = task_real_world_validation(args)
    
    if args.task in ['onnx', 'all']:
        results['onnx'] = task_onnx_export(args)
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6 EXECUTION COMPLETE")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
