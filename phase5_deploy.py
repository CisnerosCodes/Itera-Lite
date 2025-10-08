"""
Phase 5: Deployment, Validation & Edge Optimization
Main orchestration script for all Phase 5 tasks.
"""

import torch
import argparse
import logging
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.itera_lite import IteraLiteModel
from models.config import IteraLiteConfig
from utils.export import export_checkpoint
from utils.optimized_kernels import benchmark_ssm_kernels, profile_ssm_operations
from utils.advanced_quantization import AdvancedQuantizer, INT4Quantizer
from utils.benchmark import ModelBenchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def task_export_models(args):
    """Task 1: Export models to ONNX and TorchScript."""
    logger.info("=" * 80)
    logger.info("TASK: Export Models to ONNX & TorchScript")
    logger.info("=" * 80)
    
    # Get best checkpoint from Phase 4
    checkpoint_path = args.checkpoint or "checkpoints/distilled/itera_lite_micro_distilled.pt"
    
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        # Try alternative checkpoint
        checkpoint_path = "checkpoints/vocab_2000/itera_lite_vocab2000_best.pt"
        if not Path(checkpoint_path).exists():
            logger.error("No suitable checkpoint found")
            return
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create config (use micro config for distilled model)
    if 'micro' in checkpoint_path or 'distilled' in checkpoint_path:
        from models.config import get_micro_config
        config = get_micro_config(vocab_size=2000)
    else:
        config = IteraLiteConfig(
            vocab_size=2000,
            hidden_size=256,
            num_layers=6,
            ssm_state_size=64,
            num_experts=8,
            top_k_experts=2,
            max_seq_length=512
        )
    
    # Export
    try:
        results = export_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir="deployment/models",
            model_class=IteraLiteModel,
            config=config,
            model_name=args.model_name or "itera_lite_micro",
            seq_length=128
        )
        
        logger.info("\n✓ Export complete!")
        logger.info(f"  TorchScript: {results.get('torchscript', 'Failed')}")
        logger.info(f"  ONNX: {results.get('onnx', 'Failed')}")
        
        # Save results
        results_file = Path("results/phase5_export_results.json")
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()


def task_kernel_optimization(args):
    """Task 2: Benchmark and optimize SSM kernels."""
    logger.info("=" * 80)
    logger.info("TASK: Kernel & Runtime Optimization")
    logger.info("=" * 80)
    
    # Benchmark different kernel implementations
    logger.info("\n1. Benchmarking SSM kernel implementations...")
    kernel_results = benchmark_ssm_kernels(
        d_model=args.d_model,
        d_state=args.d_state,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_runs=args.num_runs
    )
    
    # Profile individual operations
    logger.info("\n2. Profiling SSM operations...")
    profile_results = profile_ssm_operations(
        d_model=args.d_model,
        d_state=args.d_state,
        seq_length=args.seq_length
    )
    
    # Save results
    results = {
        'kernel_benchmarks': kernel_results,
        'operation_profiling': profile_results,
        'config': {
            'd_model': args.d_model,
            'd_state': args.d_state,
            'seq_length': args.seq_length,
            'batch_size': args.batch_size
        }
    }
    
    results_file = Path("results/phase5_kernel_optimization.json")
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")


def task_int4_quantization(args):
    """Task 3: Apply INT4 quantization and compare with INT8."""
    logger.info("=" * 80)
    logger.info("TASK: INT4 Quantization")
    logger.info("=" * 80)
    
    # Use micro distilled model for quantization
    checkpoint_path = args.checkpoint or "checkpoints/distilled/itera_lite_micro_distilled.pt"
    
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        # Try vocab checkpoint
        checkpoint_path = "checkpoints/vocab_2000/itera_lite_vocab2000_best.pt"
        if not Path(checkpoint_path).exists():
            logger.error("No suitable checkpoint found")
            return
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint first to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create config from checkpoint or use micro config
    if 'micro' in checkpoint_path or 'distilled' in checkpoint_path:
        from models.config import get_micro_config
        config = get_micro_config(vocab_size=2000)
        logger.info(f"Using micro config for distilled model")
    elif 'config' in checkpoint and checkpoint['config'] is not None:
        config = checkpoint['config']
        logger.info(f"Using config from checkpoint")
    else:
        logger.error("Cannot determine model config")
        return
    
    # Load model
    model = IteraLiteModel(config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create quantizer
    quantizer = AdvancedQuantizer(model, model_name="itera_lite")
    
    # Apply INT8
    logger.info("\n1. Applying INT8 quantization...")
    int8_model = quantizer.apply_int8_dynamic()
    
    # Apply INT4
    logger.info("\n2. Applying INT4 quantization...")
    int4_model, quant_info = quantizer.apply_int4_simulated()
    
    # Compare quantizations
    logger.info("\n3. Comparing quantization methods...")
    sample_input = torch.randint(0, config.vocab_size, (1, 128))
    
    comparison_results = quantizer.compare_quantizations(
        int8_model=int8_model,
        int4_model=int4_model,
        sample_input=sample_input,
        num_runs=args.num_runs
    )
    
    # Save INT4 model
    logger.info("\n4. Saving INT4 model...")
    quantizer.save_quantized_model(
        model=int4_model,
        save_path="checkpoints/int4/itera_lite_int4.pt",
        quant_info=quant_info
    )
    
    # Save results
    results_file = Path("results/phase5_int4_quantization.json")
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")


def task_edge_benchmarking(args):
    """Task 4: Benchmark models on different platforms."""
    logger.info("=" * 80)
    logger.info("TASK: Edge & Cross-Platform Benchmarking")
    logger.info("=" * 80)
    
    # This would benchmark on actual edge devices
    # For now, simulate with different settings
    
    platforms = {
        'desktop_cpu': {'threads': 12, 'description': 'Desktop CPU (12 cores)'},
        'laptop_cpu': {'threads': 4, 'description': 'Laptop CPU (4 cores)'},
        'embedded_cpu': {'threads': 2, 'description': 'Embedded CPU (2 cores, simulated)'},
    }
    
    results = {}
    
    for platform_name, platform_config in platforms.items():
        logger.info(f"\nBenchmarking on {platform_config['description']}...")
        
        # Set thread count to simulate platform
        torch.set_num_threads(platform_config['threads'])
        
        # Load model (use micro for efficiency)
        checkpoint_path = "checkpoints/distilled/itera_lite_micro_distilled.pt"
        
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping")
            continue
        
        from models.config import get_micro_config
        config = get_micro_config(vocab_size=2000)
        
        model = IteraLiteModel(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Benchmark
        benchmark = ModelBenchmark(model, model_name=f"itera_lite_micro_{platform_name}")
        platform_results = benchmark.run_full_benchmark(
            batch_size=1,
            seq_length=128
        )
        
        results[platform_name] = {
            'config': platform_config,
            'metrics': platform_results
        }
        
        # Extract metrics safely
        latency = platform_results.get('inference_speed', {}).get('mean_time_ms', 0) if isinstance(platform_results.get('inference_speed'), dict) else 0
        throughput = platform_results.get('inference_speed', {}).get('throughput_tokens_per_sec', 0) if isinstance(platform_results.get('inference_speed'), dict) else 0
        logger.info(f"  Latency: {latency:.2f} ms")
        logger.info(f"  Throughput: {throughput:.0f} tok/s")
    
    # Reset thread count
    torch.set_num_threads(12)
    
    # Save results
    results_file = Path("results/phase5_edge_benchmarking.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Deployment & Optimization")
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['export', 'kernels', 'int4', 'edge', 'all'],
        help='Task to run'
    )
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, default='itera_lite_micro', help='Model name for export')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--d-state', type=int, default=64, help='State dimension')
    parser.add_argument('--seq-length', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PHASE 5: DEPLOYMENT, VALIDATION & EDGE OPTIMIZATION")
    logger.info("=" * 80)
    
    if args.task == 'export' or args.task == 'all':
        task_export_models(args)
    
    if args.task == 'kernels' or args.task == 'all':
        task_kernel_optimization(args)
    
    if args.task == 'int4' or args.task == 'all':
        task_int4_quantization(args)
    
    if args.task == 'edge' or args.task == 'all':
        task_edge_benchmarking(args)
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5 TASK COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
