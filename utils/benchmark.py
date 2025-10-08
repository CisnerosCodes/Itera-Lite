"""
Benchmarking utilities for model efficiency analysis
"""

import torch
import torch.nn as nn
import time
import psutil
import os
from typing import Dict, Optional, List
import json
import numpy as np
from pathlib import Path


class ModelBenchmark:
    """Comprehensive model benchmarking"""
    
    def __init__(self, model: nn.Module, model_name: str, device: str = 'cpu'):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Try to get embedding params if method exists
        try:
            non_embedding = self.model.get_num_params(non_embedding=True)
            embedding_params = total_params - non_embedding
        except:
            embedding_params = 0
            non_embedding = total_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_embedding': non_embedding,
            'embedding': embedding_params
        }
    
    def measure_inference_speed(
        self,
        batch_size: int = 1,
        seq_length: int = 128,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """Measure inference speed"""
        # Create dummy input
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_ids)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = self.model(input_ids)
                end = time.perf_counter()
                times.append(end - start)
        
        times = np.array(times)
        tokens_processed = batch_size * seq_length * num_iterations
        
        return {
            'mean_time_ms': float(times.mean() * 1000),
            'std_time_ms': float(times.std() * 1000),
            'min_time_ms': float(times.min() * 1000),
            'max_time_ms': float(times.max() * 1000),
            'throughput_tokens_per_sec': tokens_processed / times.sum(),
            'latency_per_token_ms': float(times.mean() * 1000 / (batch_size * seq_length))
        }
    
    def estimate_flops(
        self,
        batch_size: int = 1,
        seq_length: int = 128
    ) -> Dict[str, int]:
        """Estimate FLOPs for forward pass"""
        # Get efficiency stats if available
        try:
            stats = self.model.get_efficiency_stats()
            flops_per_token = stats.get('approx_flops_per_token', 0)
        except:
            # Rough estimate based on parameters
            params = self.count_parameters()['non_embedding']
            flops_per_token = params * 2  # Rough approximation
        
        total_flops = flops_per_token * batch_size * seq_length
        
        return {
            'flops_per_token': flops_per_token,
            'total_flops': total_flops,
            'gflops': total_flops / 1e9
        }
    
    def measure_memory(
        self,
        batch_size: int = 1,
        seq_length: int = 128
    ) -> Dict[str, float]:
        """Measure memory usage"""
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_mb = process.memory_info().rss / 1024 / 1024
        
        # Model memory (parameters)
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        param_memory_mb = param_memory / 1024 / 1024
        
        # Forward pass memory
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        peak_mb = process.memory_info().rss / 1024 / 1024
        activation_mb = peak_mb - baseline_mb - param_memory_mb
        
        return {
            'param_memory_mb': param_memory_mb,
            'activation_memory_mb': max(0, activation_mb),
            'total_memory_mb': param_memory_mb + max(0, activation_mb),
            'baseline_mb': baseline_mb,
            'peak_mb': peak_mb
        }
    
    def measure_cpu_utilization(
        self,
        batch_size: int = 1,
        seq_length: int = 128,
        duration: float = 5.0
    ) -> Dict[str, float]:
        """Measure CPU utilization during inference"""
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)
        
        # Start monitoring
        cpu_percentages = []
        start_time = time.time()
        
        with torch.no_grad():
            while time.time() - start_time < duration:
                _ = self.model(input_ids)
                cpu_percentages.append(psutil.cpu_percent(interval=0.1))
        
        return {
            'mean_cpu_percent': np.mean(cpu_percentages),
            'max_cpu_percent': np.max(cpu_percentages),
            'std_cpu_percent': np.std(cpu_percentages)
        }
    
    def calculate_perplexity(
        self,
        dataloader,
        max_batches: Optional[int] = None
    ) -> float:
        """Calculate perplexity on dataset"""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, labels=target_ids)
                if len(outputs) == 3:  # Itera-Lite
                    _, loss, _ = outputs
                elif len(outputs) == 2:  # Transformer
                    _, loss = outputs
                
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def run_full_benchmark(
        self,
        dataloader=None,
        batch_size: int = 1,
        seq_length: int = 128,
        save_path: Optional[str] = None
    ) -> Dict:
        """Run complete benchmark suite"""
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {self.model_name}")
        print(f"{'=' * 70}\n")
        
        results = {
            'model_name': self.model_name,
            'device': self.device,
        }
        
        # Parameters
        print("1. Counting parameters...")
        params = self.count_parameters()
        results['parameters'] = params
        print(f"   Total: {params['total']:,}")
        print(f"   Non-embedding: {params['non_embedding']:,}")
        
        # FLOPs
        print("\n2. Estimating FLOPs...")
        flops = self.estimate_flops(batch_size, seq_length)
        results['flops'] = flops
        print(f"   FLOPs/token: {flops['flops_per_token']:,}")
        print(f"   Total GFLOPs: {flops['gflops']:.2f}")
        
        # Inference speed
        print("\n3. Measuring inference speed...")
        speed = self.measure_inference_speed(batch_size, seq_length, num_iterations=50)
        results['inference_speed'] = speed
        print(f"   Latency: {speed['mean_time_ms']:.2f} ± {speed['std_time_ms']:.2f} ms")
        print(f"   Throughput: {speed['throughput_tokens_per_sec']:.0f} tokens/sec")
        
        # Memory
        print("\n4. Measuring memory usage...")
        memory = self.measure_memory(batch_size, seq_length)
        results['memory'] = memory
        print(f"   Parameters: {memory['param_memory_mb']:.2f} MB")
        print(f"   Total: {memory['total_memory_mb']:.2f} MB")
        
        # CPU utilization
        print("\n5. Measuring CPU utilization...")
        cpu = self.measure_cpu_utilization(batch_size, seq_length, duration=3.0)
        results['cpu_utilization'] = cpu
        print(f"   Mean: {cpu['mean_cpu_percent']:.1f}%")
        print(f"   Max: {cpu['max_cpu_percent']:.1f}%")
        
        # Perplexity (if dataloader provided)
        if dataloader:
            print("\n6. Calculating perplexity...")
            ppl = self.calculate_perplexity(dataloader, max_batches=50)
            results['perplexity'] = ppl
            print(f"   Perplexity: {ppl:.2f}")
        
        print(f"\n{'=' * 70}\n")
        
        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {save_path}")
        
        return results


def compare_models(results: List[Dict], save_path: Optional[str] = None) -> Dict:
    """Compare multiple model benchmark results"""
    if not results:
        return {}
    
    comparison = {
        'models': [r['model_name'] for r in results],
        'comparisons': {}
    }
    
    # Use first model as baseline
    baseline = results[0]
    baseline_name = baseline['model_name']
    
    print(f"\n{'=' * 70}")
    print(f"MODEL COMPARISON (baseline: {baseline_name})")
    print(f"{'=' * 70}\n")
    
    # Compare each metric
    print(f"{'Metric':<30} {'Baseline':<15} {'Ratios':<30}")
    print(f"{'-' * 70}")
    
    # Parameters
    baseline_params = baseline['parameters']['total']
    ratios = [r['parameters']['total'] / baseline_params for r in results]
    print(f"{'Total Parameters':<30} {baseline_params:>14,} {str([f'{r:.2f}x' for r in ratios]):<30}")
    comparison['comparisons']['params_ratio'] = ratios
    
    # FLOPs
    baseline_flops = baseline['flops']['flops_per_token']
    ratios = [r['flops']['flops_per_token'] / baseline_flops for r in results]
    print(f"{'FLOPs/token':<30} {baseline_flops:>14,} {str([f'{r:.2f}x' for r in ratios]):<30}")
    comparison['comparisons']['flops_ratio'] = ratios
    
    # Speed
    baseline_speed = baseline['inference_speed']['throughput_tokens_per_sec']
    ratios = [r['inference_speed']['throughput_tokens_per_sec'] / baseline_speed for r in results]
    print(f"{'Throughput (tokens/s)':<30} {baseline_speed:>14,.0f} {str([f'{r:.2f}x' for r in ratios]):<30}")
    comparison['comparisons']['speed_ratio'] = ratios
    
    # Memory
    baseline_mem = baseline['memory']['total_memory_mb']
    ratios = [r['memory']['total_memory_mb'] / baseline_mem for r in results]
    print(f"{'Memory (MB)':<30} {baseline_mem:>14.2f} {str([f'{r:.2f}x' for r in ratios]):<30}")
    comparison['comparisons']['memory_ratio'] = ratios
    
    # Perplexity (if available)
    if 'perplexity' in baseline:
        baseline_ppl = baseline['perplexity']
        ratios = [r.get('perplexity', 0) / baseline_ppl if baseline_ppl > 0 else 0 for r in results]
        print(f"{'Perplexity':<30} {baseline_ppl:>14.2f} {str([f'{r:.2f}x' for r in ratios]):<30}")
        comparison['comparisons']['perplexity_ratio'] = ratios
    
    print(f"\n{'=' * 70}\n")
    
    # Calculate efficiency score
    print("Efficiency Summary:")
    for i, result in enumerate(results):
        if i == 0:
            continue  # Skip baseline
        
        param_reduction = baseline['parameters']['total'] / result['parameters']['total']
        flop_reduction = baseline['flops']['flops_per_token'] / result['flops']['flops_per_token']
        speed_improvement = result['inference_speed']['throughput_tokens_per_sec'] / baseline_speed
        
        print(f"\n{result['model_name']} vs {baseline_name}:")
        print(f"   Parameters: {param_reduction:.2f}x smaller" if param_reduction > 1 else f"   Parameters: {1/param_reduction:.2f}x larger")
        print(f"   FLOPs: {flop_reduction:.2f}x fewer" if flop_reduction > 1 else f"   FLOPs: {1/flop_reduction:.2f}x more")
        print(f"   Speed: {speed_improvement:.2f}x faster" if speed_improvement > 1 else f"   Speed: {1/speed_improvement:.2f}x slower")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {save_path}")
    
    return comparison
