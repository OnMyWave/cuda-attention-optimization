"""
Phase 7: Comprehensive End-to-End Benchmarking
Compare optimized transformer against PyTorch baseline
"""

import torch
import torch.nn as nn
import time
import json
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_baseline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from phase1_baseline.pytorch_transformer import Transformer as PyTorchTransformer
from optimized_transformer import OptimizedTransformer


def benchmark_model(model, input_ids, num_runs=50, warmup=5):
    """
    Benchmark a model
    Args:
        model: The model to benchmark
        input_ids: Input tensor
        num_runs: Number of timing runs
        warmup: Number of warmup runs
    Returns:
        dict with timing metrics
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    torch.cuda.synchronize()

    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_ids)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_runs * 1000

    return {
        'time_ms': avg_time_ms,
        'throughput_samples_per_sec': 1000 / avg_time_ms,
        'throughput_tokens_per_sec': (input_ids.shape[0] * input_ids.shape[1] * 1000) / avg_time_ms
    }


def compare_models(batch_size, seq_len, vocab_size=10000, hidden_dim=512,
                   num_layers=4, num_heads=8):
    """
    Compare PyTorch and optimized models
    """
    print(f"\n{'='*80}")
    print(f"Config: batch={batch_size}, seq_len={seq_len}, "
          f"hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"{'='*80}")

    device = 'cuda'

    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # PyTorch baseline
    print("\nCreating PyTorch baseline model...")
    pytorch_model = PyTorchTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=2048,
        max_seq_len=512
    ).to(device)

    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"PyTorch model parameters: {pytorch_params:,}")

    print("Benchmarking PyTorch model...")
    pytorch_metrics = benchmark_model(pytorch_model, input_ids)

    # Optimized model
    print("\nCreating optimized CUDA model...")
    optimized_model = OptimizedTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=2048,
        max_seq_len=512,
        use_cuda=True
    ).to(device)

    optimized_params = sum(p.numel() for p in optimized_model.parameters())
    print(f"Optimized model parameters: {optimized_params:,}")

    print("Benchmarking optimized model...")
    optimized_metrics = benchmark_model(optimized_model, input_ids)

    # Calculate speedup
    speedup = pytorch_metrics['time_ms'] / optimized_metrics['time_ms']

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\nPyTorch Baseline:")
    print(f"  Time: {pytorch_metrics['time_ms']:.3f} ms")
    print(f"  Throughput: {pytorch_metrics['throughput_tokens_per_sec']:.0f} tokens/sec")

    print(f"\nOptimized CUDA:")
    print(f"  Time: {optimized_metrics['time_ms']:.3f} ms")
    print(f"  Throughput: {optimized_metrics['throughput_tokens_per_sec']:.0f} tokens/sec")

    print(f"\nSpeedup: {speedup:.2f}x")

    # Cleanup
    del pytorch_model
    del optimized_model
    torch.cuda.empty_cache()

    return {
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads
        },
        'pytorch': pytorch_metrics,
        'optimized': optimized_metrics,
        'speedup': speedup
    }


def run_comprehensive_benchmarks():
    """
    Run comprehensive benchmarks across different configurations
    """
    print("="*80)
    print("Phase 7: Comprehensive End-to-End Benchmarking")
    print("="*80)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Test configurations
    configs = [
        # (batch, seq_len, hidden_dim, num_layers, num_heads)
        (1, 128, 512, 4, 8),
        (4, 128, 512, 4, 8),
        (8, 128, 512, 4, 8),
        (4, 256, 512, 4, 8),
        (4, 512, 512, 4, 8),
        (4, 128, 512, 6, 8),  # More layers
        (4, 128, 768, 6, 12), # Larger model (BERT-base size)
    ]

    results = {
        'gpu_info': {
            'name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'compute_capability': torch.cuda.get_device_capability(0),
        },
        'benchmarks': []
    }

    for batch, seq_len, hidden_dim, num_layers, num_heads in configs:
        try:
            result = compare_models(batch, seq_len, 10000, hidden_dim, num_layers, num_heads)
            results['benchmarks'].append(result)
        except Exception as e:
            print(f"\nError with config batch={batch}, seq_len={seq_len}: {e}")
            continue

    # Save results
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'final_benchmark.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")

    return results


def create_summary_plots(results):
    """Create summary visualization plots"""
    import matplotlib.pyplot as plt
    import numpy as np

    benchmarks = results['benchmarks']
    configs = [b['config'] for b in benchmarks]

    # Create labels
    labels = []
    for c in configs:
        if c['num_layers'] == 4 and c['hidden_dim'] == 512:
            labels.append(f"B={c['batch_size']}, L={c['seq_len']}")
        else:
            labels.append(f"B={c['batch_size']}, L={c['seq_len']}\nH={c['hidden_dim']}, N={c['num_layers']}")

    # Extract metrics
    pytorch_times = [b['pytorch']['time_ms'] for b in benchmarks]
    optimized_times = [b['optimized']['time_ms'] for b in benchmarks]
    speedups = [b['speedup'] for b in benchmarks]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Execution time comparison
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, pytorch_times, width, label='PyTorch', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, optimized_times, width, label='Optimized CUDA', alpha=0.8, color='coral')

    ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Time (ms)', fontweight='bold', fontsize=12)
    ax.set_title('End-to-End Transformer Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Speedup
    ax = axes[1]
    bars = ax.bar(x, speedups, color='green', alpha=0.8)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (1x)')

    ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Speedup', fontweight='bold', fontsize=12)
    ax.set_title('Optimized CUDA Speedup over PyTorch', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/final_benchmark.png', dpi=300, bbox_inches='tight')
    print("✓ Benchmark plot saved to visualizations/final_benchmark.png")


def create_scaling_analysis(results):
    """Analyze performance scaling"""
    import matplotlib.pyplot as plt
    import numpy as np

    benchmarks = results['benchmarks']

    # Filter benchmarks with standard config (hidden=512, layers=4)
    standard_benchmarks = [b for b in benchmarks
                          if b['config']['hidden_dim'] == 512
                          and b['config']['num_layers'] == 4]

    if len(standard_benchmarks) < 3:
        print("Not enough data for scaling analysis")
        return

    # Group by batch size
    batch_groups = {}
    for b in standard_benchmarks:
        batch = b['config']['batch_size']
        if batch not in batch_groups:
            batch_groups[batch] = []
        batch_groups[batch].append(b)

    # Sort by sequence length
    for batch in batch_groups:
        batch_groups[batch].sort(key=lambda x: x['config']['seq_len'])

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_groups)))

    # Plot 1: Throughput vs sequence length
    ax = axes[0]
    for i, (batch, data) in enumerate(batch_groups.items()):
        seq_lens = [d['config']['seq_len'] for d in data]
        pytorch_throughput = [d['pytorch']['throughput_tokens_per_sec'] for d in data]
        optimized_throughput = [d['optimized']['throughput_tokens_per_sec'] for d in data]

        ax.plot(seq_lens, pytorch_throughput, 'o--', label=f'PyTorch (B={batch})',
               color=colors[i], alpha=0.7)
        ax.plot(seq_lens, optimized_throughput, 's-', label=f'Optimized (B={batch})',
               color=colors[i], linewidth=2)

    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax.set_title('Throughput Scaling with Sequence Length', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Speedup vs sequence length
    ax = axes[1]
    for i, (batch, data) in enumerate(batch_groups.items()):
        seq_lens = [d['config']['seq_len'] for d in data]
        speedups = [d['speedup'] for d in data]

        ax.plot(seq_lens, speedups, 'o-', label=f'Batch={batch}',
               color=colors[i], linewidth=2, markersize=8)

    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Baseline')
    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_ylabel('Speedup', fontweight='bold')
    ax.set_title('Speedup Scaling with Sequence Length', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Scaling analysis saved to visualizations/scaling_analysis.png")


def main():
    """Main benchmarking function"""

    # Run benchmarks
    results = run_comprehensive_benchmarks()

    if results is None:
        return

    # Create visualizations
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)

    create_summary_plots(results)
    create_scaling_analysis(results)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    speedups = [b['speedup'] for b in results['benchmarks']]
    avg_speedup = sum(speedups) / len(speedups)
    max_speedup = max(speedups)
    min_speedup = min(speedups)

    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    print(f"Maximum Speedup: {max_speedup:.2f}x")
    print(f"Minimum Speedup: {min_speedup:.2f}x")

    print("\n" + "="*80)
    print("Benchmarking Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
