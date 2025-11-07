"""
Profile tiled CUDA attention implementation
Compare against PyTorch and naive CUDA
"""

import torch
import torch.nn.functional as F
import math
import time
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import attention_tiled
    TILED_AVAILABLE = True
except ImportError:
    print("Error: attention_tiled module not found. Build it first:")
    print("  python setup.py install")
    TILED_AVAILABLE = False
    sys.exit(1)

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase2_naive'))
    import attention_cuda as attention_naive
    NAIVE_AVAILABLE = True
except ImportError:
    print("Warning: attention_cuda (naive) not found. Will skip comparison.")
    NAIVE_AVAILABLE = False

from utils.roofline import plot_roofline, analyze_performance, print_analysis


def pytorch_attention(Q, K, V):
    """PyTorch reference implementation"""
    batch, seq_len, head_dim = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output


def benchmark_implementation(impl_fn, Q, K, V, num_runs=100, warmup=10):
    """Benchmark an attention implementation"""
    for _ in range(warmup):
        with torch.no_grad():
            _ = impl_fn(Q, K, V)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            output = impl_fn(Q, K, V)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000
    return {'time_ms': avg_time_ms}


def calculate_metrics(batch, seq_len, head_dim, time_ms):
    """Calculate FLOPs and memory metrics"""
    qk_flops = 2 * batch * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch * seq_len * seq_len
    av_flops = 2 * batch * seq_len * seq_len * head_dim
    total_flops = qk_flops + softmax_flops + av_flops

    gflops = (total_flops / 1e9) / (time_ms / 1000)

    qkv_bytes = 3 * batch * seq_len * head_dim * 4
    intermediate_bytes = 2 * batch * seq_len * seq_len * 4
    output_bytes = batch * seq_len * head_dim * 4
    total_bytes = qkv_bytes + intermediate_bytes + output_bytes

    operational_intensity = total_flops / total_bytes
    bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

    return {
        'gflops': gflops,
        'bandwidth_gb_s': bandwidth_gb_s,
        'operational_intensity': operational_intensity,
        'total_flops': total_flops,
        'total_bytes': total_bytes
    }


def compare_implementations(batch, seq_len, head_dim, tile_size=16, num_runs=100):
    """Compare all available implementations"""
    print(f"\nConfig: batch={batch}, seq_len={seq_len}, head_dim={head_dim}, tile_size={tile_size}")

    device = 'cuda'
    Q = torch.randn(batch, seq_len, head_dim, device=device)
    K = torch.randn(batch, seq_len, head_dim, device=device)
    V = torch.randn(batch, seq_len, head_dim, device=device)

    results = {}

    # PyTorch
    print("  Benchmarking PyTorch...")
    pytorch_metrics = benchmark_implementation(pytorch_attention, Q, K, V, num_runs)
    pytorch_perf = calculate_metrics(batch, seq_len, head_dim, pytorch_metrics['time_ms'])
    results['pytorch'] = {**pytorch_metrics, **pytorch_perf}

    # Naive CUDA (if available)
    if NAIVE_AVAILABLE:
        print("  Benchmarking Naive CUDA...")
        naive_metrics = benchmark_implementation(attention_naive.forward, Q, K, V, num_runs)
        naive_perf = calculate_metrics(batch, seq_len, head_dim, naive_metrics['time_ms'])
        results['naive'] = {**naive_metrics, **naive_perf}

    # Tiled CUDA
    print("  Benchmarking Tiled CUDA...")
    tiled_fn = lambda Q, K, V: attention_tiled.forward(Q, K, V, tile_size)
    tiled_metrics = benchmark_implementation(tiled_fn, Q, K, V, num_runs)
    tiled_perf = calculate_metrics(batch, seq_len, head_dim, tiled_metrics['time_ms'])
    results['tiled'] = {**tiled_metrics, **tiled_perf}

    # Print results
    print(f"\n  PyTorch:")
    print(f"    Time: {results['pytorch']['time_ms']:.3f} ms")
    print(f"    Performance: {results['pytorch']['gflops']:.2f} GFLOPS")
    print(f"    Bandwidth: {results['pytorch']['bandwidth_gb_s']:.2f} GB/s")
    print(f"    Operational Intensity: {results['pytorch']['operational_intensity']:.3f} FLOPs/Byte")

    if NAIVE_AVAILABLE:
        print(f"\n  Naive CUDA:")
        print(f"    Time: {results['naive']['time_ms']:.3f} ms")
        print(f"    Performance: {results['naive']['gflops']:.2f} GFLOPS")
        print(f"    Speedup vs PyTorch: {results['pytorch']['time_ms'] / results['naive']['time_ms']:.2f}x")

    print(f"\n  Tiled CUDA:")
    print(f"    Time: {results['tiled']['time_ms']:.3f} ms")
    print(f"    Performance: {results['tiled']['gflops']:.2f} GFLOPS")
    print(f"    Speedup vs PyTorch: {results['pytorch']['time_ms'] / results['tiled']['time_ms']:.2f}x")
    if NAIVE_AVAILABLE:
        print(f"    Speedup vs Naive: {results['naive']['time_ms'] / results['tiled']['time_ms']:.2f}x")

    return {
        'config': {'batch': batch, 'seq_len': seq_len, 'head_dim': head_dim, 'tile_size': tile_size},
        'results': results
    }


def run_all_benchmarks():
    """Run comprehensive benchmarks"""
    print("="*80)
    print("Phase 3: Tiled CUDA Implementation - Profiling")
    print("="*80)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    configs = [
        {'batch': 1, 'seq_len': 128, 'head_dim': 64},
        {'batch': 4, 'seq_len': 128, 'head_dim': 64},
        {'batch': 8, 'seq_len': 128, 'head_dim': 64},
        {'batch': 4, 'seq_len': 256, 'head_dim': 64},
        {'batch': 4, 'seq_len': 512, 'head_dim': 64},
        {'batch': 4, 'seq_len': 128, 'head_dim': 128},
    ]

    all_results = {
        'gpu_info': {
            'name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'compute_capability': torch.cuda.get_device_capability(0),
        },
        'benchmarks': []
    }

    print("\n" + "="*80)
    print("Running Benchmarks")
    print("="*80)

    for config in configs:
        result = compare_implementations(**config, num_runs=100)
        all_results['benchmarks'].append(result)

    # Save results
    output_file = 'results/tiled_metrics.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    return all_results


def create_comparison_plots(results):
    """Create comparison plots"""
    import matplotlib.pyplot as plt

    benchmarks = results['benchmarks']
    configs = [b['config'] for b in benchmarks]
    labels = [f"B={c['batch']}, L={c['seq_len']}" for c in configs]

    # Extract times
    pytorch_times = [b['results']['pytorch']['time_ms'] for b in benchmarks]
    tiled_times = [b['results']['tiled']['time_ms'] for b in benchmarks]

    has_naive = 'naive' in benchmarks[0]['results']
    if has_naive:
        naive_times = [b['results']['naive']['time_ms'] for b in benchmarks]

    # Calculate speedups
    speedups_vs_pytorch = [p/t for p, t in zip(pytorch_times, tiled_times)]
    if has_naive:
        speedups_vs_naive = [n/t for n, t in zip(naive_times, tiled_times)]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Time comparison
    ax = axes[0]
    x = range(len(labels))
    width = 0.25

    if has_naive:
        ax.bar([i - width for i in x], pytorch_times, width, label='PyTorch', alpha=0.8)
        ax.bar([i for i in x], naive_times, width, label='Naive CUDA', alpha=0.8)
        ax.bar([i + width for i in x], tiled_times, width, label='Tiled CUDA', alpha=0.8)
    else:
        ax.bar([i - width/2 for i in x], pytorch_times, width, label='PyTorch', alpha=0.8)
        ax.bar([i + width/2 for i in x], tiled_times, width, label='Tiled CUDA', alpha=0.8)

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Execution Time Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Speedups
    ax = axes[1]

    if has_naive:
        ax.bar([i - width/2 for i in x], speedups_vs_pytorch, width,
               label='Speedup vs PyTorch', alpha=0.8, color='green')
        ax.bar([i + width/2 for i in x], speedups_vs_naive, width,
               label='Speedup vs Naive', alpha=0.8, color='blue')
    else:
        ax.bar(x, speedups_vs_pytorch, width*2,
               label='Speedup vs PyTorch', alpha=0.8, color='green')

    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1x)')
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Speedup', fontweight='bold')
    ax.set_title('Tiled CUDA Speedup', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/tiled_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved to results/tiled_comparison.png")


def create_roofline_plot(results):
    """Create roofline plot"""
    gpu_name = results['gpu_info']['name']

    # GPU specs
    gpu_specs = {'peak_gflops': 10000, 'peak_bandwidth_gb_s': 500}

    if 'V100' in gpu_name:
        gpu_specs = {'peak_gflops': 15700, 'peak_bandwidth_gb_s': 900}
    elif 'A100' in gpu_name:
        gpu_specs = {'peak_gflops': 19500, 'peak_bandwidth_gb_s': 1555}
    elif 'H100' in gpu_name:
        gpu_specs = {'peak_gflops': 51000, 'peak_bandwidth_gb_s': 3350}
    elif '3090' in gpu_name:
        gpu_specs = {'peak_gflops': 35580, 'peak_bandwidth_gb_s': 936}
    elif '4090' in gpu_name:
        gpu_specs = {'peak_gflops': 82580, 'peak_bandwidth_gb_s': 1008}
    elif 'T4' in gpu_name:
        gpu_specs = {'peak_gflops': 8100, 'peak_bandwidth_gb_s': 320}

    print(f"\nUsing GPU specs: {gpu_specs}")

    # Prepare measurements
    measurements = []
    selected_indices = [1, 3, 4]  # batch=4 configs

    for idx in selected_indices:
        benchmark = results['benchmarks'][idx]
        config = benchmark['config']
        results_data = benchmark['results']

        # PyTorch
        measurements.append({
            'name': f"PyTorch B={config['batch']}, L={config['seq_len']}",
            'operational_intensity': results_data['pytorch']['operational_intensity'],
            'gflops': results_data['pytorch']['gflops']
        })

        # Tiled
        measurements.append({
            'name': f"Tiled B={config['batch']}, L={config['seq_len']}",
            'operational_intensity': results_data['tiled']['operational_intensity'],
            'gflops': results_data['tiled']['gflops']
        })

    plot_roofline(
        peak_gflops=gpu_specs['peak_gflops'],
        peak_bandwidth_gb_s=gpu_specs['peak_bandwidth_gb_s'],
        measurements=measurements,
        title=f"Tiled CUDA Roofline Analysis - {gpu_name}",
        output_file="results/tiled_roofline.png"
    )


def main():
    """Main profiling function"""
    if not TILED_AVAILABLE:
        return

    results = run_all_benchmarks()

    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)

    create_comparison_plots(results)
    create_roofline_plot(results)

    print("\n" + "="*80)
    print("Profiling Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
