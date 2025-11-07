"""
Profile fused CUDA attention implementation
Compare against all previous implementations
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
    import attention_fused
    FUSED_AVAILABLE = True
except ImportError:
    print("Error: attention_fused module not found. Build it first:")
    print("  python setup.py install")
    FUSED_AVAILABLE = False
    sys.exit(1)

# Try to import previous implementations
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase2_naive'))
    import attention_cuda as attention_naive
    NAIVE_AVAILABLE = True
except ImportError:
    NAIVE_AVAILABLE = False

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase3_tiled'))
    import attention_tiled
    TILED_AVAILABLE = True
except ImportError:
    TILED_AVAILABLE = False

from utils.roofline import plot_roofline


def pytorch_attention(Q, K, V):
    """PyTorch reference implementation"""
    batch, seq_len, head_dim = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output


def benchmark_implementation(impl_fn, Q, K, V, num_runs=100, warmup=10):
    """Benchmark an implementation"""
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

    # For fused kernel, only Q, K, V inputs and output are touched in global memory
    qkv_bytes = 3 * batch * seq_len * head_dim * 4
    output_bytes = batch * seq_len * head_dim * 4
    total_bytes = qkv_bytes + output_bytes  # No intermediate storage!

    operational_intensity = total_flops / total_bytes
    bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

    return {
        'gflops': gflops,
        'bandwidth_gb_s': bandwidth_gb_s,
        'operational_intensity': operational_intensity,
        'total_flops': total_flops,
        'total_bytes': total_bytes
    }


def compare_all_implementations(batch, seq_len, head_dim, num_runs=100):
    """Compare all available implementations"""
    print(f"\nConfig: batch={batch}, seq_len={seq_len}, head_dim={head_dim}")

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

    # Naive CUDA
    if NAIVE_AVAILABLE:
        print("  Benchmarking Naive CUDA...")
        naive_metrics = benchmark_implementation(attention_naive.forward, Q, K, V, num_runs)
        naive_perf = calculate_metrics(batch, seq_len, head_dim, naive_metrics['time_ms'])
        results['naive'] = {**naive_metrics, **naive_perf}

    # Tiled CUDA
    if TILED_AVAILABLE:
        print("  Benchmarking Tiled CUDA...")
        tiled_fn = lambda Q, K, V: attention_tiled.forward(Q, K, V, 16)
        tiled_metrics = benchmark_implementation(tiled_fn, Q, K, V, num_runs)
        tiled_perf = calculate_metrics(batch, seq_len, head_dim, tiled_metrics['time_ms'])
        results['tiled'] = {**tiled_metrics, **tiled_perf}

    # Fused CUDA
    print("  Benchmarking Fused CUDA...")
    fused_metrics = benchmark_implementation(attention_fused.forward, Q, K, V, num_runs)
    fused_perf = calculate_metrics(batch, seq_len, head_dim, fused_metrics['time_ms'])
    results['fused'] = {**fused_metrics, **fused_perf}

    # Print results
    print(f"\n  Results:")
    for name, data in results.items():
        print(f"  {name.capitalize()}:")
        print(f"    Time: {data['time_ms']:.3f} ms")
        print(f"    Performance: {data['gflops']:.2f} GFLOPS")
        print(f"    Bandwidth: {data['bandwidth_gb_s']:.2f} GB/s")
        print(f"    Operational Intensity: {data['operational_intensity']:.3f} FLOPs/Byte")
        if name != 'pytorch':
            speedup = results['pytorch']['time_ms'] / data['time_ms']
            print(f"    Speedup vs PyTorch: {speedup:.2f}x")

    return {
        'config': {'batch': batch, 'seq_len': seq_len, 'head_dim': head_dim},
        'results': results
    }


def run_all_benchmarks():
    """Run comprehensive benchmarks"""
    print("="*80)
    print("Phase 4: Fused CUDA Implementation - Profiling")
    print("="*80)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    configs = [
        {'batch': 1, 'seq_len': 128, 'head_dim': 64},
        {'batch': 4, 'seq_len': 128, 'head_dim': 64},
        {'batch': 8, 'seq_len': 128, 'head_dim': 64},
        {'batch': 4, 'seq_len': 256, 'head_dim': 64},
        {'batch': 2, 'seq_len': 512, 'head_dim': 64},
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
        result = compare_all_implementations(**config, num_runs=100)
        all_results['benchmarks'].append(result)

    # Save results
    output_file = 'results/fused_metrics.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    return all_results


def create_comprehensive_comparison(results):
    """Create comprehensive comparison plots"""
    import matplotlib.pyplot as plt
    import numpy as np

    benchmarks = results['benchmarks']
    configs = [b['config'] for b in benchmarks]
    labels = [f"B={c['batch']}, L={c['seq_len']}" for c in configs]

    # Determine which implementations are available
    implementations = ['pytorch']
    if 'naive' in benchmarks[0]['results']:
        implementations.append('naive')
    if 'tiled' in benchmarks[0]['results']:
        implementations.append('tiled')
    implementations.append('fused')

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Execution Time
    ax = axes[0, 0]
    x = np.arange(len(labels))
    width = 0.2
    colors = {'pytorch': 'steelblue', 'naive': 'coral', 'tiled': 'lightgreen', 'fused': 'gold'}

    for i, impl in enumerate(implementations):
        times = [b['results'][impl]['time_ms'] for b in benchmarks]
        offset = (i - len(implementations)/2 + 0.5) * width
        ax.bar(x + offset, times, width, label=impl.capitalize(), color=colors[impl], alpha=0.8)

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Execution Time Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Speedup vs PyTorch
    ax = axes[0, 1]
    speedups = {}
    for impl in implementations[1:]:  # Skip PyTorch
        speedups[impl] = [b['results']['pytorch']['time_ms'] / b['results'][impl]['time_ms']
                         for b in benchmarks]

    for i, (impl, speeds) in enumerate(speedups.items()):
        offset = (i - len(speedups)/2 + 0.5) * width * 2
        bars = ax.bar(x + offset, speeds, width * 1.5, label=impl.capitalize(),
                     color=colors[impl], alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}x', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Baseline')
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Speedup vs PyTorch', fontweight='bold')
    ax.set_title('Speedup Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: GFLOPS
    ax = axes[1, 0]
    for i, impl in enumerate(implementations):
        gflops = [b['results'][impl]['gflops'] for b in benchmarks]
        offset = (i - len(implementations)/2 + 0.5) * width
        ax.bar(x + offset, gflops, width, label=impl.capitalize(), color=colors[impl], alpha=0.8)

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontweight='bold')
    ax.set_title('Compute Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Operational Intensity
    ax = axes[1, 1]
    for i, impl in enumerate(implementations):
        oi = [b['results'][impl]['operational_intensity'] for b in benchmarks]
        offset = (i - len(implementations)/2 + 0.5) * width
        ax.bar(x + offset, oi, width, label=impl.capitalize(), color=colors[impl], alpha=0.8)

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Operational Intensity (FLOPs/Byte)', fontweight='bold')
    ax.set_title('Operational Intensity Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fused_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive plot saved to results/fused_comprehensive.png")


def create_roofline_plot(results):
    """Create roofline plot for all implementations"""
    gpu_name = results['gpu_info']['name']

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

    # Select representative config (batch=4, seq_len=128)
    benchmark = results['benchmarks'][1]
    config = benchmark['config']
    results_data = benchmark['results']

    measurements = []
    for impl in results_data.keys():
        measurements.append({
            'name': impl.capitalize(),
            'operational_intensity': results_data[impl]['operational_intensity'],
            'gflops': results_data[impl]['gflops']
        })

    plot_roofline(
        peak_gflops=gpu_specs['peak_gflops'],
        peak_bandwidth_gb_s=gpu_specs['peak_bandwidth_gb_s'],
        measurements=measurements,
        title=f"All Implementations Roofline (B=4, L=128) - {gpu_name}",
        output_file="results/fused_roofline.png"
    )


def main():
    """Main profiling function"""
    if not FUSED_AVAILABLE:
        return

    results = run_all_benchmarks()

    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)

    create_comprehensive_comparison(results)
    create_roofline_plot(results)

    print("\n" + "="*80)
    print("Profiling Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
