"""
Profile naive CUDA attention implementation
Compare against PyTorch baseline
"""

import torch
import torch.nn.functional as F
import math
import time
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import attention_cuda
    CUDA_AVAILABLE = True
except ImportError:
    print("Error: attention_cuda module not found. Please build it first:")
    print("  cd phase2_naive")
    print("  python setup.py install")
    CUDA_AVAILABLE = False
    sys.exit(1)

from utils.roofline import plot_roofline, analyze_performance, print_analysis


def pytorch_attention(Q, K, V):
    """PyTorch reference implementation"""
    batch, seq_len, head_dim = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output


def benchmark_implementation(impl_fn, Q, K, V, num_runs=100, warmup=10):
    """
    Benchmark an attention implementation
    Args:
        impl_fn: Function that takes (Q, K, V) and returns output
        Q, K, V: Input tensors
        num_runs: Number of timing runs
        warmup: Number of warmup runs
    Returns:
        dict with timing metrics
    """
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = impl_fn(Q, K, V)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Timing
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            output = impl_fn(Q, K, V)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000

    return {
        'time_ms': avg_time_ms,
        'throughput_ms': 1000 / avg_time_ms
    }


def calculate_metrics(batch, seq_len, head_dim, time_ms):
    """Calculate FLOPs and memory metrics"""

    # FLOPs calculation
    # Q @ K^T: 2 * batch * seq_len * seq_len * head_dim
    # Softmax: ~5 * batch * seq_len * seq_len (exp, max, sum, div, sub)
    # Attention @ V: 2 * batch * seq_len * seq_len * head_dim
    qk_flops = 2 * batch * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch * seq_len * seq_len
    av_flops = 2 * batch * seq_len * seq_len * head_dim
    total_flops = qk_flops + softmax_flops + av_flops

    gflops = (total_flops / 1e9) / (time_ms / 1000)

    # Memory calculation (bytes)
    # Q, K, V: 3 * batch * seq_len * head_dim * 4 bytes
    # Scores & Attn: 2 * batch * seq_len * seq_len * 4 bytes
    # Output: batch * seq_len * head_dim * 4 bytes
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


def compare_implementations(batch, seq_len, head_dim, num_runs=100):
    """Compare PyTorch vs CUDA implementations"""

    print(f"\nConfig: batch={batch}, seq_len={seq_len}, head_dim={head_dim}")

    device = 'cuda'

    # Create inputs
    Q = torch.randn(batch, seq_len, head_dim, device=device)
    K = torch.randn(batch, seq_len, head_dim, device=device)
    V = torch.randn(batch, seq_len, head_dim, device=device)

    # Benchmark PyTorch
    print("  Benchmarking PyTorch...")
    pytorch_metrics = benchmark_implementation(pytorch_attention, Q, K, V, num_runs)
    pytorch_perf = calculate_metrics(batch, seq_len, head_dim, pytorch_metrics['time_ms'])

    # Benchmark CUDA
    print("  Benchmarking CUDA...")
    cuda_metrics = benchmark_implementation(attention_cuda.forward, Q, K, V, num_runs)
    cuda_perf = calculate_metrics(batch, seq_len, head_dim, cuda_metrics['time_ms'])

    # Calculate speedup
    speedup = pytorch_metrics['time_ms'] / cuda_metrics['time_ms']

    # Print results
    print(f"\n  PyTorch:")
    print(f"    Time: {pytorch_metrics['time_ms']:.3f} ms")
    print(f"    Performance: {pytorch_perf['gflops']:.2f} GFLOPS")
    print(f"    Bandwidth: {pytorch_perf['bandwidth_gb_s']:.2f} GB/s")
    print(f"    Operational Intensity: {pytorch_perf['operational_intensity']:.3f} FLOPs/Byte")

    print(f"\n  Naive CUDA:")
    print(f"    Time: {cuda_metrics['time_ms']:.3f} ms")
    print(f"    Performance: {cuda_perf['gflops']:.2f} GFLOPS")
    print(f"    Bandwidth: {cuda_perf['bandwidth_gb_s']:.2f} GB/s")
    print(f"    Operational Intensity: {cuda_perf['operational_intensity']:.3f} FLOPs/Byte")

    print(f"\n  Speedup: {speedup:.2f}x")

    return {
        'config': {'batch': batch, 'seq_len': seq_len, 'head_dim': head_dim},
        'pytorch': {**pytorch_metrics, **pytorch_perf},
        'cuda': {**cuda_metrics, **cuda_perf},
        'speedup': speedup
    }


def run_all_benchmarks():
    """Run comprehensive benchmarks"""

    if not CUDA_AVAILABLE:
        return

    print("="*80)
    print("Phase 2: Naive CUDA Implementation - Profiling")
    print("="*80)

    # GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Test configurations
    configs = [
        {'batch': 1, 'seq_len': 128, 'head_dim': 64},
        {'batch': 4, 'seq_len': 128, 'head_dim': 64},
        {'batch': 8, 'seq_len': 128, 'head_dim': 64},
        {'batch': 4, 'seq_len': 256, 'head_dim': 64},
        {'batch': 4, 'seq_len': 512, 'head_dim': 64},
        {'batch': 4, 'seq_len': 128, 'head_dim': 128},
    ]

    results = {
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
        results['benchmarks'].append(result)

    # Save results
    output_file = 'results/naive_metrics.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    return results


def create_comparison_plots(results):
    """Create comparison plots"""
    import matplotlib.pyplot as plt

    configs = [r['config'] for r in results['benchmarks']]
    labels = [f"B={c['batch']}, L={c['seq_len']}, D={c['head_dim']}" for c in configs]

    pytorch_times = [r['pytorch']['time_ms'] for r in results['benchmarks']]
    cuda_times = [r['cuda']['time_ms'] for r in results['benchmarks']]
    speedups = [r['speedup'] for r in results['benchmarks']]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Execution time comparison
    ax = axes[0]
    x = range(len(labels))
    width = 0.35

    ax.bar([i - width/2 for i in x], pytorch_times, width, label='PyTorch', alpha=0.8)
    ax.bar([i + width/2 for i in x], cuda_times, width, label='Naive CUDA', alpha=0.8)

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Execution Time Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Speedup
    ax = axes[1]
    bars = ax.bar(x, speedups, color='green', alpha=0.8)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1x)')

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Speedup', fontweight='bold')
    ax.set_title('CUDA Speedup over PyTorch', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}x', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('results/naive_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved to results/naive_comparison.png")


def create_roofline_comparison(results):
    """Create roofline plot comparing PyTorch and CUDA"""

    gpu_name = results['gpu_info']['name']

    # Estimate GPU specs (adjust based on your GPU)
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

    # Select a few representative configs
    selected_indices = [1, 3, 4]  # batch=4 configs

    for idx in selected_indices:
        benchmark = results['benchmarks'][idx]
        config = benchmark['config']

        # PyTorch
        measurements.append({
            'name': f"PyTorch B={config['batch']}, L={config['seq_len']}",
            'operational_intensity': benchmark['pytorch']['operational_intensity'],
            'gflops': benchmark['pytorch']['gflops']
        })

        # CUDA
        measurements.append({
            'name': f"CUDA B={config['batch']}, L={config['seq_len']}",
            'operational_intensity': benchmark['cuda']['operational_intensity'],
            'gflops': benchmark['cuda']['gflops']
        })

    # Create roofline plot
    plot_roofline(
        peak_gflops=gpu_specs['peak_gflops'],
        peak_bandwidth_gb_s=gpu_specs['peak_bandwidth_gb_s'],
        measurements=measurements,
        title=f"Naive CUDA vs PyTorch Roofline - {gpu_name}",
        output_file="results/naive_roofline.png"
    )


def main():
    """Main profiling function"""

    if not CUDA_AVAILABLE:
        return

    # Run benchmarks
    results = run_all_benchmarks()

    # Create plots
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)

    create_comparison_plots(results)
    create_roofline_comparison(results)

    print("\n" + "="*80)
    print("Profiling Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
