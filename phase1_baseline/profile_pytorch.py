"""
Phase 1: Profiling PyTorch Transformer
Measure performance metrics for baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import time
import json
from pytorch_transformer import Transformer, MultiHeadAttention, create_causal_mask


def get_gpu_info():
    """Get GPU information"""
    if not torch.cuda.is_available():
        return None

    info = {
        'name': torch.cuda.get_device_name(0),
        'compute_capability': torch.cuda.get_device_capability(0),
        'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'cuda_version': torch.version.cuda
    }
    return info


def profile_attention_forward(batch_size, seq_len, hidden_dim, num_heads, num_runs=100, warmup=10):
    """Profile MultiHeadAttention forward pass"""
    device = 'cuda'

    # Create model and data
    attn = MultiHeadAttention(hidden_dim, num_heads).to(device)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = attn(x)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Timing
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            output = attn(x)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000

    # Calculate FLOPs for attention
    # Q @ K^T: 2 * batch * num_heads * seq_len * seq_len * head_dim
    # Attention @ V: 2 * batch * num_heads * seq_len * seq_len * head_dim
    # Plus linear projections: 4 * (2 * batch * seq_len * hidden_dim * hidden_dim)
    head_dim = hidden_dim // num_heads
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    proj_flops = 4 * (2 * batch_size * seq_len * hidden_dim * hidden_dim)
    total_flops = qk_flops + av_flops + proj_flops

    gflops = (total_flops / 1e9) / (avg_time_ms / 1000)

    # Calculate memory usage (bytes)
    # Input: batch * seq_len * hidden_dim
    # Q, K, V: 3 * batch * seq_len * hidden_dim
    # Scores: batch * num_heads * seq_len * seq_len
    # Output: batch * seq_len * hidden_dim
    # Weights: 4 * hidden_dim * hidden_dim
    input_bytes = batch_size * seq_len * hidden_dim * 4
    qkv_bytes = 3 * batch_size * seq_len * hidden_dim * 4
    scores_bytes = batch_size * num_heads * seq_len * seq_len * 4
    output_bytes = batch_size * seq_len * hidden_dim * 4
    weights_bytes = 4 * hidden_dim * hidden_dim * 4

    total_bytes = input_bytes + qkv_bytes + scores_bytes + output_bytes

    # Operational intensity (FLOPs/Byte)
    operational_intensity = total_flops / total_bytes

    # Memory bandwidth (GB/s)
    bandwidth_gb_s = (total_bytes / 1e9) / (avg_time_ms / 1000)

    return {
        'time_ms': avg_time_ms,
        'gflops': gflops,
        'bandwidth_gb_s': bandwidth_gb_s,
        'operational_intensity': operational_intensity,
        'total_flops': total_flops,
        'total_bytes': total_bytes
    }


def profile_transformer_forward(batch_size, seq_len, vocab_size=10000, hidden_dim=512,
                                num_layers=4, num_heads=8, num_runs=50, warmup=5):
    """Profile full Transformer forward pass"""
    device = 'cuda:1'

    # Create model and data
    model = Transformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=2048,
        max_seq_len=512
    ).to(device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    mask = create_causal_mask(seq_len, device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids, mask)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Timing
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            output = model(input_ids, mask)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000

    return {
        'time_ms': avg_time_ms,
        'throughput_samples_per_sec': 1000 / avg_time_ms,
        'throughput_tokens_per_sec': (batch_size * seq_len * 1000) / avg_time_ms
    }


def benchmark_with_torch_profiler(batch_size=4, seq_len=128, hidden_dim=512, num_heads=8):
    """Use torch.profiler for detailed profiling"""
    device = 'cuda'

    attn = MultiHeadAttention(hidden_dim, num_heads).to(device)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = attn(x)

    # Profile with torch.profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            _ = attn(x)

    # Print profiling results
    print("\n" + "="*80)
    print("PyTorch Profiler Results (Top 10 operations by CUDA time)")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return prof


def run_all_benchmarks():
    """Run all benchmarks and save results"""

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print("="*80)
    print("Phase 1: PyTorch Baseline Profiling")
    print("="*80)

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
    print(f"CUDA Version: {gpu_info['cuda_version']}")

    # Test configurations
    configs = [
        {'batch_size': 1, 'seq_len': 128, 'hidden_dim': 512, 'num_heads': 8},
        {'batch_size': 4, 'seq_len': 128, 'hidden_dim': 512, 'num_heads': 8},
        {'batch_size': 8, 'seq_len': 128, 'hidden_dim': 512, 'num_heads': 8},
        {'batch_size': 4, 'seq_len': 256, 'hidden_dim': 512, 'num_heads': 8},
        {'batch_size': 4, 'seq_len': 512, 'hidden_dim': 512, 'num_heads': 8},
    ]

    results = {
        'gpu_info': gpu_info,
        'attention_benchmarks': [],
        'transformer_benchmarks': []
    }

    # Profile attention
    print("\n" + "="*80)
    print("Multi-Head Attention Benchmarks")
    print("="*80)

    for config in configs:
        print(f"\nConfig: batch={config['batch_size']}, seq_len={config['seq_len']}, "
              f"hidden_dim={config['hidden_dim']}, num_heads={config['num_heads']}")

        metrics = profile_attention_forward(**config)

        print(f"  Time: {metrics['time_ms']:.3f} ms")
        print(f"  Performance: {metrics['gflops']:.2f} GFLOPS")
        print(f"  Bandwidth: {metrics['bandwidth_gb_s']:.2f} GB/s")
        print(f"  Operational Intensity: {metrics['operational_intensity']:.3f} FLOPs/Byte")

        results['attention_benchmarks'].append({
            'config': config,
            'metrics': metrics
        })

    # Profile full transformer
    print("\n" + "="*80)
    print("Full Transformer Benchmarks")
    print("="*80)

    transformer_configs = [
        {'batch_size': 1, 'seq_len': 128},
        {'batch_size': 4, 'seq_len': 128},
        {'batch_size': 4, 'seq_len': 256},
    ]

    for config in transformer_configs:
        print(f"\nConfig: batch={config['batch_size']}, seq_len={config['seq_len']}")

        metrics = profile_transformer_forward(**config)

        print(f"  Time: {metrics['time_ms']:.3f} ms")
        print(f"  Throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/sec")

        results['transformer_benchmarks'].append({
            'config': config,
            'metrics': metrics
        })

    # Run detailed profiling
    print("\n" + "="*80)
    print("Detailed Profiling (batch=4, seq_len=128)")
    print("="*80)
    benchmark_with_torch_profiler(batch_size=4, seq_len=128)

    # Save results
    output_file = 'results/baseline_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
