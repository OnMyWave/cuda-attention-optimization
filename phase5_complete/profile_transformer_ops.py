"""
Profile LayerNorm and MLP CUDA implementations
Compare against PyTorch implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import transformer_ops
    CUDA_AVAILABLE = True
except ImportError:
    print("Error: transformer_ops module not found. Build it first:")
    print("  python setup.py install")
    CUDA_AVAILABLE = False
    sys.exit(1)

from utils.roofline import plot_roofline, analyze_performance, print_analysis


def pytorch_layer_norm(x, gamma, beta, eps=1e-5):
    """PyTorch LayerNorm reference"""
    batch, seq_len, hidden_dim = x.shape
    layer_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=True, device=x.device)
    layer_norm.weight.data = gamma.clone()
    layer_norm.bias.data = beta.clone()
    return layer_norm(x)


def pytorch_mlp(x, W1, b1, W2, b2):
    """PyTorch MLP reference"""
    h = F.linear(x, W1.t(), b1)
    h = F.gelu(h)
    output = F.linear(h, W2.t(), b2)
    return output


def benchmark_layer_norm(impl_fn, x, gamma, beta, eps, num_runs=100, warmup=10):
    """Benchmark LayerNorm implementation"""
    for _ in range(warmup):
        with torch.no_grad():
            _ = impl_fn(x, gamma, beta, eps)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            output = impl_fn(x, gamma, beta, eps)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000
    return {'time_ms': avg_time_ms}


def benchmark_mlp(impl_fn, x, W1, b1, W2, b2, num_runs=100, warmup=10):
    """Benchmark MLP implementation"""
    for _ in range(warmup):
        with torch.no_grad():
            _ = impl_fn(x, W1, b1, W2, b2)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            output = impl_fn(x, W1, b1, W2, b2)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000
    return {'time_ms': avg_time_ms}


def calculate_layer_norm_metrics(batch, seq_len, hidden_dim, time_ms):
    """Calculate FLOPs and memory metrics for LayerNorm"""
    # FLOPs: mean (1 add), variance (2 ops), normalize (3 ops), scale+shift (2 ops)
    # Total: ~8 ops per element
    total_elements = batch * seq_len * hidden_dim
    total_flops = 8 * total_elements

    gflops = (total_flops / 1e9) / (time_ms / 1000)

    # Memory: read input + gamma + beta, write output
    input_bytes = batch * seq_len * hidden_dim * 4
    params_bytes = 2 * hidden_dim * 4  # gamma + beta
    output_bytes = batch * seq_len * hidden_dim * 4
    total_bytes = input_bytes + params_bytes + output_bytes

    operational_intensity = total_flops / total_bytes
    bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

    return {
        'gflops': gflops,
        'bandwidth_gb_s': bandwidth_gb_s,
        'operational_intensity': operational_intensity,
        'total_flops': total_flops,
        'total_bytes': total_bytes
    }


def calculate_mlp_metrics(batch, seq_len, hidden_dim, ff_dim, time_ms):
    """Calculate FLOPs and memory metrics for MLP"""
    # Linear1: 2 * M * N * K FLOPs (where M=batch*seq_len, N=ff_dim, K=hidden_dim)
    linear1_flops = 2 * batch * seq_len * ff_dim * hidden_dim

    # GELU: ~8 ops per element
    gelu_flops = 8 * batch * seq_len * ff_dim

    # Linear2: 2 * M * N * K FLOPs (where M=batch*seq_len, N=hidden_dim, K=ff_dim)
    linear2_flops = 2 * batch * seq_len * hidden_dim * ff_dim

    total_flops = linear1_flops + gelu_flops + linear2_flops

    gflops = (total_flops / 1e9) / (time_ms / 1000)

    # Memory: read input, W1, b1, W2, b2, intermediate, write output
    input_bytes = batch * seq_len * hidden_dim * 4
    W1_bytes = hidden_dim * ff_dim * 4
    b1_bytes = ff_dim * 4
    intermediate_bytes = batch * seq_len * ff_dim * 4
    W2_bytes = ff_dim * hidden_dim * 4
    b2_bytes = hidden_dim * 4
    output_bytes = batch * seq_len * hidden_dim * 4

    total_bytes = input_bytes + W1_bytes + b1_bytes + intermediate_bytes + W2_bytes + b2_bytes + output_bytes

    operational_intensity = total_flops / total_bytes
    bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

    return {
        'gflops': gflops,
        'bandwidth_gb_s': bandwidth_gb_s,
        'operational_intensity': operational_intensity,
        'total_flops': total_flops,
        'total_bytes': total_bytes
    }


def profile_layer_norm(batch, seq_len, hidden_dim, num_runs=100):
    """Profile LayerNorm implementations"""
    print(f"\n{'='*80}")
    print(f"LayerNorm: batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim}")
    print('='*80)

    device = 'cuda'
    eps = 1e-5

    x = torch.randn(batch, seq_len, hidden_dim, device=device)
    gamma = torch.ones(hidden_dim, device=device)
    beta = torch.zeros(hidden_dim, device=device)

    results = {}

    # PyTorch
    print("\nBenchmarking PyTorch LayerNorm...")
    pytorch_result = benchmark_layer_norm(pytorch_layer_norm, x, gamma, beta, eps, num_runs)
    pytorch_metrics = calculate_layer_norm_metrics(batch, seq_len, hidden_dim, pytorch_result['time_ms'])
    results['pytorch'] = {**pytorch_result, **pytorch_metrics}

    # CUDA
    print("Benchmarking CUDA LayerNorm...")
    cuda_result = benchmark_layer_norm(transformer_ops.layer_norm, x, gamma, beta, eps, num_runs)
    cuda_metrics = calculate_layer_norm_metrics(batch, seq_len, hidden_dim, cuda_result['time_ms'])
    results['cuda'] = {**cuda_result, **cuda_metrics}

    # Print results
    print("\n" + "="*80)
    print("Results:")
    print("="*80)

    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Time:        {metrics['time_ms']:.4f} ms")
        print(f"  GFLOP/s:     {metrics['gflops']:.2f}")
        print(f"  Bandwidth:   {metrics['bandwidth_gb_s']:.2f} GB/s")
        print(f"  Op Intensity: {metrics['operational_intensity']:.2f} FLOPs/byte")

    if 'pytorch' in results and 'cuda' in results:
        speedup = results['pytorch']['time_ms'] / results['cuda']['time_ms']
        print(f"\nSpeedup (CUDA vs PyTorch): {speedup:.2f}x")

    return results


def profile_mlp(batch, seq_len, hidden_dim, ff_dim, num_runs=100):
    """Profile MLP implementations"""
    print(f"\n{'='*80}")
    print(f"MLP: batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim}, ff_dim={ff_dim}")
    print('='*80)

    device = 'cuda'

    x = torch.randn(batch, seq_len, hidden_dim, device=device)
    W1 = torch.randn(hidden_dim, ff_dim, device=device) * 0.02
    b1 = torch.zeros(ff_dim, device=device)
    W2 = torch.randn(ff_dim, hidden_dim, device=device) * 0.02
    b2 = torch.zeros(hidden_dim, device=device)

    results = {}

    # PyTorch
    print("\nBenchmarking PyTorch MLP...")
    pytorch_result = benchmark_mlp(pytorch_mlp, x, W1, b1, W2, b2, num_runs)
    pytorch_metrics = calculate_mlp_metrics(batch, seq_len, hidden_dim, ff_dim, pytorch_result['time_ms'])
    results['pytorch'] = {**pytorch_result, **pytorch_metrics}

    # CUDA
    print("Benchmarking CUDA MLP...")
    cuda_result = benchmark_mlp(transformer_ops.mlp_forward, x, W1, b1, W2, b2, num_runs)
    cuda_metrics = calculate_mlp_metrics(batch, seq_len, hidden_dim, ff_dim, cuda_result['time_ms'])
    results['cuda'] = {**cuda_result, **cuda_metrics}

    # Print results
    print("\n" + "="*80)
    print("Results:")
    print("="*80)

    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Time:        {metrics['time_ms']:.4f} ms")
        print(f"  GFLOP/s:     {metrics['gflops']:.2f}")
        print(f"  Bandwidth:   {metrics['bandwidth_gb_s']:.2f} GB/s")
        print(f"  Op Intensity: {metrics['operational_intensity']:.2f} FLOPs/byte")

    if 'pytorch' in results and 'cuda' in results:
        speedup = results['pytorch']['time_ms'] / results['cuda']['time_ms']
        print(f"\nSpeedup (CUDA vs PyTorch): {speedup:.2f}x")

    return results


def main():
    """Run profiling for various configurations"""
    print("="*80)
    print("Phase 5: Transformer Ops (LayerNorm, MLP) - Performance Profiling")
    print("="*80)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    all_results = {}

    # LayerNorm configurations
    print("\n" + "="*80)
    print("LAYERNORM PROFILING")
    print("="*80)

    ln_configs = [
        (4, 128, 512),
        (8, 256, 512),
        (4, 512, 768),
        (8, 128, 1024),
    ]

    all_results['layernorm'] = {}
    for batch, seq_len, hidden_dim in ln_configs:
        config_name = f"b{batch}_s{seq_len}_h{hidden_dim}"
        results = profile_layer_norm(batch, seq_len, hidden_dim)
        all_results['layernorm'][config_name] = results

    # MLP configurations
    print("\n" + "="*80)
    print("MLP PROFILING")
    print("="*80)

    mlp_configs = [
        (4, 128, 512, 2048),
        (8, 256, 512, 2048),
        (4, 512, 768, 3072),
        (8, 128, 1024, 4096),
    ]

    all_results['mlp'] = {}
    for batch, seq_len, hidden_dim, ff_dim in mlp_configs:
        config_name = f"b{batch}_s{seq_len}_h{hidden_dim}_f{ff_dim}"
        results = profile_mlp(batch, seq_len, hidden_dim, ff_dim)
        all_results['mlp'][config_name] = results

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/transformer_ops_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("Results saved to results/transformer_ops_metrics.json")
    print("="*80)


if __name__ == "__main__":
    main()
