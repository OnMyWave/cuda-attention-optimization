"""
Unified comparison script for all phases
Tests each phase with the same configurations for fair comparison
"""

import torch
import torch.nn.functional as F
import math
import time
import json
import sys
import os

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'phase2_naive'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'phase3_tiled'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'phase4_optimized'))

# Try to import each phase
try:
    import attention_cuda as attention_naive
    PHASE2_AVAILABLE = True
except ImportError:
    print("Warning: Phase 2 (naive) not available")
    PHASE2_AVAILABLE = False

try:
    import attention_tiled
    PHASE3_AVAILABLE = True
except ImportError:
    print("Warning: Phase 3 (tiled) not available")
    PHASE3_AVAILABLE = False

try:
    import attention_fused
    PHASE4_AVAILABLE = True
except ImportError:
    print("Warning: Phase 4 (optimized) not available")
    PHASE4_AVAILABLE = False


def pytorch_attention(Q, K, V):
    """Phase 1: PyTorch baseline"""
    batch, seq_len, head_dim = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output


def benchmark_attention(impl_fn, Q, K, V, name, num_runs=100, warmup=10):
    """Benchmark an attention implementation"""
    try:
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = impl_fn(Q, K, V)

        torch.cuda.synchronize()

        # Timing
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                output = impl_fn(Q, K, V)
        torch.cuda.synchronize()
        end = time.time()

        avg_time_ms = (end - start) / num_runs * 1000

        # Verify correctness
        with torch.no_grad():
            expected = pytorch_attention(Q, K, V)
            actual = impl_fn(Q, K, V)
            max_error = torch.abs(expected - actual).max().item()
            is_correct = torch.allclose(expected, actual, rtol=1e-3, atol=1e-4)

        return {
            'name': name,
            'time_ms': avg_time_ms,
            'max_error': max_error,
            'correct': is_correct,
            'available': True
        }
    except Exception as e:
        return {
            'name': name,
            'time_ms': float('inf'),
            'max_error': float('inf'),
            'correct': False,
            'available': False,
            'error': str(e)
        }


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

    bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

    return {
        'gflops': gflops,
        'bandwidth_gb_s': bandwidth_gb_s,
    }


def compare_all_phases(batch, seq_len, head_dim, num_runs=100):
    """Compare all available phases with the same configuration"""
    print(f"\n{'='*100}")
    print(f"Configuration: batch={batch}, seq_len={seq_len}, head_dim={head_dim}")
    print('='*100)

    device = 'cuda'
    Q = torch.randn(batch, seq_len, head_dim, device=device)
    K = torch.randn(batch, seq_len, head_dim, device=device)
    V = torch.randn(batch, seq_len, head_dim, device=device)

    results = []

    # Phase 1: PyTorch
    print("Benchmarking Phase 1: PyTorch Baseline...")
    result = benchmark_attention(pytorch_attention, Q, K, V, "Phase 1: PyTorch", num_runs)
    result.update(calculate_metrics(batch, seq_len, head_dim, result['time_ms']))
    results.append(result)
    baseline_time = result['time_ms']

    # Phase 2: Naive CUDA
    if PHASE2_AVAILABLE:
        print("Benchmarking Phase 2: Naive CUDA...")
        result = benchmark_attention(attention_naive.forward, Q, K, V, "Phase 2: Naive CUDA", num_runs)
        result.update(calculate_metrics(batch, seq_len, head_dim, result['time_ms']))
        results.append(result)
    else:
        results.append({'name': 'Phase 2: Naive CUDA', 'available': False})

    # Phase 3: Tiled
    if PHASE3_AVAILABLE:
        print("Benchmarking Phase 3: Tiled...")
        result = benchmark_attention(attention_tiled.forward, Q, K, V, "Phase 3: Tiled", num_runs)
        result.update(calculate_metrics(batch, seq_len, head_dim, result['time_ms']))
        results.append(result)
    else:
        results.append({'name': 'Phase 3: Tiled', 'available': False})

    # Phase 4: Optimized
    if PHASE4_AVAILABLE:
        print("Benchmarking Phase 4: Optimized...")
        result = benchmark_attention(attention_fused.forward, Q, K, V, "Phase 4: Optimized", num_runs)
        result.update(calculate_metrics(batch, seq_len, head_dim, result['time_ms']))
        results.append(result)
    else:
        results.append({'name': 'Phase 4: Optimized', 'available': False})

    # Print results table
    print(f"\n{'='*100}")
    print("RESULTS")
    print('='*100)

    table_data = []
    for r in results:
        if r['available']:
            speedup = baseline_time / r['time_ms']
            table_data.append([
                r['name'],
                f"{r['time_ms']:.4f}",
                f"{speedup:.2f}x",
                f"{r['gflops']:.2f}",
                f"{r['bandwidth_gb_s']:.2f}",
                f"{r['max_error']:.2e}",
                "✓" if r['correct'] else "✗"
            ])
        else:
            table_data.append([
                r['name'],
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            ])

    headers = ["Phase", "Time (ms)", "Speedup", "GFLOP/s", "BW (GB/s)", "Max Error", "Correct"]

    if TABULATE_AVAILABLE:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        # Simple table format
        col_widths = [25, 12, 10, 12, 12, 12, 8]

        # Print header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in table_data:
            row_line = " | ".join(str(val).ljust(w) for val, w in zip(row, col_widths))
            print(row_line)

    return {
        'config': {
            'batch': batch,
            'seq_len': seq_len,
            'head_dim': head_dim
        },
        'results': results,
        'baseline_time': baseline_time
    }


def run_all_comparisons():
    """Run comparisons for multiple configurations"""
    print("="*100)
    print("CUDA Attention Optimization - All Phases Comparison")
    print("="*100)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Test configurations (batch, seq_len, head_dim)
    configs = [
        (1, 128, 64),
        (4, 128, 64),
        (8, 128, 64),
        (4, 256, 64),
        (4, 512, 64),
        (4, 128, 128),
        (8, 256, 64),
    ]

    all_results = {
        'gpu_info': {
            'name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
        },
        'comparisons': []
    }

    for batch, seq_len, head_dim in configs:
        try:
            result = compare_all_phases(batch, seq_len, head_dim)
            all_results['comparisons'].append(result)
        except Exception as e:
            print(f"\nError with config batch={batch}, seq_len={seq_len}, head_dim={head_dim}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = 'results/all_phases_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*100}")
    print(f"Results saved to {output_file}")
    print('='*100)

    # Print summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print('='*100)

    for comparison in all_results['comparisons']:
        config = comparison['config']
        print(f"\nConfig: batch={config['batch']}, seq_len={config['seq_len']}, head_dim={config['head_dim']}")

        baseline = comparison['baseline_time']
        for result in comparison['results']:
            if result['available'] and result['name'] != 'Phase 1: PyTorch':
                speedup = baseline / result['time_ms']
                status = "✓" if result['correct'] else "✗"
                print(f"  {result['name']:25s}: {speedup:6.2f}x speedup  {status}")

    return all_results


if __name__ == "__main__":
    run_all_comparisons()
