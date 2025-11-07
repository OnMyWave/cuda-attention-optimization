"""
Visualize Phase 1 baseline results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.roofline import plot_roofline, analyze_performance, print_analysis


def load_results(filename='results/baseline_metrics.json'):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_performance_comparison(results, output_file='results/performance_comparison.png'):
    """Plot performance metrics across different configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Extract data
    configs = [r['config'] for r in results['attention_benchmarks']]
    metrics = [r['metrics'] for r in results['attention_benchmarks']]

    labels = [f"B={c['batch_size']}, L={c['seq_len']}" for c in configs]
    times = [m['time_ms'] for m in metrics]
    gflops_list = [m['gflops'] for m in metrics]
    bandwidth_list = [m['bandwidth_gb_s'] for m in metrics]
    oi_list = [m['operational_intensity'] for m in metrics]

    # Plot 1: Execution Time
    ax = axes[0, 0]
    bars = ax.bar(range(len(labels)), times, color='steelblue', alpha=0.8)
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Execution Time Comparison', fontweight='bold', pad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: GFLOPS
    ax = axes[0, 1]
    bars = ax.bar(range(len(labels)), gflops_list, color='coral', alpha=0.8)
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontweight='bold')
    ax.set_title('Compute Performance', fontweight='bold', pad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.0f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Memory Bandwidth
    ax = axes[1, 0]
    bars = ax.bar(range(len(labels)), bandwidth_list, color='lightgreen', alpha=0.8)
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Bandwidth (GB/s)', fontweight='bold')
    ax.set_title('Memory Bandwidth Utilization', fontweight='bold', pad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.0f}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Operational Intensity
    ax = axes[1, 1]
    bars = ax.bar(range(len(labels)), oi_list, color='orchid', alpha=0.8)
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Operational Intensity (FLOPs/Byte)', fontweight='bold')
    ax.set_title('Operational Intensity', fontweight='bold', pad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Performance comparison saved to {output_file}")


def create_roofline_plot(results, output_file='results/baseline_roofline.png'):
    """Create roofline plot for baseline results"""

    # Estimate GPU specs (you may need to adjust these based on your actual GPU)
    gpu_name = results['gpu_info']['name']

    # Default specs (adjust based on your GPU)
    # These are approximate values - you should use actual specs for your GPU
    gpu_specs = {
        'peak_gflops': 10000,  # Placeholder - adjust for your GPU
        'peak_bandwidth_gb_s': 500  # Placeholder - adjust for your GPU
    }

    # Try to identify GPU and use appropriate specs
    if 'V100' in gpu_name:
        gpu_specs = {'peak_gflops': 15700, 'peak_bandwidth_gb_s': 900}
    elif 'A100' in gpu_name:
        gpu_specs = {'peak_gflops': 19500, 'peak_bandwidth_gb_s': 1555}
    elif 'H100' in gpu_name:
        gpu_specs = {'peak_gflops': 51000, 'peak_bandwidth_gb_s': 3350}
    elif '3090' in gpu_name or 'RTX 3090' in gpu_name:
        gpu_specs = {'peak_gflops': 35580, 'peak_bandwidth_gb_s': 936}
    elif '4090' in gpu_name or 'RTX 4090' in gpu_name:
        gpu_specs = {'peak_gflops': 82580, 'peak_bandwidth_gb_s': 1008}
    elif 'T4' in gpu_name:
        gpu_specs = {'peak_gflops': 8100, 'peak_bandwidth_gb_s': 320}

    print(f"\nUsing GPU specs: {gpu_specs}")
    print(f"(If these don't match your GPU, please adjust manually in the code)")

    # Prepare measurements
    measurements = []
    for i, benchmark in enumerate(results['attention_benchmarks']):
        config = benchmark['config']
        metrics = benchmark['metrics']

        name = f"B={config['batch_size']}, L={config['seq_len']}"
        measurements.append({
            'name': name,
            'operational_intensity': metrics['operational_intensity'],
            'gflops': metrics['gflops']
        })

    # Create roofline plot
    plot_roofline(
        peak_gflops=gpu_specs['peak_gflops'],
        peak_bandwidth_gb_s=gpu_specs['peak_bandwidth_gb_s'],
        measurements=measurements,
        title=f"PyTorch Baseline Roofline Analysis - {gpu_name}",
        output_file=output_file
    )

    # Analyze performance
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)

    for measurement in measurements:
        print(f"\n{measurement['name']}:")
        analysis = analyze_performance(
            operational_intensity=measurement['operational_intensity'],
            achieved_gflops=measurement['gflops'],
            peak_gflops=gpu_specs['peak_gflops'],
            peak_bandwidth_gb_s=gpu_specs['peak_bandwidth_gb_s']
        )
        print_analysis(analysis)


def main():
    """Main visualization function"""
    print("="*80)
    print("Phase 1 Baseline Results Visualization")
    print("="*80)

    # Load results
    try:
        results = load_results()
        print(f"✓ Loaded results from results/baseline_metrics.json")
    except FileNotFoundError:
        print("Error: results/baseline_metrics.json not found")
        print("Please run profile_pytorch.py first")
        return

    # Print summary
    print(f"\nGPU: {results['gpu_info']['name']}")
    print(f"CUDA Version: {results['gpu_info']['cuda_version']}")
    print(f"Total Memory: {results['gpu_info']['total_memory_gb']:.2f} GB")

    print(f"\nNumber of attention benchmarks: {len(results['attention_benchmarks'])}")
    print(f"Number of transformer benchmarks: {len(results['transformer_benchmarks'])}")

    # Create visualizations
    print("\nGenerating visualizations...")

    plot_performance_comparison(results)
    create_roofline_plot(results)

    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()
