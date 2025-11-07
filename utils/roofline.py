"""
Roofline Model Visualization Utilities
"""

import matplotlib.pyplot as plt
import numpy as np
import json


def get_gpu_specs():
    """
    Get theoretical peak performance for common GPUs
    Returns dict with peak_gflops and peak_bandwidth_gb_s
    """
    # Common GPU specs (FP32)
    gpu_specs = {
        'V100': {'peak_gflops': 15700, 'peak_bandwidth_gb_s': 900},
        'A100': {'peak_gflops': 19500, 'peak_bandwidth_gb_s': 1555},
        'H100': {'peak_gflops': 51000, 'peak_bandwidth_gb_s': 3350},
        'RTX 3090': {'peak_gflops': 35580, 'peak_bandwidth_gb_s': 936},
        'RTX 4090': {'peak_gflops': 82580, 'peak_bandwidth_gb_s': 1008},
        'T4': {'peak_gflops': 8100, 'peak_bandwidth_gb_s': 320},
    }
    return gpu_specs


def plot_roofline(peak_gflops, peak_bandwidth_gb_s, measurements=None,
                  title="Roofline Model", output_file="roofline.png"):
    """
    Plot Roofline model with measurements

    Args:
        peak_gflops: Peak compute performance in GFLOPS
        peak_bandwidth_gb_s: Peak memory bandwidth in GB/s
        measurements: List of dicts with 'name', 'operational_intensity', 'gflops'
        title: Plot title
        output_file: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate operational intensity range
    intensities = np.logspace(-2, 3, 1000)

    # Memory-bound ceiling (linear with bandwidth)
    memory_bound = intensities * peak_bandwidth_gb_s

    # Compute-bound ceiling (flat at peak GFLOPS)
    compute_bound = np.ones_like(intensities) * peak_gflops

    # Roofline is the minimum of the two
    roofline = np.minimum(memory_bound, compute_bound)

    # Plot roofline
    ax.loglog(intensities, roofline, 'k-', linewidth=3, label='Roofline', zorder=2)

    # Fill areas
    ax.fill_between(intensities, 0, memory_bound,
                    where=(memory_bound <= compute_bound),
                    alpha=0.2, color='blue', label='Memory Bound Region')
    ax.fill_between(intensities, 0, compute_bound,
                    where=(compute_bound < memory_bound),
                    alpha=0.2, color='red', label='Compute Bound Region')

    # Ridge point (where memory and compute bounds meet)
    ridge_point = peak_gflops / peak_bandwidth_gb_s
    ax.axvline(x=ridge_point, color='gray', linestyle='--', linewidth=1.5,
               label=f'Ridge Point ({ridge_point:.2f} FLOPs/Byte)', zorder=1)

    # Plot measurements if provided
    if measurements:
        colors = plt.cm.tab10(np.linspace(0, 1, len(measurements)))
        for i, measurement in enumerate(measurements):
            oi = measurement['operational_intensity']
            gflops = measurement['gflops']
            name = measurement['name']

            ax.loglog(oi, gflops, 'o', markersize=10, color=colors[i],
                     label=name, zorder=3)

            # Calculate efficiency
            theoretical_max = min(oi * peak_bandwidth_gb_s, peak_gflops)
            efficiency = (gflops / theoretical_max) * 100

            # Add annotation
            ax.annotate(f'{name}\n{efficiency:.1f}% efficiency',
                       xy=(oi, gflops),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.5', fc=colors[i], alpha=0.3))

    # Labels and formatting
    ax.set_xlabel('Operational Intensity (FLOPs/Byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Set reasonable axis limits
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(1, peak_gflops * 2)

    # Add text with GPU specs
    info_text = f"Peak Compute: {peak_gflops:.0f} GFLOPS\nPeak Bandwidth: {peak_bandwidth_gb_s:.0f} GB/s"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Roofline plot saved to {output_file}")

    return fig, ax


def analyze_performance(operational_intensity, achieved_gflops,
                       peak_gflops, peak_bandwidth_gb_s):
    """
    Analyze performance and provide optimization suggestions

    Args:
        operational_intensity: Measured operational intensity (FLOPs/Byte)
        achieved_gflops: Achieved performance in GFLOPS
        peak_gflops: Peak compute performance
        peak_bandwidth_gb_s: Peak memory bandwidth

    Returns:
        dict with analysis results
    """
    ridge_point = peak_gflops / peak_bandwidth_gb_s

    # Determine if memory or compute bound
    theoretical_max = min(operational_intensity * peak_bandwidth_gb_s, peak_gflops)
    efficiency = (achieved_gflops / theoretical_max) * 100

    if operational_intensity < ridge_point:
        bottleneck = "Memory Bound"
        optimization_hint = "Focus on: data reuse, cache optimization, memory coalescing, kernel fusion"
    else:
        bottleneck = "Compute Bound"
        optimization_hint = "Focus on: ILP, better instruction mix, tensor cores"

    analysis = {
        'operational_intensity': operational_intensity,
        'achieved_gflops': achieved_gflops,
        'theoretical_max_gflops': theoretical_max,
        'efficiency_percent': efficiency,
        'bottleneck': bottleneck,
        'ridge_point': ridge_point,
        'distance_from_ridge': abs(operational_intensity - ridge_point),
        'optimization_suggestions': optimization_hint
    }

    return analysis


def print_analysis(analysis):
    """Pretty print performance analysis"""
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    print(f"Operational Intensity: {analysis['operational_intensity']:.3f} FLOPs/Byte")
    print(f"Achieved Performance: {analysis['achieved_gflops']:.2f} GFLOPS")
    print(f"Theoretical Maximum: {analysis['theoretical_max_gflops']:.2f} GFLOPS")
    print(f"Efficiency: {analysis['efficiency_percent']:.1f}%")
    print(f"\nBottleneck: {analysis['bottleneck']}")
    print(f"Ridge Point: {analysis['ridge_point']:.3f} FLOPs/Byte")
    print(f"\nOptimization Suggestions:")
    print(f"  {analysis['optimization_suggestions']}")
    print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Roofline Model Utility")
    print("="*70)

    # Example: NVIDIA A100 specs
    peak_gflops = 19500  # FP32
    peak_bandwidth_gb_s = 1555

    # Example measurements
    measurements = [
        {
            'name': 'PyTorch Baseline',
            'operational_intensity': 2.5,
            'gflops': 500
        },
        {
            'name': 'Naive CUDA',
            'operational_intensity': 1.8,
            'gflops': 350
        },
        {
            'name': 'Tiled CUDA',
            'operational_intensity': 5.2,
            'gflops': 1200
        }
    ]

    # Plot roofline
    plot_roofline(
        peak_gflops=peak_gflops,
        peak_bandwidth_gb_s=peak_bandwidth_gb_s,
        measurements=measurements,
        title="Attention Kernel Roofline Analysis",
        output_file="example_roofline.png"
    )

    # Analyze each measurement
    for measurement in measurements:
        print(f"\n{measurement['name']}:")
        analysis = analyze_performance(
            operational_intensity=measurement['operational_intensity'],
            achieved_gflops=measurement['gflops'],
            peak_gflops=peak_gflops,
            peak_bandwidth_gb_s=peak_bandwidth_gb_s
        )
        print_analysis(analysis)
