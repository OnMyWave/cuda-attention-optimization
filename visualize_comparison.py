"""
Visualization script for all phases comparison
Creates comprehensive plots from comparison results
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


def load_results(filepath='results/all_phases_comparison.json'):
    """Load comparison results from JSON"""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run compare_all_phases.py first!")
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def create_performance_comparison(results):
    """Create performance comparison bar chart"""
    comparisons = results['comparisons']

    # Extract data
    configs = []
    phase_names = ['Phase 1: PyTorch', 'Phase 2: Naive CUDA', 'Phase 3: Tiled', 'Phase 4: Optimized']
    times_by_phase = {name: [] for name in phase_names}

    for comp in comparisons:
        config = comp['config']
        config_label = f"B={config['batch']}\nL={config['seq_len']}\nH={config['head_dim']}"
        configs.append(config_label)

        for result in comp['results']:
            if result['name'] in phase_names and result['available']:
                times_by_phase[result['name']].append(result['time_ms'])

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(configs))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = []
    for i, (phase_name, color) in enumerate(zip(phase_names, colors)):
        offset = (i - 1.5) * width
        bar = ax.bar(x + offset, times_by_phase[phase_name], width,
                     label=phase_name, color=color, alpha=0.8)
        bars.append(bar)

    ax.set_xlabel('Configuration', fontweight='bold', fontsize=14)
    ax.set_ylabel('Time (ms)', fontweight='bold', fontsize=14)
    ax.set_title('Attention Performance Comparison - All Phases', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Performance comparison saved to results/performance_comparison.png")


def create_speedup_chart(results):
    """Create speedup comparison chart"""
    comparisons = results['comparisons']

    # Extract data
    configs = []
    phase_names = ['Phase 2: Naive CUDA', 'Phase 3: Tiled', 'Phase 4: Optimized']
    speedups_by_phase = {name: [] for name in phase_names}

    for comp in comparisons:
        config = comp['config']
        config_label = f"B={config['batch']}, L={config['seq_len']}, H={config['head_dim']}"
        configs.append(config_label)

        baseline_time = comp['baseline_time']

        for result in comp['results']:
            if result['name'] in phase_names and result['available']:
                speedup = baseline_time / result['time_ms']
                speedups_by_phase[result['name']].append(speedup)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(configs))
    width = 0.25
    colors = ['#ff7f0e', '#2ca02c', '#d62728']

    for i, (phase_name, color) in enumerate(zip(phase_names, colors)):
        offset = (i - 1) * width
        speedups = speedups_by_phase[phase_name]
        bars = ax.bar(x + offset, speedups, width, label=phase_name, color=color, alpha=0.8)

        # Add value labels on bars
        for j, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            label = f'{speedup:.2f}x' if speedup >= 0.1 else f'{speedup:.3f}x'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=8, rotation=0)

    # Add baseline line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='PyTorch Baseline (1x)')

    ax.set_xlabel('Configuration', fontweight='bold', fontsize=14)
    ax.set_ylabel('Speedup vs PyTorch', fontweight='bold', fontsize=14)
    ax.set_title('CUDA Kernel Speedup vs PyTorch Baseline', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('results/speedup_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Speedup comparison saved to results/speedup_comparison.png")


def create_efficiency_analysis(results):
    """Create GFLOP/s and bandwidth analysis"""
    comparisons = results['comparisons']

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Prepare data
    configs = []
    phase_names = ['Phase 1: PyTorch', 'Phase 2: Naive CUDA', 'Phase 3: Tiled', 'Phase 4: Optimized']
    gflops_by_phase = {name: [] for name in phase_names}
    bandwidth_by_phase = {name: [] for name in phase_names}

    for comp in comparisons:
        config = comp['config']
        config_label = f"B={config['batch']}, L={config['seq_len']}"
        configs.append(config_label)

        for result in comp['results']:
            if result['name'] in phase_names and result['available']:
                gflops_by_phase[result['name']].append(result.get('gflops', 0))
                bandwidth_by_phase[result['name']].append(result.get('bandwidth_gb_s', 0))

    x = np.arange(len(configs))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Plot 1: GFLOP/s
    ax = axes[0]
    for i, (phase_name, color) in enumerate(zip(phase_names, colors)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, gflops_by_phase[phase_name], width,
               label=phase_name, color=color, alpha=0.8)

    ax.set_ylabel('GFLOP/s', fontweight='bold', fontsize=14)
    ax.set_title('Computational Throughput (GFLOP/s)', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Bandwidth
    ax = axes[1]
    for i, (phase_name, color) in enumerate(zip(phase_names, colors)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, bandwidth_by_phase[phase_name], width,
               label=phase_name, color=color, alpha=0.8)

    ax.set_xlabel('Configuration', fontweight='bold', fontsize=14)
    ax.set_ylabel('Bandwidth (GB/s)', fontweight='bold', fontsize=14)
    ax.set_title('Memory Bandwidth Utilization', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('results/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Efficiency analysis saved to results/efficiency_analysis.png")


def create_scaling_analysis(results):
    """Analyze how performance scales with problem size"""
    comparisons = results['comparisons']

    # Group by batch size
    batch_groups = {}
    for comp in comparisons:
        batch = comp['config']['batch']
        if batch not in batch_groups:
            batch_groups[batch] = []
        batch_groups[batch].append(comp)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    phase_names = ['Phase 1: PyTorch', 'Phase 2: Naive CUDA', 'Phase 3: Tiled', 'Phase 4: Optimized']

    # Plot 1: Time vs Sequence Length for each batch size
    ax = axes[0]
    for batch in sorted(batch_groups.keys()):
        data = sorted(batch_groups[batch], key=lambda x: x['config']['seq_len'])
        seq_lens = [d['config']['seq_len'] for d in data]

        for i, phase_name in enumerate(phase_names):
            times = []
            for d in data:
                for result in d['results']:
                    if result['name'] == phase_name and result['available']:
                        times.append(result['time_ms'])
                        break

            if times:
                marker = 'o' if i == 0 else 's' if i == 1 else '^' if i == 2 else 'D'
                linestyle = '--' if i == 0 else '-'
                ax.plot(seq_lens, times, marker=marker, linestyle=linestyle,
                       label=f"{phase_name.split(':')[1].strip()} (B={batch})",
                       color=colors[i], linewidth=2, markersize=6, alpha=0.8)

    ax.set_xlabel('Sequence Length', fontweight='bold', fontsize=12)
    ax.set_ylabel('Time (ms)', fontweight='bold', fontsize=12)
    ax.set_title('Performance Scaling with Sequence Length', fontweight='bold', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Speedup vs Sequence Length
    ax = axes[1]
    for batch in sorted(batch_groups.keys()):
        data = sorted(batch_groups[batch], key=lambda x: x['config']['seq_len'])
        seq_lens = [d['config']['seq_len'] for d in data]

        for i, phase_name in enumerate(['Phase 3: Tiled', 'Phase 4: Optimized']):
            speedups = []
            for d in data:
                baseline_time = d['baseline_time']
                for result in d['results']:
                    if result['name'] == phase_name and result['available']:
                        speedups.append(baseline_time / result['time_ms'])
                        break

            if speedups:
                marker = '^' if i == 0 else 'D'
                ax.plot(seq_lens, speedups, marker=marker, linestyle='-',
                       label=f"{phase_name.split(':')[1].strip()} (B={batch})",
                       linewidth=2, markersize=8, alpha=0.8)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (1x)')
    ax.set_xlabel('Sequence Length', fontweight='bold', fontsize=12)
    ax.set_ylabel('Speedup vs PyTorch', fontweight='bold', fontsize=12)
    ax.set_title('Speedup Scaling with Sequence Length', fontweight='bold', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Scaling analysis saved to results/scaling_analysis.png")


def create_summary_table(results):
    """Create a summary table visualization"""
    comparisons = results['comparisons']

    # Calculate overall statistics
    phase_names = ['Phase 2: Naive CUDA', 'Phase 3: Tiled', 'Phase 4: Optimized']
    stats = {name: {'speedups': [], 'times': []} for name in phase_names}

    for comp in comparisons:
        baseline_time = comp['baseline_time']
        for result in comp['results']:
            if result['name'] in phase_names and result['available']:
                speedup = baseline_time / result['time_ms']
                stats[result['name']]['speedups'].append(speedup)
                stats[result['name']]['times'].append(result['time_ms'])

    # Create summary plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Title
    title_text = "CUDA Attention Optimization - Performance Summary\n"
    title_text += f"GPU: {results['gpu_info']['name']}\n"
    title_text += f"CUDA Version: {results['gpu_info']['cuda_version']}\n"
    title_text += f"Total Configurations Tested: {len(comparisons)}"

    ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=14,
            fontweight='bold', transform=ax.transAxes)

    # Statistics table
    table_data = []
    headers = ['Phase', 'Avg Speedup', 'Max Speedup', 'Min Speedup',
               'Avg Time (ms)', 'Best Config']

    for phase_name in phase_names:
        speedups = stats[phase_name]['speedups']
        times = stats[phase_name]['times']

        if speedups:
            avg_speedup = np.mean(speedups)
            max_speedup = np.max(speedups)
            min_speedup = np.min(speedups)
            avg_time = np.mean(times)

            # Find best config
            best_idx = np.argmax(speedups)
            best_config = comparisons[best_idx]['config']
            best_config_str = f"B={best_config['batch']}, L={best_config['seq_len']}"

            table_data.append([
                phase_name.split(':')[1].strip(),
                f'{avg_speedup:.2f}x',
                f'{max_speedup:.2f}x',
                f'{min_speedup:.2f}x',
                f'{avg_time:.3f}',
                best_config_str
            ])

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0.05, 0.3, 0.9, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Color code rows
    colors = ['#FFE0B2', '#C8E6C9', '#FFCDD2']
    for i, row in enumerate(table_data):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors[i])

    # Add notes
    notes = "Note: Speedup values > 1.0 indicate faster than PyTorch baseline.\n"
    notes += "Negative speedups (<1.0) occur due to kernel launch overhead on small problem sizes."
    ax.text(0.5, 0.15, notes, ha='center', va='top', fontsize=10,
            style='italic', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight')
    print("✓ Summary table saved to results/summary_table.png")


def create_all_visualizations():
    """Create all visualizations"""
    print("="*80)
    print("Creating Visualizations for All Phases Comparison")
    print("="*80)

    # Load results
    results = load_results()
    if results is None:
        return

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Create all visualizations
    print("\nGenerating plots...")
    create_performance_comparison(results)
    create_speedup_chart(results)
    create_efficiency_analysis(results)
    create_scaling_analysis(results)
    create_summary_table(results)

    print("\n" + "="*80)
    print("All visualizations created successfully!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/performance_comparison.png")
    print("  - results/speedup_comparison.png")
    print("  - results/efficiency_analysis.png")
    print("  - results/scaling_analysis.png")
    print("  - results/summary_table.png")
    print("\n" + "="*80)


if __name__ == "__main__":
    create_all_visualizations()
