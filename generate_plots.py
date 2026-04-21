#!/usr/bin/env python3
"""
Generate benchmark visualization plots for Wan2.2 Optimized Benchmark Suite.

Usage:
    python3 generate_plots.py [results/benchmark_results.json]

If no JSON file is provided, uses placeholder data structure.
After benchmarks complete, re-run with actual results JSON.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs('plots', exist_ok=True)

# ================================================================
# Color palette
# ================================================================
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'teal': '#009688',
    'grey': '#9E9E9E',
    'dark_blue': '#1565C0',
    'dark_green': '#2E7D32',
    'amber': '#FFC107',
    'indigo': '#3F51B5',
    'pink': '#E91E63',
    'cyan': '#00BCD4',
    'lime': '#CDDC39',
}

# Category colors
CAT_COLORS = {
    'baseline': COLORS['grey'],
    'sequence_parallelism': COLORS['blue'],
    'attention_backend': COLORS['orange'],
    'p2p_communication': COLORS['green'],
    'combined': COLORS['purple'],
    'caching': COLORS['teal'],
    'multi_host': COLORS['indigo'],
}

def load_results(json_path=None):
    """Load benchmark results from JSON. Returns placeholder if not available."""
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return None


def plot_01_optimization_comparison_t2v(results):
    """Bar chart comparing all optimizations for T2V."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Will be populated with actual data after benchmarks run
    # For now, create the structure
    configs = []
    times = []
    colors = []
    categories = []

    if results:
        for bench in results.get('benchmarks', []):
            bid = bench['benchmark_id']
            t2v = bench.get('t2v', {})
            if t2v and t2v.get('total_sec'):
                configs.append(bid.replace('_', '\n'))
                times.append(t2v['total_sec'])
                # Determine category
                cat = 'baseline'
                if 'sp_' in bid: cat = 'sequence_parallelism'
                elif 'sage' in bid: cat = 'attention_backend'
                elif 'p2p' in bid: cat = 'p2p_communication'
                elif 'cache' in bid: cat = 'caching'
                elif 'multi' in bid: cat = 'multi_host'
                if 'sage' in bid and 'sp' in bid: cat = 'combined'
                if 'p2p_sp_sage' in bid: cat = 'combined'
                colors.append(CAT_COLORS.get(cat, COLORS['grey']))
                categories.append(cat)

    if not configs:
        # Placeholder
        ax.text(0.5, 0.5, 'Run benchmarks first, then regenerate plots\n\npython3 generate_plots.py results/benchmark_results.json',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('T2V: Optimization Comparison\n(awaiting benchmark results)', fontsize=15, fontweight='bold')
    else:
        x = np.arange(len(configs))
        bars = ax.bar(x, times, color=colors, edgecolor='white', linewidth=0.5)

        # Add value labels
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{bar.get_height():.0f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Baseline reference line
        if times:
            baseline = times[0]  # First config should be baseline
            ax.axhline(y=baseline, color=COLORS['red'], linestyle='--', alpha=0.5, label=f'Baseline: {baseline:.0f}s')

        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8, rotation=45, ha='right')
        ax.set_ylabel('Total Generation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Wan2.2-A14B T2V: Optimization Comparison\n(720P, 81 frames, 40 steps, RTX PRO 6000)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('plots/01_t2v_optimization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 1: T2V optimization comparison")


def plot_02_optimization_comparison_i2v(results):
    """Bar chart comparing all optimizations for I2V."""
    fig, ax = plt.subplots(figsize=(14, 7))

    configs = []
    times = []
    colors = []

    if results:
        for bench in results.get('benchmarks', []):
            bid = bench['benchmark_id']
            i2v = bench.get('i2v', {})
            if i2v and i2v.get('total_sec'):
                configs.append(bid.replace('_', '\n'))
                times.append(i2v['total_sec'])
                cat = 'baseline'
                if 'sp_' in bid: cat = 'sequence_parallelism'
                elif 'sage' in bid: cat = 'attention_backend'
                elif 'p2p' in bid: cat = 'p2p_communication'
                elif 'cache' in bid: cat = 'caching'
                elif 'multi' in bid: cat = 'multi_host'
                if 'sage' in bid and 'sp' in bid: cat = 'combined'
                if 'p2p_sp_sage' in bid: cat = 'combined'
                colors.append(CAT_COLORS.get(cat, COLORS['grey']))

    if not configs:
        ax.text(0.5, 0.5, 'Run benchmarks first, then regenerate plots',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('I2V: Optimization Comparison\n(awaiting benchmark results)', fontsize=15, fontweight='bold')
    else:
        x = np.arange(len(configs))
        bars = ax.bar(x, times, color=colors, edgecolor='white', linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{bar.get_height():.0f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
        if times:
            ax.axhline(y=times[0], color=COLORS['red'], linestyle='--', alpha=0.5, label=f'Baseline: {times[0]:.0f}s')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8, rotation=45, ha='right')
        ax.set_ylabel('Total Generation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Wan2.2-A14B I2V: Optimization Comparison\n(720P, 81 frames, 40 steps, RTX PRO 6000)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('plots/02_i2v_optimization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 2: I2V optimization comparison")


def plot_03_speedup_by_optimization(results):
    """Horizontal bar chart showing speedup vs baseline for each optimization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    labels = []
    speedups = []
    colors = []
    baseline_time = None

    if results:
        for bench in results.get('benchmarks', []):
            bid = bench['benchmark_id']
            t2v = bench.get('t2v', {})
            if t2v and t2v.get('total_sec'):
                if 'baseline_tp4' in bid:
                    baseline_time = t2v['total_sec']
                    break

        if baseline_time:
            for bench in results.get('benchmarks', []):
                bid = bench['benchmark_id']
                t2v = bench.get('t2v', {})
                if t2v and t2v.get('total_sec') and 'baseline_tp4' not in bid:
                    speedup = baseline_time / t2v['total_sec']
                    labels.append(bid.replace('_', ' '))
                    speedups.append(speedup)
                    cat = 'baseline'
                    if 'sp_' in bid: cat = 'sequence_parallelism'
                    elif 'sage' in bid: cat = 'attention_backend'
                    elif 'p2p' in bid: cat = 'p2p_communication'
                    elif 'cache' in bid: cat = 'caching'
                    elif 'multi' in bid: cat = 'multi_host'
                    if 'sage' in bid and 'sp' in bid: cat = 'combined'
                    if 'p2p_sp_sage' in bid: cat = 'combined'
                    colors.append(CAT_COLORS.get(cat, COLORS['grey']))

    if not labels:
        ax.text(0.5, 0.5, 'Run benchmarks first, then regenerate plots',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
    else:
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, speedups, color=colors, edgecolor='white')
        for i, (bar, s) in enumerate(zip(bars, speedups)):
            ax.text(s + 0.02, i, f'{s:.2f}x', va='center', fontweight='bold', fontsize=10)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(x=1.0, color=COLORS['red'], linestyle='--', alpha=0.7, label='Baseline (1.0x)')
        ax.set_xlabel('Speedup vs Baseline TP=4', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)

    ax.set_title('T2V: Speedup by Optimization\n(vs Baseline TP=4, higher is better)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('plots/03_speedup_by_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 3: Speedup by optimization")


def plot_04_denoising_per_step(results):
    """Grouped bar chart of denoising time per step across configurations."""
    fig, ax = plt.subplots(figsize=(14, 6))

    configs = []
    t2v_steps = []
    i2v_steps = []

    if results:
        for bench in results.get('benchmarks', []):
            bid = bench['benchmark_id']
            t2v = bench.get('t2v', {})
            i2v = bench.get('i2v', {})
            t2v_step = t2v.get('denoising_sec_per_step') if t2v else None
            i2v_step = i2v.get('denoising_sec_per_step') if i2v else None
            if t2v_step or i2v_step:
                configs.append(bid.replace('_', '\n'))
                t2v_steps.append(t2v_step or 0)
                i2v_steps.append(i2v_step or 0)

    if not configs:
        ax.text(0.5, 0.5, 'Run benchmarks first, then regenerate plots',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
    else:
        x = np.arange(len(configs))
        width = 0.35
        bars1 = ax.bar(x - width/2, t2v_steps, width, label='T2V', color=COLORS['blue'])
        bars2 = ax.bar(x + width/2, i2v_steps, width, label='I2V', color=COLORS['orange'])
        for bar in list(bars1) + list(bars2):
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    ax.set_ylabel('Seconds per Denoising Step', fontsize=12, fontweight='bold')
    ax.set_title('Denoising Time per Step (40 steps total)\nWan2.2-A14B on RTX PRO 6000', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('plots/04_denoising_per_step.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 4: Denoising time per step")


def plot_05_optimization_category_impact(results):
    """Grouped comparison by optimization category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    categories = {
        'Baseline\n(TP only)': ['baseline_tp4', 'baseline_tp8'],
        'Sequence\nParallelism': ['sp_4gpu_u2r2', 'sp_8gpu_u4r2', 'sp_8gpu_u2r4'],
        'SageAttention': ['sage_attn_tp4', 'sage_attn_tp8'],
        'P2P\nCommunication': ['p2p_tp4', 'p2p_tp8'],
        'Combined\nOptimizations': ['sage_sp_4gpu', 'sage_sp_8gpu', 'p2p_sp_sage_8gpu'],
        'Cache-DiT\n(1 GPU)': ['cache_dit_1gpu'],
        'Multi-Host\n(16 GPU)': ['multihost_16gpu_torchrun'],
    }

    for ax_idx, (model_type, ax) in enumerate(zip(['t2v', 'i2v'], axes)):
        cat_names = []
        best_times = []
        bar_colors = []

        if results:
            bench_map = {b['benchmark_id']: b for b in results.get('benchmarks', [])}
            for cat_name, bench_ids in categories.items():
                best_time = None
                for bid in bench_ids:
                    bench = bench_map.get(bid)
                    if bench:
                        data = bench.get(model_type, {})
                        if data and data.get('total_sec'):
                            if best_time is None or data['total_sec'] < best_time:
                                best_time = data['total_sec']
                if best_time:
                    cat_names.append(cat_name)
                    best_times.append(best_time)
                    cat_key = list(categories.keys()).index(cat_name)
                    color_keys = list(CAT_COLORS.values())
                    bar_colors.append(color_keys[cat_key % len(color_keys)])

        if not cat_names:
            ax.text(0.5, 0.5, 'Awaiting results', ha='center', va='center', fontsize=12, transform=ax.transAxes)
        else:
            x = np.arange(len(cat_names))
            bars = ax.bar(x, best_times, color=bar_colors, edgecolor='white')
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                        f'{bar.get_height():.0f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(cat_names, fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        ax.set_ylabel('Best Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title(f'{"T2V" if model_type == "t2v" else "I2V"}: Best Time per Category', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Optimization Category Impact (Best Config per Category)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/05_category_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 5: Optimization category impact")


def plot_06_gpu_scaling(results):
    """GPU scaling chart: 1, 4, 8, 16 GPUs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    gpu_counts = [1, 4, 8, 16]
    # Map benchmark IDs to GPU counts for best optimized config
    gpu_map = {
        1: ['cache_dit_1gpu'],
        4: ['baseline_tp4', 'sp_4gpu_u2r2', 'sage_attn_tp4', 'sage_sp_4gpu', 'p2p_tp4'],
        8: ['baseline_tp8', 'sp_8gpu_u4r2', 'sage_attn_tp8', 'sage_sp_8gpu', 'p2p_tp8', 'p2p_sp_sage_8gpu'],
        16: ['multihost_16gpu_torchrun', 'multihost_16gpu_ulysses4_ring2'],
    }

    for model_type, ax in zip(['t2v', 'i2v'], [ax1, ax2]):
        baseline_times = {}
        optimized_times = {}

        if results:
            bench_map = {b['benchmark_id']: b for b in results.get('benchmarks', [])}
            for gpu_count, bench_ids in gpu_map.items():
                for bid in bench_ids:
                    bench = bench_map.get(bid)
                    if bench:
                        data = bench.get(model_type, {})
                        if data and data.get('total_sec'):
                            t = data['total_sec']
                            if 'baseline' in bid:
                                baseline_times[gpu_count] = t
                            if gpu_count not in optimized_times or t < optimized_times[gpu_count]:
                                optimized_times[gpu_count] = t

        if optimized_times:
            gpus = sorted(optimized_times.keys())
            opt_times = [optimized_times[g] for g in gpus]
            ax.plot(gpus, opt_times, 'o-', color=COLORS['green'], linewidth=3, markersize=10,
                    label='Best Optimized', markeredgecolor='white', markeredgewidth=2)

            if baseline_times:
                base_gpus = sorted(baseline_times.keys())
                base_t = [baseline_times[g] for g in base_gpus]
                ax.plot(base_gpus, base_t, 's--', color=COLORS['grey'], linewidth=2, markersize=8,
                        label='Baseline (TP only)')

            for g, t in zip(gpus, opt_times):
                ax.annotate(f'{t:.0f}s', (g, t), textcoords="offset points", xytext=(0, 12),
                            ha='center', fontweight='bold', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Awaiting results', ha='center', va='center', fontsize=12, transform=ax.transAxes)

        ax.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{"T2V" if model_type == "t2v" else "I2V"}: GPU Scaling', fontsize=13, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 4, 8, 16])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('GPU Scaling: Baseline vs Best Optimized\nWan2.2-A14B on G4 (RTX PRO 6000)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/06_gpu_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 6: GPU scaling")


def plot_07_optimization_legend():
    """Create a legend/reference plot showing all optimization techniques."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    optimizations = [
        ('Baseline (TP)', 'Tensor Parallelism only\n(original Google recipe)', CAT_COLORS['baseline']),
        ('Sequence Parallelism', 'USP: Ulysses + Ring attention\nReduces memory, improves scaling', CAT_COLORS['sequence_parallelism']),
        ('SageAttention', 'Optimized attention kernel\nfor Blackwell GPUs (SM120)', CAT_COLORS['attention_backend']),
        ('P2P Communication', 'NCCL_P2P_LEVEL=5\nG4 proprietary GPU P2P bypass', CAT_COLORS['p2p_communication']),
        ('Combined', 'SP + SageAttn + P2P\nAll optimizations together', CAT_COLORS['combined']),
        ('Cache-DiT', 'Block-level caching with TaylorSeer\nUp to 1.69x speedup (1-GPU)', CAT_COLORS['caching']),
        ('Multi-Host', 'torchrun across 2 nodes (16 GPU)\nUlysses sequence parallelism', CAT_COLORS['multi_host']),
    ]

    y_start = 0.92
    for i, (name, desc, color) in enumerate(optimizations):
        y = y_start - i * 0.13
        ax.add_patch(plt.Rectangle((0.02, y - 0.04), 0.04, 0.08, facecolor=color, edgecolor='white', linewidth=2, transform=ax.transAxes))
        ax.text(0.08, y, name, fontsize=13, fontweight='bold', va='center', transform=ax.transAxes)
        ax.text(0.35, y, desc, fontsize=10, va='center', transform=ax.transAxes, color='#555')

    ax.set_title('Optimization Techniques Applied', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('plots/07_optimization_legend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 7: Optimization legend")


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = load_results(results_path)

    if results:
        print(f"Loaded {len(results.get('benchmarks', []))} benchmark results from {results_path}")
    else:
        print("No results file provided. Generating placeholder plots.")
        print("After benchmarks complete, re-run: python3 generate_plots.py results/benchmark_results.json")

    plot_01_optimization_comparison_t2v(results)
    plot_02_optimization_comparison_i2v(results)
    plot_03_speedup_by_optimization(results)
    plot_04_denoising_per_step(results)
    plot_05_optimization_category_impact(results)
    plot_06_gpu_scaling(results)
    plot_07_optimization_legend()

    print("\n🎉 All plots generated in plots/ directory!")


if __name__ == "__main__":
    main()
