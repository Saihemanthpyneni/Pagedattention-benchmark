"""
Generate Individual Publication-Ready Figures for vLLM Benchmark Report

"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = 'results/figures'
os.makedirs(output_dir, exist_ok=True)

# Color scheme
COLOR_HF = '#4472C4'      # Blue for HuggingFace
COLOR_VLLM = '#ED7D31'    # Orange for vLLM
COLOR_BEST = '#70AD47'    # Green for best result

# ============================================================================
# FIGURE 1: Throughput vs Batch Size (Bar Chart)
# ============================================================================
def create_figure1():
    """Create Figure 1: Throughput comparison across batch sizes"""
    
    batch_sizes = [1, 2, 4, 8, 16]
    hf_throughput = [42.1, 78.3, 124.6, 163.2, 171.8]
    vllm_throughput = [44.8, 87.5, 168.4, 298.7, 421.3]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hf_throughput, width, label='HuggingFace baseline', 
                   color=COLOR_HF, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, vllm_throughput, width, label='vLLM (PagedAttention)', 
                   color=COLOR_VLLM, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 1: Throughput (tokens/sec) vs Batch Size\nvLLM vs HuggingFace Baseline. OPT-1.3B, RTX 4060', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylim([0, 450])  # ✅ CORRECTED: Was 300, now 450 to match presentation
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure1_throughput_batch.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure1_throughput_batch.png")
    plt.close()

# ============================================================================
# FIGURE 2: Throughput Trend (Line Chart)
# ============================================================================
def create_figure2():
    """Create Figure 2: Throughput trend showing scaling"""
    
    batch_sizes = [1, 2, 4, 8, 16]
    hf_throughput = [42.1, 78.3, 124.6, 163.2, 171.8]
    vllm_throughput = [44.8, 87.5, 168.4, 298.7, 421.3]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(batch_sizes, hf_throughput, marker='o', linewidth=2.5, markersize=10,
           label='HuggingFace baseline', color=COLOR_HF)
    ax.plot(batch_sizes, vllm_throughput, marker='s', linewidth=2.5, markersize=10,
           label='vLLM (PagedAttention)', color=COLOR_VLLM)
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Throughput Trend\nvLLM Advantage Widens as Batch Size Grows. HF Baseline Plateaus due to Memory Fragmentation', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(batch_sizes)
    ax.set_ylim([0, 450])  # ✅ CORRECTED: Was 300, now 450 to match presentation
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure2_throughput_trend.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure2_throughput_trend.png")
    plt.close()

# ============================================================================
# FIGURE 3: GPU Memory Usage (Bar Chart with Limit Line)
# ============================================================================
def create_figure3():
    """Create Figure 3: Peak GPU memory usage comparison"""
    
    batch_sizes = [1, 2, 4, 8, 16]
    hf_memory = [3.12, 3.98, 5.61, 7.44, 7.98]
    vllm_memory = [3.05, 3.42, 4.18, 5.31, 6.24]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hf_memory, width, label='HuggingFace baseline',
                   color=COLOR_HF, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, vllm_memory, width, label='vLLM (PagedAttention)',
                   color=COLOR_BEST, edgecolor='black', linewidth=1.2)
    
    # Add VRAM limit line
    ax.axhline(y=8, color='red', linestyle='--', linewidth=2.5, label='RTX 4060 VRAM limit (8 GB)')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak GPU Memory (GB)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: Peak GPU Memory Usage\nHF Baseline Hits the 8 GB VRAM Limit at Batch=16. vLLM Stays within Safe Range', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylim([0, 9])
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure3_memory.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure3_memory.png")
    plt.close()

# ============================================================================
# FIGURE 4: Latency vs Sequence Length (Line Chart)
# ============================================================================
def create_figure4():
    """Create Figure 4: Latency comparison across sequence lengths"""
    
    seq_lengths = [64, 128, 256, 512]
    hf_latency = [8.24, 9.73, 12.18, 14.37]
    vllm_latency = [7.64, 8.60, 9.57, 9.76]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(seq_lengths, hf_latency, marker='o', linewidth=2.5, markersize=10,
           label='HuggingFace baseline', color=COLOR_HF)
    ax.plot(seq_lengths, vllm_latency, marker='s', linewidth=2.5, markersize=10,
           label='vLLM (PagedAttention)', color=COLOR_VLLM)
    
    ax.set_xlabel('Input Sequence Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms/token)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 4: Latency vs Input Sequence Length (Batch=4)\nvLLM Advantage Grows for Longer Sequences: 1.08× at 64 Tokens, 1.47× at 512 Tokens', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(seq_lengths)
    ax.set_ylim([0, 15])  # ✅ CORRECTED: Was 12, now 15 to accommodate 14.37 value
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure4_latency.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure4_latency.png")
    plt.close()

# ============================================================================
# FIGURE 5: Block Size Ablation (Bar Chart)
# ============================================================================
def create_figure5():
    """Create Figure 5: Block size ablation study"""
    
    block_sizes = [8, 16, 32]
    block_throughput = [318.2, 421.3, 389.7]  # ✅ CORRECTED: Using presentation values
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color bars - highlight best (block 16)
    colors = [COLOR_HF, COLOR_VLLM, COLOR_HF]
    bars = ax.bar(block_sizes, block_throughput, width=5, color=colors, 
                  edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for i, (bs, tp) in enumerate(zip(block_sizes, block_throughput)):
        ax.text(bs, tp + 5, f'{tp:.1f}', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
    
    # Add annotation for best
    ax.annotate('Best (block=16)', xy=(16, 421.3), xytext=(20, 440),
               arrowprops=dict(arrowstyle='->', color='orange', lw=2),
               fontsize=11, fontweight='bold', color='orange')
    
    ax.set_xlabel('Block Size (tokens per KV block)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5: Block Size Ablation Study\nBlock=16 Gives Best Throughput: Small Enough to Avoid Fragmentation, Large Enough to Use GPU Parallelism', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(block_sizes)
    ax.set_ylim([0, 450])  # ✅ CORRECTED: Was 300, now 450 to match presentation
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure5_blocksize.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure5_blocksize.png")
    plt.close()

# ============================================================================
# FIGURE 6: Speedup Ratio (Bar Chart with Progressive Colors)
# ============================================================================
def create_figure6():
    """Create Figure 6: Speedup ratio showing progressive improvement"""
    
    batch_sizes = [1, 2, 4, 8, 16]
    speedups = [1.06, 1.12, 1.35, 1.83, 2.45]
    
    # Progressive color scheme
    colors = ['#4472C4', '#5A7FD4', '#ED7D31', '#F59D42', '#70AD47']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(batch_sizes, speedups, width=0.6, color=colors, 
                  edgecolor='black', linewidth=1.2)
    
    # Add baseline reference line
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2.5, label='Baseline (1×)')
    
    # Add value labels
    for i, (bs, sp) in enumerate(zip(batch_sizes, speedups)):
        ax.text(bs, sp + 0.08, f'{sp:.2f}×', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (vLLM / HF Baseline)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 6: Speedup Ratio (vLLM / HF Baseline) vs Batch Size\nProgressive Improvement from 1.06× to 2.45×. Consistent with Paper\'s Reported 2–4× on A100 Hardware', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(batch_sizes)
    ax.set_ylim([0, 3.0])  # ✅ CORRECTED: Was 2.5, now 3.0 to match presentation
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure6_speedup.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: Figure6_speedup.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("Generating 6 Individual Publication-Ready Figures")
    print("=" * 80)
    print()
    
    # Generate all figures
    create_figure1()
    create_figure2()
    create_figure3()
    create_figure4()
    create_figure5()
    create_figure6()
    
    print()
    print("=" * 80)
    print("✅ All figures generated successfully!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  1. Figure1_throughput_batch.png")
    print("  2. Figure2_throughput_trend.png")
    print("  3. Figure3_memory.png")
    print("  4. Figure4_latency.png")
    print("  5. Figure5_blocksize.png")
    print("  6. Figure6_speedup.png")
    print()
    print(f"Location: {output_dir}/")
    print()
    print("Ready to use in LaTeX report!")
    print("=" * 80)
