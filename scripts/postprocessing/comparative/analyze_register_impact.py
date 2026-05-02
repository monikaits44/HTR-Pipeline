#!/usr/bin/env python3
"""
Analyze Register Impact on ViT-RGTS Performance

Compares the effect of different register token counts (0, 2, 4, 8, 16, ...)
on ViT-RGTS model performance using bar charts and statistical summaries.
For training curve plots (loss, CER, WER over epochs), use plot_training_metrics.py.

Supported Architectures: vit_rgts only (requires register count metadata)
Input: results.csv from each run directory
Output: 6-panel comparison figure + terminal statistical summary

Usage:
    # Compare ViT-RGTS v2 register configurations (runs 50-54)
    python scripts/postprocessing/analyze_register_impact.py \
        --model-path saved_models/experiments/run_50 --registers 0 \
        --model-path saved_models/experiments/run_51 --registers 2 \
        --model-path saved_models/experiments/run_52 --registers 4 \
        --model-path saved_models/experiments/run_53 --registers 8 \
        --model-path saved_models/experiments/run_54 --registers 16

    # Compare two configurations only
    python scripts/postprocessing/analyze_register_impact.py \
        --model-path saved_models/experiments/run_50 --registers 0 \
        --model-path saved_models/experiments/run_54 --registers 16

    # Show plots without saving
    python scripts/postprocessing/analyze_register_impact.py \
        --model-path saved_models/experiments/run_50 --registers 0 \
        --model-path saved_models/experiments/run_54 --registers 16 \
        --no-save

4 Visualizations (2×2 grid):
  1. Best CER vs Register Count    - Bar chart of best Test CER per config
  2. Best WER vs Register Count    - Bar chart of best Test WER per config
  3. Training Stability            - Loss std dev in last 5 epochs (lower = more stable)
  4. Summary Table                 - Registers / Best CER / Final Loss per config

Statistical Analysis (printed to terminal):
  - Best configuration identification
  - Fastest convergence analysis
  - Training stability comparison
  - Overfitting detection
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Professional color palette
COLOR_PALETTE = {
    'reg0': "#15bad0",    # Red - no registers
    'reg2': "#004359",    # Orange - 2 registers
    'reg4': "#34677d",    # Green - 4 registers
    'reg8': '#1f77b4',    # Blue - 8 registers
    'reg16': '#0a360a',   # Purple - 16 registers
    'reg32': "#4b4d8c",   # Brown - 32 registers
}

sns.set_style("white")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def load_results_with_registers(model_path: str, num_registers: int) -> Tuple[pd.DataFrame, str]:
    """Load results.csv and create label with register count."""
    results_path = os.path.join(model_path, 'results.csv')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_csv(results_path)
    
    # Create descriptive label
    model_name = Path(model_path).stem if Path(model_path).stem else Path(model_path).parent.stem
    if not model_name or model_name == 'experiments':
        model_name = os.path.basename(os.path.normpath(model_path))
    
    label = f"{num_registers} registers ({model_name})"
    
    return df, label, num_registers


def plot_register_performance_comparison(results_dict: Dict, output_dir: str = None):
    """Compare performance metrics across different register configurations.

    Generates a 2×2 grid: Best CER, Best WER, Training Stability, Summary Table.
    For training curve plots (loss, CER, WER over epochs), use plot_training_metrics.py.
    """
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    fig.suptitle('Register Impact Analysis: Performance Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # Sort by register count for consistent ordering
    sorted_items = sorted(results_dict.items(), key=lambda x: x[1]['num_registers'])
    reg_counts = [data['num_registers'] for _, data in sorted_items]
    colors_bar = [COLOR_PALETTE.get(f'reg{r}', plt.cm.tab10(r % 10)) for r in reg_counts]

    # ── Plot 1: Best Test CER vs Register Count ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    best_test_cers = []
    for _, data in sorted_items:
        df = data['df']
        if 'test/cer' in df.columns:
            best_test_cers.append(df['test/cer'].min() * 100)
        else:
            best_test_cers.append(np.nan)

    bars = ax1.bar(range(len(reg_counts)), best_test_cers, color=colors_bar,
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    ax1.set_xticks(range(len(reg_counts)))
    ax1.set_xticklabels([f'{r} regs' for r in reg_counts])
    ax1.set_ylabel('Best Test CER (%)', fontweight='bold')
    ax1.set_xlabel('Number of Registers', fontweight='bold')
    ax1.set_title('Best CER vs Register Count', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, val in zip(bars, best_test_cers):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{val:.2f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    # ── Plot 2: Best Test WER vs Register Count ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    best_test_wers = []
    for _, data in sorted_items:
        df = data['df']
        if 'test/wer' in df.columns:
            best_test_wers.append(df['test/wer'].min() * 100)
        else:
            best_test_wers.append(np.nan)

    bars = ax2.bar(range(len(reg_counts)), best_test_wers, color=colors_bar,
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    ax2.set_xticks(range(len(reg_counts)))
    ax2.set_xticklabels([f'{r} regs' for r in reg_counts])
    ax2.set_ylabel('Best Test WER (%)', fontweight='bold')
    ax2.set_xlabel('Number of Registers', fontweight='bold')
    ax2.set_title('Best WER vs Register Count', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar, val in zip(bars, best_test_wers):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{val:.2f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    # ── Plot 3: Training Stability — Loss Std Dev (Last 5 Epochs) ───────────
    ax3 = fig.add_subplot(gs[1, 0])
    loss_variances = []
    for _, data in sorted_items:
        df = data['df']
        loss_variances.append(df['train/ctc_loss'].tail(5).std())

    bars = ax3.bar(range(len(reg_counts)), loss_variances, color=colors_bar,
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    ax3.set_xticks(range(len(reg_counts)))
    ax3.set_xticklabels([f'{r} regs' for r in reg_counts])
    ax3.set_ylabel('Loss Std Dev (Last 5 Epochs)', fontweight='bold')
    ax3.set_xlabel('Number of Registers', fontweight='bold')
    ax3.set_title('Training Stability (Lower is Better)', fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ── Plot 4: Summary Statistics Table ────────────────────────────────────
    ax9 = fig.add_subplot(gs[1, 1])
    ax9.axis('off')
    
    # Create summary table
    table_data = []
    for label, data in sorted_items:
        df = data['df']
        num_regs = data['num_registers']
        
        best_test_cer = df['test/cer'].min() * 100 if 'test/cer' in df.columns else np.nan
        final_loss = df['train/ctc_loss'].iloc[-1]
        
        table_data.append([
            f"{num_regs}",
            f"{best_test_cer:.2f}%" if not np.isnan(best_test_cer) else "N/A",
            f"{final_loss:.2f}"
        ])
    
    table = ax9.table(cellText=table_data,
                     colLabels=['Registers', 'Best CER', 'Final Loss'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#002f6c')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by register count
    for i, (_, data) in enumerate(sorted_items, start=1):
        num_regs = data['num_registers']
        color = COLOR_PALETTE.get(f'reg{num_regs}', '#ffffff')
        for j in range(3):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)
    
    ax9.set_title('Performance Summary', fontweight='bold', pad=20, fontsize=12)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, 'register_impact_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_register_analysis(results_dict: Dict, output_dir: str = None):
    """Save detailed statistical analysis of register impact to a text file."""
    sorted_items = sorted(results_dict.items(), key=lambda x: x[1]['num_registers'])

    lines = []
    lines.append("=" * 80)
    lines.append("REGISTER IMPACT ANALYSIS REPORT")
    lines.append("=" * 80)

    # ── Per-configuration metrics table ─────────────────────────────────────
    lines.append("")
    header = (f"{'Registers':<12} {'Best CER':<12} {'Final CER':<12}"
              f" {'Best WER':<12} {'Final WER':<12}"
              f" {'Epochs<10%':<12} {'Overfit Gap':<12} {'Loss StdDev':<12}")
    lines.append(header)
    lines.append("-" * len(header))

    for label, data in sorted_items:
        df = data['df']
        num_regs = data['num_registers']

        best_cer  = df['test/cer'].min() * 100  if 'test/cer' in df.columns else float('nan')
        final_cer = df['test/cer'].iloc[-1] * 100 if 'test/cer' in df.columns else float('nan')
        best_wer  = df['test/wer'].min() * 100  if 'test/wer' in df.columns else float('nan')
        final_wer = df['test/wer'].iloc[-1] * 100 if 'test/wer' in df.columns else float('nan')
        overfit   = final_cer - best_cer if not np.isnan(best_cer) else float('nan')
        loss_std  = df['train/ctc_loss'].tail(5).std()

        below_10 = df[df['test/cer'] <= 0.10] if 'test/cer' in df.columns else pd.DataFrame()
        epochs_to_10 = str(int(below_10.iloc[0]['epoch'])) if len(below_10) > 0 else '>max'

        def fmt(v, suffix='%'): return f'{v:.2f}{suffix}' if not np.isnan(v) else 'N/A'

        lines.append(
            f"{num_regs:<12} {fmt(best_cer):<12} {fmt(final_cer):<12}"
            f" {fmt(best_wer):<12} {fmt(final_wer):<12}"
            f" {epochs_to_10:<12} {fmt(overfit):<12} {loss_std:<12.4f}"
        )

    # ── Key insights ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 80)
    lines.append("KEY INSIGHTS")
    lines.append("=" * 80)

    # Best CER configuration
    valid_cer = [(l, d) for l, d in sorted_items if 'test/cer' in d['df'].columns]
    if valid_cer:
        best = min(valid_cer, key=lambda x: x[1]['df']['test/cer'].min())
        lines.append(f"\n[+] Best CER Configuration : {best[1]['num_registers']} registers"
                     f"  (Test CER = {best[1]['df']['test/cer'].min()*100:.2f}%)")

    # Best WER configuration
    valid_wer = [(l, d) for l, d in sorted_items if 'test/wer' in d['df'].columns]
    if valid_wer:
        best_w = min(valid_wer, key=lambda x: x[1]['df']['test/wer'].min())
        lines.append(f"[+] Best WER Configuration : {best_w[1]['num_registers']} registers"
                     f"  (Test WER = {best_w[1]['df']['test/wer'].min()*100:.2f}%)")

    # Fastest convergence
    fastest_epoch = float('inf')
    fastest_config = None
    for _, data in valid_cer:
        below = data['df'][data['df']['test/cer'] <= 0.10]
        if len(below) > 0:
            ep = below.iloc[0]['epoch']
            if ep < fastest_epoch:
                fastest_epoch, fastest_config = ep, data['num_registers']
    if fastest_config is not None:
        lines.append(f"[+] Fastest Convergence    : {fastest_config} registers"
                     f"  (reached <10% CER at epoch {int(fastest_epoch)})")

    # Most stable training
    stabilities = [(d['num_registers'], d['df']['train/ctc_loss'].tail(5).std())
                   for _, d in sorted_items]
    most_stable = min(stabilities, key=lambda x: x[1])
    lines.append(f"[+] Most Stable Training   : {most_stable[0]} registers"
                 f"  (loss std dev = {most_stable[1]:.4f})")

    # Lowest overfitting
    if valid_cer:
        gaps = [(d['num_registers'],
                 (d['df']['test/cer'].iloc[-1] - d['df']['test/cer'].min()) * 100)
                for _, d in valid_cer]
        least_overfit = min(gaps, key=lambda x: abs(x[1]))
        lines.append(f"[+] Least Overfitting      : {least_overfit[0]} registers"
                     f"  (CER gap = {least_overfit[1]:.2f}%)")

    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    # Print to terminal
    print(report_text)

    # Save to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'register_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text + "\n")
        print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Register Impact on ViT-RGTS Performance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model-path',
        type=str,
        action='append',
        required=True,
        help='Path to model directory (can be specified multiple times)'
    )
    parser.add_argument(
        '--registers',
        type=int,
        action='append',
        required=True,
        help='Number of registers for corresponding model (must match --model-path order)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./visualizations/register_analysis',
        help='Directory to save analysis plots'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save plots (show only)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.model_path) != len(args.registers):
        print("Error: Number of --model-path and --registers must match!")
        return 1
    
    # Create output directory
    output_dir = None if args.no_save else args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Load all results
    results_dict = {}
    print("\n" + "="*80)
    print("LOADING EXPERIMENT RESULTS")
    print("="*80)
    
    for model_path, num_regs in zip(args.model_path, args.registers):
        try:
            df, label, num_registers = load_results_with_registers(model_path, num_regs)
            results_dict[label] = {
                'df': df,
                'num_registers': num_registers,
                'path': model_path
            }
            print(f"✓ Loaded: {label} ({len(df)} epochs)")
        except Exception as e:
            print(f"✗ Error loading {model_path}: {e}")
    
    if not results_dict:
        print("Error: No valid results found!")
        return 1
    
    # Print detailed analysis
    save_register_analysis(results_dict, output_dir)
    
    # Generate comprehensive comparison plot
    print("\n" + "="*80)
    print("GENERATING ANALYSIS PLOTS")
    print("="*80)
    
    plot_register_performance_comparison(results_dict, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    if output_dir:
        print(f"\nAnalysis saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
