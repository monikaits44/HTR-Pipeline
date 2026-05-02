#!/usr/bin/env python3
"""
Plot Training Metrics from Experiment Results

Visualizes training curves from results.csv for any architecture.
Supports single model analysis or multi-model comparison.

Supported Architectures: ALL (cnn_rnn, vit_rgts, torchvision_vit, trocr)
    Reads results.csv and config.json from run directories.
Input: One or more run directories containing results.csv
Output: Training curve plots (saved or displayed)

Metrics Plotted:
- Training CTC Loss vs Epoch
- Validation/Test CER vs Epoch
- Validation/Test WER vs Epoch
- Learning Rate Schedule

Usage:
    # Single model (any architecture)
    python scripts/postprocessing/plot_training_metrics.py \
        --model-path saved_models/experiments/run_32/

    # Compare multiple models (e.g., all ViT-RGTS v2 runs)
    python scripts/postprocessing/plot_training_metrics.py \
        --model-path saved_models/experiments/run_50/ \
        --model-path saved_models/experiments/run_51/ \
        --model-path saved_models/experiments/run_52/ \
        --model-path saved_models/experiments/run_53/ \
        --model-path saved_models/experiments/run_54/

    # Compare across architectures
    python scripts/postprocessing/plot_training_metrics.py \
        --model-path saved_models/experiments/run_32/ \
        --model-path saved_models/experiments/run_39/ \
        --model-path saved_models/experiments/run_40/ \
        --model-path saved_models/experiments/run_54/

    # Plot specific metrics only
    python scripts/postprocessing/plot_training_metrics.py \
        --model-path saved_models/experiments/run_32/ \
        --metrics loss cer

    # Custom output directory
    python scripts/postprocessing/plot_training_metrics.py \
        --model-path saved_models/experiments/run_32/ \
        --output-dir ./output/training_plots/
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Tuple

# Professional color palette
COLOR_PALETTE = {
    'primary': "#84fa30",      # Deep blue
    'secondary': '#004359',     # Teal blue
    'accent1': '#386846',       # Forest green
    'accent2': "#63065f",       # Steel blue
    'model1': "#081d39",        # Deep blue
    'model2': '#004359',        # Teal
    'model3': "#B3D03F",        # Green
    'model4': '#34677d',        # Steel blue
    'model5': '#d62728',        # Red
    'model6': '#ff7f0e',        # Orange
    'model7': "#0a360a",        # Green
    'model8': '#9467bd',        # Purple
}

# Set style for professional plots
sns.set_style("white")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def load_results(model_path: str) -> Tuple[pd.DataFrame, str]:
    """Load results.csv from model path and extract descriptive model name from config."""
    results_path = os.path.join(model_path, 'results.csv')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_csv(results_path)
    
    # Load config.json to get architecture type
    config_path = os.path.join(model_path, 'config.json')
    run_name = Path(model_path).stem if Path(model_path).stem else Path(model_path).parent.stem
    if not run_name or run_name == 'experiments':
        run_name = os.path.basename(os.path.normpath(model_path))
    
    # Try to extract architecture information from config
    model_name = run_name  # Default fallback
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract architecture type
            arch_type = config.get('arch', {}).get('type', 'unknown')
            
            # Build descriptive name based on architecture
            if arch_type == 'cnn_rnn':
                model_name = "CNN-RNN"
            elif arch_type == 'vit_rgts':
                # Check for number of registers
                num_registers = config.get('arch', {}).get('num_registers', None)
                if num_registers is not None:
                    model_name = f"ViT-RGTS ({num_registers} reg)"
                else:
                    model_name = "ViT-RGTS"
            elif arch_type == 'torchvision_vit':
                # Check for model name (e.g., vit_b_16)
                model_variant = config.get('arch', {}).get('model_name', 'vit_b_16')
                num_registers = config.get('arch', {}).get('num_registers', 0)
                if num_registers > 0:
                    model_name = f"TorchVision-{model_variant.upper()} ({num_registers} reg)"
                else:
                    model_name = f"TorchVision-{model_variant.upper()}"
            elif arch_type == 'trocr':
                model_name = "TrOCR"
            else:
                model_name = f"{arch_type} ({run_name})"
            
            # Append run name if it provides additional context
            if run_name and run_name != 'experiments' and not run_name.startswith('run_'):
                model_name = f"{model_name} [{run_name}]"
            elif run_name.startswith('run_'):
                # Add run number as suffix for disambiguation
                model_name = f"{model_name} (Run {run_name.split('_')[-1]})"
                
        except Exception as e:
            print(f"Warning: Could not parse config.json for {model_path}: {e}")
            model_name = run_name
    else:
        print(f"Warning: config.json not found for {model_path}, using run name")
    
    return df, model_name


def plot_training_loss(results: Dict[str, pd.DataFrame], output_dir: str = None):
    """Plot Training CTC Loss vs Epoch."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = [COLOR_PALETTE[f'model{i+1}'] if f'model{i+1}' in COLOR_PALETTE 
              else plt.cm.tab10(i) for i in range(len(results))]
    
    for idx, (model_name, df) in enumerate(results.items()):
        if 'train/ctc_loss' not in df.columns:
            print(f"Warning: 'train/ctc_loss' not found in {model_name}")
            continue
        
        epochs = df['epoch']
        train_loss = df['train/ctc_loss']
        
        ax.plot(epochs, train_loss, marker='o', markersize=4, linewidth=2.5,
               label=model_name, color=colors[idx], alpha=0.85)
    
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
    ax.set_ylabel('CTC Loss', fontweight='bold', fontsize=13)
    ax.set_title('Training CTC Loss vs Epoch', fontweight='bold', fontsize=16, pad=15)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add best loss annotation for single model
    if len(results) == 1:
        model_name, df = list(results.items())[0]
        best_epoch = df['train/ctc_loss'].idxmin() + 1
        best_loss = df['train/ctc_loss'].min()
        ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.text(0.02, 0.98, f'Best: Epoch {best_epoch}, Loss: {best_loss:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, 'training_ctc_loss.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_validation_cer(results: Dict[str, pd.DataFrame], output_dir: str = None):
    """Plot Validation and Test CER vs Epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Character Error Rate (CER) vs Epoch', fontsize=18, fontweight='bold', y=0.98)
    
    colors = [COLOR_PALETTE[f'model{i+1}'] if f'model{i+1}' in COLOR_PALETTE 
              else plt.cm.tab10(i) for i in range(len(results))]
    
    # Plot 1: Validation CER
    for idx, (model_name, df) in enumerate(results.items()):
        if 'val/cer' not in df.columns:
            continue
        
        epochs = df['epoch']
        val_cer = df['val/cer'] * 100  # Convert to percentage
        
        axes[0].plot(epochs, val_cer, marker='o', markersize=4, linewidth=2.5,
                    label=model_name, color=colors[idx], alpha=0.85)
    
    axes[0].set_xlabel('Epoch', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Validation CER (%)', fontweight='bold', fontsize=13)
    axes[0].set_title('Validation CER', fontweight='bold', fontsize=14, pad=10)
    axes[0].legend(loc='upper right', frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot 2: Test CER
    for idx, (model_name, df) in enumerate(results.items()):
        if 'test/cer' not in df.columns:
            continue
        
        epochs = df['epoch']
        test_cer = df['test/cer'] * 100  # Convert to percentage
        
        axes[1].plot(epochs, test_cer, marker='s', markersize=4, linewidth=2.5,
                    label=model_name, color=colors[idx], alpha=0.85)
    
    axes[1].set_xlabel('Epoch', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Test CER (%)', fontweight='bold', fontsize=13)
    axes[1].set_title('Test CER', fontweight='bold', fontsize=14, pad=10)
    axes[1].legend(loc='upper right', frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Add best CER annotations for single model
    if len(results) == 1:
        model_name, df = list(results.items())[0]
        
        if 'val/cer' in df.columns:
            best_val_epoch = df['val/cer'].idxmin() + 1
            best_val_cer = df['val/cer'].min() * 100
            axes[0].axvline(best_val_epoch, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
            axes[0].text(0.02, 0.98, f'Best: Epoch {best_val_epoch}\nCER: {best_val_cer:.2f}%',
                       transform=axes[0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10, fontweight='bold')
        
        if 'test/cer' in df.columns:
            best_test_epoch = df['test/cer'].idxmin() + 1
            best_test_cer = df['test/cer'].min() * 100
            axes[1].axvline(best_test_epoch, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
            axes[1].text(0.02, 0.98, f'Best: Epoch {best_test_epoch}\nCER: {best_test_cer:.2f}%',
                       transform=axes[1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, 'cer_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_validation_wer(results: Dict[str, pd.DataFrame], output_dir: str = None):
    """Plot Validation and Test WER vs Epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Word Error Rate (WER) vs Epoch', fontsize=18, fontweight='bold', y=0.98)
    
    colors = [COLOR_PALETTE[f'model{i+1}'] if f'model{i+1}' in COLOR_PALETTE 
              else plt.cm.tab10(i) for i in range(len(results))]
    
    # Plot 1: Validation WER
    for idx, (model_name, df) in enumerate(results.items()):
        if 'val/wer' not in df.columns:
            continue
        
        epochs = df['epoch']
        val_wer = df['val/wer'] * 100  # Convert to percentage
        
        axes[0].plot(epochs, val_wer, marker='o', markersize=4, linewidth=2.5,
                    label=model_name, color=colors[idx], alpha=0.85)
    
    axes[0].set_xlabel('Epoch', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Validation WER (%)', fontweight='bold', fontsize=13)
    axes[0].set_title('Validation WER', fontweight='bold', fontsize=14, pad=10)
    axes[0].legend(loc='upper right', frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot 2: Test WER
    for idx, (model_name, df) in enumerate(results.items()):
        if 'test/wer' not in df.columns:
            continue
        
        epochs = df['epoch']
        test_wer = df['test/wer'] * 100  # Convert to percentage
        
        axes[1].plot(epochs, test_wer, marker='s', markersize=4, linewidth=2.5,
                    label=model_name, color=colors[idx], alpha=0.85)
    
    axes[1].set_xlabel('Epoch', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Test WER (%)', fontweight='bold', fontsize=13)
    axes[1].set_title('Test WER', fontweight='bold', fontsize=14, pad=10)
    axes[1].legend(loc='upper right', frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Add best WER annotations for single model
    if len(results) == 1:
        model_name, df = list(results.items())[0]
        
        if 'val/wer' in df.columns:
            best_val_epoch = df['val/wer'].idxmin() + 1
            best_val_wer = df['val/wer'].min() * 100
            axes[0].axvline(best_val_epoch, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
            axes[0].text(0.02, 0.98, f'Best: Epoch {best_val_epoch}\nWER: {best_val_wer:.2f}%',
                       transform=axes[0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10, fontweight='bold')
        
        if 'test/wer' in df.columns:
            best_test_epoch = df['test/wer'].idxmin() + 1
            best_test_wer = df['test/wer'].min() * 100
            axes[1].axvline(best_test_epoch, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
            axes[1].text(0.02, 0.98, f'Best: Epoch {best_test_epoch}\nWER: {best_test_wer:.2f}%',
                       transform=axes[1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, 'wer_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_learning_rate(results: Dict[str, pd.DataFrame], output_dir: str = None):
    """Plot Learning Rate Schedule vs Epoch."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = [COLOR_PALETTE[f'model{i+1}'] if f'model{i+1}' in COLOR_PALETTE 
              else plt.cm.tab10(i) for i in range(len(results))]
    
    for idx, (model_name, df) in enumerate(results.items()):
        if 'lr' not in df.columns:
            print(f"Warning: 'lr' not found in {model_name}")
            continue
        
        epochs = df['epoch']
        lr = df['lr']
        
        ax.plot(epochs, lr, marker='o', markersize=4, linewidth=2.5,
               label=model_name, color=colors[idx], alpha=0.85)
    
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
    ax.set_ylabel('Learning Rate', fontweight='bold', fontsize=13)
    ax.set_title('Learning Rate Schedule', fontweight='bold', fontsize=16, pad=15)
    ax.set_yscale('log')
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, 'learning_rate_schedule.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comprehensive_metrics(results: Dict[str, pd.DataFrame], output_dir: str = None):
    """Plot all metrics in a comprehensive 2x2 grid."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    model_name_single = list(results.keys())[0] if len(results) == 1 else "Model Comparison"
    fig.suptitle(f'Training Metrics Overview: {model_name_single}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    colors = [COLOR_PALETTE[f'model{i+1}'] if f'model{i+1}' in COLOR_PALETTE 
              else plt.cm.tab10(i) for i in range(len(results))]
    
    # Plot 1: Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, (model_name, df) in enumerate(results.items()):
        if 'train/ctc_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/ctc_loss'], marker='o', markersize=3,
                    linewidth=2, label=model_name, color=colors[idx], alpha=0.85)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('CTC Loss', fontweight='bold')
    ax1.set_title('Training CTC Loss', fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Validation CER
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, (model_name, df) in enumerate(results.items()):
        if 'val/cer' in df.columns:
            ax2.plot(df['epoch'], df['val/cer'] * 100, marker='o', markersize=3,
                    linewidth=2, label=model_name, color=colors[idx], alpha=0.85)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Validation CER (%)', fontweight='bold')
    ax2.set_title('Validation Character Error Rate', fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Test CER
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, (model_name, df) in enumerate(results.items()):
        if 'test/cer' in df.columns:
            ax3.plot(df['epoch'], df['test/cer'] * 100, marker='s', markersize=3,
                    linewidth=2, label=model_name, color=colors[idx], alpha=0.85)
    
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Test CER (%)', fontweight='bold')
    ax3.set_title('Test Character Error Rate', fontweight='bold', pad=10)
    ax3.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Learning Rate
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, (model_name, df) in enumerate(results.items()):
        if 'lr' in df.columns:
            ax4.plot(df['epoch'], df['lr'], marker='o', markersize=3,
                    linewidth=2, label=model_name, color=colors[idx], alpha=0.85)
    
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontweight='bold')
    ax4.set_title('Learning Rate Schedule', fontweight='bold', pad=10)
    ax4.set_yscale('log')
    ax4.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--', which='both')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, 'comprehensive_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_statistics(results: Dict[str, pd.DataFrame]):
    """Print summary statistics for all models."""
    print("\n" + "="*80)
    print("TRAINING METRICS SUMMARY")
    print("="*80)
    
    for model_name, df in results.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        print(f"Total Epochs: {len(df)}")
        
        if 'train/ctc_loss' in df.columns:
            best_loss_epoch = df['train/ctc_loss'].idxmin() + 1
            best_loss = df['train/ctc_loss'].min()
            final_loss = df['train/ctc_loss'].iloc[-1]
            print(f"\nTraining CTC Loss:")
            print(f"  Best:  {best_loss:.4f} (Epoch {best_loss_epoch})")
            print(f"  Final: {final_loss:.4f} (Epoch {len(df)})")
            print(f"  Reduction: {((df['train/ctc_loss'].iloc[0] - final_loss) / df['train/ctc_loss'].iloc[0] * 100):.2f}%")
        
        if 'val/cer' in df.columns:
            best_val_cer_epoch = df['val/cer'].idxmin() + 1
            best_val_cer = df['val/cer'].min() * 100
            final_val_cer = df['val/cer'].iloc[-1] * 100
            print(f"\nValidation CER:")
            print(f"  Best:  {best_val_cer:.2f}% (Epoch {best_val_cer_epoch})")
            print(f"  Final: {final_val_cer:.2f}% (Epoch {len(df)})")
        
        if 'test/cer' in df.columns:
            best_test_cer_epoch = df['test/cer'].idxmin() + 1
            best_test_cer = df['test/cer'].min() * 100
            final_test_cer = df['test/cer'].iloc[-1] * 100
            print(f"\nTest CER:")
            print(f"  Best:  {best_test_cer:.2f}% (Epoch {best_test_cer_epoch})")
            print(f"  Final: {final_test_cer:.2f}% (Epoch {len(df)})")
        
        if 'val/wer' in df.columns:
            best_val_wer_epoch = df['val/wer'].idxmin() + 1
            best_val_wer = df['val/wer'].min() * 100
            final_val_wer = df['val/wer'].iloc[-1] * 100
            print(f"\nValidation WER:")
            print(f"  Best:  {best_val_wer:.2f}% (Epoch {best_val_wer_epoch})")
            print(f"  Final: {final_val_wer:.2f}% (Epoch {len(df)})")
        
        if 'test/wer' in df.columns:
            best_test_wer_epoch = df['test/wer'].idxmin() + 1
            best_test_wer = df['test/wer'].min() * 100
            final_test_wer = df['test/wer'].iloc[-1] * 100
            print(f"\nTest WER:")
            print(f"  Best:  {best_test_wer:.2f}% (Epoch {best_test_wer_epoch})")
            print(f"  Final: {final_test_wer:.2f}% (Epoch {len(df)})")
        
        if 'lr' in df.columns:
            lr_schedule = df['lr'].unique()
            print(f"\nLearning Rate Schedule:")
            print(f"  Initial: {df['lr'].iloc[0]:.2e}")
            print(f"  Final:   {df['lr'].iloc[-1]:.2e}")
            print(f"  Unique values: {len(lr_schedule)}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Training Metrics from Experiment Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python scripts/postprocessing/plot_training_metrics.py --model-path saved_models/experiments/run_33/
  
  # Multiple models comparison
  python scripts/postprocessing/plot_training_metrics.py \\
      --model-path saved_models/experiments/run_33/ \\
      --model-path saved_models/experiments/run_34/ \\
      --model-path saved_models/experiments/run_35/
  
  # Plot specific metrics only
  python scripts/postprocessing/plot_training_metrics.py \\
      --model-path saved_models/experiments/run_33/ \\
      --metrics loss cer
        """
    )
    parser.add_argument(
        '--model-path',
        type=str,
        action='append',
        required=True,
        help='Path to model directory containing results.csv (can be specified multiple times)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/training_plots',
        help='Directory to save output plots (default: ./output/training_plots)'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['loss', 'cer', 'wer', 'lr', 'all'],
        default=['all'],
        help='Metrics to plot (default: all)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save plots (show only)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = None if args.no_save else args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Load results from all model paths
    results = {}
    print("\n" + "="*80)
    print("LOADING EXPERIMENT RESULTS")
    print("="*80)
    
    for model_path in args.model_path:
        try:
            df, model_name = load_results(model_path)
            results[model_name] = df
            print(f"✓ Loaded: {model_name} ({len(df)} epochs)")
        except Exception as e:
            print(f"✗ Error loading {model_path}: {e}")
    
    if not results:
        print("Error: No valid results found!")
        return 1
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Determine which metrics to plot
    plot_all = 'all' in args.metrics
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Plot metrics
    if plot_all or 'loss' in args.metrics:
        print("\nPlotting Training CTC Loss...")
        plot_training_loss(results, output_dir)
    
    if plot_all or 'cer' in args.metrics:
        print("Plotting CER metrics...")
        plot_validation_cer(results, output_dir)
    
    if plot_all or 'wer' in args.metrics:
        print("Plotting WER metrics...")
        plot_validation_wer(results, output_dir)
    
    if plot_all or 'lr' in args.metrics:
        print("Plotting Learning Rate Schedule...")
        plot_learning_rate(results, output_dir)
    
    # Always plot comprehensive view for single models
    if len(results) == 1 and plot_all:
        print("Plotting Comprehensive Metrics Overview...")
        plot_comprehensive_metrics(results, output_dir)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE!")
    print("="*80)
    
    if output_dir:
        print(f"\nAll plots saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
