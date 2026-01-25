#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for IAM Handwriting Dataset

This script performs comprehensive analysis and visualization of the IAM dataset:
- Dataset statistics (split sizes, class distribution)
- Text length analysis
- Character frequency analysis
- Image dimension analysis
- Writer distribution

Usage:
    python scripts/preprocessing/exploratory_data_analysis.py --data-path ./data/IAM/processed_lines --output-dir ./output
    
    # Quick analysis without saving images
    python scripts/preprocessing/exploratory_data_analysis.py --data-path ./data/IAM/processed_lines --no-save-images
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# Professional color palette from template
COLOR_PALETTE = {
    'primary': '#002f6c',      # Deep blue
    'secondary': '#004359',     # Teal blue
    'accent1': '#386846',       # Forest green
    'accent2': '#34677d',       # Steel blue
    'train': '#002f6c',         # Deep blue for train
    'val': '#004359',           # Teal for validation
    'test': '#386846',          # Green for test
    'highlight': '#34677d',     # Steel blue for highlights
    'dark': '#1e2737',          # Dark background
    'grid': '#cccccc'           # Light gray for grids
}

# Set style for professional plots
sns.set_style("white")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16


def load_ground_truth(gt_path):
    """Load ground truth file and return list of (image_id, text) tuples."""
    samples = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                img_id, text = parts
                samples.append((img_id, text))
            else:
                # Handle edge case of empty text
                img_id = parts[0]
                samples.append((img_id, ''))
    return samples


def analyze_dataset_overview(data_path, output_dir=None):
    """Analyze basic dataset statistics with visualization."""
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    
    splits = ['train', 'val', 'test']
    stats = {}
    
    for split in splits:
        gt_path = os.path.join(data_path, split, 'gt.txt')
        if not os.path.exists(gt_path):
            print(f"Warning: {gt_path} not found")
            continue
        
        samples = load_ground_truth(gt_path)
        
        # Calculate character counts
        total_chars = sum(len(text) for _, text in samples)
        total_words = sum(len(text.split()) for _, text in samples)
        
        stats[split] = {
            'num_samples': len(samples),
            'samples': samples,
            'total_characters': total_chars,
            'total_words': total_words
        }
        
        # Check for corresponding images
        img_dir = os.path.join(data_path, split)
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        stats[split]['num_images'] = len(img_files)
    
    # Print detailed statistics
    total_samples = sum(s['num_samples'] for s in stats.values())
    total_chars = sum(s['total_characters'] for s in stats.values())
    total_words = sum(s['total_words'] for s in stats.values())
    
    print(f"\n{'='*60}")
    print(f"OVERALL DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total lines:       {total_samples:>10,}")
    print(f"Total characters:  {total_chars:>10,}")
    print(f"Total words:       {total_words:>10,}")
    print(f"{'='*60}")
    
    print(f"\n{'Split':<10} {'Lines':<10} {'Chars':<12} {'Words':<10} {'Line %':<10} {'Char %':<10}")
    print("-" * 70)
    
    for split in splits:
        if split in stats:
            num_samples = stats[split]['num_samples']
            num_chars = stats[split]['total_characters']
            num_words = stats[split]['total_words']
            sample_pct = (num_samples / total_samples) * 100
            char_pct = (num_chars / total_chars) * 100
            print(f"{split:<10} {num_samples:<10,} {num_chars:<12,} {num_words:<10,} {sample_pct:>6.2f}%    {char_pct:>6.2f}%")
    
    # Create visualization - 2 separate graphs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Dataset Split Distribution Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    split_names = ['Train', 'Val', 'Test']
    # Use professional color palette from template
    split_colors = [COLOR_PALETTE['train'], COLOR_PALETTE['val'], COLOR_PALETTE['test']]
    
    # Plot 1: Distribution by split (Train/Val/Test)
    line_counts = [stats[split]['num_samples'] for split in splits if split in stats]
    
    bars1 = axes[0].bar(split_names, line_counts, color=split_colors, 
                        edgecolor='black', linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Data Split', fontsize=13, fontweight='bold')
    axes[0].set_title('Sample Distribution Across Splits', fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(True, alpha=0.2, linestyle='--', axis='y')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Add value labels on bars with percentage
    for bar, count in zip(bars1, line_counts):
        height = bar.get_height()
        percentage = (count / total_samples) * 100
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add total count annotation
    axes[0].text(0.5, 0.95, f'Total: {total_samples:,} lines', 
                transform=axes[0].transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4),
                fontsize=11, fontweight='bold')
    
    # Plot 2: Distribution by category (Lines vs Characters)
    categories = ['Lines', 'Characters']
    train_data = [stats['train']['num_samples'], stats['train']['total_characters']]
    val_data = [stats['val']['num_samples'], stats['val']['total_characters']]
    test_data = [stats['test']['num_samples'], stats['test']['total_characters']]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars2_1 = axes[1].bar(x - width, train_data, width, label='Train', 
                         color=COLOR_PALETTE['train'], edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2_2 = axes[1].bar(x, val_data, width, label='Val', 
                         color=COLOR_PALETTE['val'], edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2_3 = axes[1].bar(x + width, test_data, width, label='Test', 
                         color=COLOR_PALETTE['test'], edgecolor='black', linewidth=1.2, alpha=0.85)
    
    axes[1].set_ylabel('Count', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Category', fontsize=13, fontweight='bold')
    axes[1].set_title('Lines vs Characters by Split', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=12)
    axes[1].legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    axes[1].grid(True, alpha=0.2, linestyle='--', axis='y')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bars in [bars2_1, bars2_2, bars2_3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'dataset_split_distribution.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved: dataset_split_distribution.png")
    else:
        plt.show()
    
    plt.close()
    
    return stats


def analyze_text_lengths(stats, output_dir=None):
    """Analyze text length distribution - simplified for critical analysis."""
    print("\n" + "="*80)
    print("TEXT LENGTH ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Text Length Distribution Analysis', fontsize=16, y=0.98, fontweight='bold')
    
    all_lengths = []
    split_lengths = {}
    
    for split, data in stats.items():
        lengths = [len(text) for _, text in data['samples']]
        split_lengths[split] = lengths
        all_lengths.extend(lengths)
        
        # Statistics
        print(f"\n{split.upper()} split:")
        print(f"  Samples: {len(lengths):,}")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        print(f"  Mean length: {np.mean(lengths):.2f}")
        print(f"  Median length: {np.median(lengths):.2f}")
        print(f"  Std dev: {np.std(lengths):.2f}")
    
    # Overall statistics
    print(f"\nOVERALL:")
    print(f"  Total samples: {len(all_lengths):,}")
    print(f"  Min length: {min(all_lengths)}")
    print(f"  Max length: {max(all_lengths)}")
    print(f"  Mean length: {np.mean(all_lengths):.2f}")
    print(f"  Median length: {np.median(all_lengths):.2f}")
    print(f"  Std dev: {np.std(all_lengths):.2f}")
    
    # Plot 1: Text length distribution by split (overlaid histograms)
    split_order = ['train', 'val', 'test']
    colors_split = [COLOR_PALETTE['train'], COLOR_PALETTE['val'], COLOR_PALETTE['test']]
    
    for split, color in zip(split_order, colors_split):
        if split in split_lengths:
            axes[0].hist(split_lengths[split], bins=60, alpha=0.6, 
                        label=f'{split.capitalize()} (n={len(split_lengths[split]):,})', 
                        color=color, edgecolor='black', linewidth=0.5)
    
    axes[0].set_xlabel('Text Length (characters)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Text Length Distribution by Split', fontweight='bold', pad=10)
    axes[0].legend(loc='upper right', frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.2, linestyle='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot 2: Overall text length distribution with statistical markers
    axes[1].hist(all_lengths, bins=80, color=COLOR_PALETTE['primary'], 
                alpha=0.7, edgecolor='black', linewidth=0.5, label='Distribution')
    
    mean_val = np.mean(all_lengths)
    median_val = np.median(all_lengths)
    std_val = np.std(all_lengths)
    
    axes[1].axvline(mean_val, color='#d62728', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_val:.1f}')
    axes[1].axvline(median_val, color='#ff7f0e', linestyle='--', linewidth=2.5, 
                   label=f'Median: {median_val:.1f}')
    axes[1].axvspan(mean_val - std_val, mean_val + std_val, 
                   alpha=0.15, color='gray', label=f'±1 Std Dev ({std_val:.1f})')
    
    axes[1].set_xlabel('Text Length (characters)', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Overall Text Length Distribution', fontweight='bold', pad=10)
    axes[1].legend(loc='upper right', frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.2, linestyle='--')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Add text annotation with key statistics
    stats_text = f'Total Samples: {len(all_lengths):,}\nRange: [{min(all_lengths)}, {max(all_lengths)}]'
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'text_length_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"\nSaved: text_length_analysis.png")
    else:
        plt.show()
    
    plt.close()
    
    return split_lengths


def analyze_character_frequency(stats, classes_path, output_dir=None):
    """Analyze character frequency distribution with critical insights."""
    print("\n" + "="*80)
    print("CHARACTER FREQUENCY ANALYSIS")
    print("="*80)
    
    # Load classes
    classes = np.load(classes_path, allow_pickle=True)
    print(f"\nTotal character classes: {len(classes)}")
    print(f"Character set preview: {' '.join(repr(c) for c in classes[:20])}...")
    
    # Count character frequencies
    char_counter = Counter()
    split_char_counters = {split: Counter() for split in stats.keys()}
    
    for split, data in stats.items():
        for _, text in data['samples']:
            for char in text:
                char_counter[char] += 1
                split_char_counters[split][char] += 1
    
    # Overall statistics
    total_chars = sum(char_counter.values())
    unique_chars = len(char_counter)
    
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Unique characters: {unique_chars}")
    
    # K for top and bottom characters
    K = 5
    
    print(f"\nTop {K} most frequent characters:")
    print(f"{'Char':<6} {'Count':<10} {'Percentage':<12}")
    print("-" * 30)
    
    for char, count in char_counter.most_common(K):
        percentage = (count / total_chars) * 100
        char_repr = repr(char) if char != ' ' else "'space'"
        print(f"{char_repr:<6} {count:<10,} {percentage:>6.2f}%")
    
    print(f"\nBottom {K} least frequent characters:")
    print(f"{'Char':<6} {'Count':<10} {'Percentage':<12}")
    print("-" * 30)
    
    for char, count in char_counter.most_common()[-K:]:
        percentage = (count / total_chars) * 100
        char_repr = repr(char) if char != ' ' else "'space'"
        print(f"{char_repr:<6} {count:<10,} {percentage:>6.2f}%")
    
    # Analyze character distribution imbalance
    top_10_percent = sum(count for _, count in char_counter.most_common(10)) / total_chars * 100
    print(f"\nClass Imbalance Analysis:")
    print(f"  Top 10 characters account for: {top_10_percent:.2f}% of all characters")
    print(f"  Imbalance ratio (most/least): {char_counter.most_common(1)[0][1] / char_counter.most_common()[-1][1]:.1f}x")
    
    # Create visualization - 2x2 grid for comprehensive analysis
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('Character Frequency Analysis: Distribution & Class Imbalance', 
                 fontsize=18, y=0.98, fontweight='bold')
    
    # Plot 1: Top K vs Bottom K frequency comparison
    ax1 = fig.add_subplot(gs[0, 0])
    top_k_chars = char_counter.most_common(K)
    bottom_k_chars = char_counter.most_common()[-K:]
    
    all_chars = top_k_chars + [('---', 0)] + bottom_k_chars[::-1]
    chars = [repr(c) if c not in [' ', '---'] else ('space' if c == ' ' else '...') for c, _ in all_chars]
    counts = [count for _, count in all_chars]
    colors = [COLOR_PALETTE['train']]*K + ['#e0e0e0'] + [COLOR_PALETTE['test']]*K
    
    bars = ax1.bar(range(len(chars)), counts, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:  # Skip separator
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xticks(range(len(chars)))
    ax1.set_xticklabels(chars, rotation=45, ha='right', fontsize=10)
    ax1.set_xlabel('Character', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Frequency (log scale)', fontweight='bold', fontsize=12)
    ax1.set_title(f'Top {K} vs Bottom {K}: Class Imbalance Visualization', 
                 fontweight='bold', pad=10, fontsize=13)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLOR_PALETTE['train'], edgecolor='black', label='Most Frequent'),
                      Patch(facecolor=COLOR_PALETTE['test'], edgecolor='black', label='Least Frequent')]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True)
    
    # Plot 2: Character type distribution (pie chart with professional styling)
    ax2 = fig.add_subplot(gs[0, 1])
    char_types = {
        'Lowercase': sum(count for char, count in char_counter.items() if char.islower()),
        'Uppercase': sum(count for char, count in char_counter.items() if char.isupper()),
        'Digits': sum(count for char, count in char_counter.items() if char.isdigit()),
        'Space': char_counter.get(' ', 0),
        'Punctuation': sum(count for char, count in char_counter.items() 
                          if not char.isalnum() and char != ' ')
    }
    
    colors_pie = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
                  COLOR_PALETTE['accent1'], COLOR_PALETTE['accent2'], '#34677d']
    
    wedges, texts, autotexts = ax2.pie(char_types.values(), labels=char_types.keys(), 
                                        autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'},
                                        explode=[0.05, 0.05, 0.05, 0.05, 0.05])
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax2.set_title('Character Type Distribution', fontweight='bold', pad=10, fontsize=13)
    
    # Plot 3: Split-wise comparison of top K characters
    ax3 = fig.add_subplot(gs[1, 0])
    top_k_chars_list = [c for c, _ in char_counter.most_common(K)]
    x = np.arange(len(top_k_chars_list))
    width = 0.25
    
    split_order = ['train', 'val', 'test']
    colors_split = [COLOR_PALETTE['train'], COLOR_PALETTE['val'], COLOR_PALETTE['test']]
    
    for i, (split, color) in enumerate(zip(split_order, colors_split)):
        if split in split_char_counters:
            counts = [split_char_counters[split].get(c, 0) for c in top_k_chars_list]
            offset = (i - 1) * width
            ax3.bar(x + offset, counts, width, label=split.capitalize(), 
                   color=color, edgecolor='black', linewidth=0.8)
    
    ax3.set_xlabel('Character', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax3.set_title(f'Top {K} Character Frequency: Split Consistency Analysis', 
                 fontweight='bold', pad=10, fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels([repr(c) if c != ' ' else 'sp' for c in top_k_chars_list], 
                        rotation=0, ha='center', fontsize=11)
    ax3.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax3.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Full character frequency distribution (showing class imbalance)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Sort all characters by frequency
    sorted_chars = char_counter.most_common()
    char_ranks = list(range(1, len(sorted_chars) + 1))
    char_freqs = [count for _, count in sorted_chars]
    
    ax4.plot(char_ranks, char_freqs, linewidth=2.5, color=COLOR_PALETTE['primary'], 
            marker='o', markersize=4, alpha=0.7, label='Character Frequency')
    
    # Highlight regions
    ax4.axhline(char_freqs[0], color='red', linestyle='--', alpha=0.5, 
               label=f'Max: {char_freqs[0]:,}')
    ax4.axhline(char_freqs[-1], color='orange', linestyle='--', alpha=0.5, 
               label=f'Min: {char_freqs[-1]:,}')
    
    ax4.set_xlabel('Character Rank (sorted by frequency)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Frequency (log scale)', fontweight='bold', fontsize=12)
    ax4.set_title('Class Imbalance: Full Character Frequency Distribution', 
                 fontweight='bold', pad=10, fontsize=13)
    ax4.set_yscale('log')
    ax4.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    ax4.grid(True, alpha=0.2, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add annotation about imbalance
    imbalance_ratio = char_freqs[0] / char_freqs[-1]
    ax4.text(0.5, 0.05, f'Imbalance Ratio: {imbalance_ratio:.0f}:1\n(Most frequent / Least frequent)', 
            transform=ax4.transAxes, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'character_frequency_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved: character_frequency_analysis.png")
    else:
        plt.show()
    
    plt.close()
    
    return char_counter


def analyze_image_dimensions(data_path, stats, output_dir=None, max_samples=1000):
    """Analyze image dimensions and aspect ratios."""
    print("\n" + "="*80)
    print("IMAGE DIMENSION ANALYSIS")
    print("="*80)
    
    print(f"\nAnalyzing up to {max_samples} images per split...")
    
    dimensions = []
    split_dimensions = {split: [] for split in stats.keys()}
    
    for split, data in stats.items():
        img_dir = os.path.join(data_path, split)
        samples = data['samples'][:max_samples]
        
        print(f"\nProcessing {split} split ({len(samples)} samples)...")
        for img_id, _ in tqdm(samples, desc=f"Loading {split} images"):
            img_path = os.path.join(img_dir, f"{img_id}.png")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    dimensions.append((w, h))
                    split_dimensions[split].append((w, h))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    if not dimensions:
        print("No images found!")
        return
    
    # Calculate statistics
    widths = [w for w, h in dimensions]
    heights = [h for w, h in dimensions]
    aspect_ratios = [w / h for w, h in dimensions]
    areas = [w * h for w, h in dimensions]
    
    print(f"\nImage Statistics (from {len(dimensions)} images):")
    print(f"\nWidth:")
    print(f"  Min: {min(widths)} px")
    print(f"  Max: {max(widths)} px")
    print(f"  Mean: {np.mean(widths):.2f} px")
    print(f"  Median: {np.median(widths):.2f} px")
    print(f"  Std dev: {np.std(widths):.2f} px")
    
    print(f"\nHeight:")
    print(f"  Min: {min(heights)} px")
    print(f"  Max: {max(heights)} px")
    print(f"  Mean: {np.mean(heights):.2f} px")
    print(f"  Median: {np.median(heights):.2f} px")
    print(f"  Std dev: {np.std(heights):.2f} px")
    
    print(f"\nAspect Ratio (W/H):")
    print(f"  Min: {min(aspect_ratios):.2f}")
    print(f"  Max: {max(aspect_ratios):.2f}")
    print(f"  Mean: {np.mean(aspect_ratios):.2f}")
    print(f"  Median: {np.median(aspect_ratios):.2f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Image Dimension Analysis', fontsize=16, y=0.995)
    
    # Plot 1: Width distribution
    axes[0, 0].hist(widths, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(widths):.0f}')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Image Width Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Height distribution
    axes[0, 1].hist(heights, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(heights):.0f}')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Image Height Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Aspect ratio distribution
    axes[1, 0].hist(aspect_ratios, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(aspect_ratios), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(aspect_ratios):.2f}')
    axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: 2D histogram (heatmap) - Width vs Height
    axes[1, 1].hist2d(widths, heights, bins=50, cmap='YlOrRd')
    axes[1, 1].set_xlabel('Width (pixels)')
    axes[1, 1].set_ylabel('Height (pixels)')
    axes[1, 1].set_title('Width vs Height Density')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'image_dimension_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved: image_dimension_analysis.png")
    else:
        plt.show()
    
    plt.close()
    
    return dimensions


def analyze_writer_distribution(stats, output_dir=None):
    """Analyze writer distribution across samples with visualization."""
    print("\n" + "="*80)
    print("WRITER DISTRIBUTION ANALYSIS")
    print("="*80)
    
    writer_counter = Counter()
    split_writer_counters = {split: Counter() for split in stats.keys()}
    split_writers_set = {split: set() for split in stats.keys()}
    
    for split, data in stats.items():
        for img_id, _ in data['samples']:
            # Extract writer ID from image ID (e.g., 'a01-000u-00' -> 'a01')
            writer_id = img_id.split('-')[0]
            writer_counter[writer_id] += 1
            split_writer_counters[split][writer_id] += 1
            split_writers_set[split].add(writer_id)
    
    unique_writers = len(writer_counter)
    total_samples = sum(writer_counter.values())
    
    print(f"\nTotal unique writers: {unique_writers}")
    print(f"Total samples: {total_samples:,}")
    print(f"Average samples per writer: {total_samples / unique_writers:.2f}")
    print(f"Min samples per writer: {min(writer_counter.values())}")
    print(f"Max samples per writer: {max(writer_counter.values())}")
    print(f"Median samples per writer: {np.median(list(writer_counter.values())):.2f}")
    
    # Note about full IAM dataset
    print(f"\nNote: The full IAM database contains 657 writers.")
    print(f"      Your processed_lines subset contains {unique_writers} writers.")
    
    print(f"\nTop 10 writers by sample count:")
    print(f"{'Writer ID':<12} {'Count':<10} {'Percentage':<12}")
    print("-" * 35)
    
    for writer, count in writer_counter.most_common(10):
        percentage = (count / total_samples) * 100
        print(f"{writer:<12} {count:<10} {percentage:>6.2f}%")
    
    # Writer distribution across splits
    print(f"\nWriter distribution across splits:")
    for split in ['train', 'val', 'test']:
        if split in split_writer_counters:
            unique_in_split = len(split_writer_counters[split])
            samples_in_split = sum(split_writer_counters[split].values())
            print(f"  {split}: {unique_in_split} unique writers, {samples_in_split:,} samples")
    
    # Check for writer overlap between splits
    print(f"\nWriter overlap between splits:")
    train_writers = split_writers_set.get('train', set())
    val_writers = split_writers_set.get('val', set())
    test_writers = split_writers_set.get('test', set())
    
    train_val_overlap = len(train_writers & val_writers)
    train_test_overlap = len(train_writers & test_writers)
    val_test_overlap = len(val_writers & test_writers)
    
    print(f"  Train ∩ Val: {train_val_overlap} writers")
    print(f"  Train ∩ Test: {train_test_overlap} writers")
    print(f"  Val ∩ Test: {val_test_overlap} writers")
    
    if train_val_overlap > 0 or train_test_overlap > 0:
        print(f"  ⚠ Warning: Writer overlap detected between splits!")
        print(f"             This may lead to overfitting (model sees same writer's style in train and test)")
    else:
        print(f"  ✓ Good: No writer overlap between splits (writer-independent split)")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Writer Distribution Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: Samples per writer distribution
    ax1 = fig.add_subplot(gs[0, 0])
    samples_per_writer = list(writer_counter.values())
    ax1.hist(samples_per_writer, bins=30, color=COLOR_PALETTE['primary'], 
            alpha=0.7, edgecolor='black', linewidth=1.2)
    ax1.axvline(np.mean(samples_per_writer), color='red', linestyle='--', 
               linewidth=2.5, label=f'Mean: {np.mean(samples_per_writer):.1f}')
    ax1.axvline(np.median(samples_per_writer), color='orange', linestyle='--', 
               linewidth=2.5, label=f'Median: {np.median(samples_per_writer):.1f}')
    ax1.set_xlabel('Samples per Writer', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Number of Writers', fontweight='bold', fontsize=12)
    ax1.set_title('Distribution of Samples per Writer', fontweight='bold', pad=10, fontsize=13)
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add statistics text
    stats_text = f'Total Writers: {unique_writers}\nTotal Samples: {total_samples:,}\nAvg: {total_samples/unique_writers:.1f}'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4),
            fontsize=10)
    
    # Plot 2: Top 15 writers
    ax2 = fig.add_subplot(gs[0, 1])
    top_15_writers = writer_counter.most_common(15)
    writers = [w for w, _ in top_15_writers]
    counts = [c for _, c in top_15_writers]
    
    bars = ax2.barh(range(len(writers)), counts, color=COLOR_PALETTE['primary'], 
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    ax2.set_yticks(range(len(writers)))
    ax2.set_yticklabels(writers, fontsize=10)
    ax2.set_xlabel('Number of Samples', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Writer ID', fontweight='bold', fontsize=12)
    ax2.set_title('Top 15 Writers by Sample Count', fontweight='bold', pad=10, fontsize=13)
    ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, f' {count}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Plot 3: Writer distribution across splits
    ax3 = fig.add_subplot(gs[0, 2])
    split_names = ['Train', 'Val', 'Test']
    split_order = ['train', 'val', 'test']
    colors_split = [COLOR_PALETTE['train'], COLOR_PALETTE['val'], COLOR_PALETTE['test']]
    
    writer_counts = [len(split_writers_set[split]) for split in split_order if split in split_writers_set]
    
    bars = ax3.bar(split_names, writer_counts, color=colors_split, 
                  edgecolor='black', linewidth=1.2, alpha=0.85)
    ax3.set_ylabel('Number of Unique Writers', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Data Split', fontweight='bold', fontsize=12)
    ax3.set_title('Unique Writers per Split', fontweight='bold', pad=10, fontsize=13)
    ax3.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, count in zip(bars, writer_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 4: Writer overlap Venn diagram (simplified bar chart)
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Calculate exclusive and shared writers
    train_only = len(train_writers - val_writers - test_writers)
    val_only = len(val_writers - train_writers - test_writers)
    test_only = len(test_writers - train_writers - val_writers)
    train_val_only = len((train_writers & val_writers) - test_writers)
    train_test_only = len((train_writers & test_writers) - val_writers)
    val_test_only = len((val_writers & test_writers) - train_writers)
    all_three = len(train_writers & val_writers & test_writers)
    
    categories = ['Train\nOnly', 'Val\nOnly', 'Test\nOnly', 'Train∩Val', 'Train∩Test', 'Val∩Test', 'All 3']
    values = [train_only, val_only, test_only, train_val_only, train_test_only, val_test_only, all_three]
    colors_overlap = [COLOR_PALETTE['train'], COLOR_PALETTE['val'], COLOR_PALETTE['test'], 
                     '#8B4513', '#4B0082', '#2F4F4F', '#8B0000']
    
    bars = ax4.bar(categories, values, color=colors_overlap, edgecolor='black', 
                  linewidth=1.2, alpha=0.85)
    ax4.set_ylabel('Number of Writers', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Writer Overlap Category', fontweight='bold', fontsize=12)
    ax4.set_title('Writer Overlap Between Splits', fontweight='bold', pad=10, fontsize=13)
    ax4.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars, values):
        if val > 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 5: Cumulative distribution of samples
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_counts = sorted(writer_counter.values(), reverse=True)
    cumulative_samples = np.cumsum(sorted_counts)
    cumulative_percent = (cumulative_samples / total_samples) * 100
    
    ax5.plot(range(1, len(sorted_counts) + 1), cumulative_percent, 
            linewidth=2.5, color=COLOR_PALETTE['primary'], marker='o', 
            markersize=3, alpha=0.7)
    ax5.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% of samples')
    ax5.axhline(80, color='orange', linestyle='--', alpha=0.5, label='80% of samples')
    ax5.axhline(90, color='green', linestyle='--', alpha=0.5, label='90% of samples')
    
    # Find how many writers account for 50%, 80%, 90%
    writers_for_50 = np.argmax(cumulative_percent >= 50) + 1
    writers_for_80 = np.argmax(cumulative_percent >= 80) + 1
    writers_for_90 = np.argmax(cumulative_percent >= 90) + 1
    
    ax5.set_xlabel('Number of Writers (ranked by sample count)', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Cumulative % of Samples', fontweight='bold', fontsize=12)
    ax5.set_title('Cumulative Sample Distribution', fontweight='bold', pad=10, fontsize=13)
    ax5.legend(loc='lower right', frameon=True, shadow=True)
    ax5.grid(True, alpha=0.2, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # Add annotation
    annotation_text = f'Top {writers_for_50} writers → 50% samples\nTop {writers_for_80} writers → 80% samples\nTop {writers_for_90} writers → 90% samples'
    ax5.text(0.02, 0.98, annotation_text, transform=ax5.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4),
            fontsize=9)
    
    # Plot 6: Sample imbalance across writers
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_counts_full = sorted(writer_counter.values(), reverse=True)
    writer_ranks = list(range(1, len(sorted_counts_full) + 1))
    
    ax6.plot(writer_ranks, sorted_counts_full, linewidth=2.5, 
            color=COLOR_PALETTE['primary'], marker='o', markersize=4, alpha=0.7)
    ax6.axhline(np.mean(sorted_counts_full), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(sorted_counts_full):.1f}')
    ax6.set_xlabel('Writer Rank (by sample count)', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax6.set_title('Writer Sample Imbalance', fontweight='bold', pad=10, fontsize=13)
    ax6.legend(loc='upper right', frameon=True, shadow=True)
    ax6.grid(True, alpha=0.2, linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    # Add imbalance ratio
    imbalance_ratio = max(sorted_counts_full) / min(sorted_counts_full)
    ax6.text(0.5, 0.05, f'Imbalance Ratio: {imbalance_ratio:.1f}:1\n(Max/Min samples per writer)', 
            transform=ax6.transAxes, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'writer_distribution_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved: writer_distribution_analysis.png")
    else:
        plt.show()
    
    plt.close()
    
    return writer_counter


def save_summary_report(stats, char_counter, writer_counter, output_dir):
    """Save a JSON summary report."""
    summary = {
        'dataset_overview': {
            'total_samples': sum(s['num_samples'] for s in stats.values()),
            'splits': {
                split: {
                    'num_samples': data['num_samples'],
                    'num_images': data['num_images'],
                    'total_characters': data['total_characters'],
                    'total_words': data['total_words']
                }
                for split, data in stats.items()
            }
        },
        'text_statistics': {
            'total_characters': sum(char_counter.values()),
            'unique_characters': len(char_counter)
        },
        'writer_statistics': {
            'unique_writers': len(writer_counter),
            'total_samples': sum(writer_counter.values())
        },
        'top_characters': [
            {'char': char, 'count': count} 
            for char, count in char_counter.most_common(20)
        ]
    }
    
    report_path = os.path.join(output_dir, 'eda_summary.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved summary report: eda_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Exploratory Data Analysis for IAM Handwriting Dataset'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/IAM/processed_lines',
        help='Path to processed IAM dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/data_analysis',
        help='Directory to save output visualizations and reports'
    )
    parser.add_argument(
        '--no-save-images',
        action='store_true',
        help='Do not save visualization images (show only)'
    )
    parser.add_argument(
        '--max-image-samples',
        type=int,
        default=1000,
        help='Maximum number of images to analyze per split'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = None if args.no_save_images else args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path not found: {args.data_path}")
        return
    
    classes_path = os.path.join(args.data_path, 'classes.npy')
    if not os.path.exists(classes_path):
        print(f"Error: Classes file not found: {classes_path}")
        return
    
    print("\n" + "="*80)
    print("IAM DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    print(f"Data path: {args.data_path}")
    
    # Run analyses
    stats = analyze_dataset_overview(args.data_path, output_dir)
    analyze_text_lengths(stats, output_dir)
    char_counter = analyze_character_frequency(stats, classes_path, output_dir)
    analyze_image_dimensions(args.data_path, stats, output_dir, args.max_image_samples)
    writer_counter = analyze_writer_distribution(stats, output_dir)
    
    # Save summary report
    if output_dir:
        save_summary_report(stats, char_counter, writer_counter, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    if output_dir:
        print(f"\nAll outputs saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - dataset_split_distribution.png")
        print("  - text_length_analysis.png")
        print("  - character_frequency_analysis.png")
        print("  - image_dimension_analysis.png")
        print("  - eda_summary.json")


if __name__ == '__main__':
    main()
