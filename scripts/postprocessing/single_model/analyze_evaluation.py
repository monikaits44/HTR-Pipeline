#!/usr/bin/env python3
"""
Analyze Per-Sample Evaluation Results

Parses evaluation_details.csv to provide detailed error analysis including
worst/best samples, error distributions, and length-based performance.

Supported Architectures: ALL (cnn_rnn, vit_rgts, torchvision_vit, trocr)
    Architecture-agnostic — reads CSV output from any model evaluation.
Input: evaluation_details.csv from any run directory
Output: Terminal analysis with statistics and error breakdown

Usage:
    # Analyze CNN-RNN evaluation (run_32)
    python scripts/postprocessing/analyze_evaluation.py \
        saved_models/experiments/run_32/evaluation_details.csv

    # Analyze ViT-RGTS evaluation (run_54)
    python scripts/postprocessing/analyze_evaluation.py \
        saved_models/experiments/run_54/evaluation_details.csv

    # Analyze TrOCR evaluation (run_40)
    python scripts/postprocessing/analyze_evaluation.py \
        saved_models/experiments/run_40/evaluation_details.csv

Analysis Provided:
    - Overall statistics per epoch and dataset
    - Worst performing samples (highest CER/WER)
    - Best performing samples (lowest CER/WER)
    - Error distribution analysis
    - Length-based analysis (performance vs text length)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_evaluation(csv_path):
    """Analyze the evaluation details CSV file."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {csv_path}")
    print(f"{'='*80}\n")
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Epochs evaluated: {sorted(df['epoch'].unique())}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print()
    
    # Overall statistics per epoch and dataset
    print("="*80)
    print("OVERALL STATISTICS BY EPOCH AND DATASET")
    print("="*80)
    summary = df.groupby(['epoch', 'dataset']).agg({
        'sample_cer': ['mean', 'std', 'min', 'max', 'median'],
        'sample_wer': ['mean', 'std', 'min', 'max', 'median'],
        'sample_idx': 'count'
    }).round(4)
    print(summary)
    print()
    
    # Analyze by epoch
    print("="*80)
    print("PROGRESSION OVER EPOCHS")
    print("="*80)
    epoch_summary = df.groupby(['epoch', 'dataset']).agg({
        'sample_cer': 'mean',
        'sample_wer': 'mean'
    }).round(4)
    print(epoch_summary)
    print()
    
    # Find worst performing samples (highest errors) in the latest epoch
    latest_epoch = df['epoch'].max()
    latest_data = df[df['epoch'] == latest_epoch]
    
    print("="*80)
    print(f"WORST PERFORMING SAMPLES (Epoch {latest_epoch})")
    print("="*80)
    
    for dataset in latest_data['dataset'].unique():
        dataset_data = latest_data[latest_data['dataset'] == dataset]
        worst_samples = dataset_data.nlargest(10, 'sample_cer')[
            ['sample_idx', 'ground_truth', 'prediction', 'sample_cer', 'sample_wer', 'gt_length']
        ]
        print(f"\n{dataset.upper()} Set - Top 10 Worst CER:")
        print(worst_samples.to_string(index=False))
        print()
    
    # Find best performing samples (lowest errors)
    print("="*80)
    print(f"BEST PERFORMING SAMPLES (Epoch {latest_epoch})")
    print("="*80)
    
    for dataset in latest_data['dataset'].unique():
        dataset_data = latest_data[latest_data['dataset'] == dataset]
        perfect_samples = dataset_data[dataset_data['sample_cer'] == 0.0]
        print(f"\n{dataset.upper()} Set - Perfect predictions (CER=0): {len(perfect_samples)}/{len(dataset_data)}")
        print(f"Percentage: {100*len(perfect_samples)/len(dataset_data):.2f}%")
    print()
    
    # Error distribution
    print("="*80)
    print(f"ERROR DISTRIBUTION (Epoch {latest_epoch})")
    print("="*80)
    
    for dataset in latest_data['dataset'].unique():
        dataset_data = latest_data[latest_data['dataset'] == dataset]
        print(f"\n{dataset.upper()} Set CER Distribution:")
        print(f"  0.00-0.10: {len(dataset_data[dataset_data['sample_cer'] <= 0.10])}")
        print(f"  0.10-0.20: {len(dataset_data[(dataset_data['sample_cer'] > 0.10) & (dataset_data['sample_cer'] <= 0.20)])}")
        print(f"  0.20-0.30: {len(dataset_data[(dataset_data['sample_cer'] > 0.20) & (dataset_data['sample_cer'] <= 0.30)])}")
        print(f"  0.30-0.50: {len(dataset_data[(dataset_data['sample_cer'] > 0.30) & (dataset_data['sample_cer'] <= 0.50)])}")
        print(f"  0.50-1.00: {len(dataset_data[dataset_data['sample_cer'] > 0.50])}")
    print()
    
    # Length-based analysis
    print("="*80)
    print(f"PERFORMANCE BY TEXT LENGTH (Epoch {latest_epoch})")
    print("="*80)
    
    for dataset in latest_data['dataset'].unique():
        dataset_data = latest_data[latest_data['dataset'] == dataset]
        
        # Create length bins
        dataset_data['length_bin'] = pd.cut(dataset_data['gt_length'], 
                                             bins=[0, 20, 40, 60, 80, 100, 200],
                                             labels=['0-20', '20-40', '40-60', '60-80', '80-100', '100+'])
        
        length_analysis = dataset_data.groupby('length_bin').agg({
            'sample_cer': ['mean', 'count'],
            'sample_wer': 'mean'
        }).round(4)
        
        print(f"\n{dataset.upper()} Set:")
        print(length_analysis)
    print()
    
    # Character-level statistics
    print("="*80)
    print(f"CHARACTER-LEVEL STATISTICS (Epoch {latest_epoch})")
    print("="*80)
    
    for dataset in latest_data['dataset'].unique():
        dataset_data = latest_data[latest_data['dataset'] == dataset]
        total_gt_chars = dataset_data['gt_length'].sum()
        total_pred_chars = dataset_data['pred_length'].sum()
        avg_cer = dataset_data['sample_cer'].mean()
        
        print(f"\n{dataset.upper()} Set:")
        print(f"  Total ground truth characters: {total_gt_chars}")
        print(f"  Total predicted characters: {total_pred_chars}")
        print(f"  Average sample length: {dataset_data['gt_length'].mean():.1f} chars")
        print(f"  Average CER: {avg_cer:.4f}")
        print(f"  Estimated total character errors: ~{int(avg_cer * total_gt_chars)}")
    print()
    
    print("="*80)
    print("Analysis complete!")
    print("="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_evaluation.py <path_to_evaluation_details.csv>")
        print("\nExample:")
        print("  python scripts/postprocessing/analyze_evaluation.py saved_models/experiments/run_10/evaluation_details.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    analyze_evaluation(csv_path)


if __name__ == '__main__':
    main()
