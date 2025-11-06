# Evaluation Details CSV

## Overview

During training, the system now creates a detailed `evaluation_details.csv` file that logs every single prediction made during evaluation. This file is saved in each experiment directory (e.g., `saved_models/experiments/run_X/evaluation_details.csv`).

## CSV Structure

The evaluation CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `epoch` | Training epoch when evaluation was performed (0 = before training) |
| `dataset` | Which dataset was evaluated ('test' or 'val') |
| `sample_idx` | Index of the sample within the evaluation run |
| `ground_truth` | The actual text (ground truth) |
| `prediction` | The model's predicted text |
| `sample_cer` | Character Error Rate for this sample (0.0-1.0, lower is better) |
| `sample_wer` | Word Error Rate for this sample (0.0-1.0, lower is better) |
| `gt_length` | Length of ground truth text (number of characters) |
| `pred_length` | Length of predicted text (number of characters) |

## When Evaluations Happen

1. **Epoch 0**: Baseline evaluation before any training
2. **During training**: Every `save_every_k_epochs` epochs (configured in `config.yaml`)
3. **Both datasets**: Validation set first, then test set

## Example Data

```csv
epoch,dataset,sample_idx,ground_truth,prediction,sample_cer,sample_wer,gt_length,pred_length
0,test,0,"The quick brown fox","Th quik brwn fox",0.1579,0.2500,19,17
0,test,1,"jumps over lazy dog","jumps over lazy dog",0.0000,0.0000,19,19
5,val,0,"Hello World","Hello World",0.0000,0.0000,11,11
```

## Analysis Tools

### 1. Quick Analysis with Pandas

```python
import pandas as pd

# Load the data
df = pd.read_csv('saved_models/experiments/run_10/evaluation_details.csv')

# Get summary statistics
print(df.groupby(['epoch', 'dataset'])['sample_cer'].describe())

# Find worst predictions in latest epoch
latest = df[df['epoch'] == df['epoch'].max()]
worst = latest.nlargest(10, 'sample_cer')[['ground_truth', 'prediction', 'sample_cer']]
print(worst)

# Count perfect predictions
perfect = df[df['sample_cer'] == 0.0]
print(f"Perfect predictions: {len(perfect)}/{len(df)}")
```

### 2. Use the Analysis Script

A comprehensive analysis script is provided:

```bash
python scripts/postprocessing/analyze_evaluation.py saved_models/experiments/run_10/evaluation_details.csv
```

This script provides:
- Overall statistics per epoch and dataset
- Worst performing samples (highest CER/WER)
- Best performing samples (lowest CER/WER)
- Error distribution analysis
- Length-based analysis

### 3. Custom Analysis

Common use cases:

**Track improvement over epochs:**
```python
epoch_stats = df.groupby(['epoch', 'dataset']).agg({
    'sample_cer': 'mean',
    'sample_wer': 'mean'
})
```

**Find samples that got worse:**
```python
epoch0 = df[df['epoch'] == 0].set_index('sample_idx')
epoch5 = df[df['epoch'] == 5].set_index('sample_idx')
worse = epoch5[epoch5['sample_cer'] > epoch0['sample_cer']]
```

**Analyze by text length:**
```python
df['length_category'] = pd.cut(df['gt_length'], bins=[0, 20, 40, 60, 100], labels=['short', 'medium', 'long', 'very_long'])
length_analysis = df.groupby('length_category')['sample_cer'].mean()
```

## Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot CER distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df[df['epoch']==df['epoch'].max()], x='sample_cer', hue='dataset', bins=50)
plt.title('CER Distribution (Latest Epoch)')
plt.xlabel('Character Error Rate')
plt.show()

# Plot improvement over epochs
epoch_means = df.groupby(['epoch', 'dataset'])['sample_cer'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=epoch_means, x='epoch', y='sample_cer', hue='dataset', marker='o')
plt.title('Average CER Over Training')
plt.ylabel('Average Character Error Rate')
plt.show()

# Plot length vs error
plt.figure(figsize=(10, 6))
latest = df[df['epoch'] == df['epoch'].max()]
plt.scatter(latest['gt_length'], latest['sample_cer'], alpha=0.3)
plt.xlabel('Ground Truth Length (characters)')
plt.ylabel('Character Error Rate')
plt.title('Text Length vs Error Rate')
plt.show()
```

## Benefits

1. **Error Analysis**: Identify specific samples causing problems
2. **Pattern Detection**: Find common error patterns in predictions
3. **Length Analysis**: Understand how performance varies with text length
4. **Progress Tracking**: See which samples improve over training
5. **Dataset Comparison**: Compare test vs validation performance
6. **Model Debugging**: Debug specific failure cases

## File Location

The evaluation details CSV is saved alongside other experiment files:

```
saved_models/experiments/run_X/
├── config.yaml                 # Configuration used
├── log.txt                     # Training log
├── results.csv                 # High-level metrics per epoch
├── evaluation_details.csv      # ← Detailed per-sample results
├── metrics_description.txt     # Metrics documentation
└── model.pt                    # Trained model
```
