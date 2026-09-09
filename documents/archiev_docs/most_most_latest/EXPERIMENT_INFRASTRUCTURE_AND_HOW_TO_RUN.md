# Experiment Infrastructure & How to Run Everything

> Complete guide to running experiments, understanding the SLURM infrastructure, and analyzing results.

---

## 1. Quick Start

### Train CNN-RNN Baseline (Best Performing)

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate

# Direct run (interactive)
python scripts/trainer.py configs/baseline.yaml configs/config.yaml

# SLURM submission
cd experiments_execution
./submit_single.sh 1   # Submits baseline experiment
```

### Train ViT-RGTS v2 (With Explainability)

```bash
# Default (4 registers)
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml configs/config.yaml

# Custom register count
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml configs/config.yaml arch.num_registers=8
```

---

## 2. Configuration System

### Config Merging Order

```bash
python scripts/trainer.py <base_config>.yaml [additional.yaml ...] [key=value ...]
```

1. Load base YAML (e.g., `baseline_vit_rgts_v2.yaml`)
2. Merge additional YAMLs (e.g., `config.yaml` for device/data paths)
3. Apply CLI dotlist overrides (e.g., `arch.num_registers=8`)

Later values override earlier ones.

### `configs/config.yaml` — Shared Settings

This file typically contains:
- `device: cuda:0`
- `data.path: ./data/IAM/processed_lines`
- `train`/`eval` worker counts
- Any site-specific overrides

### Common CLI Overrides

```bash
# Architecture parameters
arch.num_registers=0          # Number of register tokens
arch.use_cnn_stem=true        # Use CNN stem (always true for v2)
arch.dim=256                  # Embedding dimension
arch.depth=6                  # Transformer layers

# Training parameters
train.lr=0.0005               # Learning rate
train.num_epochs=50           # Number of epochs
train.batch_size=16           # Batch size
train.augmentation=vit        # Force ViT augmentation

# Data mode
data.mode=synthetic           # Use synthetic data
data.synthetic_path=/path/to/synthetic_processed_lines

# Resume from checkpoint
resume=saved_models/experiments/run_67/model.pt
```

---

## 3. SLURM Experiment Infrastructure

### Directory Layout

```
experiments_execution/
├── slurm_scripts/
│   ├── 01_baseline.slurm           # CNN-RNN baseline
│   ├── 02_vit_rgts_0reg.slurm      # ViT-RGTS v2, 0 registers
│   ├── 03_vit_rgts_2reg.slurm      # ViT-RGTS v2, 2 registers
│   ├── 04_vit_rgts_4reg.slurm      # ViT-RGTS v2, 4 registers ← default
│   ├── 05_vit_rgts_8reg.slurm      # ViT-RGTS v2, 8 registers
│   ├── 06_vit_rgts_16reg.slurm     # ViT-RGTS v2, 16 registers
│   ├── 07_torchvision_vit_b16.slurm # Pretrained ViT-B/16
│   └── 08_trocr_base.slurm         # TrOCR encoder
│
├── submit_all_experiments.sh        # Submit all 8 experiments
├── submit_single.sh                 # Submit one experiment by number
├── submit_vit_rgts_v2_sweep.sh      # Register count sweep
├── submit_vit_rgts_v2_full_reg_sweep.sh  # Full 0-16 register sweep
│
├── logs/                            # SLURM stdout/stderr
│   ├── baseline/
│   ├── vit_rgts_registers/
│   └── pretrained/
│
├── OVERVIEW.md
├── README.md
├── QUICK_REFERENCE.txt
└── experiment_tracker.ipynb         # Notebook for tracking results
```

### Submitting Experiments

```bash
cd experiments_execution

# Submit a single experiment (1-8)
./submit_single.sh 1    # CNN-RNN baseline
./submit_single.sh 4    # ViT-RGTS v2, 4 registers

# Submit all experiments
./submit_all_experiments.sh
# Prompts for mode: 1=sequential, 2=parallel, 3=manual

# Register sweep (0, 2, 4, 8, 16)
./submit_vit_rgts_v2_sweep.sh

# Full register sweep (0-16, all values)
./submit_vit_rgts_v2_full_reg_sweep.sh
```

### Expected Training Times

| Experiment | Time (single GPU) |
|-----------|-------------------|
| CNN-RNN baseline | 3-4 hours |
| ViT-RGTS v2 (any register count) | 3.5-4.5 hours |
| Pretrained ViT-B/16 | 4-8 hours |
| TrOCR | 4-8 hours |
| **Full register sweep (17 runs)** | **60-75 hours sequential** |

---

## 4. Output Structure

Every training run creates a directory:

```
saved_models/experiments/run_N/
├── config.json              # Full configuration snapshot
├── model.pt                 # Model state dict (latest epoch)
├── results.csv              # Per-epoch metrics
├── evaluation_details.csv   # Per-sample predictions
├── training.log             # Console output
└── attention_weights/       # (ViT only)
    ├── epoch_001/
    │   ├── sample_000_layer_00.npy   # [H, S, S] attention
    │   ├── sample_000_layer_01.npy
    │   ├── ...
    │   ├── sample_000_token_norms.npy
    │   ├── sample_000_register_tokens.npy
    │   ├── sample_000_groundtruth.txt
    │   └── metadata.json
    ├── epoch_005/
    ├── epoch_010/
    └── ...
```

### results.csv Format

```csv
epoch,lr,train/ctc_loss,val/cer,val/wer,test/cer,test/wer,data/train_lines,...,model/params,time/epoch(s),device,seed
1,0.001,15.234,0.45,0.78,0.47,0.80,6482,...,6000000,120.5,cuda:0,-1
2,0.001,8.123,0.25,0.55,0.28,0.58,...
```

### evaluation_details.csv Format

```csv
epoch,dataset,sample_idx,ground_truth,prediction,sample_cer,sample_wer,gt_length,pred_length
80,test,0," The quick brown fox "," The quck brown fox ",0.05,0.25,22,21
```

---

## 5. Analyzing Results

### Quick Result Summary

```bash
# View final results for a run
tail -1 saved_models/experiments/run_67/results.csv

# Compare all runs
for run in saved_models/experiments/run_*/; do
  echo "$(basename $run): $(tail -1 $run/results.csv | awk -F, '{print "val_cer=" $4, "test_cer=" $6}')"
done | sort -t= -k2 -n
```

### Load Results in Python

```python
import pandas as pd
import json

# Load training curves
df = pd.read_csv('saved_models/experiments/run_67/results.csv')
print(df[['epoch', 'val/cer', 'test/cer']].tail(10))

# Load config
with open('saved_models/experiments/run_67/config.json') as f:
    config = json.load(f)
print(f"Architecture: {config['arch']['type']}")
print(f"Registers: {config['arch'].get('num_registers', 'N/A')}")

# Load per-sample evaluation
eval_df = pd.read_csv('saved_models/experiments/run_67/evaluation_details.csv')
worst_samples = eval_df[eval_df['dataset']=='test'].nlargest(10, 'sample_cer')
```

### Experiment Tracker Notebook

`experiments_execution/experiment_tracker.ipynb` provides interactive tracking:
- Training curve plots
- CER/WER comparison across runs
- Register sweep visualization

---

## 6. Attention Analysis After Training

### Load and Analyze Saved Attention

```python
import numpy as np
import json

# Load attention weights from a specific epoch
epoch_dir = 'saved_models/experiments/run_67/attention_weights/epoch_080'

# Load attention map for sample 0, layer 3
attn = np.load(f'{epoch_dir}/sample_000_layer_03.npy')  # [H, S, S] = [8, 132, 132]

# Load metadata
with open(f'{epoch_dir}/metadata.json') as f:
    meta = json.load(f)
print(f"Layers: {meta['num_layers']}, Heads: {meta['num_heads']}")

# Load token norms
norms = np.load(f'{epoch_dir}/sample_000_token_norms.npy')  # [S]

# Load register embeddings
reg = np.load(f'{epoch_dir}/sample_000_register_tokens.npy')  # [R, D]
```

### Run Attention Visualization Script

```bash
python scripts/attention_visualization.py \
    --config configs/baseline_vit_rgts_v2.yaml \
    --checkpoint saved_models/experiments/run_67/model.pt \
    --output outputs/attention_maps/
```

### Generate Publication Figures

```bash
# Paper Fig 5 (character-level attention)
python scripts/pub_fig1_paper_fig5.py

# Register comparison
python scripts/pub_fig2_register_comparison.py

# GradCAM analysis
python scripts/pub_fig3_gradcam_quantitative.py
```

---

## 7. Model Evaluation

### Evaluate a Saved Model

```bash
cd evaluation_execution
bash model_evaluation.sh
```

### Manual Evaluation

```python
import torch
from omegaconf import OmegaConf
from models import HTRNet

# Load config
config = OmegaConf.load('saved_models/experiments/run_67/config.json')

# Build model
net = HTRNet(config['arch'], nclasses=80)
net.load_state_dict(torch.load('saved_models/experiments/run_67/model.pt'))
net.eval()

# Run inference
img = torch.randn(1, 1, 128, 1024)  # Replace with real image
with torch.no_grad():
    logits = net(img)  # [T, 1, 80]
    prediction = logits.argmax(2).squeeze()  # [T]
```

---

## 8. Synthetic Data Workflow

### Step 1: Generate Synthetic Data

```bash
cd synthetic_data_generation
# Follow numbered scripts 0-6 in order
# See SYNTHETIC_DATA_PIPELINE_DOCUMENTATION.md for details
```

### Step 2: Extract to IAM Format

```bash
python scripts/extract_synthetic_to_iam_format.py \
    --lmdb_path /path/to/synthetic_lmdb \
    --output_path /path/to/synthetic_processed_lines
```

### Step 3: Train with Synthetic Data

```bash
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml \
    data.mode=synthetic \
    data.synthetic_path=/path/to/synthetic_processed_lines \
    train.num_epochs=10
```

### Step 4: Fine-tune on IAM

```bash
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_N/model.pt \
    data.mode=iam \
    train.num_epochs=50 \
    train.lr=0.0001
```

---

## 9. Existing Experiment Runs

### Runs 60-71: ViT-RGTS v2 Register Sweep (80 epochs)

| Run | Registers | Val CER | Test CER |
|-----|-----------|---------|----------|
| 60 | 13 | 4.29% | 6.16% |
| 61 | 10 | 4.22% | 6.03% |
| 62 | 14 | 4.19% | 6.03% |
| 63 | 8 | 4.29% | 6.26% |
| 64 | 1 | 4.23% | 6.04% |
| 65 | 5 | 4.17% | 6.04% |
| 66 | 3 | 4.25% | 6.27% |
| **67** | **6** | **4.08%** | **6.05%** |
| **68** | **7** | **4.11%** | **5.99%** |
| 69 | 9 | 4.17% | 6.19% |
| 70 | 15 | 4.09% | 6.05% |
| 71 | 16 | 4.19% | 6.10% |

### Run 73: CNN-RNN Baseline (80 epochs)

| Run | Val CER | Test CER |
|-----|---------|----------|
| **73** | **3.50%** | **5.02%** |

### Runs 76-92: Second Register Sweep (50 epochs)

All ViT-RGTS v2. Val CER range: 4.24-4.54%. Test CER range: 6.30-6.47%.

### Runs 96-101: Pretrained Models

| Run | Model | Val CER | Test CER | Notes |
|-----|-------|---------|----------|-------|
| 96 | TrOCR | 66.4% | 66.5% | Insufficient epochs |
| 97 | ViT-B/16 | 73.4% | 73.7% | Training collapsed |
| **98** | **TrOCR** | **13.2%** | **15.5%** | Best pretrained |
| 99 | ViT-B/16 | 85.7% | 86.3% | Only 10 epochs |
| 100 | ViT-B/16 | 26.1% | 28.6% | Moderate |
| 101 | ViT-B/16 | 77.8% | 77.8% | Collapsed |

---

## 10. Troubleshooting

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| CER stuck at 75%+ | ViT v1 patch embedding (wrong token order) | Use `use_cnn_stem: true` (v2) |
| CER not decreasing | Weight decay too high for small data | Reduce WD to 5e-4 / 5e-3 |
| CUDA OOM | Batch too large for GPU | Reduce `train.batch_size`, increase `gradient_accumulation` |
| Pretrained model garbage output | Missing ImageNet normalization | Ensure `self.pretrained = True` triggers normalization |
| TrOCR CER ~66% | Aspect ratio squash | Already fixed with aspect-ratio resize + pad |
| Run number collision | Concurrent SLURM jobs | Handled automatically via file lock |
| `classes.npy` mismatch | Different charset between runs | Delete old `classes.npy`, re-run |

### Monitoring Training

```bash
# Watch training log live
tail -f saved_models/experiments/run_N/training.log

# Check SLURM job status
squeue -u $USER

# View SLURM output
cat experiments_execution/logs/vit_rgts_registers/vit_rgts_4reg_JOBID.out
```
