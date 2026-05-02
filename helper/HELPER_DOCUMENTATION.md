# HTR-Pipeline — Helper Documentation

> **Handwritten Text Recognition (HTR) Pipeline** supporting multiple architectures (CNN-RNN, ViT-RGTS, TorchVision ViT, TrOCR) with CTC-based decoding, trained and evaluated on the IAM Handwriting Dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Environment Setup](#3-environment-setup)
4. [Data Preparation](#4-data-preparation)
5. [Configuration System](#5-configuration-system)
6. [Model Architecture](#6-model-architecture)
7. [Training](#7-training)
8. [Evaluation & Inference](#8-evaluation--inference)
9. [Visualization & Explainability](#9-visualization--explainability)
10. [Experiment Execution (SLURM)](#10-experiment-execution-slurm)
11. [Utility Modules](#11-utility-modules)
12. [Execution Sequence (Quick Reference)](#12-execution-sequence-quick-reference)
13. [Script Reference Table](#13-script-reference-table)

---

## 1. Project Overview

This pipeline implements an end-to-end HTR system with four backbone architectures:

| Architecture | Config File | Description |
|---|---|---|
| **CNN-RNN** | `configs/baseline.yaml` | ResNet-style CNN encoder + BiLSTM decoder (baseline) |
| **ViT-RGTS** | `configs/baseline_vit_rgts.yaml` | Custom Vision Transformer with Register Tokens |
| **TorchVision ViT** | `configs/torchvision_vit.yaml` | Pretrained `vit_b_16` from TorchVision |
| **TrOCR** | `configs/trocr.yaml` | HuggingFace TrOCR encoder for feature extraction |

All architectures share a unified CTC-based training pipeline with architecture-aware optimizer settings.

---

## 2. Folder Structure

```
HTR-Pipeline/
│
├── models.py                          # All neural network architecture definitions
├── eval.sh                            # SLURM script for batch evaluation
├── requirements.txt                   # Python package dependencies
│
├── configs/                           # YAML configuration files
│   ├── config.yaml                    #   Global/shared training settings
│   ├── baseline.yaml                  #   CNN-RNN architecture config
│   ├── baseline_vit_rgts.yaml         #   ViT-RGTS architecture config
│   ├── torchvision_vit.yaml           #   TorchVision ViT architecture config
│   └── trocr.yaml                     #   TrOCR architecture config
│
├── data/IAM/                          # IAM Handwriting Dataset
│   ├── processed_lines/               #   Preprocessed line images + ground truth
│   │   ├── train/                     #     Training images + gt.txt
│   │   ├── val/                       #     Validation images + gt.txt
│   │   ├── test/                      #     Test images + gt.txt
│   │   └── classes.npy                #     Character class mapping (79 classes)
│   └── splits/                        #   Official IAM split files
│       ├── train.uttlist
│       ├── validation.uttlist
│       └── test.uttlist
│
├── scripts/                           # Executable scripts
│   ├── trainer.py                     #   Main training loop
│   ├── pipeline_summary.py            #   ASCII pipeline overview
│   ├── verify_architectures.py        #   Architecture shape verification
│   ├── run_register_experiments.py    #   Batch register-token experiments
│   ├── preprocessing/                 #   Data preparation scripts
│   │   ├── prepare_iam.py             #     IAM dataset preprocessing
│   │   ├── validate_setup.py          #     Environment & data validation
│   │   └── exploratory_data_analysis.py  # Dataset EDA & statistics
│   └── postprocessing/                #   Evaluation & visualization scripts
│       ├── evaluate.py                #     Standalone model evaluation
│       ├── demo.py                    #     Single-image inference demo
│       ├── demo_explainability.py     #     Full explainability demo
│       ├── plot_training_metrics.py   #     Training curve visualization
│       ├── analyze_evaluation.py      #     Per-sample error analysis
│       ├── analyze_register_impact.py #     Register token comparison
│       ├── extract_vit_rgts_features.py  # ViT feature extraction
│       ├── gradcam_vit_rgts.py        #     GradCAM for ViT
│       ├── visualize_register_attention.py  # Register attention maps
│       ├── visualize_htr_register_attention.py  # Multi-model register comparison
│       ├── visualize_character_vit.py          # Character-level attention
│       ├── visualize_character_vit_updated.py  # Enhanced character attention
│       ├── visualize_character_gradcam.py      # Character-level GradCAM
│       ├── batch_process_samples.py   #     Batch attention visualization
│       ├── tsne_register_tokens.py    #     t-SNE of register embeddings
│       └── visualize_token_norms.py   #     Output token norm analysis
│
├── utils/                             # Shared utility modules
│   ├── __init__.py
│   ├── htr_dataset.py                 #   Dataset class & data loading
│   ├── transforms.py                  #   Albumentations augmentation pipelines
│   ├── metrics.py                     #   CER & WER metric classes
│   ├── preprocessing.py               #   Image preprocessing functions
│   ├── attention_extractor.py         #   Attention map extraction from ViT
│   └── visualizer.py                  #   Attention visualization & hooks
│
├── asset/                             # Pretrained model management
│   ├── download_pretrained_models.py  #   Download TorchVision/HuggingFace models
│   └── pretrained_models/             #   Cached pretrained weights
│
├── saved_models/experiments/          # Training output (auto-created)
│   └── run_X/                         #   Each run: config.json, results.csv,
│                                      #   checkpoints, attention_weights/
│
├── experiments_execution/             # SLURM job management
│   ├── submit_all_experiments.sh      #   Submit all 8 experiments
│   ├── submit_single.sh              #   Submit one experiment by number
│   ├── slurm_scripts/                #   Individual SLURM job scripts
│   │   ├── 01_baseline.slurm
│   │   ├── 02_vit_rgts_0reg.slurm
│   │   ├── 03_vit_rgts_2reg.slurm
│   │   ├── 04_vit_rgts_4reg.slurm
│   │   ├── 05_vit_rgts_8reg.slurm
│   │   ├── 06_vit_rgts_16reg.slurm
│   │   ├── 07_torchvision_vit_b16.slurm
│   │   └── 08_trocr_base.slurm
│   └── logs/                         #   SLURM stdout/stderr logs
│
├── output/                            # Generated outputs
│   ├── data_analysis/                #   EDA plots & statistics
│   └── training_plots/               #   Training metric plots
│
├── visualizations/                    # Generated attention visualizations
│   ├── character_attention/
│   ├── character_attention_batch/
│   ├── character_gradcam/
│   └── register_analysis/
│
├── notebook/                          # Jupyter notebooks
│   ├── test.ipynb
│   ├── attention_maps_explore.ipynb
│   └── verify_attention_sample.ipynb
│
├── documents/                         # Project documentation
│   └── *.md                          #   Architecture guides, fix logs, etc.
│
└── helper/                            # This documentation
```

---

## 3. Environment Setup

### Step 1 — Install Dependencies

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Download Pretrained Models (if using TorchVision ViT or TrOCR)

> Downloads and caches pretrained TorchVision ViT and HuggingFace TrOCR weights locally.

```bash
python asset/download_pretrained_models.py
```

### Step 3 — Validate Environment

> Checks Python packages, data directories, config files, model availability, GPU, and dataset integrity.

```bash
python scripts/preprocessing/validate_setup.py
```

---

## 4. Data Preparation

### Step 1 — Prepare IAM Dataset

> Parses IAM XML annotations, crops line images from forms, splits into train/val/test, and writes `gt.txt` ground truth files.

```bash
python scripts/preprocessing/prepare_iam.py \
    <path_to_form_images> \
    <path_to_xml_annotations> \
    <path_to_split_files> \
    <output_directory>
```

**Expected output structure:**
```
data/IAM/processed_lines/
├── train/          # 6482 images + gt.txt
├── val/            # 976 images + gt.txt
├── test/           # 2915 images + gt.txt
└── classes.npy     # 79 character classes
```

### Step 2 — Exploratory Data Analysis (Optional)

> Generates comprehensive statistics and publication-quality plots: split sizes, character frequencies, text lengths, and image dimensions.

```bash
python scripts/preprocessing/exploratory_data_analysis.py \
    --data_path data/IAM/processed_lines \
    --output_dir output/data_analysis
```

---

## 5. Configuration System

The pipeline uses **OmegaConf** for hierarchical YAML configuration. Training is launched by merging a **global config** with an **architecture config**, plus optional CLI overrides.

### Global Config — `configs/config.yaml`

| Parameter | Default | Description |
|---|---|---|
| `data.data_path` | `data/IAM/processed_lines` | Path to processed dataset |
| `data.img_height` | 128 | Input image height (pixels) |
| `data.img_width` | 1024 | Input image width (pixels) |
| `train.lr` | 0.001 | Base learning rate |
| `train.num_epochs` | 50 | Training epochs |
| `train.batch_size` | 8 | Batch size |
| `train.scheduler` | `mstep` | LR scheduler type |
| `train.num_workers` | 8 | DataLoader workers |
| `eval.wer_mode` | `tokenizer` | WER computation mode |

### Architecture Configs

Each architecture overrides `arch.*` parameters:

| Config File | `arch.backbone` | Key Settings |
|---|---|---|
| `baseline.yaml` | `cnn_rnn` | head_type=both, rnn_type=lstm, rnn_layers=3, hidden_size=256 |
| `baseline_vit_rgts.yaml` | `vit_rgts` | patch=8×32, dim=256, depth=6, heads=8, registers=0, dropout=0.1 |
| `torchvision_vit.yaml` | `torchvision_vit` | model=vit_b_16, pretrained=true, registers=0 |
| `trocr.yaml` | `trocr` | model=trocr-base-handwritten, img=384×384 |

### Config Merging Pattern

```bash
python scripts/trainer.py configs/config.yaml configs/<arch>.yaml [key=value overrides]
```

Example:
```bash
python scripts/trainer.py configs/config.yaml configs/baseline_vit_rgts.yaml \
    arch.num_register_tokens=4 \
    train.num_epochs=80
```

---

## 6. Model Architecture

### `models.py` — Unified Model Definitions

> Defines all neural network architectures for HTR under a single `HTRModel` entry point, supporting CNN-RNN, ViT-RGTS, TorchVision ViT, and TrOCR backbones with CTC-based decoding heads.

**Classes:**

| Class | Purpose |
|---|---|
| `BasicBlock` | ResNet-style convolutional block with skip connections |
| `CNN` | Configurable CNN feature extractor with BasicBlocks |
| `CTCtopC` | CNN-only CTC head: Conv2d → class logits |
| `CTCtop` | Linear CTC head: Dropout → Linear → class logits |
| `CTCtopR` | RNN CTC head: BiLSTM/GRU → FC (with optional LayerNorm for ViT) |
| `CTCtopB` | Dual CTC head: RNN + CNN shortcut (architecture-aware return behavior) |
| `ViT_RGTS` | Custom ViT with register tokens and `forward_explain()` |
| `TorchVisionViT` | Pretrained ViT wrapper with `forward_explain()` |
| `TrOCREncoder` | HuggingFace TrOCR encoder with `forward_explain()` |
| `HTRModel` | **Main entry point** — dispatches to backbone + head |

**Architecture-Aware Behavior:**
- ViT models (`is_vit=True`): LayerNorm before head, dropout=0.1
- CNN-RNN models (`is_vit=False`): No LayerNorm, dropout=0.5
- `CTCtopB` with `return_both=True`: Returns tuple during training, single tensor during eval

**Key Method — `HTRModel.forward_explain(x)`:**
Returns `(output, attention_dict)` where `attention_dict` contains per-layer attention weights, register attention patterns, and token norms. Used for all explainability visualizations.

---

## 7. Training

### `scripts/trainer.py` — Main Training Script

> Loads configuration, initializes architecture-aware optimizer/scheduler, trains the model with CTC loss, logs metrics to CSV, saves checkpoints, and optionally extracts attention weights for ViT models.

```bash
python scripts/trainer.py configs/config.yaml configs/<arch>.yaml [overrides]
```

**What it does step-by-step:**

1. Merges config YAML files + CLI overrides via OmegaConf
2. Creates experiment directory: `saved_models/experiments/run_X/`
3. Saves `config.json` for reproducibility
4. Loads IAM dataset with architecture-specific augmentation
5. Initializes `HTRModel` and moves to GPU
6. Sets up **architecture-aware optimizer**:
   - CNN-RNN: AdamW, weight_decay=0.00005, MultiStepLR at [50%, 75%] epochs
   - ViT: AdamW, weight_decay=0.0001, LR×0.5, Warmup(5 epochs) + CosineAnnealingLR
7. Training loop with CTC loss (log_softmax, reduction='sum', batch-normalized)
8. **Architecture-aware gradient clipping**: ViT only (max_norm=5.0), CNN-RNN disabled
9. Validation every epoch, saves best model by CER
10. Logs epoch metrics to `results.csv`
11. For ViT: extracts and saves attention weights every 5 epochs

**Output per run:**
```
saved_models/experiments/run_X/
├── config.json              # Frozen config snapshot
├── results.csv              # Epoch-wise metrics (loss, CER, WER, LR)
├── best_model.pth           # Best checkpoint by validation CER
├── latest_model.pth         # Latest checkpoint
└── attention_weights/       # (ViT only) Saved attention maps
```

### Training Helper Scripts

| Script | What it Does |
|---|---|
| `scripts/pipeline_summary.py` | Prints an ASCII flowchart of the full pipeline for quick reference |
| `scripts/verify_architectures.py` | Validates CNN-RNN and ViT-RGTS forward-pass output shapes with synthetic data |
| `scripts/run_register_experiments.py` | Launches multiple training runs varying register token counts (0/2/4/8/16) |

---

## 8. Evaluation & Inference

### 8.1 Standalone Evaluation

> Loads a trained checkpoint and evaluates CER/WER on validation and/or test splits.

```bash
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/<arch>.yaml \
    model_path=saved_models/experiments/run_X/best_model.pth \
    eval.sets=test
```

### 8.2 Single-Image Inference

> Runs CTC decoding on a single image and prints the recognized text.

```bash
python scripts/postprocessing/demo.py \
    configs/config.yaml configs/<arch>.yaml \
    model_path=saved_models/experiments/run_X/best_model.pth \
    <image_path>
```

### 8.3 Per-Sample Error Analysis

> Analyzes detailed evaluation output: worst/best performing samples, error distribution by length, and per-epoch statistics.

```bash
python scripts/postprocessing/analyze_evaluation.py <path_to_eval_results.csv>
```

### 8.4 Batch Evaluation via SLURM

> Submits evaluation as a SLURM job on the GPU cluster.

```bash
sbatch eval.sh
```

---

## 9. Visualization & Explainability

### 9.1 Training Metrics

> Plots training curves (CTC loss, CER, WER, LR) from experiment `results.csv` files, supporting multi-model comparison.

```bash
# Single model
python scripts/postprocessing/plot_training_metrics.py \
    --experiment_dir saved_models/experiments/run_X

# Compare multiple models
python scripts/postprocessing/plot_training_metrics.py \
    --experiment_dir saved_models/experiments/run_32 \
    --experiment_dir saved_models/experiments/run_37 \
    --output_dir output/training_plots
```

### 9.2 Register Token Impact Analysis

> Compares performance across different register token counts with convergence, overfitting, and stability plots.

```bash
python scripts/postprocessing/analyze_register_impact.py \
    --model_dirs saved_models/experiments/run_A saved_models/experiments/run_B \
    --register_counts 0 4 \
    --output_dir visualizations/register_analysis
```

### 9.3 Attention & Explainability

#### Full Explainability Demo

> Generates model predictions, attention maps, register analysis, and architecture comparison on a single image.

```bash
python scripts/postprocessing/demo_explainability.py \
    --base_config configs/config.yaml \
    --arch_config configs/baseline_vit_rgts.yaml \
    --checkpoint saved_models/experiments/run_X/best_model.pth \
    --image <image_path> \
    --output_dir demo_output
```

#### Character-Level Attention (Enhanced)

> Generates per-character attention heatmaps using real transformer attention weights with layer selection and gradient saliency.

```bash
python scripts/postprocessing/visualize_character_vit_updated.py \
    --image <image_path> \
    --text "ground truth" \
    --weights saved_models/experiments/run_X/best_model.pth \
    --layer last \
    --gradient
```

#### Character-Level GradCAM

> Generates per-character gradient-weighted activation maps showing model focus for each recognized character.

```bash
python scripts/postprocessing/visualize_character_gradcam.py \
    --image <image_path> \
    --text "ground truth" \
    --output_dir visualizations/character_gradcam
```

#### Batch Character Attention

> Processes all sample images from `notebook/sample_images/` with ground truth, generating attention maps for each.

```bash
python scripts/postprocessing/batch_process_samples.py
```

### 9.4 ViT-RGTS Feature Extraction & Analysis

#### Extract Features

> Runs `forward_explain()` on all test samples, saving register embeddings, attention maps, and token norms.

```bash
python scripts/postprocessing/extract_vit_rgts_features.py \
    configs/config.yaml configs/baseline_vit_rgts.yaml \
    model_path=saved_models/experiments/run_X/best_model.pth
```

#### GradCAM for ViT

> Generates GradCAM visualizations showing which patches influence predictions.

```bash
python scripts/postprocessing/gradcam_vit_rgts.py \
    configs/config.yaml configs/baseline_vit_rgts.yaml \
    model_path=saved_models/experiments/run_X/best_model.pth
```

#### Register Attention Visualization

> Displays how register tokens attend to different image regions from pre-extracted features.

```bash
python scripts/postprocessing/visualize_register_attention.py \
    <path_to_vit_rgts_explain_folder> \
    --sample_idx 0 \
    --save <output_path>
```

#### Multi-Model Register Comparison

> Compares register attention patterns across models with different register counts on a single image.

```bash
python scripts/postprocessing/visualize_htr_register_attention.py \
    --image <image_path> \
    --runs run_A run_B run_C \
    --registers 0 4 8 \
    --output_dir visualizations/register_analysis
```

#### t-SNE of Register Embeddings

> Creates 2D t-SNE projections showing how register token embeddings cluster across samples.

```bash
python scripts/postprocessing/tsne_register_tokens.py \
    <path_to_vit_rgts_explain_folder> \
    --max_samples 800
```

#### Token Norm Analysis

> Visualizes L2 norms of output tokens (patches + registers), highlighting norm artifacts absorbed by registers.

```bash
python scripts/postprocessing/visualize_token_norms.py \
    <path_to_vit_rgts_explain_folder> \
    --sample_idx 0
```

---

## 10. Experiment Execution (SLURM)

The `experiments_execution/` directory manages batch experiment submission on the TinyGPU SLURM cluster.

### Submit All 8 Experiments

> Queues all experiments (1 CNN-RNN baseline + 5 ViT-RGTS register variants + 1 TorchVision ViT + 1 TrOCR) via SLURM.

```bash
cd experiments_execution
bash submit_all_experiments.sh
```

### Submit a Single Experiment

> Submits one experiment by number.

```bash
bash experiments_execution/submit_single.sh <experiment_number>
```

| # | Experiment | Config |
|---|---|---|
| 1 | Baseline CNN-RNN | `baseline.yaml` |
| 2 | ViT-RGTS, 0 registers | `baseline_vit_rgts.yaml`, registers=0 |
| 3 | ViT-RGTS, 2 registers | `baseline_vit_rgts.yaml`, registers=2 |
| 4 | ViT-RGTS, 4 registers | `baseline_vit_rgts.yaml`, registers=4 |
| 5 | ViT-RGTS, 8 registers | `baseline_vit_rgts.yaml`, registers=8 |
| 6 | ViT-RGTS, 16 registers | `baseline_vit_rgts.yaml`, registers=16 |
| 7 | TorchVision ViT (vit_b_16) | `torchvision_vit.yaml` |
| 8 | TrOCR Base | `trocr.yaml` |

### SLURM Settings (per job)

- **Partition:** `rtx3080`
- **GPUs:** 1
- **CPUs:** 4
- **Time limit:** 2–6 hours (varies by architecture)

### Monitor Jobs

```bash
squeue -u $USER          # List running/queued jobs
scancel <job_id>          # Cancel a specific job
tail -f experiments_execution/logs/<exp_name>/<job_id>.out  # Live log
```

---

## 11. Utility Modules

All shared code lives in `utils/` and is imported by scripts.

### `utils/htr_dataset.py`

> PyTorch Dataset class that loads IAM line images and transcriptions, applies grayscale normalization, aspect-ratio-preserving resize, padding, and optional augmentation.

- **Class:** `HTRDataset(data_path, split, img_height, img_width, augment, transform)`
- **Used by:** `trainer.py`, `evaluate.py`, `demo.py`, `extract_vit_rgts_features.py`

### `utils/transforms.py`

> Defines three tiers of Albumentations augmentation pipelines for training data.

| Function | Intensity | Used For |
|---|---|---|
| `get_default_transforms()` | Moderate | CNN-RNN baseline |
| `get_vit_strong_transforms()` | Strong | ViT-RGTS, TorchVision ViT |
| `get_extra_strong_transforms()` | Extra strong | TrOCR / large models |

### `utils/metrics.py`

> CER and WER metric classes using edit distance with incremental batch updates.

- **Classes:** `CER`, `WER(mode='tokenizer'|'space')`
- **Used by:** `trainer.py`, `evaluate.py`

### `utils/preprocessing.py`

> Core image preprocessing: grayscale normalization and aspect-ratio-preserving resize with padding.

- **Functions:** `preprocess(img_path)`, `pad_and_resize(img, target_h, target_w)`
- **Used by:** `htr_dataset.py`, `demo.py`, all visualization scripts

### `utils/attention_extractor.py`

> Extracts and analyzes attention maps from ViT models via the `forward_explain()` method.

- **Class:** `AttentionExtractor` — register-to-patch attention, Shannon entropy, layer-specific extraction
- **Function:** `extract_attention_weights(model, images)` — batch extraction
- **Used by:** `trainer.py`, `extract_vit_rgts_features.py`, visualization scripts

### `utils/visualizer.py`

> Modular attention visualization: forward hooks for CNN/RNN activations, heatmap overlays, transformer self-attention visualization, and attention rollout.

- **Classes:** `AttentionVisualizer`, `ActivationCapture` (context manager)
- **Used by:** All visualization scripts in `scripts/postprocessing/`

---

## 12. Execution Sequence (Quick Reference)

Below is the recommended end-to-end execution order:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: SETUP                                                 │
├─────────────────────────────────────────────────────────────────┤
│  1. pip install -r requirements.txt                             │
│  2. python asset/download_pretrained_models.py                  │
│  3. python scripts/preprocessing/validate_setup.py              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: DATA PREPARATION                                      │
├─────────────────────────────────────────────────────────────────┤
│  4. python scripts/preprocessing/prepare_iam.py <args>          │
│  5. python scripts/preprocessing/exploratory_data_analysis.py   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: TRAINING                                              │
├─────────────────────────────────────────────────────────────────┤
│  6. python scripts/trainer.py configs/config.yaml               │
│     configs/<arch>.yaml [overrides]                             │
│                                                                 │
│  — OR for SLURM cluster —                                       │
│  6. bash experiments_execution/submit_all_experiments.sh        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: EVALUATION                                            │
├─────────────────────────────────────────────────────────────────┤
│  7. python scripts/postprocessing/evaluate.py <configs>         │
│     model_path=<checkpoint> eval.sets=test                      │
│  8. python scripts/postprocessing/analyze_evaluation.py <csv>   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: ANALYSIS & VISUALIZATION                              │
├─────────────────────────────────────────────────────────────────┤
│  9.  python scripts/postprocessing/plot_training_metrics.py     │
│  10. python scripts/postprocessing/analyze_register_impact.py   │
│  11. python scripts/postprocessing/demo_explainability.py       │
│  12. python scripts/postprocessing/extract_vit_rgts_features.py │
│  13. python scripts/postprocessing/visualize_*.py               │
│  14. python scripts/postprocessing/tsne_register_tokens.py      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Script Reference Table

A complete alphabetical listing of every executable script with a one-sentence description.

| # | Script Path | What It Does |
|---|---|---|
| 1 | `asset/download_pretrained_models.py` | Downloads and caches pretrained TorchVision ViT and HuggingFace TrOCR model weights locally. |
| 2 | `eval.sh` | SLURM batch script that runs model evaluation on the test set via the compute cluster. |
| 3 | `models.py` | Defines all HTR neural network architectures (CNN-RNN, ViT-RGTS, TorchVision ViT, TrOCR) with CTC heads. |
| 4 | `scripts/pipeline_summary.py` | Prints a human-readable ASCII flowchart of the entire HTR pipeline for quick reference. |
| 5 | `scripts/run_register_experiments.py` | Launches multiple training runs with varying register token counts (0/2/4/8/16) as subprocesses. |
| 6 | `scripts/trainer.py` | Main training loop: loads configs, trains model with CTC loss, logs metrics, saves checkpoints. |
| 7 | `scripts/verify_architectures.py` | Validates that CNN-RNN and ViT-RGTS architectures produce correct output shapes with synthetic data. |
| 8 | `scripts/preprocessing/exploratory_data_analysis.py` | Generates comprehensive dataset statistics and publication-quality EDA plots for the IAM dataset. |
| 9 | `scripts/preprocessing/prepare_iam.py` | Parses IAM XML annotations, crops line images, splits into train/val/test, and writes ground truth files. |
| 10 | `scripts/preprocessing/validate_setup.py` | Validates the entire environment: packages, data paths, configs, GPU availability, and dataset integrity. |
| 11 | `scripts/postprocessing/analyze_evaluation.py` | Analyzes per-sample evaluation results: worst/best samples, error distribution, and length-based analysis. |
| 12 | `scripts/postprocessing/analyze_register_impact.py` | Compares model performance across different register token counts with convergence and stability plots. |
| 13 | `scripts/postprocessing/batch_process_samples.py` | Batch-processes all sample images to generate character-level attention visualizations from ground truth. |
| 14 | `scripts/postprocessing/demo.py` | Runs single-image CTC inference on a trained model and prints the recognized text. |
| 15 | `scripts/postprocessing/demo_explainability.py` | Full explainability demo: predictions, attention maps, register analysis, and architecture comparison. |
| 16 | `scripts/postprocessing/evaluate.py` | Standalone evaluation script that computes CER/WER on validation and/or test splits from a checkpoint. |
| 17 | `scripts/postprocessing/extract_vit_rgts_features.py` | Extracts register embeddings, attention maps, and token norms from ViT-RGTS for all test samples. |
| 18 | `scripts/postprocessing/gradcam_vit_rgts.py` | Generates GradCAM visualizations showing which image patches most influence ViT predictions. |
| 19 | `scripts/postprocessing/plot_training_metrics.py` | Plots training curves (loss, CER, WER, LR) from experiment CSV files with multi-model comparison. |
| 20 | `scripts/postprocessing/tsne_register_tokens.py` | Creates t-SNE 2D projections of register token embeddings to visualize clustering patterns. |
| 21 | `scripts/postprocessing/visualize_character_gradcam.py` | Generates per-character GradCAM heatmaps showing model focus for each recognized character. |
| 22 | `scripts/postprocessing/visualize_character_vit.py` | Generates per-character attention heatmaps using ViT self-attention weights overlaid on input images. |
| 23 | `scripts/postprocessing/visualize_character_vit_updated.py` | Enhanced character attention with real transformer weights, layer selection, and gradient saliency. |
| 24 | `scripts/postprocessing/visualize_htr_register_attention.py` | Compares register attention patterns across multiple models with different register configurations. |
| 25 | `scripts/postprocessing/visualize_register_attention.py` | Visualizes how register tokens attend to image regions from pre-extracted ViT-RGTS features. |
| 26 | `scripts/postprocessing/visualize_token_norms.py` | Visualizes L2 norms of output tokens to highlight norm artifacts absorbed by register tokens. |
| 27 | `utils/attention_extractor.py` | Extracts and analyzes attention maps from ViT models via the `forward_explain()` method. |
| 28 | `utils/htr_dataset.py` | PyTorch Dataset class for loading IAM images with preprocessing, resizing, and augmentation. |
| 29 | `utils/metrics.py` | CER and WER metric classes using edit distance with incremental batch updates. |
| 30 | `utils/preprocessing.py` | Core image preprocessing: grayscale normalization and aspect-ratio-preserving resize with padding. |
| 31 | `utils/transforms.py` | Defines three-tier Albumentations augmentation pipelines (moderate, strong, extra-strong). |
| 32 | `utils/visualizer.py` | Modular attention visualization utilities: hooks, heatmaps, rollout, and self-attention displays. |

---

*Document generated for the HTR-Pipeline project. Refer to `documents/` folder for in-depth technical guides on specific topics.*
