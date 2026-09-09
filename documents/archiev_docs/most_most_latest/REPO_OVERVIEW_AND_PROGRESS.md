# HTR-Pipeline: Complete Repository Overview & Progress Report

> **Last Updated**: June 2025  
> **Purpose**: Comprehensive walkthrough of the entire HTR (Handwritten Text Recognition) pipeline — architecture, codebase, experiments, results, and current status.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Repository Structure](#2-repository-structure)
3. [Architecture Deep-Dive](#3-architecture-deep-dive)
4. [Configuration System](#4-configuration-system)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Pipeline](#6-training-pipeline)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Experiment Results](#8-experiment-results)
9. [Attention & Explainability](#9-attention--explainability)
10. [Synthetic Data Generation](#10-synthetic-data-generation)
11. [Fine-Tuning System](#11-fine-tuning-system)
12. [Notebooks & Visualization](#12-notebooks--visualization)
13. [Key Lessons & Gotchas](#13-key-lessons--gotchas)
14. [Current Status & Next Steps](#14-current-status--next-steps)

---

## 1. Project Summary

This project implements a **Handwritten Text Recognition (HTR)** pipeline based on the DAS 2022 "Best Practices" paper, extended with:

- **Vision Transformer (ViT-RGTS v2)** — a custom hybrid CNN-stem + Transformer model with register tokens
- **Pretrained model support** — TorchVision ViT-B/16 and HuggingFace TrOCR
- **Register tokens** — from "Vision Transformers Need Registers" (Darcet et al. 2023) for improved attention maps and explainability
- **Comprehensive experiment infrastructure** — SLURM-based, with automatic run numbering, CSV logging, attention extraction
- **Synthetic data pipeline** — CC100 text + crawled fonts → rendered line images

### Best Results Achieved

| Model | Best Val CER | Best Test CER | Run |
|-------|-------------|---------------|-----|
| **CNN-RNN Baseline** | **3.50%** | **5.02%** | run_73 |
| ViT-RGTS v2 (6 reg) | 4.08% | 6.05% | run_67 |
| ViT-RGTS v2 (7 reg) | 4.11% | 5.99% | run_68 |
| ViT-RGTS v2 (5 reg) | 4.17% | 6.04% | run_65 |
| ViT-RGTS v2 (14 reg) | 4.19% | 6.03% | run_62 |
| TrOCR (fine-tuned) | 13.21% | 15.47% | run_98 |
| TorchVision ViT-B/16 | 26.06% | 28.61% | run_100 |

> The CNN-RNN baseline remains the strongest in pure CER terms. ViT-RGTS v2 is competitive (~6% test CER) while providing **explainability via attention maps and register tokens** — its primary value proposition.

---

## 2. Repository Structure

```
HTR-Pipeline/
│
├── models.py                    # ALL model architectures (single file)
├── index2letter.json            # Character index → letter mapping
├── letter2index.json            # Letter → character index mapping
├── requirements.txt             # Python dependencies
├── README.md                    # Project README
│
├── configs/                     # YAML configuration files
│   ├── baseline.yaml            # CNN-RNN baseline
│   ├── baseline_vit_rgts.yaml   # ViT-RGTS v1 (DEPRECATED)
│   ├── baseline_vit_rgts_v2.yaml # ViT-RGTS v2 (RECOMMENDED)
│   ├── torchvision_vit.yaml     # Pretrained ViT-B/16
│   ├── trocr.yaml               # HuggingFace TrOCR
│   ├── config.yaml              # Legacy default config
│   └── finetune_*.yaml          # Fine-tuning configs
│
├── scripts/                     # Executable scripts
│   ├── trainer.py               # Main training loop (~1115 lines)
│   ├── attention_visualization.py
│   ├── register_attention_analysis.py
│   ├── fig5_attention_maps.py   # Paper figure reproduction
│   ├── pub_fig*.py              # Publication figure scripts
│   ├── extract_synthetic_to_iam_format.py
│   └── postprocessing/          # Post-training analysis scripts
│
├── utils/                       # Utility modules
│   ├── preprocessing.py         # Image loading & preprocessing
│   ├── htr_dataset.py           # PyTorch Dataset class
│   ├── transforms.py            # Data augmentation (3 tiers)
│   ├── metrics.py               # CER & WER computation
│   ├── attention_extractor.py   # Attention map extraction & analysis
│   ├── finetuning.py            # LLRD, gradual unfreezing, parameter groups
│   └── visualizer.py            # Hook-based feature visualization
│
├── data/IAM/                    # IAM dataset (Aachen splits)
│   └── processed_lines/
│       ├── train/gt.txt         # Training split
│       ├── val/gt.txt           # Validation split
│       └── test/gt.txt          # Test split
│
├── saved_models/experiments/    # All experiment outputs
│   ├── run_60/ through run_101/ # Individual experiment directories
│   │   ├── config.json          # Full config snapshot
│   │   ├── model.pt             # Model weights
│   │   ├── results.csv          # Epoch-wise CER/WER/loss
│   │   ├── evaluation_details.csv # Per-sample predictions
│   │   ├── training.log         # Training console output
│   │   └── attention_weights/   # Saved attention maps (ViT only)
│   └── ...
│
├── experiments_execution/       # SLURM job submission infrastructure
│   ├── slurm_scripts/           # Per-experiment SLURM scripts
│   ├── submit_all_experiments.sh
│   ├── submit_single.sh
│   └── logs/
│
├── synthetic_data_generation/   # Synthetic data creation pipeline
├── notebook/                    # Jupyter notebooks for analysis
├── outputs/                     # Generated visualizations
│   ├── attention_maps/
│   ├── gradcam/
│   └── pub_figures/
│
├── documents/                   # Project documentation
│   ├── most_latest/             # Current documentation (YOU ARE HERE)
│   └── latest/
│
└── helper/                      # Guides and reference material
```

---

## 3. Architecture Deep-Dive

All models are defined in **`models.py`** (~1200 lines). The main entry point is `HTRNet`, which routes to one of four architecture families.

### 3.1 HTRNet — Unified Wrapper

```python
class HTRNet(nn.Module):
    """
    arch_cfg.type determines the backbone:
      - 'cnn_rnn'         → CNN + CTC head
      - 'vit_rgts'        → ViT-RGTS v2 (custom transformer)
      - 'torchvision_vit'  → Pretrained ViT-B/16
      - 'trocr'           → HuggingFace TrOCR encoder
    
    Forward returns: logits [T, B, nclasses] (CTC-ready)
    """
```

All architectures follow the same pattern:
```
Input Image [B, 1, 128, 1024] → Backbone → Features [T, B, D] → CTC Head → Logits [T, B, nclasses]
```

### 3.2 CNN-RNN Baseline (Best CER: 5.02%)

```
Input [B, 1, 128, 1024]
  → Conv 7×7 (stride 4,2) → 32 channels
  → [2× BasicBlock(64)] → MaxPool
  → [3× BasicBlock(128)] → MaxPool
  → [2× BasicBlock(256)]
  → MaxPool (collapse height → 1) → [B, 256, 1, W]
  → CTCtopB (dual head):
      ├── 3-layer BiLSTM (hidden=256) → [T, B, nclasses]  (primary)
      └── Conv 1×3                     → [T, B, nclasses]  (auxiliary, 0.1× weight)
```

**Why it works**: CNN + pooling naturally collapses height and preserves left-to-right order. BiLSTM captures long-range dependencies. Dual supervision provides gradient regularization.

### 3.3 ViT-RGTS v2 — Custom Transformer (Best CER: 5.99%)

The key innovation — a **hybrid CNN-stem + Transformer** designed for CTC-based HTR:

```
Input [B, 1, 128, 1024]
  → CNN Stem (4 conv layers):
      Conv 7×7 (stride 4,2) → [B, 32, 32, 512]
      Conv 3×3 (stride 2,2) → [B, 64, 16, 256]
      Conv 3×3 (stride 2,2) → [B, 128, 8, 128]
      Conv 3×3 (stride 2,1) → [B, 256, 4, 128]
      AdaptiveMaxPool(1,_)  → [B, 256, 1, 128]  ← height collapsed to 1!
  → 128 column tokens [B, 128, 256]
  → Prepend R register tokens → [B, R+128, 256]
  → Add positional embeddings
  → 6-layer Transformer Encoder (dim=256, 8 heads, MLP=1024)
  → Split: register_tokens [B, R, 256] + patch_tokens [B, 128, 256]
  → patch_tokens → CTCtopB → logits [T, B, nclasses]
```

**Critical Design Decisions**:
- **CNN stem collapses height → 1**: This produces a left-to-right token sequence that CTC needs. v1 used raw patch embedding (8×64 grid flattened row-major) which destroyed spatial order → CER stuck at 75%.
- **~3 tokens per character**: 128 tokens for ~40 chars = 3.2 tokens/char, ideal for CTC alignment.
- **Register tokens**: Learnable global tokens (indices 0..R-1) that attend to all patches. They capture holistic style/writer information without polluting patch token representations.

**Model Size**: ~6M parameters (comparable to CNN baseline's 7.4M)

### 3.4 TorchVision ViT-B/16 — Pretrained (Best CER: 28.6%)

```
Input [B, 1, 128, 1024]
  → gray_to_rgb (1→3 channels)
  → ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  → Patch embedding (16×16 patches) → [B, Hp×Wp, 768]
  → CLS token + optional register tokens
  → Positional embedding interpolation (224² → 128×1024)
  → 12-layer Transformer (dim=768, 12 heads)
  → Extract patch tokens → CTC head
```

**Current status**: Not competitive yet (28.6% CER). Needs better fine-tuning strategy or more epochs.

### 3.5 TrOCR Encoder — HuggingFace (Best CER: 15.5%)

```
Input [B, 1, 128, 1024]
  → gray_to_rgb
  → Aspect-ratio resize + pad → [B, 3, 384, 384]
  → TrOCR-base encoder (ViT, 12 layers, dim=768)
  → Remove CLS token → [B, 24×24, 768]
  → Mean-pool over height (24→1) → [B, 24, 768]
  → Repeat-interleave ×4 → [B, 96, 768]  (enough timesteps for CTC)
  → CTC head
```

**Key design**: Aspect-ratio preservation prevents 8× horizontal squash that destroys character shapes.

### 3.5 CTC Head Variants

| Head | Class | Description |
|------|-------|-------------|
| `cnn` | `CTCtopC` | Conv 1×3 + Dropout(0.5) |
| `rnn` | `CTCtopR` | BiGRU/BiLSTM + Linear. LayerNorm for ViT. Reduced dropout (0.1) for ViT. |
| `both` | `CTCtopB` | RNN + CNN shortcut. Returns `(rnn_out, cnn_out)` during training for dual CTC loss. Eval returns only RNN output. |
| `linear` | `CTCtopLinear` | Simple linear projection (for testing) |

---

## 4. Configuration System

Uses **OmegaConf** for YAML-based configuration with CLI override support.

### Usage Pattern

```bash
# Use a config file directly
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml

# Override specific parameters via CLI
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml arch.num_registers=8 train.lr=0.0005
```

### Key Configs

| Config File | Architecture | Key Settings |
|-------------|-------------|--------------|
| `baseline.yaml` | CNN-RNN | cnn_cfg=[[2,64],'M',[3,128],'M',[2,256]], head=both, LSTM×3, hidden=256 |
| `baseline_vit_rgts_v2.yaml` | ViT-RGTS v2 | use_cnn_stem=true, dim=256, depth=6, heads=8, registers=4, head=both |
| `torchvision_vit.yaml` | ViT-B/16 | pretrained=true, 768-dim, 12 layers, optional registers |
| `trocr.yaml` | TrOCR | microsoft/trocr-base-handwritten, 384×384, freeze_encoder configurable |

### Config Structure

```yaml
resume: null          # Path to checkpoint for resume
device: cuda:0

data:
  path: ./data/IAM/processed_lines
  mode: iam           # 'iam' or 'synthetic'

preproc:
  image_height: 128
  image_width: 1024

train:
  lr: 0.001
  num_epochs: 80
  batch_size: 8
  scheduler: mstep
  save_every_k_epochs: 1
  num_workers: 8
  augmentation: cnn    # 'auto', 'cnn', 'vit', or 'vit_strong'

eval:
  batch_size: 8
  num_workers: 8
  wer_mode: tokenizer  # NLTK tokenizer-based WER

arch:
  type: vit_rgts       # 'cnn_rnn', 'vit_rgts', 'torchvision_vit', 'trocr'
  # ... architecture-specific parameters
```

---

## 5. Data Pipeline

### 5.1 Dataset: IAM Handwriting Database

- **Aachen splits**: train / val / test
- **Format**: `processed_lines/{split}/gt.txt` where each line: `image_path transcription`
- **Image format**: Grayscale line images, variable width
- **Character set**: 79 characters (A-Z, a-z, digits, punctuation, space) → **80 classes** (79 + CTC blank at index 0)
- **Training set**: ~6,482 lines
- **Validation set**: ~976 lines
- **Test set**: ~2,915 lines

### 5.2 Image Preprocessing (`utils/preprocessing.py`)

```python
def load_image(image_path):
    image = imread(image_path)
    image = rgb2gray(image)     # if color
    image = 1 - image / 255.0   # invert: white background → 0, dark ink → 1
    return image

def preprocess(img, input_size=(128, 1024), border_size=8):
    # Aspect-ratio preserving resize to fit within (128-16) × (1024-16)
    # Then pad to 128×1024 with median value
```

**Training-time random resize**: width ×[0.75, 1.25], height ×[0.9, 1.1] — simple but effective augmentation built into the dataset.

### 5.3 Data Augmentation (`utils/transforms.py`)

Three augmentation tiers using **Albumentations**:

| Tier | Name | Used For | Key Transforms |
|------|------|----------|-----------------|
| `aug_transforms_cnn` | Moderate | CNN-RNN, pretrained models | Affine(±1°), GridDistortion, Morphological, BrightnessContrast |
| `aug_transforms_vit` | Strong | ViT from scratch | Affine(±10°), Perspective, ElasticTransform, GaussNoise, CoarseDropout, Blur |
| `aug_transforms_vit_strong` | Extra Strong | Large pretrained (>100M params) | Affine(±15°), stronger distortion/noise/erasing |

**Auto-selection** in trainer.py: architecture type → appropriate augmentation tier (unless overridden via `train.augmentation`).

### 5.4 Character Encoding

- **CTC blank** is at **index 0** (PyTorch convention)
- Character at model output index `i` → `charset[i - 1]`
- Dictionary built fresh from training data's gt.txt
- Space is always added to the character set
- For synthetic data mode: unified charset = synthetic ∪ IAM test

---

## 6. Training Pipeline

### 6.1 Entry Point: `scripts/trainer.py`

The complete training workflow:

```
Parse config (OmegaConf + CLI overrides)
  → setup_experiment_dir() [auto run numbering with file lock]
  → HTRTrainer.__init__()
      ├── prepare_dataloaders()    [IAM or synthetic mode]
      ├── prepare_net()            [HTRNet init, optional resume]
      ├── prepare_losses()         [CTC loss with sum reduction]
      └── prepare_optimizers()     [Architecture-specific optimizer/scheduler]
  → for epoch in 1..max_epochs:
      ├── train(epoch)             [forward, CTC loss, backward, gradient accumulation]
      ├── scheduler.step()
      ├── save(epoch)              [model.pt to experiment dir]
      ├── test(epoch, 'val')       [CER/WER on validation]
      ├── test(epoch, 'test')      [CER/WER on test]
      ├── extract_attention_weights() [ViT: every 5 epochs]
      └── CSV logging              [epoch, lr, loss, val/test CER/WER, timing]
```

### 6.2 Experiment Directory Structure

Each run creates `saved_models/experiments/run_N/` containing:

| File | Description |
|------|-------------|
| `config.json` | Full configuration snapshot |
| `model.pt` | Model state dict (overwritten each save) |
| `results.csv` | Per-epoch metrics: epoch, lr, loss, val/test CER/WER, params, time |
| `evaluation_details.csv` | Per-sample predictions: ground truth, prediction, sample CER/WER |
| `training.log` | Full console output |
| `attention_weights/epoch_NNN/` | NumPy arrays: `sample_XX_layer_YY.npy`, `token_norms.npy`, `register_tokens.npy`, `groundtruth.txt` |

**Race-condition safe**: Uses `fcntl.LOCK_EX` file lock to prevent concurrent SLURM jobs from claiming the same run number.

### 6.3 Architecture-Specific Optimizer Settings

| Architecture | Optimizer | LR Strategy | Weight Decay | Scheduler |
|-------------|-----------|-------------|-------------|-----------|
| **CNN-RNN** | AdamW | Single LR=0.001 | 5e-5 | MultiStepLR at 50%/75% |
| **ViT-RGTS v2** | AdamW (3 groups) | stem=1e-3, transformer=5e-4, head=1e-3 | stem=5e-4, transformer=5e-3, head=1e-4 | 5-epoch warmup + cosine annealing |
| **TorchVision ViT** | AdamW (2 groups or LLRD) | backbone=2e-5, head=5e-4 | backbone=0.01, head=1e-4 | Warmup + cosine |
| **TrOCR** | AdamW | encoder=3e-5, head=5e-4 (or frozen encoder) | 0.01 / 1e-4 | Warmup + cosine |

**ViT-RGTS v2 key fix**: Previous runs used too-high weight decay (0.01-0.05 vs baseline's 5e-5) which killed learning. Fixed by reducing WD 10-20× and separating head from transformer group.

### 6.4 Training Features

- **Gradient accumulation**: Effective batch = batch_size × accum_steps (default 2 for ViT, 1 for CNN)
- **Gradient clipping**: max_norm=10.0 for ViT architectures only
- **Dual CTC loss** (head_type='both'): `loss = CTC(rnn_out) + 0.1 × CTC(cnn_out)`
- **CTC loss normalization**: `CTCLoss(sum) / batch_size`

---

## 7. Evaluation & Metrics

### 7.1 CER (Character Error Rate)

```python
CER = edit_distance(prediction, target) / len(target)
```

Accumulated across all samples, then: `total_distance / total_length`.

### 7.2 WER (Word Error Rate)

Two modes:
- **`tokenizer`** (default): Uses NLTK `word_tokenize()` — handles punctuation correctly
- **`space`**: Simple space-based splitting

### 7.3 CTC Decoding

Greedy decoding (no beam search):
```python
def decode(tdec, tdict, blank_id=0):
    tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j-1]]  # remove consecutive duplicates
    return ''.join([tdict[t] for t in tt if t != blank_id])            # remove blanks
```

---

## 8. Experiment Results

### 8.1 Complete Results Table (runs 60-101)

#### ViT-RGTS v2 Register Sweep (80 epochs)

| Run | Registers | Val CER | Test CER | Notes |
|-----|-----------|---------|----------|-------|
| run_67 | 6 | **4.08%** | **6.05%** | Best ViT val CER |
| run_70 | 15 | 4.09% | 6.05% | |
| run_68 | 7 | 4.11% | **5.99%** | Best ViT test CER |
| run_65 | 5 | 4.17% | 6.04% | |
| run_62 | 14 | 4.19% | 6.03% | |
| run_61 | 10 | 4.22% | 6.03% | |
| run_64 | 1 | 4.23% | 6.04% | |
| run_66 | 3 | 4.25% | 6.27% | |
| run_60 | 13 | 4.29% | 6.16% | |
| run_63 | 8 | 4.29% | 6.26% | |
| run_71 | 16 | 4.19% | 6.10% | |
| run_69 | 9 | 4.17% | 6.19% | |

**Key finding**: Register count has **minimal impact on CER** (all within 4.08-4.29% val CER). The model performs similarly with 1-16 registers. The optimal range appears to be **5-7 registers** for the best balance of performance and attention quality.

#### ViT-RGTS v2 Register Sweep (50 epochs, different seed/run)

| Run | Registers | Val CER | Test CER |
|-----|-----------|---------|----------|
| run_81 | 15 | 4.24% | 6.37% |
| run_82 | 2 | 4.31% | 6.34% |
| run_89 | 13 | 4.30% | 6.35% |
| run_91 | 9 | 4.34% | 6.34% |
| run_92 | 7 | 4.36% | 6.30% |
| run_76 | 0 | 4.45% | 6.43% |
| run_77 | 16 | 4.35% | 6.42% |
| run_84 | 4 | 4.39% | 6.47% |

#### CNN-RNN Baseline

| Run | Val CER | Test CER |
|-----|---------|----------|
| run_73 | **3.50%** | **5.02%** |

#### Pretrained Models

| Run | Model | Val CER | Test CER | Epochs | Notes |
|-----|-------|---------|----------|--------|-------|
| run_98 | TrOCR | 13.21% | 15.47% | 50 | Best pretrained result |
| run_100 | ViT-B/16 | 26.06% | 28.61% | 50 | With 4 registers |
| run_96 | TrOCR | 66.37% | 66.49% | 30 | Insufficient training |
| run_101 | ViT-B/16 | 77.81% | 77.84% | 60 | Training collapsed |
| run_97 | ViT-B/16 | 73.41% | 73.66% | 40 | Training collapsed |
| run_99 | ViT-B/16 | 85.73% | 86.31% | 10 | Very early |

### 8.2 Summary of Findings

1. **CNN-RNN baseline is hard to beat** on IAM with only ~6.5K training samples
2. **ViT-RGTS v2 is competitive** (~6% test CER vs 5% baseline) while providing explainability
3. **Register count barely affects CER** — the model is robust to R=1..16
4. **Pretrained models struggle** — likely need better fine-tuning strategy (LLRD, gradual unfreezing) or more data
5. **80 epochs > 50 epochs** — the 80-epoch runs consistently beat 50-epoch runs by ~0.3-0.4% CER

---

## 9. Attention & Explainability

### 9.1 Forward Explain

Each ViT backbone implements `forward_explain()` which returns:
```python
seq_tokens,    # [T, B, D]           — patch embeddings (CTC input)
reg_tokens,    # [B, R, D]           — register token embeddings
attn_maps,     # List[L] of [B, H, S, S]  — per-layer, per-head attention weights
token_norms,   # [B, S]              — L2 norm per token at final layer
grid_size      # (Hp, Wp)            — spatial grid dimensions
```

For ViT-RGTS v2: S = R + 128 (registers + column tokens), L = 6 layers, H = 8 heads.

### 9.2 Attention Extractor (`utils/attention_extractor.py`)

```python
extractor = AttentionExtractor(model)
result = extractor.extract_from_forward(images)
# result['attention_maps']  → List of [B, H, S, S] per layer
# result['register_tokens'] → [B, R, D]
# result['token_norms']     → [B, S]

# Analyze register behavior
stats = extractor.analyze_register_attention(attn_maps, num_registers=4)
# stats['layer_N']['register_to_patch']  → [R, P] attention distribution
# stats['layer_N']['register_attention_entropy'] → scalar
```

### 9.3 Automatic Attention Saving

During training, attention weights are automatically saved every 5 epochs to:
```
saved_models/experiments/run_N/attention_weights/epoch_NNN/
  ├── sample_000_layer_00.npy  through  sample_000_layer_05.npy
  ├── sample_000_token_norms.npy
  ├── sample_000_register_tokens.npy
  ├── sample_000_groundtruth.txt
  ├── ... (5 samples per epoch)
  └── metadata.json
```

### 9.4 Visualization Scripts

| Script | Purpose |
|--------|---------|
| `scripts/attention_visualization.py` | Load checkpoint → forward_explain → heatmaps |
| `scripts/register_attention_analysis.py` | Quantitative register-to-patch attention analysis |
| `scripts/fig5_attention_maps.py` | Reproduce paper Figure 5 (character-level attention) |
| `scripts/pub_fig1_paper_fig5.py` | Publication-quality attention figure |
| `scripts/pub_fig2_register_comparison.py` | Compare attention with/without registers |
| `scripts/pub_fig3_gradcam_quantitative.py` | GradCAM visualization |
| `scripts/pub_fig5_character_attention.py` | Per-character attention heatmaps |

---

## 10. Synthetic Data Generation

Located in `synthetic_data_generation/`. A multi-step pipeline to create rendered handwriting-style line images.

### Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 0 | `font_crawling_with_license.ipynb` | Crawl open-license handwriting fonts |
| 1 | `font_download.py` / `1_extract_fonts.py` | Download and extract font files |
| 2 | `lines_extraction_from_CC100_multiprocessing.py` | Extract English text lines from CommonCrawl CC100 |
| 2-alt | `2_alternative_text_generation.py` | Alternative text generation method |
| 3 | `get_length_distribution_of_IAM.py` | Analyze IAM line length distribution (match it) |
| 4 | `cc100_random_subset_preprocessing_multiprocessing.py` | Preprocess and filter CC100 text |
| 4-1 | `remove_unknown_characters_multiprocessing.py` | Remove chars not in target charset |
| 5-x | `font_check_*.py` | Validate fonts render all characters correctly |
| 6 | `text_rendering_filter_...LMDB.py` | Render text with fonts → LMDB database |

### Integration with Training

- **Extraction**: `scripts/extract_synthetic_to_iam_format.py` converts LMDB → IAM directory structure
- **Training**: Set `data.mode: synthetic` and `data.synthetic_path: /path/to/synthetic_processed_lines`
- **Character unification**: Trainer automatically computes `synth_chars ∪ iam_chars` for the model vocabulary
- **SLURM**: `render_1M_images.slurm` for HPC rendering

---

## 11. Fine-Tuning System

Located in `utils/finetuning.py`. Implements state-of-the-art fine-tuning strategies for pretrained models.

### 11.1 Layer-wise Learning Rate Decay (LLRD)

```python
groups = get_vit_layer_groups(net, 'torchvision_vit')
# Returns groups from deepest to shallowest:
# embedding → block_0..block_11 → final_ln → adapters → head

# Each deeper layer gets higher LR:
# layer_lr = base_lr × decay_rate^(num_layers - layer_idx)
```

### 11.2 Gradual Unfreezing

```python
unfreezer = GradualUnfreezer(net, arch_type='torchvision_vit',
                              warmup_frozen=3,    # epochs with frozen backbone
                              unfreeze_every=3)   # unfreeze 1 layer every N epochs
# Epoch 1-3:  Only head trains
# Epoch 4:    Unfreeze block_11
# Epoch 7:    Unfreeze block_10
# ...
```

### 11.3 Configuration

```yaml
finetune:
  enabled: true
  base_lr: 1e-5
  head_lr: 5e-4
  lr_decay_rate: 0.65
  weight_decay: 0.01
  head_weight_decay: 0.0001
  gradual_unfreeze: true
  unfreeze_warmup: 3
  unfreeze_every: 3
```

---

## 12. Notebooks & Visualization

### Key Notebooks (in `notebook/`)

| Notebook | Purpose |
|----------|---------|
| `01_attention_maps_register_effect.ipynb` | Register contribution analysis across layers |
| `02_gradcam_register_effect.ipynb` | GradCAM visualization with/without registers |
| `attention_final.ipynb` | Comprehensive attention study |
| `character_attention_visualization.ipynb` | Per-character attention heatmaps |
| `fig5_attention_maps.ipynb` | Paper Figure 5 reproduction |
| `pretrained_model_use.ipynb` | TorchVision/TrOCR evaluation |
| `basic_visualization.ipynb` | Quick tests and sanity checks |

### Output Directories (in `outputs/`)

```
outputs/
├── attention_maps/     # Attention heatmaps
├── gradcam/            # GradCAM visualizations
├── pub_figures/        # Publication-ready figures
└── viz_gradcam/        # GradCAM overlays on images
```

---

## 13. Key Lessons & Gotchas

### Critical Bugs Fixed

1. **CTC Blank at Index 0**: PyTorch CTC expects blank at index 0. Character indices start at 1. Getting this wrong causes silent training failure.

2. **ViT v1 → v2 (CNN Stem Fix)**: v1 used 16×16 patches producing an 8×64 grid flattened row-major. This destroyed left-to-right order needed by CTC → 75% CER. The CNN stem collapses height to 1, producing proper sequential order → 6% CER.

3. **ImageNet Normalization**: Pretrained ViTs **must** receive ImageNet-normalized inputs. Without it: out-of-distribution features → garbage output.

4. **Weight Decay Too High**: Early ViT runs used WD=0.01-0.05 (following DeiT). On 6.5K samples this killed learning. Fixed by reducing to 5e-4 to 5e-3 (closer to CNN baseline's 5e-5).

5. **TrOCR Aspect Ratio**: Naive resize from 128×1024 → 384×384 squashes text 8× horizontally. Fixed with aspect-ratio-preserving resize + padding.

### Design Insights

- **Small data (6.5K samples) favors CNNs**: CNN's built-in inductive bias (locality, translation invariance) is a huge advantage with limited data.
- **Register tokens don't hurt CER**: Adding registers (even up to 16) doesn't degrade recognition accuracy.
- **Moderate augmentation > strong augmentation** for from-scratch ViT on small data: too-strong augmentation destroys signal before the model learns basics.
- **Dual supervision (CTCtopB)** consistently helps across architectures.
- **Gradient accumulation** (effective batch 16-32) stabilizes ViT training.

---

## 14. Current Status & Next Steps

### What's Complete

- [x] CNN-RNN baseline achieving 5.02% test CER (competitive with published results)
- [x] ViT-RGTS v2 with CNN stem achieving ~6% test CER
- [x] Full register sweep (0-16 registers) showing minimal CER impact
- [x] Attention extraction and visualization pipeline
- [x] Synthetic data generation pipeline
- [x] SLURM experiment infrastructure
- [x] Publication figure generation scripts
- [x] Fine-tuning utilities (LLRD, gradual unfreezing)

### What Needs Work

- [ ] **Pretrained models**: TrOCR and ViT-B/16 not yet competitive. Need:
  - Proper LLRD fine-tuning with `finetune.enabled: true`
  - More training epochs
  - Possibly synthetic pretraining → IAM fine-tuning
- [ ] **Synthetic pretraining**: Pipeline exists but end-to-end synthetic→IAM training not yet validated
- [ ] **Beam search decoding**: Currently using greedy decoding; beam search + language model could improve CER
- [ ] **Writer identification**: Registers theoretically capture writer style, but writer ID experiments not run yet
- [ ] **Register attention analysis**: Quantitative analysis of what registers learn at different layers

### Dependencies

Key packages from `requirements.txt`:
- `torch`, `torchvision`, `transformers` (ML stack)
- `albumentations==2.0.8` (augmentation)
- `omegaconf` (config)
- `editdistance`, `nltk` (metrics)
- `scikit-image`, `opencv-python` (image I/O)
- `matplotlib`, `seaborn` (plotting)
- `einops` (tensor reshaping)
- `tqdm` (progress bars)

### How to Run

```bash
# Setup
conda create -n htr python=3.9
pip install -r requirements.txt

# Train CNN-RNN baseline
python scripts/trainer.py configs/baseline.yaml

# Train ViT-RGTS v2 with 4 registers
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml

# Register sweep
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml arch.num_registers=0
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml arch.num_registers=8

# SLURM batch submission
cd experiments_execution
./submit_single.sh 4   # Submit experiment #4

# Evaluate saved model
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml resume=saved_models/experiments/run_67/model.pt
```
