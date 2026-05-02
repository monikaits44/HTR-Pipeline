# Post-Processing Execution Guide

> **Scope**: 13 scripts in `scripts/postprocessing/` for analyzing trained HTR models.
> **Architectures**: CNN-RNN, ViT-RGTS (with 0/2/4/8/16 registers), TorchVision ViT, TrOCR

---

## Architecture Compatibility Matrix

| # | Script | CNN-RNN | ViT-RGTS | TorchVision ViT | TrOCR |
|:-:|--------|:-------:|:--------:|:---------------:|:-----:|
| 1 | `plot_training_metrics.py` | ✅ | ✅ | ✅ | ✅ |
| 2 | `analyze_register_impact.py` | ❌ | ✅ | ❌ | ❌ |
| 3 | `analyze_evaluation.py` | ✅ | ✅ | ✅ | ✅ |
| 4 | `evaluate.py` | ✅ | ✅ | ✅ | ✅ |
| 5 | `demo.py` | ✅ | ✅ | ✅ | ✅ |
| 6 | `demo_explainability.py` | ❌ | ✅ | ✅ | ❌ |
| 7 | `extract_vit_rgts_features.py` | ❌ | ✅ | ❌ | ❌ |
| 8 | `visualize_token_norms.py` | ❌ | ✅ | ❌ | ❌ |
| 9 | `visualize_register_attention.py` | ❌ | ✅ (R>0) | ❌ | ❌ |
| 10 | `tsne_register_tokens.py` | ❌ | ✅ (R>0) | ❌ | ❌ |
| 11 | `gradcam_vit_rgts.py` | ❌ | ✅ | ❌ | ❌ |
| 12 | `visualize_htr_register_attention.py` | ❌ | ✅ | ❌ | ❌ |
| 13 | `visualize_character_vit_updated.py` | ❌ | ✅ | ✅ | ❌ |

> **R>0** = Requires models trained with at least 1 register token.

---

## Run Directory Reference

| Run | Architecture | Registers | Config | Notes |
|-----|-------------|:---------:|--------|-------|
| run_32 | cnn_rnn | — | `baseline.yaml` | 3-layer LSTM, hidden=256 |
| run_39 | torchvision_vit | 4 | `torchvision_vit.yaml` | Pretrained vit_b_16 |
| run_40 | trocr | — | `trocr.yaml` | trocr-base-handwritten, frozen encoder |
| run_50 | vit_rgts (v2) | 0 | `baseline_vit_rgts_v2.yaml` | dim=256, depth=6, cnn_stem |
| run_51 | vit_rgts (v2) | 2 | `baseline_vit_rgts_v2.yaml` | dim=256, depth=6, cnn_stem |
| run_52 | vit_rgts (v2) | 4 | `baseline_vit_rgts_v2.yaml` | dim=256, depth=6, cnn_stem |
| run_53 | vit_rgts (v2) | 8 | `baseline_vit_rgts_v2.yaml` | dim=256, depth=6, cnn_stem |
| run_54 | vit_rgts (v2) | 16 | `baseline_vit_rgts_v2.yaml` | dim=256, depth=6, cnn_stem |

---

## Execution Order

```
Phase A: Results Analysis (no GPU, no model loading)
  1. plot_training_metrics.py ─────── Training curves from results.csv
  2. analyze_register_impact.py ───── Register comparison bar charts
  3. analyze_evaluation.py ────────── Per-sample error analysis

Phase B: Model Evaluation (GPU recommended)
  4. evaluate.py ──────────────────── Re-run CER/WER on val/test sets

Phase C: Quick Demo (GPU recommended)
  5. demo.py ──────────────────────── Single image prediction
  6. demo_explainability.py ───────── Prediction + attention + registers

Phase D: Feature Extraction (GPU recommended, prerequisite for Phase E)
  7. extract_vit_rgts_features.py ── Extract tokens/features for all test samples

Phase E: Token Analysis (no GPU, requires Phase D output)
  8. visualize_token_norms.py ─────── Token norm heatmaps
  9. visualize_register_attention.py  Register→patch similarity maps
 10. tsne_register_tokens.py ──────── t-SNE of register embeddings

Phase F: Attention Visualization (GPU recommended)
 11. gradcam_vit_rgts.py ──────────── Grad-CAM patch importance
 12. visualize_htr_register_attention.py ── Multi-run attention comparison
 13. visualize_character_vit_updated.py ── Per-character attention maps
```

---

## Detailed Execution — Script by Script

### Prerequisites

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate
```

---

### Phase A: Results Analysis (No GPU Required)

#### 1. `plot_training_metrics.py` — Training Curves

**Purpose**: Plot loss, CER, WER, and learning rate curves from `results.csv`.

**Architectures**: ALL

```bash
# Single run (any architecture)
python scripts/postprocessing/plot_training_metrics.py \
    --model-path saved_models/experiments/run_32/

# Compare all ViT-RGTS v2 register configurations
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
    --model-path saved_models/experiments/run_54/ \
    --metrics loss cer

# Custom output directory
python scripts/postprocessing/plot_training_metrics.py \
    --model-path saved_models/experiments/run_54/ \
    --output-dir ./output/training_plots/
```

**Input**: `saved_models/experiments/run_X/results.csv`, `config.json`
**Output**: `./output/training_plots/*.png`

---

#### 2. `analyze_register_impact.py` — Register Comparison

**Purpose**: Compare register token configurations with bar charts showing best CER, convergence speed, overfitting analysis, training stability, and improvement rates.

**Architectures**: ViT-RGTS only

```bash
# Compare all 5 ViT-RGTS v2 register configurations
python scripts/postprocessing/analyze_register_impact.py \
    --model-path saved_models/experiments/run_50 --registers 0 \
    --model-path saved_models/experiments/run_51 --registers 2 \
    --model-path saved_models/experiments/run_52 --registers 4 \
    --model-path saved_models/experiments/run_53 --registers 8 \
    --model-path saved_models/experiments/run_54 --registers 16

# Compare just two configurations
python scripts/postprocessing/analyze_register_impact.py \
    --model-path saved_models/experiments/run_50 --registers 0 \
    --model-path saved_models/experiments/run_54 --registers 16

# Show plots without saving
python scripts/postprocessing/analyze_register_impact.py \
    --model-path saved_models/experiments/run_50 --registers 0 \
    --model-path saved_models/experiments/run_54 --registers 16 \
    --no-save
```

**Input**: `saved_models/experiments/run_X/results.csv`
**Output**: `./visualizations/register_analysis/register_impact_analysis.png` + terminal statistics

> **Note**: Training curve plots have been moved to `plot_training_metrics.py`. This script focuses exclusively on register-specific comparative analysis (bar charts, convergence speed, stability metrics).

---

#### 3. `analyze_evaluation.py` — Per-Sample Error Analysis

**Purpose**: Analyze `evaluation_details.csv` for worst/best samples, error distributions, and length-based analysis.

**Architectures**: ALL (architecture-agnostic, reads CSV only)

```bash
# Analyze any run's evaluation results
python scripts/postprocessing/analyze_evaluation.py \
    saved_models/experiments/run_32/evaluation_details.csv

python scripts/postprocessing/analyze_evaluation.py \
    saved_models/experiments/run_54/evaluation_details.csv

python scripts/postprocessing/analyze_evaluation.py \
    saved_models/experiments/run_40/evaluation_details.csv
```

**Input**: `evaluation_details.csv`
**Output**: Terminal output with statistics

---

### Phase B: Model Evaluation (GPU Recommended)

#### 4. `evaluate.py` — Re-run CER/WER Evaluation

**Purpose**: Re-evaluate a saved model checkpoint on validation/test sets.

**Architectures**: ALL

```bash
# CNN-RNN (run_32)
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml \
    resume=saved_models/experiments/run_32/model.pt

# ViT-RGTS v2 (run_54)
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt

# TorchVision ViT (run_39)
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml configs/torchvision_vit.yaml \
    resume=saved_models/experiments/run_39/model.pt

# TrOCR (run_40)
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml configs/trocr.yaml \
    resume=saved_models/experiments/run_40/model.pt

# Test set only
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml \
    resume=saved_models/experiments/run_32/model.pt eval_sets=test
```

**Input**: Config YAML(s) + `model.pt`
**Output**: Terminal CER/WER results, `evaluation_details.csv`

---

### Phase C: Quick Demo (GPU Recommended)

#### 5. `demo.py` — Single Image Prediction (KEEP AS-IS)

**Purpose**: Load a trained model and predict text from a single image.

**Architectures**: ALL

> This script is kept as-is per project requirements. See the script header for usage.

---

#### 6. `demo_explainability.py` — Explainability Demo (KEEP AS-IS)

**Purpose**: Prediction with attention maps and register token visualization.

**Architectures**: ViT-RGTS, TorchVision ViT

> This script is kept as-is per project requirements. See the script header for usage.

---

### Phase D: Feature Extraction (GPU Recommended)

#### 7. `extract_vit_rgts_features.py` — Extract Tokens & Features

**Purpose**: Run the trained ViT-RGTS model on the test set and extract per-sample register tokens, sequence tokens, logits, and token norms as `.npy` files.

**Architectures**: ViT-RGTS only (uses `forward_explain()` method)

**This is a prerequisite for Phase E scripts (8, 9, 10).**

```bash
# Extract features from run_54 (16 registers)
python scripts/postprocessing/extract_vit_rgts_features.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt

# Extract features from run_50 (0 registers)
python scripts/postprocessing/extract_vit_rgts_features.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_50/model.pt

# Use GPU for faster extraction
python scripts/postprocessing/extract_vit_rgts_features.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt device=cuda
```

**Input**: Config YAML(s) + `model.pt`
**Output**: `saved_models/experiments/run_X/vit_rgts_explain/` containing:
- `register_tokens_XXXX.npy` — Register token embeddings per sample
- `sequence_tokens_XXXX.npy` — Patch token embeddings per sample
- `logits_XXXX.npy` — CTC logits per sample
- `token_norms_XXXX.npy` — Token norm values per sample
- `metadata.csv` — Predictions, ground truth, CER, WER per sample

---

### Phase E: Token Analysis (No GPU, Requires Phase D)

> **Prerequisite**: Run `extract_vit_rgts_features.py` first to generate the `vit_rgts_explain/` directory.

#### 8. `visualize_token_norms.py` — Token Norm Heatmaps

**Purpose**: Visualize token norm magnitudes as heatmaps overlaid on the input image.

**Architectures**: ViT-RGTS only (reads `vit_rgts_explain/`)

```bash
# Visualize first sample from run_54
python scripts/postprocessing/visualize_token_norms.py \
    saved_models/experiments/run_54/vit_rgts_explain

# Specific sample
python scripts/postprocessing/visualize_token_norms.py \
    saved_models/experiments/run_54/vit_rgts_explain --sample-idx 5

# Save output
python scripts/postprocessing/visualize_token_norms.py \
    saved_models/experiments/run_54/vit_rgts_explain \
    --save output/token_norms/run_54_sample0.png
```

**Input**: `vit_rgts_explain/` directory
**Output**: Token norm heatmap (displayed or saved)

---

#### 9. `visualize_register_attention.py` — Register→Patch Similarity

**Purpose**: Create heatmaps showing cosine/dot-product similarity between each register token and every patch token.

**Architectures**: ViT-RGTS only (R>0, reads `vit_rgts_explain/`)

```bash
# Default (cosine similarity, first sample)
python scripts/postprocessing/visualize_register_attention.py \
    saved_models/experiments/run_54/vit_rgts_explain

# Dot-product similarity, specific sample
python scripts/postprocessing/visualize_register_attention.py \
    saved_models/experiments/run_54/vit_rgts_explain \
    --sample-idx 3 --similarity dot

# Save output
python scripts/postprocessing/visualize_register_attention.py \
    saved_models/experiments/run_54/vit_rgts_explain \
    --save output/register_attention/run_54_sample0.png
```

**Input**: `vit_rgts_explain/` directory (must have register tokens)
**Output**: Per-register attention heatmap (displayed or saved)

---

#### 10. `tsne_register_tokens.py` — t-SNE of Register Embeddings

**Purpose**: Run t-SNE on mean-pooled register embeddings and plot colored by CER to see if register content correlates with prediction quality.

**Architectures**: ViT-RGTS only (R>0, reads `vit_rgts_explain/`)

```bash
# Default (up to 800 samples)
python scripts/postprocessing/tsne_register_tokens.py \
    saved_models/experiments/run_54/vit_rgts_explain

# Limit samples and save
python scripts/postprocessing/tsne_register_tokens.py \
    saved_models/experiments/run_54/vit_rgts_explain \
    --max-samples 500 --save output/tsne/run_54.png
```

**Input**: `vit_rgts_explain/` directory (must have register tokens)
**Output**: t-SNE scatter plot (displayed or saved)

---

### Phase F: Attention Visualization (GPU Recommended)

#### 11. `gradcam_vit_rgts.py` — Grad-CAM Patch Importance

**Purpose**: Generate Grad-CAM style heatmaps showing which patches contribute most to the prediction.

**Architectures**: ViT-RGTS only (accesses backbone patch embeddings)

```bash
# First test sample from run_54
python scripts/postprocessing/gradcam_vit_rgts.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt

# Specific sample
python scripts/postprocessing/gradcam_vit_rgts.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt --img-idx 5

# Save output
python scripts/postprocessing/gradcam_vit_rgts.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt \
    --save output/gradcam/run_54_sample0.png
```

**Input**: Config YAML(s) + `model.pt` + test dataset
**Output**: Grad-CAM overlay image (displayed or saved)

---

#### 12. `visualize_htr_register_attention.py` — Multi-Run Attention Comparison

**Purpose**: Compare attention patterns across multiple runs with different register counts. Produces paper-style side-by-side visualizations.

**Architectures**: ViT-RGTS only (multi-run comparison)

```bash
# Compare all 5 ViT-RGTS v2 runs
python scripts/postprocessing/visualize_htr_register_attention.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --runs run_50 run_51 run_52 run_53 run_54 \
    --register-counts 0 2 4 8 16 \
    --output-dir visualizations/register_analysis

# Compare two configurations
python scripts/postprocessing/visualize_htr_register_attention.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --runs run_50 run_54 \
    --register-counts 0 16 \
    --layer-type last

# Middle layer attention
python scripts/postprocessing/visualize_htr_register_attention.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --runs run_53 --register-counts 8 \
    --layer-type middle
```

**Input**: Image file + run directories (config.json + model.pt)
**Output**: Multi-panel comparison figure in `visualizations/register_analysis/`

---

#### 13. `visualize_character_vit_updated.py` — Per-Character Attention

**Purpose**: Visualize which image regions the model focuses on for each predicted character. Uses real attention extraction via forward hooks.

**Architectures**: ViT-RGTS (`--model-type htr`), TorchVision ViT (`--model-type vit`)

```bash
# HTR model (vit_rgts) - single image
python scripts/postprocessing/visualize_character_vit_updated.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --model-path saved_models/experiments/run_54/model.pt \
    --model-type htr --attention-layer last

# Multi-layer analysis
python scripts/postprocessing/visualize_character_vit_updated.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --model-path saved_models/experiments/run_54/model.pt \
    --model-type htr --multi-layer --output output/char_attention/

# TorchVision ViT pretrained model
python scripts/postprocessing/visualize_character_vit_updated.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --model-path asset/pretrained_models/torchvision/vit_b_16.pth \
    --model-type vit --text "sample text"

# Batch mode
python scripts/postprocessing/visualize_character_vit_updated.py \
    --batch --batch-images img1.png img2.png img3.png \
    --model-path saved_models/experiments/run_54/model.pt \
    --model-type htr --output output/char_attention/

# With gradient saliency
python scripts/postprocessing/visualize_character_vit_updated.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --model-path saved_models/experiments/run_54/model.pt \
    --model-type htr --show-saliency
```

**Input**: Image file + model checkpoint
**Output**: Character attention heatmaps (displayed or saved)

---

## Deprecated Scripts

Three scripts were moved to `scripts/postprocessing/_deprecated/` as they are superseded:

| Deprecated Script | Reason | Replacement |
|-------------------|--------|-------------|
| `visualize_character_vit.py` | Superseded by updated version | `visualize_character_vit_updated.py` |
| `visualize_character_gradcam.py` | Only works with torchvision ViT-B/16, not HTR models | `gradcam_vit_rgts.py` |
| `batch_process_samples.py` | Hardcoded wrapper for old run_23, calls deprecated script | `visualize_character_vit_updated.py --batch` |

---

## Architecture-Specific Workflow Examples

### CNN-RNN (run_32)

CNN-RNN has no transformer components, so only results analysis and evaluation scripts apply:

```bash
# 1. Plot training curves
python scripts/postprocessing/plot_training_metrics.py \
    --model-path saved_models/experiments/run_32/

# 2. Re-evaluate
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml \
    resume=saved_models/experiments/run_32/model.pt

# 3. Analyze errors
python scripts/postprocessing/analyze_evaluation.py \
    saved_models/experiments/run_32/evaluation_details.csv
```

### ViT-RGTS v2 (runs 50–54) — Full Pipeline

```bash
# Phase A: Results analysis
python scripts/postprocessing/plot_training_metrics.py \
    --model-path saved_models/experiments/run_50/ \
    --model-path saved_models/experiments/run_51/ \
    --model-path saved_models/experiments/run_52/ \
    --model-path saved_models/experiments/run_53/ \
    --model-path saved_models/experiments/run_54/

python scripts/postprocessing/analyze_register_impact.py \
    --model-path saved_models/experiments/run_50 --registers 0 \
    --model-path saved_models/experiments/run_51 --registers 2 \
    --model-path saved_models/experiments/run_52 --registers 4 \
    --model-path saved_models/experiments/run_53 --registers 8 \
    --model-path saved_models/experiments/run_54 --registers 16

# Phase B: Evaluate best model
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt

python scripts/postprocessing/analyze_evaluation.py \
    saved_models/experiments/run_54/evaluation_details.csv

# Phase D: Extract features (for run with registers)
python scripts/postprocessing/extract_vit_rgts_features.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt

# Phase E: Token analysis (requires Phase D)
python scripts/postprocessing/visualize_token_norms.py \
    saved_models/experiments/run_54/vit_rgts_explain

python scripts/postprocessing/visualize_register_attention.py \
    saved_models/experiments/run_54/vit_rgts_explain

python scripts/postprocessing/tsne_register_tokens.py \
    saved_models/experiments/run_54/vit_rgts_explain

# Phase F: Attention visualization
python scripts/postprocessing/gradcam_vit_rgts.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    resume=saved_models/experiments/run_54/model.pt

python scripts/postprocessing/visualize_htr_register_attention.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --runs run_50 run_51 run_52 run_53 run_54 \
    --register-counts 0 2 4 8 16

python scripts/postprocessing/visualize_character_vit_updated.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --model-path saved_models/experiments/run_54/model.pt \
    --model-type htr
```

### TorchVision ViT (run_39)

```bash
# 1. Plot training curves
python scripts/postprocessing/plot_training_metrics.py \
    --model-path saved_models/experiments/run_39/

# 2. Evaluate
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml configs/torchvision_vit.yaml \
    resume=saved_models/experiments/run_39/model.pt

# 3. Analyze errors
python scripts/postprocessing/analyze_evaluation.py \
    saved_models/experiments/run_39/evaluation_details.csv

# 4. Character attention (uses model-type vit)
python scripts/postprocessing/visualize_character_vit_updated.py \
    --image-path notebook/sample_images/a01-000u-00.png \
    --model-path saved_models/experiments/run_39/model.pt \
    --model-type vit
```

### TrOCR (run_40)

```bash
# 1. Plot training curves
python scripts/postprocessing/plot_training_metrics.py \
    --model-path saved_models/experiments/run_40/

# 2. Evaluate
python scripts/postprocessing/evaluate.py \
    configs/config.yaml configs/baseline.yaml configs/trocr.yaml \
    resume=saved_models/experiments/run_40/model.pt

# 3. Analyze errors
python scripts/postprocessing/analyze_evaluation.py \
    saved_models/experiments/run_40/evaluation_details.csv
```

> **Note**: TrOCR uses HuggingFace's TrOCRProcessor for decoding — the attention visualization scripts are not directly compatible with TrOCR's encoder-decoder architecture.

---

## Script Summary Table

| # | Script | Purpose | Single/Multi | GPU |
|:-:|--------|---------|:------------:|:---:|
| 1 | `plot_training_metrics.py` | Training curves (loss/CER/WER/LR) | Both | No |
| 2 | `analyze_register_impact.py` | Register config comparison (bar charts) | Multi-run | No |
| 3 | `analyze_evaluation.py` | Per-sample error analysis | Single | No |
| 4 | `evaluate.py` | Re-run CER/WER evaluation | Single | Yes |
| 5 | `demo.py` | Quick single-image prediction | Single | Yes |
| 6 | `demo_explainability.py` | Prediction + attention + registers | Single | Yes |
| 7 | `extract_vit_rgts_features.py` | Extract tokens/features for test set | Single | Yes |
| 8 | `visualize_token_norms.py` | Token norm heatmaps | Single sample | No |
| 9 | `visualize_register_attention.py` | Register→patch similarity maps | Single sample | No |
| 10 | `tsne_register_tokens.py` | t-SNE of register embeddings | Single run | No |
| 11 | `gradcam_vit_rgts.py` | Grad-CAM patch importance | Single sample | Yes |
| 12 | `visualize_htr_register_attention.py` | Multi-run attention comparison | Multi-run | Yes |
| 13 | `visualize_character_vit_updated.py` | Per-character attention maps | Single/Batch | Yes |

---

## Output Directories

| Script | Default Output |
|--------|---------------|
| `plot_training_metrics.py` | `./output/training_plots/` |
| `analyze_register_impact.py` | `./visualizations/register_analysis/` |
| `extract_vit_rgts_features.py` | `<run_dir>/vit_rgts_explain/` |
| `visualize_htr_register_attention.py` | `visualizations/register_analysis/` |
| Others | Displayed or specified via `--save`/`--output` |
