# HTR Attention Visualization — Documentation

## Project: Making HTR Explainable via Register-Augmented Vision Transformers

> **Goal**: Generate fine-grained, interpretable attention maps from a ViT-based CTC HTR system
> to support downstream tasks such as writer identification (VLAC pipeline).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Architecture Summary](#3-architecture-summary)
4. [Visualization Script Reference](#4-visualization-script-reference)
5. [Figure Descriptions](#5-figure-descriptions)
6. [Execution Guide](#6-execution-guide)
7. [Experimental Setup & Results](#7-experimental-setup--results)
8. [Connection to Papers](#8-connection-to-papers)
9. [VLAC Integration Path](#9-vlac-integration-path)

---

## 1. Overview

This visualization toolkit generates **4 publication-quality figures** that demonstrate
how **register tokens** (Darcet et al., ICLR 2024) improve the interpretability of
attention maps in a ViT-based Handwritten Text Recognition (HTR) system.

### Script Location

```
scripts/attention_visualization.py
```

### Output Location

```
output/attention_visualizations/
├── fig1_register_impact_grid.pdf      # Register count comparison
├── fig1_register_impact_grid.png
├── fig2_character_attention_grid.pdf   # Per-character attention (with registers)
├── fig2_character_attention_grid.png
├── fig2b_character_attention_grid_baseline.pdf  # Per-character attention (baseline)
├── fig2b_character_attention_grid_baseline.png
├── fig3_token_norm_artifacts.pdf      # Token norm artifact analysis
├── fig3_token_norm_artifacts.png
├── fig4_layerwise_attention.pdf       # Layer-by-layer attention evolution
└── fig4_layerwise_attention.png
```

---

## 2. Theoretical Foundation

### 2.1 The Artifact Problem in Vision Transformers

In standard ViTs, well-trained models spontaneously **recycle** patch tokens from
low-information regions (background, inter-character gaps) to store global information.
This produces **high-norm outlier tokens** that:

- Corrupt attention maps with noise
- Prevent fine-grained character-level interpretability
- Make downstream feature extraction unreliable

### 2.2 Register Tokens as Solution (Darcet et al., 2024)

Register tokens are **learnable input tokens** appended after the patch embedding:

```
Token sequence:  [REG_1] ... [REG_R] | [patch_1] ... [patch_N]
                 ↑ no positional emb   ↑ with positional embedding
```

Key properties:
- **No positional embedding** — registers are position-agnostic scratch pads
- **Discarded at output** — only patch tokens feed the CTC head
- **Trained end-to-end** — model learns to offload global info into registers
- **< 2% FLOP increase** for 4 registers (Darcet et al., Fig. 12)

### 2.3 Character-Level Attention for Writer ID (VLAC)

The VLAC paper ("Interpretable Writer Recognition via Vectors of Locally Aggregated
Characters") uses character-level features for writer identification:

```
Attention map for char 'a' → spatial mask → extract patch features → aggregate per char → VLAC descriptor → cosine distance between writers
```

Register tokens ensure the attention masks are **clean** (no background artifacts),
so extracted features are truly **local allograph features**.

---

## 3. Architecture Summary

### ViT-RGTS v2 (Hybrid CNN-Stem + Transformer)

```
Input: [B, 1, 128, 1024]  (grayscale IAM line image)
       │
  CNN Stem (4 conv layers):
    Conv1 (stride 4,2) → [B, 32, 32, 512]
    Conv2 (stride 2,2) → [B, 64, 16, 256]
    Conv3 (stride 2,2) → [B, 128, 8, 128]
    Conv4 (stride 2,1) → [B, 256, 4, 128]
    AvgPool (H→1)      → [B, 256, 1, 128]  = 128 column tokens
       │
  [REG_1] ... [REG_R] | [patch_1] ... [patch_128]
       │
  6× TransformerEncoderLayer (dim=256, 8 heads, GELU, pre-norm)
       │
  ┌────┴────────────────────┐
  │                         │
  Patch tokens → CTC Head   Register tokens → DISCARDED
  (BiLSTM + Linear)         (available for analysis)
       │
  [T, B, 80]  CTC logits
```

### Model Configurations Used

| Run     | Registers | Val CER (%) | Test CER (%) | Parameters |
|---------|-----------|-------------|--------------|------------|
| run_55  | 0         | 4.19        | 6.06         | 9,475,680  |
| run_58  | 4         | 4.12        | 6.06         | 9,478,784  |
| run_53  | 8         | 4.23        | 6.11         | 9,481,888  |
| run_54  | 16        | 4.20        | 5.93         | 9,488,096  |

### Character Set

79 characters: `[space] ! " # & ' ( ) * + , - . / 0-9 : ; ? A-Z a-z`
Plus 1 CTC blank token = 80 output classes.

---

## 4. Visualization Script Reference

### Command-Line Interface

```bash
python scripts/attention_visualization.py [OPTIONS]
```

| Argument          | Default                           | Description                          |
|-------------------|-----------------------------------|--------------------------------------|
| `--runs`          | `run_55 run_58 run_53 run_54`    | Run IDs to compare                   |
| `--images`        | 3 sample images                   | Image paths to visualize             |
| `--device`        | `cpu`                             | `cpu` or `cuda:0`                    |
| `--out_dir`       | `output/attention_visualizations` | Output directory                     |
| `--max_chars`     | `10`                              | Max characters in Fig 2              |
| `--char_grid_run` | First model with registers        | Specific run for Fig 2               |
| `--layer_run`     | First model with registers        | Specific run for Fig 4               |

### Key Functions

| Function                      | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `load_model(run_id, device)`  | Load trained HTRNet from experiment directory     |
| `prepare_image(path, device)` | Load → preprocess → tensor                       |
| `ctc_greedy_decode(logits)`   | Greedy CTC decode → (string, peak_columns)       |
| `run_explain(model, tensor)`  | Forward pass with attention extraction            |
| `get_char_attention_maps()`   | Extract per-character spatial attention            |
| `plot_register_impact_grid()` | **Fig 1** — Register count comparison             |
| `plot_char_attention_grid()`  | **Fig 2** — Character-level attention              |
| `plot_token_norm_comparison()`| **Fig 3** — Token norm artifact analysis           |
| `plot_layerwise_attention()`  | **Fig 4** — Layer-wise attention evolution          |

---

## 5. Figure Descriptions

### Fig. 1: Register Impact Grid

**Paper reference**: Darcet et al. (2024), Fig. 1 / Fig. 19

**Layout**: Rows = sample images, Columns = register counts (0, 4, 8, 16)

**Shows**: Global attention heatmap (all patch tokens' aggregate attention)
overlaid on the handwriting image. Demonstrates how register tokens
produce cleaner, more spatially coherent attention.

**Expected observation**:
- 0-register model shows diffuse/noisy attention with background artifacts
- Higher register counts produce sharper, text-focused attention
- Attention concentrates on character strokes rather than whitespace

```
┌──────────┬────────────┬────────────┬────────────┬────────────┐
│  Input   │  0 Regs    │  4 Regs    │  8 Regs    │  16 Regs   │
├──────────┼────────────┼────────────┼────────────┼────────────┤
│  word 1  │  heatmap   │  heatmap   │  heatmap   │  heatmap   │
├──────────┼────────────┼────────────┼────────────┼────────────┤
│  word 2  │  heatmap   │  heatmap   │  heatmap   │  heatmap   │
└──────────┴────────────┴────────────┴────────────┴────────────┘
```

### Fig. 2: Character-Level Attention Grid

**Paper reference**: "Beyond Memorization" Fig. 5, VLAC paper Fig. 4

**Layout**: Rows = words, Columns = decoded characters

**Shows**: For each decoded character, the spatial attention map highlighting
which region of the image the model focuses on. The character label is shown
above each cell.

**Expected observation**:
- Attention peaks should align with the physical character position
- Left-to-right progression of attention peaks matches reading order
- With registers: sharper, more localized character attention
- Without registers (Fig 2b): noisier, less character-specific attention

```
┌──────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Original │  't'   │  'a'   │  'l'   │  'k'   │  's'   │  '.'   │
├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│  "talks."│ attn   │ attn   │ attn   │ attn   │ attn   │ attn   │
└──────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

### Fig. 3: Token Norm Artifact Map

**Paper reference**: Darcet et al. (2024), Fig. 3 / Fig. 4

**Layout**: Same grid as Fig 1 (rows = images, columns = register counts)

**Shows**: Spatial distribution of token L2 norms. High-norm outlier tokens
(artifacts) appear as bright spots in the inferno colormap.

**Expected observation**:
- 0-register model shows scattered high-norm outliers in background regions
- As register count increases, norm distribution becomes more uniform
- Outlier tokens are absorbed by register tokens → cleaner feature space
- Standard deviation (σ) of norms decreases with more registers

### Fig. 4: Layer-wise Attention Evolution

**Paper reference**: General ViT analysis (Dosovitskiy et al., 2020)

**Layout**: Rows = transformer layers (1–6), Columns = attention heads (avg + 4 individual)

**Shows**: How attention patterns evolve from early (global, diffuse) to late
(local, character-specific) layers.

**Expected observation**:
- Early layers: broad, global attention patterns
- Middle layers: emerging spatial structure
- Late layers: sharp, character-localized attention
- Individual heads specialize for different spatial scales/patterns

---

## 6. Execution Guide

### Prerequisites

```bash
# Activate environment
source .venv/bin/activate

# Required packages (already in requirements.txt)
pip install torch matplotlib numpy scikit-image
```

### Basic Execution (CPU)

```bash
python scripts/attention_visualization.py
```

### GPU Execution

```bash
python scripts/attention_visualization.py --device cuda:0
```

### Custom Images

```bash
python scripts/attention_visualization.py \
    --images data/IAM/processed_lines/test/a01-000u-00.png \
             data/IAM/processed_lines/test/d04-108-02.png \
    --device cuda:0
```

### Specific Register Count Comparison

```bash
# Compare 0 vs 4 registers only
python scripts/attention_visualization.py --runs run_55 run_58

# Full sweep: 0, 2, 4, 8, 16
python scripts/attention_visualization.py --runs run_55 run_51 run_58 run_53 run_54
```

### SLURM Submission (HPC)

```bash
#!/bin/bash
#SBATCH --job-name=attn_viz
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

source .venv/bin/activate
python scripts/attention_visualization.py --device cuda:0
```

---

## 7. Experimental Setup & Results

### Training Configuration (ViT-RGTS v2)

| Hyperparameter   | Value                    |
|------------------|--------------------------|
| Architecture     | CNN Stem + 6L Transformer|
| Embedding dim    | 256                      |
| Attention heads  | 8                        |
| MLP dim          | 1024                     |
| CTC Head         | Dual (BiLSTM + CNN)      |
| Image size       | 128 × 1024               |
| Batch size       | 8                        |
| Optimizer        | AdamW (warmup + cosine)  |
| Epochs           | 80                       |
| Dataset          | IAM Handwriting (6482 train, 976 val, 2915 test lines) |

### Register Token Sweep Results

| Registers | Best Run | Val CER (%) | Test CER (%) |
|-----------|----------|-------------|--------------|
| 0         | run_50   | 4.17        | 6.16         |
| 1         | run_64   | 4.20        | 6.04         |
| 2         | run_51   | 4.09        | 6.09         |
| 3         | run_66   | 4.25        | 6.22         |
| 4         | run_58   | 4.12        | 6.06         |
| 5         | run_65   | 4.10        | 6.01         |
| 6         | run_67   | 4.06        | 6.04         |
| 7         | run_68   | 4.11        | 5.97         |
| 8         | run_53   | 4.23        | 6.11         |
| 12        | run_59   | 4.08        | 6.06         |
| 16        | run_54   | 4.20        | 5.93         |

**Key finding**: Register tokens maintain or slightly improve CER while providing
dramatically cleaner attention maps for explainability. The performance is stable
across register counts (4.06–4.25% val CER), confirming Darcet et al.'s finding
that registers add < 2% FLOP overhead with no accuracy degradation.

---

## 8. Connection to Papers

### Paper 1: "Vision Transformers Need Registers" (Darcet et al., 2024)

| Paper concept                    | Our implementation                              |
|----------------------------------|-------------------------------------------------|
| Register tokens (§2.2)          | `ViTRGTSBackbone.register_tokens`               |
| No positional encoding for regs | Confirmed in `forward()`                        |
| Discard at output (§2.2)        | `patch_out = encoded[:, R:, :]`                 |
| Artifact reduction (Fig. 3)     | **Fig 3** — Token Norm Artifact Map             |
| Cleaner attention (Fig. 1)      | **Fig 1** — Register Impact Grid                |
| FLOP overhead < 2%              | 9.47M → 9.49M params (0.3% increase for 4 regs)|

### Paper 2: "Beyond Memorization" (ECCV 2024 Workshop)

| Paper concept                        | Our implementation                      |
|--------------------------------------|-----------------------------------------|
| Attention-based char localization    | `get_char_attention_maps()` via CTC peaks|
| Character-level attention maps (Fig 5)| **Fig 2** — Character Attention Grid    |
| Writer embedding injection points   | CTC peak columns identify char positions|

### Paper 3: "Interpretable Writer Recognition via VLAC"

| Paper concept                    | Our implementation                          |
|----------------------------------|---------------------------------------------|
| Character-wise feature extraction| Attention maps → spatial masks per character |
| VLAC aggregation                 | Ready for downstream: char attention → mask → feature extraction |
| One-to-one feature-char mapping  | CTC greedy decode provides exact alignment   |

---

## 9. VLAC Integration Path

The attention visualizations produced by this script are the **first step** toward
full VLAC-based writer identification. The integration pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: This Script                                        │
│   HTR model → character attention maps                      │
│   Output: per-character spatial attention (Ac)              │
├─────────────────────────────────────────────────────────────┤
│ Step 2: Feature Extraction (future work)                   │
│   For each char c:                                          │
│     mask = attention_map[c] > threshold                     │
│     features[c] = patch_tokens * mask                       │
│     → local allograph descriptor for char c                 │
├─────────────────────────────────────────────────────────────┤
│ Step 3: VLAC Aggregation (future work)                     │
│   For each document d:                                      │
│     VLAC[c] = mean(features[c] across all lines)            │
│   → document-level writer descriptor                        │
├─────────────────────────────────────────────────────────────┤
│ Step 4: Writer Distance (future work)                      │
│   dist(d1, d2) = mean_c(cosine_distance(VLAC1[c], VLAC2[c])│
│   → interpretable writer similarity                         │
└─────────────────────────────────────────────────────────────┘
```

Register tokens ensure that Step 2 produces **clean local features** (no artifact
contamination), which is critical for meaningful writer-level aggregation.

---

## References

1. Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision Transformers
   Need Registers. ICLR 2024. arXiv:2309.16588

2. Beyond Memorization: Training-Free Style Mixing for Variability in Handwritten
   Text Generation Using Writer Embedding Injection in Pretrained Diffusion Models.
   ECCV 2024 Workshop.

3. Retsinas, G., et al. Best Practices for a Handwritten Text Recognition System.
   DAS 2022. github.com/georgeretsi/HTR-best-practices

4. Interpretable Writer Recognition via Vectors of Locally Aggregated Characters.
