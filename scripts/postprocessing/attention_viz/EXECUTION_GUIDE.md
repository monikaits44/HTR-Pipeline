# Attention Visualization System — Step-by-Step Execution Guide

**Location**: `scripts/postprocessing/attention_viz/`  
**Date**: April 24, 2026  
**Purpose**: Generate all attention-related figures for the project report showing the impact of register tokens on HTR attention quality.

---

## Prerequisites

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate
```

**Required**: 17 trained ViT-RGTS v2 models (runs 55–71, register sweep 0–16), all at 80 epochs.  
**GPU**: Not required for visualization — all scripts run on CPU.  
**Time**: ~5 minutes per script on CPU (comprehensive_comparison loads 17 models sequentially).

### Model → Register Mapping

| Run | Registers | Run | Registers | Run | Registers |
|-----|-----------|-----|-----------|-----|-----------|
| 55  | 0         | 61  | 10        | 67  | 6         |
| 56  | 2         | 62  | 14        | 68  | 7         |
| 57  | 11        | 63  | 8         | 69  | 9         |
| 58  | 4         | 64  | 1         | 70  | 15        |
| 59  | 12        | 65  | 5         | 71  | 16        |
| 60  | 13        | 66  | 3         |     |           |

---

## Script 1: Comprehensive Comparison (KEY DELIVERABLE)

### Purpose
Produces a single multi-panel figure that tells the complete register-token story:
- **Panel A** (top-left): CER vs register count (0–16) — proves registers don't hurt recognition
- **Panel A** (top-right): WER vs register count — same message
- **Panel B** (mid-left): Diagonality (ρ) and Entropy vs register count — quantitative attention quality
- **Panel B** (mid-right): The input image for reference
- **Panel C** (wide): Fig.5-style character attention grid showing Reg-0, Reg-4, Reg-8, Reg-16 side-by-side
- **Panel D** (bottom): Register token behavior — what registers attend to (should absorb background)

### Command
```bash
python scripts/postprocessing/attention_viz/comprehensive_comparison.py \
  --image notebook/sample_images/a01-038-12.png \
  --runs-dir saved_models/experiments \
  --device cpu \
  --dpi 200 \
  --gamma 3.0 \
  --max-chars 12 \
  --save-dir visualizations/comprehensive_comparison
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Path to a single handwriting image |
| `--runs-dir` | `saved_models/experiments` | Directory containing run_XX folders |
| `--device` | `cpu` | `cpu` or `cuda:0` |
| `--dpi` | `200` | Output resolution |
| `--gamma` | `3.0` | Contrast exponent for heatmaps (>1 = sharper peaks) |
| `--max-chars` | `15` | Max characters shown in Panel C |
| `--save-dir` | `visualizations/comprehensive_comparison` | Output directory |

### Output
```
visualizations/comprehensive_comparison/comprehensive_<image_name>.png
```

### How to Interpret
- **Panel A**: CER/WER should be flat or slightly improving with registers (~4.1–4.3%). This proves registers don't hurt recognition.
- **Panel B**: Diagonality ρ should stay near 1.0 (left-to-right reading order). Entropy may change.
- **Panel C**: Compare rows — with more registers, character attention heatmaps should be tighter and more localized.
- **Panel D**: Register tokens should attend to non-text regions (borders, whitespace, padding) — they "absorb" background attention.

### Run for Multiple Images
```bash
for img in a01-038-12.png a06-110-08.png c04-110-00.png; do
  python scripts/postprocessing/attention_viz/comprehensive_comparison.py \
    --image notebook/sample_images/$img \
    --device cpu --dpi 200 --gamma 3.0 --max-chars 12
done
```

---

## Script 2: Attention Rollout Comparison

### Purpose
Computes **attention rollout** (Abnar & Zuidema, 2020) — multiplying attention matrices across all 6 layers to reveal the TRUE effective attention from input to output. This is more accurate than single-layer attention for understanding what the model actually uses.

Produces two columns per model:
- **Left**: Rollout heatmap overlaid on original image (saliency-style)
- **Right**: Per-character rollout carpet (characters × patches)

### Command
```bash
python scripts/postprocessing/attention_viz/rollout_comparison.py \
  --image notebook/sample_images/a01-038-12.png \
  --runs 55 58 63 71 \
  --runs-dir saved_models/experiments \
  --device cpu \
  --dpi 200 \
  --alpha 0.55 \
  --gamma 2.0 \
  --save-dir visualizations/rollout_comparison
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Path to a single handwriting image |
| `--runs` | `55 58 63 71` | Run numbers to compare (Reg-0, 4, 8, 16) |
| `--alpha` | `0.55` | Overlay opacity (0=transparent, 1=opaque) |
| `--gamma` | `2.0` | Contrast exponent |
| `--max-chars` | `20` | Max characters in carpet plot |

### Output
```
visualizations/rollout_comparison/rollout_<image_name>.png
```

### How to Interpret
- **Rollout overlay**: Bright regions = where the model's effective attention accumulates across all layers. Should highlight actual text strokes.
- **Character carpet**: Diagonal pattern = correct left-to-right reading. With more registers, diagonals should be sharper.
- Compare Reg-0 (top) vs Reg-16 (bottom) — rollout should be more focused on text with registers.

---

## Script 3: Head Specialization Analysis

### Purpose
Analyzes how individual attention heads specialize. Different heads may learn to focus on different aspects:
- Head 0: Left-context characters
- Head 1: Right-context characters  
- Head 2: Character structure (strokes)
- Head 3: Whitespace/punctuation

Produces:
1. **Per-head character attention grid**: Rows = 8 heads, columns = decoded characters
2. **Head diversity bar chart**: Entropy per head (lower = more focused/specialized)

### Command
```bash
python scripts/postprocessing/attention_viz/head_specialization.py \
  --image notebook/sample_images/a01-038-12.png \
  --runs 55 58 71 \
  --runs-dir saved_models/experiments \
  --device cpu \
  --dpi 200 \
  --gamma 2.0 \
  --max-chars 10 \
  --save-dir visualizations/head_specialization
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Input image |
| `--runs` | `55 58 71` | Runs to analyze (Reg-0, 4, 16) |
| `--gamma` | `2.0` | Contrast exponent |
| `--max-chars` | `12` | Max characters |

### Output (per model)
```
visualizations/head_specialization/heads_run55_reg0_<image>.png       # 8-row grid
visualizations/head_specialization/diversity_run55_reg0_<image>.png   # Bar chart
```

### How to Interpret
- **Head grid**: Each row is one attention head's character attention. Well-specialized heads have distinct patterns (some look left, some look local, some look right).
- **Diversity bar**: Lower entropy = more focused head. With registers, heads may become MORE specialized because registers absorb the "catch-all" global attention.
- Compare Reg-0 vs Reg-16: Look for heads that become more focused (lower entropy) with registers.

---

## Script 4: Layer-wise Attention Evolution

### Purpose
Shows how character attention evolves through the 6 transformer layers. Early layers have broad/diffuse attention; later layers should converge to focused character-aligned peaks.

Produces:
1. **Per-model layer grid**: Rows = 6 layers, columns = characters. Shows attention sharpening from Layer 1 to Layer 6.
2. **Progression metrics**: Diagonality (ρ) and Entropy plotted per layer.
3. **Cross-model comparison**: Overlay of diagonality/entropy curves for different register counts.

### Command
```bash
python scripts/postprocessing/attention_viz/layerwise_evolution.py \
  --image notebook/sample_images/a01-038-12.png \
  --runs 55 58 71 \
  --runs-dir saved_models/experiments \
  --device cpu \
  --dpi 200 \
  --gamma 2.0 \
  --max-chars 10 \
  --save-dir visualizations/layerwise_evolution
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Input image |
| `--runs` | `55 58 71` | Runs to analyze |
| `--gamma` | `2.0` | Contrast exponent |
| `--max-chars` | `15` | Max characters |

### Output
```
visualizations/layerwise_evolution/
  layers_run55_reg0_<image>.png           # 6-row layer grid
  progression_run55_reg0_<image>.png      # Diag+entropy vs layer
  layers_run58_reg4_<image>.png
  progression_run58_reg4_<image>.png
  layers_run71_reg16_<image>.png
  progression_run71_reg16_<image>.png
  cross_model_layers_<image>.png          # Cross-model comparison
```

### How to Interpret
- **Layer grid**: Layer 1 (top) should be diffuse, Layer 6 (bottom) should be sharp. The ρ and H values on the right quantify this.
- **Progression plot**: Diagonality should increase from ~0.5 → ~1.0 across layers; entropy should decrease.
- **Cross-model**: With registers, the attention should converge FASTER (higher diagonality at earlier layers).

---

## Script 5: Attention Overlay

### Purpose
Generates transparent heatmap overlays directly on handwriting images, showing spatial attention localization.

Two modes:
- **`sidebyside`**: For each model: (1) Original image, (2) Alignment carpet, (3) Mean attention overlay — all in one row.
- **`overlay`**: Per-character heatmaps overlaid on the original image — shows exactly where the model looks when decoding each character.

### Commands

**Side-by-side mode** (comparison across register counts):
```bash
python scripts/postprocessing/attention_viz/attention_overlay.py \
  --image notebook/sample_images/a01-038-12.png \
  --runs 55 58 71 \
  --mode sidebyside \
  --device cpu --dpi 200 \
  --save-dir visualizations/attention_overlay
```

**Overlay mode** (per-character saliency):
```bash
python scripts/postprocessing/attention_viz/attention_overlay.py \
  --image notebook/sample_images/a06-110-08.png \
  --runs 55 71 \
  --mode overlay \
  --device cpu --dpi 200 \
  --save-dir visualizations/attention_overlay
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `sidebyside` | `overlay` (per-char) or `sidebyside` (comparison) |
| `--alpha` | `0.55` | Heatmap opacity |
| `--gamma` | `2.5` | Contrast |

### Output
```
# sidebyside mode
visualizations/attention_overlay/sidebyside_<image>.png

# overlay mode (one per model)
visualizations/attention_overlay/overlay_run55_reg0_<image>.png
visualizations/attention_overlay/overlay_run71_reg16_<image>.png
```

### How to Interpret
- **Sidebyside**: Alignment carpet (middle column) should show a clean diagonal. Mean attention overlay (right) shows overall focus regions.
- **Overlay**: Each sub-image highlights WHERE the model looks when decoding that specific character. Good = bright spot over the correct character. Bad = spread across the whole image.

---

## Script 6: Attention Concentration Analysis (Full Quantitative Sweep)

### Purpose
The most comprehensive quantitative analysis. Runs ALL 17 register configurations (0–16) across ALL sample images and computes:
- **Diagonality** (Spearman ρ): How well attention follows left-to-right reading order
- **Entropy** (bits): How spread out the attention is (lower = more focused)
- **Peak Sharpness** (max/mean ratio): How peaked the attention is (higher = sharper)

Produces:
1. **Metrics vs Registers** (with error bars across images)
2. **CER vs Attention Quality** correlation scatter (labeled by register count)
3. **CSV with all raw data** for statistical analysis

### Command
```bash
python scripts/postprocessing/attention_viz/attention_concentration.py \
  --images-dir notebook/sample_images \
  --runs-dir saved_models/experiments \
  --device cpu \
  --dpi 200 \
  --max-images 10 \
  --save-dir visualizations/attention_concentration
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--images-dir` | `notebook/sample_images` | Directory of test images |
| `--images` | (none) | Override with specific image paths |
| `--max-images` | `10` | Max images to analyze |

### Output
```
visualizations/attention_concentration/
  metrics_vs_registers.png           # 3 subplots: diag, entropy, sharpness vs #regs
  cer_vs_attention.png               # 2 subplots: CER vs diag, CER vs entropy
  attention_metrics_full.csv         # Raw data (136 rows: 17 models × 8 images)
```

### CSV Format
```csv
n_reg,run,image,diagonality,entropy,sharpness,cer,text
0,55,a01-038-12.png,1.0,4.84,20.25,0.0606," talks. "
4,58,a01-038-12.png,0.99,4.62,22.10,0.0412," talks. "
...
```

### How to Interpret
- **Metrics vs Registers**: 
  - Diagonality should stay near 1.0 (all models decode left-to-right)
  - Entropy may decrease with more registers (attention becomes more focused)
  - Sharpness may increase with registers (attention peaks become more pronounced)
  - Error bars show variance across images
- **CER vs Quality**: Points labeled by register count. Ideally, models with better attention quality also have lower CER. Even if CER is similar, attention quality differences matter for downstream writer identification.

---

## Script 7: Generate All Figures at Once

### Purpose
Batch runner that invokes all 6 scripts above with sensible defaults. Useful for regenerating all figures.

### Command
```bash
python scripts/postprocessing/attention_viz/generate_all.py \
  --device cpu \
  --dpi 200
```

This will process 3 sample images and generate ~20 figures total.

---

## Output Summary

After running all scripts, the following directories contain outputs:

```
visualizations/
├── comprehensive_comparison/        # 1-3 PNGs — KEY DELIVERABLE
│   └── comprehensive_<image>.png
├── rollout_comparison/              # 1-3 PNGs — Effective attention
│   └── rollout_<image>.png
├── head_specialization/             # 6 PNGs — Per-head analysis
│   ├── heads_run*_reg*_<image>.png
│   └── diversity_run*_reg*_<image>.png
├── layerwise_evolution/             # 7 PNGs — Layer progression
│   ├── layers_run*_reg*_<image>.png
│   ├── progression_run*_reg*_<image>.png
│   └── cross_model_layers_<image>.png
├── attention_overlay/               # 3+ PNGs — Saliency overlays
│   ├── sidebyside_<image>.png
│   └── overlay_run*_reg*_<image>.png
└── attention_concentration/         # 2 PNGs + 1 CSV — Full sweep
    ├── metrics_vs_registers.png
    ├── cer_vs_attention.png
    └── attention_metrics_full.csv
```

---

## Figures for Report (Recommended Selection)

| Report Figure | Script | File |
|---------------|--------|------|
| Fig. 1: Register ablation (performance) | `comprehensive_comparison.py` | Panel A of `comprehensive_*.png` |
| Fig. 2: Character attention grid (Fig.5 style) | `comprehensive_comparison.py` | Panel C of `comprehensive_*.png` |
| Fig. 3: Register token behavior | `comprehensive_comparison.py` | Panel D of `comprehensive_*.png` |
| Fig. 4: Attention quality vs registers | `attention_concentration.py` | `metrics_vs_registers.png` |
| Fig. 5: CER vs attention quality | `attention_concentration.py` | `cer_vs_attention.png` |
| Fig. 6: Layer-wise evolution | `layerwise_evolution.py` | `cross_model_layers_*.png` |
| Fig. 7: Head specialization | `head_specialization.py` | `heads_run*_reg*.png` |
| Fig. 8: Attention rollout | `rollout_comparison.py` | `rollout_*.png` |
| Fig. 9: Saliency overlay | `attention_overlay.py` | `overlay_run*_reg*.png` |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: classes.npy` | Missing character class file | Run `ls data/IAM/processed_lines/classes.npy` — if missing, run training once |
| `RuntimeError: size mismatch` | Wrong `num_registers` for model | The script auto-sets registers from the run map — verify RUN_MAP matches your runs |
| Heatmaps look flat/uniform | Gamma too low | Increase `--gamma` to 3.0–5.0 |
| Heatmaps look binary (all 0 or 1) | Gamma too high | Decrease `--gamma` to 1.0–2.0 |
| Script takes >10 min | Loading many large models | Use `--device cuda:0` if GPU available, or reduce `--max-images` |
| `ModuleNotFoundError` | Wrong working directory | Must run from project root: `cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline` |

---

## Architecture Reference

### Where Attention Comes From

```
Image [1,1,128,1024]
  → CNN Stem (4 conv layers) → [1, 256, 1, 128]
  → AdaptiveMaxPool → 128 patch tokens [1, 128, 256]
  → Prepend R register tokens → [1, R+128, 256]
  → 6-layer TransformerEncoder (8 heads each)
      Layer i: Q·K^T/√d → softmax → attn_weights [1, 8, R+128, R+128]
                                      ↑ THIS is extracted per layer
  → Split: registers [1, R, 256] + patches [1, 128, 256]
  → CTCtopB (BiLSTM) → logits [128, 1, nclasses]
  → Greedy CTC decode → characters + positions
```

### How Per-Character Attention Is Extracted

For decoded character at CTC position `t`:
```
char_attention = attn_maps[layer][0].mean(dim=0)  # Average over 8 heads → [S, S]
char_attention = char_attention[R + t, R:]          # Row t (skip registers), patches only → [128]
char_attention = normalize(char_attention)           # Min-max → [0, 1]
char_attention = char_attention ** gamma             # Contrast adjustment
```

This gives a 128-element vector showing spatial attention for each character.
