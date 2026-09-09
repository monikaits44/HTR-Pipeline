# Register Attention Comparison — Character-Level Analysis

## Script

```
scripts/register_attention_comparison.py
```

## Output Location

```
/home/woody/iwi5/iwi5369h/projects/attn_register_comparison/
```

All output goes to **woody** (not hpchome) due to disk quota constraints.

---

## Purpose

Generates publication-quality figures comparing **character-level self-attention**
across register token configurations (0, 4, 8, 16 registers), following the style
of "Beyond Memorization" Figure 5 as requested by Prof. Christlein.

The script answers:
- How do register tokens affect per-character attention focus?
- Do registers absorb noisy/artifact attention, producing cleaner spatial maps?
- How does attention entropy, sharpness, and register absorption vary per character?

---

## Models Used

| Registers | Run ID   | Config   |
|-----------|----------|----------|
| 0         | run_76   | ViT-RGTS v2 baseline |
| 4         | run_84   | ViT-RGTS v2          |
| 8         | run_63   | ViT-RGTS v2          |
| 16        | run_71   | ViT-RGTS v2          |

All from `saved_models/experiments/`.

---

## Figures Generated (per image)

### Figure 1 — Character Attention Overlay Grid (`*_attention_grid.png/.pdf`)

- **Layout**: Rows = register counts (0, 4, 8, 16), Columns = input image + per-character attention
- **Content**: Cropped text image with semi-transparent attention heatmap overlay (inferno colormap)
- **Markers**: Dashed white vertical line at Xmax_c (weighted centroid of attention)
- **Follows**: "Beyond Memorization" Fig. 5 style — character-based spatial attention maps

### Figure 2 — 1D Attention Profiles (`*_attention_profiles.png`)

- **Layout**: One subplot per character
- **Content**: 4 overlaid curves (one per register count) showing the raw 1D attention distribution
- **Purpose**: Direct quantitative comparison of attention shape, spread, and peak location
- **Color coding**: Blue (0 reg), Green (4 reg), Orange (8 reg), Red (16 reg)

### Figure 3 — Attention Metrics Bar Chart (`*_attention_metrics.png`)

Three sub-panels:
- **(a) Entropy** — Shannon entropy in bits (lower = more focused attention)
- **(b) Sharpness** — Peak-to-mean ratio (higher = sharper peak)
- **(c) Register Absorption** — % of total attention directed to register tokens

---

## Attention Extraction Method

For each decoded character `c` at CTC timestep `t_c`:

1. Run `model.forward_explain(image)` → get per-layer, per-head attention `[B, H, S, S]`
2. Average over heads in the **last layer** → `[B, S, S]`
3. Extract row `A[0, R + t_c, R : R + Wp]` — attention FROM character timestep TO patch tokens
4. `R` = number of register tokens, `Wp` = 128 (patch columns from CNN stem)
5. Normalize via percentile clipping (p2/p98) + Gaussian smoothing

`Xmax_c` = weighted centroid: $X^{(c)}_{max} = \frac{\sum_i i \cdot A_c[i]}{\sum_i A_c[i]}$

Register absorption = $\sum_{r=0}^{R-1} A[0, R+t_c, r]$ (fraction of attention to register tokens)

---

## Usage

```bash
# Auto-discover short words from IAM test set
python scripts/register_attention_comparison.py

# Use sample image ("talks.")
python scripts/register_attention_comparison.py --use_samples

# Specific image(s)
python scripts/register_attention_comparison.py --images data/IAM/processed_lines/test/d04-086-06.png

# Custom max characters (default: 5)
python scripts/register_attention_comparison.py --max_chars 7

# GPU
python scripts/register_attention_comparison.py --device cuda:0

# Custom output directory
python scripts/register_attention_comparison.py --output /path/to/output

# Custom run IDs (order: 0-reg 4-reg 8-reg 16-reg)
python scripts/register_attention_comparison.py --runs run_76 run_84 run_63 run_71
```

---

## Current Results

Output files in `/home/woody/iwi5/iwi5369h/projects/attn_register_comparison/`:

| Image | Ground Truth | Notes |
|-------|-------------|-------|
| a01-038-12 | "talks." | All 4 models agree on "talks" — clean comparison |
| c04-150-07 | "vision." | 16-reg decodes "vislo" (minor error), rest decode "visto" |
| d06-104-10 | "3 p.m.." | Models diverge — interesting edge case |
| m02-087-07 | "of law." | All agree on "of la" (5-char cap) |

### Key Findings

| Metric | 0 reg | 4 reg | 8 reg | 16 reg |
|--------|-------|-------|-------|--------|
| Avg Entropy (bits) | 5.4–6.5 | 4.8–5.6 | 4.8–5.6 | 4.9–5.5 |
| Avg Sharpness | 8–21 | 12–27 | 12–26 | 14–20 |
| Avg Reg Absorption | 0.0% | 2.2–2.5% | 5.0–6.3% | 8.4–11.7% |

- Register absorption increases monotonically with register count
- 4 registers shows the best entropy/sharpness balance
- Registers act as attention sinks, absorbing diffuse/noisy attention

---

## Relationship to Other Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/attention_visualization.py` | 5 publication figures (register grid, char attention, norms, layer flow, GradCAM) | Existing — DO NOT MODIFY |
| `scripts/register_attention_comparison.py` | Focused character-level register comparison (this script) | **New** |
| `utils/attention_extractor.py` | AttentionExtractor class (hook-based) | Utility |
| `utils/visualizer.py` | CNN feature visualization | CNN-RNN only |
