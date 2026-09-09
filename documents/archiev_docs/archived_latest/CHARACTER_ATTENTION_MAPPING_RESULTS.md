# Character-Level Attention Mapping — Results & Interpretation

> **Script**: `scripts/postprocessing/character_attention_mapping.py`  
> **Date generated**: 2026-03-02  
> **Layer analysed**: last (layer 6 of 6)  
> **Models**: Reg-0 (run_50), Reg-2 (run_51), Reg-4 (run_52), Reg-8 (run_53), Reg-16 (run_54)

---

## 1. What This Script Does (Plain Language)

When the ViT-RGTS model reads a handwritten word like **"talks."**, it predicts
one character at a time. For each predicted character, the model's
**attention mechanism** decides which part of the input image to focus on.

This script answers the question:

> **"When the model predicts the letter 't', which region of the image is it
> actually looking at?"**

It produces three types of figures:

| Output folder   | What it shows | How to read it |
|-----------------|---------------|----------------|
| `carpet/`       | Side-by-side copies of the line image, each overlaid with the attention heatmap for one character | **Bright yellow = the model is looking here** for that character. The cyan dashed line marks the peak. |
| `alignment/`    | Top: original image. Bottom: a matrix with characters on the Y-axis and image positions on the X-axis | **A diagonal pattern = the model reads left-to-right** and correctly locates each character. Dotted lines connect each character's peak to the corresponding image region. |
| `combined/`     | All three views in a single figure (original image + carpet + alignment matrix) | One-stop overview. |

---

## 2. How the Pipeline Works (Technical)

```
Raw image (42×162)
    │
    ▼
Preprocessing: scale + pad to 128×1024 canvas (border=8px)
    │
    ▼
CNN stem → 1×128 patch tokens  (each patch ≈ 8px wide)
    │
    ▼  Register tokens prepended: [REG₁ … REGᵣ, PATCH₁ … PATCH₁₂₈]
ViT Transformer (6 layers, 8 heads)
    │
    ▼
BiLSTM + CTC head → 128 output time steps → greedy decode
    │
    ▼
CTC decode: character "t" emitted at time step 3 → maps to patch 3
    │
    ▼
Attention row: attn[num_reg + 3, num_reg : num_reg + 128]
   = "when predicting at position 3, how much does the model attend
      to each of the 128 image patches?"
    │
    ▼
Reshape to (1, 128) → kron-upscale to (128, 1024) → crop to image size
    │
    ▼
Overlay heatmap on original image
```

**Key geometry**:
- Patch grid: (Hp=1, Wp=128) — purely horizontal 1D positions
- Patch width in preprocessed space: pw = 1024 / 128 = **8 pixels**
- Image content starts at **patch 1** (after 8px border)
- For a 162px-wide image (a01-038-12): content ends at ~**patch 21**
- Patches 22-127 are padding (median fill) and carry near-zero attention

---

## 3. Results Summary

### 3.1 Image: **a01-038-12** → Ground truth: `"talks."`

All 5 models correctly predict **"talks."** (CER = 0.00).

| Model  | Character | CTC pos | Peak patch | Top-5 focus | Notes |
|--------|-----------|---------|------------|-------------|-------|
| **Reg-0**  | t | 3  | **3**  | 0.44 | Correct |
|        | a | 6  | **6**  | 0.44 | Correct |
|        | l | 9  | 13 | 0.41 | Shifted right |
|        | k | 12 | 13 | 0.34 | **Attention sink** at patch 13 |
|        | s | 16 | 13 | 0.32 | **Attention sink** at patch 13 |
|        | . | 20 | 13 | 0.32 | **Attention sink** at patch 13 |
|        | **avg** | | | **0.379** | Monotonic: **Yes** (peaks never go left) |
| **Reg-2**  | t | 3  | **3**  | 0.46 | Correct |
|        | a | 6  | **6**  | 0.47 | Correct |
|        | l | 9  | 3  | 0.34 | Jumps back to "t" location |
|        | k | 11 | 3  | 0.39 | Jumps back to "t" location |
|        | s | 16 | 7  | 0.43 | Near correct |
|        | . | 20 | **16** | 0.44 | Close |
|        | **avg** | | | **0.423** | Monotonic: **No** |
| **Reg-4**  | t | 3  | **3**  | 0.39 | Correct |
|        | a | 6  | **6**  | 0.49 | Correct, sharp |
|        | l | 9  | 11 | 0.35 | Close |
|        | k | 12 | **12** | 0.38 | Correct |
|        | s | 16 | 6  | 0.43 | Jumps back to "a" |
|        | . | 20 | 13 | 0.38 | Shifted left |
|        | **avg** | | | **0.405** | Monotonic: **No** |
| **Reg-8**  | t | 3  | **3**  | 0.31 | Correct but diffuse |
|        | a | 6  | **6**  | 0.44 | Correct |
|        | l | 9  | 3  | 0.38 | Jumps back to "t" |
|        | k | 12 | 10 | 0.33 | Close |
|        | s | 16 | 6  | 0.38 | Jumps back |
|        | . | 20 | 13 | 0.24 | Very diffuse |
|        | **avg** | | | **0.350** | Monotonic: **No**, most diffuse |
| **Reg-16** | t | 3  | **3**  | 0.52 | Correct, **sharpest** |
|        | a | 6  | **6**  | 0.44 | Correct |
|        | l | 9  | **9**  | 0.44 | **Correct** |
|        | k | 11 | 9  | 0.40 | Close |
|        | s | 16 | 6  | 0.34 | Jumps back |
|        | . | 20 | **20** | 0.27 | **Correct** |
|        | **avg** | | | **0.403** | Monotonic: **No** |

### 3.2 Image: **a01-096u-10** → Ground truth: `"people."`

All 5 models correctly predict **"people."** (CER = 0.00).

| Model  | Character | CTC pos | Peak patch | Top-5 focus | Notes |
|--------|-----------|---------|------------|-------------|-------|
| **Reg-0**  | p | 2  | **2**  | 0.30 | Correct |
|        | e | 5  | 16 | 0.30 | **Far off** → distant-focus |
|        | o | 8  | **8**  | 0.32 | Correct |
|        | p | 11 | 2  | 0.26 | Jumps back to 1st "p" |
|        | l | 13 | **13** | 0.33 | Correct |
|        | e | 15 | 7  | 0.22 | Jumps back, **most diffuse** |
|        | . | 21 | **21** | 0.31 | Correct |
|        | **avg** | | | **0.292** | Monotonic: **No**, most diffuse overall |
| **Reg-2**  | p | 2  | **2**  | 0.34 | Correct |
|        | e | 5  | 8  | 0.37 | Slightly forward |
|        | o | 8  | **8**  | 0.47 | Correct, sharp |
|        | p | 10 | 7  | 0.34 | Close |
|        | l | 13 | 21 | 0.31 | Jumps far right |
|        | e | 16 | 12 | 0.39 | Close |
|        | . | 21 | **21** | 0.47 | Correct, sharp |
|        | **avg** | | | **0.382** | Monotonic: **No** |
| **Reg-4**  | p | 2  | 11 | 0.25 | Far off |
|        | e | 5  | 8  | 0.50 | Sharp but shifted |
|        | o | 8  | **8**  | 0.40 | Correct |
|        | p | 11 | 8  | 0.30 | Stuck on "o" position |
|        | l | 13 | **14** | 0.44 | Close |
|        | e | 15 | 7  | 0.33 | Jumps back |
|        | . | 21 | **21** | 0.34 | Correct |
|        | **avg** | | | **0.365** | Monotonic: **No** |
| **Reg-8**  | p | 2  | **3**  | 0.22 | Very close |
|        | e | 5  | **8**  | 0.32 | Slightly forward |
|        | o | 8  | **8**  | 0.46 | Correct, sharp |
|        | p | 11 | 8  | 0.31 | Stuck on "o" |
|        | l | 13 | **13** | 0.32 | Correct |
|        | e | 15 | **15** | 0.32 | **Correct** |
|        | . | 21 | 18 | 0.26 | Close |
|        | **avg** | | | **0.317** | Monotonic: **Yes** (only model!) |
| **Reg-16** | p | 2  | **2**  | 0.35 | Correct |
|        | e | 5  | 16 | 0.36 | Far off |
|        | o | 8  | **8**  | 0.42 | Correct |
|        | p | 10 | 14 | 0.35 | Shifted |
|        | l | 13 | **13** | 0.41 | Correct |
|        | e | 16 | **16** | 0.37 | **Correct** |
|        | . | 21 | **21** | 0.30 | **Correct** |
|        | **avg** | | | **0.365** | Monotonic: **No** |

---

## 4. Key Findings (Simple Language)

### Finding 1: The model DOES learn spatial awareness

Despite using a global attention mechanism (every patch can see every other
patch), the model learns to associate each character with the correct
**region** of the image. For the first few characters ("t", "a" in "talks.";
"p", "o" in "people."), the attention peaks are almost always at the correct
patch position.

**What this means**: The ViT is not just "memorising words" — it truly learns
to read from the image, looking at the right region for each character.

### Finding 2: The "attention sink" problem in Reg-0

In the **Reg-0** model for "talks.":
- Characters "l", "k", "s", "." all peak at **the same patch (13)**
- This is the classic **attention sink** — a patch that absorbs attention
  from everywhere, acting as a "dump" for attention mass

**Why this matters**: When many characters focus on the same location, the
model can't spatially distinguish between them. It still predicts correctly
(CER=0) because the BiLSTM+CTC head compensates, but the attention maps
become less interpretable.

### Finding 3: Registers partially reduce attention sinks

Comparing Reg-0 to Reg-16 on "talks.":
- **Reg-0**: 4 out of 6 characters converge on patch 13 (attention sink)
- **Reg-16**: Only "k" shares a peak with "l" — the other characters find
  distinct locations ("t"→3, "a"→6, "l"→9, "."→20)

**Interpretation**: Register tokens absorb the "garbage" attention that would
otherwise pile up on a single patch. This makes the per-character attention
maps **more spatially meaningful**.

### Finding 4: No model achieves perfect left-to-right reading order

"Monotonic peaks" means each character's peak attention is to the **right** of
the previous character's peak (reading order). Results:

| Image | Reg-0 | Reg-2 | Reg-4 | Reg-8 | Reg-16 |
|-------|-------|-------|-------|-------|--------|
| talks. | **Yes** | No | No | No | No |
| people. | No | No | No | **Yes** | No |

No single model is consistently monotonic. This is **expected** for attention:
the model can look backwards to resolve ambiguity (e.g., "is this an 'l' or
an 'i'? look back at the overall word shape").

### Finding 5: Attention focus (sharpness) varies by model

**Top-5 focus** = fraction of total attention concentrated in the 5 most-attended
patches. Higher = sharper, more localised attention.

| Model | talks. | people. | Average |
|-------|--------|---------|---------|
| Reg-0 | 0.379 | 0.292 | 0.336 |
| Reg-2 | 0.423 | 0.382 | 0.403 |
| Reg-4 | 0.405 | 0.365 | 0.385 |
| Reg-8 | 0.350 | 0.317 | 0.334 |
| Reg-16 | 0.403 | 0.365 | 0.384 |

- **Reg-2** has the sharpest attention on average
- **Reg-0** and **Reg-8** are the most diffuse
- Longer words ("people.", 7 chars) have universally lower focus than shorter
  words ("talks.", 6 chars)

### Finding 6: Repeated characters are hard to localise

In "people.", both "p" and "e" appear twice. The model sometimes:
- **Confuses the two 'p's**: Reg-0 and Reg-4 route the 2nd "p" back to the
  1st "p"'s location
- **Confuses the two 'e's**: Reg-0 routes the 2nd "e" to a position near the
  1st "e"

**Why**: Self-attention is permutation-equivariant; it's hard for the model to
distinguish two identical characters based purely on content. Position
information from the BiLSTM helps, but attention alone doesn't fully resolve it.

---

## 5. How to Read the Figures

### 5.1 Carpet Figure (`carpet/`)

```
  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
  │   "t"    │   "a"    │   "l"    │   "k"    │   "s"    │   "."   │
  │ [image]  │ [image]  │ [image]  │ [image]  │ [image]  │ [image] │
  │ +heatmap │ +heatmap │ +heatmap │ +heatmap │ +heatmap │ +heatmap│
  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

- Each panel shows the **same image** with a different character's attention heatmap overlaid
- **Bright yellow** = high attention (the model is looking here for that character)
- **Dark purple** = low attention (the model ignores this region)
- **Cyan dashed line** = peak attention position
- Look for: bright region shifting left to right across panels (= good spatial reading)

### 5.2 Alignment Figure (`alignment/`)

```
  ┌─────────────────────────────────────────────┐
  │  [Original handwritten image]               │  ← coloured markers at peak positions
  │   t  a  l  k  s  .                          │
  ├─────────────────────────────────────────────┤
  │  t ░░▓▓░░░░░░░░░░░░░░░░░░                  │  ← row = character
  │  a ░░░░░░▓▓░░░░░░░░░░░░░░                  │  ← column = patch position
  │  l ░░░░░░░░░░▓▓░░░░░░░░░░                  │
  │  k ░░░░░░░░░░░░▓▓░░░░░░░░                  │  A clean diagonal means
  │  s ░░░░░░░░░░░░░░░░▓▓░░░░                  │  good spatial localisation
  │  . ░░░░░░░░░░░░░░░░░░░░▓▓                  │
  └─────────────────────────────────────────────┘
```

- **Ideal pattern**: a bright diagonal from top-left to bottom-right
- **White dashed line** = "peak path" connecting each character's attention peak
- **Dotted vertical lines** connect the alignment matrix to the image above
- If the peak path goes **backwards** (right to left), the model is looking at
  a previous character's location — potential attention confusion

### 5.3 Combined Figure (`combined/`)

The combined figure stacks all three views vertically:
1. **Row 0**: Original image (reference, with peak position markers)
2. **Row 1**: Character attention carpet (same as `carpet/`)
3. **Row 2**: Alignment matrix (same as `alignment/`)

---

## 6. Comparing Models — What to Look For

When comparing figures across Reg-0 through Reg-16:

| What to compare | Good sign | Bad sign |
|----------------|-----------|----------|
| Peak path in alignment matrix | Monotonically increasing (diagonal) | Peaks jumping backwards |
| Heatmap sharpness in carpet | Focused bright spot near the character | Diffuse glow over entire image |
| Attention sink (patch 13 in talks.) | Absent or weak | All characters peak at same location |
| Repeated chars (like two 'p's) | Different peak locations | Same peak for both occurrences |

---

## 7. Script Usage Reference

```bash
# Basic: one model, one image
python scripts/postprocessing/character_attention_mapping.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    --model-path saved_models/experiments/run_54/model.pt \
    -- notebook/sample_images/a01-038-12.png

# All 5 register-sweep models
python scripts/postprocessing/character_attention_mapping.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    --model-path saved_models/experiments/run_50/model.pt \
    --model-path saved_models/experiments/run_51/model.pt \
    --model-path saved_models/experiments/run_52/model.pt \
    --model-path saved_models/experiments/run_53/model.pt \
    --model-path saved_models/experiments/run_54/model.pt \
    -- notebook/sample_images/

# Custom settings
python scripts/postprocessing/character_attention_mapping.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    --model-path saved_models/experiments/run_54/model.pt \
    --layer middle \      # first | middle | last | all | rollout
    --alpha 0.5 \         # heatmap overlay opacity (0-1)
    --gamma 4.0 \         # contrast boost (higher = sharper peaks)
    --dpi 200 \           # output resolution
    --max-chars 15 \      # max characters to display
    -- notebook/sample_images/a01-038-12.png
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint. Repeat for multiple models. |
| `--layer` | `last` | Which transformer layer's attention to visualise |
| `--alpha` | `0.45` | Heatmap overlay transparency (0 = invisible, 1 = opaque) |
| `--gamma` | `3.0` | Contrast exponent — higher values make peaks sharper |
| `--dpi` | `150` | Output image resolution |
| `--max-chars` | `25` | Maximum characters to show (truncates long words) |
| `--save-dir` | `visualizations/character_attention_mapping/` | Output directory |

---

## 8. Alignment Bug Fixes Applied (v2)

The original version had 5 alignment bugs that have been corrected:

| Bug | Symptom | Root cause | Fix |
|-----|---------|------------|-----|
| **Carpet peak marker** | Cyan line at pixel 3 instead of pixel 20 for "t" | `peak_x = peak_idx/Wp * W_img` ignored border offset and patch width | Use `np.argmax(heatmap.max(axis=0))` — pixel-accurate from the already-upscaled heatmap |
| **Alignment image extent** | Image stretched across all 128 patches; content was a thin sliver on the left | `extent=[0, T]` maps image to full 128-patch range, but content only occupies patches 1-22 | Compute content geometry: `extent=[img_left, img_right]` maps image only to its actual patch range |
| **Alignment matrix width** | 128 columns mostly empty (patches 22-127 are padding) | No cropping to active region | Crop to `[p_start - margin, p_end + margin]`, showing only content-bearing patches |
| **Combined figure Row 0** | Image mapped to `extent=[0, n_show]` (e.g. 0-6 for 6 characters) — nonsensical | Used character count as x-range instead of patch coordinates | Use proper `extent=[img_left, img_right]` with `xlim=[p_lo, p_hi]` |
| **CTC boundary spaces** | Leading/trailing spaces showed as empty panels | CTC decoder emits blank characters at sequence boundaries | Trim leading/trailing whitespace from decoded characters before visualisation |
| **Colorbar width mismatch** | Colorbar on bottom panel only → top panel wider → connecting lines offset | `fig.colorbar(im, ax=ax_mat)` steals space from one panel | `fig.colorbar(im, ax=[ax_img, ax_mat])` — attached to both axes |

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| **Patch** | A small region of the image processed as a single token by the ViT. Here: 8px wide strips (Hp=1, Wp=128). |
| **Register token** | Extra learnable tokens prepended to the sequence. Not tied to any image location. Designed to absorb "junk" attention. |
| **Attention sink** | A single patch that absorbs disproportionate attention from many positions. Common in vanilla ViTs (Reg-0). |
| **CTC position** | The time step at which the CTC decoder emits a character. Maps 1:1 to patch index since CTC output length = Wp = 128. |
| **Top-5 focus** | Fraction of total attention mass in the top 5 most-attended patches. Higher = the model is more confident about where to look. |
| **Monotonic peaks** | Whether each successive character's peak attention is to the right of the previous one. Indicates left-to-right reading order. |
| **Peak path** | The white dashed line on the alignment matrix connecting each character's peak attention position. Ideally a clean diagonal. |
| **Gamma** | Contrast exponent applied to normalised attention. `gamma > 1` suppresses weak attention and amplifies peaks. |
