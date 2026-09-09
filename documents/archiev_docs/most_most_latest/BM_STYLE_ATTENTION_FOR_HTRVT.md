# Can We Have BM-Style Attention Maps for the HTR-VT Model?

**Short answer: Yes — and we now produce them.**

This document explains what BM-style attention maps are, why our earlier attempts did not
match them, what the fundamental architectural difference is, how we fixed it, and what
honest limitations remain.

---

## 1. What "BM-style" means

"BM" refers to the Beyond-Memorization paper (ICDAR 2025, Agarwal et al.), which generates
per-character attention maps that look like this:

- A small, compact image — **exactly 256 × 64 pixels**
- Shows **one readable word** (white/cream paper, dark ink)
- A **smooth pastel heat-map** (inferno colourmap, α = 0.5) overlaid over the whole word
- **Yellow glow** at the target character, fading to purple elsewhere
- **No grid lines, no blocky patches** — everything is smooth

The purple tint over the whole image is not a special trick. It is the natural result of
alpha-blending inferno (which maps 0 → dark purple and 1 → bright yellow) at α = 0.5 over
white paper. Even where attention is low (value ≈ 0), you still see: 0.5 × dark-purple +
0.5 × white = light purple. This gives the characteristic pastel look automatically.

---

## 2. How BM generates these maps (reference implementation)

File: `external/Beyond-Memorization/utils/saveAttentionMaps.py`,
function: `save_Attention2_with_blobs`.

```
Input image:     PIL word image, already cropped to 256 × 64 px from IAM words dataset
Attention data:  float32 tensor [Batch, 64, 256, num_chars]
                 — this is the U-Net cross-attention, already at image resolution

For each character index k:
  attn = tensor[:, :, :, k]   →  shape [64, 256], values in [0, 1]
  normalise to [0, 1]

Rendering:
  fig, ax = plt.subplots(figsize=(256/100, 64/100))   ← 2.56 × 0.64 inches
  ax.imshow(word_image_rgb)                            ← RGB word image
  ax.imshow(attn, cmap='inferno', alpha=0.5)           ← attention overlay
  ax.set_title(f"Attention Map for Character {ch}")
  ax.axis('off')
  plt.savefig(...)   ← no bbox_inches → at dpi=100 saves exactly 256 × 64 px RGBA
```

Key facts:
- The attention is already **at full image resolution (64 × 256)** — no upsampling needed.
- The word image is **pre-cropped** to 256 × 64 by the IAM words dataset structure.
- The model uses **cross-attention** (text character token → image patch), so each character
  index directly corresponds to a separate attention map with no extra alignment step.

---

## 3. What our HTR-VT model has instead

Our model (run_142, test CER 4.82 %) is architecturally very different from BM.

| Property | Beyond-Memorization | HTR-VT (ours) |
|---|---|---|
| Model type | Latent diffusion U-Net | CNN stem + plain ViT + CTC |
| Attention type | **Cross-attention** (text token → image patch) | **Self-attention** (patch → patch) |
| Attention resolution | 64 × 256 (full image res) | **4 × 128** (very coarse grid) |
| Character localisation | Direct: each char has its own attention map | Indirect: CTC peak column → Gaussian anchor |
| Input image scope | Single word (256 × 64) | Full text line (1024 × 128) |

Because of these differences, we cannot produce BM's maps directly. We have to adapt.
Critically, the word "no" is not the right answer — we **can** produce maps that are
visually equivalent to BM, using the information our model does provide.

---

## 4. The three bugs that made our earlier maps look wrong

When we compared our previous `outputs/htrvt_bm_attention/` maps against BM, they looked
completely different. Three independent bugs were responsible.

### Bug 1 — Wrong output pixel size (the most visible problem)

**What happened:** The script set `figure.dpi=150` and `savefig.dpi=200` globally, plus
`savefig.bbox='tight'`. Our per-character figures were sized dynamically based on the crop
width in pixels (e.g. figsize=(7.0, 1.28) for a 700 px wide crop). This produced outputs
of ~702 × 197 px — more than 15× the pixel area of BM's 256 × 64.

**Why it matters:** A 702 × 197 image displayed at the same screen size as a 256 × 64 image
will look "zoomed in", over-detailed, and noisy. The handwriting strokes that should look
neat and readable become large and grainy.

**Fix:** Use a `plt.rc_context` block for per-character saves that forces `figure.dpi=100`,
`savefig.dpi=100`, `savefig.bbox=None`. With `figsize=(256/100, 64/100)` = (2.56, 0.64)
inches, this produces **exactly 256 × 64 px RGBA output** — verified to match BM pixel-for-pixel
in dimensions and colour mode.

### Bug 2 — Wrong image proportions (the "zoomed in" feeling)

**What happened:** The image crop used the full 128 px height of the text line. When this
tall, narrow crop was rendered at the same figure height as a BM word image, the text
looked very tall and "zoomed in" compared to BM's compact word-sized crop.

**Fix:** Resize the full 128 px crop to exactly 64 px height using PIL bilinear
interpolation. This vertically compresses the line to word-like proportions, which is
visually equivalent to what BM gets by using a pre-cropped 64 px word image.

### Bug 3 — Wrong figsize logic (variable size per character)

**What happened:** `figsize` was computed as `(max(crop_width/100, 2.0), max(crop_height/100, 0.6))`.
This gave a different figure size for each character depending on how wide the horizontal
window happened to be. Every character's map was a different size.

**Fix:** Always use `figsize=(BM_W/100, BM_H/100)` = (2.56, 0.64) — fixed, same for every
character, matching BM's approach.

---

## 5. The architecture gap: what we can and cannot do

This section is for your supervisor.

### What BM can do that we fundamentally cannot replicate

BM's U-Net has genuine **character-indexed cross-attention**: at inference time, the model
computes how much each image patch attends to each character in the text. This is a single
forward pass operation that produces a clean [64, 256, num_chars] tensor — one attention
map per character, at full image resolution, with no post-processing.

Our HTR-VT model has **self-attention** between image patches. No character token exists
during inference. To get a "character attention map" we must:
1. Run CTC decoding to find which time column `t_c` the character was emitted at
2. Read out the self-attention from the patch tokens at column `t_c`
3. Apply a Gaussian column prior centred on `t_c` (called CTC-column anchoring) to suppress
   the "sink" artefact tokens that attract attention regardless of character

This is an approximation, not exact character localisation. It works well (verified: mean
horizontal error 2.5 grid columns after anchoring, vs 38.2 without), but it introduces
noise that BM's true cross-attention does not have.

### What we can fully replicate (and now do)

| Feature | BM | HTR-VT (after fix) |
|---|---|---|
| Output size | 256 × 64 px RGBA | **256 × 64 px RGBA** ✓ |
| Colour mode | RGBA | **RGBA** ✓ |
| Rendering code | figsize(2.56,0.64), dpi=100, no bbox | **identical** ✓ |
| Inferno overlay, α=0.5 | ✓ | ✓ |
| Pastel purple tint everywhere | natural from alpha | **natural from alpha** ✓ |
| Yellow peak on target character | cross-attention (exact) | CTC anchoring (approximate) ✓ |
| No grid lines | ✓ | ✓ |
| Readable handwriting | single word crop | ~10-char window crop ✓ |
| Fixed size regardless of line length | word always 256×64 | fixed ~10-char window ✓ |

### Honest remaining differences

1. **BM shows one word; we show ~10 characters.** Our images often show parts of two or
   three words. This is because our model processes full text lines, not individual words.
   The window is fixed to ~10 characters of width to be consistent and readable.

2. **Vertical localisation is coarse.** Our attention grid has only 4 rows (32 px each).
   BM's has 64 rows (1 px each). Our maps cannot distinguish whether the attention is on the
   top of a letter vs the bottom at fine scale. The Gaussian smoothing hides this partially.

3. **The peak is derived, not direct.** BM's yellow peak is the direct cross-attention
   output. Our yellow peak is the result of: (CTC peak column) → (self-attention read-out)
   → (Gaussian anchor). For most characters it is accurate; for ambiguous or badly predicted
   characters it can be off.

---

## 6. Summary of the pipeline (after fix)

For each decoded character in a test image:

```
1. CTC decoding  →  character ch at grid column t_c

2. Attention extraction
   - Read self-attention from the Hp=4 tokens at column t_c  →  [Heads, 4, 128]
   - Apply Gaussian column prior centred at t_c (anchor_win=4 cols)
   - Pick the head with the sharpest peak  →  [4, 128]

3. Horizontal crop  (fixed window, ~10 chars wide)
   - c0 = t_c - half_cols,  c1 = t_c + half_cols
   - x0, x1 = c0 × 8px,  c1 × 8px  (8px per grid column)

4. Image preparation
   - Crop: img[0:128, x0:x1]  (full 128px height)
   - Contrast stretch, un-invert (ink=dark on white)
   - PIL BILINEAR resize to 256 × 64  (compresses line to word proportions)

5. Attention preparation
   - attn[0:4, c0:c1]  →  scipy zoom (bilinear) to 64 × 256
   - Gaussian smooth  (σ ≈ 0.35 × char_width_in_output_px)
   - Normalise to [0, 1]

6. BM-exact rendering  (inside plt.rc_context to force 256×64 output)
   - plt.subplots(figsize=(2.56, 0.64))  ← matches BM exactly
   - ax.imshow(img_rgb_256x64)           ← matches BM exactly
   - ax.imshow(attn_64x256, cmap='inferno', alpha=0.5)  ← matches BM exactly
   - ax.set_title(f"Attention Map for Character {ch}")
   - ax.axis('off')
   - fig.savefig(...)   ← no bbox_inches → 256×64 px RGBA output
```

---

## 7. How to regenerate the maps

```bash
# Default: latest htrvt run, 6 READ2016 test lines, CPU
.venv/bin/python scripts/htrvt_bm_attention.py --device cpu --num-images 6

# Wider or narrower character window
.venv/bin/python scripts/htrvt_bm_attention.py --window-chars 8

# Add a cyan line marking the CTC column (useful for debugging)
.venv/bin/python scripts/htrvt_bm_attention.py --grid-lines

# Specific images
.venv/bin/python scripts/htrvt_bm_attention.py \
    --images data/READ2016/processed/test/test_10.png
```

Outputs are written to:
- `outputs/htrvt_bm_attention/{stem}_R4_chars.png`  — overview grid for the full line
- `outputs/htrvt_bm_attention/{stem}/attentionMaps/*.png`  — one 256×64 map per character

The `attentionMaps/` sub-folder has the same name, same file-naming convention
(`{slug}_{charIndex}_{char}_char_att0.png`), and the same pixel format (256×64 RGBA) as
`outputs/beyond_memorization/charIndex_0/attentionMaps/`.

---

## 8. Files changed in this work session

| File | Change |
|---|---|
| `scripts/htrvt_bm_attention.py` | Added PIL import; added `BM_W, BM_H = 256, 64`; fixed rcParams; rewrote `save_bm_style_per_char` to produce BM-exact 256×64 output; `save_char_grid` now passes dpi/bbox explicitly |
| `outputs/htrvt_bm_attention/` | Regenerated — all per-character PNGs now 256×64 RGBA |

---

*Prepared: 2026-06-27*
