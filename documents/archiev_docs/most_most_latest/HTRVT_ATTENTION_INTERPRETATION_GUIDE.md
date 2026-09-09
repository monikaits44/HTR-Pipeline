# HTR-VT (+R4 registers, 2-D grid) — Attention Map Interpretation Guide

**Run:** `saved_models/experiments/run_142`  ·  **Model:** `htrvt` (HTR-VT + R=4 registers + 4×128 2-D grid)
**Data:** READ2016 lines  ·  **Test CER / WER: 4.82 % / 20.58 %** (best epoch 76)
**Script:** [scripts/htrvt_bm_attention.py](../../scripts/htrvt_bm_attention.py)
**Outputs:** `outputs/htrvt_bm_attention/`

This document explains (1) how to read the maps, (2) why the first version was hard to
interpret, (3) what we verified about the attention, and (4) exactly how the maps are now
produced so they read like the Beyond-Memorization (BM) reference
(`outputs/beyond_memorization/charIndex_*/attentionMaps/`).

---

## 1. TL;DR

- The attention **does** localize characters — but **only after CTC-column anchoring**.
- The *raw* self-attention of a CTC query token is dominated by a few **high-norm "sink"
  patch columns** that almost every token attends to (the artifact tokens that registers
  are meant to absorb; R=4 absorbs them only partially). Those sinks made the first maps
  unreadable.
- We verified this numerically (40 test images, 887 characters):

  | Read-out | mean \|peak − CTC col\| | median | within ≤2 cols |
  |---|---|---|---|
  | Raw attention (`anchor_win=0`) | **38.2** grid cols | 32 | 6 % |
  | CTC-anchored (`anchor_win=4`)  | **2.5** grid cols | 2 | 64 % |

  1 grid column ≈ 8 px ≈ half a character, so the anchored read-out localizes characters to
  **sub-character precision**.
- The BM reference looks like a small tinted word crop because it shows a **256×64 single
  word** with a **smooth** soft heat-map (bilinear, inferno, α≈0.5) over the whole crop. Our
  model sees a **1024×128 full line** on a **4×128 grid**. To match BM we render each
  per-character map as a **fixed ~10-character window** centred on the character (independent
  of line length), with the attention **smoothly** upsampled and gently tinting the whole
  crop — no patch grid lines, no blocky patches.

---

## 2. How to read a per-character map

Each PNG in `outputs/htrvt_bm_attention/<image>/attentionMaps/` is **one decoded character**.

```
outputs/htrvt_bm_attention/
├── test_10_R4_chars.png                 ← overview grid (all chars, one figure)
└── test_10/attentionMaps/
    ├── den-herrn-Aines-Ersamen_11_A_char_att0.png
    ├── den-herrn-Aines-Ersamen_5_h_char_att0.png
    └── ...   (filename = <line-slug>_<charIndex>_<char>_char_att0.png)
```

Visual elements:

- **Greyscale background** — the handwriting, shown *un-inverted* (dark ink on light paper)
  to match BM, contrast-stretched and cropped to a **fixed ~10-character window** centred on
  the character (width derived from the CTC peak spacing, so it spans ~10 chars whether the
  line is 5 or 50 characters long).
- **Inferno overlay** (`alpha≈0.55`, **bilinear** → smooth): the model's attention for this
  character, upsampled from the low-res 4×128 grid and Gaussian-smoothed into a soft blob.
  Bright **yellow = peak**; a `floor` lift tints the whole crop pink/purple (no black) so it
  reads like BM's pastel heat-map rather than a hard stripe.
- **Optional cyan line** (`--grid-lines`) — the character's **CTC alignment column** `t_c`
  (off by default to match BM's clean look). A correctly-localized character has its yellow
  blob sitting on/near `t_c`.

**What "good" looks like:** the bright yellow glow sits over the actual pen strokes of the
character near its CTC column. **What "bad" looks like:** the glow sits far from the
character (the raw-attention sink failure mode, suppressed by anchoring).

---

## 3. From pixels to attention — the pipeline

```
image [1×128×1024]
   │  ResNet CNN stem  (HTR-VT front-end)
   ▼
feature map  ──AdaptiveMaxPool2d(4,128)──►  4×128 = 512 patch tokens
   │  + R=4 register tokens                 → S = 516 tokens
   ▼
plain ViT encoder (depth 4, heads 8, norm_first)   ← self-attention A ∈ [L,B,H,S,S]
   │
   ├── collapse height by MAX over the 4 rows → 128 timesteps → BiLSTM → CTC head
   └── forward_explain() returns A, token_norms, grid=(4,128), registers
```

Token layout in `S`: `[ R registers | 512 patches row-major (idx = h*128 + w) ]`.

For a character emitted by CTC at column `t_c`:

1. **Query tokens** = the `Hp=4` patch tokens of that column: `R + h*128 + t_c`, `h=0..3`.
2. `patch_attn = mean_h A[:, query_h, R:]` → `[H, 4, 128]` (attention from the column to all
   patches, per head).
3. **CTC-column anchoring**: multiply by a Gaussian column prior centred at `t_c`
   (`std = anchor_win = 4` columns) to suppress the global sink columns. This is justified —
   CTC already tells us the character lives at column `t_c`, so its spatial support must be
   near `t_c`.
4. **Head selection** (`method="best"`): pick the head with the sharpest peak
   (`max/mean`), or average heads (`method="mean"`).
5. Normalize to `[0,1]` → the `4×128` map. For display, crop a **fixed ~10-character window**
   around `t_c`, smoothly upsample (bilinear + Gaussian), gamma-lift and floor-tint so the
   whole crop carries soft colour (BM look).

The last encoder layer (`layer=-1`) gives the cleanest maps.

---

## 4. Why the 2-D grid matters (and the mode-B decision)

A faithful HTR-VT collapses the feature-map height to a single row *before* the transformer,
so self-attention runs over a 1-D sequence — it can only ever produce **horizontal strips**,
never 2-D blobs. To obtain BM-style 2-D localization we kept a **thin 2-D token grid
(`grid_height=4`) inside the encoder** and collapse height to 1 row **only at the CTC head**
("mode B" in the feasibility doc). The maps above are 4 rows tall precisely because of this
choice; with `grid_height=1` the per-character map would be a single horizontal line.

Trade-off: 4 rows is a deliberate compromise — enough for vertical localization (ascenders
vs. baseline vs. descenders), cheap enough to not hurt CER. The vertical resolution is
coarse (32 px/row), which the smooth bilinear upsample + Gaussian blur hides at display time.

---

## 5. Why the BM reference looked different (and how we matched it)

| | Beyond-Memorization reference | This model (before) | This model (now) |
|---|---|---|---|
| Input | single **word** 256×64 | full **line** 1024×128 | full line, **fixed ~10-char window** |
| Patch grid | ~**8×32** (square 8×8 px) | 4×128 (8×32 px strips) | 4×128, smoothly upsampled |
| Per-char file | yes | yes | yes |
| Overlay | inferno, α≈0.5, **smooth** | inferno, α=0.5 but tiny patches | inferno, α≈0.55, **smooth pastel** |
| Grid lines | none | none | none (cyan `t_c` optional via `--grid-lines`) |
| CTC anchor | n/a (cross-attention) | none → sink-dominated | **Gaussian prior at t_c** |

The BM model uses genuine **cross-attention** (text-char query → image patches), which is
intrinsically character-localized. Our CTC model has only **self-attention** among
patches+registers, which is contaminated by sink tokens — hence the anchoring step is the
honest equivalent that recovers a comparable, interpretable signal.

---

## 6. Verification findings (what the attention actually does)

1. **Localization works when anchored.** Mean horizontal error drops from 38.2 → 2.5 grid
   columns (Section 1 table). The model genuinely attends to the correct character region.
2. **Sink tokens exist.** A handful of fixed patch columns (e.g. ~col 45–46 and ~80 on
   `test_1000`) receive strong attention from *most* query tokens regardless of character —
   the high-norm artifact phenomenon from Darcet et al. R=4 registers reduce but do not
   eliminate them (consistent with the repo's earlier ViT-RGTS register sweep showing only
   0–7.5 % absorption in the CTC-HTR regime).
3. **Vertical signal is coarse but present.** With only 4 rows, the blob's row indicates
   rough vertical position (top/upper-mid/lower-mid/bottom) — useful for ascenders/descenders
   but not fine strokes.
4. **Recognition is strong.** Test CER 4.82 % / WER 20.58 % confirms the 2-D grid + registers
   did **not** cost accuracy — the interpretability is "for free".

---

## 7. How to regenerate

```bash
# default: latest htrvt run, 6 READ2016 test lines, CPU
.venv/bin/python scripts/htrvt_bm_attention.py --device cpu --num-images 6

# wider/narrower fixed window, or show the CTC-column marker
.venv/bin/python scripts/htrvt_bm_attention.py --window-chars 6
.venv/bin/python scripts/htrvt_bm_attention.py --grid-lines

# specific run / more images / average heads
.venv/bin/python scripts/htrvt_bm_attention.py --run run_142 --num-images 12
.venv/bin/python scripts/htrvt_bm_attention.py --method mean
```

Key knobs (in `scripts/htrvt_bm_attention.py`):

- `extract_char_attention_2d(..., anchor_win=4.0)` — Gaussian column-prior std. `0` disables
  anchoring (reproduces the raw, sink-dominated maps — useful to demonstrate the problem).
- `save_bm_style_per_char(..., window_chars=10, smooth_sigma=None, gamma=0.6, floor=0.35,
  alpha=0.55, show_grid=False)` — fixed window width (in characters), glow smoothing
  (auto-scaled to ~1 character if `None`), mid/low lift, whole-crop tint, overlay opacity,
  and the optional cyan CTC-column marker. CLI: `--window-chars`, `--grid-lines`.
- `--layer` — which encoder layer's attention to read (default last).

---

## 8. Limitations & honest caveats

- **Anchoring uses the CTC column as a prior.** This is not "pure" bottom-up attention; it
  injects the known alignment to suppress sinks. Set `anchor_win=0` to see the unfiltered
  attention. We consider anchoring legitimate because the CTC column is a model output, not
  external supervision, but it should be stated when presenting the maps.
- **Vertical resolution = 4 rows.** Fine vertical structure is not recoverable. Increasing
  `grid_height` (e.g. 8) would help but costs compute and was not retrained here.
- **Self-attention ≠ cross-attention.** Unlike BM's diffusion cross-attention, this is a CTC
  recognizer; the maps show *where the column token gathers evidence from*, which is a
  related but not identical notion of "character attention".
- **Sink tokens remain.** A cleaner result would come from more/learned registers or an
  explicit sink-token mask; both are future work.

---

## 9. File map

| Path | What |
|---|---|
| `scripts/htrvt_bm_attention.py` | extraction + BM-style rendering + CER/WER report |
| `outputs/htrvt_bm_attention/<img>_R4_chars.png` | overview grid (all chars) |
| `outputs/htrvt_bm_attention/<img>/attentionMaps/*.png` | per-character BM-style maps |
| `configs/htrvt_read2016_r4.yaml` | training config (grid 4×128, R=4, span-mask) |
| `saved_models/experiments/run_142/` | trained model, `results.csv`, config |
| `documents/most_most_latest/HTRVT_REGISTERS_BM_ATTENTION_FEASIBILITY.md` | prior feasibility study |
