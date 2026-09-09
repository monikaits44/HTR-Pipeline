
**Script path:** `scripts/postprocessing/character_attention_mapping.py`  
**Generates (among others):** `visualizations/character_attention_mapping/alignment/a01-038-12_run_50.png`

---

## 1. Purpose

`character_attention_mapping.py` is a post-processing visualisation tool for
the ViT-RGTS Handwritten Text Recognition (HTR) model.  Given one or more
trained model checkpoints and one or more line-image files, it produces
three families of publication-ready PNG figures that show **where** the model
"looks" on the input image when predicting each character.

---

## 2. Output artefacts and directory structure

Running the script creates (or re-creates) three sub-directories under the
output root (`visualizations/character_attention_mapping/` by default):

| Sub-directory | Figure type | What it shows |
|---|---|---|
| `carpet/` | Character attention carpet | The same line image repeated N times, once per predicted character, each copy overlaid with the heat-map for that character (mirrors the classic `sample_attention.png` style). |
| **`alignment/`** | **Character × Patch alignment matrix** | A two-panel figure: the source line image on top and a 2-D matrix (characters × patch positions) at the bottom. **This is the sub-directory that contains `a01-038-12_run_50.png`.** |
| `combined/` | All-in-one figure | Three stacked panels: reference image, character carpet, and alignment matrix in a single figure. |

### Filename convention

```
{image_stem}_{run_name}.png
```

For `a01-038-12_run_50.png`:
- `a01-038-12` — stem of the input image file (e.g. `notebook/sample_images/a01-038-12.png`)
- `run_50` — the checkpoint directory name inside `saved_models/experiments/`
  (i.e. the model was loaded from `saved_models/experiments/run_50/model.pt`)

---

## 3. Invocation

```bash
python scripts/postprocessing/character_attention_mapping.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    --model-path saved_models/experiments/run_50/model.pt \
    --model-path saved_models/experiments/run_54/model.pt \
    -- notebook/sample_images/a01-038-12.png
```

Key CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--model-path P` | *(required)* | Path to a model checkpoint; repeat for multiple models. |
| `--layer LAYER` | `last` | Which transformer layer's attention to use: `first`, `middle`, `last`, `all` (mean over all layers), `rollout`. |
| `--alpha FLOAT` | `0.45` | Heatmap overlay opacity on the carpet panels. |
| `--dpi INT` | `150` | Output PNG resolution. |
| `--gamma FLOAT` | `3.0` | Contrast-boost exponent applied to normalised attention vectors. |
| `--max-chars INT` | `0` (all) | Cap the number of characters shown per figure. |
| `--save-dir DIR` | `visualizations/character_attention_mapping/` | Root output directory. |
| `--` | — | Separator; image paths (files or directories) follow. |

---

## 4. Step-by-step execution flow

### 4.1 Argument parsing (`parse_args`)

- Splits `sys.argv` at the literal `--` separator into *config args* and
  *image args*.
- Extracts flags (`--model-path`, `--layer`, etc.) and positional YAML files.
- Merges YAML configs in order with `OmegaConf.merge`, then applies any
  trailing `key=value` overrides.
- Expands image directories to individual file paths (png/jpg/bmp/tiff).

### 4.2 Environment and vocabulary setup

```
classes = np.load(dataset_folder/classes.npy)
num_classes = len(classes) + 1          # +1 for CTC blank
i2c = {i+1: c for i, c in enumerate(classes)}   # integer → character
fixed_size = (conf.preproc.image_height, conf.preproc.image_width)
```

- `classes.npy` holds the vocabulary array saved alongside the training data.
- `fixed_size` is the canonical (H, W) the model was trained on (e.g. 64 × 1536).

### 4.3 Model loading (`get_run_info` + `build_model`)

For each `--model-path`:
1. `get_run_info` reads `config.json` from the checkpoint's directory to
   recover `arch.num_registers` (number of ViT register tokens).
2. `build_model` instantiates `HTRNet` with the merged config, loads the
   `state_dict` (strict), moves the model to the target device, and sets
   `eval()` mode.
3. Models are sorted ascending by `num_registers` before processing so output
   figures are consistently ordered.

### 4.4 Ground-truth loading (`load_gt`)

Scans every directory that contains an input image for a `gt.txt` file.
Lines have the format `<image_stem> <transcription>`.  Results are stored in
a dict keyed by image stem so they can be printed on the figures.

### 4.5 Per-image processing loop

For each `(model, image)` pair:

#### a) Preprocessing
```python
raw_img  = load_image(img_path)          # utils/preprocessing.py
processed = preprocess(raw_img, fixed_size)
img_tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
```
The image is resized/padded to `fixed_size` with an 8-pixel border, then
normalised to [0, 1].

#### b) Attention extraction (`extract_attention`)
```python
logits, reg_tokens, attn_maps_list, token_norms, (Hp, Wp) = \
    net.forward_explain(img_tensor)
```
`forward_explain` is a special forward pass that returns, in addition to the
CTC logits, a list of per-layer attention maps
`[L × (1, num_heads, S, S)]` where `S = num_registers + Hp*Wp`.

`select_layer` then reduces this list to a single `[S, S]` attention matrix
according to the chosen `--layer` strategy:

| `--layer` | Reduction |
|---|---|
| `first` | Layer 0 attention, averaged over heads |
| `middle` | Layer L//2 attention, averaged over heads |
| `last` | Layer -1 attention, averaged over heads |
| `all` | Mean of all layers, averaged over heads |
| `rollout` | Attention rollout: recursive product with residual identity, averaged over heads at each layer |

#### c) CTC decoding (`ctc_char_positions`)
Greedy argmax over the logits time-axis, collapsing repeated tokens and
blanks (standard CTC decoding).  Returns two parallel lists:
- `chars` — predicted character strings
- `positions` — CTC time-step (= patch column index) at which each character
  peaked

Leading/trailing whitespace tokens are trimmed.

#### d) Per-character attention extraction (`get_char_attention`)

For each character at CTC position `t`:

1. The row `attn[num_reg + t, num_reg : num_reg + T]` is read from the
   attention matrix, giving a length-`T = Hp × Wp` vector over patch tokens.
2. The vector is percentile-clipped (1 %–99 %), linearly re-scaled to [0, 1],
   then raised to the power `gamma_adaptive` (base `gamma` plus a small
   adaptive increment for long words) to sharpen the contrast.
3. All normalised vectors are stacked into `alignment[n_chars, T]`.

#### e) Figure generation

Three figures are saved per `(model, image)` pair:

**Carpet** (`visualize_char_attention_carpet`) — `carpet/`
- Opens the grayscale image.
- For each character, reshapes its normalised attention vector to `(Hp, Wp)`,
  up-scales it to image pixel size via `np.kron` (nearest-neighbour), and
  crops to the actual content region.
- Plots a 1 × N grid of axes (one axis per character).  Each axis shows the
  greyscale image with the heat-map overlaid (`cmap="inferno"`, `alpha`
  controls opacity) and a cyan dashed vertical line at the peak-attention
  pixel column.

**Alignment matrix** (`visualize_alignment_matrix`) — `alignment/`  ← **this produces `a01-038-12_run_50.png`**
- Creates a 2-row figure (`gridspec` with `height_ratios=[1.5, …]`).
- **Top panel (ax_img):** The greyscale line image is rendered with the
  x-axis mapped to patch-column coordinates (via `compute_content_geometry`)
  so it aligns exactly with the matrix below.  A coloured dashed vertical
  line and a small character label are drawn at each character's peak patch
  position.
- **Bottom panel (ax_mat):** The cropped `alignment[n_chars, p_lo:p_hi]`
  matrix is displayed as an `imshow` heat-map (`cmap="inferno"`).  The
  y-axis lists the predicted characters; the x-axis lists patch positions
  (left-to-right).  A white dashed "peak path" line connects successive peak
  positions — a diagonal trend indicates correct left-to-right spatial
  localisation.
- `matplotlib.patches.ConnectionPatch` draws dotted lines from each
  alignment-peak cell in the matrix up to the corresponding position in the
  image above.
- A shared colorbar (attached to both axes) labels attention intensity as
  *normalised attention*.

**Combined** (`visualize_combined`) — `combined/`
- Three-row `gridspec` stacking: reference image row, carpet row (sub-grid of
  N panels), alignment matrix row.

### 4.6 Cleanup

After all images for a model are processed:
- The model is deleted and `gc.collect()` is called to free GPU/CPU memory
  before the next model is loaded.

---

## 5. Key data structures

| Name | Shape | Description |
|---|---|---|
| `attn` | `[S, S]` | Selected single-layer (or aggregated) attention matrix. `S = num_reg + Hp*Wp`. |
| `char_vectors` | `list of [T]` | Raw attention rows per character, `T = Hp*Wp`. |
| `char_normed` | `list of [T]` | Percentile-clipped, normalised, gamma-boosted versions. |
| `alignment` | `[n_chars, T]` | Stacked `char_normed` — the character × patch alignment matrix. |
| `alignment_crop` | `[n_chars, p_hi−p_lo]` | `alignment` cropped to the patch columns that overlap actual image content. |
| `heatmaps` | `list of [H_img, W_img]` | Per-character attention up-scaled to full image pixel resolution via `np.kron`. |

---

## 6. Dependencies

| Module | Role |
|---|---|
| `models.HTRNet` | ViT-RGTS model with `forward_explain()` hook |
| `utils.preprocessing` | `load_image`, `preprocess` (resize, pad, normalise) |
| `omegaconf` | YAML config merging |
| `numpy` | Attention arithmetic, `np.kron` upscaling |
| `torch` | Model inference, tensor operations |
| `matplotlib` | All figure rendering (`gridspec`, `ConnectionPatch`, `imshow`) |
| `PIL.Image` | Loading source images for figure rendering |

---

## 7. Reproducing `a01-038-12_run_50.png`

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate

python scripts/postprocessing/character_attention_mapping.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    --model-path saved_models/experiments/run_50/model.pt \
    --layer last \
    --alpha 0.45 \
    --dpi 150 \
    --gamma 3.0 \
    -- notebook/sample_images/a01-038-12.png
```

The alignment figure is then written to:
```
visualizations/character_attention_mapping/alignment/a01-038-12_run_50.png
```

---

## 8. Reading the alignment figure

| Visual element | Meaning |
|---|---|
| Top panel — greyscale image | The handwritten line image, x-axis aligned with patch positions |
| Coloured dashed verticals | Peak attention patch for each predicted character |
| Small coloured labels above image | Predicted characters at their peak positions |
| Bottom panel — inferno heat-map | `alignment[char_i, patch_j]` — brighter = stronger attention |
| White dashed diagonal | Connects sequential peak positions; a clean diagonal = correct left-to-right reading order |
| Dotted `ConnectionPatch` lines | Link each matrix row's peak cell to the image above |
| Shared colorbar | Maps colour to *normalised attention* (0–1) |

A prominent near-diagonal bright band in the bottom panel indicates that the
model spatially localises each character to the correct horizontal region of
the image, which is a sign of well-learned character-level alignment.
