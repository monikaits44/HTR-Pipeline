# Beyond-Memorization: End-to-End Implementation Documentation

## Paper Reference

**"Beyond Memorization: Training-Free Style Mixing for Variability in Handwritten Text Generation Using Writer Embedding Injection in Pretrained Diffusion Models"**
- Authors: Aniket Gurav, Shivakumara Chanda, Rangaraj M. Krishnan
- Published: ICDAR 2025
- Official Repository: https://github.com/aniketntnu/Beyond-Memorization
- PDF Location: `archive/papers/3.Beyond Memorization Training Free Style Mixing for.pdf`

---

## 1. Method Summary

### Core Idea

The paper introduces a **training-free** technique to increase handwriting style variability in a pretrained **WordStylist** latent diffusion model. Instead of training/fine-tuning, it performs:

1. **Writer Embedding Injection** — shuffles writer embeddings mid-denoising using character-position masks derived from cross-attention maps.
2. **Character Localization via Attention** — extracts cross-attention maps `A ∈ (B, L, H, W)` from the UNet middle block, slices per character, finds the spatial boundary `Xmax_c` of each character, and creates binary masks `Mc`.
3. **Style Mixing Formula**: `I(Wo, Wr)_c = (1−Mc) ⊙ I(Wo) + Mc ⊙ I(Wr)` where `Wo` = original writer, `Wr` = shuffled writer, applied progressively from character position `c`.

### charLocation Sweep

One invocation of the inference script runs 5 configurations:
| charLocation | Meaning |
|---|---|
| -1 | Original writer (no mixing, baseline) — produces `noChange/` output |
| 0 | Style mixed from character position 0 (leftmost) — `charIndex_0/` |
| 1 | Style mixed from character position 1 — `charIndex_1/` |
| 2 | Style mixed from character position 2 — `charIndex_2/` |
| 3 | Style mixed from character position 3 — `charIndex_3/` |

Each sweep produces generated handwriting images AND per-character cross-attention map overlays.

### Model Architecture

- **UNet**: WordStylist latent diffusion UNet (`UNetModel` from `unetVarStleMixExp4.py`)
  - `image_size=(64, 256)`, `in_channels=4`, `model_channels=320`, `out_channels=4`
  - `num_res_blocks=1`, `attention_resolutions=(1,1)`, `channel_mult=(1,1)`
  - `num_heads=4`, `num_classes=339` (writer styles), `context_dim=320`
  - `vocab_size=53` (character vocabulary), `max_seq_len=25` (MAX_CHARS)
- **VAE**: Stable Diffusion v1.5 `AutoencoderKL` (latent space: 64×256 → 8×32×4)
- **Diffusion**: 600-step DDPM, denoises from pure Gaussian noise

---

## 2. Repository & File Structure

### Workspace Layout

```
HTR-Pipeline/
├── configs/
│   └── beyond_memorization.yaml          ← Paper-model config (our creation)
├── data/
│   └── IAM/
│       ├── words_raw/                    ← Extracted words.tgz (115,320 PNGs)
│       └── words_64x256/                 ← Preprocessed 64×256 crops (16+ images)
├── external/
│   └── Beyond-Memorization/              ← Cloned official repo + our additions
│       ├── regFrmTrnVariStyleMixOcr.py   ← MAIN inference script (~1600 lines)
│       ├── unetVarStleMixExp4.py         ← UNet model architecture
│       ├── config.py                     ← Constants (MAX_CHARS, paths, etc.)
│       ├── gt/gany.filter27              ← Ground truth file (16 word entries)
│       ├── htr/                          ← HTR model code (optional, use_ocr=0)
│       ├── models/
│       │   └── ema_ckpt.pt              ← WordStylist EMA checkpoint (162 MB)
│       ├── stable-diffusion-v1-5/
│       │   └── vae/                     ← SD v1.5 VAE weights (335 MB)
│       ├── run_from_config.py           ← Config launcher (our creation)
│       ├── prepare_images.py            ← IAM preprocessing (our creation)
│       ├── download_pretrained.py       ← Weight downloader (our creation)
│       └── download_iam_words.sh        ← IAM data downloader (our creation)
├── experiments_execution/
│   └── slurm_modular/
│       └── beyond_memorization.slurm    ← GPU job script (our creation)
├── outputs/
│   └── beyond_memorization/             ← Generated outputs (80 images + 450 att maps)
│       ├── noChange/                    ← Baseline (charLocation=-1)
│       │   ├── *.png                    ← 16 generated word images
│       │   └── attentionMaps/           ← 90 per-character attention overlays
│       ├── charIndex_0/                 ← Style mixed from char 0
│       │   ├── *.png
│       │   └── attentionMaps/
│       ├── charIndex_1/
│       ├── charIndex_2/
│       └── charIndex_3/
├── archive/
│   └── IAM dataset/
│       └── words.tgz                    ← Source IAM word images (820 MB)
└── experiments_execution/
    └── logs/beyond_memorization/
        ├── output_*.log
        └── error_*.log
```

### Files Created by Us

| File | Purpose |
|------|---------|
| `configs/beyond_memorization.yaml` | All parameters: paths, model dims, batch size, flags |
| `external/Beyond-Memorization/run_from_config.py` | YAML→CLI arg translator; validates assets; supports `--dry-run` |
| `external/Beyond-Memorization/prepare_images.py` | Resizes raw IAM word PNGs to 64×256 (aspect-ratio + white pad) |
| `external/Beyond-Memorization/download_pretrained.py` | Downloads WordStylist EMA ckpt (Google Drive) + SD VAE (HuggingFace) |
| `external/Beyond-Memorization/download_iam_words.sh` | IAM login + download + extract + preprocess (if words.tgz not local) |
| `experiments_execution/slurm_modular/beyond_memorization.slurm` | SLURM job: partition rtx3080, gpu:1, cpus 8, 8h wall |

### Patches Applied to Official Repo

Two surgical patches to `regFrmTrnVariStyleMixOcr.py` (hardcoded author-machine paths):

1. **Line ~285** — `cropStyleDict_Numpy.pkl` loading:
   - **Before**: `open("/cluster/datastore/aniketag/.../cropStyleDict_Numpy.pkl")`
   - **After**: Lazy conditional load, only when `wrdChrWrStyl==1` (unused in our flow)

2. **Line ~351** — `__getitem__` image loading (vaeFromDict==0 branch):
   - **Before**: Looks up `charImgDict`/`wordImgDict` pointing to author's `/cluster/datastore/...`
   - **After**: Loads directly from `self.args.iam_path` (the preprocessed 64×256 dir)

---

## 3. Configuration File

**Location**: `configs/beyond_memorization.yaml`

```yaml
# Repository
repo_dir: external/Beyond-Memorization
entrypoint: regFrmTrnVariStyleMixOcr.py

# Pretrained assets
model_path: external/Beyond-Memorization/models/ema_ckpt.pt
stable_dif_path: external/Beyond-Memorization/stable-diffusion-v1-5
htr_model_path: ""

# Data
iam_path: data/IAM/words_64x256
gt_train: external/Beyond-Memorization/gt/gany.filter27

# Output
save_path: outputs/beyond_memorization

# Model architecture (fixed — matches paper)
img_height: 64
img_width: 256
channels: 4
emb_dim: 320
num_heads: 4
num_res_blocks: 1
max_chars: 25
style_classes: 339
noise_steps: 600

# Runtime
device: cuda:0
batch_size: 4
epochs: 1
num_workers: 4
latent: true
vae_from_dict: 0       # Encode real images with VAE on the fly
use_ocr: 0             # Skip OCR quality filter (no HTR model needed)
attention_visualization: 1  # Generate per-character attention maps
```

### Key Configuration Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `vae_from_dict: 0` | Encode on the fly | Avoids needing precomputed latent pickle files |
| `use_ocr: 0` | No OCR filtering | Saves all generated images; no HTR checkpoint needed |
| `attention_visualization: 1` | Generate maps | Core output for paper reproduction |
| `batch_size: 4` | Small batch | 16 images ÷ 4 = 4 batches; RTX 3080 10GB VRAM safe |

---

## 4. Data Preparation

### Source Data

- **IAM Handwriting Database** (words level): `archive/IAM dataset/words.tgz` (820 MB)
- Contains 115,320 handwritten word images at varying resolutions
- Organized in nested form-based directories: `a01/a01-000u/a01-000u-00-00.png`

### Ground Truth File

**File**: `external/Beyond-Memorization/gt/gany.filter27`
**Format**: `writerID,imageID word` (16 entries)

```
049,a03-034-01-03 booklet
049,a03-034-01-04 called
049,a03-034-01-07 Speaks
049,a03-034-03-08 three
049,a03-034-04-06 there
049,a03-034-04-08 little
049,a03-034-06-04 fallen
049,a03-034-07-05 economy
128,b06-019-00-07 Herr
128,b06-019-01-00 Strauss
128,b06-019-04-05 supply
128,b06-019-09-00 British
116,b05-032-01-05 before
116,b05-032-03-03 all
116,b05-032-07-02 took
116,b05-032-07-06 looms
```

**Writers**: 3 distinct (049, 128, 116) — the model generates words in one writer's style while injecting another writer's style from a specific character position.

### Preprocessing Pipeline

```bash
# Step 1: Extract words.tgz
tar -xzf "archive/IAM dataset/words.tgz" -C data/IAM/words_raw

# Step 2: Resize/pad to 64×256 RGB PNGs
python external/Beyond-Memorization/prepare_images.py \
    --iam_path data/IAM/words_raw \
    --save_dir data/IAM/words_64x256
```

**Processing logic** (`prepare_images.py`):
1. Open image → convert to RGB
2. Resize to height=64, keeping aspect ratio (`LANCZOS` resampling)
3. If width > 256: force resize to (256, 64)
4. If width ≤ 256: pad to (256, 64) with white background (`ImageOps.pad`)
5. Save as PNG

---

## 5. Pretrained Weights

### WordStylist EMA Checkpoint

| Property | Value |
|----------|-------|
| File | `external/Beyond-Memorization/models/ema_ckpt.pt` |
| Size | 162 MB |
| Source | Google Drive folder `15jdDCoYuWAohKW_OEjD2LXWce0pxM7ux` |
| Contents | EMA-averaged UNet state dict |
| Load method | `model.load_state_dict(torch.load(path), strict=False)` |

### Stable Diffusion v1.5 VAE

| Property | Value |
|----------|-------|
| Directory | `external/Beyond-Memorization/stable-diffusion-v1-5/vae/` |
| Size | ~335 MB (diffusion_pytorch_model.bin) |
| Source | HuggingFace: `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Load method | `AutoencoderKL.from_pretrained(sd_path, subfolder="vae")` |
| Function | Encodes 64×256 RGB → 8×32×4 latent; Decodes latent → image |
| Scale factor | latent = encode(img).sample() × 0.18215 |

### Download Command

```bash
python external/Beyond-Memorization/download_pretrained.py
# Downloads both ema_ckpt.pt (from Google Drive) and SD v1.5 vae/ (from HF)
```

---

## 6. Launcher: run_from_config.py

### Purpose

Translates the YAML config into CLI arguments for the official inference script, validates all prerequisites, and launches with the correct working directory.

### Usage

```bash
# Dry run (prints command, validates assets, does not execute)
python external/Beyond-Memorization/run_from_config.py \
    --config configs/beyond_memorization.yaml --dry-run

# Real run
python external/Beyond-Memorization/run_from_config.py \
    --config configs/beyond_memorization.yaml
```

### What it validates before launching

1. ✅ `repo_dir` exists
2. ✅ `model_path` (ema_ckpt.pt) exists
3. ✅ `stable_dif_path/vae/` directory exists
4. ✅ `gt_train` file exists
5. ✅ `iam_path` directory exists and contains `.png` files

### Generated CLI command

```bash
python regFrmTrnVariStyleMixOcr.py \
    --iam_path /abs/path/data/IAM/words_64x256 \
    --model_path /abs/path/external/Beyond-Memorization/models/ema_ckpt.pt \
    --stable_dif_path /abs/path/external/Beyond-Memorization/stable-diffusion-v1-5 \
    --gt_train /abs/path/external/Beyond-Memorization/gt/gany.filter27 \
    --save_path /abs/path/outputs/beyond_memorization/ \
    --loadPrevPath ./models/htr_model.pt \
    --batch_size 4 --epochs 1 --num_workers 4 \
    --emb_dim 320 --num_heads 4 --num_res_blocks 1 --channels 4 \
    --device cuda:0 --latent True --vaeFromDict 0 \
    --use_ocr 0 --attentionVisualition 1 --lang ENG
```

**Critical**: Script is executed with `cwd=repo_dir` because it uses relative imports (`from config import *`, `from unetVarStleMixExp4 import UNetModel`) and relative paths (`./gt/`, `./flags/`, `./logs/`).

---

## 7. SLURM Execution

### Job Script

**Location**: `experiments_execution/slurm_modular/beyond_memorization.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=beyond_mem
#SBATCH --output=.../experiments_execution/logs/beyond_memorization/output_%j.log
#SBATCH --error=.../experiments_execution/logs/beyond_memorization/error_%j.log
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx3080
```

### Submission

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
mkdir -p experiments_execution/logs/beyond_memorization
sbatch experiments_execution/slurm_modular/beyond_memorization.slurm
```

### Monitoring

```bash
# Check job status
squeue -u $USER

# Watch output log in real time
tail -f experiments_execution/logs/beyond_memorization/output_<JOBID>.log

# Check for errors
tail -20 experiments_execution/logs/beyond_memorization/error_<JOBID>.log
```

### Performance (Observed)

| Metric | Value |
|--------|-------|
| Job ID | 1709209 |
| Node | tg085 (RTX 3080, 10GB VRAM) |
| Runtime | 8 minutes 18 seconds |
| Peak RAM | 2.85 GB |
| Status | COMPLETED |

---

## 8. Inference Pipeline (Internal Flow)

### High-Level Execution Flow

```
main() [line 1556+]
  ├── Parse args → print
  ├── HTRNet(cnn_cfg, head_cfg, 54, ...) → net [line 1385]
  │   └── Only loads weights if file exists (skipped with use_ocr=0)
  ├── UNetModel(...) → unet [line 1434+]
  │   └── Load state dict from ema_ckpt.pt (strict=False)
  ├── EMA(unet) → ema_model
  ├── AutoencoderKL.from_pretrained(sd_path, "vae") → vae
  ├── createDataLoader1(args) → train_loader, style_classes, wr_dict, ...
  │   ├── Read gt/gany.filter27
  │   ├── Build writer dict {s_id: index}
  │   ├── IAMDataset(full_dict, iam_path, ...) → dataset
  │   └── CustomOrderedSampler → DataLoader(batch_size=4)
  └── FOR charLocation in [-1, 0, 1, 2, 3]:  [line 1556-1605]
      └── train(epoch=0, diffusion, net, unet, ema, ema_model, vae, ...,
               charLocation=charLocation)
```

### train() Function [line 918]

```
train(epoch, diffusion, net, model, ema, ema_model, vae, ...)
  ├── FOR batch in train_loader:
  │   ├── Extract: image_name, images, word_embedding, wr_id, label
  │   ├── images = vae.encode(images).latent_dist.sample() * 0.18215
  │   ├── Shuffle writers for style mixing
  │   └── sampling3(model, images, labels, ...)
  │       ├── x = torch.randn((n, 4, 8, 32))  [pure noise start]
  │       ├── FOR t in reversed(600 steps):
  │       │   ├── predicted_noise = model(x, t, text_features, y, ...)
  │       │   └── DDPM update: x = (x - β*noise) / sqrt(α) + σ*z
  │       ├── IF charLocation >= 0:
  │       │   ├── Extract attention maps from UNet middle block
  │       │   ├── Compute Xmax_c for target character
  │       │   ├── Create mask Mc from Xmax_c
  │       │   └── Second denoising pass with writer injection via mask
  │       └── RETURN denoised latent
  ├── decoded_image = vae.decode(latent / 0.18215)
  ├── Save generated image as PNG
  └── IF attentionVisualition:
      └── Save per-character attention map overlays
```

### Attention Map Extraction

The cross-attention maps come from the UNet's **middle block** (best character localization per paper):
- Shape: `(B, L=MAX_CHARS, H, W)` where H=8, W=32 (latent spatial dims)
- Each slice `A[:, c, :, :]` shows where character `c` attends in the image
- Character boundary: `Xmax_c = max(x-coordinates where A[:, c, :, :] > threshold)`
- Mask: `Mc[x > Xmax_c] = 1` (everything to the right of character c)

---

## 9. Output Structure

### Generated Images

**Naming format**: `{imageID}_{writerOriginal}_{writerShuffled}_{charLoc}__{word}_{epoch}.png`

Examples:
- `a03-034-01-03_049_116_New__booklet_0.png` (noChange: writer 049 style, shuffled with 116 but no mixing)
- `a03-034-01-03_049_116_2__booklet_0.png` (charIndex_2: style mixing from char position 2)

### Attention Map Overlays

**Naming format**: `{imageID}_{wrO}_{wrSh}_{charLoc}__{word}_{charIdx}_{char}_{epoch}_char_att0_val2_rollMins4.png`

Examples:
- `a03-034-01-03_049_116_New__booklet_3_k_0_char_att0_val2_rollMins4.png`
  - Image: `a03-034-01-03`, Writer: 049, Shuffled: 116, charLocation: -1 (New)
  - Word: "booklet", Character index: 3, Character: 'k'
  - The overlay shows the generated word with a heatmap highlighting where character 'k' is localized

### Output Statistics (Successful Run)

| Directory | Generated Words | Attention Maps | Total |
|-----------|----------------|----------------|-------|
| `noChange/` | 16 | 90 | 106 |
| `charIndex_0/` | 16 | 90 | 106 |
| `charIndex_1/` | 16 | 90 | 106 |
| `charIndex_2/` | 16 | 90 | 106 |
| `charIndex_3/` | 16 | 90 | 106 |
| **TOTAL** | **80** | **450** | **530** |

The 90 attention maps per directory = 16 words × ~5.6 characters average (each character in each word gets its own attention map).

---

## 10. Reproducing From Scratch

### Complete Step-by-Step

```bash
# 0. Navigate to workspace
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate

# 1. Download pretrained weights (if not already done)
python external/Beyond-Memorization/download_pretrained.py
# Downloads: models/ema_ckpt.pt (Google Drive) + stable-diffusion-v1-5/vae/ (HF)

# 2. Prepare IAM word crops (if not already done)
mkdir -p data/IAM/words_raw
tar -xzf "archive/IAM dataset/words.tgz" -C data/IAM/words_raw
python external/Beyond-Memorization/prepare_images.py \
    --iam_path data/IAM/words_raw \
    --save_dir data/IAM/words_64x256

# 3. Validate prerequisites (dry run)
python external/Beyond-Memorization/run_from_config.py \
    --config configs/beyond_memorization.yaml --dry-run
# Expected: "Launching Beyond-Memorization inference:" + full command

# 4. Submit SLURM job
mkdir -p experiments_execution/logs/beyond_memorization
sbatch experiments_execution/slurm_modular/beyond_memorization.slurm
# Expected runtime: ~8 minutes on RTX 3080

# 5. Verify outputs
find outputs/beyond_memorization -name "*.png" | wc -l
# Expected: 530 (80 word images + 450 attention maps)
```

### If Running Locally (non-SLURM)

```bash
python external/Beyond-Memorization/run_from_config.py \
    --config configs/beyond_memorization.yaml
```

---

## 11. Troubleshooting

### Error: `No such file or directory: '/cluster/datastore/aniketag/...'`

**Cause**: Author-machine hardcoded paths in the repo that were not patched.
**Fix**: Apply the two patches described in Section 2 (cropStyleDict lazy load + iam_path direct loading in __getitem__).

### Error: `FileNotFoundError` for images during inference

**Cause**: `data/IAM/words_64x256/` missing the referenced word images.
**Fix**: Ensure the 16 gt-referenced images are preprocessed:
```bash
# Quick check
for id in a03-034-01-03 a03-034-01-04 b06-019-00-07 b05-032-01-05; do
    ls "data/IAM/words_64x256/${id}.png"
done
```

### Error: `../../.venv/bin/activate: No such file or directory`

**Cause**: SLURM script sources venv with relative path before `cd` to workspace.
**Fix**: Use absolute path: `source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate` (already fixed).

### Low VRAM / OOM

**Fix**: Reduce batch_size in `configs/beyond_memorization.yaml` (try 2 or 1).

### Want more/different words

**Modify**: `external/Beyond-Memorization/gt/gany.filter27` — add more lines in format `writerID,imageID word`. Ensure the referenced word images exist in `data/IAM/words_64x256/`.

---

## 12. Extending the Experiment

### Adding More Writers/Words

1. Find writer IDs and image IDs from `external/Beyond-Memorization/htr/utils/words.txt` (115,320 entries with format: `imageID ok/err graylevel x y w h POS word`)
2. Add entries to `gt/gany.filter27` in format `writerID,imageID word`
3. Ensure the word images are in `data/IAM/words_64x256/`
4. Re-run the SLURM job

### Using a Different Partition

Edit `experiments_execution/slurm_modular/beyond_memorization.slurm`:
```bash
#SBATCH --partition=a100    # or v100
```

### Disabling Attention Map Generation (faster)

In `configs/beyond_memorization.yaml`:
```yaml
attention_visualization: 0
```

### Running Only Specific charLocations

Requires modifying `regFrmTrnVariStyleMixOcr.py` lines 1556-1605 (the main loop is hardcoded). Comment out unwanted iterations.

---

## 13. Understanding the Attention Maps

### What They Show

Each attention map PNG shows:
- **Background**: The generated handwritten word (after VAE decode)
- **Overlay**: A heatmap (yellow/orange/red) from the UNet middle-block cross-attention
- **Meaning**: Where the model "looks" when generating that specific character

### Interpretation (Paper Fig. 5/6)

- Characters are localized left-to-right (attention peak moves rightward)
- The `Xmax_c` boundary determines where style mixing occurs
- Comparing the SAME word across `noChange/` vs `charIndex_*/` shows:
  - Same characters on the LEFT of the mixing point (original writer style)
  - Different characters on the RIGHT (injected writer style)
- This demonstrates the model has learned character-level spatial structure without explicit segmentation training

### Naming Convention Decoded

```
a03-034-01-03_049_116_New__booklet_3_k_0_char_att0_val2_rollMins4.png
│             │   │   │    │       │ │ │
│             │   │   │    │       │ │ └── epoch
│             │   │   │    │       │ └── character letter
│             │   │   │    │       └── character index in word
│             │   │   │    └── word label
│             │   │   └── charLocation indicator (New=-1, 0/1/2/3)
│             │   └── shuffled writer ID
│             └── original writer ID
└── IAM image ID
```

---

## 14. Dependencies

### Python Environment

All dependencies satisfied by the workspace venv (`.venv/`):
- `torch >= 2.4.1` (with CUDA)
- `diffusers >= 0.35.1`
- `transformers`
- `einops`
- `timm`
- `Pillow >= 10.0` (requires `Image.LANCZOS` not deprecated `ANTIALIAS`)
- `omegaconf` (for YAML config loading)
- `gdown >= 5.0` (for Google Drive downloads)
- `huggingface_hub` (for SD VAE download)
- `numpy`, `scipy`, `matplotlib`, `tqdm`

### Hardware Requirements

- **GPU**: Any CUDA GPU with ≥ 8GB VRAM (tested on RTX 3080 10GB)
- **RAM**: ~3 GB system RAM
- **Disk**: ~2 GB for weights + outputs
- **Time**: ~8 minutes for 16 words × 5 charLocations on RTX 3080

---

## 15. Relation to HTR-Pipeline Project

This Beyond-Memorization reproduction is **complementary** to the main HTR-Pipeline work:

| Aspect | HTR-Pipeline (Main) | Beyond-Memorization |
|--------|--------------------|--------------------|
| Task | Handwriting recognition (HTR) | Handwriting generation + style analysis |
| Model | ViT-RGTS v2 with registers | WordStylist latent diffusion UNet |
| Attention | Self-attention (encoder→CTC) | Cross-attention (text→image) |
| Purpose | Recognize text from images | Generate images from text + analyze style |
| Data level | Line images (128×1024) | Word images (64×256) |
| Registers | Learnable register tokens for global info | N/A (different architecture) |

The attention maps from both systems provide complementary insights:
- **HTR self-attention**: Shows which image patches the recognizer attends to for each output character
- **Beyond-Memorization cross-attention**: Shows where the generator "places" each character in the generated image

Both demonstrate that transformer models learn meaningful character-level spatial structure.

---

## 16. Verified Run Log (Job 1709209)

```
Started at: Fri Jun 19 04:14:30 PM CEST 2026
Node: tg085  GPU: NVIDIA GeForce RTX 3080, 10240 MiB

Arguments loaded from config
Latent is true - Working on latent space
3 writer styles detected (049, 128, 116)
character vocabulary size: 53
DataLoader: 4 batches

charLocation=-1 → noChange/ (16 images + 90 attention maps) ✓
charLocation=0  → charIndex_0/ (16 images + 90 attention maps) ✓
charLocation=1  → charIndex_1/ (16 images + 90 attention maps) ✓
charLocation=2  → charIndex_2/ (16 images + 90 attention maps) ✓
charLocation=3  → charIndex_3/ (16 images + 90 attention maps) ✓

Completed at: Fri Jun 19 04:22:48 PM CEST 2026
Elapsed: 00:08:18
Exit: COMPLETED (0)
```

---

*Document created: 2026-06-19*
*Implementation status: FULLY WORKING — end-to-end reproduction verified*
