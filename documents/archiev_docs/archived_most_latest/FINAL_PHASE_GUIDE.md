# Final Phase Guide: Attention Maps, Experiments & Repo Cleanup

## Table of Contents

1. [Attention Map System — Complete Reference](#1-attention-map-system)
2. [Running Attention Visualizations — Step by Step](#2-running-attention-visualizations)
3. [Experiment Status & What to Run Next](#3-experiment-status)
4. [Repo Cleanup Guide](#4-repo-cleanup-guide)

---

## 1. Attention Map System

### 1.1 Architecture Overview

The attention extraction system has three layers:

```
┌─────────────────────────────────────────────────────────┐
│ scripts/attention_visualization.py                       │
│   Main script: generates 5 publication-quality figures   │
│   Loads models, runs forward_explain, plots results      │
├─────────────────────────────────────────────────────────┤
│ models.py → forward_explain()                            │
│   Per-architecture attention extraction:                 │
│   - ViTRGTSBackbone.forward_explain()   (vit_rgts)      │
│   - TorchVisionViTBackbone.forward_explain() (tv_vit)   │
│   - TrOCREncoderBackbone.forward_explain()  (trocr)     │
│   All return: (logits, reg_tokens, attn_maps,            │
│                token_norms, grid_size)                   │
├─────────────────────────────────────────────────────────┤
│ utils/attention_extractor.py                             │
│   Utility class for analysis (entropy, register stats)   │
│ utils/visualizer.py                                      │
│   Hook-based CNN feature visualization (optional)        │
└─────────────────────────────────────────────────────────┘
```

### 1.2 How `forward_explain()` Works (ViT-RGTS)

This is the core of the attention system. Located in `models.py` at `ViTRGTSBackbone.forward_explain()` (line 383).

**What it does**: Re-implements the transformer forward pass manually to extract per-layer, per-head attention weights that PyTorch's `nn.TransformerEncoder` normally discards.

**Step-by-step flow:**

```python
# 1. Patch embedding (CNN stem or linear projection)
x = self.cnn_stem(input_image)          # [B, D, 1, Wp]
patch_tokens = x.squeeze(2).transpose(1,2)  # [B, Wp, D]

# 2. Prepend register tokens
reg_tokens = self.register_tokens.expand(B, -1, -1)  # [B, R, D]
tokens = cat([reg_tokens, patch_tokens])  # [B, R+Wp, D] = [B, S, D]

# 3. Add positional encoding
tokens = tokens + self.pos_embed[:, :S, :]

# 4. Manual transformer loop (extracting attention)
for each layer:
    normed = layer.norm1(x_tokens)       # Pre-norm
    Q, K, V = linear(normed).chunk(3)    # QKV projection
    Q = reshape to [B, H, S, D_h]       # Multi-head reshape
    
    attn_weights = softmax(Q @ K^T / √d_h)  # [B, H, S, S]  ← CAPTURED
    attn_maps.append(attn_weights)
    
    attn_output = attn_weights @ V       # Apply attention
    x_tokens = x_tokens + out_proj(attn_output)  # Residual
    x_tokens = x_tokens + layer.mlp(layer.norm2(x_tokens))  # FFN

# 5. Return everything
token_norms = encoded.norm(dim=-1)  # [B, S] L2 norms
return seq_tokens, reg_tokens, attn_maps, token_norms, grid_size
```

**Output format:**

| Return Value | Shape | Description |
|---|---|---|
| `seq_tokens` | `[T, B, D]` | Patch token sequence (input to CTC head) |
| `reg_tokens` | `[B, R, D]` | Register token embeddings |
| `attn_maps` | `List[L]` of `[B, H, S, S]` | Per-layer, per-head attention matrices |
| `token_norms` | `[B, S]` | L2 norm of each token at final layer |
| `grid_size` | `(Hp, Wp)` | Spatial grid (Hp=1 with CNN stem, Wp=128) |

Where: `S = R + Hp×Wp` (registers + patches), `H` = num_heads (8), `L` = depth (6).

### 1.3 Understanding the Attention Matrix `[B, H, S, S]`

The attention matrix `attn_maps[layer][b, h, i, j]` answers: **"How much does token i attend to token j at layer `layer`, head `h`?"**

Token ordering in the sequence dimension `S`:

```
Index:   0    1    ...  R-1  |  R    R+1   ...  R+Wp-1
         ┌──────────────┐    |  ┌────────────────────┐
Token:   reg_0  reg_1 ... reg_{R-1}  patch_0  patch_1 ... patch_{Wp-1}
         └──────────────┘    |  └────────────────────┘
         Register tokens     |  Patch tokens (left→right of image)
```

**Key sub-matrices:**

| Slice | Shape | Meaning |
|---|---|---|
| `attn[:,:, :R, R:]` | `[B,H,R,Wp]` | Registers → Patches (what registers look at) |
| `attn[:,:, R:, :R]` | `[B,H,Wp,R]` | Patches → Registers (information offloading) |
| `attn[:,:, R:, R:]` | `[B,H,Wp,Wp]` | Patch ↔ Patch (spatial self-attention) |
| `attn[:,:, :R, :R]` | `[B,H,R,R]` | Register ↔ Register (inter-register comm.) |

### 1.4 The 5 Figures — What Each Shows

**Fig 1: Register Impact Grid** (`plot_register_impact_grid`)
- Grid: rows = sample images, columns = register counts (0, 4, 8, 16)
- Shows global attention heatmap (last layer, head-averaged, patch-to-patch) overlaid on input
- Demonstrates how register tokens absorb high-attention artifacts (Darcet et al., 2024)
- Attention: `avg_attn[0, R:, R:].mean(dim=0)` → 1D profile → interpolate to image width → crop to text

**Fig 2: Character Attention Grid** (`plot_char_attention_grid`)
- Grid: rows = words, columns = decoded characters
- For each character `c` at CTC timestep `t_c`: attention row `A[R+t_c, R:R+Wp]`
- Shows which spatial patches each character "looks at" during recognition
- $X^{max}_c$ (weighted centroid) marked with white dashed line
- Normalization: percentile clip + adaptive gamma correction
- Paper reference: "Beyond Memorization" Fig. 5

**Fig 3: Token Norm Artifact Map** (`plot_token_norm_comparison`)
- Grid: rows = images, columns = register counts
- Shows L2 norm of each patch token (spatial distribution)
- Without registers: high-norm outlier tokens appear in low-information regions
- With registers: norms become more uniform (artifacts absorbed by registers)

**Fig 4: Layer-wise Attention Flow** (`plot_layerwise_attention`)
- Grid: rows = transformer layers (1→6), columns = individual heads + head-averaged
- Shows how attention patterns evolve from shallow (local) to deep (global)
- Single image, one model (typically best 4-register run)

**Fig 5: Per-Character Grad-CAM** (`plot_gradcam` + `compute_gradcam`)
- Grid: rows = words, columns = decoded characters
- True gradient-based attribution (not just attention weights)
- For each character: backpropagate from its specific CTC logit
- `importance_c[t] = ReLU(Σ_d ∂logit(c)/∂A_{t,d} · A_{t,d})`
- Requires `retain_graph=True` for multiple backward passes

### 1.5 Supported Architectures

| Architecture | `forward_explain` | Location in `models.py` | Notes |
|---|---|---|---|
| `vit_rgts` | ✓ Full support | Line 383 | Manual QKV decomposition, full attention extraction |
| `torchvision_vit` | ✓ Full support | Line 710 | Uses `need_weights=True` on `nn.MultiheadAttention` |
| `trocr` | ✓ Full support | Line 985 | Hooks into HuggingFace encoder layers, `reg_tokens=None` |
| `cnn_rnn` | ✗ Not supported | — | No transformer → no attention maps (use `utils/visualizer.py` for CNN feature maps) |

### 1.6 Critical Bug Fix Applied

The default run IDs in `attention_visualization.py` were updated from non-existent runs (run_53, 54, 55, 58 from v1 sweep) to actual v2 sweep runs:

| Registers | Old (broken) | New (verified) | Val CER |
|---|---|---|---|
| 0 | run_55 ✗ | run_76 ✓ | 0.0442 |
| 4 | run_58 ✗ | run_84 ✓ | 0.0433 |
| 8 | run_53 ✗ | run_63 ✓ | 0.0425 |
| 16 | run_54 ✗ | run_71 ✓ | 0.0419 |

---

## 2. Running Attention Visualizations — Step by Step

### 2.1 Prerequisites

```bash
# Verify models exist
ls saved_models/experiments/run_{76,84,63,71}/model.pt

# Verify sample images
ls notebook/sample_images/a01-038-12.png

# Verify charset
python3 -c "import numpy as np; print(len(np.load('saved_models/experiments/classes.npy', allow_pickle=True)))"
# Expected: 79
```

### 2.2 Generate All 5 Figures (CPU — login node)

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate

# Output to woody (hpchome is over quota)
python scripts/attention_visualization.py \
    --device cpu \
    --out_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/attention_visualizations
```

**Output** (verified working — 12 files, ~10 MB total):
```
fig1_register_impact_grid.pdf/.png      — Register comparison across images
fig2_character_attention_grid.pdf/.png   — Per-char attention (4-reg model)
fig2b_character_attention_grid_baseline.pdf/.png — Per-char attention (0-reg)
fig3_token_norm_artifacts.pdf/.png       — Token norm spatial distribution
fig4_layerwise_attention.pdf/.png        — Layer 1→6 attention evolution
fig5_gradcam.pdf/.png                    — Per-char gradient attribution
```

### 2.3 Custom Images

```bash
# Use your own IAM images
python scripts/attention_visualization.py \
    --device cpu \
    --images data/IAM/processed_lines/test/a01-000u-00.png \
             data/IAM/processed_lines/test/a01-003-00.png \
    --out_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/custom_attention
```

### 2.4 Specific Models / Runs

```bash
# Only 0-reg and 4-reg comparison
python scripts/attention_visualization.py \
    --runs run_76 run_84 \
    --device cpu \
    --out_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/attention_0vs4

# Character grid for specific run
python scripts/attention_visualization.py \
    --char_grid_run run_63 \
    --layer_run run_63 \
    --device cpu \
    --out_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/attention_8reg
```

### 2.5 GPU (SLURM Job — Faster)

```bash
# Create a quick SLURM job
cat > /tmp/attn_viz.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=attn_viz
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/attn_viz_%j.out

source /home/hpc/iwi5/iwi5369h/HTR-Pipeline/.venv/bin/activate
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

python scripts/attention_visualization.py \
    --device cuda:0 \
    --out_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/attention_visualizations
EOF

sbatch /tmp/attn_viz.slurm
```

### 2.6 Programmatic Usage (Notebooks / Custom Scripts)

```python
import sys, json, torch, numpy as np
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, '/home/hpc/iwi5/iwi5369h/HTR-Pipeline')

from models import HTRNet
from utils.preprocessing import load_image, preprocess
from utils.attention_extractor import AttentionExtractor

# Load model
EXPERIMENTS = Path('saved_models/experiments')
charset = list(np.load(str(EXPERIMENTS / 'classes.npy'), allow_pickle=True))
cfg = json.load(open(EXPERIMENTS / 'run_84' / 'config.json'))
model = HTRNet(SimpleNamespace(**cfg['arch']), len(charset) + 1)
model.load_state_dict(torch.load(EXPERIMENTS / 'run_84' / 'model.pt', map_location='cpu', weights_only=False), strict=False)
model.eval()

# Prepare image
img = preprocess(load_image('notebook/sample_images/a01-038-12.png'), (128, 1024))
tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

# Extract attention
with torch.no_grad():
    logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(tensor)

# attn_maps[layer][batch, head, query_token, key_token]
# Layer 5 (last), head-averaged, patch-to-patch attention
R = reg_tokens.shape[1]  # 4
Hp, Wp = grid             # (1, 128)
last_attn = attn_maps[-1].mean(dim=1)  # [B, S, S] head-averaged
patch_attn = last_attn[0, R:, R:]       # [Wp, Wp] patch self-attention
spatial_profile = patch_attn.mean(dim=0).numpy()  # [Wp] — which columns are attended

# Use AttentionExtractor for analysis
extractor = AttentionExtractor(model)
result = extractor.extract_from_forward(tensor)
reg_stats = extractor.analyze_register_attention(result['attention_maps'], R)
```

---

## 3. Experiment Status & What to Run Next

### 3.1 Completed Experiments

| Runs | Architecture | Description | Best Test CER |
|---|---|---|---|
| run_73 | cnn_rnn | CNN-RNN baseline | **5.02%** |
| run_60–71 | vit_rgts | Register sweep (0–16 regs), v1 config | 5.97–6.23% |
| run_76–92 | vit_rgts | Register sweep (0–16 regs), v2 config | 6.27–6.46% |
| run_72, 96, 98 | trocr | TrOCR encoder + CTC | 15.5–66.5% |
| run_97, 99–101 | torchvision_vit | Pretrained ViT-B/16 + CTC | 26–77% |

**Key findings:**
- CNN-RNN (run_73) is the strongest single model: 5.02% test CER
- ViT-RGTS with 7 registers (run_68) achieves best ViT result: 5.97%
- Register tokens consistently improve ViT-RGTS (0 regs = 6.43% → 7 regs = 5.97%)
- Pretrained ViT/TrOCR underperform significantly (fine-tuning not converged)

### 3.2 Remaining Experiments to Consider

**A. Synthetic pretraining (highest priority — infrastructure ready)**
```bash
# Step 1: Extract LMDB to IAM format (once, ~3-6 hours)
sbatch scripts/extract_synthetic.slurm

# Step 2: Pretrain on synthetic data
python scripts/trainer.py configs/config.yaml configs/baseline.yaml \
    data.mode=synthetic train.num_epochs=10

# Step 3: Fine-tune on IAM (resume from checkpoint)
python scripts/trainer.py configs/config.yaml configs/baseline.yaml \
    resume=saved_models/experiments/run_N/model.pt train.num_epochs=80
```

**B. Best-model attention analysis (for paper)**
```bash
# Generate final publication figures with all 4 register counts
python scripts/attention_visualization.py \
    --runs run_76 run_84 run_63 run_71 \
    --device cpu \
    --out_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/final_attention_figures
```

**C. Additional attention analysis on best CNN-RNN**
Note: CNN-RNN does NOT have transformer attention maps. Use `utils/visualizer.py` hook-based CNN feature maps instead.

### 3.3 Key Run Reference

| Purpose | Run ID | Regs | Test CER | Use For |
|---|---|---|---|---|
| CNN-RNN baseline | run_73 | N/A | 5.02% | Best overall CER |
| ViT-RGTS 0-reg | run_76 | 0 | 6.43% | Attention baseline (no registers) |
| ViT-RGTS 4-reg | run_84 | 4 | 6.40% | Character attention analysis |
| ViT-RGTS 7-reg (best) | run_68 | 7 | 5.97% | Best ViT result |
| ViT-RGTS 8-reg | run_63 | 8 | 6.23% | Attention with many registers |
| ViT-RGTS 16-reg | run_71 | 16 | 6.09% | Max register comparison |

---

## 4. Repo Cleanup Guide

### 4.1 Current State

The repo has accumulated significant development artifacts:

| Directory | Size | Status |
|---|---|---|
| `saved_models/` | 107 GB | 37 experiment runs — **KEEP** (or selectively prune) |
| `output/` | 1.9 GB | Generated visualizations — can regenerate |
| `visualizations/` | 708 MB | Old visualization outputs — can regenerate |
| `data/` | 663 MB | IAM dataset — **KEEP** |
| `notebook/` | 110 MB | Jupyter notebooks — most are exploratory |
| `backup/` | 45 MB | Old files — can archive or remove |
| `papers/` | 24 MB | Reference papers — keep if needed |
| `problem statement/` | 26 MB | Project spec — keep separately |
| `documents/` | 1.8 MB | 32 markdown docs — consolidate |
| `helper/` | 707 KB | Helper/instruction files — archive |

### 4.2 Recommended Final Repo Structure

```
HTR-Pipeline/
├── configs/                          # KEEP — all config files
│   ├── config.yaml
│   ├── baseline.yaml
│   ├── baseline_vit_rgts_v2.yaml
│   ├── torchvision_vit.yaml
│   └── trocr.yaml
├── data/                             # KEEP — IAM dataset
│   └── IAM/processed_lines/
├── models.py                         # KEEP — all model architectures
├── requirements.txt                  # KEEP
├── README.md                         # UPDATE — final project README
├── scripts/
│   ├── trainer.py                    # KEEP — training pipeline
│   ├── attention_visualization.py    # KEEP — publication figures
│   ├── extract_synthetic_to_iam_format.py  # KEEP — synthetic data tool
│   ├── extract_synthetic.slurm       # KEEP — extraction job
│   ├── preprocessing/
│   │   └── prepare_iam.py            # KEEP
│   └── postprocessing/               # REVIEW — keep only working scripts
│       ├── attention_viz/            # KEEP core.py, generate_all.py
│       ├── single_model/            # KEEP evaluate.py, gradcam, char_attention
│       └── comparative/             # KEEP compare scripts
├── utils/
│   ├── htr_dataset.py               # KEEP
│   ├── preprocessing.py             # KEEP
│   ├── transforms.py                # KEEP
│   ├── metrics.py                   # KEEP
│   ├── finetuning.py                # KEEP
│   ├── attention_extractor.py       # KEEP
│   └── visualizer.py                # KEEP
├── saved_models/experiments/         # KEEP key runs, prune duplicates
├── synthetic_data_generation/        # KEEP — all 6 pipeline scripts
├── notebook/
│   └── sample_images/               # KEEP — needed by attention_viz
├── experiments_execution/            # KEEP — SLURM submission scripts
└── documents/
    └── FINAL_DOCUMENTATION.md        # This file (consolidation)
```

### 4.3 Step-by-Step Cleanup Procedure

**IMPORTANT**: Do these steps carefully. Do NOT delete anything without verifying first.

#### Step 1: Back up everything to woody first

```bash
# Create a full backup on woody before any cleanup
mkdir -p /home/woody/iwi5/iwi5369h/projects/htr_backup
rsync -av --exclude='.venv' --exclude='saved_models' --exclude='data' \
    /home/hpc/iwi5/iwi5369h/HTR-Pipeline/ \
    /home/woody/iwi5/iwi5369h/projects/htr_backup/repo_snapshot_$(date +%Y%m%d)/
```

#### Step 2: Remove regeneratable output files

These can all be regenerated by running the visualization scripts:

```bash
# Move to woody instead of deleting (safe)
mv /home/hpc/iwi5/iwi5369h/HTR-Pipeline/output \
   /home/woody/iwi5/iwi5369h/projects/htr_backup/output_archive
mv /home/hpc/iwi5/iwi5369h/HTR-Pipeline/visualizations \
   /home/woody/iwi5/iwi5369h/projects/htr_backup/visualizations_archive
```

#### Step 3: Archive old notebooks

```bash
# Keep only sample_images/ in notebook/
mkdir -p /home/woody/iwi5/iwi5369h/projects/htr_backup/notebooks_archive
mv /home/hpc/iwi5/iwi5369h/HTR-Pipeline/notebook/*.ipynb \
   /home/woody/iwi5/iwi5369h/projects/htr_backup/notebooks_archive/
mv /home/hpc/iwi5/iwi5369h/HTR-Pipeline/notebook/*.py \
   /home/woody/iwi5/iwi5369h/projects/htr_backup/notebooks_archive/
# Keep notebook/sample_images/ in place
```

#### Step 4: Consolidate documents

```bash
# Archive all old development docs
mkdir -p /home/woody/iwi5/iwi5369h/projects/htr_backup/docs_archive
mv /home/hpc/iwi5/iwi5369h/HTR-Pipeline/documents/depricated \
   /home/woody/iwi5/iwi5369h/projects/htr_backup/docs_archive/
mv /home/hpc/iwi5/iwi5369h/HTR-Pipeline/documents/latest \
   /home/woody/iwi5/iwi5369h/projects/htr_backup/docs_archive/
mv /home/hpc/iwi5/iwi5369h/HTR-Pipeline/backup \
   /home/woody/iwi5/iwi5369h/projects/htr_backup/backup_archive
mv /home/hpc/iwi5/iwi5369h/HTR-Pipeline/helper \
   /home/woody/iwi5/iwi5369h/projects/htr_backup/helper_archive

# Keep in documents/: only this file + SYNTHETIC_DATA_INTEGRATION.md + TRAINING_PIPELINE_DOCUMENTATION.md
```

#### Step 5: Prune duplicate experiment runs

You have TWO register sweep sets (run_60–71 and run_76–92). The first set has better results. Consider archiving the weaker set:

```bash
# Archive v2 sweep (run_76-92) — v1 sweep (run_60-71) has better CER
mkdir -p /home/woody/iwi5/iwi5369h/projects/htr_backup/experiments_v2
for run in run_76 run_77 run_78 run_79 run_80 run_81 run_82 run_83 run_84 run_85 run_86 run_87 run_88 run_89 run_90 run_91 run_92; do
    mv saved_models/experiments/$run /home/woody/iwi5/iwi5369h/projects/htr_backup/experiments_v2/
done
```

**WARNING**: run_76 (0-reg baseline) is used by `attention_visualization.py`. If you archive v2 runs, update the DEFAULT_RUNS in the script to use v1 equivalents (run_55 etc. — verify they exist, or keep run_76).

**Safer alternative**: Keep all runs but move model.pt checkpoints of duplicate runs to woody:

```bash
# Keep config.json and results.csv (tiny), move model.pt (~3GB each) to woody
for run in run_77 run_78 run_79 run_80 run_81 run_85 run_86 run_88 run_89 run_90 run_91 run_92; do
    if [[ -f saved_models/experiments/$run/model.pt ]]; then
        mv saved_models/experiments/$run/model.pt \
           /home/woody/iwi5/iwi5369h/projects/htr_backup/model_checkpoints/${run}_model.pt
    fi
done
```

#### Step 6: Clean postprocessing scripts

Many scripts in `scripts/postprocessing/` are exploratory duplicates. Keep only the ones that work:

```
scripts/postprocessing/
├── attention_viz/
│   ├── core.py              # KEEP — shared utilities
│   ├── generate_all.py      # KEEP — batch generation
│   └── (others)             # ARCHIVE — individual components
├── single_model/
│   ├── evaluate.py           # KEEP — model evaluation
│   ├── gradcam_vit_rgts.py   # KEEP — GradCAM
│   ├── character_attention_grid.py  # KEEP
│   └── (others)             # ARCHIVE
└── comparative/
    ├── compare_attention_across_models.py  # KEEP
    ├── plot_training_metrics.py            # KEEP
    └── (others)             # ARCHIVE
```

#### Step 7: Verify nothing is broken

```bash
# After cleanup, verify core pipeline still works
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
source .venv/bin/activate

# 1. Config loads
python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/config.yaml'); print('Config OK')"

# 2. Dataset loads
python -c "
from utils.htr_dataset import HTRDataset
ds = HTRDataset('./data/IAM/processed_lines', 'train', fixed_size=(128,1024))
print(f'Dataset OK: {len(ds)} samples')
"

# 3. Model loads and forward_explain works
python -c "
import json, torch, numpy as np
from types import SimpleNamespace
from models import HTRNet
from utils.preprocessing import load_image, preprocess

charset = list(np.load('saved_models/experiments/classes.npy', allow_pickle=True))
cfg = json.load(open('saved_models/experiments/run_84/config.json'))
model = HTRNet(SimpleNamespace(**cfg['arch']), len(charset)+1)
model.load_state_dict(torch.load('saved_models/experiments/run_84/model.pt', map_location='cpu', weights_only=False), strict=False)
model.eval()
img = preprocess(load_image('notebook/sample_images/a01-038-12.png'), (128,1024))
t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
with torch.no_grad():
    logits, reg, attn, norms, grid = model.forward_explain(t)
print(f'forward_explain OK: {len(attn)} layers, grid={grid}')
"

# 4. Attention visualization generates figures
python scripts/attention_visualization.py --device cpu \
    --out_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/verify_test
```

### 4.4 Estimated Space Recovery

| Action | Space Freed |
|---|---|
| Move `output/` to woody | ~1.9 GB |
| Move `visualizations/` to woody | ~708 MB |
| Move `notebook/*.ipynb` to woody | ~110 MB |
| Archive duplicate run model.pt files | ~30-60 GB |
| Remove `backup/` | ~45 MB |
| **Total potential** | **~33-63 GB** |

Current hpchome usage: 171 GB / 105 GB (over quota). Moving output and archiving duplicate model checkpoints should bring you back under quota.

### 4.5 Files That MUST NOT Be Deleted

| File/Directory | Why |
|---|---|
| `models.py` | All model architectures including `forward_explain` |
| `scripts/trainer.py` | Training pipeline with dual-mode data loading |
| `scripts/attention_visualization.py` | The main attention figure generator |
| `utils/*.py` | All utility modules (dataset, preprocessing, transforms, metrics, finetuning, attention_extractor, visualizer) |
| `configs/*.yaml` | All configuration files |
| `data/IAM/processed_lines/` | The IAM dataset |
| `saved_models/experiments/classes.npy` | Character set used by all models |
| `saved_models/experiments/run_{73,76,84,63,68,71}/` | Key experiment runs (CNN baseline + register sweep reference points) |
| `notebook/sample_images/` | Sample images used by attention_visualization.py |
| `synthetic_data_generation/` | All 6 pipeline scripts + documentation |
| `requirements.txt` | Python dependencies |

---

## Appendix A: Attention Map Interpretation Guide

### Reading Fig 2 (Character Attention Grid)

Each cell `A_c` shows where the model "looks" when predicting character `c`:
- **Bright regions** = high attention (model focuses here)
- **White dashed line** = $X^{max}_c$ = weighted centroid of attention
- Characters should attend to their correct spatial position in the handwriting
- Adjacent characters may show overlapping attention (context-dependent recognition)
- Registers absorb "junk" attention that would otherwise form artifacts on low-info patches

### Reading Fig 3 (Token Norms)

- **Without registers (0-reg)**: Some tokens develop abnormally high norms → these are "artifact" tokens
- **With registers**: Norms become more uniform → registers absorb the function that artifact tokens served
- The σ (standard deviation) statistic quantifies this: lower σ = more uniform = better

### Reading Fig 5 (Grad-CAM)

- Each cell shows gradient-weighted activation for one character
- Unlike attention (which shows where the model looks), Grad-CAM shows what image regions **caused** the prediction
- Grad-CAM and attention often agree but can differ when the model uses indirect reasoning

## Appendix B: Quick Command Reference

```bash
# ===== ATTENTION VISUALIZATION =====
# Generate all 5 figures (CPU, ~2 min)
python scripts/attention_visualization.py --device cpu \
    --out_dir /home/woody/iwi5/iwi5369h/projects/synth_htr/attention_figures

# ===== TRAINING (IAM mode — existing) =====
python scripts/trainer.py configs/config.yaml configs/baseline.yaml

# ===== TRAINING (synthetic mode — new) =====
python scripts/trainer.py configs/config.yaml configs/baseline.yaml data.mode=synthetic

# ===== EXTRACT SYNTHETIC DATA =====
sbatch scripts/extract_synthetic.slurm

# ===== EVALUATE A MODEL =====
python scripts/postprocessing/single_model/evaluate.py \
    --run run_84 --device cpu
```
