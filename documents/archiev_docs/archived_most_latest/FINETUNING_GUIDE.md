# Fine-Tuning Pretrained Models for HTR

## Overview

This module implements **advanced fine-tuning** for two pretrained backbones on IAM handwriting:

| Model | Source | Params | Domain Gap | Strategy |
|-------|--------|--------|------------|----------|
| **TorchVision ViT-B/16** | ImageNet-1K | 86M | High (natural images → handwriting) | Aggressive LLRD + gradual unfreeze |
| **TrOCR Base** | Handwriting data | 85M | Low (already handwriting-trained) | Conservative LLRD + gradual unfreeze |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HTRNet (Unified Interface)                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────────┐   │
│  │  TorchVision ViT-B   │    │  TrOCR Encoder (DeiT)    │   │
│  │  ─────────────────   │    │  ─────────────────────   │   │
│  │  gray_to_rgb (1→3)   │    │  gray_to_rgb (1→3)       │   │
│  │  ImageNet norm        │    │  Aspect-ratio resize+pad │   │
│  │  Patch embed (16×16)  │    │  TrOCR normalization     │   │
│  │  + Register tokens    │    │  Patch embed (16×16)     │   │
│  │  Pos embed interp     │    │  12× Transformer blocks  │   │
│  │  12× Transformer      │    │  Height pool + upsample  │   │
│  │  blocks               │    │                          │   │
│  └──────────┬───────────┘    └────────────┬─────────────┘   │
│             │                              │                  │
│             └──────────┬───────────────────┘                  │
│                        ▼                                      │
│              [T, B, D] seq_tokens                             │
│                        ▼                                      │
│             ┌──────────────────┐                              │
│             │  CTC Head (RNN)  │ ← Highest LR                │
│             │  BiLSTM + Linear │                              │
│             └──────────────────┘                              │
│                        ▼                                      │
│              [T, B, nclasses]                                 │
└─────────────────────────────────────────────────────────────┘
```

## Fine-Tuning Techniques

### 1. Layer-wise Learning Rate Decay (LLRD)

Each transformer layer gets a different learning rate — deeper (earlier) layers get lower LR to preserve generic features:

```
Head/Adapters  →  head_lr        (e.g., 5e-4)
Final LayerNorm → base_lr × 1.0  (e.g., 2e-5)
Block 11       →  base_lr × γ¹   (e.g., 1.7e-5 with γ=0.85)
Block 10       →  base_lr × γ²   (e.g., 1.4e-5)
  ...
Block 0        →  base_lr × γ¹²  (e.g., 2.8e-6)
Embedding      →  base_lr × γ¹³  (e.g., 2.4e-6)
```

**Rationale**: Lower layers learn universal features (edges, textures) — preserve them. Upper layers learn task-specific features — adapt them.

### 2. Gradual Unfreezing

Progressive training schedule:

```
Epochs 1-N:       Only head + adapters trainable (learn CTC alignment)
Epochs N+1 to M:  Unfreeze one layer every K epochs (top → bottom)
After all unfrozen: Full model fine-tuning with LLRD
```

**Rationale**: Prevents catastrophic forgetting. Head learns CTC mapping first, then backbone gradually adapts.

### 3. Weight Decay Separation

- **Backbone weights**: Higher WD (0.01-0.05) for regularization
- **Bias/LayerNorm**: Zero WD (1-D parameters excluded automatically)
- **Head (RNN)**: Zero WD (RNNs sensitive to regularization)

## File Structure

```
utils/
  finetuning.py            ← Core module (LLRD, unfreezer, builders)

configs/
  finetune_torchvision_vit.yaml  ← ViT-B/16 fine-tuning config
  finetune_trocr.yaml            ← TrOCR Base fine-tuning config

experiments_execution/slurm_scripts/
  15_finetune_vit_b16.slurm      ← SLURM job for ViT-B fine-tuning
  16_finetune_trocr_base.slurm   ← SLURM job for TrOCR fine-tuning
```

## Usage

### Quick Start

```bash
# Fine-tune TorchVision ViT-B/16
python scripts/trainer.py configs/config.yaml configs/finetune_torchvision_vit.yaml

# Fine-tune TrOCR Base
python scripts/trainer.py configs/config.yaml configs/finetune_trocr.yaml

# Override hyperparameters via CLI
python scripts/trainer.py configs/config.yaml configs/finetune_torchvision_vit.yaml \
    finetune.base_lr=1e-5 finetune.lr_decay_rate=0.9 train.num_epochs=60
```

### SLURM Submission

```bash
cd experiments_execution/slurm_scripts
sbatch 15_finetune_vit_b16.slurm
sbatch 16_finetune_trocr_base.slurm
```

### Programmatic API

```python
from utils.finetuning import (
    build_finetune_optimizer,
    build_finetune_scheduler,
    GradualUnfreezer,
    get_vit_layer_groups,
)

# Inspect layer groups
groups = get_vit_layer_groups(net, 'torchvision_vit')
for name, params in groups:
    print(f"{name}: {sum(p.numel() for p in params):,} params")

# Build optimizer with LLRD
optimizer = build_finetune_optimizer(net, 'torchvision_vit', config)

# Build warmup + cosine scheduler
scheduler = build_finetune_scheduler(optimizer, config, max_epochs=40)

# Gradual unfreezing
unfreezer = GradualUnfreezer(net, 'torchvision_vit', warmup_frozen=3, unfreeze_every=3)
for epoch in range(1, 41):
    n_unfrozen = unfreezer.step(epoch)
    train_one_epoch(...)
    scheduler.step()
```

## Configuration Reference

### `finetune` namespace in YAML

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | False | Activate LLRD fine-tuning mode |
| `base_lr` | float | 2e-5 | LR for deepest backbone layer |
| `head_lr` | float | 5e-4 | LR for head and adapter layers |
| `lr_decay_rate` | float | 0.9 | Multiplicative decay per layer (γ) |
| `weight_decay` | float | 0.01 | Backbone weight decay |
| `head_wd` | float | 0.0 | Head weight decay |
| `warmup_epochs` | int | 5 | Linear warmup epochs |
| `gradual_unfreeze` | bool | False | Enable progressive unfreezing |
| `unfreeze_warmup` | int | 3 | Epochs to keep backbone fully frozen |
| `unfreeze_every` | int | 3 | Epochs between unfreezing layers |

### Recommended Settings

| Model | base_lr | head_lr | γ | WD | Epochs |
|-------|---------|---------|---|----|----|
| ViT-B/16 (ImageNet) | 2e-5 | 5e-4 | 0.85 | 0.05 | 40 |
| TrOCR Base (HW) | 5e-6 | 3e-4 | 0.9 | 0.01 | 30 |

## Backward Compatibility

- Setting `finetune.enabled: False` (or omitting the `finetune` section) falls back to the **legacy 2-group optimizer** — no LLRD, no gradual unfreezing.
- Existing configs (`configs/torchvision_vit.yaml`, `configs/trocr.yaml`) work unchanged.
- The `freeze_backbone` / `freeze_encoder` flags still work in legacy mode.

## Design Decisions

1. **Why LLRD over full freeze + head-only?**  
   Full encoder freezing leaves significant CER on the table. LLRD allows the model to adapt pretrained features without catastrophic forgetting.

2. **Why gradual unfreezing?**  
   With only 6.5K IAM training lines, unfreezing 85M params simultaneously causes overfitting. Gradual unfreezing provides implicit regularization.

3. **Why different γ for ViT vs TrOCR?**  
   TrOCR is already in-domain (handwriting) — its features need less adaptation, so we use a more conservative schedule (γ=0.9 vs 0.85).

4. **Why separate no-decay groups?**  
   Standard practice from BERT/ViT fine-tuning: bias and LayerNorm params should not be regularized, as they have clear optimal values independent of model complexity.
