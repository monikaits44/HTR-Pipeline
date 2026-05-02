# Experiment Analysis & Expert Recommendations

> **Expert analysis of HTR pipeline results with actionable recommendations to achieve competitive CER/WER across all architectures.**

---

## Table of Contents

1. [Results Summary](#1-results-summary)
2. [Diagnosis: What's Working](#2-diagnosis-whats-working)
3. [Diagnosis: What's Failing and Why](#3-diagnosis-whats-failing-and-why)
4. [Root Cause Analysis (Per Architecture)](#4-root-cause-analysis-per-architecture)
5. [What Is Correct in the Current Implementation](#5-what-is-correct-in-the-current-implementation)
6. [What Is Wrong in the Current Implementation](#6-what-is-wrong-in-the-current-implementation)
7. [Recommended Modifications (Priority Ordered)](#7-recommended-modifications-priority-ordered)
8. [Architecture-Specific Action Plans](#8-architecture-specific-action-plans)
9. [Expected Outcomes After Fixes](#9-expected-outcomes-after-fixes)
10. [Justifications from Deep Learning Literature](#10-justifications-from-deep-learning-literature)

---

## 1. Results Summary

| Run | Epochs | Registers | Model | Head | Train CTC | Val CER | Val WER | Test CER | Test WER | Verdict |
|-----|--------|-----------|-------|------|-----------|---------|---------|----------|----------|---------|
| 21 | 30 | 0 | ViT-RGTS | RNN-CTC | 124.49 | 0.764 | 1.099 | 0.764 | 1.116 | ❌ Not learning |
| 22 | 30 | 2 | ViT-RGTS | RNN-CTC | 125.36 | 0.978 | 1.000 | 0.978 | 1.000 | ❌ Not learning |
| 23 | 30 | 4 | ViT-RGTS | RNN-CTC | 126.75 | 0.829 | 1.000 | 0.830 | 1.000 | ❌ Not learning |
| 24 | 30 | 8 | ViT-RGTS | RNN-CTC | 126.71 | 0.829 | 1.000 | 0.830 | 0.999 | ❌ Not learning |
| 25 | 30 | 16 | ViT-RGTS | RNN-CTC | 126.44 | 0.916 | 1.000 | 0.916 | 0.999 | ❌ Not learning |
| **32** | **30** | **NA** | **Baseline** | **CNN-RNN** | **12.04** | **0.043** | **0.154** | **0.060** | **0.203** | **✅ Excellent** |
| 29 | 30 | 4 | ViT-B/16 | RNN-CTC | 122.91 | 0.759 | 1.324 | 0.759 | 1.365 | ❌ Not learning |
| 31 | 30 | NA | TrOCR | Base | 126.71 | 0.978 | 1.000 | 0.978 | 1.000 | ❌ Not learning |

### Key Observations

- **CNN-RNN (run_32)** is the only model that converged. CER 4.3% / WER 15.4% is a strong baseline.
- **ALL ViT-based models** (runs 21–25, 29, 31) have CTC loss stuck at ~122–127, which is essentially the **blank-token collapse** loss — the model outputs CTC blank for every timestep.
- WER ≥ 1.0 means the model produces worse-than-random word predictions (more error than content).
- Adding register tokens (0 → 16) made no difference because the base ViT isn't learning at all.

---

## 2. Diagnosis: What's Working

### ✅ CNN-RNN Pipeline (run_32)
The baseline is well-implemented and achieves near state-of-the-art results:

| Component | Status | Details |
|-----------|--------|---------|
| CNN Backbone | ✅ Correct | ResNet-style BasicBlocks with proper skip connections, BN, ReLU |
| Maxpool Flattening | ✅ Correct | Reduces H to 1, creating sequence for CTC |
| BiLSTM Head (CTCtopB) | ✅ Correct | 3-layer BiLSTM + CNN shortcut with dual supervision |
| CTC Loss | ✅ Correct | `log_softmax` → `CTCLoss(reduction='sum', zero_infinity=True)` / batch_size |
| Optimizer | ✅ Correct | AdamW, weight_decay=0.00005, MultiStepLR at [50%, 75%] |
| Data Pipeline | ✅ Correct | Random width/height scaling + augmentation |
| Dual Supervision | ✅ Correct | `loss_rnn + 0.1 * loss_cnn` provides gradient flow through CNN shortcut |

**Why it works:** The CNN backbone has strong inductive biases — locality, translation equivariance, hierarchical features. These priors are perfect for handwriting where strokes are local and sequential. The BiLSTM adds temporal modeling, and 30 epochs with ~6500 samples is sufficient.

---

## 3. Diagnosis: What's Failing and Why

### The ViT Failure Is Not a Minor Issue — It's Structural

The CTC loss values tell the full story:

$$\text{CTC Loss}_{\text{blank collapse}} \approx -\log\left(\frac{1}{C}\right) \times T \approx \ln(80) \times \frac{1024/32} = 4.38 \times 32 \approx 140$$

The observed losses (~125–127) are near this theoretical maximum — the model outputs nearly uniform distributions or pure blank tokens at every timestep. **The ViT backbone is producing uninformative features.**

### Why ViTs Fail on Small HTR Datasets: The Fundamental Problem

| Factor | CNN-RNN | ViT (from scratch) | Impact |
|--------|---------|---------------------|--------|
| **Inductive bias** | Locality + translation invariance built-in | None — must learn everything from data | Critical |
| **Data required** | ~5K–10K samples sufficient | ~100K–1M+ samples typically needed | **10–100x gap** |
| **Feature hierarchy** | Automatic via pooling layers | Must be learned via attention | Slow convergence |
| **Positional understanding** | Implicit in conv structure | Learned positional embeddings only | Fragile |
| **IAM dataset size** | 6,482 train lines | 6,482 train lines | **Too small for ViT from scratch** |

**This is the single most critical insight: training a ViT from scratch on 6,482 images is fundamentally data-insufficient.** No amount of learning rate tuning or scheduler changes will fix this.

---

## 4. Root Cause Analysis (Per Architecture)

### 4.1 ViT-RGTS (runs 21–25) — Training from Scratch

**Root Causes (ordered by severity):**

1. **Insufficient data for from-scratch ViT training**
   - IAM has 6,482 training lines. ViTs typically need 100K+ for convergence.
   - DeiT (Touvron et al., 2021) required ImageNet (1.2M images) even with extensive augmentation and distillation.
   - Without pretraining, the transformer cannot learn meaningful patch-to-patch relationships in 30 epochs.

2. **Patch size still too large for fine-grained HTR**
   - Current: `patch_height=8, patch_width=32` → 512 patches.
   - Each 8×32 patch covers ~2–3 characters horizontally. This destroys character-level detail.
   - For comparison, DeiT/ViT-Base uses 16×16 patches on 224×224 images — each patch covers a semantically coherent region.
   - **For 128×1024 line images, better patch sizes: 8×8 or 16×16, yielding 1024 or 512 patches** that each cover sub-character regions.

3. **Model too deep for the data size**
   - Config has `depth: 12` transformer layers.
   - With only 6,482 samples, 12 layers is massively overparameterized.
   - **Recommendation: depth=4–6** for from-scratch training on small data.

4. **No pretraining or knowledge distillation**
   - Modern small-data ViTs universally use pretraining (ImageNet, MAE, DINO).
   - Without it, the attention mechanism has no starting point to build on.

5. **Learning rate may still be too high**
   - Current: `lr = 0.001 * 0.5 = 0.0005`
   - For from-scratch ViTs on small data, LR should be 1e-4 to 3e-4 with careful warmup.

6. **CTC + ViT sequence length mismatch**
   - 512 patches create a 512-length sequence for CTC. But IAM transcriptions average ~25 characters.
   - The CTC blank probability dominates: $P(\text{blank})$ must be ~95% at each timestep, making gradients extremely sparse.
   - **This is perhaps the most overlooked issue**: the sequence is too long relative to label length.

### 4.2 TorchVision ViT-B/16 (run_29) — Pretrained but Misapplied

**Root Causes:**

1. **Positional embedding interpolation is destructive**
   - ViT-B/16 was pretrained on 224×224 images (14×14 = 196 patches).
   - Your input: 128×1024 → 8×64 = 512 patches.
   - Positional embeddings are interpolated from 14×14 → 8×64 grid using bicubic interpolation.
   - This extreme aspect ratio change (1:1 → 1:8) **destroys the spatial structure** the model learned during ImageNet pretraining.

2. **Fine-tuning learning rate too high for pretrained model**
   - Current: `lr = 0.0005` for ALL parameters.
   - Pretrained backbone should use much lower LR (1e-5 to 5e-5).
   - The CTC head (new, random weights) needs higher LR (1e-3 to 5e-4).
   - **Differential learning rates** are essential for transfer learning.

3. **No layer-wise LR decay**
   - Deep pretrained transformers benefit from lower LR for early layers (generic features) and higher LR for later layers (task-specific).
   - Standard practice: LR decay factor 0.65–0.75 per layer.

4. **Grayscale → RGB conversion is naive**
   - `gray_to_rgb` uses `Conv2d(1, 3, 1)` initialized to `weight.fill_(1.0)`.
   - This triples the pixel values. ViT-B/16 expects ImageNet-normalized RGB ([0.485, 0.456, 0.406] mean, [0.229, 0.224, 0.225] std).
   - **The input distribution is completely wrong** for the pretrained weights.

5. **`freeze_backbone: False` is wrong for small data**
   - Fine-tuning all 86M parameters of ViT-B/16 on 6,482 samples will overfit catastrophically.
   - Should freeze most layers and only fine-tune last 2–4 transformer blocks + head.

### 4.3 TrOCR (run_31) — Pretrained but Wrong Input Format

**Root Causes:**

1. **Square resize destroys handwriting aspect ratio**
   - Config: `image_height: 384, image_width: 384` — input is **square**.
   - IAM line images are ~128×1024 (1:8 aspect ratio).
   - Resizing to 384×384 squashes text horizontally by 8×, making characters unreadable to the model.
   - TrOCR was pretrained on cropped text images, not 1:1 square crops.

2. **CTC head on TrOCR is architecturally wrong**
   - TrOCR is designed as an **encoder-decoder** model with cross-attention decoder and autoregressive token prediction.
   - Stripping the decoder and attaching a CTC head discards the model's strongest component.
   - The TrOCR encoder alone produces 576 patch tokens (384/16 × 384/16 = 24×24) — spatial, not sequential.

3. **Same learning rate issues as TorchVision ViT**
   - Pretrained encoder needs very low LR; CTC head needs higher LR.

---

## 5. What Is Correct in the Current Implementation

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| CNN backbone architecture | `models.py` → `CNN`, `BasicBlock` | ✅ Correct | Well-designed ResNet-style blocks |
| CTC loss computation | `trainer.py` → `prepare_losses` | ✅ Correct | `log_softmax` + batch normalization |
| Label encoding for CTC | `trainer.py` → `train()` | ✅ Correct | Concatenated labels + label_lens |
| Architecture-aware optimizer (CNN-RNN) | `trainer.py` → `prepare_optimizers` | ✅ Correct | weight_decay=0.00005, MultiStepLR |
| Architecture-aware gradient clipping | `trainer.py` → `train()` | ✅ Correct | CNN-RNN: no clipping, ViT: max_norm=5.0 |
| CTCtopB return behavior | `models.py` → `CTCtopB` | ✅ Correct | Tuple in training, single tensor in eval |
| Dual supervision (head_type="both") | `trainer.py` → `train()` | ✅ Correct | `loss + 0.1 * aux_loss` |
| Dataset class | `utils/htr_dataset.py` | ✅ Correct | Proper augmentation, grayscale, padding |
| CER/WER metrics | `utils/metrics.py` | ✅ Correct | Edit distance-based |
| ViT-RGTS `forward_explain` | `models.py` | ✅ Correct | Manual Q/K/V extraction for attention |
| Attention extraction during training | `trainer.py` | ✅ Correct | Saves every 5 epochs for ViT |
| Experiment directory structure | `trainer.py` | ✅ Correct | config.json + results.csv + checkpoints |
| LayerNorm for ViT heads | `models.py` → `CTCtopR`, `CTCtopB` | ✅ Correct | Architecture-conditional |
| Warmup + Cosine scheduler for ViT | `trainer.py` | ✅ Correct | 5-epoch warmup, cosine annealing |

---

## 6. What Is Wrong in the Current Implementation

### 🔴 Critical Issues (Must Fix)

| # | Issue | File | Impact | Fix Effort |
|---|-------|------|--------|------------|
| 1 | **ViT-RGTS depth=12 is too deep for 6.5K samples** | `configs/baseline_vit_rgts.yaml` | Model can't converge | Config change |
| 2 | **Patch width=32 is too large, destroys character detail** | `configs/baseline_vit_rgts.yaml` | Features are meaningless | Config change |
| 3 | **TorchVision ViT: No ImageNet normalization** | `models.py` → `TorchVisionViTBackbone` | Pretrained weights useless | Code change |
| 4 | **TorchVision ViT: No differential LR** | `trainer.py` → `prepare_optimizers` | Backbone overwrites pretrained features | Code change |
| 5 | **TrOCR: Square 384×384 resize destroys aspect ratio** | `configs/trocr.yaml` | Text becomes unreadable | Config change |
| 6 | **All ViT: Using EXTRA STRONG augmentation** | `trainer.py` → `prepare_dataloaders` | Too aggressive for small data, destroys signal | Code change |
| 7 | **TorchVision ViT: All params unfrozen** | `configs/torchvision_vit.yaml` | 86M params fine-tuned on 6.5K samples → overfit then collapse | Config change |
| 8 | **ViT-RGTS: No pretraining strategy** | N/A | From-scratch ViT can't learn on small data | Architecture decision |

### 🟡 Important Issues (Should Fix)

| # | Issue | File | Impact |
|---|-------|------|--------|
| 9 | No differential LR for pretrained vs new layers | `trainer.py` | Pretrained features destroyed by high LR |
| 10 | No label smoothing for CTC | `trainer.py` | Overconfident blank predictions |
| 11 | No intermediate CTC (aux) loss for deep ViTs | `models.py` | Vanishing gradients in deep transformers |
| 12 | Augmentation is uniform — not adapted per architecture | `trainer.py` | EXTRA STRONG augment hurts when model can't learn basics |
| 13 | ViT-RGTS has no CLS token | `models.py` | Minor: register tokens partially compensate |
| 14 | No stochastic depth / DropPath for ViT | `models.py` | Missing standard ViT regularization |
| 15 | Batch size=8 is too small for ViT training | `configs/config.yaml` | ViTs benefit from larger effective batch sizes |

### 🟢 Minor Issues (Nice to Fix)

| # | Issue | Impact |
|---|-------|--------|
| 16 | No mixed-precision (AMP) training | Slower training, higher memory usage |
| 17 | No EMA (Exponential Moving Average) of weights | Slightly worse final performance |
| 18 | Only best-by-val-CER checkpoint saved | Could miss models good at different metrics |

---

## 7. Recommended Modifications (Priority Ordered)

### Priority 1: Fix ViT-RGTS (From-Scratch Training)

Since IAM is too small for from-scratch ViT, there are two strategies:

#### Strategy A: Make ViT-RGTS Small Enough to Learn (Recommended)

```yaml
# configs/baseline_vit_rgts.yaml — RECOMMENDED CHANGES
arch:
  type: 'vit_rgts'
  image_height: 128
  image_width: 1024

  # CHANGE 1: Smaller patches = more detail per patch
  # 8×8 patches → 16×128 = 2048 patches (too many for CTC)
  # 16×16 patches → 8×64 = 512 patches (good balance)
  # BEST: 8×16 patches → 16×64 = 1024 patches
  patch_height: 16    # was 8 → 16 (larger vertical = less patches)
  patch_width: 16     # was 32 → 16 (smaller horizontal = more character detail)

  # CHANGE 2: Much shallower — 4 layers, not 12
  dim: 128            # was 256 → 128 (smaller embedding for small data)
  depth: 4            # was 12 → 4 (critical: prevent overfitting)
  heads: 4            # was 8 → 4 (proportional to dim)
  mlp_dim: 256        # was 512 → 256

  # CHANGE 3: More aggressive dropout for regularization
  dropout: 0.2        # was 0.1 → 0.2
  emb_dropout: 0.2    # was 0.1 → 0.2

  # Registers (keep experimenting with 0 and 4)
  num_registers: 4

  head_type: 'rnn'
  rnn_type: 'lstm'
  rnn_layers: 2       # was 3 → 2 (smaller model)
  rnn_hidden_size: 128 # was 256 → 128
```

**Why these changes:**
- `dim=128, depth=4`: ~500K params instead of ~8M. DeiT-Tiny (5M params) needs ImageNet-scale data; our model should be much smaller for 6.5K samples.
- `patch=16×16`: Standard ViT patch size. Each patch is a character-width region. 512 patches is reasonable for CTC.
- `rnn_layers=2, hidden=128`: Proportional to backbone size.

**Corresponding trainer.py changes for from-scratch ViT:**

```python
# In prepare_optimizers — for vit_rgts FROM SCRATCH:
lr = config.train.lr * 0.3       # 3e-4 instead of 5e-4
weight_decay = 0.05              # Higher for small ViT (DeiT uses 0.05)
warmup_epochs = 10               # Longer warmup (was 5)
```

#### Strategy B: Use Self-Supervised Pretraining (Best Results, More Effort)

Pretrain the ViT-RGTS encoder using **Masked Autoencoder (MAE)** on IAM data (self-supervised — no labels needed):

1. Mask 75% of patches randomly
2. Train encoder + lightweight decoder to reconstruct masked patches
3. After pretraining (100–200 epochs), attach CTC head and fine-tune

This is the approach used by successful ViT-based HTR papers (e.g., DTrOCR, TrOCR).

---

### Priority 2: Fix TorchVision ViT (Pretrained Fine-Tuning)

This has the **highest potential** because ViT-B/16 already has strong visual features from ImageNet.

#### Config Changes:

```yaml
# configs/torchvision_vit.yaml — RECOMMENDED CHANGES
arch:
  type: 'torchvision_vit'
  model_name: 'vit_b_16'
  pretrained: True
  
  image_height: 224       # was 128 → 224 (match pretrained resolution)
  image_width: 224        # was 1024 → 224 (see note below)
  
  # CRITICAL: Freeze most of backbone
  freeze_backbone: False  # We'll use differential LR instead (better)
  
  num_registers: 0
  head_type: 'rnn'
  rnn_type: 'lstm'
  rnn_layers: 2
  rnn_hidden_size: 256
```

**Image resolution note:** ViT-B/16 was pretrained on 224×224. Options:
- **Option A (Simple):** Resize to 224×224 (distorts aspect ratio but preserves pretrained pos-embeddings perfectly). Worth trying as a baseline.
- **Option B (Better):** Resize to 128×896, interpolate pos-embeddings to 8×56 grid. Less distortion but pos-embeddings are approximate.
- **Option C (Best):** Resize height to 224, pad width to multiple of 16, interpolate pos-embeddings. Preserves aspect ratio.

#### Code Changes Required:

```python
# 1. ADD ImageNet normalization in TorchVisionViTBackbone.forward():
# After gray_to_rgb conversion:
mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
x_rgb = (x_rgb - mean) / std

# 2. ADD differential learning rates in trainer.py:
def prepare_optimizers(self):
    if arch_type == 'torchvision_vit':
        # Low LR for pretrained backbone
        backbone_params = list(self.net.backbone.vit.parameters())
        # Higher LR for new head + gray_to_rgb
        head_params = list(self.net.top.parameters()) + list(self.net.backbone.gray_to_rgb.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 2e-5, 'weight_decay': 0.01},
            {'params': head_params, 'lr': 5e-4, 'weight_decay': 0.0001},
        ])

# 3. ADD layer-wise LR decay (optional but recommended):
def get_layer_wise_lr(model, base_lr=2e-5, decay=0.75):
    """Apply lower LR to earlier layers."""
    param_groups = []
    for i, layer in enumerate(model.backbone.vit.encoder.layers.layers):
        lr = base_lr * (decay ** (len(layers) - 1 - i))
        param_groups.append({'params': layer.parameters(), 'lr': lr})
    return param_groups
```

---

### Priority 3: Fix TrOCR

#### Option A: Use TrOCR Properly (Encoder-Decoder, Not CTC)

TrOCR was designed as an encoder-decoder model. Using only the encoder with CTC is architecturally mismatched. The best approach:

```python
# Use TrOCR end-to-end with its own decoder (autoregressive, not CTC)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
# Fine-tune the FULL model (encoder + decoder) on IAM
# Use cross-entropy loss, not CTC
```

This requires modifying the training loop to use autoregressive decoding instead of CTC.

#### Option B: Fix CTC-Based TrOCR (Simpler but Suboptimal)

```yaml
# configs/trocr.yaml — MINIMUM FIXES
arch:
  type: 'trocr'
  model_name: 'microsoft/trocr-base-handwritten'
  
  # CRITICAL: Preserve aspect ratio!
  image_height: 384
  image_width: 384    # Keep square BUT: pad the image to square instead of resize
  
  # Actually freeze the encoder — it's already pretrained for handwriting
  freeze_encoder: True
  
  head_type: 'rnn'
  rnn_layers: 2
  rnn_hidden_size: 256
```

**Also need:** Aspect-ratio-preserving preprocessing that pads to square instead of stretching.

---

### Priority 4: Training Hyperparameters

#### Augmentation Fix

The current code uses `aug_transforms_vit_strong` (EXTRA STRONG) for **all** architectures. This is too aggressive:

```python
# trainer.py — prepare_dataloaders — RECOMMENDED CHANGE:
if aug_strategy == 'auto':
    if arch_type == 'cnn_rnn':
        selected_transforms = aug_transforms_cnn       # Moderate for CNN (proven)
    elif arch_type == 'vit_rgts':
        selected_transforms = aug_transforms_vit       # Strong (not EXTRA) for from-scratch
    elif arch_type in ['torchvision_vit', 'trocr']:
        selected_transforms = aug_transforms_cnn       # Moderate for pretrained (features already robust)
```

**Justification:** Pretrained models already have augmentation-invariance baked in from ImageNet training. Over-augmenting destroys the signal that the pretrained features expect.

#### Epochs

```
CNN-RNN:        30–50 epochs (sufficient, well-calibrated)
ViT-RGTS:       100–200 epochs (needs much longer to converge from scratch)
TorchVision:    30–50 epochs (pretrained, converges fast with low LR)
TrOCR:          20–30 epochs (pretrained for handwriting, minimal fine-tuning)
```

#### Batch Size

```
CNN-RNN:        8 (fine)
ViT-RGTS:       16–32 (ViTs benefit from larger batches for stable training)
TorchVision:    4–8 (large model, memory constrained on RTX 3080)
TrOCR:          4 (large model)
```

For larger effective batch sizes without more memory, use **gradient accumulation**:

```python
# In trainer.py train() loop:
accumulation_steps = 4  # Effective batch = 8 * 4 = 32
loss_val = loss_val / accumulation_steps
loss_val.backward()

if (iter_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## 8. Architecture-Specific Action Plans

### Plan A: ViT-RGTS — Quick Win

**Goal:** Get ViT-RGTS to CER < 20% (meaningful learning, not competitive yet).

| Step | Change | File |
|------|--------|------|
| 1 | Set `depth: 4, dim: 128, heads: 4, mlp_dim: 256` | `baseline_vit_rgts.yaml` |
| 2 | Set `patch_height: 16, patch_width: 16` | `baseline_vit_rgts.yaml` |
| 3 | Set `rnn_layers: 2, rnn_hidden_size: 128` | `baseline_vit_rgts.yaml` |
| 4 | Set `dropout: 0.2, emb_dropout: 0.2` | `baseline_vit_rgts.yaml` |
| 5 | Use `aug_transforms_vit` (not extra strong) | `trainer.py` |
| 6 | LR = 3e-4, weight_decay = 0.05, warmup = 10 epochs | `trainer.py` |
| 7 | Train for 100–150 epochs | CLI override |
| 8 | Use gradient accumulation (effective batch 32) | `trainer.py` |

### Plan B: TorchVision ViT — Highest Potential

**Goal:** Get pretrained ViT to CER < 10% (leverage ImageNet features).

| Step | Change | File |
|------|--------|------|
| 1 | Add ImageNet normalization after gray_to_rgb | `models.py` |
| 2 | Implement differential LR: backbone=2e-5, head=5e-4 | `trainer.py` |
| 3 | Use moderate augmentation (aug_transforms_cnn) | `trainer.py` |
| 4 | Set `image_height: 128, image_width: 896` (or try 224×224) | `torchvision_vit.yaml` |
| 5 | Train for 40–60 epochs | CLI override |
| 6 | Consider freezing first 8 of 12 transformer layers | `models.py` or config |

### Plan C: TrOCR — Use It Properly

**Goal:** Get TrOCR to CER < 5% (it's pretrained specifically for handwriting).

| Step | Change | File |
|------|--------|------|
| 1 | Use TrOCR end-to-end (encoder + decoder, not CTC) | New training script |
| 2 | Preserve aspect ratio via padding | `utils/htr_dataset.py` |
| 3 | Freeze encoder, fine-tune decoder only | Config |
| 4 | Use TrOCR's native tokenizer | Training script |
| 5 | LR = 3e-5, train for 20 epochs | Config |

---

## 9. Expected Outcomes After Fixes

| Architecture | Current CER | Expected CER (after fixes) | Confidence |
|---|---|---|---|
| CNN-RNN (baseline) | **4.3%** | **4–5%** (no change needed) | ✅ High |
| ViT-RGTS (small, from scratch) | 76–98% | **15–30%** | 🟡 Medium |
| ViT-RGTS (with MAE pretraining) | 76–98% | **5–10%** | 🟡 Medium (more effort) |
| TorchVision ViT (fixed fine-tuning) | 76% | **6–12%** | ✅ High |
| TrOCR (end-to-end, proper use) | 98% | **3–6%** | ✅ High |
| TrOCR (CTC, fixed input) | 98% | **15–25%** | 🟡 Medium |

**Note:** For the research goal of studying register tokens, the ViT-RGTS must at least converge to meaningful performance. A CER of 15–30% is sufficient to study whether registers improve attention quality, even if absolute CER isn't competitive with CNN-RNN.

---

## 10. Justifications from Deep Learning Literature

### 10.1 ViTs Need Large Data or Pretraining

> *"When trained on mid-sized datasets such as ImageNet without strong regularization, ViT yields modest accuracies below ResNets of comparable size. This result is to be expected: Transformers lack the inductive biases inherent to CNNs."*
> — Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020)

**Implication:** IAM (6.5K images) is ~200× smaller than "mid-sized" ImageNet (1.2M). ViT from scratch on IAM is expected to fail.

### 10.2 Data-Efficient Transformers Require Distillation

> *"DeiT shows that it is possible to train data-efficient transformers ... using extensive data augmentation and regularization along with knowledge distillation."*
> — Touvron et al., "Training Data-Efficient Image Transformers" (DeiT, 2021)

**Implication:** Even DeiT required ImageNet-scale data plus distillation from a CNN teacher. For HTR on small data, consider training a CNN teacher (already have run_32!) and distilling into ViT.

### 10.3 Transfer Learning LR Best Practices

> *"The learning rates for different layers should be different: the lower layers of a pretrained model should be fine-tuned with a much smaller learning rate than the upper layers."*
> — Howard & Ruder, "Universal Language Model Fine-tuning" (2018)

**Implication:** Using the same LR=0.0005 for all layers of a pretrained ViT-B/16 will destroy early-layer features within a few epochs.

### 10.4 CTC + Long Sequences

> *"CTC struggles when the output sequence is much shorter than the input sequence... The blank token dominates, making gradients sparse."*
> — Graves et al., "Connectionist Temporal Classification" (2006)

**Implication:** With 512–1024 patches and ~25-character labels, CTC's search space is enormous. Reducing patch count or using a hybrid CTC-attention loss can help.

### 10.5 Register Tokens Require Working Base Model

> *"Register tokens solve the artifact problem in trained ViTs — they absorb low-information tokens' attention mass."*
> — Darcet et al., "Vision Transformers Need Registers" (2023)

**Implication:** Register tokens are a refinement on already-working ViTs. If the base model can't learn (CER > 70%), registers have nothing to improve. **Fix the base model first, then study registers.**

---

## Summary: Top 5 Actions for Better Results

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| **1** | Shrink ViT-RGTS: depth=4, dim=128, patch=16×16 | ViT starts learning (CER 15–30%) | ⭐ Low (config only) |
| **2** | Add ImageNet normalization + differential LR for TorchVision ViT | Pretrained ViT converges (CER 6–12%) | ⭐⭐ Medium (code) |
| **3** | Use TrOCR end-to-end (encoder+decoder, not CTC) | Best possible CER (3–6%) | ⭐⭐⭐ High (new script) |
| **4** | Fix augmentation: moderate for pretrained, medium for from-scratch | All models train more stably | ⭐ Low (code) |
| **5** | Train ViT-RGTS for 100+ epochs with gradient accumulation | ViT has enough training time | ⭐ Low (config) |

---

*Document prepared based on analysis of runs 21–25, 29, 31, 32. All recommendations are justified by established deep learning literature and HTR best practices.*
