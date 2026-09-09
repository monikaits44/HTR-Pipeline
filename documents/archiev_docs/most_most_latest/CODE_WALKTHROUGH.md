# Code Walkthrough: Step-by-Step Through Every Module

> A detailed walkthrough of every source file in the HTR-Pipeline, explaining what each function does, how data flows, and where key decisions are made.

---

## Table of Contents

1. [models.py — All Architectures](#1-modelspy)
2. [scripts/trainer.py — Training Loop](#2-scriptstrainerpy)
3. [utils/preprocessing.py — Image Loading](#3-utilspreprocessingpy)
4. [utils/htr_dataset.py — Dataset Class](#4-utilshtr_datasetpy)
5. [utils/transforms.py — Augmentation](#5-utilstransformspy)
6. [utils/metrics.py — CER & WER](#6-utilsmetricspy)
7. [utils/attention_extractor.py — Explainability](#7-utilsattention_extractorpy)
8. [utils/finetuning.py — Transfer Learning](#8-utilsfinetuningpy)
9. [utils/visualizer.py — Feature Visualization](#9-utilsvisualizerpy)
10. [Config Files — YAML Reference](#10-config-files)

---

## 1. models.py

**Location**: `/models.py` (~1200 lines)  
**Purpose**: Defines all neural network architectures for the HTR pipeline.

### Class Hierarchy

```
models.py
├── BasicBlock          — ResNet-style residual block (conv-bn-conv-bn + shortcut)
├── CNN                 — Feature extraction backbone (configurable depth)
├── weight_init()       — Xavier initialization for conv layers
├── CTCtopC             — CTC head: Conv 1×3 (simple, fast)
├── CTCtopLinear        — CTC head: Linear projection
├── CTCtopR             — CTC head: BiRNN (GRU/LSTM) + Linear
├── CTCtopB             — CTC head: BiRNN + CNN shortcut (dual supervision)
├── ViTRGTSBackbone     — Custom ViT with CNN stem + register tokens
├── TorchVisionViTBackbone — Pretrained ViT-B/16 wrapper
├── TrOCREncoderBackbone   — HuggingFace TrOCR encoder wrapper
└── HTRNet              — Unified entry point (routes to correct backbone+head)
```

### Key: BasicBlock

```python
class BasicBlock(nn.Module):
    """Standard ResNet residual block.
    
    Flow: x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU
    If dimensions change: shortcut = Conv1×1 + BN
    """
```

### Key: CNN Backbone

```python
class CNN(nn.Module):
    """Configurable CNN feature extractor.
    
    cnn_cfg example: [[2, 64], 'M', [3, 128], 'M', [2, 256]]
    Meaning: 2 blocks of 64 channels → MaxPool → 3 blocks of 128 → MaxPool → 2 blocks of 256
    
    First layer: Conv 7×7 (stride 4,2) reduces spatial dims early
    Final: MaxPool(height, 1) collapses to [B, C, 1, W]
    """
```

### Key: CTCtopB (Dual Head)

```python
class CTCtopB(nn.Module):
    """The recommended CTC head — dual supervision.
    
    Training:  returns (rnn_output, cnn_output) — both get CTC loss
    Inference: returns rnn_output only — CNN path is auxiliary
    
    Why dual: CNN shortcut provides direct gradient path to backbone,
    preventing vanishing gradients through deep BiLSTM.
    The 0.1× weight on CNN loss prevents it from dominating.
    """
    def forward(self, x):
        # RNN path
        y = x.squeeze(2).permute(2, 0, 1)  # [B,C,1,W] → [T,B,C]
        y = self.rec(y)[0]                   # BiLSTM → [T,B,2H]
        if self.is_vit:
            y = self.layer_norm(y)           # Stabilize ViT features
        y = self.fnl(y)                      # → [T,B,nclasses]
        
        if self.training and self.return_both:
            cnn_out = self.cnn(x).squeeze(2).permute(2, 0, 1)
            return y, cnn_out   # Dual supervision
        return y                # Single output at eval
```

### Key: ViTRGTSBackbone

```python
class ViTRGTSBackbone(nn.Module):
    """
    The core innovation: CNN stem + Transformer + Register tokens.
    
    CNN Stem (when use_cnn_stem=True):
        128×1024 input → 4 conv layers → height collapse → 128 tokens
        Each token ≈ 8px horizontal span ≈ 3 tokens per character
    
    Register tokens:
        R learnable [D]-dim vectors prepended to the sequence
        Attend to all patches via self-attention
        Capture global information (writer style, line statistics)
    
    Positional embeddings:
        Learned [1, max_seq_len, D] embeddings
        Sliced to actual sequence length: pos_embed[:, :S, :]
    
    Transformer:
        nn.TransformerEncoder with norm_first=True (pre-norm)
        6 layers × 8 heads × 256 dim × 1024 MLP
    """
    
    def forward(self, x):
        # 1. Feature extraction
        if self.use_cnn_stem:
            x = self.cnn_stem(x)          # [B,1,128,1024] → [B,D,H',W']
            x = self.stem_pool(x)         # → [B,D,1,128]
            patch_tokens = x.squeeze(2).transpose(1,2)  # [B,128,D]
        else:
            x = self.patch_embed(x)       # → [B,D,Hp,Wp]
            patch_tokens = x.flatten(2).transpose(1,2)  # [B,Hp×Wp,D]
        
        # 2. Prepend registers
        reg = self.register_tokens.expand(B, -1, -1)  # [B,R,D]
        tokens = torch.cat([reg, patch_tokens], dim=1) # [B, R+128, D]
        
        # 3. Add positional embeddings
        tokens = tokens + self.pos_embed[:, :S, :]
        tokens = self.emb_dropout(tokens)
        
        # 4. Transformer encoding
        encoded = self.encoder(tokens)  # [B, S, D]
        
        # 5. Split outputs
        reg_out = encoded[:, :R, :]     # Register embeddings
        patch_out = encoded[:, R:, :]   # Patch sequence
        seq_tokens = patch_out.transpose(0, 1)  # [T, B, D] for CTC
        
        return seq_tokens, reg_out, (Hp, Wp)
    
    def forward_explain(self, x):
        """Manual transformer loop to extract per-layer attention.
        
        Instead of using self.encoder(tokens), iterates through each
        TransformerEncoderLayer manually:
        1. Extract Q, K, V from in_proj_weight
        2. Compute attention: softmax(QK^T / √d)
        3. Save attention weights [B, H, S, S]
        4. Apply attention to values
        5. Residual + FFN
        
        Returns attention_maps: List[6] of [B, 8, S, S] tensors
        """
```

### Key: HTRNet Router

```python
class HTRNet(nn.Module):
    """Dispatches to correct backbone based on arch_cfg.type.
    
    Handles all the config parsing:
    - Reads arch.dim, arch.depth, arch.heads etc.
    - Computes max_seq_len automatically
    - Creates appropriate backbone + head combination
    
    forward():     returns logits [T, B, nclasses]
    forward_explain(): returns (logits, reg_tokens, attn_maps, token_norms, grid)
    """
```

---

## 2. scripts/trainer.py

**Location**: `/scripts/trainer.py` (~1115 lines)  
**Purpose**: Complete training pipeline — from config to saved model.

### Flow: main block

```python
if __name__ == '__main__':
    config = parse_args()          # OmegaConf: load YAML + CLI overrides
    experiment_dir, log_file, run_number, csv_file, csv_writer = setup_experiment_dir(config)
    htr_trainer = HTRTrainer(config, experiment_dir, ...)
    
    for epoch in range(1, max_epochs + 1):
        # Optional: gradual unfreezing for pretrained models
        if htr_trainer.unfreezer:
            htr_trainer.unfreezer.step(epoch)
        
        htr_trainer.train(epoch)
        htr_trainer.scheduler.step()
        
        htr_trainer.save(epoch)
        val_cer, val_wer = htr_trainer.test(epoch, 'val')
        test_cer, test_wer = htr_trainer.test(epoch, 'test')
        
        # Extract attention every 5 epochs (ViT only)
        if arch_type in ['vit_rgts', ...] and epoch % 5 == 0:
            htr_trainer.extract_attention_weights(epoch)
        
        # Log to CSV
        csv_writer.writerow([epoch, lr, loss, val_cer, val_wer, ...])
```

### HTRTrainer Methods

#### `prepare_dataloaders()`

```
1. Read config: data.path, preproc.image_height/width
2. Auto-select augmentation based on architecture type:
   - CNN-RNN → moderate (aug_transforms_cnn)
   - ViT from scratch → strong (aug_transforms_vit)
   - Pretrained → moderate (aug_transforms_cnn)
3. Choose data mode:
   - 'iam': train/val/test all from IAM
   - 'synthetic': train/val from synthetic, test from IAM
     → Computes unified charset = synth_chars ∪ iam_chars
4. Create DataLoaders with proper batch sizes
5. Build character dictionaries: c2i (char→index+1), i2c (index+1→char)
   Note: index 0 reserved for CTC blank
```

#### `prepare_net()`

```
1. Create HTRNet(config.arch, nclasses=len(classes)+1)
   - +1 for CTC blank at index 0
2. Optional: load checkpoint (config.resume)
3. Move to device
4. Log parameter count
```

#### `prepare_optimizers()`

Architecture-specific setup:
```
CNN-RNN:
  - Single param group, lr=0.001, wd=5e-5
  - MultiStepLR at 50%/75% epochs

ViT-RGTS v2 (CNN stem):
  - 3 param groups: stem(lr=1e-3, wd=5e-4), transformer(lr=5e-4, wd=5e-3), head(lr=1e-3, wd=1e-4)
  - 5-epoch linear warmup + cosine annealing

Pretrained (ViT-B/16, TrOCR):
  - Either LLRD (finetune.enabled=true) or 2-group differential LR
  - Optional gradual unfreezing
  - Warmup + cosine annealing
```

#### `train(epoch)`

```python
for img, transcr in train_loader:
    output = self.net(img)             # Forward pass
    # If dual head: output, aux_output = output
    
    # CTC loss computation:
    act_lens = [seq_length] × batch_size
    labels = concatenated character indices for all samples
    label_lens = [len(transcript) for each sample]
    
    loss = CTC(output, labels, act_lens, label_lens)
    if dual_head:
        loss += 0.1 * CTC(aux_output, labels, act_lens, label_lens)
    
    (loss / accum_steps).backward()
    
    if (iter + 1) % accum_steps == 0:
        clip_grad_norm_(params, 10.0)  # ViT only
        optimizer.step()
        optimizer.zero_grad()
```

#### `test(epoch, tset)`

```python
for imgs, transcrs in loader:
    o = self.net(imgs)
    tdecs = o.argmax(2).permute(1, 0)  # Greedy decoding
    
    for tdec, transcr in zip(tdecs, transcrs):
        prediction = decode(tdec)  # Remove blanks + consecutive duplicates
        cer.update(prediction, transcr.strip())
        wer.update(prediction, transcr.strip())
        # Also log per-sample to evaluation_details.csv
```

#### `extract_attention_weights(epoch)`

```
Only for ViT architectures. Runs forward_explain on 5 validation samples.
Saves per-sample:
  - sample_XXX_layer_YY.npy  (attention map [H, S, S])
  - sample_XXX_token_norms.npy
  - sample_XXX_register_tokens.npy
  - sample_XXX_groundtruth.txt
  - metadata.json (stats)
```

---

## 3. utils/preprocessing.py

**Location**: `/utils/preprocessing.py` (40 lines)

### `load_image(path)`

```python
image = imread(path)           # Load as-is
image = rgb2gray(image)        # Convert to grayscale if needed
image = 1 - image / 255.0     # Invert: white bg → 0, dark ink → 1
# Returns: float array [H, W], values in [0, 1]
```

### `preprocess(img, input_size=(128, 1024), border_size=8)`

```python
# 1. Compute target dimensions (minus border)
n_height = min(128 - 16, img.height)
scale = n_height / img.height
n_width = min(1024 - 16, int(scale * img.width))

# 2. Resize to fit within (112 × 1008)
img = resize(img, (n_height, n_width))

# 3. Pad to exact 128×1024 with median value
# Border of 8px on each side, remaining padding on right/bottom
img = np.pad(img, ((8, ...), (8, ...)), mode='median')
```

This preserves aspect ratio and uses median padding (avoids harsh white/black borders).

---

## 4. utils/htr_dataset.py

**Location**: `/utils/htr_dataset.py` (61 lines)

```python
class HTRDataset(Dataset):
    def __init__(self, basefolder, subset, fixed_size=(128, None),
                 transforms=None, character_classes=None):
        # Load gt.txt: "image_path transcription"
        # Auto-compute character_classes if not provided
    
    def __getitem__(self, index):
        img = load_image(path)
        
        # Training: random resize (width ×0.75-1.25, height ×0.9-1.1)
        if self.subset == 'train':
            nwidth = int(uniform(0.75, 1.25) * img.width)
            nheight = int((uniform(0.9, 1.1) * img.height / img.width) * nwidth)
            img = resize(img, (nheight, nwidth))
        
        img = preprocess(img, fixed_size)    # Aspect-ratio resize + pad
        if self.transforms:
            img = self.transforms(image=img)['image']  # Albumentations
        
        transcr = " " + transcr + " "  # Add space padding
        return torch.Tensor(img).unsqueeze(0), transcr  # [1, H, W]
```

**Important**: Transcriptions are padded with spaces: `" text "`. This means spaces are always in the character set.

---

## 5. utils/transforms.py

**Location**: `/utils/transforms.py` (183 lines)

Three tiers of Albumentations augmentation:

### `aug_transforms_cnn` (Moderate)
- Affine: ±1° rotation, shear x±30° y±5°, scale 0.6-1.2
- GridDistortion or ElasticTransform (p=0.5)
- Morphological dilation/erosion (p=0.5)
- BrightnessContrast, Gamma

### `aug_transforms_vit` (Strong)
All of the above PLUS:
- Affine: ±10° rotation (10× more than CNN)
- Perspective transform (p=0.4)
- Stronger elastic distortion
- GaussNoise (p=0.5)
- CoarseDropout: 1-3 small holes with white fill (p=0.4)
- GaussianBlur / MotionBlur (p=0.3)
- Sharpen (p=0.3)

### `aug_transforms_vit_strong` (Extra Strong)
Even more aggressive versions of all transforms.

### Selection Logic (in trainer.py)
```python
if aug_strategy == 'auto':
    if arch_type == 'cnn_rnn':        → aug_transforms_cnn
    elif arch_type == 'vit_rgts':     → aug_transforms_vit
    elif arch_type in ['torchvision_vit', 'trocr']:  → aug_transforms_cnn  # pretrained
```

---

## 6. utils/metrics.py

**Location**: `/utils/metrics.py` (56 lines)

### CER (Character Error Rate)

```python
class CER:
    def update(self, prediction, target):
        dist = editdistance.eval(prediction, target)  # Levenshtein distance
        self.total_dist += dist
        self.total_len += len(target)
    
    def score(self):
        return self.total_dist / self.total_len
```

### WER (Word Error Rate)

```python
class WER:
    def __init__(self, mode='tokenizer'):
        # mode='tokenizer': uses NLTK word_tokenize (handles punctuation)
        # mode='space': simple space split
    
    def update(self, prediction, target):
        if self.mode == 'tokenizer':
            target_words = word_tokenize(target)
            pred_words = word_tokenize(prediction)
        else:
            target_words = target.split(' ')
            pred_words = prediction.split(' ')
        
        dist = editdistance.eval(pred_words, target_words)
        # ...
```

---

## 7. utils/attention_extractor.py

**Location**: `/utils/attention_extractor.py` (249 lines)

### AttentionExtractor

High-level interface for ViT-RGTS attention analysis.

```python
extractor = AttentionExtractor(model)

# Full extraction
result = extractor.extract_from_forward(images)
# → attention_maps: List[6] of [B, 8, S, S]
# → register_tokens: [B, R, D]
# → token_norms: [B, S]

# Register analysis
stats = extractor.analyze_register_attention(attn_maps, num_registers=4)
# Per-layer:
#   register_to_patch: [R, P]  — what each register attends to
#   patch_to_register: [P, R]  — how patches attend to registers
#   entropy: scalar             — higher = more distributed attention

# Spatial attention map
patch_attn = extractor.extract_patch_attention(images, layer_idx=-1)
# → [B, Hp, Wp] attention heatmap
```

### `_compute_entropy(attn)`

Shannon entropy of attention distribution:
```python
entropy = -(attn * log(attn + eps)).sum(dim=-1).mean()
# High entropy = distributed/uniform attention
# Low entropy = focused/peaked attention
```

---

## 8. utils/finetuning.py

**Location**: `/utils/finetuning.py` (371 lines)

### `get_vit_layer_groups(net, arch_type)`

Partitions model parameters into ordered groups for LLRD:

```
TorchVision ViT (15 groups):
  Group 0:  embedding (conv_proj, class_token, pos_embedding)
  Group 1:  block_0  (shallowest transformer layer)
  ...
  Group 12: block_11 (deepest transformer layer)
  Group 13: final_ln
  Group 14: adapters (gray_to_rgb, register_tokens)
  Group 15: head (CTC top — highest LR)
```

### `build_finetune_optimizer(net, arch_type, config)`

Creates AdamW with per-layer LR:
```python
for i, (name, params) in enumerate(reversed(groups)):
    lr = head_lr if name == 'head' else base_lr * (decay_rate ** (num_groups - i))
    wd = head_wd if name == 'head' else weight_decay
    param_groups.append({'params': params, 'lr': lr, 'weight_decay': wd})
```

### `GradualUnfreezer`

```python
class GradualUnfreezer:
    def __init__(self, net, arch_type, warmup_frozen=3, unfreeze_every=3):
        # Initially: freeze all backbone layers
        # After warmup: unfreeze deepest layer
        # Every unfreeze_every epochs: unfreeze next deeper layer
    
    def step(self, epoch):
        if epoch <= self.warmup_frozen:
            return  # Keep frozen
        elapsed = epoch - self.warmup_frozen
        layers_to_unfreeze = elapsed // self.unfreeze_every
        # Unfreeze from deepest to shallowest
```

---

## 9. utils/visualizer.py

**Location**: `/utils/visualizer.py` (488 lines)

Hook-based feature visualization for both CNN and transformer models.

### `AttentionVisualizer`

```python
class AttentionVisualizer:
    """
    Registers forward hooks on model layers to capture intermediate features.
    Supports:
    - CNN feature map extraction
    - Transformer attention weight capture
    - GradCAM visualization
    """
```

---

## 10. Config Files

### `configs/baseline_vit_rgts_v2.yaml` (The Recommended Config)

```yaml
# Architecture
arch:
  type: vit_rgts
  use_cnn_stem: true        # Critical: CNN stem for left-to-right tokens
  dim: 256                  # Embedding dimension
  depth: 6                  # Transformer layers
  heads: 8                  # Attention heads
  mlp_dim: 1024             # FFN hidden dim
  num_registers: 4          # Register tokens (sweepable via CLI)
  dropout: 0.1
  emb_dropout: 0.1
  head_type: both           # Dual CTC supervision
  rnn_type: lstm
  rnn_layers: 3
  rnn_hidden_size: 256

# Data
data:
  path: ./data/IAM/processed_lines

# Preprocessing
preproc:
  image_height: 128
  image_width: 1024

# Training
train:
  lr: 0.001
  num_epochs: 80
  batch_size: 8
  augmentation: cnn         # Moderate (strong hurts on small data)
  save_every_k_epochs: 1

# Evaluation
eval:
  batch_size: 8
  wer_mode: tokenizer
```

### `configs/baseline.yaml` (CNN-RNN)

```yaml
arch:
  type: cnn_rnn
  cnn_cfg: [[2, 64], 'M', [3, 128], 'M', [2, 256]]
  head_type: both
  rnn_type: lstm
  rnn_layers: 3
  rnn_hidden_size: 256
  flattening: maxpool
```

### `configs/trocr.yaml`

```yaml
arch:
  type: trocr
  model_name: microsoft/trocr-base-handwritten
  freeze_encoder: true      # Only train CTC head
  image_height: 384
  image_width: 384
```

### CLI Override Examples

```bash
# Change register count
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml arch.num_registers=8

# Change learning rate
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml train.lr=0.0005

# Use synthetic data
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml \
    data.mode=synthetic \
    data.synthetic_path=/path/to/synthetic_processed_lines

# Multiple overrides
python scripts/trainer.py configs/baseline_vit_rgts_v2.yaml \
    arch.num_registers=0 \
    train.num_epochs=50 \
    train.augmentation=vit
```
