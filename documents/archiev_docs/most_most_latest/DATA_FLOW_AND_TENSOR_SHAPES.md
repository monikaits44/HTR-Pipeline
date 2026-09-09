# Data Flow & Tensor Shapes: End-to-End

> Traces the exact tensor shapes and transformations from raw image to CTC prediction, for each architecture.

---

## 1. Overall Pipeline

```
Raw Image File (.png)
    │
    ▼
load_image()         →  [H_orig, W_orig]  float32, values in [0, 1], inverted
    │
    ▼
preprocess()         →  [128, 1024]         float32, aspect-ratio resized + median-padded
    │
    ▼
Dataset.__getitem__  →  [1, 128, 1024]      float32 tensor, unsqueeze(0) for channel
    │
    ▼
DataLoader           →  [B, 1, 128, 1024]   batched input
    │
    ▼
HTRNet.forward()     →  [T, B, nclasses]    CTC logits (T = sequence length)
    │
    ▼
argmax(dim=2)        →  [T, B]              greedy decoded indices
    │
    ▼
decode()             →  string              remove blanks + consecutive duplicates
```

---

## 2. CNN-RNN Baseline

```
Input:    [B, 1, 128, 1024]
              │
              ▼
Conv 7×7 (s=4,2):  [B, 32, 32, 512]     # Initial downsampling
              │
              ▼
BasicBlock×2 (64ch):  [B, 64, 32, 512]
MaxPool 2×2:          [B, 64, 16, 256]
              │
              ▼
BasicBlock×3 (128ch): [B, 128, 16, 256]
MaxPool 2×2:          [B, 128, 8, 128]
              │
              ▼
BasicBlock×2 (256ch): [B, 256, 8, 128]
              │
              ▼
MaxPool [H, 1]:       [B, 256, 1, 128]   # Collapse height → sequence
              │
              ▼ (CTCtopB)
squeeze(2):           [B, 256, 128]
permute(2,0,1):       [128, B, 256]       # [T, B, D] for RNN
              │
    ┌─────────┴──────────┐
    │ RNN path           │ CNN path (aux)
    ▼                    ▼
BiLSTM×3              Conv 1×3
[128, B, 512]         [B, 80, 1, 128]
LayerNorm(opt)        squeeze+permute
Dropout + Linear      [128, B, 80]
[128, B, 80]                │
    │                       │
    ▼                       ▼
logits_rnn            logits_cnn

Loss = CTC(logits_rnn) + 0.1 × CTC(logits_cnn)

T = 128 timesteps for 128 column positions
nclasses = 80 (79 chars + 1 CTC blank)
```

---

## 3. ViT-RGTS v2 (CNN Stem)

```
Input:    [B, 1, 128, 1024]
              │
              ▼ (CNN Stem)
Conv 7×7 (s=4,2):    [B, 32, 32, 512]
Conv 3×3 (s=2,2):    [B, 64, 16, 256]
Conv 3×3 (s=2,2):    [B, 128, 8, 128]
Conv 3×3 (s=2,1):    [B, 256, 4, 128]   # Note: stride 1 in width!
AdaptiveMaxPool(1,_): [B, 256, 1, 128]   # Height → 1
              │
              ▼
squeeze(2):           [B, 128, 256]      # 128 column tokens, each 256-dim
transpose(1,2):       [B, 128, 256]      # [B, Np, D]
              │
              ▼ (Prepend registers)
register_tokens:      [B, R, 256]        # R learnable tokens
cat([reg, patch]):    [B, R+128, 256]    # [B, S, D]
              │
              ▼
+ pos_embed[:, :S]:   [B, S, 256]        # Add positional embeddings
emb_dropout:          [B, S, 256]
              │
              ▼ (TransformerEncoder, 6 layers)
Layer 0..5 each:
  ├── LayerNorm → Q,K,V → MultiHead Self-Attention (8 heads, d_k=32)
  │   Attention shape: [B, 8, S, S]
  ├── + Residual
  ├── LayerNorm → FFN (256→1024→256, GELU)
  └── + Residual
              │
              ▼
Output:       [B, S, 256]
              │
    ┌─────────┴──────────┐
    │ Split               │
    ▼                     ▼
reg_out:  [B, R, 256]   patch_out: [B, 128, 256]
(analysis)               │
                          ▼
                    transpose(0,1): [128, B, 256]  # [T, B, D]
                          │
                          ▼ (CTCtopB, same as above)
                    [128, B, 80]  # logits

S = R + 128  (e.g., 4 + 128 = 132 with 4 registers)
Attention maps: [B, 8, 132, 132] per layer
```

### Token Layout in Sequence

```
Position:  [0] [1] [2] [3] | [4] [5] [6] ... [131]
Content:   R0  R1  R2  R3  | P0  P1  P2  ... P127
           ←─registers──→  | ←───patch tokens────→
```

---

## 4. ViT-RGTS v2 (Original Patches, DEPRECATED)

```
Input:    [B, 1, 128, 1024]
              │
              ▼
patch_embed (Conv 16×16, s=16):  [B, 256, 8, 64]  # 8 rows × 64 cols
              │
              ▼
flatten(2).transpose(1,2):       [B, 512, 256]     # 512 = 8×64 patches
              │
              ▼
+ registers + pos_embed → Transformer → [512, B, 256]

PROBLEM: 512 patches flattened row-major:
  [row0_col0, row0_col1, ..., row0_col63, row1_col0, ...]
  
CTC sees: left→right for first row, then jumps back to left for second row.
This destroys the left-to-right ordering CTC needs → CER stuck at 75%.

SOLUTION: CNN stem collapses height first → only 128 column tokens in order.
```

---

## 5. TorchVision ViT-B/16

```
Input:    [B, 1, 128, 1024]
              │
              ▼
gray_to_rgb (Conv 1×1):  [B, 3, 128, 1024]
              │
              ▼ (ImageNet normalization)
(x - mean) / std:        [B, 3, 128, 1024]
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
              │
              ▼
conv_proj (16×16, s=16): [B, 768, 8, 64]
flatten + transpose:     [B, 512, 768]
              │
              ▼ (Prepend CLS + optional registers)
[CLS, R0..R_r, P0..P511]: [B, 1+R+512, 768]
              │
              ▼
+ interpolated pos_embed: [B, S, 768]
  (original: 14×14=196 → interpolated to 8×64=512)
              │
              ▼ (12-layer Transformer)
              │
              ▼
CLS_out: [B, 768]         # For analysis
patch_out: [B, 512, 768]  # Skip CLS + registers
              │
              ▼
transpose(0,1): [512, B, 768]  # [T, B, D]
              │
              ▼ (CTC head)
[512, B, 80]               # logits

T = 512 timesteps (much more than needed for CTC)
```

---

## 6. TrOCR Encoder

```
Input:    [B, 1, 128, 1024]
              │
              ▼
gray_to_rgb (Conv 1×1):  [B, 3, 128, 1024]
              │
              ▼ (Aspect-ratio resize + pad)
scale = min(384/128, 384/1024) = 0.375
new_h = 48, new_w = 384
resize:   [B, 3, 48, 384]
pad:      [B, 3, 384, 384]   # Pad bottom (48→384)
              │
              ▼ (TrOCR normalization)
(x - 0.5) / 0.5:  [B, 3, 384, 384]
              │
              ▼ (TrOCR ViT encoder, 16×16 patches)
Patches:     384/16 = 24 rows × 24 cols = 576 patches
+ CLS token: [B, 577, 768]
12-layer Transformer
              │
              ▼
Remove CLS:  [B, 576, 768]
Reshape:     [B, 24, 24, 768]    # 2D grid
              │
              ▼
Mean over height (dim=1):  [B, 24, 768]   # Collapse to 1D
              │
              ▼
repeat_interleave(4):      [B, 96, 768]   # Upsample for CTC
              │
              ▼
transpose(0,1):            [96, B, 768]    # [T, B, D]
              │
              ▼ (CTC head)
[96, B, 80]                # logits

T = 96 timesteps (24 columns × 4 upsample)
Note: Only 24 meaningful columns because most of the 384×384 input
is padding (the actual image occupies only 48×384, top portion).
```

---

## 7. CTC Loss Computation

```python
# Outputs from model
output:     [T, B, nclasses]   # T=128 for ViT-RGTS v2

# Target encoding
act_lens:   [B] = [T, T, ..., T]   # All T (full sequence length)

labels:     [sum(label_lens)]       # Concatenated character indices
            # e.g., "Hi" → [c2i['H'], c2i['i']] = [8, 35]
            # Indices are 1-based (0 = CTC blank)

label_lens: [B]                     # Length of each transcription

# Loss
log_probs = log_softmax(output, dim=2)  # [T, B, nclasses]
loss = CTCLoss(reduction='sum')(log_probs, labels, act_lens, label_lens)
loss = loss / batch_size                 # Normalize
```

---

## 8. CTC Decoding

```python
# From model output
logits:    [T, B, nclasses]
indices = logits.argmax(dim=2)  # [T, B]
indices = indices.permute(1, 0) # [B, T]

# For each sample in batch:
tdec = indices[i]  # [T] array of class indices

# Step 1: Remove consecutive duplicates
# [0, 0, 8, 8, 8, 0, 35, 35, 0, 0] → [0, 8, 0, 35, 0]

# Step 2: Remove blanks (index 0)
# [0, 8, 0, 35, 0] → [8, 35]

# Step 3: Map to characters
# [8, 35] → "Hi"  (using i2c dictionary where i2c[8]='H', i2c[35]='i')
```

---

## 9. Attention Map Shapes (forward_explain)

### ViT-RGTS v2 (4 registers, CNN stem)

```
attn_maps:    List of 6 tensors, each [B, 8, 132, 132]
              │
              └── 132 = 4 registers + 128 patches
                  8 attention heads
                  6 transformer layers

token_norms:  [B, 132]   — L2 norm of each token embedding
reg_tokens:   [B, 4, 256] — final register embeddings
grid_size:    (1, 128)     — Hp=1 (height collapsed), Wp=128

# Interpreting attention[layer][batch, head, i, j]:
#   = how much token i attends to token j
#   Rows 0..3:     register tokens attending
#   Rows 4..131:   patch tokens attending
#   Columns 0..3:  attending TO register tokens
#   Columns 4..131: attending TO patch tokens
```

### TorchVision ViT-B/16 (4 registers)

```
attn_maps:    List of 12 tensors, each [B, 12, 517, 517]
              │
              └── 517 = 1 CLS + 4 registers + 512 patches
                  12 attention heads
                  12 transformer layers

token_norms:  [B, 517]
```

### TrOCR

```
attn_maps:    List of 12 tensors, each [B, 12, 577, 577]
              │
              └── 577 = 1 CLS + 576 patches (24×24)
                  12 attention heads
                  12 transformer layers

token_norms:  [B, 577]
```
