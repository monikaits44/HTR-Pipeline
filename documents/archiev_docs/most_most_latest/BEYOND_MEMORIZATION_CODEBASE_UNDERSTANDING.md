# Beyond Memorization — Complete Codebase Understanding

> **Paper**: Gurav, Chanda, Krishnan — ICDAR 2025, Springer (pp. 465–484)  
> **Repo**: [github.com/aniketntnu/Beyond-Memorization](https://github.com/aniketntnu/Beyond-Memorization)  
> **Core Idea**: Training-free style mixing for handwritten text generation using character-level attention map localization and writer embedding injection in a pretrained diffusion model (WordStylist).

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [File-by-File Code Walkthrough](#2-file-by-file-code-walkthrough)
3. [The Diffusion Pipeline in Detail](#3-the-diffusion-pipeline)
4. [The UNet with Style Mixing — `unetVarStleMixExp4.py`](#4-the-unet-with-style-mixing)
5. [Cross-Attention and Attention Map Extraction](#5-cross-attention-and-attention-map-extraction)
6. [The Style Mixing Mechanism — Character-Level Injection](#6-the-style-mixing-mechanism)
7. [Attention Map Visualization — `utils/saveAttentionMaps.py`](#7-attention-map-visualization)
8. [Data Flow: End-to-End Inference Trace](#8-data-flow-end-to-end)
9. [Output Structure and File Naming](#9-output-structure)
10. [Key Differences from Our HTR Pipeline](#10-key-differences)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  INFERENCE PIPELINE                     │
│                                                         │
│  IAM word images ──→ VAE encoder ──→ latent z₀          │
│                                                         │
│  Word label "hello" ──→ CharacterEncoder ──→ context     │
│                          (embed + positional + attn)     │
│                                                         │
│  Writer ID ──→ nn.Embedding(339, 1280) ──→ emb_writer   │
│                                                         │
│  Timestep t ──→ sinusoidal ──→ MLP ──→ emb_time         │
│                                                         │
│  emb = emb_time + emb_writer                            │
│                                                         │
│  ┌─── DENOISING LOOP (600 steps, reverse) ───────────┐  │
│  │                                                    │  │
│  │  noise ──→ UNet(x_t, emb, context) ──→ ε_pred     │  │
│  │                                                    │  │
│  │  FIRST PASS (charIndex=-1): normal generation      │  │
│  │    → returns predicted_noise + attn2Original       │  │
│  │                                                    │  │
│  │  SECOND PASS (charIndex≥0): style-mixed generation │  │
│  │    → uses attn2Original to locate char boundaries  │  │
│  │    → applies mask-based writer embedding injection │  │
│  │    → returns style-mixed predicted_noise           │  │
│  │                                                    │  │
│  │  x_{t-1} = denoise_step(x_t, ε_pred)              │  │
│  └────────────────────────────────────────────────────┘  │
│                                                         │
│  x_0 ──→ VAE decoder ──→ generated image (64×256 RGB)   │
│                                                         │
│  Optional: HTR/OCR filter (keep only correct words)     │
└─────────────────────────────────────────────────────────┘
```

### Components

| Component | Source | Purpose |
|-----------|--------|---------|
| **WordStylist** | Pretrained `ema_ckpt.pt` | Base latent diffusion model for handwriting generation |
| **VAE** | Stable Diffusion v1.5 `vae/` | Encode/decode between pixel and latent space (64×256 → 8×32 latent) |
| **UNet** | `unetVarStleMixExp4.py` | Noise prediction with cross-attention and style injection |
| **CharacterEncoder** | Inside UNet file | Word label → context embeddings via character embedding + positional encoding + attention |
| **Writer Embedding** | `nn.Embedding(339, 1280)` | Maps writer ID to style embedding vector |
| **HTR/OCR** | `htr/models.py` (HTR-best-practices) | Optional post-generation filter — keeps only correctly recognized words |
| **Attention Maps** | `utils/saveAttentionMaps.py` | Extract, threshold, and visualize per-character cross-attention heatmaps |

---

## 2. File-by-File Code Walkthrough

### 2.1 `regFrmTrnVariStyleMixOcr.py` — Main Inference Script (~1200 lines)

This is the entry point. Key sections:

| Section | Lines (approx) | What it does |
|---------|----------------|-------------|
| **Imports & config** | 1–50 | Import UNet, HTR model, utilities; load `config.py` |
| **`label_padding()`** | 50–70 | Pad character labels to `MAX_CHARS=25` with PAD_TOKEN |
| **`labelDictionary()`** | 70–100 | Build letter↔index mappings (52 chars: A-Z, a-z) |
| **`IAMDataset`** | 100–350 | PyTorch Dataset for IAM word images with writer IDs, labels, optional VAE precomputed latents |
| **`EMA`** | 350–390 | Exponential moving average for model parameters |
| **`Diffusion`** | 390–500 | DDPM diffusion: noise schedule, `noise_images()`, `sampling3()` (main sampling loop) |
| **`process_sampling()`** | 500–530 | Wrapper that calls `diffusion.sampling3()` |
| **`callOCR()`** | 530–560 | Run HTR inference → CTC decode → compare with ground truth |
| **`process_tensors_and_ocr()`** | 560–620 | Post-process: resize, OCR filter, collect correct images |
| **`train()`** (really inference) | 620–900 | Main inference loop over dataloader batches |
| **`createDataLoader1()`** | 900–1050 | Build IAM dataset + dataloader with custom sampler |
| **`main()`** | 1050–1200 | Parse args, load models, run epochs with charLocation = -1, 0, 1, 2, 3 |

### 2.2 `unetVarStleMixExp4.py` — Modified UNet (~1200 lines)

The core model with attention extraction and style injection. [Detailed in Section 4](#4-the-unet-with-style-mixing).

### 2.3 `unetAuthor.py` — Original WordStylist UNet

The unmodified UNet from WordStylist. Used as reference/fallback.

### 2.4 `unet.py` — Attention-extraction-only UNet

A UNet variant that returns attention maps but doesn't do style mixing.

### 2.5 `config.py` — Configuration

```python
device = "cuda:0"
MAX_CHARS = 25
iam_path = ""          # now via CLI --iam_path
save_path = ""         # now via CLI --save_path
gt_train = "./gt/gany.filter27"
saveModelName = "ema_ckpt.pt"
```

All paths are now CLI arguments (no hardcoded cluster paths).

### 2.6 `utils/saveAttentionMaps.py` — Attention Visualization (~800 lines)

Multiple functions for saving per-character attention maps. [Detailed in Section 7](#7-attention-map-visualization).

### 2.7 `utils/tensorProcess.py` — Tensor Processing

- `torchProcess()`: Convert tensor list to proper format
- `tensor_centered()`: Center-crop/pad tensor to fixed size (64×256)

### 2.8 `htr/` — HTR Model for OCR Filtering

Contains `HTRNet` from [HTR-best-practices](https://github.com/georgeretsi/HTR-best-practices). Used for post-generation quality control.

### 2.9 `ResPhoSCNetZSL/` — PHOSC Features

Optional module for phonological/visual features (PHOSC embeddings). Not used in the main style mixing experiments.

### 2.10 `gt/gany.filter27` — Ground Truth File

Format: `writerID,imageID label` — one line per word, used to build the dataset.

---

## 3. The Diffusion Pipeline

### 3.1 Noise Schedule

```python
class Diffusion:
    noise_steps = 600
    beta_start = 1e-4
    beta_end = 0.02
    
    beta = linspace(beta_start, beta_end, noise_steps)   # [600]
    alpha = 1 - beta                                       # [600]
    alpha_hat = cumprod(alpha)                             # [600] cumulative product
```

### 3.2 Forward Process (Add Noise)

$$x_t = \sqrt{\hat{\alpha}_t} \cdot x_0 + \sqrt{1 - \hat{\alpha}_t} \cdot \epsilon$$

### 3.3 Reverse Process (Denoise) — `sampling3()`

This is the critical function. For each timestep from 600 to 1:

```python
for i in reversed(range(1, 600)):
    t = tensor([i] * batch_size)
    
    if charLocation >= 0:  # STYLE MIXING MODE
        # Pass 1: Get attention maps from original writer
        predicted_noise, attn1, attn2, attn3, attn2Original = model(
            x, original_images=latents, timesteps=t,
            context=text_features, y=writer_ids,
            y1=shuffled_writers, charIndx=charLocation
        )
        
        # Pass 2: Use attention maps to guide style injection
        predicted_noise, _, _, _, _ = model(
            x, original_images=latents, timesteps=t,
            context=text_features, y=writer_ids,
            y1=shuffled_writers, Attnmap=attn2Original,
            charIndx=charLocation
        )
    else:  # NORMAL MODE (no style mixing)
        predicted_noise, attn1, attn2, attn3, attn2Original = model(
            x, original_images=latents, timesteps=t,
            context=text_features, y=writer_ids
        )
    
    # DDPM update
    x = (1/√α) * (x - (1-α)/√(1-α̂) * predicted_noise) + √β * noise
```

### 3.4 Latent Space

Images are processed in VAE latent space:
- Input: `[B, 3, 64, 256]` RGB image
- Encoded: `[B, 4, 8, 32]` latent representation
- UNet works in this 8×32 spatial resolution
- Decoded back to pixel space after denoising

---

## 4. The UNet with Style Mixing — `unetVarStleMixExp4.py`

### 4.1 Architecture Overview

```
Input: x [B, 4, 8, 32] (latent)
       emb [B, 1280] (time + writer embedding)
       context [B, MAX_CHARS, 320] (character embeddings)

INPUT BLOCKS:
  → Conv2d(4, 320)
  → ResBlock(320, 1280) + SpatialTransformer(320)  ← cross-attention here
  → Downsample(320 → 320)
  → ResBlock(320, 1280) + SpatialTransformer(320)  ← cross-attention here

MIDDLE BLOCK:
  → ResBlock(320, 1280) + SpatialTransformer(320)  ← cross-attention here (MAIN)
  → ResBlock(320, 1280)

OUTPUT BLOCKS:
  → ResBlock(640→320, 1280) + SpatialTransformer(320)  ← cross-attention here
  → Upsample(320)
  → ResBlock(640→320, 1280) + SpatialTransformer(320)  ← cross-attention here

Output: [B, 4, 8, 32] (predicted noise)
        + attn1, attn2, attn3, attn2Original
```

### 4.2 `CharacterEncoder` — Text Conditioning

```python
class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len):
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 54 → 320
        self.attention = Word_Attention(hidden_size, hidden_size)
        self.positional_encoding = sinusoidal_PE(max_seq_len, hidden_size)
    
    def forward(self, x):
        # x: [B, MAX_CHARS] integer token IDs
        x = self.embedding(x)                    # [B, 25, 320]
        x += self.positional_encoding[:25, :]    # + sinusoidal PE
        word_emb = self.attention(x)             # [B, 25, 320] self-attention
        return word_emb
```

Output: `context = [B, 25, 320]` — one 320-dim vector per character position.

### 4.3 Writer Embedding

```python
self.label_emb = nn.Embedding(339, 1280)  # 339 IAM writers → 1280-dim

# In forward():
emb = time_embed + label_emb(writer_id)  # [B, 1280]
```

### 4.4 `CrossAttention` — Where Attention Maps Come From

```python
class CrossAttention(nn.Module):
    def forward(self, x, context=None):
        # x: [B, H*W, C] image features (query)
        # context: [B, MAX_CHARS, C] character embeddings (key/value)
        
        q = self.to_q(x)          # [B, H*W, inner_dim]
        k = self.to_k(context)    # [B, MAX_CHARS, inner_dim]
        v = self.to_v(context)    # [B, MAX_CHARS, inner_dim]
        
        # Reshape for multi-head: (B*heads) × N × d_head
        q, k, v = rearrange('b n (h d) -> (b h) n d')
        
        # Compute attention scores
        sim = einsum('b i d, b j d -> b i j', q, k) * scale
        # sim: [(B*heads), H*W, MAX_CHARS]
        
        attn = softmax(sim, dim=-1)
        # attn: [(B*heads), H*W, MAX_CHARS]
        #   Each spatial position attends to each character
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        
        # Reshape attn for return: [B, heads, H*W, MAX_CHARS]
        attn = rearrange('(b h) 1 n d -> b h n d', h=heads)
        
        return self.to_out(out), attn
```

**This is the KEY mechanism**: The attention matrix `attn[b, h, spatial_pos, char_idx]` tells you how much each spatial position in the image attends to each character. This is what makes per-character localization possible.

### 4.5 How Attention is Reshaped for Visualization

In the `SpatialTransformer`:
```python
# attn2 shape from CrossAttention: [B, heads, H*W, MAX_CHARS]
# Reshape to spatial grid:
attn2 = rearrange(attn2, 'b h (height width) c -> b h height width c',
                   height=h, width=w)
# Result: [B, heads, 8, 32, 25]  (for 8×32 latent, 25 chars)
```

In the `UNetModel.forward()`:
```python
# Sum over attention heads:
attn2 = attn2.sum(dim=1)           # [B, 8, 32, 25]
attn2Original = attn2.clone()      # Save for style injection

# Upsample to image resolution:
attn2 = F.interpolate(
    attn2.permute(0,3,1,2),        # [B, 25, 8, 32]
    scale_factor=(16, 16),         # 8×16=128, but displayed as 64×256 by swapping
    mode="nearest"
)
attn2 = attn2.permute(0,2,3,1)    # [B, 128, 512, 25] or similar
```

---

## 5. Cross-Attention and Attention Map Extraction

### 5.1 What the attention maps represent

The cross-attention computes:

$$\text{attn}[b, h, (y,x), c] = \frac{\exp(q_{(y,x)} \cdot k_c / \sqrt{d})}{\sum_{c'} \exp(q_{(y,x)} \cdot k_{c'} / \sqrt{d})}$$

Where:
- $(y,x)$ is a spatial position in the UNet feature map
- $c$ is a character index (0 to MAX_CHARS-1)
- $q_{(y,x)}$ comes from the image features (projected)
- $k_c$ comes from the character embedding (projected)

**Interpretation**: `attn[b, h, (y,x), c]` = "how much does spatial position (y,x) in the image attend to character c". High values mean this spatial region is associated with this character.

### 5.2 Multi-resolution attention

The UNet has cross-attention at multiple resolutions:
- Input blocks: 8×32 feature map → `attn` at 8×32 resolution
- Middle block: 8×32 feature map → `attn` at 8×32 resolution (this is `attn2`, the main one used)
- Output blocks: 8×32 and potentially higher

The paper primarily uses the **middle block** cross-attention (`attn2Original`) because:
1. It has the richest feature representation
2. It's at the bottleneck where global information converges
3. It provides the most reliable character localization

### 5.3 Finding character spatial boundaries — `get_blob_centroids()`

```python
def get_blob_centroids(self, attn_tensor, sigma_multiplier=2):
    # attn_tensor: [B, H, W, MAX_CHARS]
    
    for b in range(batch_size):
        for c in range(num_chars):
            curr_attn = attn_tensor[b, :, :, c]  # [H, W]
            
            # Normalize to [0, 1]
            curr_attn = (curr_attn - min) / (max - min)
            
            # Threshold: mean + sigma_multiplier * 2 * std
            threshold = mean + sigma_multiplier * 2 * std
            thresholded = curr_attn >= threshold
            
            # Find connected components (blobs)
            labeled_map, num_features = scipy.ndimage.label(thresholded)
            
            # Find the blob with highest peak attention
            max_blob = argmax within thresholded region
            blob_mask = labeled_map == max_blob_label
            
            # Compute centroid of that blob
            centroid = scipy.ndimage.center_of_mass(blob_mask)
            
            max_x_coords[b, c] = centroid[1]  # X position
            max_y_coords[b, c] = centroid[0]  # Y position
    
    return max_x_coords, max_y_coords
    # max_x_coords: [B, MAX_CHARS] — the x-coordinate of each character's center
```

**This is the bridge between attention maps and style injection**: `max_x_coords[:, charIndex]` gives the horizontal position where character `charIndex` is centered. Everything left of this position belongs to one writer's style, everything right belongs to another.

---

## 6. The Style Mixing Mechanism — Character-Level Injection

### 6.1 The two-pass approach

For each denoising step:

**Pass 1** — Normal forward, get attention maps:
```python
predicted_noise, _, _, _, attn2Original = model(
    x, ..., y=original_writer, charIndx=charLocation)
```

**Pass 2** — Use attention maps to create spatial masks:
```python
predicted_noise, _, _, _, _ = model(
    x, ..., y=original_writer, y1=shuffled_writer,
    Attnmap=attn2Original, charIndx=charLocation)
```

### 6.2 Inside `ResBlock._forward()` — The injection point

```python
def _forward(self, x, emb, extraDict):
    h = self.in_layers(x)
    
    emb_out = self.emb_layers(emb)      # Original writer embedding [B, C, 1, 1]
    charIndx = extraDict["charIndx"]
    
    if charIndx >= 0:
        emb_out1 = self.emb_layers(extraDict["emb1"])  # Shuffled writer embedding
    
    max_x_coords = extraDict["max_x_coords"]
    
    if max_x_coords is not None and charIndx >= 0:
        h = self.add_emb_to_h_character_based(
            h, emb_out, emb_out1, max_x_coords, charIndx, extraDict)
    else:
        h = h + emb_out  # Standard: add writer embedding uniformly
    
    h = self.out_layers(h)
    return self.skip_connection(x) + h
```

### 6.3 `add_emb_to_h_character_based()` — The core injection formula

```python
def add_emb_to_h_character_based(self, h, emb_out, emb_out1, max_x_coords, 
                                  char_index, extraDict):
    # h: [B, C, 8, 32] — feature map
    # emb_out: [B, C, 1, 1] — original writer's embedding  
    # emb_out1: [B, C, 1, 1] — shuffled writer's embedding
    # max_x_coords: [B, MAX_CHARS] — character positions from attention
    # char_index: int — which character to split at
    
    # Get the x-coordinate for this character for each batch element
    selected_x = max_x_coords[:, char_index]  # [B]
    
    # Expand embeddings to full spatial dimensions
    expanded_emb = emb_out.expand(-1, -1, height, width)    # [B, C, 8, 32]
    expanded_emb1 = emb_out1.expand(-1, -1, height, width)  # [B, C, 8, 32]
    
    # Create spatial masks based on character position
    width_indices = torch.arange(width)  # [0, 1, ..., 31]
    
    mask  = width_indices < selected_x.view(B, 1, 1, 1)  # LEFT of character
    mask1 = width_indices > selected_x.view(B, 1, 1, 1)  # RIGHT of character
    
    # THE INJECTION FORMULA:
    h = mask * expanded_emb + h + mask1 * expanded_emb1
    
    #   LEFT of char position: h + original_writer_embedding
    #   RIGHT of char position: h + shuffled_writer_embedding
    #   AT char position: just h (no style injection)
    
    return h
```

**Visually**:
```
Character position (from attention map): x=15
                     ↓
Feature map: [....ORIGINAL_STYLE.....][....SHUFFLED_STYLE.....]
             ← emb_out applied here → ← emb_out1 applied here →
Position:    0  1  2  ...  14  15  16  17  ...  31
```

### 6.4 The charLocation sweep

The main script runs inference with `charLocation = -1, 0, 1, 2, 3`:

| charLocation | Behavior |
|-------------|----------|
| -1 | No style mixing. Original writer only. Baseline. |
| 0 | Split at character 0 position. Almost entirely shuffled writer style. |
| 1 | Split at character 1 position. First char is original, rest is shuffled. |
| 2 | Split at character 2 position. First 2 chars original, rest shuffled. |
| 3 | Split at character 3 position. First 3 chars original, rest shuffled. |

This produces **progressive style variability** — more characters from the original writer's style → more similarity to original.

---

## 7. Attention Map Visualization — `utils/saveAttentionMaps.py`

### 7.1 Overview

The file contains multiple visualization functions, evolved over development:

| Function | Purpose | Used in production? |
|----------|---------|-------------------|
| `save_Attention()` | Basic overlay, saves per-layer | No (early version) |
| `save_Attention2()` | Threshold-based, max activation point | No (intermediate) |
| `save_Attention2_above_threshold()` | sigma-based threshold | No (intermediate) |
| `save_Attention2_updated()` | Rightmost extent of attention | No (intermediate) |
| **`save_Attention2_with_blobs()`** | **Blob detection + bounding box + heatmap** | **Yes — main function** |
| `save_images_and_attention_maps_1_()` | Full pipeline: save images + call blob viz | **Yes — called from train()** |
| `get_blob_centroids()` | Extract character positions (standalone) | Yes (in UNet) |
| `extract_character_from_attention()` | Extract masked character images | Optional |

### 7.2 The production visualization: `save_Attention2_with_blobs()`

Step-by-step for each character:

```python
def save_Attention2_with_blobs(txt, tempIndx, currImg, attn1, attn2, attn3,
                                attn2Original, imgNameWrite, attenMapDict, dumpPath):
    # attn3: [B, H, W, MAX_CHARS] — upsampled attention maps
    # currImg: PIL.Image — the generated word image
    
    mapPath = os.path.join(dumpPath, "attentionMaps")
    os.makedirs(mapPath, exist_ok=True)
    
    for charIndx in range(len(txt)):
        # 1. Extract attention for this character
        currCharAttn = attn3[:, :, :, charIndx]       # [B, H, W]
        currAttnImg = currCharAttn[tempIndx, :, :]    # [H, W]
        
        # 2. Normalize to [0, 1]
        currAttnImg = (currAttnImg - min) / (max - min)
        
        # 3. Threshold: mean + std
        threshold = mean + std
        thresholded = currAttnImg >= threshold
        
        # 4. Find connected components (blobs)
        labeled_map, num_features = scipy.ndimage.label(thresholded)
        
        # 5. Find the blob with highest peak
        max_blob_label = labeled_map at argmax location
        blob_mask = labeled_map == max_blob_label
        
        # 6. Get bounding box
        rows, cols = np.where(blob_mask)
        min_x, max_x = cols.min(), cols.max()
        min_y, max_y = rows.min(), rows.max()
        
        # 7. Create masked image (character isolation)
        masked_img = np.full_like(currImg_np, 255)  # white background
        mask = currAttnImg >= threshold + std
        masked_img[mask] = currImg_np[mask]
        
        # 8. Save attention heatmap overlay
        fig, ax = plt.subplots(figsize=(256/100, 64/100))
        ax.imshow(currImg_np)
        ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)
        ax.set_title(f"Attention Map for Character {txt[charIndx]}")
        ax.axis('off')
        plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}.png")
        plt.close()
```

### 7.3 Visualization output

For word "hello":
```
attentionMaps/
├── word_h_0_h_char_att.png    ← heatmap showing model attending to 'h' region
├── word_h_1_e_char_att.png    ← heatmap showing model attending to 'e' region
├── word_h_2_l_char_att.png    ← heatmap attending to first 'l'
├── word_h_3_l_char_att.png    ← heatmap attending to second 'l'
└── word_h_4_o_char_att.png    ← heatmap attending to 'o' region
```

Each image shows:
- The generated word image as background
- A semi-transparent `inferno` colormap overlay showing attention intensity
- High intensity (bright yellow/white) at the spatial region corresponding to that character
- Low intensity (dark purple/black) elsewhere

---

## 8. Data Flow: End-to-End Inference Trace

```
INPUT:
  IAM image: "a01-000u-01-00.png" (64×256 grayscale, word "A")
  Writer ID: 000 → mapped to index 42
  Label: "A" → padded to [0+54, 52, 52, ..., 52] (25 tokens)

STEP 1: VAE Encode
  [B, 3, 64, 256] → VAE encoder → [B, 4, 8, 32] latent × 0.18215

STEP 2: Character Encoding
  [B, 25] integer tokens
    → nn.Embedding(54, 320)  → [B, 25, 320]
    → + positional_encoding   → [B, 25, 320]
    → Word_Attention           → [B, 25, 320] context

STEP 3: Writer Embedding
  writer_id=42 → nn.Embedding(339, 1280) → [B, 1280]
  emb = time_embed(t) + writer_embed     → [B, 1280]

STEP 4: Denoising Loop (600 steps)
  For each step i = 599, 598, ..., 1:
    UNet forward:
      Input blocks:  h → [B, 320, 8, 32]
      Middle block:  h → [B, 320, 8, 32], cross-attention returns attn2
      Output blocks: h → [B, 320, 8, 32] → [B, 4, 8, 32]
      
    Cross-attention in middle block:
      q = project(h_flat)        → [B*4, 256, 80]  (4 heads, 256=8*32 positions, 80 dim/head)
      k = project(context)       → [B*4, 25, 80]   (25 characters)
      v = project(context)       → [B*4, 25, 80]
      
      attn = softmax(q @ k.T / √80)  → [B*4, 256, 25]
      
      Reshaped: → [B, 4, 8, 32, 25]
      Sum over heads: → [B, 8, 32, 25]  ← attn2Original
      Upsampled: → [B, 128, 512, 25]    ← attn2 (for visualization)
      
    For charLocation ≥ 0 (second pass):
      get_blob_centroids(attn2Original) → max_x_coords [B, 25]
      In each ResBlock:
        mask = width < max_x_coords[:, charIdx]
        h = mask*emb_original + h + (1-mask)*emb_shuffled

STEP 5: VAE Decode  
  [B, 4, 8, 32] / 0.18215 → VAE decoder → [B, 3, 64, 256] → clamp(0,1)

STEP 6: OCR Filter (optional)
  HTRNet inference → CTC decode → compare with ground truth
  Keep only correctly recognized words

STEP 7: Save
  Generated image: output/noChange/word.png  or  output/charIndex_0/word.png
  Attention maps:  output/noChange/attentionMaps/word_c_0_h.png
```

---

## 9. Output Structure

```
output/
├── noChange/                          ← charLocation=-1 (original style)
│   ├── imgID_writerID__New__word_0.png
│   └── attentionMaps/
│       ├── imgID_word_0_h_0_char_att.png
│       └── ...
├── charIndex_0/                       ← charLocation=0 (split at char 0)
│   ├── imgID_writerID_shuffledID_0__word_0.png
│   └── attentionMaps/
│       └── ...
├── charIndex_1/                       ← charLocation=1
├── charIndex_2/                       ← charLocation=2
└── charIndex_3/                       ← charLocation=3
```

Filename format: `{imageID}_{writerID}_{shuffledWriterID}_New__{word}_{epoch}.png`

---

## 10. Key Differences from Our HTR Pipeline

| Aspect | Beyond Memorization | Our HTR Pipeline |
|--------|-------------------|-----------------|
| **Task** | **Generation** (create handwriting images) | **Recognition** (read handwriting) |
| **Model** | UNet (denoising diffusion) | ViT-RGTS (CTC recognition) |
| **Attention type** | **Cross-attention**: image→text query | **Self-attention**: token↔token |
| **What attends** | Image spatial positions attend to character embeddings | All tokens (registers + patches) attend to all tokens |
| **Attention shape** | `[B, heads, H×W, MAX_CHARS]` | `[B, heads, S, S]` where S = R+P |
| **Per-character maps** | Direct: `attn[:,:,:,charIdx]` gives spatial map for char c | Indirect: must map CTC output positions to characters |
| **Writer style** | Explicit `nn.Embedding(339, 1280)` | Not modeled (task is recognition, not generation) |
| **Training** | Pretrained WordStylist, inference-only modifications | Trained from scratch on IAM |
| **Image size** | 64×256 RGB | 128×1024 grayscale |
| **Latent space** | Yes (VAE, 8×32) | No (direct pixel processing via CNN stem) |
