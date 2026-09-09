# Attention Map Visualization: Beyond-Memorization → HTR Pipeline Adaptation

> **Goal**: Understand the per-character attention map visualization from Beyond-Memorization, and create equivalent visualizations for our ViT-RGTS HTR pipeline with register tokens.

---

## Table of Contents

1. [What Beyond-Memorization Does (Summary)](#1-what-beyond-memorization-does)
2. [Why Their Approach Doesn't Directly Apply](#2-why-it-doesnt-directly-apply)
3. [What We Already Have](#3-what-we-already-have)
4. [Three Visualization Approaches for Our Pipeline](#4-three-visualization-approaches)
5. [Approach A: CTC-Aligned Character Attention Maps](#5-approach-a-ctc-aligned)
6. [Approach B: Register Token Attention Heatmaps](#6-approach-b-register-attention)
7. [Approach C: Patch-to-Patch Self-Attention Flow](#7-approach-c-patch-self-attention)
8. [Complete Implementation Code](#8-implementation-code)
9. [How to Run](#9-how-to-run)
10. [Expected Output and Comparison](#10-expected-output)

---

## 1. What Beyond-Memorization Does

Their system produces **per-character spatial heatmaps** showing which image region the diffusion model associates with each character in the word.

### Their Pipeline

```
UNet cross-attention: image features (query) × character embeddings (key/value)
    → attn: [B, heads, H×W, MAX_CHARS]
    → For character c: attn[:, :, :, c] is a spatial heatmap
    → Upsample to image resolution
    → Overlay on generated image with 'inferno' colormap
    → Save per-character PNG files
```

### Their Key Function: `save_Attention2_with_blobs()`

```python
for charIndx in range(len(word)):
    # 1. Get attention for this character: [H, W]
    char_attn = attn[:, :, :, charIndx][sample_idx]
    
    # 2. Normalize to [0,1]
    char_attn = (char_attn - min) / (max - min)
    
    # 3. Threshold: mean + std
    threshold = mean + std
    
    # 4. Connected components → find strongest blob
    labeled, n = scipy.ndimage.label(char_attn >= threshold)
    blob_mask = labeled == strongest_blob_label
    
    # 5. Bounding box of blob
    rows, cols = np.where(blob_mask)
    min_x, max_x, min_y, max_y = cols.min(), cols.max(), rows.min(), rows.max()
    
    # 6. Overlay heatmap on image
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(char_attn, cmap='inferno', alpha=0.5)
    ax.set_title(f"Attention for '{char}'")
    plt.savefig(f"attentionMaps/word_{charIndx}_{char}.png")
```

### What makes this work for them

Their model has **cross-attention** where image spatial positions (queries) attend to character embeddings (keys). This means `attn[:,:,:,c]` directly gives a spatial heat map for character `c`.

---

## 2. Why Their Approach Doesn't Directly Apply

| Aspect | Beyond-Memorization | Our HTR Pipeline |
|--------|-------------------|-----------------|
| **Attention type** | Cross-attention (image → text) | Self-attention (token ↔ token) |
| **Query source** | Image spatial features | All tokens (registers + column patches) |
| **Key/Value source** | Character embeddings (one per char) | Same tokens (registers + column patches) |
| **Direct char→space mapping?** | **Yes** — attn[:,:,:,c] is the spatial map for char c | **No** — must derive from CTC alignment |
| **Attention shape** | `[B, H, spatial, chars]` | `[B, H, S, S]` where S = R + num_patches |

**The fundamental difference**: They have explicit character slots in their cross-attention. We have self-attention where tokens attend to other tokens, with no explicit character dimension.

### How to bridge this gap

We need to **create** the character↔space mapping by combining:
1. **CTC decoder output** → which characters are predicted at which time positions
2. **Self-attention maps** → which token positions attend to which other positions
3. **Register attention** → how register tokens aggregate information from patches

---

## 3. What We Already Have

### 3.1 `models.py` — `forward_explain()`

Returns:
```python
logits,        # [T, B, C] — CTC logits per time step
reg_tokens,    # [B, R, D] — register token embeddings
attn_maps,     # List[L] of [B, H, S, S] — per-layer self-attention matrices
token_norms,   # [B, S] — L2 norm of each token at final layer
grid_size      # (Hp, Wp) — spatial grid
```

Where `S = R + num_patches` (e.g., 4 + 128 = 132 tokens).

### 3.2 `scripts/trainer.py` — `extract_attention_weights()`

Already saves:
- Per-layer attention maps as `.npy` files
- Token norms
- Register tokens
- Ground truth text

But only saves raw numpy arrays — **no visualization**.

### 3.3 `utils/attention_extractor.py` — `AttentionExtractor`

Has infrastructure for:
- Register-to-patch attention analysis
- Patch-to-register attention analysis
- Entropy computation
- Layer-wise attention flow

But **no per-character heatmap visualization**.

### 3.4 Existing visualization scripts

- `scripts/postprocessing/comparative/visualize_attention_maps.py` — raw token×token heatmaps with register boundary lines. Good for understanding attention structure, but not per-character.

---

## 4. Three Visualization Approaches for Our Pipeline

### Approach A: CTC-Aligned Character Attention Maps (Closest to Beyond-Memorization)

Use CTC output to assign characters to time positions, then extract the self-attention rows for those positions and overlay on the image.

**Result**: Per-character heatmaps similar to Beyond-Memorization's output.

### Approach B: Register Token Attention Heatmaps

Visualize what each register token attends to across the image. This is unique to our architecture and maps to the "registers develop specialization" finding from the Registers paper.

**Result**: Per-register spatial heatmaps showing learned specialization.

### Approach C: Patch-to-Patch Self-Attention Flow

Visualize how each column patch attends to other column patches, creating an attention flow visualization along the text sequence.

**Result**: Attention flow matrix showing which parts of the image interact.

---

## 5. Approach A: CTC-Aligned Character Attention Maps

### The Idea

1. Run `forward_explain()` to get CTC logits + attention maps
2. CTC-decode the logits to get the predicted text
3. For each predicted character, find which time positions (columns) produced it
4. For those time positions, look at their self-attention pattern (which other columns they attend to)
5. Map the attention pattern back to image spatial coordinates
6. Overlay as a heatmap on the input image

### Step-by-Step

```python
# STEP 1: Forward pass
logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(image)
# logits: [T=128, B=1, C=54]  (128 time steps, 54 classes including blank)
# attn_maps: 6 × [B=1, H=8, S=132, S=132]  (6 layers, 8 heads, 132 tokens)

# STEP 2: CTC decode — find character-to-position mapping
probs = logits[:, 0, :].softmax(dim=-1)  # [128, 54]
pred_indices = probs.argmax(dim=-1)       # [128]

# CTC alignment: group consecutive same predictions, skip blanks
char_positions = {}  # char_idx → list of time positions
current_char_idx = 0
for t in range(128):
    pred = pred_indices[t].item()
    if pred == 0:  # blank
        continue
    if t > 0 and pred == pred_indices[t-1].item():  # repeat
        continue
    # New character at position t
    if current_char_idx not in char_positions:
        char_positions[current_char_idx] = []
    char_positions[current_char_idx].append(t)
    # Extend to cover the full span of this character
    # Look ahead for repeats
    for t2 in range(t+1, 128):
        if pred_indices[t2].item() == pred:
            char_positions[current_char_idx].append(t2)
        else:
            break
    current_char_idx += 1

# STEP 3: Extract attention for each character
# Use last layer, average over heads
R = model.backbone.num_registers  # e.g., 4
last_attn = attn_maps[-1][0].mean(dim=0)  # [S, S] averaged over heads
# Focus on patch-to-patch attention (skip register rows/cols)
patch_attn = last_attn[R:, R:]  # [128, 128]

for char_idx, positions in char_positions.items():
    # Average attention FROM these positions TO all other positions
    char_attn = patch_attn[positions, :].mean(dim=0)  # [128]
    
    # char_attn[j] = "how much does character char_idx attend to column j"
    # Map back to image width: each column token covers ~8 pixels
    # Upsample to image width
    char_heatmap = char_attn.numpy()
    char_heatmap = np.repeat(char_heatmap, image_width // 128)  # [1024]
    
    # Expand to full 2D: [H, W]
    char_heatmap_2d = np.tile(char_heatmap, (image_height, 1))  # [128, 1024]
    
    # Normalize
    char_heatmap_2d = (char_heatmap_2d - char_heatmap_2d.min()) / 
                       (char_heatmap_2d.max() - char_heatmap_2d.min())
    
    # Overlay on image
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(image_np, cmap='gray')
    ax.imshow(char_heatmap_2d, cmap='inferno', alpha=0.5)
    ax.set_title(f"Attention for character '{decoded_char}'")
    ax.axis('off')
    plt.savefig(f"attn_char_{char_idx}_{decoded_char}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
```

### Why this works

In our CNN-stem ViT, each column token corresponds to a horizontal strip of the image (~8 pixels wide). Self-attention between column tokens tells us which image columns interact during the transformer computation. By finding which column token positions produce each CTC character prediction, we can see what spatial context the model uses for each character.

---

## 6. Approach B: Register Token Attention Heatmaps

### The Idea

Register tokens attend to all patch tokens. Their attention patterns show which spatial regions they specialize in — analogous to Figure 9 in the Registers paper.

```python
# Register-to-patch attention
R = 4  # number of registers
last_attn = attn_maps[-1][0]  # [H, S, S] for sample 0

for reg_idx in range(R):
    for head_idx in range(num_heads):
        # Attention FROM register reg_idx TO all patches
        reg_attn = last_attn[head_idx, reg_idx, R:]  # [128]
        
        # Upsample to image width
        heatmap = reg_attn.numpy()
        heatmap = np.repeat(heatmap, image_width // 128)
        heatmap_2d = np.tile(heatmap, (image_height, 1))
        
        # Normalize and overlay
        heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min())
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.imshow(image_np, cmap='gray')
        ax.imshow(heatmap_2d, cmap='inferno', alpha=0.5)
        ax.set_title(f"Register {reg_idx}, Head {head_idx}")
        ax.axis('off')
        plt.savefig(f"reg_{reg_idx}_head_{head_idx}.png")
        plt.close()
```

### What this reveals

- Do different registers specialize in different text regions?
- Do some registers attend to character strokes while others attend to spacing?
- Does the pattern change across layers (early vs late)?
- How does the number of registers (0, 4, 7, 16) affect specialization?

---

## 7. Approach C: Patch-to-Patch Self-Attention Flow

### The Idea

Visualize the full patch×patch attention matrix as a heatmap, showing which columns attend to which other columns.

```python
R = 4
last_attn = attn_maps[-1][0].mean(dim=0)  # [S, S] averaged over heads
patch_attn = last_attn[R:, R:]            # [128, 128]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: attention matrix
im = axes[0].imshow(patch_attn.numpy(), cmap='viridis', aspect='auto')
axes[0].set_xlabel('Key (column position)')
axes[0].set_ylabel('Query (column position)')
axes[0].set_title('Patch Self-Attention')
plt.colorbar(im, ax=axes[0])

# Right: input image with CTC predictions
axes[1].imshow(image_np, cmap='gray', aspect='auto')
axes[1].set_title(f'Input: "{ground_truth}"')

plt.tight_layout()
plt.savefig("attention_flow.png", dpi=150)
```

---

## 8. Complete Implementation Code

The following script implements all three approaches in a single file:

```python
#!/usr/bin/env python3
"""
Character-Level Attention Map Visualization for HTR-Pipeline.
Inspired by Beyond-Memorization (ICDAR 2025) attention visualization.

Produces:
  1. Per-character CTC-aligned attention heatmaps (Beyond-Memorization style)
  2. Per-register spatial attention heatmaps
  3. Full self-attention flow visualization

Usage:
    python scripts/postprocessing/character_attention_viz.py \
        --config configs/baseline_vit_rgts_v2.yaml \
        --model saved_models/experiments/run_68/model.pt \
        --image notebook/sample_images/a06-110-08.png \
        --save-dir visualizations/character_attention
"""

import argparse, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from PIL import Image
from scipy.ndimage import label, center_of_mass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models import HTRNet
from utils.preprocessing import preprocess
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--model", required=True, help="Model checkpoint .pt")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--save-dir", default="visualizations/character_attention")
    p.add_argument("--layer", type=int, default=-1, help="Layer index (-1=last)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_model(config_path, model_path, device):
    """Load HTRNet model."""
    cfg = OmegaConf.load(config_path)
    
    with open("letter2index.json") as f:
        l2i = json.load(f)
    with open("index2letter.json") as f:
        i2l = json.load(f)
    
    num_classes = len(l2i) + 1  # +1 for CTC blank
    model = HTRNet(cfg, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, cfg, l2i, i2l


def ctc_decode(logits, i2l, blank_idx=0):
    """
    CTC greedy decode with character-to-position mapping.
    
    Returns:
        decoded_text: str
        char_positions: dict mapping char_idx → list of time positions
    """
    # logits: [T, C]
    probs = logits.softmax(dim=-1)
    pred_indices = probs.argmax(dim=-1).cpu().numpy()  # [T]
    
    char_positions = {}
    decoded_chars = []
    current_char_idx = 0
    prev_idx = -1
    
    for t in range(len(pred_indices)):
        idx = pred_indices[t]
        if idx == blank_idx:
            prev_idx = idx
            continue
        if idx == prev_idx:  # collapse repeats
            # But still record this position for the current character
            if current_char_idx - 1 in char_positions:
                char_positions[current_char_idx - 1].append(t)
            prev_idx = idx
            continue
        
        # New character
        char = i2l.get(str(idx - 1), '?')  # -1 because blank is at index 0
        decoded_chars.append(char)
        char_positions[current_char_idx] = [t]
        current_char_idx += 1
        prev_idx = idx
    
    return ''.join(decoded_chars), char_positions


def find_blob_centroid(attn_map, sigma=1.0):
    """Find the centroid of the strongest attention blob (Beyond-Memorization style)."""
    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    mean_val = attn_map.mean()
    std_val = attn_map.std()
    threshold = mean_val + sigma * std_val
    
    thresholded = attn_map >= threshold
    labeled_map, n_features = label(thresholded)
    
    if n_features > 0:
        max_idx = np.argmax(attn_map[thresholded])
        max_blob_label = labeled_map[thresholded][max_idx]
        blob_mask = labeled_map == max_blob_label
        centroid = center_of_mass(blob_mask)
        rows, cols = np.where(blob_mask)
        return {
            'centroid': centroid,
            'bbox': (rows.min(), cols.min(), rows.max(), cols.max()),
            'blob_mask': blob_mask
        }
    return None


# ═════════════════════════════════════════════════════════════════════════════
# APPROACH A: CTC-Aligned Per-Character Heatmaps
# ═════════════════════════════════════════════════════════════════════════════

def visualize_ctc_character_attention(
    image_np, attn_maps, logits, i2l, num_registers,
    save_dir, layer_idx=-1, dpi=150
):
    """
    Create per-character attention heatmaps using CTC alignment.
    Similar to Beyond-Memorization's per-character attention maps.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # CTC decode with position mapping
    decoded_text, char_positions = ctc_decode(logits[:, 0, :], i2l)
    
    if not decoded_text:
        print("Warning: Empty CTC decode, skipping character attention.")
        return
    
    # Get attention from specified layer
    R = num_registers
    attn = attn_maps[layer_idx][0]  # [H, S, S] for sample 0
    num_heads = attn.shape[0]
    
    # Average over heads
    attn_avg = attn.mean(dim=0).numpy()  # [S, S]
    
    # Patch-to-patch attention (skip registers)
    patch_attn = attn_avg[R:, R:]  # [P, P]
    num_patches = patch_attn.shape[0]
    
    H_img, W_img = image_np.shape[:2]
    pixels_per_patch = W_img / num_patches
    
    print(f"Decoded: '{decoded_text}' ({len(decoded_text)} chars, {num_patches} patches)")
    
    # Create per-character heatmaps
    all_heatmaps = []
    
    for char_idx, char in enumerate(decoded_text):
        if char_idx not in char_positions:
            continue
        
        positions = char_positions[char_idx]
        
        # What do these character positions attend to?
        # Average attention FROM character positions TO all patch positions
        char_attn = patch_attn[positions, :].mean(axis=0)  # [P]
        
        # Create 2D heatmap by upsampling horizontally
        heatmap_1d = char_attn
        
        # Upsample to image width
        heatmap_upsampled = np.repeat(heatmap_1d, max(1, int(pixels_per_patch)))
        if len(heatmap_upsampled) < W_img:
            heatmap_upsampled = np.pad(heatmap_upsampled, 
                                        (0, W_img - len(heatmap_upsampled)))
        heatmap_upsampled = heatmap_upsampled[:W_img]
        
        # Tile vertically
        heatmap_2d = np.tile(heatmap_upsampled, (H_img, 1))
        
        # Normalize
        heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)
        all_heatmaps.append((char, char_idx, heatmap_2d, positions))
        
        # Individual character plot (Beyond-Memorization style)
        fig, ax = plt.subplots(figsize=(10, 2.5))
        if len(image_np.shape) == 2:
            ax.imshow(image_np, cmap='gray', aspect='auto')
        else:
            ax.imshow(image_np, aspect='auto')
        ax.imshow(heatmap_2d, cmap='inferno', alpha=0.5, aspect='auto')
        ax.set_title(f"Attention for character '{char}' (pos {positions[0]}–{positions[-1]})",
                     fontsize=12)
        ax.axis('off')
        
        filepath = os.path.join(save_dir, f"char_{char_idx:02d}_{char}.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    # Combined grid (all characters in one figure)
    if all_heatmaps:
        n_chars = len(all_heatmaps)
        fig, axes = plt.subplots(n_chars + 1, 1, figsize=(12, 2 * (n_chars + 1)))
        
        if n_chars == 0:
            return
        
        if n_chars + 1 == 1:
            axes = [axes]
        
        # First row: original image
        if len(image_np.shape) == 2:
            axes[0].imshow(image_np, cmap='gray', aspect='auto')
        else:
            axes[0].imshow(image_np, aspect='auto')
        axes[0].set_title(f"Input: '{decoded_text}'", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Character heatmaps
        for i, (char, char_idx, heatmap, positions) in enumerate(all_heatmaps):
            ax = axes[i + 1]
            if len(image_np.shape) == 2:
                ax.imshow(image_np, cmap='gray', aspect='auto')
            else:
                ax.imshow(image_np, aspect='auto')
            ax.imshow(heatmap, cmap='inferno', alpha=0.5, aspect='auto')
            ax.set_title(f"'{char}'", fontsize=11, loc='left')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "all_characters.png"), 
                    dpi=dpi, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(all_heatmaps)} character attention maps to {save_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# APPROACH B: Register Token Attention Heatmaps
# ═════════════════════════════════════════════════════════════════════════════

def visualize_register_attention(
    image_np, attn_maps, num_registers,
    save_dir, layer_idx=-1, dpi=150
):
    """
    Visualize what each register token attends to across the image.
    Analogous to Figure 9/16 in 'Vision Transformers Need Registers'.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    R = num_registers
    if R == 0:
        print("No register tokens, skipping register attention visualization.")
        return
    
    attn = attn_maps[layer_idx][0]  # [H, S, S]
    num_heads = attn.shape[0]
    H_img, W_img = image_np.shape[:2]
    num_patches = attn.shape[1] - R
    pixels_per_patch = W_img / num_patches
    
    # Head-averaged register attention
    attn_avg = attn.mean(dim=0).numpy()  # [S, S]
    
    fig_rows = R
    fig_cols = 1
    fig, axes = plt.subplots(fig_rows + 1, fig_cols, figsize=(12, 2.5 * (fig_rows + 1)))
    
    # Original image
    if len(image_np.shape) == 2:
        axes[0].imshow(image_np, cmap='gray', aspect='auto')
    else:
        axes[0].imshow(image_np, aspect='auto')
    axes[0].set_title("Input Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    for reg_idx in range(R):
        # Register reg_idx attending to all patches
        reg_attn = attn_avg[reg_idx, R:]  # [P]
        
        # Upsample
        heatmap = np.repeat(reg_attn, max(1, int(pixels_per_patch)))[:W_img]
        if len(heatmap) < W_img:
            heatmap = np.pad(heatmap, (0, W_img - len(heatmap)))
        heatmap_2d = np.tile(heatmap, (H_img, 1))
        heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)
        
        ax = axes[reg_idx + 1]
        if len(image_np.shape) == 2:
            ax.imshow(image_np, cmap='gray', aspect='auto')
        else:
            ax.imshow(image_np, aspect='auto')
        ax.imshow(heatmap_2d, cmap='inferno', alpha=0.5, aspect='auto')
        ax.set_title(f"Register {reg_idx} attention", fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "register_attention.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Per-head register attention (detailed view)
    fig, axes = plt.subplots(R, num_heads, figsize=(3 * num_heads, 2.5 * R))
    if R == 1:
        axes = axes.reshape(1, -1)
    
    for reg_idx in range(R):
        for head_idx in range(num_heads):
            reg_attn = attn[head_idx, reg_idx, R:].numpy()
            
            heatmap = np.repeat(reg_attn, max(1, int(pixels_per_patch)))[:W_img]
            if len(heatmap) < W_img:
                heatmap = np.pad(heatmap, (0, W_img - len(heatmap)))
            heatmap_2d = np.tile(heatmap, (H_img, 1))
            heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)
            
            ax = axes[reg_idx, head_idx]
            if len(image_np.shape) == 2:
                ax.imshow(image_np, cmap='gray', aspect='auto')
            else:
                ax.imshow(image_np, aspect='auto')
            ax.imshow(heatmap_2d, cmap='inferno', alpha=0.5, aspect='auto')
            if reg_idx == 0:
                ax.set_title(f"Head {head_idx}", fontsize=9)
            if head_idx == 0:
                ax.set_ylabel(f"Reg {reg_idx}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle("Register Attention per Head", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "register_attention_per_head.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved register attention maps ({R} registers × {num_heads} heads)")


# ═════════════════════════════════════════════════════════════════════════════
# APPROACH C: Self-Attention Flow Matrix
# ═════════════════════════════════════════════════════════════════════════════

def visualize_attention_flow(
    image_np, attn_maps, logits, i2l, num_registers,
    save_dir, layer_idx=-1, dpi=150
):
    """
    Visualize patch-to-patch self-attention flow alongside the image.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    R = num_registers
    decoded_text, char_positions = ctc_decode(logits[:, 0, :], i2l)
    
    attn = attn_maps[layer_idx][0].mean(dim=0).numpy()  # [S, S]
    patch_attn = attn[R:, R:]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), 
                              gridspec_kw={'width_ratios': [1, 2]})
    
    # Left: attention flow matrix
    im = axes[0].imshow(patch_attn, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Key position (column token)')
    axes[0].set_ylabel('Query position (column token)')
    axes[0].set_title('Patch Self-Attention Flow')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Mark character boundaries
    if char_positions:
        for char_idx, positions in char_positions.items():
            mid_pos = positions[len(positions)//2]
            if char_idx < len(decoded_text):
                axes[0].axhline(y=mid_pos, color='red', alpha=0.3, linewidth=0.5)
                axes[0].axvline(x=mid_pos, color='red', alpha=0.3, linewidth=0.5)
    
    # Right: input image
    if len(image_np.shape) == 2:
        axes[1].imshow(image_np, cmap='gray', aspect='auto')
    else:
        axes[1].imshow(image_np, aspect='auto')
    axes[1].set_title(f"Input: '{decoded_text}'", fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_flow.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Multi-layer view
    n_layers = len(attn_maps)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    for l_idx in range(n_layers):
        a = attn_maps[l_idx][0].mean(dim=0).numpy()
        pa = a[R:, R:]
        axes[l_idx].imshow(pa, cmap='viridis', aspect='auto')
        axes[l_idx].set_title(f"Layer {l_idx}")
        axes[l_idx].set_xticks([])
        axes[l_idx].set_yticks([])
    
    plt.suptitle(f"Self-Attention Flow Across Layers — '{decoded_text}'",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_flow_all_layers.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved attention flow visualizations ({n_layers} layers)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, cfg, l2i, i2l = load_model(args.config, args.model, device)
    
    # Get architecture info
    arch_type = getattr(cfg.arch, 'type', 'cnn_rnn')
    num_registers = getattr(cfg.arch, 'num_registers', 0)
    
    print(f"Architecture: {arch_type}, Registers: {num_registers}")
    
    # Load and preprocess image
    img_height = getattr(cfg.data, 'img_height', 128)
    img_width = getattr(cfg.data, 'img_width', 1024)
    
    image = preprocess(args.image, img_height, img_width)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Also load original image for display
    orig_img = np.array(Image.open(args.image).convert('L'))
    display_img = image[0] if len(image.shape) > 2 else image
    
    # Forward pass with attention extraction
    print("Running forward_explain()...")
    with torch.no_grad():
        logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(image_tensor)
    
    print(f"  Logits: {logits.shape}")
    print(f"  Attention maps: {len(attn_maps)} layers, shape {attn_maps[0].shape}")
    print(f"  Token norms: {token_norms.shape}")
    print(f"  Grid: {grid}")
    
    # Create output directories
    base_dir = args.save_dir
    
    # Run all three visualization approaches
    print("\n--- Approach A: CTC-Aligned Character Attention ---")
    visualize_ctc_character_attention(
        display_img, attn_maps, logits, i2l, num_registers,
        os.path.join(base_dir, "character_attention"),
        layer_idx=args.layer, dpi=args.dpi
    )
    
    print("\n--- Approach B: Register Token Attention ---")
    visualize_register_attention(
        display_img, attn_maps, num_registers,
        os.path.join(base_dir, "register_attention"),
        layer_idx=args.layer, dpi=args.dpi
    )
    
    print("\n--- Approach C: Self-Attention Flow ---")
    visualize_attention_flow(
        display_img, attn_maps, logits, i2l, num_registers,
        os.path.join(base_dir, "attention_flow"),
        layer_idx=args.layer, dpi=args.dpi
    )
    
    print(f"\nAll visualizations saved to: {base_dir}")


if __name__ == "__main__":
    main()
```

---

## 9. How to Run

### Basic usage

```bash
cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline

python scripts/postprocessing/character_attention_viz.py \
    --config configs/baseline_vit_rgts_v2.yaml \
    --model saved_models/experiments/run_68/model.pt \
    --image notebook/sample_images/a06-110-08.png \
    --save-dir visualizations/character_attention
```

### Compare multiple models (0 vs 4 vs 7 registers)

```bash
for run in run_60 run_68 run_73; do
    python scripts/postprocessing/character_attention_viz.py \
        --config configs/baseline_vit_rgts_v2.yaml \
        --model saved_models/experiments/${run}/model.pt \
        --image notebook/sample_images/a06-110-08.png \
        --save-dir visualizations/character_attention/${run}
done
```

### All layers view

```bash
# Last layer (default)
python scripts/postprocessing/character_attention_viz.py \
    --config configs/baseline_vit_rgts_v2.yaml \
    --model saved_models/experiments/run_68/model.pt \
    --image notebook/sample_images/a06-110-08.png \
    --layer -1

# First layer
python scripts/postprocessing/character_attention_viz.py \
    ... --layer 0

# Middle layer (layer 3 of 6)
python scripts/postprocessing/character_attention_viz.py \
    ... --layer 2
```

---

## 10. Expected Output and Comparison

### Output directory structure

```
visualizations/character_attention/
├── character_attention/
│   ├── all_characters.png        ← Grid: input image + all character heatmaps
│   ├── char_00_h.png             ← Individual heatmap for 'h'
│   ├── char_01_e.png             ← Individual heatmap for 'e'
│   ├── char_02_l.png             ← ...
│   └── ...
├── register_attention/
│   ├── register_attention.png     ← Head-averaged register attention
│   └── register_attention_per_head.png  ← R × H grid
└── attention_flow/
    ├── attention_flow.png         ← Attention matrix + input image
    └── attention_flow_all_layers.png  ← Layer-by-layer flow
```

### Side-by-Side Comparison

| Aspect | Beyond-Memorization Output | Our Output |
|--------|--------------------------|------------|
| **Source** | UNet cross-attention (image→char) | ViT self-attention (CTC-aligned) |
| **Per-character?** | Yes — direct from attention dim | Yes — via CTC alignment |
| **Spatial resolution** | 8×32 upsampled to 64×256 | 1×128 tiled to 128×1024 |
| **Colormap** | inferno overlay, alpha=0.5 | inferno overlay, alpha=0.5 |
| **Extra features** | Blob detection, bounding boxes | Register attention, layer flow |
| **Unique to us** | — | Register specialization visualization |

### What to look for in the output

1. **Character attention maps**: Each character should have a concentrated heatmap over its spatial region in the image. If attention is diffuse, the model may not be localizing characters well.

2. **Register attention**: Different registers should attend to different parts of the image. If all registers look identical, they haven't specialized. If one register attends to the left side and another to the right, that's the slot-attention-like behavior from the Registers paper.

3. **Attention flow**: The matrix should show near-diagonal patterns (local attention) plus some long-range connections. Register-augmented models should show cleaner patterns than models without registers.

4. **Layer progression**: Early layers should show more local/diffuse attention, later layers more focused/character-specific attention.
