#!/usr/bin/env python3
"""
Attention Map Visualization for HTR-Pipeline.
==============================================

Unified character-level attention visualization following the approach from:
  "Beyond Memorization: Training-Free Style Mixing for Variability in
   Handwritten Text Generation" (ICDAR 2025, Gurav et al.)
   https://github.com/aniketntnu/Beyond-Memorization

Their paper (Fig. 5) visualizes per-character attention maps as heatmaps
overlaid on the word image, highlighting spatial focus for each character.
They use U-Net cross-attention (image queries x character keys) for direct
character maps.  We adapt this using CTC-aligned self-attention:
  - CTC greedy decode maps characters -> encoder timestep spans
  - Self-attention rows at those timesteps -> per-character spatial attention
  - Blob thresholding (mean + sigma*std) + connected-component detection
    finds the strongest attention region per character (matching their
    find_max_activation_above_threshold / get_blob_centroids approach)

Produces:
  1. Fig. 5-style character attention grid (rows=words, cols=characters)
  2. Per-character blob-thresholded heatmaps with bounding boxes
  3. Multi-model register comparison (0 vs 4 vs 8 vs 16 registers)
  4. Register token spatial attention heatmaps
  5. Self-attention flow matrices across layers
  6. Attention quality metrics (entropy, sparsity, diagonality)

Usage:
    # Single model -- all visualizations
    python scripts/postprocessing/beyond_memorization_viz.py \\
        --config configs/baseline_vit_rgts_v2.yaml \\
        --model saved_models/experiments/run_146/model.pt \\
        --image notebook/sample_images/a01-038-12.png \\
        --save-dir outputs/attention_maps

    # Multi-model register comparison
    python scripts/postprocessing/beyond_memorization_viz.py \\
        --config configs/baseline_vit_rgts_v2.yaml \\
        --models-dir saved_models/experiments \\
        --runs run_144 run_146 run_151 run_152 \\
        --image notebook/sample_images/a01-038-12.png \\
        --save-dir outputs/attention_maps/comparison

    # Fig. 5 grid from multiple images
    python scripts/postprocessing/beyond_memorization_viz.py \\
        --config configs/baseline_vit_rgts_v2.yaml \\
        --model saved_models/experiments/run_146/model.pt \\
        --image-dir notebook/sample_images \\
        --num-samples 4 \\
        --save-dir outputs/attention_maps/fig5

Reference:
    @InProceedings{10.1007/978-3-032-04627-7_27,
      author="Gurav, Aniket and Chanda, Sukalpa and Krishnan, Narayanan C.",
      title="Beyond Memorization: Training-Free Style Mixing ...",
      booktitle="Document Analysis and Recognition -- ICDAR 2025",
      year="2026", pages="465--484"
    }
"""

import argparse, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import torch
from pathlib import Path
from scipy.ndimage import label as ndlabel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models import HTRNet
from utils.preprocessing import load_image, preprocess
from omegaconf import OmegaConf


# ===========================================================================
# MODEL & DATA LOADING
# ===========================================================================

def _load_charset(cfg):
    """Load character classes from the dataset classes.npy file."""
    data_path = getattr(getattr(cfg, 'data', None), 'path', 'data/IAM/processed_lines')
    classes_path = os.path.join(data_path, 'classes.npy')
    classes = list(np.load(classes_path, allow_pickle=True))
    i2l = {str(i): str(c) for i, c in enumerate(classes)}
    num_classes = len(classes) + 1  # +1 for CTC blank
    return i2l, num_classes


def load_model_from_run(config_path, model_path, device):
    """Load an HTRNet model from config + checkpoint."""
    cfg = OmegaConf.load(config_path)
    i2l, num_classes = _load_charset(cfg)
    model = HTRNet(cfg.arch, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    num_registers = getattr(cfg.arch, 'num_registers', 0)
    return model, cfg, i2l, num_registers


def load_and_preprocess_image(image_path, cfg):
    """Load and preprocess a single image."""
    img_height = getattr(cfg.preproc, 'image_height',
                         getattr(cfg.data, 'img_height', 128))
    img_width = getattr(cfg.preproc, 'image_width',
                        getattr(cfg.data, 'img_width', 1024))
    raw_img = load_image(image_path)
    processed = preprocess(raw_img, (img_height, img_width))
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
    return processed, tensor


# ===========================================================================
# CTC DECODE
# ===========================================================================

def ctc_greedy_decode(logits, i2l, blank_idx=0):
    """
    CTC greedy decode with character-to-position span mapping.

    Returns:
        text: decoded string
        char_spans: list of (char, start_pos, end_pos) tuples
    """
    probs = logits.softmax(dim=-1)
    pred_indices = probs.argmax(dim=-1).cpu().numpy()

    char_spans = []
    prev_idx = -1
    current_start = None
    current_char = None

    for t in range(len(pred_indices)):
        idx = pred_indices[t]
        if idx == blank_idx:
            if current_char is not None:
                char_spans.append((current_char, current_start, t - 1))
                current_char = None
            prev_idx = idx
            continue
        if idx == prev_idx:
            prev_idx = idx
            continue
        # New character -- close previous
        if current_char is not None:
            char_spans.append((current_char, current_start, t - 1))
        current_char = i2l.get(str(idx - 1), '?')
        current_start = t
        prev_idx = idx

    # Close last character
    if current_char is not None:
        char_spans.append((current_char, current_start, len(pred_indices) - 1))

    text = ''.join(c for c, _, _ in char_spans)
    return text, char_spans


# ===========================================================================
# ATTENTION EXTRACTION
# ===========================================================================

def upsample_1d_to_width(attn_1d, target_width):
    """Upsample a 1D attention vector to image width."""
    num_patches = len(attn_1d)
    ppp = target_width / num_patches
    heatmap = np.repeat(attn_1d, max(1, int(np.ceil(ppp))))
    if len(heatmap) < target_width:
        heatmap = np.pad(heatmap, (0, target_width - len(heatmap)), mode='edge')
    return heatmap[:target_width]


def extract_attention_data(model, image_tensor, i2l, num_registers):
    """Run forward_explain and extract attention + CTC decode."""
    with torch.no_grad():
        logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(image_tensor)
    text, char_spans = ctc_greedy_decode(logits[:, 0, :], i2l)
    return {
        'logits': logits,
        'reg_tokens': reg_tokens,
        'attn_maps': attn_maps,
        'token_norms': token_norms,
        'grid': grid,
        'text': text,
        'char_spans': char_spans,
        'num_registers': num_registers,
    }


def compute_char_heatmap(attn_data, char_idx, char_span, layer_idx=-1):
    """
    Compute the 1D attention vector for a single character.

    Uses self-attention: average attention FROM character CTC positions
    TO all patch positions (excluding register tokens).
    """
    R = attn_data['num_registers']
    attn = attn_data['attn_maps'][layer_idx][0]  # [H, S, S]
    attn_avg = attn.mean(dim=0).numpy()  # [S, S]
    patch_attn = attn_avg[R:, R:]  # [P, P]

    _, start, end = char_span
    positions = list(range(start, end + 1))

    num_patches = patch_attn.shape[0]
    positions = [p for p in positions if p < num_patches]
    if not positions:
        return None

    char_attn = patch_attn[positions, :].mean(axis=0)  # [P]
    return char_attn


# ===========================================================================
# BLOB DETECTION (Beyond-Memorization approach)
# ===========================================================================

def find_strongest_blob(heatmap_2d, sigma=1.0):
    """
    Beyond-Memorization blob detection: threshold at mean + sigma*std,
    find connected components, return the strongest blob.

    This follows the approach from saveAttentionMaps.py in the
    Beyond-Memorization codebase (find_max_activation_above_threshold /
    get_blob_centroids), adapted for our 1D-upsampled attention maps.

    Returns dict with 'mask', 'bbox' (row_min, col_min, row_max, col_max),
    'centroid', or None if no blob found.
    """
    normed = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)
    threshold = normed.mean() + sigma * normed.std()
    binary = normed >= threshold

    labeled, n_features = ndlabel(binary)
    if n_features == 0:
        return None

    # Find blob with highest total attention
    best_label = None
    best_sum = -1
    for lbl in range(1, n_features + 1):
        mask = labeled == lbl
        s = normed[mask].sum()
        if s > best_sum:
            best_sum = s
            best_label = lbl

    mask = labeled == best_label
    rows, cols = np.where(mask)

    return {
        'mask': mask,
        'bbox': (rows.min(), cols.min(), rows.max(), cols.max()),
        'centroid': (rows.mean(), cols.mean()),
    }


# ===========================================================================
# VIZ 1: PER-CHARACTER HEATMAPS (Beyond-Memorization style)
# ===========================================================================

def visualize_character_attention(
    image_np, attn_data, save_dir, layer_idx=-1, sigma=1.0, dpi=150
):
    """
    Per-character attention maps: inferno heatmap overlay with blob
    bounding boxes (Beyond-Memorization style).

    Produces:
      - Per-character PNGs (char_00_X.png, char_01_Y.png, ...)
      - Summary grid (summary_grid.png)
    """
    os.makedirs(save_dir, exist_ok=True)

    text = attn_data['text']
    char_spans = attn_data['char_spans']
    H_img, W_img = image_np.shape[:2]
    R = attn_data['num_registers']
    num_patches = attn_data['attn_maps'][layer_idx][0].shape[1] - R

    if not text:
        print("  Warning: Empty CTC decode, skipping.")
        return []

    print(f"  Decoded: '{text}' ({len(char_spans)} chars, {num_patches} patches)")

    char_results = []

    for ci, (char, start, end) in enumerate(char_spans):
        char_attn_1d = compute_char_heatmap(attn_data, ci, (char, start, end), layer_idx)
        if char_attn_1d is None:
            continue

        heatmap_1d = upsample_1d_to_width(char_attn_1d, W_img)
        heatmap_2d = np.tile(heatmap_1d, (H_img, 1))
        heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)

        blob = find_strongest_blob(heatmap_2d, sigma=sigma)

        char_results.append({
            'char': char, 'idx': ci, 'span': (start, end),
            'heatmap': heatmap_2d, 'attn_1d': char_attn_1d, 'blob': blob,
        })

        # Individual character plot
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.imshow(image_np, cmap='gray', aspect='auto')
        ax.imshow(heatmap_2d, cmap='inferno', alpha=0.5, aspect='auto')

        if blob is not None:
            r0, c0, r1, c1 = blob['bbox']
            rect = mpatches.FancyBboxPatch(
                (c0, r0), c1 - c0, r1 - r0,
                linewidth=2, edgecolor='lime', facecolor='none',
                boxstyle='round,pad=2'
            )
            ax.add_patch(rect)

        ax.set_title(f"'{char}' (tokens {start}-{end})", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.savefig(os.path.join(save_dir, f"char_{ci:02d}_{char}.png"),
                    dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        plt.close()

    # Summary grid (vertical: input + per-character rows)
    if char_results:
        _draw_summary_grid(image_np, text, char_results, save_dir, dpi)

    print(f"  Saved {len(char_results)} character attention maps to {save_dir}")
    return char_results


def _draw_summary_grid(image_np, text, char_results, save_dir, dpi):
    """Vertical summary: input image + one row per character with blob boxes."""
    n = len(char_results)
    fig, axes = plt.subplots(n + 1, 1, figsize=(12, 2.0 * (n + 1)))
    if n + 1 == 1:
        axes = [axes]

    axes[0].imshow(image_np, cmap='gray', aspect='auto')
    axes[0].set_title(f"Input - predicted: '{text}'", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    for i, cr in enumerate(char_results):
        ax = axes[i + 1]
        ax.imshow(image_np, cmap='gray', aspect='auto')
        ax.imshow(cr['heatmap'], cmap='inferno', alpha=0.5, aspect='auto')
        if cr['blob'] is not None:
            r0, c0, r1, c1 = cr['blob']['bbox']
            rect = mpatches.FancyBboxPatch(
                (c0, r0), c1 - c0, r1 - r0,
                linewidth=1.5, edgecolor='lime', facecolor='none',
                boxstyle='round,pad=1'
            )
            ax.add_patch(rect)
        ax.set_title(f"'{cr['char']}'", fontsize=10, loc='left', pad=2)
        ax.axis('off')

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(save_dir, "summary_grid.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()


# ===========================================================================
# VIZ 2: FIG. 5 MULTI-WORD GRID
# ===========================================================================

def visualize_fig5_grid(image_word_results, save_dir, sigma=1.0, dpi=200):
    """
    Beyond-Memorization Fig. 5 style grid:
      Rows = different words (images)
      Col 0 = word-level attention A (sum of all character attentions)
      Cols 1+ = per-character attention map Ac

    Each cell shows the word image with inferno heatmap overlay and
    blob bounding box for that character.

    Args:
        image_word_results: list of (image_np, attn_data, char_results) tuples
    """
    os.makedirs(save_dir, exist_ok=True)
    if not image_word_results:
        return

    max_chars = max(len(cr) for _, _, cr in image_word_results)
    n_words = len(image_word_results)

    # Grid: n_words rows x (max_chars + 1) cols (+1 for word-level A)
    fig, axes = plt.subplots(
        n_words, max_chars + 1,
        figsize=(2.0 * (max_chars + 1), 2.0 * n_words),
        squeeze=False
    )

    for row, (image_np, attn_data, char_results) in enumerate(image_word_results):
        text = attn_data['text']

        # Column 0: full word-level attention A
        ax = axes[row, 0]
        ax.imshow(image_np, cmap='gray', aspect='auto')
        if char_results:
            combined = np.zeros_like(char_results[0]['heatmap'])
            for cr in char_results:
                combined += cr['heatmap']
            combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
            ax.imshow(combined, cmap='inferno', alpha=0.45, aspect='auto')
        ax.set_title(f"A ('{text}')", fontsize=8, fontweight='bold')
        ax.axis('off')

        # Columns 1..max_chars: per-character attention Ac
        for col_idx in range(max_chars):
            ax = axes[row, col_idx + 1]
            if col_idx < len(char_results):
                cr = char_results[col_idx]
                ax.imshow(image_np, cmap='gray', aspect='auto')
                ax.imshow(cr['heatmap'], cmap='inferno', alpha=0.5, aspect='auto')
                if cr['blob'] is not None:
                    r0, c0, r1, c1 = cr['blob']['bbox']
                    rect = mpatches.FancyBboxPatch(
                        (c0, r0), c1 - c0, r1 - r0,
                        linewidth=1.0, edgecolor='lime', facecolor='none',
                        boxstyle='round,pad=1'
                    )
                    ax.add_patch(rect)
                ax.set_title(f"Ac: '{cr['char']}'", fontsize=7)
            else:
                ax.set_visible(False)
            ax.axis('off')

    plt.suptitle(
        "Fig. 5: Character-Level Attention Maps\n"
        "Rows = words, Col 0 = word attention A, Cols 1+ = per-character Ac",
        fontsize=11, fontweight='bold', y=1.02
    )
    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(save_dir, "fig5_character_attention_grid.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved Fig. 5 grid ({n_words} words x {max_chars} chars) to {save_dir}")


# ===========================================================================
# VIZ 3: MULTI-MODEL REGISTER COMPARISON
# ===========================================================================

def visualize_register_comparison(
    image_np, model_results, save_dir, layer_idx=-1, sigma=1.0, dpi=150
):
    """
    Side-by-side comparison of character attention across different
    register counts. Rows = characters, Cols = models (register configs).
    """
    os.makedirs(save_dir, exist_ok=True)

    n_models = len(model_results)
    if n_models == 0:
        return

    ref_text = model_results[0][2]['text']
    ref_spans = model_results[0][2]['char_spans']
    if not ref_text:
        print("  Warning: Empty decode, skipping comparison.")
        return

    H_img, W_img = image_np.shape[:2]
    n_chars = min(len(ref_spans), 15)

    fig, axes = plt.subplots(
        n_chars + 1, n_models,
        figsize=(5 * n_models, 1.8 * (n_chars + 1)),
        squeeze=False
    )

    # Header row: input image per model
    for col, (run_name, n_reg, attn_data) in enumerate(model_results):
        axes[0, col].imshow(image_np, cmap='gray', aspect='auto')
        axes[0, col].set_title(f"{run_name}\nreg={n_reg}, pred='{attn_data['text']}'",
                               fontsize=9, fontweight='bold')
        axes[0, col].axis('off')

    # Character rows
    for row_idx in range(n_chars):
        char, start, end = ref_spans[row_idx]

        for col, (run_name, n_reg, attn_data) in enumerate(model_results):
            ax = axes[row_idx + 1, col]
            ax.imshow(image_np, cmap='gray', aspect='auto')

            if row_idx < len(attn_data['char_spans']):
                m_char, m_start, m_end = attn_data['char_spans'][row_idx]
                char_attn_1d = compute_char_heatmap(
                    attn_data, row_idx, (m_char, m_start, m_end), layer_idx
                )
            else:
                char_attn_1d = None

            if char_attn_1d is not None:
                heatmap_1d = upsample_1d_to_width(char_attn_1d, W_img)
                heatmap_2d = np.tile(heatmap_1d, (H_img, 1))
                heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)
                ax.imshow(heatmap_2d, cmap='inferno', alpha=0.5, aspect='auto')
                blob = find_strongest_blob(heatmap_2d, sigma=sigma)
                if blob is not None:
                    r0, c0, r1, c1 = blob['bbox']
                    rect = mpatches.FancyBboxPatch(
                        (c0, r0), c1 - c0, r1 - r0,
                        linewidth=1.5, edgecolor='lime', facecolor='none',
                        boxstyle='round,pad=1'
                    )
                    ax.add_patch(rect)

            if col == 0:
                ax.set_ylabel(f"'{char}'", fontsize=10, rotation=0, labelpad=20)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Register Token Effect on Character Attention",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(save_dir, "register_comparison.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved register comparison ({n_models} models x {n_chars} chars)")


# ===========================================================================
# VIZ 4: REGISTER TOKEN SPATIAL ATTENTION
# ===========================================================================

def visualize_register_attention(
    image_np, attn_data, save_dir, layer_idx=-1, dpi=150
):
    """
    What does each register token attend to across the image?
    Analogous to Figure 9/16 in 'Vision Transformers Need Registers'.

    Produces:
      - register_attention.png: head-averaged register -> patch heatmaps
      - register_attention_per_head.png: per-head detail view
    """
    os.makedirs(save_dir, exist_ok=True)

    R = attn_data['num_registers']
    if R == 0:
        print("  No register tokens, skipping register attention.")
        return

    attn_maps = attn_data['attn_maps']
    attn = attn_maps[layer_idx][0]  # [H, S, S]
    num_heads = attn.shape[0]
    H_img, W_img = image_np.shape[:2]
    num_patches = attn.shape[1] - R

    attn_avg = attn.mean(dim=0).numpy()

    # Head-averaged view
    n_rows = R + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    axes[0].imshow(image_np, cmap='gray', aspect='auto')
    axes[0].set_title("Input Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    for reg_idx in range(R):
        reg_attn = attn_avg[reg_idx, R:]
        heatmap_1d = upsample_1d_to_width(reg_attn, W_img)
        heatmap_2d = np.tile(heatmap_1d, (H_img, 1))
        heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)

        ax = axes[reg_idx + 1]
        ax.imshow(image_np, cmap='gray', aspect='auto')
        ax.imshow(heatmap_2d, cmap='inferno', alpha=0.5, aspect='auto')
        ax.set_title(f"Register {reg_idx} attention", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "register_attention.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()

    # Per-head detail view
    fig, axes = plt.subplots(R, num_heads, figsize=(3 * num_heads, 2.5 * R),
                             squeeze=False)

    for reg_idx in range(R):
        for head_idx in range(num_heads):
            reg_attn = attn[head_idx, reg_idx, R:].numpy()
            heatmap_1d = upsample_1d_to_width(reg_attn, W_img)
            heatmap_2d = np.tile(heatmap_1d, (H_img, 1))
            heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)

            ax = axes[reg_idx, head_idx]
            ax.imshow(image_np, cmap='gray', aspect='auto')
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

    print(f"  Saved register attention maps ({R} registers x {num_heads} heads)")


# ===========================================================================
# VIZ 5: SELF-ATTENTION FLOW
# ===========================================================================

def visualize_attention_flow(
    image_np, attn_data, save_dir, layer_idx=-1, dpi=150
):
    """
    Patch-to-patch self-attention flow matrices across layers.

    Produces:
      - attention_flow.png: last-layer attention matrix + input image
      - attention_flow_all_layers.png: all layers side-by-side
    """
    os.makedirs(save_dir, exist_ok=True)

    R = attn_data['num_registers']
    text = attn_data['text']
    char_spans = attn_data['char_spans']
    attn_maps = attn_data['attn_maps']

    attn = attn_maps[layer_idx][0].mean(dim=0).numpy()
    patch_attn = attn[R:, R:]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                             gridspec_kw={'width_ratios': [1, 2]})

    im = axes[0].imshow(patch_attn, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Key position')
    axes[0].set_ylabel('Query position')
    axes[0].set_title('Patch Self-Attention Flow')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Mark character positions
    for ci, (char, start, end) in enumerate(char_spans):
        mid = (start + end) // 2
        axes[0].axhline(y=mid, color='red', alpha=0.3, linewidth=0.5)
        axes[0].axvline(x=mid, color='red', alpha=0.3, linewidth=0.5)

    axes[1].imshow(image_np, cmap='gray', aspect='auto')
    axes[1].set_title(f"Input: '{text}'", fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_flow.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()

    # Multi-layer view
    n_layers = len(attn_maps)
    fig, axes_l = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes_l = [axes_l]

    for l_idx in range(n_layers):
        a = attn_maps[l_idx][0].mean(dim=0).numpy()
        pa = a[R:, R:]
        axes_l[l_idx].imshow(pa, cmap='viridis', aspect='auto')
        axes_l[l_idx].set_title(f"Layer {l_idx}")
        axes_l[l_idx].set_xticks([])
        axes_l[l_idx].set_yticks([])

    plt.suptitle(f"Self-Attention Flow Across Layers - '{text}'",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_flow_all_layers.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved attention flow visualizations ({n_layers} layers)")


# ===========================================================================
# ATTENTION QUALITY METRICS
# ===========================================================================

def compute_attention_metrics(attn_data, layer_idx=-1):
    """
    Quantitative attention quality metrics.

    Returns:
      - entropy: mean attention entropy (lower = more focused)
      - sparsity: fraction of attention mass in top-10% of tokens
      - diagonality: alignment between query position and max-attended key
      - norm_ratio: max/mean token norm (high = outlier artifacts)
    """
    R = attn_data['num_registers']
    attn = attn_data['attn_maps'][layer_idx][0]  # [H, S, S]
    attn_avg = attn.mean(dim=0).numpy()  # [S, S]
    patch_attn = attn_avg[R:, R:]  # [P, P]
    P = patch_attn.shape[0]

    eps = 1e-10
    row_entropy = -np.sum(patch_attn * np.log(patch_attn + eps), axis=1)
    entropy = row_entropy.mean()

    top_k = max(1, P // 10)
    sorted_attn = np.sort(patch_attn, axis=1)[:, ::-1]
    sparsity = sorted_attn[:, :top_k].sum(axis=1).mean()

    argmaxes = patch_attn.argmax(axis=1)
    diag_offsets = np.abs(argmaxes - np.arange(P))
    diagonality = 1.0 - diag_offsets.mean() / P

    norms = attn_data['token_norms'][0].numpy()
    patch_norms = norms[R:]
    norm_ratio = patch_norms.max() / (patch_norms.mean() + 1e-8)

    return {
        'entropy': float(entropy),
        'sparsity': float(sparsity),
        'diagonality': float(diagonality),
        'norm_ratio': float(norm_ratio),
    }


def write_metrics_table(all_metrics, save_dir):
    """Write metrics comparison table as CSV + bar chart."""
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "attention_metrics.csv")
    with open(csv_path, 'w') as f:
        f.write("run,registers,entropy,sparsity,diagonality,norm_ratio\n")
        for run_name, n_reg, metrics in all_metrics:
            f.write(f"{run_name},{n_reg},{metrics['entropy']:.4f},"
                    f"{metrics['sparsity']:.4f},{metrics['diagonality']:.4f},"
                    f"{metrics['norm_ratio']:.4f}\n")

    print(f"\n{'Run':<10} {'Reg':>4} {'Entropy':>9} {'Sparsity':>9} "
          f"{'Diagonal':>9} {'NormRatio':>10}")
    print("-" * 60)
    for run_name, n_reg, metrics in all_metrics:
        print(f"{run_name:<10} {n_reg:>4} {metrics['entropy']:>9.4f} "
              f"{metrics['sparsity']:>9.4f} {metrics['diagonality']:>9.4f} "
              f"{metrics['norm_ratio']:>10.4f}")

    print(f"\n  Metrics saved to {csv_path}")

    if len(all_metrics) > 1:
        _plot_metrics_bars(all_metrics, save_dir)


def _plot_metrics_bars(all_metrics, save_dir):
    """Bar chart comparing metrics across models."""
    names = [f"reg={n}" for _, n, _ in all_metrics]
    metrics_keys = ['entropy', 'sparsity', 'diagonality', 'norm_ratio']
    titles = ['Attention Entropy (lower=better)', 'Sparsity top-10% (higher=better)',
              'Diagonality (higher=better)', 'Norm Ratio (lower=better)']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(names))

    for ax, key, title in zip(axes, metrics_keys, titles):
        values = [m[key] for _, _, m in all_metrics]
        bars = ax.bar(x, values, color='steelblue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        if 'lower' in title:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_color('forestgreen')

    plt.suptitle("Attention Quality Metrics Across Register Counts",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_metrics_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


# ===========================================================================
# CLI & MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Beyond-Memorization attention map visualization for HTR")
    p.add_argument("--config", required=True, help="YAML config file")

    # Single model mode
    p.add_argument("--model", help="Single model checkpoint .pt")

    # Multi-model comparison mode
    p.add_argument("--models-dir", default="saved_models/experiments",
                   help="Base directory containing run_* folders")
    p.add_argument("--runs", nargs='+',
                   help="Run names for comparison (e.g. run_76 run_84 run_63)")

    # Image input
    p.add_argument("--image", help="Single image path")
    p.add_argument("--image-dir", help="Directory of images (batch/Fig.5 mode)")
    p.add_argument("--num-samples", type=int, default=5,
                   help="Number of samples for batch/Fig.5 mode")

    # Options
    p.add_argument("--save-dir", default="outputs/attention_maps")
    p.add_argument("--layer", type=int, default=-1, help="Layer index (-1=last)")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Blob threshold: mean + sigma*std")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--skip-register-attn", action='store_true',
                   help="Skip register token attention visualization")
    p.add_argument("--skip-flow", action='store_true',
                   help="Skip self-attention flow visualization")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = OmegaConf.load(args.config)

    multi_mode = args.runs is not None and len(args.runs) > 0

    # Collect images
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        all_imgs = sorted([
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        image_paths = all_imgs[:args.num_samples]
    else:
        print("Error: provide --image or --image-dir")
        sys.exit(1)

    # Load models
    if multi_mode:
        models_info = []
        for run_name in args.runs:
            model_path = os.path.join(args.models_dir, run_name, "model.pt")
            if not os.path.exists(model_path):
                print(f"  Warning: {model_path} not found, skipping")
                continue
            run_config = os.path.join(args.models_dir, run_name, "config.json")
            with open(run_config) as f:
                run_cfg = json.load(f)
            n_reg = run_cfg.get("arch", {}).get("num_registers", 0)

            cfg_copy = OmegaConf.load(args.config)
            cfg_copy.arch.num_registers = n_reg

            i2l, num_classes = _load_charset(cfg_copy)
            model = HTRNet(cfg_copy.arch, num_classes).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            models_info.append((run_name, n_reg, model, i2l))
            print(f"  Loaded {run_name} (registers={n_reg})")
    else:
        model, cfg, i2l, num_registers = load_model_from_run(
            args.config, args.model, device)
        models_info = [("model", num_registers, model, i2l)]
        print(f"  Loaded model (registers={num_registers})")

    # Collect results for Fig. 5 grid (batch mode)
    fig5_data = []

    # Process each image
    for img_path in image_paths:
        img_name = Path(img_path).stem
        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print(f"{'='*60}")

        image_np, image_tensor = load_and_preprocess_image(img_path, cfg)
        image_tensor = image_tensor.to(device)

        model_results = []
        all_metrics = []

        for run_name, n_reg, model, i2l in models_info:
            attn_data = extract_attention_data(model, image_tensor, i2l, n_reg)
            model_results.append((run_name, n_reg, attn_data))
            metrics = compute_attention_metrics(attn_data, layer_idx=args.layer)
            all_metrics.append((run_name, n_reg, metrics))

        if not multi_mode:
            run_name, n_reg, attn_data = model_results[0]
            img_save_dir = os.path.join(args.save_dir, img_name)

            # 1. Per-character attention with blob detection
            print("\n--- Character Attention (Beyond-Memorization style) ---")
            char_results = visualize_character_attention(
                image_np, attn_data, os.path.join(img_save_dir, "character_attention"),
                layer_idx=args.layer, sigma=args.sigma, dpi=args.dpi
            )

            # Collect for Fig. 5 grid
            fig5_data.append((image_np, attn_data, char_results))

            # 2. Register token attention
            if not args.skip_register_attn:
                print("\n--- Register Token Attention ---")
                visualize_register_attention(
                    image_np, attn_data,
                    os.path.join(img_save_dir, "register_attention"),
                    layer_idx=args.layer, dpi=args.dpi
                )

            # 3. Self-attention flow
            if not args.skip_flow:
                print("\n--- Self-Attention Flow ---")
                visualize_attention_flow(
                    image_np, attn_data,
                    os.path.join(img_save_dir, "attention_flow"),
                    layer_idx=args.layer, dpi=args.dpi
                )

        else:
            img_save_dir = os.path.join(args.save_dir, img_name)

            # Multi-model register comparison
            print("\n--- Register Comparison ---")
            visualize_register_comparison(
                image_np, model_results, img_save_dir,
                layer_idx=args.layer, sigma=args.sigma, dpi=args.dpi
            )

            # Individual model grids
            for run_name, n_reg, attn_data in model_results:
                individual_dir = os.path.join(img_save_dir, f"{run_name}_reg{n_reg}")
                visualize_character_attention(
                    image_np, attn_data, individual_dir,
                    layer_idx=args.layer, sigma=args.sigma, dpi=args.dpi
                )

        # Metrics
        write_metrics_table(all_metrics, os.path.join(args.save_dir, img_name))

    # Fig. 5 multi-word grid (single-model batch mode with multiple images)
    if not multi_mode and len(fig5_data) > 1:
        print(f"\n{'='*60}")
        print("Generating Fig. 5 multi-word grid")
        print(f"{'='*60}")
        visualize_fig5_grid(fig5_data, args.save_dir, sigma=args.sigma, dpi=args.dpi)

    print(f"\nAll outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
