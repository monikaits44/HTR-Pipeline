#!/usr/bin/env python3
"""
HTR Attention Visualization Script
===================================
Generates 4 publication-quality figures for the HTR register-augmented ViT project.

Figures:
  1. Register Impact Grid     — CLS-to-patch attention + token norms across register counts
                                 (Darcet et al., 2024, Fig. 1 / Fig. 19)
  2. Character Attention Grid  — Per-character spatial attention maps
                                 ("Beyond Memorization" Fig. 5 / VLAC Fig. 4)
  3. Token Norm Artifact Map   — Spatial distribution of high-norm outlier tokens
  4. Layer-wise Attention Flow — Attention evolution through transformer layers

Usage:
    python scripts/attention_visualization.py                          # CPU, sample images
    python scripts/attention_visualization.py --device cuda:0          # GPU
    python scripts/attention_visualization.py --runs run_55 run_58     # specific runs
    python scripts/attention_visualization.py --images path1 path2     # specific images
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import HTRNet
from utils.preprocessing import load_image, preprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = ROOT / "saved_models" / "experiments"
SAMPLE_DIR = ROOT / "notebook" / "sample_images"
OUTPUT_DIR = ROOT / "output" / "attention_visualizations"
CLASSES_PATH = EXPERIMENTS_DIR / "classes.npy"

IMAGE_H, IMAGE_W = 128, 1024

# Default runs: 0-reg, 4-reg, 8-reg, 16-reg (best val CER from sweep)
DEFAULT_RUNS = {
    "run_55": 0,   # 0 registers  — baseline
    "run_58": 4,   # 4 registers
    "run_53": 8,   # 8 registers
    "run_54": 16,  # 16 registers
}

# Default sample images (short + long words for variety)
DEFAULT_IMAGES = [
    "a01-038-12.png",   # "talks."
    "a06-095-10.png",   # "he said."
    "c04-110-01.png",   # long sentence
]

# Matplotlib styling for publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ============================================================================
# Model Loading
# ============================================================================

def load_charset():
    """Load character set from saved experiments."""
    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    return list(classes)


def load_model(run_id: str, device: str = "cpu"):
    """Load a trained HTRNet model from an experiment directory."""
    run_dir = EXPERIMENTS_DIR / run_id
    config_path = run_dir / "config.json"
    model_path = run_dir / "model.pt"

    with open(config_path) as f:
        cfg = json.load(f)

    arch_cfg = cfg["arch"]
    arch_ns = SimpleNamespace(**arch_cfg)

    charset = load_charset()
    nclasses = len(charset) + 1  # +1 for CTC blank

    model = HTRNet(arch_ns, nclasses)
    state = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    num_regs = arch_cfg.get("num_registers", 0)
    return model, charset, num_regs


def prepare_image(image_path: str, device: str = "cpu"):
    """Load and preprocess a single image to model input tensor.
    Returns:
        tensor: [1, 1, H, W] model input
        img_proc: preprocessed numpy image (128×1024)
        text_bbox: (x_start, x_end) pixel range of actual text in preprocessed image
        orig_shape: (h, w) of the original image before preprocessing
    """
    img_np = load_image(str(image_path))
    orig_shape = img_np.shape  # (h, w)
    img_proc = preprocess(img_np, (IMAGE_H, IMAGE_W))
    tensor = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).float().to(device)

    # Detect actual text region (non-padding) in the preprocessed image
    # Padding uses median value → low column variance
    col_var = np.var(img_proc, axis=0)
    threshold = np.median(col_var) * 0.1 + 1e-6
    active_cols = np.where(col_var > threshold)[0]
    if len(active_cols) > 0:
        x_start = max(0, active_cols[0] - 4)    # small margin
        x_end = min(IMAGE_W, active_cols[-1] + 4)
    else:
        x_start, x_end = 0, IMAGE_W

    return tensor, img_proc, (x_start, x_end), orig_shape


# ============================================================================
# CTC Decoding
# ============================================================================

def ctc_greedy_decode(logits, charset):
    """
    Greedy CTC decode.
    logits: [T, B, C]  (T=time, B=batch, C=classes)
    CTC blank is at index 0 (PyTorch convention).
    Character at model index i corresponds to charset[i-1].
    Returns: list of (decoded_string, peak_columns) per batch element.
    """
    probs = torch.softmax(logits, dim=-1)
    B = probs.shape[1]
    blank = 0  # CTC blank is at index 0

    results = []
    for b in range(B):
        seq = probs[:, b, :].argmax(dim=-1).cpu().numpy()  # [T]
        chars = []
        cols = []
        prev = -1
        for t, idx in enumerate(seq):
            if idx != blank and idx != prev:
                char_idx = idx - 1  # offset: model index → charset index
                if 0 <= char_idx < len(charset):
                    chars.append(charset[char_idx])
                    cols.append(t)
            prev = idx
        results.append(("".join(chars), cols))
    return results


# ============================================================================
# Attention Extraction Helpers
# ============================================================================

def run_explain(model, img_tensor):
    """Run forward_explain and return structured results."""
    with torch.no_grad():
        logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(img_tensor)
    return {
        "logits": logits,          # [T, B, C]
        "reg_tokens": reg_tokens,  # [B, R, D] or None
        "attn_maps": attn_maps,    # list of [B, H, S, S] per layer
        "token_norms": token_norms,  # [B, S]
        "grid_size": grid,         # (Hp, Wp)
    }


def get_patch_attention(attn_map, num_registers, grid_size):
    """
    Extract patch-to-patch attention averaged over heads.
    attn_map: [B, H, S, S]  — single layer
    Returns: [B, Wp] — 1D attention profile (CNN stem: Hp=1)
    """
    B, H, S, _ = attn_map.shape
    Hp, Wp = grid_size
    R = num_registers

    # Average over heads → [B, S, S]
    avg = attn_map.mean(dim=1)

    # Patch tokens start at index R
    # Aggregate: mean attention FROM all patch tokens TO each patch token
    patch_attn = avg[:, R:, R:]  # [B, Np, Np]
    col_attn = patch_attn.mean(dim=1)  # [B, Np] — how much each col is attended to

    return col_attn.numpy()


def get_char_attention_maps(attn_maps, num_registers, peak_cols, grid_size,
                            layer_idx=-1, gamma=0.4):
    """
    Extract per-character attention maps following "Beyond Memorization" Fig. 5.

    For each decoded character c at CTC timestep t_c, the attention row
    A[R+t_c, R:R+Wp] reveals which spatial patch tokens timestep t_c
    attended to — the self-attention analogue of the paper's cross-attention
    row A_c.

    Applies percentile normalization + adaptive gamma correction per the
    paper_fig5_ctc documentation for clean, high-contrast heatmaps.

    Xmax_c is computed as the WEIGHTED CENTROID (not argmax) following
    the paper's definition: X^(c)_max = Σ_i (i * A_c[i]) / Σ_i A_c[i].

    Returns:
        char_maps: list of 1D attention arrays [Wp], one per character
        xmax_positions: list of Xmax_c centroid positions (patch-space)
    """
    attn = attn_maps[layer_idx]  # [B, H, S, S]
    avg = attn.mean(dim=1)       # [B, S, S] — head-averaged
    R = num_registers
    Hp, Wp = grid_size

    char_maps = []
    xmax_positions = []
    for c in peak_cols:
        token_idx = R + c  # offset by register tokens
        if token_idx < avg.shape[1]:
            # Attention FROM timestep t_c TO all patch tokens
            row = avg[0, token_idx, R:R + Wp].numpy()  # [Wp]

            # Percentile normalization (robust to outliers)
            p1, p99 = np.percentile(row, 1), np.percentile(row, 99)
            if p99 > p1:
                row = np.clip((row - p1) / (p99 - p1), 0, 1)
            else:
                row = (row - row.min()) / (row.max() - row.min() + 1e-8)

            # Adaptive gamma correction (< 1 brightens dim regions)
            # Adjust based on signal concentration: if attention is very
            # peaked, use stronger gamma to reveal context
            peak_ratio = row.max() / (row.mean() + 1e-8)
            gamma_eff = gamma * min(peak_ratio / 5.0, 2.0)
            gamma_eff = max(0.2, min(gamma_eff, 1.0))
            row = np.power(row, gamma_eff)

            # Re-normalize to [0, 1]
            row = (row - row.min()) / (row.max() - row.min() + 1e-8)

            char_maps.append(row)

            # Xmax_c: weighted centroid (paper definition)
            positions = np.arange(Wp, dtype=np.float64)
            weight_sum = row.sum()
            if weight_sum > 0:
                xmax = float(np.sum(positions * row) / weight_sum)
            else:
                xmax = float(np.argmax(row))
            xmax_positions.append(xmax)
    return char_maps, xmax_positions


def attn_to_image_crop(attn_1d, Wp, text_bbox, smooth_sigma=2):
    """
    Convert 1D patch-level attention to image-space, cropped to text region.
    Applies Gaussian smoothing as post-processing (per paper recommendation).

    Args:
        attn_1d: [Wp] normalized attention array
        Wp: number of patch columns
        text_bbox: (x_start, x_end) pixel range of text in the 1024-wide image
        smooth_sigma: Gaussian sigma for smoothing
    Returns:
        attn_cropped: 2D array [IMAGE_H, crop_w] attention overlay
        crop_w: width of the cropped region
    """
    x_start, x_end = text_bbox
    crop_w = x_end - x_start

    # Interpolate 1D attention to full image width, then crop
    attn_full = np.interp(
        np.linspace(0, 1, IMAGE_W), np.linspace(0, 1, Wp), attn_1d
    )
    attn_crop = attn_full[x_start:x_end]

    # Gaussian smoothing for cleaner localization
    if smooth_sigma > 0 and len(attn_crop) > 1:
        from scipy.ndimage import gaussian_filter1d
        attn_crop = gaussian_filter1d(attn_crop, sigma=smooth_sigma)
        attn_crop = (attn_crop - attn_crop.min()) / (attn_crop.max() - attn_crop.min() + 1e-8)

    # Tile to 2D
    attn_2d = np.tile(attn_crop, (IMAGE_H, 1))
    return attn_2d, crop_w


# ============================================================================
# Figure 1: Register Impact Grid
# ============================================================================

def plot_register_impact_grid(
    image_paths, models_info, charset, device, output_path
):
    """
    Fig 1: Register Impact Grid.
    Rows = sample images, Columns = register counts (0, 4, 8, 16).
    Each cell shows the global attention heatmap overlaid on the image,
    **cropped to the actual text region** to handle variable-length inputs.

    Inspired by Darcet et al. (2024), Fig. 1 / Fig. 19.
    """
    n_images = len(image_paths)
    n_models = len(models_info)

    fig, axes = plt.subplots(
        n_images, n_models + 1,
        figsize=(3.0 * (n_models + 1), 2.2 * n_images),
        gridspec_kw={"wspace": 0.08, "hspace": 0.25},
    )
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for row, img_path in enumerate(image_paths):
        img_tensor, img_np, text_bbox, orig_shape = prepare_image(img_path, device)
        x_start, x_end = text_bbox
        img_crop = img_np[:, x_start:x_end]

        # Column 0: original image (cropped to text)
        axes[row, 0].imshow(img_crop, cmap="gray", aspect="auto")
        axes[row, 0].set_ylabel(Path(img_path).stem, fontsize=8, rotation=0,
                                labelpad=60, va="center")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("Input Image", fontweight="bold")

        for col, (run_id, model, n_regs) in enumerate(models_info, start=1):
            result = run_explain(model, img_tensor)
            attn_maps = result["attn_maps"]
            grid_size = result["grid_size"]
            Hp, Wp = grid_size
            R = n_regs

            # Get last-layer patch attention profile
            last_attn = attn_maps[-1]  # [B, H, S, S]
            avg_attn = last_attn.mean(dim=1)  # [B, S, S]
            patch_attn = avg_attn[0, R:, R:].mean(dim=0).numpy()  # [Wp]
            patch_attn = (patch_attn - patch_attn.min()) / (patch_attn.max() - patch_attn.min() + 1e-8)

            # Crop attention to text region
            attn_2d, _ = attn_to_image_crop(patch_attn, Wp, text_bbox, smooth_sigma=3)

            # Decode prediction
            decoded, _ = ctc_greedy_decode(result["logits"], charset)[0]

            # Show cropped image + heatmap overlay
            axes[row, col].imshow(img_crop, cmap="gray", aspect="auto", alpha=0.6)
            im = axes[row, col].imshow(
                attn_2d, cmap="jet", aspect="auto", alpha=0.5,
                vmin=0, vmax=1
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_xlabel(f'"{decoded}"', fontsize=7, style="italic")

            if row == 0:
                axes[row, col].set_title(
                    f"{n_regs} Registers", fontweight="bold"
                )

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Attention Intensity")

    fig.suptitle(
        "Fig. 1: Register Impact on Attention Maps\n"
        "(Darcet et al., 2024 — Vision Transformers Need Registers)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# Figure 2: Character-Level Attention Grid
# ============================================================================

def plot_char_attention_grid(
    image_paths, model, charset, num_registers, device, output_path,
    max_chars=10, run_label=""
):
    """
    Fig 2: Character-Level Attention Grid.
    Rows = words, Columns = characters.
    Each cell overlays the character's spatial attention on the handwriting image,
    **cropped to the text region** with Xmax_c peak markers.

    Following "Beyond Memorization" Fig. 5:
    - Each row = a word/line (complete attention map A)
    - Each column = character-specific attention Ac
    - Attention maps highlight spatial focus per character
    - Padding columns shown when word < max_chars (uniform shape)
    - Xmax_c (peak horizontal coordinate) marked with vertical line
    """
    n_images = len(image_paths)

    # First pass: decode all to find max chars
    all_data = []
    for img_path in image_paths:
        img_tensor, img_np, text_bbox, orig_shape = prepare_image(img_path, device)
        result = run_explain(model, img_tensor)
        decoded, peak_cols = ctc_greedy_decode(result["logits"], charset)[0]

        char_maps, xmax_positions = get_char_attention_maps(
            result["attn_maps"], num_registers, peak_cols,
            result["grid_size"], layer_idx=-1
        )
        n_chars_show = min(len(decoded), max_chars, len(char_maps))
        all_data.append((
            img_np, decoded, peak_cols, char_maps, xmax_positions,
            result["grid_size"], n_chars_show, text_bbox
        ))

    actual_max = max(d[6] for d in all_data) if all_data else max_chars
    n_cols = actual_max + 1  # +1 for original image

    fig, axes = plt.subplots(
        n_images, n_cols,
        figsize=(1.8 * n_cols, 2.0 * n_images),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for row, (img_np, decoded, peak_cols, char_maps, xmax_pos,
              grid_size, n_show, text_bbox) in enumerate(all_data):
        Hp, Wp = grid_size
        x_start, x_end = text_bbox
        img_crop = img_np[:, x_start:x_end]
        crop_w = x_end - x_start

        # Column 0: original image with full decoded text
        axes[row, 0].imshow(img_crop, cmap="gray", aspect="auto")
        axes[row, 0].set_title("A (full)", fontsize=8, fontweight="bold")
        axes[row, 0].set_xlabel(f'"{decoded}"', fontsize=7, style="italic")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for c_idx in range(n_show):
            ax = axes[row, c_idx + 1]
            char = decoded[c_idx] if c_idx < len(decoded) else ""
            attn_1d = char_maps[c_idx] if c_idx < len(char_maps) else np.zeros(Wp)

            # Crop attention to text region with smoothing
            attn_2d, _ = attn_to_image_crop(attn_1d, Wp, text_bbox, smooth_sigma=2)

            ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.25)
            ax.imshow(attn_2d, cmap="inferno", aspect="auto", alpha=0.75, vmin=0, vmax=1)

            # Mark Xmax_c: vertical dashed line at attention centroid
            if c_idx < len(xmax_pos):
                # Convert patch-space Xmax_c (float centroid) to cropped image px
                xmax_px = xmax_pos[c_idx] / Wp * IMAGE_W - x_start
                xmax_px = np.clip(xmax_px, 0, crop_w - 1)
                ax.axvline(x=xmax_px, color="white", linewidth=1.2,
                           linestyle="--", alpha=0.85)

            # Character label
            display_char = char if char != " " else "SPC"
            ax.set_title(f"$A_{{c}}$='{display_char}'", fontsize=8, fontweight="bold",
                         color="darkblue")
            ax.set_xticks([])
            ax.set_yticks([])

        # Padded columns (uniform shape, per paper)
        for c_idx in range(n_show, actual_max):
            ax = axes[row, c_idx + 1]
            ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)
            ax.set_title("(pad)", fontsize=7, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Fig. 2: Character-Level Attention Maps  ({run_label})\n"
        "Rows = word attention A, Columns = character attention $A_c$  |  "
        "Cyan line = $X^{{max}}_c$ peak position",
        fontsize=11, fontweight="bold", y=1.03,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# Figure 3: Token Norm Artifact Map
# ============================================================================

def plot_token_norm_comparison(
    image_paths, models_info, charset, device, output_path
):
    """
    Fig 3: Token Norm Artifact Map.
    Shows spatial distribution of token L2 norms for different register counts,
    **cropped to the actual text region**.
    """
    n_images = len(image_paths)
    n_models = len(models_info)

    fig, axes = plt.subplots(
        n_images, n_models + 1,
        figsize=(3.0 * (n_models + 1), 2.2 * n_images),
        gridspec_kw={"wspace": 0.08, "hspace": 0.35},
    )
    if n_images == 1:
        axes = axes[np.newaxis, :]

    global_vmin, global_vmax = float("inf"), float("-inf")

    all_norm_data = []
    for row, img_path in enumerate(image_paths):
        img_tensor, img_np, text_bbox, _ = prepare_image(img_path, device)
        x_start, x_end = text_bbox
        img_crop = img_np[:, x_start:x_end]
        row_data = []

        axes[row, 0].imshow(img_crop, cmap="gray", aspect="auto")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("Input Image", fontweight="bold")

        for col, (run_id, model, n_regs) in enumerate(models_info, start=1):
            result = run_explain(model, img_tensor)
            norms = result["token_norms"][0].numpy()  # [S]
            Hp, Wp = result["grid_size"]
            R = n_regs

            patch_norms = norms[R:R + Wp]
            global_vmin = min(global_vmin, patch_norms.min())
            global_vmax = max(global_vmax, patch_norms.max())
            row_data.append((patch_norms, Wp, n_regs, col))

        all_norm_data.append((img_crop, row_data, text_bbox))

    for row, (img_crop, row_data, text_bbox) in enumerate(all_norm_data):
        for patch_norms, Wp, n_regs, col in row_data:
            # Interpolate to full width, then crop to text region
            norms_full = np.interp(
                np.linspace(0, 1, IMAGE_W), np.linspace(0, 1, len(patch_norms)), patch_norms
            )
            x_start, x_end = text_bbox
            norms_crop = norms_full[x_start:x_end]
            norm_2d = np.tile(norms_crop, (IMAGE_H, 1))

            axes[row, col].imshow(img_crop, cmap="gray", aspect="auto", alpha=0.4)
            im = axes[row, col].imshow(
                norm_2d, cmap="inferno", aspect="auto", alpha=0.7,
                vmin=global_vmin, vmax=global_vmax,
            )

            threshold = np.percentile(patch_norms, 95)
            outlier_count = (patch_norms > threshold).sum()
            norm_std = patch_norms.std()

            axes[row, col].set_xlabel(
                f"\u03c3={norm_std:.1f}  outliers={outlier_count}", fontsize=7
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            if row == 0:
                axes[row, col].set_title(f"{n_regs} Registers", fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Token L2 Norm")

    fig.suptitle(
        "Fig. 3: Token Norm Distribution \u2014 Artifact Reduction with Registers\n"
        "(Darcet et al., 2024 \u2014 High-norm outliers absorbed by register tokens)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# Figure 4: Layer-wise Attention Evolution
# ============================================================================

def plot_layerwise_attention(
    image_path, model, charset, num_registers, device, output_path,
    run_label=""
):
    """
    Fig 4: Layer-wise Attention Flow.
    Shows how attention patterns evolve through transformer layers (1 → 6),
    **cropped to the actual text region**.
    """
    img_tensor, img_np, text_bbox, _ = prepare_image(image_path, device)
    x_start, x_end = text_bbox
    img_crop = img_np[:, x_start:x_end]

    result = run_explain(model, img_tensor)
    attn_maps = result["attn_maps"]
    grid_size = result["grid_size"]
    Hp, Wp = grid_size
    R = num_registers

    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[1]
    heads_to_show = min(n_heads, 4)

    fig, axes = plt.subplots(
        n_layers, heads_to_show + 1,
        figsize=(3.2 * (heads_to_show + 1), 1.8 * n_layers),
        gridspec_kw={"wspace": 0.05, "hspace": 0.25},
    )

    for layer in range(n_layers):
        attn = attn_maps[layer]  # [B, H, S, S]

        # Column 0: head-averaged attention
        avg = attn[0].mean(dim=0)  # [S, S]
        patch_avg = avg[R:, R:].mean(dim=0).numpy()  # [Wp]
        patch_avg = (patch_avg - patch_avg.min()) / (patch_avg.max() - patch_avg.min() + 1e-8)

        attn_2d, _ = attn_to_image_crop(patch_avg, Wp, text_bbox, smooth_sigma=2)

        axes[layer, 0].imshow(img_crop, cmap="gray", aspect="auto", alpha=0.5)
        axes[layer, 0].imshow(attn_2d, cmap="viridis", aspect="auto", alpha=0.6, vmin=0, vmax=1)
        axes[layer, 0].set_ylabel(f"Layer {layer + 1}", fontsize=9, fontweight="bold")
        axes[layer, 0].set_xticks([])
        axes[layer, 0].set_yticks([])
        if layer == 0:
            axes[layer, 0].set_title("Avg (all heads)", fontsize=9, fontweight="bold")

        for h in range(heads_to_show):
            head_attn = attn[0, h]  # [S, S]
            patch_head = head_attn[R:, R:].mean(dim=0).numpy()  # [Wp]
            patch_head = (patch_head - patch_head.min()) / (patch_head.max() - patch_head.min() + 1e-8)

            head_2d, _ = attn_to_image_crop(patch_head, Wp, text_bbox, smooth_sigma=2)

            axes[layer, h + 1].imshow(img_crop, cmap="gray", aspect="auto", alpha=0.5)
            axes[layer, h + 1].imshow(head_2d, cmap="viridis", aspect="auto", alpha=0.6, vmin=0, vmax=1)
            axes[layer, h + 1].set_xticks([])
            axes[layer, h + 1].set_yticks([])
            if layer == 0:
                axes[layer, h + 1].set_title(f"Head {h + 1}", fontsize=9, fontweight="bold")

    decoded, _ = ctc_greedy_decode(result["logits"], charset)[0]

    fig.suptitle(
        f'Fig. 4: Layer-wise Attention Evolution  ({run_label})\n'
        f'Prediction: "{decoded}"   |   {n_layers} layers \u00d7 {n_heads} heads',
        fontsize=12, fontweight="bold", y=1.02,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# Figure 5: GradCAM Visualization
# ============================================================================

def compute_gradcam(model, img_tensor, charset, gamma=0.4):
    """
    Per-character Grad-CAM following the paper's CE-style approach
    (paper_fig5_ce documentation).

    For each predicted character c at CTC timestep t_c with class id k_c:
      1. Forward backbone → activations A [T, 1, D] (patch-token outputs)
      2. Forward CTC head → logits [T, 1, C]
      3. Backpropagate from logits[t_c, 0, k_c] (single character logit)
      4. importance_c[t] = ReLU(Σ_d ∂logit(c)/∂A_{t,d} · A_{t,d})  — per-patch
      5. Normalize + percentile clip + gamma correction

    This hooks into the TRANSFORMER OUTPUT (not CNN stem), computing true
    per-character gradient-based attribution as described in the documentation.

    Args:
        model: HTRNet model (must be vit_rgts arch)
        img_tensor: [1, 1, H, W]
        charset: character list
        gamma: power-law contrast for gamma correction (< 1 brightens dim)

    Returns:
        per_char_cams: dict {f"{char}_{idx}": 1D array [Wp]} per character
        decoded: decoded string
        Wp: number of patch columns
    """
    backbone = model.backbone

    # We need to capture the patch-token activations AFTER the transformer
    # but BEFORE the CTC head. We do this by splitting the forward pass:
    #   1. Run backbone to get seq_tokens with gradients
    #   2. Run CTC head to get logits
    #   3. Backpropagate per-character

    model.eval()
    img_input = img_tensor.detach()

    # --- Forward through backbone (need grad on the activations) ---
    B, C, H, W = img_input.shape

    if backbone.use_cnn_stem:
        x = backbone.cnn_stem(img_input)
        x = backbone.stem_pool(x)
        B, D, Hp, Wp = x.shape
        patch_tokens = x.squeeze(2).transpose(1, 2)  # [B, Wp, D]
    else:
        x = backbone.patch_embed(img_input)
        B, D, Hp, Wp = x.shape
        patch_tokens = x.flatten(2).transpose(1, 2)

    reg_tokens = backbone.register_tokens.expand(B, -1, -1)
    tokens = torch.cat([reg_tokens, patch_tokens], dim=1)
    S = tokens.size(1)
    pos = backbone.pos_embed[:, :S, :]
    tokens = tokens + pos
    tokens = backbone.emb_dropout(tokens)

    # Run transformer encoder (standard forward — supports grad)
    encoded = backbone.encoder(tokens)  # [B, S, D]

    R = backbone.num_registers
    patch_out = encoded[:, R:, :]  # [B, Wp, D]

    # Retain grad on patch activations — this is what we differentiate through
    patch_out.retain_grad()

    # Forward CTC head: expects [B, D, 1, T] format
    seq_4d = patch_out.transpose(1, 2).unsqueeze(2)  # [B, D, 1, Wp]
    logits = model.top(seq_4d)  # [T, B, C]

    # CTC greedy decode
    decoded, peak_cols = ctc_greedy_decode(logits, charset)[0]

    # Identify each character's timestep and class index
    probs = logits[:, 0, :]  # [T, C]
    pred_seq = probs.argmax(dim=-1)  # [T]

    # Build per-character targets: (timestep, class_id) for each decoded char
    char_targets = []
    prev = -1
    for t in range(logits.shape[0]):
        idx = pred_seq[t].item()
        if idx != 0 and idx != prev:
            char_targets.append((t, idx))
        prev = idx

    per_char_cams = {}

    for i, (t_c, k_c) in enumerate(char_targets):
        if i >= len(decoded):
            break
        char = decoded[i]
        label = char if char != " " else "SPC"

        # Zero gradients
        model.zero_grad()
        if patch_out.grad is not None:
            patch_out.grad.zero_()

        # Backward from this character's logit only
        target = logits[t_c, 0, k_c]
        target.backward(retain_graph=True)

        # Grad-CAM: importance_c[t] = ReLU(Σ_d grad_{t,d} * A_{t,d})
        grad = patch_out.grad[0]  # [Wp, D]
        act = patch_out[0].detach()  # [Wp, D]
        cam = torch.relu((grad * act).sum(dim=-1))  # [Wp]
        cam = cam.detach().cpu().numpy()

        # Percentile normalization
        if cam.max() > 0:
            p1, p99 = np.percentile(cam, 1), np.percentile(cam, 99)
            if p99 > p1:
                cam = np.clip((cam - p1) / (p99 - p1), 0, 1)
            else:
                cam = cam / cam.max()

            # Adaptive gamma correction
            peak_ratio = cam.max() / (cam.mean() + 1e-8)
            gamma_eff = gamma * min(peak_ratio / 5.0, 2.0)
            gamma_eff = max(0.2, min(gamma_eff, 1.0))
            cam = np.power(cam, gamma_eff)

            # Final [0,1] normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        per_char_cams[f"{label}_{i}"] = cam

    return per_char_cams, decoded, Wp


def plot_gradcam(
    image_paths, model, charset, num_registers, device, output_path,
    run_label=""
):
    """
    Fig 5: Per-Character Grad-CAM Visualization.

    Following the paper's CE-style approach (paper_fig5_ce documentation):
    - One backward pass per character from its specific CTC logit
    - importance_c[t] = ReLU(Σ_d ∂logit(c)/∂A_{t,d} · A_{t,d})
    - Hooks into transformer output activations (NOT CNN stem)
    - Uses inferno colormap with α=0.75 overlay, matching the paper

    Layout: rows = images, col 0 = input image, cols 1..N = per-char Grad-CAM
    """
    n_images = len(image_paths)
    max_chars = 10

    all_data = []
    for img_path in image_paths:
        img_tensor, img_np, text_bbox, _ = prepare_image(img_path, device)
        per_char_cams, decoded, Wp = compute_gradcam(model, img_tensor, charset)
        n_show = min(len(decoded), max_chars)
        all_data.append((img_np, decoded, per_char_cams, Wp, text_bbox, n_show))

    actual_max = max(d[5] for d in all_data) if all_data else max_chars
    n_cols = actual_max + 1

    fig, axes = plt.subplots(
        n_images, n_cols,
        figsize=(1.8 * n_cols, 2.0 * n_images),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for row, (img_np, decoded, per_char_cams, Wp, text_bbox, n_show) in enumerate(all_data):
        x_start, x_end = text_bbox
        img_crop = img_np[:, x_start:x_end]
        crop_w = x_end - x_start

        # Column 0: input image with decoded text
        axes[row, 0].imshow(img_crop, cmap="gray", aspect="auto")
        axes[row, 0].set_title("Input", fontsize=8, fontweight="bold")
        axes[row, 0].set_xlabel(f'"{decoded}"', fontsize=7, style="italic")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for c_idx in range(n_show):
            ax = axes[row, c_idx + 1]
            char = decoded[c_idx] if c_idx < len(decoded) else ""
            display_char = char if char != " " else "SPC"

            key = f"{display_char}_{c_idx}"
            if key in per_char_cams:
                cam_1d = per_char_cams[key]
                # Convert 1D patch-level cam to image space, cropped
                attn_2d, _ = attn_to_image_crop(cam_1d, Wp, text_bbox, smooth_sigma=2)
            else:
                attn_2d = np.zeros((IMAGE_H, crop_w))

            ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.25)
            ax.imshow(attn_2d, cmap="inferno", aspect="auto", alpha=0.75,
                      vmin=0, vmax=1)
            ax.set_title(f"'{display_char}'", fontsize=9, fontweight="bold",
                         color="darkred")
            ax.set_xticks([])
            ax.set_yticks([])

        for c_idx in range(n_show, actual_max):
            ax = axes[row, c_idx + 1]
            ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)
            ax.set_title("(pad)", fontsize=7, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Fig. 5: Per-Character Grad-CAM  ({run_label})\n"
        r"$\mathrm{importance}_c[t] = \mathrm{ReLU}(\sum_d \frac{\partial \mathrm{logit}(c)}{\partial A_{t,d}} \cdot A_{t,d})$"
        "  —  one backward pass per character",
        fontsize=11, fontweight="bold", y=1.03,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HTR Attention Visualization — Publication-Quality Figures"
    )
    parser.add_argument(
        "--runs", nargs="*", default=list(DEFAULT_RUNS.keys()),
        help="Run IDs to load (default: run_55 run_58 run_53 run_54)",
    )
    parser.add_argument(
        "--images", nargs="*", default=None,
        help="Image paths (default: sample images from notebook/sample_images/)",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu / cuda:0)")
    parser.add_argument(
        "--out_dir", default=str(OUTPUT_DIR),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--max_chars", type=int, default=10,
        help="Max characters per row in Fig 2",
    )
    parser.add_argument(
        "--char_grid_run", default=None,
        help="Specific run for character attention grid (default: best 4-reg model)",
    )
    parser.add_argument(
        "--layer_run", default=None,
        help="Specific run for layer-wise attention (default: best 4-reg model)",
    )
    args = parser.parse_args()

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve image paths
    if args.images:
        image_paths = [str(Path(p).resolve()) for p in args.images]
    else:
        image_paths = [str(SAMPLE_DIR / name) for name in DEFAULT_IMAGES]

    # Verify images exist
    for p in image_paths:
        if not os.path.isfile(p):
            print(f"ERROR: Image not found: {p}")
            sys.exit(1)

    print("=" * 70)
    print("HTR Attention Visualization")
    print("=" * 70)
    print(f"Device:  {device}")
    print(f"Output:  {out_dir}")
    print(f"Images:  {len(image_paths)}")
    for p in image_paths:
        print(f"  - {Path(p).name}")
    print()

    # Load models
    print("Loading models...")
    charset = load_charset()
    models_info = []  # list of (run_id, model, num_registers)

    for run_id in args.runs:
        try:
            model, _, n_regs = load_model(run_id, device)
            models_info.append((run_id, model, n_regs))
            print(f"  {run_id}: {n_regs} registers, loaded OK")
        except Exception as e:
            print(f"  {run_id}: FAILED — {e}")

    if not models_info:
        print("ERROR: No models loaded successfully.")
        sys.exit(1)

    # Sort by register count for consistent ordering
    models_info.sort(key=lambda x: x[2])
    print()

    # Load ground truth
    gt_path = SAMPLE_DIR / "gt.txt"
    gt_map = {}
    if gt_path.exists():
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    gt_map[parts[0]] = parts[1]

    # ---- Figure 1: Register Impact Grid ----
    print("Generating Fig 1: Register Impact Grid...")
    plot_register_impact_grid(
        image_paths, models_info, charset, device,
        str(out_dir / "fig1_register_impact_grid.pdf"),
    )

    # ---- Figure 2: Character Attention Grid ----
    # Use the best 4-register model by default, or user-specified
    char_run = args.char_grid_run
    if char_run is None:
        # Pick first model with registers > 0, or fallback to first
        char_candidates = [(r, m, n) for r, m, n in models_info if n > 0]
        if char_candidates:
            char_run_id, char_model, char_nregs = char_candidates[0]
        else:
            char_run_id, char_model, char_nregs = models_info[0]
    else:
        char_model, _, char_nregs = load_model(char_run, device)
        char_run_id = char_run

    print(f"Generating Fig 2: Character Attention Grid ({char_run_id}, {char_nregs} regs)...")
    plot_char_attention_grid(
        image_paths, char_model, charset, char_nregs, device,
        str(out_dir / "fig2_character_attention_grid.pdf"),
        max_chars=args.max_chars,
        run_label=f"{char_run_id} — {char_nregs} registers",
    )

    # Also generate Fig 2 for 0-register baseline for comparison
    baseline_candidates = [(r, m, n) for r, m, n in models_info if n == 0]
    if baseline_candidates:
        bl_run_id, bl_model, bl_nregs = baseline_candidates[0]
        print(f"Generating Fig 2b: Character Attention Grid baseline ({bl_run_id}, 0 regs)...")
        plot_char_attention_grid(
            image_paths, bl_model, charset, bl_nregs, device,
            str(out_dir / "fig2b_character_attention_grid_baseline.pdf"),
            max_chars=args.max_chars,
            run_label=f"{bl_run_id} — 0 registers (baseline)",
        )

    # ---- Figure 3: Token Norm Artifact Map ----
    print("Generating Fig 3: Token Norm Artifact Map...")
    plot_token_norm_comparison(
        image_paths, models_info, charset, device,
        str(out_dir / "fig3_token_norm_artifacts.pdf"),
    )

    # ---- Figure 4: Layer-wise Attention ----
    layer_run = args.layer_run
    if layer_run is None:
        layer_run_id, layer_model, layer_nregs = (
            char_candidates[0] if char_candidates else models_info[0]
        )
    else:
        layer_model, _, layer_nregs = load_model(layer_run, device)
        layer_run_id = layer_run

    # Use first image for layerwise (cleaner visualization)
    print(f"Generating Fig 4: Layer-wise Attention ({layer_run_id}, {layer_nregs} regs)...")
    plot_layerwise_attention(
        image_paths[0], layer_model, charset, layer_nregs, device,
        str(out_dir / "fig4_layerwise_attention.pdf"),
        run_label=f"{layer_run_id} — {layer_nregs} registers",
    )

    # ---- Figure 5: GradCAM ----
    print(f"Generating Fig 5: GradCAM ({char_run_id}, {char_nregs} regs)...")
    plot_gradcam(
        image_paths, char_model, charset, char_nregs, device,
        str(out_dir / "fig5_gradcam.pdf"),
        run_label=f"{char_run_id} — {char_nregs} registers",
    )

    # ---- Summary ----
    print()
    print("=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {out_dir}")
    print("Files:")
    for f in sorted(out_dir.glob("fig*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} ({size_kb:.0f} KB)")

    # Print model comparison summary
    print("\n--- Model Comparison ---")
    print(f"{'Run':>10s}  {'Regs':>4s}  {'Prediction (first image)':>40s}")
    print("-" * 60)
    img_tensor, _, _, _ = prepare_image(image_paths[0], device)
    gt_key = Path(image_paths[0]).stem
    gt_text = gt_map.get(gt_key, "N/A")
    print(f"{'GT':>10s}  {'':>4s}  {gt_text:>40s}")
    for run_id, model, n_regs in models_info:
        with torch.no_grad():
            logits = model.forward_explain(img_tensor)[0]
        decoded, _ = ctc_greedy_decode(logits, charset)[0]
        print(f"{run_id:>10s}  {n_regs:>4d}  {decoded:>40s}")


if __name__ == "__main__":
    main()
