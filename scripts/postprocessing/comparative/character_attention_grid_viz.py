#!/usr/bin/env python3
"""
Character-Level Patch Attention Grid – Publication-Quality Visualization
========================================================================

Generates a figure matching the layout of notebook/sample_attention.png:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Row 0: Original grayscale image (full-width, with GT & prediction) │
  ├───────┬───────┬───────┬───────┬───────┬──────┬───────┬──────────────┤
  │  "A"  │  " "  │  "m"  │  "o"  │  "v"  │  "e" │  …   │  char labels │
  │ [2D block-patch heatmap overlay on the original image per char]     │
  │  Each cell = full copy of the line image + character's attention    │
  │  heatmap as 2D block patches (not vertical strips), showing where  │
  │  in the image the model attends when predicting that character.    │
  ├───────┴───────┴───────┴───────┴───────┴──────┴───────┴──────────────┤
  │  Row 2: Character × Patch alignment matrix (attention carpet)       │
  │    X-axis = patch positions aligned with original image             │
  │    Y-axis = decoded characters                                      │
  │    Diagonal pattern = correct left-to-right spatial localisation    │
  └──────────────────────────────────────────────────────────────────────┘

Key features:
  • 2D block patches (not 1D vertical strips) — the heatmap has visible
    patch boundaries with per-patch attention intensity
  • Per-character attention varies across patches: bright blocks where
    the model "looks" for that character, dark elsewhere
  • Patch grid overlay shows the actual ViT patch structure
  • Supports multi-model comparison (rows = register counts 0→16)
  • Alignment carpet at the bottom shows character × patch matrix

Usage:
    python scripts/postprocessing/character_attention_grid_viz.py \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --image notebook/sample_images/a01-038-12.png

    # Multiple models (register sweep comparison)
    python scripts/postprocessing/character_attention_grid_viz.py \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_51/model.pt \\
        --model-path saved_models/experiments/run_52/model.pt \\
        --model-path saved_models/experiments/run_53/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --image notebook/sample_images/a01-038-12.png

    # Custom options
    python scripts/postprocessing/character_attention_grid_viz.py \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --image notebook/sample_images/a01-038-12.png \\
        --layer last --max-chars 25 --gamma 3.0 --alpha 0.6 --dpi 200 \\
        --save-dir output/attn_grid/ --style both
"""

import os
import sys
import json
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import ConnectionPatch
from PIL import Image

# ── Project root ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import HTRNet
from utils.preprocessing import load_image, preprocess

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FIXED_SIZE = (128, 1024)
BORDER_SIZE = 8
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "IAM", "processed_lines")

# Visual constants for the two output styles
PAPER_BG   = "#0B0B0F"
PAPER_FG   = "#FFFFFF"
PAPER_CMAP = "hot"       # black→red→orange→yellow→white
OVERLAY_BG = "#FFFFFF"
OVERLAY_CMAP = "inferno"  # perceptually uniform


# ═════════════════════════════════════════════════════════════════════════════
# CLI Argument Parsing
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Character-level patch attention grid visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-path", action="append", required=True,
                   help="Path to model checkpoint (.pt). Repeat for multi-model.")
    p.add_argument("--image", required=True,
                   help="Path to input image or directory of images.")
    p.add_argument("--layer", default="last",
                   choices=["first", "middle", "last", "all", "rollout"],
                   help="Which attention layer to use (default: last)")
    p.add_argument("--max-chars", type=int, default=25,
                   help="Max characters to show per row (default: 25)")
    p.add_argument("--alpha", type=float, default=0.60,
                   help="Heatmap overlay opacity (default: 0.60)")
    p.add_argument("--gamma", type=float, default=3.0,
                   help="Contrast exponent (default: 3.0)")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output DPI (default: 150)")
    p.add_argument("--save-dir", default="",
                   help="Output directory (default: backup/post processing_v1/char_attn_grid/)")
    p.add_argument("--style", default="both", choices=["paper", "overlay", "both"],
                   help="Output style: paper (dark), overlay (white), both")
    p.add_argument("--show-grid-lines", action="store_true",
                   help="Draw visible patch grid lines on heatmap overlays")
    p.add_argument("--config-yaml", nargs="*", default=[],
                   help="Optional config YAML files (auto-detected from config.json)")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Model Utilities
# ═════════════════════════════════════════════════════════════════════════════

def get_run_info(model_path):
    """Extract run metadata from config.json in the model directory."""
    run_dir = os.path.dirname(os.path.abspath(model_path))
    run_name = os.path.basename(run_dir)
    config_json = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_json):
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(config_json) as f:
        cfg = json.load(f)
    num_reg = cfg.get("arch", {}).get("num_registers", 0)
    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "num_registers": num_reg,
        "label": f"Reg-{num_reg}",
        "config": cfg,
    }


def build_model(config_dict, model_path, num_classes, device):
    """Build and load a ViT-RGTS model from a config dict."""
    from omegaconf import OmegaConf
    arch_cfg = OmegaConf.create(config_dict["arch"])
    net = HTRNet(arch_cfg, num_classes)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt, strict=True)
    net.to(device).eval()
    return net


def load_classes():
    classes_path = os.path.join(DATASET_PATH, "classes.npy")
    classes = np.load(classes_path, allow_pickle=True)
    num_classes = len(classes) + 1
    i2c = {(i + 1): str(c) for i, c in enumerate(classes)}
    return classes, num_classes, i2c


# ═════════════════════════════════════════════════════════════════════════════
# Attention Extraction
# ═════════════════════════════════════════════════════════════════════════════

def attention_rollout(attn_maps_list):
    """Compute attention rollout across all layers → [S, S] numpy."""
    result = None
    for attn in attn_maps_list:
        attn_avg = attn.mean(dim=1)
        S = attn_avg.size(-1)
        I = torch.eye(S, device=attn_avg.device).unsqueeze(0)
        attn_res = 0.5 * attn_avg + 0.5 * I
        attn_res = attn_res / (attn_res.sum(dim=-1, keepdim=True) + 1e-8)
        result = attn_res if result is None else torch.bmm(attn_res, result)
    return result[0].cpu().numpy()


def select_layer(attn_maps_list, layer="last"):
    """Select and head-average one layer → [S, S] numpy."""
    L = len(attn_maps_list)
    if layer == "rollout":
        return attention_rollout(attn_maps_list)
    pick = {"first": 0, "middle": L // 2, "last": -1, "all": None}[layer]
    if pick is None:
        attn = torch.stack(attn_maps_list, 0).mean(0)
    else:
        attn = attn_maps_list[pick]
    if attn.dim() == 4:
        attn = attn[0].mean(0)
    elif attn.dim() == 3:
        attn = attn[0]
    return attn.cpu().numpy()


def extract_attention(net, img_tensor, device, layer):
    """Run forward_explain → attention matrix, logits, grid."""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits, reg_tokens, attn_maps_list, token_norms, (Hp, Wp) = \
            net.forward_explain(img_tensor)
    if isinstance(logits, tuple):
        logits = logits[0]
    num_reg = reg_tokens.shape[1] if reg_tokens is not None else 0
    attn = select_layer(attn_maps_list, layer)
    return {
        "attn": attn,
        "logits": logits.cpu().numpy(),
        "grid_size": (Hp, Wp),
        "num_reg": num_reg,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CTC Decoding
# ═════════════════════════════════════════════════════════════════════════════

def ctc_char_positions(logits_np, i2c, blank_id=0):
    """CTC greedy decode → (chars, positions)."""
    tdec = logits_np.argmax(2).squeeze()
    chars, positions, prev = [], [], -1
    for t, v in enumerate(tdec):
        v = int(v)
        if v != blank_id and v != prev:
            chars.append(str(i2c.get(v, "?")))
            positions.append(t)
        prev = v
    return chars, positions


def char_display(c):
    """Safe display label for a character."""
    c = str(c)
    if c in (" ", "\t", "\n"):
        return repr(c)
    return c


# ═════════════════════════════════════════════════════════════════════════════
# Patch-Aligned Spatial Mapping
# ═════════════════════════════════════════════════════════════════════════════

def upscale_patch_aligned(patch_map, H_img, W_img,
                          fixed_size=FIXED_SIZE, border_size=BORDER_SIZE):
    """Map (Hp, Wp) patch heatmap → (H_img, W_img) pixel heatmap.
    Uses nearest-neighbor (block) upscaling to preserve patch boundaries."""
    Hp, Wp = patch_map.shape
    H_pre, W_pre = fixed_size
    ph = H_pre // Hp
    pw = W_pre // Wp

    # Nearest-neighbor block upscale — preserves crisp patch boundaries
    heat_full = np.kron(patch_map, np.ones((ph, pw)))
    r0, c0 = border_size, border_size
    crop = heat_full[r0:r0 + H_img, c0:c0 + W_img]

    if crop.shape[0] < H_img or crop.shape[1] < W_img:
        padded = np.zeros((H_img, W_img), dtype=crop.dtype)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    return crop


def compute_content_geometry(H_img, W_img, grid_size,
                             fixed_size=FIXED_SIZE, border_size=BORDER_SIZE):
    """Compute patch ↔ pixel mapping geometry."""
    H_pre, W_pre = fixed_size
    Hp, Wp = grid_size
    pw = W_pre // Wp
    n_height = min(H_pre - 2 * border_size, H_img)
    scale = n_height / H_img
    n_width = min(W_pre - 2 * border_size, int(scale * W_img))
    p_start = border_size // pw
    p_end = min(Wp, (border_size + n_width + pw - 1) // pw)
    img_left = border_size / pw
    img_right = (border_size + n_width) / pw
    return dict(scale=scale, n_width=n_width, pw=pw,
                p_start=p_start, p_end=p_end,
                img_left=img_left, img_right=img_right)


# ═════════════════════════════════════════════════════════════════════════════
# Per-Character Attention Heatmap Builder
# ═════════════════════════════════════════════════════════════════════════════

def build_char_heatmaps(attn, grid_size, num_reg, chars, positions,
                        H_img, W_img, max_chars, gamma, pclip=99.0):
    """
    Build per-character attention heatmaps as 2D block-patch grids.

    Each heatmap preserves the patch structure: the attention for character i
    at time step t is a row in the attention matrix, which is reshaped into
    the (Hp, Wp) spatial grid and upscaled with nearest-neighbor to produce
    visible block patches.

    Returns:
        heatmaps  : list of [H_img, W_img] arrays, normalised [0, 1]
        n_show    : number of characters shown
        xmax_list : pixel x-position of attention centroid for each char
        alignment : [n_show, T] matrix for alignment carpet
    """
    Hp, Wp = grid_size
    T = Hp * Wp
    n_show = min(len(chars), max_chars)

    # Adaptive gamma
    n_chars = max(1, len(chars))
    gamma_eff = float(np.clip(gamma + 0.05 * (n_chars - 6), gamma, gamma + 3.0))

    heatmaps = []
    xmax_list = []
    char_vectors = []

    for ci in range(n_show):
        t = positions[ci]
        row_idx = num_reg + t

        if row_idx >= attn.shape[0]:
            patch_attn = np.ones(T) / T
        else:
            patch_attn = attn[row_idx, num_reg:num_reg + T].copy()
            if patch_attn.shape[0] < T:
                patch_attn = np.pad(patch_attn, (0, T - patch_attn.shape[0]))

        char_vectors.append(patch_attn.copy())

        # Percentile normalisation
        p_lo = max(0.0, np.percentile(patch_attn, 100.0 - pclip))
        p_hi = np.percentile(patch_attn, pclip)
        if p_hi - p_lo < 1e-8:
            p_lo, p_hi = patch_attn.min(), patch_attn.max()
        patch_attn = np.clip(patch_attn, p_lo, p_hi)
        patch_attn = (patch_attn - p_lo) / (p_hi - p_lo + 1e-8)

        # Power contrast
        patch_attn = patch_attn ** gamma_eff

        # Xmax: attention centroid in patch space → pixel space
        patch_grid = patch_attn.reshape(Hp, Wp)
        col_weight = patch_grid.sum(axis=0)
        col_weight = col_weight / (col_weight.sum() + 1e-12)
        xmax_patch = float(np.dot(col_weight, np.arange(Wp)))
        pw = FIXED_SIZE[1] // Wp
        xmax_px = int(xmax_patch * pw + pw // 2) - BORDER_SIZE
        xmax_px = max(0, min(xmax_px, W_img - 1))
        xmax_list.append(xmax_px)

        # Upscale to image space (block patches preserved)
        heatmap = upscale_patch_aligned(patch_grid, H_img, W_img)
        heatmaps.append(heatmap)

    # Build alignment matrix
    alignment = np.zeros((n_show, T))
    for ci in range(n_show):
        vec = char_vectors[ci]
        p_lo = max(0.0, np.percentile(vec, 100.0 - pclip))
        p_hi = np.percentile(vec, pclip)
        if p_hi - p_lo < 1e-8:
            p_lo, p_hi = vec.min(), vec.max()
        normed = np.clip(vec, p_lo, p_hi)
        normed = (normed - p_lo) / (p_hi - p_lo + 1e-8)
        normed = normed ** gamma_eff
        alignment[ci] = normed

    return heatmaps, n_show, xmax_list, alignment


# ═════════════════════════════════════════════════════════════════════════════
# Ground Truth
# ═════════════════════════════════════════════════════════════════════════════

def load_gt(image_paths):
    gt = {}
    seen = set()
    for p in image_paths:
        d = os.path.dirname(os.path.abspath(p))
        if d in seen:
            continue
        seen.add(d)
        gf = os.path.join(d, "gt.txt")
        if os.path.isfile(gf):
            with open(gf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        gt[parts[0]] = parts[1]
    return gt


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: Paper-Style Grid (dark bg, 2D block-patch heatmaps)
# ═════════════════════════════════════════════════════════════════════════════

def render_paper_grid(rows, save_path, dpi=150, show_grid_lines=False):
    """
    Generate the publication figure matching sample_attention.png:

    Layout:
      Row 0        : Original image (full width) with pred/GT text
      Rows 1..N    : Per-model character attention (each character is a cell
                     showing the full image with block-patch heatmap overlay)
      Bottom       : Alignment matrix (character × patch position)

    Each heatmap cell shows clear 2D block patches — the attention value
    is constant within each patch region, creating a visible grid pattern
    that varies per character.
    """
    if not rows:
        return

    n_models = len(rows)
    # Use the first row to determine character count
    max_chars = max(r["n_show"] for r in rows)
    if max_chars == 0:
        return

    img_np = rows[0]["img_np"]
    H_img, W_img = img_np.shape[:2]
    aspect = W_img / H_img

    # ── Sizing ────────────────────────────────────────────────────────
    cell_h = 1.8
    cell_w = max(0.9, min(cell_h * aspect / max_chars * 2, 3.0))
    # Make sure cells are wide enough when few characters
    cell_w = max(cell_w, min(2.5, 14.0 / max_chars))

    fig_w = max(10, cell_w * max_chars + 1.5)
    orig_h = max(1.5, cell_h * 0.55)
    carpet_h = cell_h * n_models
    align_h = max(1.8, max_chars * 0.22) if n_models == 1 else 0
    fig_h = orig_h + carpet_h + align_h + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PAPER_BG)

    n_grid_rows = 1 + n_models + (1 if align_h > 0 else 0)
    h_ratios = [orig_h] + [cell_h] * n_models
    if align_h > 0:
        h_ratios.append(align_h)

    gs_main = gridspec.GridSpec(
        n_grid_rows, 1,
        figure=fig,
        height_ratios=h_ratios,
        hspace=0.08,
        left=0.04, right=0.96,
        top=0.93, bottom=0.04,
    )

    # ── Row 0: Original image ─────────────────────────────────────────
    ax_orig = fig.add_subplot(gs_main[0])
    ax_orig.set_facecolor(PAPER_BG)
    ax_orig.imshow(img_np, cmap="gray", aspect="auto")
    ax_orig.axis("off")

    # Title with prediction/GT
    r0 = rows[0]
    title_parts = [f'{r0.get("image_name", "")}']
    if r0.get("gt_text"):
        title_parts.append(f'GT: "{r0["gt_text"]}"')
    title_parts.append(f'Pred: "{r0["pred_text"]}"')
    ax_orig.set_title(
        "  |  ".join(title_parts),
        fontsize=9, color=PAPER_FG, fontweight="bold", pad=6,
        fontfamily="monospace",
    )

    # ── Rows 1..N: Per-model character attention grids ────────────────
    for mi, row in enumerate(rows):
        chars = row["chars"]
        heatmaps = row["heatmaps"]
        n_show = row["n_show"]
        xmax = row["xmax"]
        label = row["label"]

        gs_row = gs_main[1 + mi].subgridspec(
            1, max_chars + 1,  # +1 for label
            width_ratios=[0.15] + [1.0] * max_chars,
            wspace=0.03,
        )

        # Label cell
        ax_label = fig.add_subplot(gs_row[0, 0])
        ax_label.set_facecolor(PAPER_BG)
        ax_label.text(
            0.5, 0.5, label,
            transform=ax_label.transAxes,
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            color=PAPER_FG, fontfamily="monospace",
            rotation=90 if len(label) > 5 else 0,
        )
        ax_label.axis("off")

        # Character cells
        for ci in range(max_chars):
            ax = fig.add_subplot(gs_row[0, ci + 1])
            ax.set_facecolor(PAPER_BG)

            if ci < n_show:
                # Show heatmap as 2D block patches (no smoothing)
                ax.imshow(img_np, cmap="gray", aspect="auto")
                ax.imshow(heatmaps[ci], cmap=PAPER_CMAP, alpha=0.65,
                          vmin=0.0, vmax=1.0, aspect="auto",
                          interpolation="nearest")  # block patches!

                # Optional patch grid lines
                if show_grid_lines:
                    Hp, Wp = row["grid_size"]
                    pw = FIXED_SIZE[1] // Wp
                    ph = FIXED_SIZE[0] // Hp
                    for px in range(0, W_img, pw):
                        ax.axvline(x=px, color="#ffffff", linewidth=0.15, alpha=0.3)
                    for py in range(0, H_img, ph):
                        ax.axhline(y=py, color="#ffffff", linewidth=0.15, alpha=0.3)

                # Xmax marker
                ax.axvline(x=xmax[ci], color="#00FF88", linewidth=0.7,
                           linestyle="--", alpha=0.8)

                # Character label (top row only)
                if mi == 0:
                    ch_label = char_display(chars[ci])
                    ax.set_title(
                        ch_label, color=PAPER_FG, fontsize=7,
                        fontweight="bold", pad=2, fontfamily="monospace",
                    )
            else:
                ax.imshow(np.zeros((4, 4)), cmap=PAPER_CMAP,
                          vmin=0, vmax=1, aspect="auto")

            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")
                sp.set_linewidth(0.4)

    # ── Bottom: Alignment matrix (single model only) ──────────────────
    if align_h > 0 and n_models == 1:
        row = rows[0]
        alignment = row["alignment"]
        grid_size = row["grid_size"]
        chars = row["chars"]
        n_show = row["n_show"]
        num_reg = row["num_reg"]

        geo = compute_content_geometry(H_img, W_img, grid_size)
        margin = 2
        T = grid_size[0] * grid_size[1]
        p_lo = max(0, geo["p_start"] - margin)
        p_hi = min(T, geo["p_end"] + margin)
        alignment_crop = alignment[:n_show, p_lo:p_hi]

        ax_mat = fig.add_subplot(gs_main[n_grid_rows - 1])
        ax_mat.set_facecolor(PAPER_BG)
        im = ax_mat.imshow(
            alignment_crop, cmap=PAPER_CMAP, aspect="auto",
            vmin=0.0, vmax=1.0, interpolation="nearest",
            extent=[p_lo, p_hi, n_show - 0.5, -0.5],
        )

        char_labels = [char_display(c) for c in chars[:n_show]]
        ax_mat.set_yticks(range(n_show))
        ax_mat.set_yticklabels(
            char_labels, fontsize=5, fontfamily="monospace", color=PAPER_FG,
        )
        ax_mat.set_xlabel(
            "Patch position (left → right)", fontsize=7, color=PAPER_FG,
        )
        ax_mat.set_ylabel("Char", fontsize=7, color=PAPER_FG)
        ax_mat.tick_params(colors=PAPER_FG, labelsize=5)

        # Peak path diagonal
        if n_show >= 2:
            peaks = [int(alignment[ci].argmax()) + 0.5 for ci in range(n_show)]
            ax_mat.plot(peaks, range(n_show), "w--", linewidth=0.8, alpha=0.6,
                        label="peak path")
            ax_mat.legend(fontsize=5, loc="lower right",
                          framealpha=0.5, edgecolor="none",
                          facecolor="#333333", labelcolor=PAPER_FG)

        cbar = fig.colorbar(im, ax=ax_mat, fraction=0.015, pad=0.01)
        cbar.ax.tick_params(labelsize=5, colors=PAPER_FG)
        cbar.set_label("attention", fontsize=6, color=PAPER_FG)

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=PAPER_BG, edgecolor="none")
    plt.close(fig)
    print(f"  [paper] saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: Overlay-Style Grid (white bg, image + heatmap overlay)
# ═════════════════════════════════════════════════════════════════════════════

def render_overlay_grid(rows, save_path, alpha=0.55, dpi=150,
                        show_grid_lines=False):
    """
    White-background overlay grid. Same layout as paper grid but with
    grayscale image + inferno heatmap overlay per character cell.
    """
    if not rows:
        return

    n_models = len(rows)
    max_chars = max(r["n_show"] for r in rows)
    if max_chars == 0:
        return

    img_np = rows[0]["img_np"]
    H_img, W_img = img_np.shape[:2]
    aspect = W_img / H_img

    # ── Sizing ────────────────────────────────────────────────────────
    cell_h = 2.0
    cell_w = max(1.0, min(cell_h * aspect / max_chars * 2, 3.5))
    cell_w = max(cell_w, min(3.0, 16.0 / max_chars))

    n_cols = 1 + max_chars  # original + chars
    fig_w = max(10, cell_w * n_cols + 0.5)
    fig_h = cell_h * n_models + 1.2

    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        facecolor=OVERLAY_BG,
        gridspec_kw=dict(hspace=0.08, wspace=0.03),
    )

    for mi, row in enumerate(rows):
        img = row["img_np"]
        chars = row["chars"]
        heatmaps = row["heatmaps"]
        n_show = row["n_show"]
        xmax = row["xmax"]
        label = row["label"]

        # Column 0: original image
        ax0 = axes[mi, 0]
        ax0.imshow(img, cmap="gray", aspect="auto")
        ax0.set_facecolor(OVERLAY_BG)
        if mi == 0:
            ax0.set_title("Original", fontsize=8, fontweight="bold",
                          color="#222222", pad=3)

        info = label
        if row.get("pred_text"):
            pt = row["pred_text"]
            if len(pt) > 25:
                pt = pt[:22] + "…"
            info += f'\n"{pt}"'
        ax0.text(
            0.02, 0.97, info,
            transform=ax0.transAxes, va="top", ha="left", fontsize=5,
            color="white", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#222", alpha=0.8),
        )
        ax0.set_xticks([])
        ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_edgecolor("#DDD")
            sp.set_linewidth(0.5)

        # Columns 1..N: character overlays
        last_im = None
        for ci in range(max_chars):
            ax = axes[mi, 1 + ci]
            ax.set_facecolor(OVERLAY_BG)

            if ci < n_show:
                ax.imshow(img, cmap="gray", aspect="auto")
                im = ax.imshow(
                    heatmaps[ci], cmap=OVERLAY_CMAP, alpha=alpha,
                    vmin=0.0, vmax=1.0, aspect="auto",
                    interpolation="nearest",  # crisp block patches
                )
                last_im = im

                # Grid lines
                if show_grid_lines:
                    Hp, Wp = row["grid_size"]
                    pw = FIXED_SIZE[1] // Wp
                    ph = FIXED_SIZE[0] // Hp
                    for px in range(0, W_img, pw):
                        ax.axvline(x=px, color="#000", linewidth=0.12, alpha=0.2)
                    for py in range(0, H_img, ph):
                        ax.axhline(y=py, color="#000", linewidth=0.12, alpha=0.2)

                # Xmax marker
                ax.axvline(x=xmax[ci], color="#00FF88", linewidth=0.8,
                           linestyle="--", alpha=0.9)

                if mi == 0:
                    ax.set_title(
                        f'"{char_display(chars[ci])}"',
                        fontsize=7, fontweight="bold", color="#222", pad=3,
                        fontfamily="monospace",
                    )
            else:
                ax.imshow(np.ones_like(img) * 240, cmap="gray",
                          vmin=0, vmax=255, aspect="auto")

            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#DDD")
                sp.set_linewidth(0.5)

    fig.suptitle(
        f"Character-Level Attention Grid | Colormap: {OVERLAY_CMAP} | α={alpha}",
        fontsize=9, color="#444", y=1.01,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=OVERLAY_BG, edgecolor="none")
    plt.close(fig)
    print(f"  [overlay] saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Collect images
# ═════════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def collect_images(path):
    """Collect image paths from a file or directory."""
    if os.path.isdir(path):
        return sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
    elif os.path.isfile(path):
        return [path]
    else:
        print(f"Warning: {path} not found")
        return []


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load classes
    classes, num_classes, i2c = load_classes()

    # Collect images
    image_paths = collect_images(args.image)
    if not image_paths:
        sys.exit(f"Error: no images found at {args.image}")

    # Load ground truth
    gt_dict = load_gt(image_paths)

    # Model info
    model_infos = []
    for mp in args.model_path:
        info = get_run_info(mp)
        info["model_path"] = mp
        model_infos.append(info)

    # Sort by register count
    model_infos.sort(key=lambda x: x["num_registers"])

    # Output directory
    save_dir = args.save_dir or os.path.join(
        PROJECT_ROOT, "backup", "post processing_v1", "char_attn_grid"
    )

    print("=" * 70)
    print("  Character-Level Patch Attention Grid Visualization")
    print("=" * 70)
    print(f"  Images      : {len(image_paths)}")
    print(f"  Models      : {len(model_infos)}")
    for info in model_infos:
        print(f"    {info['label']:>8s}  ({info['run_name']})")
    print(f"  Layer       : {args.layer}")
    print(f"  Max chars   : {args.max_chars}")
    print(f"  Gamma       : {args.gamma}")
    print(f"  Alpha       : {args.alpha}")
    print(f"  Style       : {args.style}")
    print(f"  Grid lines  : {args.show_grid_lines}")
    print(f"  Output      : {save_dir}")
    print()

    # ── Process each image ────────────────────────────────────────────
    for img_idx, img_path in enumerate(image_paths):
        image_name = Path(img_path).stem
        print(f"[{img_idx + 1}/{len(image_paths)}] {image_name}")

        # Load and preprocess
        raw_img = load_image(img_path)
        processed = preprocess(raw_img, FIXED_SIZE, BORDER_SIZE)
        img_tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)

        img_np = np.array(Image.open(img_path).convert("L"))
        H_img, W_img = img_np.shape
        gt_text = gt_dict.get(image_name, "")

        # Process each model
        rows = []
        for mi, info in enumerate(model_infos):
            print(f"  Model: {info['label']}")

            net = build_model(info["config"], info["model_path"],
                              num_classes, device)

            result = extract_attention(net, img_tensor, device, args.layer)
            chars, positions = ctc_char_positions(result["logits"], i2c)
            pred_text = "".join(chars)

            heatmaps, n_show, xmax, alignment = build_char_heatmaps(
                result["attn"], result["grid_size"], result["num_reg"],
                chars, positions, H_img, W_img,
                args.max_chars, args.gamma,
            )

            print(f"    Pred: \"{pred_text}\"")
            if gt_text:
                print(f"    GT:   \"{gt_text}\"")
            print(f"    Chars: {n_show}, Grid: {result['grid_size']}, "
                  f"Reg: {result['num_reg']}")

            rows.append({
                "label": info["label"],
                "chars": chars[:n_show],
                "heatmaps": heatmaps,
                "n_show": n_show,
                "xmax": xmax,
                "alignment": alignment,
                "img_np": img_np,
                "grid_size": result["grid_size"],
                "num_reg": result["num_reg"],
                "pred_text": pred_text,
                "gt_text": gt_text,
                "image_name": image_name,
            })

            # Free GPU memory
            del net
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # ── Render figures ────────────────────────────────────────────
        if len(model_infos) > 1:
            suffix = f"_{args.layer}"
        else:
            suffix = f"_{model_infos[0]['run_name']}_{args.layer}"

        if args.style in ("paper", "both"):
            paper_path = os.path.join(
                save_dir, "paper_grid", f"{image_name}{suffix}.png"
            )
            render_paper_grid(rows, paper_path, dpi=args.dpi,
                              show_grid_lines=args.show_grid_lines)

        if args.style in ("overlay", "both"):
            overlay_path = os.path.join(
                save_dir, "overlay_grid", f"{image_name}{suffix}.png"
            )
            render_overlay_grid(
                rows, overlay_path, alpha=args.alpha, dpi=args.dpi,
                show_grid_lines=args.show_grid_lines,
            )

        print()

    print(f"All outputs → {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
