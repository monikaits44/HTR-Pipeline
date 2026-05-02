#!/usr/bin/env python3
"""
Register Effect on CTC Self-Attention — Clean Visualization
============================================================

Architectural context:
  This ViT uses CTC + self-attention (encoder-only).
  Unlike cross-attention models ("Beyond Memorization" Fig 5),
  there are NO dedicated character query embeddings.

  Self-attention A[t,j] = "how much does column t look at column j"
  Character attention = A[t_c, :] where t_c is CTC peak for char c.

  CNN stem: 128x1024 input → 128 column tokens (Hp=1, Wp=128).
  Each token covers ~8px horizontally. Attention is inherently 1D.

Why previous maps looked blurry:
  1. Head averaging: 8 heads averaged, destroying localized heads
  2. Over-processing: gamma + percentile norm + Gaussian smoothing
  3. Fake 2D: 1D attention tiled vertically

This script fixes all three issues:
  - Uses best-head selection (most spatially localized head per character)
  - Minimal processing (min-max only, no gamma/smoothing)
  - Honest 1D attention profiles + clean 2D overlay from best head

Figures:
  Fig A: Best-head character attention grid (2D overlay, per register count)
  Fig B: 1D attention profiles comparison (direct register effect)
  Fig C: Register absorption mechanism (what registers attend to)

Usage:
    python scripts/register_effect_visualization.py --device cpu
    python scripts/register_effect_visualization.py --device cuda:0
    python scripts/register_effect_visualization.py --images path1.png path2.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

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

RUNS = {
    "run_55": 0,
    "run_58": 4,
    "run_53": 8,
    "run_54": 16,
}

REG_COLORS = {0: "#d62728", 4: "#ff7f0e", 8: "#2ca02c", 16: "#1f77b4"}
REG_LABELS = {0: "0 reg (baseline)", 4: "4 reg", 8: "8 reg", 16: "16 reg"}

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
# Core utilities (shared with attention_visualization.py)
# ============================================================================

def load_charset():
    classes = np.load(str(CLASSES_PATH), allow_pickle=True)
    return list(classes)


def load_model(run_id, device="cpu"):
    run_dir = EXPERIMENTS_DIR / run_id
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    arch_ns = SimpleNamespace(**cfg["arch"])
    charset = load_charset()
    model = HTRNet(arch_ns, len(charset) + 1)
    state = torch.load(str(run_dir / "model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    n_regs = cfg["arch"].get("num_registers", 0)
    return model, charset, n_regs


def prepare_image(image_path, device="cpu"):
    img_np = load_image(str(image_path))
    orig_shape = img_np.shape
    img_proc = preprocess(img_np, (IMAGE_H, IMAGE_W))
    tensor = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).float().to(device)
    col_var = np.var(img_proc, axis=0)
    threshold = np.median(col_var) * 0.1 + 1e-6
    active_cols = np.where(col_var > threshold)[0]
    if len(active_cols) > 0:
        x_start = max(0, active_cols[0] - 4)
        x_end = min(IMAGE_W, active_cols[-1] + 4)
    else:
        x_start, x_end = 0, IMAGE_W
    return tensor, img_proc, (x_start, x_end), orig_shape


def run_explain(model, img_tensor):
    with torch.no_grad():
        logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(img_tensor)
    return {
        "logits": logits,
        "reg_tokens": reg_tokens,
        "attn_maps": attn_maps,
        "token_norms": token_norms,
        "grid_size": grid,
    }


def ctc_greedy_decode(logits, charset):
    probs = torch.softmax(logits, dim=-1)
    B = probs.shape[1]
    results = []
    for b in range(B):
        seq = probs[:, b, :].argmax(dim=-1).cpu().numpy()
        chars, cols = [], []
        prev = -1
        for t, idx in enumerate(seq):
            if idx != 0 and idx != prev:
                if 0 <= idx - 1 < len(charset):
                    chars.append(charset[idx - 1])
                    cols.append(t)
            prev = idx
        results.append(("".join(chars), cols))
    return results


# ============================================================================
# Module 1: Per-Head Attention Extraction
# ============================================================================

def extract_per_head_attention(attn_maps, num_registers, peak_cols, grid_size,
                               layer_idx=-1):
    """
    Extract per-character attention for EACH head separately.

    Returns:
        per_head: dict[char_idx] -> np.array [H, Wp]  (raw attention per head)
        best_heads: dict[char_idx] -> int  (index of most localized head)
        best_maps: dict[char_idx] -> np.array [Wp]  (best-head attention, min-max normed)
        avg_maps: dict[char_idx] -> np.array [Wp]  (head-averaged attention, min-max normed)
    """
    attn = attn_maps[layer_idx]  # [B, H, S, S]
    R = num_registers
    _, H, S, _ = attn.shape
    Hp, Wp = grid_size

    per_head = {}
    best_heads = {}
    best_maps = {}
    avg_maps = {}

    for ci, t_c in enumerate(peak_cols):
        token_idx = R + t_c
        if token_idx >= S:
            continue

        # Per-head attention from this character's token to all patch tokens
        heads = attn[0, :, token_idx, R:R + Wp].numpy()  # [H, Wp]
        per_head[ci] = heads

        # Find most localized head (highest peak-to-mean ratio)
        localizations = heads.max(axis=1) / (heads.mean(axis=1) + 1e-8)
        best_h = int(np.argmax(localizations))
        best_heads[ci] = best_h

        # Best-head map: simple min-max normalization only
        bmap = heads[best_h].copy()
        bmap = (bmap - bmap.min()) / (bmap.max() - bmap.min() + 1e-8)
        best_maps[ci] = bmap

        # Head-averaged map: simple min-max normalization only
        amap = heads.mean(axis=0)
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        avg_maps[ci] = amap

    return per_head, best_heads, best_maps, avg_maps


def extract_register_attention(attn_maps, num_registers, grid_size, layer_idx=-1):
    """
    Extract register token attention patterns.

    Returns:
        reg_to_patch: [R, Wp] — what each register attends to (head-averaged)
        patch_to_reg: [Wp] — how much each patch token sends to registers (total)
    """
    if num_registers == 0:
        return None, None

    attn = attn_maps[layer_idx]  # [B, H, S, S]
    R = num_registers
    Hp, Wp = grid_size

    avg = attn.mean(dim=1)  # [B, S, S]

    # What each register attends to among patch tokens
    reg_to_patch = avg[0, :R, R:R + Wp].numpy()  # [R, Wp]

    # How much each patch token attends to register tokens (summed over all regs)
    patch_to_reg = avg[0, R:R + Wp, :R].sum(dim=-1).numpy()  # [Wp]

    return reg_to_patch, patch_to_reg


# ============================================================================
# Module 2: Attention-to-Image Mapping
# ============================================================================

def attn_to_image_1d(attn_1d, Wp, text_bbox):
    """
    Map 1D patch attention to image pixel space, cropped to text region.
    NO smoothing, NO gamma. Just interpolation and crop.

    Returns:
        attn_crop: 1D array [crop_w] in image pixel space
        crop_w: int
    """
    x_start, x_end = text_bbox
    crop_w = x_end - x_start
    attn_full = np.interp(
        np.linspace(0, 1, IMAGE_W), np.linspace(0, 1, Wp), attn_1d
    )
    attn_crop = attn_full[x_start:x_end]
    return attn_crop, crop_w


def attn_to_image_2d(attn_1d, Wp, text_bbox):
    """
    Map 1D patch attention to 2D image overlay, cropped to text region.
    NO smoothing, NO gamma.

    Returns:
        attn_2d: [IMAGE_H, crop_w]
        crop_w: int
    """
    attn_crop, crop_w = attn_to_image_1d(attn_1d, Wp, text_bbox)
    attn_2d = np.tile(attn_crop, (IMAGE_H, 1))
    return attn_2d, crop_w


# ============================================================================
# Figure A: Best-Head Character Attention Grid
# ============================================================================

def plot_best_head_grid(image_paths, models_dict, charset, device, output_dir,
                        max_chars=10):
    """
    One figure per image: rows=register counts, cols=characters.
    Each cell: best-head attention overlay (inferno) on grayscale image.

    This shows the CLEAREST possible attention for each character,
    using the most spatially localized head (not head-average).
    """
    os.makedirs(output_dir, exist_ok=True)

    reg_counts = sorted(models_dict.keys())

    for img_path in image_paths:
        img_name = Path(img_path).stem
        img_tensor, img_np, text_bbox, _ = prepare_image(img_path, device)
        x_start, x_end = text_bbox
        img_crop = img_np[:, x_start:x_end]

        # Collect data for all register counts
        all_data = {}
        common_decoded = None
        for n_regs in reg_counts:
            model = models_dict[n_regs]
            result = run_explain(model, img_tensor)
            decoded, peak_cols = ctc_greedy_decode(result["logits"], charset)[0]
            if common_decoded is None:
                common_decoded = decoded

            per_head, best_heads, best_maps, avg_maps = extract_per_head_attention(
                result["attn_maps"], n_regs, peak_cols, result["grid_size"]
            )
            all_data[n_regs] = {
                "decoded": decoded,
                "peak_cols": peak_cols,
                "best_heads": best_heads,
                "best_maps": best_maps,
                "avg_maps": avg_maps,
                "grid_size": result["grid_size"],
            }

        # Use the decoded text from baseline (0-reg) as reference
        ref_data = all_data[reg_counts[0]]
        n_chars = min(len(ref_data["decoded"]), max_chars)
        n_rows = len(reg_counts)
        n_cols = n_chars + 1  # +1 for image column

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.6 * n_cols, 1.8 * n_rows),
            gridspec_kw={"wspace": 0.04, "hspace": 0.30},
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        Wp = ref_data["grid_size"][1]

        for ri, n_regs in enumerate(reg_counts):
            data = all_data[n_regs]
            decoded = data["decoded"]
            n_show = min(len(decoded), n_chars, len(data["best_maps"]))

            # Column 0: original image
            axes[ri, 0].imshow(img_crop, cmap="gray", aspect="auto")
            axes[ri, 0].set_ylabel(f"{n_regs} reg", fontsize=9, fontweight="bold",
                                   rotation=0, labelpad=30, va="center")
            axes[ri, 0].set_xticks([])
            axes[ri, 0].set_yticks([])
            if ri == 0:
                axes[ri, 0].set_title("Input", fontsize=9, fontweight="bold")

            for ci in range(n_show):
                ax = axes[ri, ci + 1]
                bmap = data["best_maps"].get(ci, np.zeros(Wp))
                best_h = data["best_heads"].get(ci, -1)

                attn_2d, _ = attn_to_image_2d(bmap, Wp, text_bbox)

                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.25)
                ax.imshow(attn_2d, cmap="inferno", aspect="auto", alpha=0.75,
                          vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

                if ri == 0:
                    ch = decoded[ci] if ci < len(decoded) else "?"
                    display = ch if ch != " " else "SPC"
                    ax.set_title(f"'{display}'", fontsize=9, fontweight="bold",
                                 color="darkblue")

                # Annotate head index in corner
                ax.text(0.95, 0.05, f"H{best_h}", transform=ax.transAxes,
                        fontsize=6, color="white", ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

            # Blank remaining columns
            for ci in range(n_show, n_chars):
                axes[ri, ci + 1].axis("off")

        fig.suptitle(
            f"Best-Head Character Attention — \"{common_decoded}\"\n"
            f"Each cell: most localized head (annotated H#), no averaging",
            fontsize=11, fontweight="bold", y=1.04,
        )

        out = os.path.join(output_dir, f"figA_best_head_{img_name}")
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        fig.savefig(f"{out}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}.pdf")


# ============================================================================
# Figure B: 1D Attention Profile Comparison
# ============================================================================

def plot_attention_profiles(image_paths, models_dict, charset, device, output_dir,
                            max_chars=8):
    """
    For each image: one subplot per character.
    Each subplot: 1D attention curve (x=patch position, y=attention weight),
    with one line per register count.

    This is the most HONEST representation of CTC self-attention.
    Directly shows how register count affects the attention shape.
    """
    os.makedirs(output_dir, exist_ok=True)

    reg_counts = sorted(models_dict.keys())

    for img_path in image_paths:
        img_name = Path(img_path).stem
        img_tensor, img_np, text_bbox, _ = prepare_image(img_path, device)
        x_start, x_end = text_bbox

        # Collect per-register data
        all_data = {}
        ref_decoded = None
        for n_regs in reg_counts:
            model = models_dict[n_regs]
            result = run_explain(model, img_tensor)
            decoded, peak_cols = ctc_greedy_decode(result["logits"], charset)[0]
            if ref_decoded is None:
                ref_decoded = decoded

            per_head, best_heads, best_maps, avg_maps = extract_per_head_attention(
                result["attn_maps"], n_regs, peak_cols, result["grid_size"]
            )
            all_data[n_regs] = {
                "decoded": decoded,
                "peak_cols": peak_cols,
                "best_maps": best_maps,
                "avg_maps": avg_maps,
                "per_head": per_head,
                "best_heads": best_heads,
                "grid_size": result["grid_size"],
            }

        n_chars = min(len(ref_decoded), max_chars)
        Wp = all_data[reg_counts[0]]["grid_size"][1]

        # Layout: 2 rows x n_chars columns
        # Row 0: best-head profiles, Row 1: head-averaged profiles
        fig, axes = plt.subplots(
            2, n_chars,
            figsize=(2.5 * n_chars, 5),
            gridspec_kw={"wspace": 0.25, "hspace": 0.45},
        )
        if n_chars == 1:
            axes = axes[:, np.newaxis]

        for ci in range(n_chars):
            ch = ref_decoded[ci] if ci < len(ref_decoded) else "?"
            display = ch if ch != " " else "SPC"

            for n_regs in reg_counts:
                data = all_data[n_regs]
                color = REG_COLORS[n_regs]
                label = REG_LABELS[n_regs]

                # Best-head profile
                if ci in data["best_maps"]:
                    bmap = data["best_maps"][ci]
                    bh = data["best_heads"][ci]
                    attn_crop, _ = attn_to_image_1d(bmap, Wp, text_bbox)
                    axes[0, ci].plot(attn_crop, color=color, linewidth=1.2,
                                    label=f"{label} (H{bh})", alpha=0.85)

                # Head-averaged profile
                if ci in data["avg_maps"]:
                    amap = data["avg_maps"][ci]
                    attn_crop, _ = attn_to_image_1d(amap, Wp, text_bbox)
                    axes[1, ci].plot(attn_crop, color=color, linewidth=1.2,
                                    label=label, alpha=0.85)

            # Mark CTC peak position
            ref_cols = all_data[reg_counts[0]]["peak_cols"]
            if ci < len(ref_cols):
                peak_px = ref_cols[ci] / Wp * IMAGE_W - x_start
                peak_px = np.clip(peak_px, 0, text_bbox[1] - text_bbox[0] - 1)
                for row in range(2):
                    axes[row, ci].axvline(x=peak_px, color="gray", linewidth=0.8,
                                         linestyle=":", alpha=0.6)

            axes[0, ci].set_title(f"'{display}'", fontsize=10, fontweight="bold",
                                  color="darkblue")
            axes[0, ci].set_ylim(-0.05, 1.05)
            axes[1, ci].set_ylim(-0.05, 1.05)
            axes[0, ci].set_xticks([])
            axes[1, ci].set_xlabel("pixel pos", fontsize=7)

            if ci == 0:
                axes[0, ci].set_ylabel("Best head", fontsize=9)
                axes[1, ci].set_ylabel("Head avg", fontsize=9)

        # Single legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles[:len(reg_counts)], [REG_LABELS[r] for r in reg_counts],
                   loc="lower center", ncol=len(reg_counts), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f"1D Attention Profiles — \"{ref_decoded}\"\n"
            f"Top: best (most localized) head per char  |  Bottom: head average",
            fontsize=11, fontweight="bold", y=1.04,
        )

        out = os.path.join(output_dir, f"figB_attn_profiles_{img_name}")
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        fig.savefig(f"{out}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}.pdf")


# ============================================================================
# Figure C: Register Absorption Mechanism
# ============================================================================

def plot_register_absorption(image_paths, models_dict, charset, device, output_dir):
    """
    Shows what register tokens attend to (reg→patch attention).
    Demonstrates that registers absorb global/background information,
    freeing patch tokens for local character-specific attention.

    Layout: one column per register count (4, 8, 16).
    Row 0: input image with patch-to-register attention overlay
           (how much each patch position "leaks" attention to registers)
    Row 1: register-to-patch heatmap (what each register attends to)
    """
    os.makedirs(output_dir, exist_ok=True)

    reg_counts_with_regs = [r for r in sorted(models_dict.keys()) if r > 0]
    if not reg_counts_with_regs:
        print("  No register models to visualize absorption")
        return

    for img_path in image_paths:
        img_name = Path(img_path).stem
        img_tensor, img_np, text_bbox, _ = prepare_image(img_path, device)
        x_start, x_end = text_bbox
        img_crop = img_np[:, x_start:x_end]

        n_cols = len(reg_counts_with_regs) + 1  # +1 for baseline

        fig, axes = plt.subplots(
            2, n_cols,
            figsize=(3.5 * n_cols, 4.5),
            gridspec_kw={"wspace": 0.15, "hspace": 0.35},
        )

        # Column 0: baseline (0 registers)
        if 0 in models_dict:
            model = models_dict[0]
            result = run_explain(model, img_tensor)
            decoded, _ = ctc_greedy_decode(result["logits"], charset)[0]
            Wp = result["grid_size"][1]

            # Show input image
            axes[0, 0].imshow(img_crop, cmap="gray", aspect="auto")
            axes[0, 0].set_title("0 reg (baseline)", fontsize=9, fontweight="bold")
            axes[0, 0].set_ylabel("Patch→Reg\nattn", fontsize=8)
            axes[0, 0].text(0.5, 0.5, "No registers\n(no absorption)",
                           transform=axes[0, 0].transAxes, ha="center", va="center",
                           fontsize=9, color="red", fontweight="bold",
                           bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])

            axes[1, 0].text(0.5, 0.5, "N/A",
                           transform=axes[1, 0].transAxes, ha="center", va="center",
                           fontsize=12, color="gray")
            axes[1, 0].set_ylabel("Reg→Patch\nheatmap", fontsize=8)
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])

        for ci, n_regs in enumerate(reg_counts_with_regs, start=1):
            model = models_dict[n_regs]
            result = run_explain(model, img_tensor)
            decoded, _ = ctc_greedy_decode(result["logits"], charset)[0]
            Wp = result["grid_size"][1]

            reg_to_patch, patch_to_reg = extract_register_attention(
                result["attn_maps"], n_regs, result["grid_size"]
            )

            # Row 0: patch→register attention overlay on image
            p2r_norm = (patch_to_reg - patch_to_reg.min()) / (patch_to_reg.max() - patch_to_reg.min() + 1e-8)
            p2r_2d, _ = attn_to_image_2d(p2r_norm, Wp, text_bbox)

            axes[0, ci].imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)
            axes[0, ci].imshow(p2r_2d, cmap="Reds", aspect="auto", alpha=0.7,
                               vmin=0, vmax=1)
            axes[0, ci].set_title(f"{n_regs} reg", fontsize=9, fontweight="bold")
            axes[0, ci].set_xticks([])
            axes[0, ci].set_yticks([])

            # Row 1: register→patch heatmap
            # Normalize each register's attention independently for visibility
            r2p_show = np.zeros_like(reg_to_patch)
            for r in range(n_regs):
                row = reg_to_patch[r]
                r2p_show[r] = (row - row.min()) / (row.max() - row.min() + 1e-8)

            # Crop to text region
            patch_per_px = Wp / IMAGE_W
            p_start = int(x_start * patch_per_px)
            p_end = int(x_end * patch_per_px)
            p_end = min(p_end, Wp)
            r2p_crop = r2p_show[:, p_start:p_end]

            im = axes[1, ci].imshow(r2p_crop, cmap="YlOrRd", aspect="auto",
                                     vmin=0, vmax=1)
            axes[1, ci].set_xlabel("patch position", fontsize=7)
            axes[1, ci].set_ylabel("reg idx" if ci == 1 else "", fontsize=7)
            axes[1, ci].set_yticks(range(0, n_regs, max(1, n_regs // 4)))

        fig.suptitle(
            f"Register Absorption Mechanism — \"{decoded}\"\n"
            f"Top: where patches send attention to registers (red=high)  |  "
            f"Bottom: what each register attends to",
            fontsize=10, fontweight="bold", y=1.05,
        )

        out = os.path.join(output_dir, f"figC_reg_absorption_{img_name}")
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        fig.savefig(f"{out}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}.pdf")


# ============================================================================
# Figure D: Head-Averaged vs Best-Head Side-by-Side (single register count)
# ============================================================================

def plot_avg_vs_best_comparison(image_paths, models_dict, charset, device,
                                 output_dir, target_reg=16):
    """
    Side-by-side: same characters, head-averaged vs best-head.
    Shows WHY head selection matters.

    Layout: 2 rows (avg, best) × N columns (characters)
    """
    os.makedirs(output_dir, exist_ok=True)

    if target_reg not in models_dict:
        target_reg = max(models_dict.keys())
    model = models_dict[target_reg]

    for img_path in image_paths:
        img_name = Path(img_path).stem
        img_tensor, img_np, text_bbox, _ = prepare_image(img_path, device)
        x_start, x_end = text_bbox
        img_crop = img_np[:, x_start:x_end]

        result = run_explain(model, img_tensor)
        decoded, peak_cols = ctc_greedy_decode(result["logits"], charset)[0]
        Wp = result["grid_size"][1]

        per_head, best_heads, best_maps, avg_maps = extract_per_head_attention(
            result["attn_maps"], target_reg, peak_cols, result["grid_size"]
        )

        n_chars = min(len(decoded), 10, len(best_maps))
        n_cols = n_chars + 1

        fig, axes = plt.subplots(
            2, n_cols,
            figsize=(1.6 * n_cols, 3.6),
            gridspec_kw={"wspace": 0.04, "hspace": 0.30},
        )

        row_labels = ["Head avg", "Best head"]

        for ri, (maps_dict, label) in enumerate([(avg_maps, "Head avg"),
                                                   (best_maps, "Best head")]):
            axes[ri, 0].imshow(img_crop, cmap="gray", aspect="auto")
            axes[ri, 0].set_ylabel(label, fontsize=9, fontweight="bold",
                                   rotation=0, labelpad=40, va="center")
            axes[ri, 0].set_xticks([])
            axes[ri, 0].set_yticks([])
            if ri == 0:
                axes[ri, 0].set_title("Input", fontsize=9, fontweight="bold")

            for ci in range(n_chars):
                ax = axes[ri, ci + 1]
                amap = maps_dict.get(ci, np.zeros(Wp))
                attn_2d, _ = attn_to_image_2d(amap, Wp, text_bbox)

                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.25)
                ax.imshow(attn_2d, cmap="inferno", aspect="auto", alpha=0.75,
                          vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

                if ri == 0:
                    ch = decoded[ci] if ci < len(decoded) else "?"
                    display = ch if ch != " " else "SPC"
                    ax.set_title(f"'{display}'", fontsize=9, fontweight="bold",
                                 color="darkblue")

                if ri == 1 and ci in best_heads:
                    ax.text(0.95, 0.05, f"H{best_heads[ci]}", transform=ax.transAxes,
                            fontsize=6, color="white", ha="right", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

        fig.suptitle(
            f"Head-Average vs Best-Head — \"{decoded}\" ({target_reg} reg)\n"
            f"Head averaging dilutes localized attention patterns",
            fontsize=11, fontweight="bold", y=1.05,
        )

        out = os.path.join(output_dir, f"figD_avg_vs_best_{img_name}")
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        fig.savefig(f"{out}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}.pdf")


# ============================================================================
# Figure E: Per-Head Heatmap Matrix
# ============================================================================

def plot_per_head_matrix(image_paths, models_dict, charset, device, output_dir):
    """
    Full 8-head × N-character matrix for baseline vs 16-reg.
    Shows which heads specialize for which characters, and how registers change this.

    Layout: 8 rows (heads) × N cols (characters), side by side for 0 and 16 reg.
    """
    os.makedirs(output_dir, exist_ok=True)

    reg_pair = [0, max(r for r in models_dict.keys() if r > 0)] if len(models_dict) > 1 else list(models_dict.keys())

    for img_path in image_paths:
        img_name = Path(img_path).stem
        img_tensor, img_np, text_bbox, _ = prepare_image(img_path, device)
        x_start, x_end = text_bbox
        img_crop = img_np[:, x_start:x_end]

        all_data = {}
        for n_regs in reg_pair:
            model = models_dict[n_regs]
            result = run_explain(model, img_tensor)
            decoded, peak_cols = ctc_greedy_decode(result["logits"], charset)[0]
            per_head, best_heads, _, _ = extract_per_head_attention(
                result["attn_maps"], n_regs, peak_cols, result["grid_size"]
            )
            all_data[n_regs] = {
                "decoded": decoded, "per_head": per_head,
                "best_heads": best_heads, "grid_size": result["grid_size"],
            }

        ref = all_data[reg_pair[0]]
        n_chars = min(len(ref["decoded"]), 8)
        n_heads = 8
        Wp = ref["grid_size"][1]

        fig, axes = plt.subplots(
            n_heads, n_chars * len(reg_pair),
            figsize=(1.3 * n_chars * len(reg_pair), 1.0 * n_heads),
            gridspec_kw={"wspace": 0.05, "hspace": 0.10},
        )

        for mi, n_regs in enumerate(reg_pair):
            data = all_data[n_regs]
            decoded = data["decoded"]

            for hi in range(n_heads):
                for ci in range(n_chars):
                    col = mi * n_chars + ci
                    ax = axes[hi, col]

                    if ci in data["per_head"]:
                        heads = data["per_head"][ci]  # [H, Wp]
                        row = heads[hi]
                        row_norm = (row - row.min()) / (row.max() - row.min() + 1e-8)
                        attn_2d, _ = attn_to_image_2d(row_norm, Wp, text_bbox)

                        ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.2)
                        ax.imshow(attn_2d, cmap="inferno", aspect="auto", alpha=0.8,
                                  vmin=0, vmax=1)

                        # Highlight if this is the best head
                        if data["best_heads"].get(ci) == hi:
                            for spine in ax.spines.values():
                                spine.set_edgecolor("lime")
                                spine.set_linewidth(2)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    if hi == 0:
                        ch = decoded[ci] if ci < len(decoded) else "?"
                        display = ch if ch != " " else "SP"
                        title = f"'{display}'" if mi == 0 else f"'{display}'"
                        ax.set_title(title, fontsize=7, pad=2)

                    if ci == 0 and mi == 0:
                        ax.set_ylabel(f"H{hi}", fontsize=7, rotation=0, labelpad=12)

            # Group labels
            mid_col = mi * n_chars + n_chars // 2
            axes[0, mid_col].text(0.5, 1.35, f"{n_regs} reg",
                                   transform=axes[0, mid_col].transAxes,
                                   fontsize=11, fontweight="bold", ha="center",
                                   color=REG_COLORS.get(n_regs, "black"))

        fig.suptitle(
            f"Per-Head Attention Matrix — \"{ref['decoded']}\"\n"
            f"Green border = most localized head for that character",
            fontsize=11, fontweight="bold", y=1.06,
        )

        out = os.path.join(output_dir, f"figE_per_head_matrix_{img_name}")
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        fig.savefig(f"{out}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}.pdf")


# ============================================================================
# Quantitative Summary
# ============================================================================

def compute_and_print_metrics(image_paths, models_dict, charset, device, output_dir):
    """
    Compute per-head and best-head metrics for all register counts.
    Print table + save CSV.
    """
    import csv
    from scipy.stats import spearmanr

    os.makedirs(output_dir, exist_ok=True)
    reg_counts = sorted(models_dict.keys())

    rows = []
    summary = {r: {"best_entropy": [], "avg_entropy": [],
                    "best_loc": [], "avg_loc": [],
                    "best_xmax_rho": [], "avg_xmax_rho": []} for r in reg_counts}

    for img_path in image_paths:
        img_name = Path(img_path).stem
        img_tensor, _, _, _ = prepare_image(img_path, device)

        for n_regs in reg_counts:
            model = models_dict[n_regs]
            result = run_explain(model, img_tensor)
            decoded, peak_cols = ctc_greedy_decode(result["logits"], charset)[0]

            per_head, best_heads, best_maps, avg_maps = extract_per_head_attention(
                result["attn_maps"], n_regs, peak_cols, result["grid_size"]
            )

            Wp = result["grid_size"][1]

            # Compute metrics for best-head and avg
            for mode, maps_dict in [("best", best_maps), ("avg", avg_maps)]:
                entropies = []
                localizations = []
                xmax_positions = []

                for ci, amap in maps_dict.items():
                    # Entropy on raw (pre-normalized) attention
                    if mode == "best" and ci in per_head:
                        bh = best_heads[ci]
                        raw = per_head[ci][bh]
                    else:
                        raw = per_head[ci].mean(axis=0) if ci in per_head else amap
                    # Normalize to distribution
                    raw_dist = raw / (raw.sum() + 1e-10)
                    ent = -np.sum(raw_dist * np.log2(raw_dist + 1e-10))
                    entropies.append(ent)

                    loc = amap.max() / (amap.mean() + 1e-8)
                    localizations.append(loc)

                    positions = np.arange(Wp, dtype=np.float64)
                    w = raw_dist.sum()
                    xmax = float(np.sum(positions * raw_dist) / w) if w > 0 else 0
                    xmax_positions.append(xmax)

                if len(xmax_positions) > 2:
                    rho, _ = spearmanr(range(len(xmax_positions)), xmax_positions)
                else:
                    rho = 0.0

                avg_ent = np.mean(entropies)
                avg_loc = np.mean(localizations)

                summary[n_regs][f"{mode}_entropy"].append(avg_ent)
                summary[n_regs][f"{mode}_loc"].append(avg_loc)
                summary[n_regs][f"{mode}_xmax_rho"].append(rho)

                rows.append({
                    "image": img_name, "registers": n_regs, "mode": mode,
                    "entropy": f"{avg_ent:.4f}", "localization": f"{avg_loc:.2f}",
                    "xmax_rho": f"{rho:.4f}", "decoded": decoded,
                })

    # Print summary
    print("\n" + "=" * 75)
    print("METRICS SUMMARY (averaged over images)")
    print("=" * 75)
    print(f"{'Regs':>5}  {'Mode':<6}  {'Entropy↓':>10}  {'Localiz↑':>10}  {'Xmax ρ↑':>10}")
    print("-" * 55)
    for n_regs in reg_counts:
        for mode in ["avg", "best"]:
            e = np.mean(summary[n_regs][f"{mode}_entropy"])
            l = np.mean(summary[n_regs][f"{mode}_loc"])
            r = np.mean(summary[n_regs][f"{mode}_xmax_rho"])
            marker = " ←" if mode == "best" else ""
            print(f"{n_regs:>5}  {mode:<6}  {e:>10.3f}  {l:>10.2f}  {r:>10.4f}{marker}")
    print("=" * 75)

    # Save CSV
    csv_path = os.path.join(output_dir, "register_metrics_v2.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {csv_path}")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Register Effect Visualization")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--figures", nargs="+",
                        default=["A", "B", "C", "D", "E", "metrics"],
                        help="Which figures to generate: A B C D E metrics")
    args = parser.parse_args()

    # Load all models
    print("Loading models...")
    charset = load_charset()
    models_dict = {}
    for run_id, n_regs in RUNS.items():
        model, _, n_regs_actual = load_model(run_id, args.device)
        models_dict[n_regs] = model
        print(f"  {run_id}: {n_regs} registers")

    # Image paths
    if args.images:
        image_paths = args.images
    else:
        image_paths = [
            str(SAMPLE_DIR / "a01-038-12.png"),
            str(SAMPLE_DIR / "a06-095-10.png"),
        ]

    out_dir = str(OUTPUT_DIR)
    figs = [f.upper() for f in args.figures]

    print(f"\nGenerating figures: {', '.join(figs)}")
    print(f"Images: {[Path(p).stem for p in image_paths]}\n")

    if "A" in figs:
        print("--- Fig A: Best-Head Character Attention Grid ---")
        plot_best_head_grid(image_paths, models_dict, charset, args.device, out_dir)

    if "B" in figs:
        print("\n--- Fig B: 1D Attention Profiles ---")
        plot_attention_profiles(image_paths, models_dict, charset, args.device, out_dir)

    if "C" in figs:
        print("\n--- Fig C: Register Absorption ---")
        plot_register_absorption(image_paths, models_dict, charset, args.device, out_dir)

    if "D" in figs:
        print("\n--- Fig D: Head-Average vs Best-Head ---")
        plot_avg_vs_best_comparison(image_paths, models_dict, charset, args.device, out_dir)

    if "E" in figs:
        print("\n--- Fig E: Per-Head Matrix ---")
        plot_per_head_matrix(image_paths, models_dict, charset, args.device, out_dir)

    if "METRICS" in figs:
        print("\n--- Quantitative Metrics ---")
        compute_and_print_metrics(image_paths, models_dict, charset, args.device, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
