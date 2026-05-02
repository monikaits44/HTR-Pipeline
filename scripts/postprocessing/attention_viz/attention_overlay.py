#!/usr/bin/env python3
"""
Attention Overlay Visualization
================================
Generates transparent heatmap overlays on handwriting images showing
where the model attends when decoding each character.

Two modes:
  1. overlay   — Heatmap overlaid on original image (saliency-style)
  2. sidebyside — Original + heatmap + carpet plot in one figure

Supports multiple models for side-by-side register comparison.

Usage:
    python scripts/postprocessing/attention_viz/attention_overlay.py \\
        --image notebook/sample_images/a01-038-12.png \\
        --runs 55 58 71 \\
        --mode overlay \\
        --device cpu \\
        --save-dir visualizations/attention_overlay
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.postprocessing.attention_viz.core import (
    load_configs, load_model, extract_attention, ctc_decode,
    load_htr_image, get_char_attention, render_heatmap_on_image,
    infer_num_registers,
)

RUN_MAP = {
    55: 0, 64: 1, 56: 2, 66: 3, 58: 4, 65: 5, 67: 6, 68: 7,
    63: 8, 69: 9, 61: 10, 57: 11, 59: 12, 60: 13, 62: 14, 70: 15, 71: 16,
}


def overlay_mode(args, cfg, image_tensor, raw_image):
    """Generate per-character transparent overlays on the original image."""
    for run_num in args.runs:
        n_reg = RUN_MAP.get(run_num, 0)
        model_path = os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")

        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg
        net, classes, i2c = load_model(cfg_run, model_path, args.device)
        result = extract_attention(net, image_tensor, args.device)
        text, positions = ctc_decode(result["logits"], i2c)

        Hp, Wp = result["grid_size"]
        n_chars = min(len(text), args.max_chars, len(positions))

        if n_chars == 0:
            del net
            continue

        char_attn = get_char_attention(
            result["attn_maps"], positions[:n_chars], result["num_registers"],
            layer="last", gamma=args.gamma,
        )

        # One figure: original on top, then per-char overlays
        cols = min(n_chars, 6)
        rows = (n_chars + cols - 1) // cols + 1  # +1 for original
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows), dpi=args.dpi)
        axes = np.atleast_2d(axes)

        # Original image spanning first row
        for c in range(cols):
            if c == 0:
                axes[0, c].imshow(1.0 - raw_image, cmap="gray", aspect="auto")
                axes[0, c].set_title(f"Reg-{n_reg}: '{text[:n_chars]}'", fontsize=10, fontweight="bold")
            axes[0, c].axis("off")

        # Per-character overlays
        for ci in range(n_chars):
            r = 1 + ci // cols
            c_col = ci % cols
            if r < axes.shape[0] and c_col < axes.shape[1]:
                overlay = render_heatmap_on_image(
                    char_attn[ci], raw_image, Wp, alpha=args.alpha,
                )
                axes[r, c_col].imshow(overlay)
                axes[r, c_col].set_title(f"'{text[ci]}'", fontsize=9, fontfamily="monospace")
                axes[r, c_col].axis("off")

        # Hide unused subplots
        for r in range(axes.shape[0]):
            for c in range(axes.shape[1]):
                if r == 0 and c > 0:
                    axes[r, c].axis("off")

        plt.tight_layout()
        out = os.path.join(
            args.save_dir,
            f"overlay_run{run_num}_reg{n_reg}_{Path(args.image).stem}.png",
        )
        fig.savefig(out, bbox_inches="tight", dpi=args.dpi, facecolor="white")
        plt.close(fig)
        print(f"Saved: {out}")
        del net


def sidebyside_mode(args, cfg, image_tensor, raw_image):
    """Generate side-by-side comparison: original + carpet + mean attention."""
    num_runs = len(args.runs)
    fig, axes = plt.subplots(num_runs, 3, figsize=(18, 4 * num_runs), dpi=args.dpi)
    if num_runs == 1:
        axes = axes[np.newaxis, :]

    for ri, run_num in enumerate(args.runs):
        n_reg = RUN_MAP.get(run_num, 0)
        model_path = os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")

        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg
        net, classes, i2c = load_model(cfg_run, model_path, args.device)
        result = extract_attention(net, image_tensor, args.device)
        text, positions = ctc_decode(result["logits"], i2c)

        Hp, Wp = result["grid_size"]
        n_chars = min(len(text), args.max_chars, len(positions))

        # Col 0: Original image
        axes[ri, 0].imshow(1.0 - raw_image, cmap="gray", aspect="auto")
        axes[ri, 0].set_title(f"Reg-{n_reg} — '{text[:20]}'", fontsize=10, fontweight="bold")
        axes[ri, 0].axis("off")

        if n_chars == 0:
            axes[ri, 1].axis("off")
            axes[ri, 2].axis("off")
            del net
            continue

        char_attn = get_char_attention(
            result["attn_maps"], positions[:n_chars], result["num_registers"],
            layer="last", gamma=args.gamma,
        )

        # Col 1: Character-patch carpet (alignment matrix)
        axes[ri, 1].imshow(char_attn, aspect="auto", cmap="inferno", interpolation="nearest")
        axes[ri, 1].set_yticks(range(n_chars))
        axes[ri, 1].set_yticklabels(
            [repr(c)[1:-1] for c in text[:n_chars]], fontsize=7, fontfamily="monospace",
        )
        axes[ri, 1].set_xlabel("Patch", fontsize=9)
        axes[ri, 1].set_title("Alignment Carpet", fontsize=10)

        # Col 2: Mean attention overlay
        mean_attn = char_attn.mean(axis=0)
        mn, mx = mean_attn.min(), mean_attn.max()
        if mx - mn > 1e-8:
            mean_attn = (mean_attn - mn) / (mx - mn)
        overlay = render_heatmap_on_image(mean_attn, raw_image, Wp, alpha=args.alpha)
        axes[ri, 2].imshow(overlay)
        axes[ri, 2].set_title("Mean Attention", fontsize=10)
        axes[ri, 2].axis("off")

        del net

    fig.suptitle("Attention Analysis — Register Comparison",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(
        args.save_dir,
        f"sidebyside_{Path(args.image).stem}.png",
    )
    fig.savefig(out, bbox_inches="tight", dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Attention overlay visualization")
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=[55, 58, 71])
    parser.add_argument("--runs-dir", default="saved_models/experiments")
    parser.add_argument("--mode", choices=["overlay", "sidebyside"], default="sidebyside")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="visualizations/attention_overlay")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--max-chars", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    project_root = Path(__file__).resolve().parents[3]
    cfg = load_configs(
        str(project_root / "configs" / "config.yaml"),
        str(project_root / "configs" / "baseline_vit_rgts_v2.yaml"),
    )
    cfg.device = args.device

    h = getattr(cfg.preproc, "image_height", 128)
    w = getattr(cfg.preproc, "image_width", 1024)
    image_tensor, raw_image = load_htr_image(args.image, h, w)

    if args.mode == "overlay":
        overlay_mode(args, cfg, image_tensor, raw_image)
    else:
        sidebyside_mode(args, cfg, image_tensor, raw_image)


if __name__ == "__main__":
    main()
