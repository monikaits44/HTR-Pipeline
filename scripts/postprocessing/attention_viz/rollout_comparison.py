#!/usr/bin/env python3
"""
Attention Rollout Comparison Across Register Counts
====================================================
Computes attention rollout (Abnar & Zuidema 2020) for each model in the 
register sweep and produces a comparison figure.

Attention rollout multiplies attention matrices across layers to reveal
the TRUE effective attention from input patches to final representations.

Output: side-by-side rollout heatmaps overlaid on input image for
        Reg-0, Reg-4, Reg-8, Reg-16 (or user-selected runs).

Usage:
    python scripts/postprocessing/attention_viz/rollout_comparison.py \\
        --image notebook/sample_images/a01-038-12.png \\
        --runs 55 58 63 71 \\
        --device cpu \\
        --save-dir visualizations/rollout_comparison
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.postprocessing.attention_viz.core import (
    load_configs, load_model, extract_attention, ctc_decode,
    load_htr_image, attention_rollout, infer_num_registers,
    render_heatmap_on_image, get_char_attention,
)

RUN_MAP = {
    55: 0, 64: 1, 56: 2, 66: 3, 58: 4, 65: 5, 67: 6, 68: 7,
    63: 8, 69: 9, 61: 10, 57: 11, 59: 12, 60: 13, 62: 14, 70: 15, 71: 16,
}


def main():
    parser = argparse.ArgumentParser(description="Attention rollout comparison")
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=[55, 58, 63, 71],
                        help="Run numbers to compare (default: 55 58 63 71)")
    parser.add_argument("--runs-dir", default="saved_models/experiments")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="visualizations/rollout_comparison")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-chars", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=2.0)
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

    num_runs = len(args.runs)
    fig, axes = plt.subplots(num_runs, 2, figsize=(16, 3.5 * num_runs), dpi=args.dpi)
    if num_runs == 1:
        axes = axes[np.newaxis, :]

    for ri, run_num in enumerate(args.runs):
        n_reg = RUN_MAP.get(run_num, infer_num_registers(
            os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")))
        model_path = os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")

        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg
        net, classes, i2c = load_model(cfg_run, model_path, args.device)
        result = extract_attention(net, image_tensor, args.device)
        text, positions = ctc_decode(result["logits"], i2c)

        R = result["num_registers"]
        Hp, Wp = result["grid_size"]

        # ── Rollout ──────────────────────────────────────────────────────
        rollout = attention_rollout(result["attn_maps"], R)  # [S, S]

        # Mean rollout from all patch tokens to each patch token
        patch_rollout = rollout[R:, R:].mean(axis=0)  # [P]
        mn, mx = patch_rollout.min(), patch_rollout.max()
        if mx - mn > 1e-8:
            patch_rollout = (patch_rollout - mn) / (mx - mn)
        patch_rollout = np.power(patch_rollout, args.gamma)

        overlay = render_heatmap_on_image(patch_rollout, raw_image, Wp, alpha=args.alpha)

        axes[ri, 0].imshow(overlay)
        axes[ri, 0].set_title(f"Reg-{n_reg} — Attention Rollout", fontsize=11, fontweight="bold")
        axes[ri, 0].axis("off")

        # ── Per-character rollout attention ───────────────────────────────
        # Use rollout matrix row at each CTC position
        char_strips = []
        n = min(len(text), args.max_chars)
        for ci in range(n):
            pos = positions[ci]
            row = rollout[R + pos, R:]  # [P]
            mn, mx = row.min(), row.max()
            if mx - mn > 1e-8:
                row = (row - mn) / (mx - mn)
            row = np.power(row, args.gamma)
            char_strips.append(row)

        if char_strips:
            char_mat = np.stack(char_strips)  # [n_chars, P]
            axes[ri, 1].imshow(char_mat, aspect="auto", cmap="inferno", interpolation="nearest")
            axes[ri, 1].set_yticks(range(n))
            axes[ri, 1].set_yticklabels([repr(c)[1:-1] for c in text[:n]], fontsize=7, fontfamily="monospace")
            axes[ri, 1].set_xlabel("Patch position", fontsize=9)
            axes[ri, 1].set_title(f"Per-char rollout: '{text[:n]}'", fontsize=10)
        else:
            axes[ri, 1].text(0.5, 0.5, "No decode", ha="center", va="center",
                             transform=axes[ri, 1].transAxes)
            axes[ri, 1].axis("off")

        del net

    fig.suptitle("Attention Rollout Comparison Across Register Counts",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    img_stem = Path(args.image).stem
    out_path = os.path.join(args.save_dir, f"rollout_{img_stem}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
