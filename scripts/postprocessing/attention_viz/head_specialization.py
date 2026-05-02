#!/usr/bin/env python3
"""
Head Specialization Analysis
==============================
Compares how individual attention heads specialise across register counts.

Produces:
  1. Per-head character attention grids (one column per head, rows = characters)
  2. Head diversity metric: how different heads attend to different regions

Usage:
    python scripts/postprocessing/attention_viz/head_specialization.py \\
        --image notebook/sample_images/a01-038-12.png \\
        --runs 55 58 71 \\
        --device cpu \\
        --save-dir visualizations/head_specialization
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
    load_htr_image, get_char_attention, infer_num_registers,
    compute_entropy,
)

RUN_MAP = {
    55: 0, 64: 1, 56: 2, 66: 3, 58: 4, 65: 5, 67: 6, 68: 7,
    63: 8, 69: 9, 61: 10, 57: 11, 59: 12, 60: 13, 62: 14, 70: 15, 71: 16,
}


def main():
    parser = argparse.ArgumentParser(description="Head specialization analysis")
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=[55, 58, 71])
    parser.add_argument("--runs-dir", default="saved_models/experiments")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="visualizations/head_specialization")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--max-chars", type=int, default=12)
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

    for run_num in args.runs:
        n_reg = RUN_MAP.get(run_num, infer_num_registers(
            os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")))
        model_path = os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")

        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg
        net, classes, i2c = load_model(cfg_run, model_path, args.device)
        result = extract_attention(net, image_tensor, args.device)
        text, positions = ctc_decode(result["logits"], i2c)

        num_heads = result["attn_maps"][0].shape[1]
        n_chars = min(len(text), args.max_chars, len(positions))

        if n_chars == 0:
            del net
            continue

        # ── Per-head attention grids ────────────────────────────────────
        fig, axes = plt.subplots(
            num_heads, 1, figsize=(14, 1.8 * num_heads), dpi=args.dpi,
            sharex=True,
        )
        if num_heads == 1:
            axes = [axes]

        head_entropies = []

        for hi in range(num_heads):
            char_attn = get_char_attention(
                result["attn_maps"], positions[:n_chars], result["num_registers"],
                layer="last", head=hi, gamma=args.gamma,
            )  # [n_chars, P]

            axes[hi].imshow(char_attn, aspect="auto", cmap="inferno", interpolation="nearest")
            axes[hi].set_ylabel(f"Head {hi}", fontsize=9, fontweight="bold")
            axes[hi].set_yticks(range(n_chars))
            axes[hi].set_yticklabels(
                [repr(c)[1:-1] for c in text[:n_chars]],
                fontsize=7, fontfamily="monospace",
            )

            # Compute per-head entropy
            mean_ent = np.mean([compute_entropy(row) for row in char_attn])
            head_entropies.append(mean_ent)
            axes[hi].text(
                1.01, 0.5, f"H={mean_ent:.1f}",
                transform=axes[hi].transAxes, fontsize=7, va="center",
            )

        axes[-1].set_xlabel("Patch Position", fontsize=10)
        fig.suptitle(
            f"Head Specialization — Reg-{n_reg} (run_{run_num}) — '{text[:n_chars]}'",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 0.98, 0.96])

        out_path = os.path.join(
            args.save_dir,
            f"heads_run{run_num}_reg{n_reg}_{Path(args.image).stem}.png",
        )
        fig.savefig(out_path, bbox_inches="tight", dpi=args.dpi, facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")

        # ── Head diversity bar chart ────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=args.dpi)
        colors = plt.cm.tab10(np.linspace(0, 1, num_heads))
        ax2.bar(range(num_heads), head_entropies, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("Attention Head", fontsize=11)
        ax2.set_ylabel("Mean Entropy (bits)", fontsize=11)
        ax2.set_title(
            f"Head Diversity — Reg-{n_reg} (lower = more focused)",
            fontsize=12, fontweight="bold",
        )
        ax2.set_xticks(range(num_heads))
        ax2.grid(axis="y", alpha=0.3)

        out2 = os.path.join(
            args.save_dir,
            f"diversity_run{run_num}_reg{n_reg}_{Path(args.image).stem}.png",
        )
        fig2.savefig(out2, bbox_inches="tight", dpi=args.dpi, facecolor="white")
        plt.close(fig2)
        print(f"Saved: {out2}")

        del net


if __name__ == "__main__":
    main()
