#!/usr/bin/env python3
"""
Layer-wise Attention Evolution Comparison
==========================================
Shows how character attention refines through transformer layers,
compared side-by-side across register counts.

Output: Grid where rows = layers (1→6), columns = characters,
        with one such grid per register configuration.

Usage:
    python scripts/postprocessing/attention_viz/layerwise_evolution.py \\
        --image notebook/sample_images/a01-038-12.png \\
        --runs 55 58 71 \\
        --device cpu \\
        --save-dir visualizations/layerwise_evolution
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
    compute_entropy, compute_diagonality,
)

RUN_MAP = {
    55: 0, 64: 1, 56: 2, 66: 3, 58: 4, 65: 5, 67: 6, 68: 7,
    63: 8, 69: 9, 61: 10, 57: 11, 59: 12, 60: 13, 62: 14, 70: 15, 71: 16,
}


def main():
    parser = argparse.ArgumentParser(description="Layer-wise attention evolution")
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=[55, 58, 71])
    parser.add_argument("--runs-dir", default="saved_models/experiments")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="visualizations/layerwise_evolution")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=2.0)
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

    num_runs = len(args.runs)

    # ── One figure per model, with layers as rows ────────────────────────
    for run_num in args.runs:
        n_reg = RUN_MAP.get(run_num, infer_num_registers(
            os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")))
        model_path = os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")

        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg
        net, classes, i2c = load_model(cfg_run, model_path, args.device)
        result = extract_attention(net, image_tensor, args.device)
        text, positions = ctc_decode(result["logits"], i2c)

        num_layers = len(result["attn_maps"])
        n_chars = min(len(text), args.max_chars, len(positions))

        if n_chars == 0:
            del net
            continue

        fig, axes = plt.subplots(
            num_layers, 1, figsize=(14, 2.0 * num_layers), dpi=args.dpi,
            sharex=True,
        )
        if num_layers == 1:
            axes = [axes]

        layer_diags = []
        layer_ents = []

        for li in range(num_layers):
            char_attn = get_char_attention(
                result["attn_maps"], positions[:n_chars], result["num_registers"],
                layer=str(li), gamma=args.gamma,
            )

            axes[li].imshow(char_attn, aspect="auto", cmap="inferno", interpolation="nearest")
            axes[li].set_ylabel(f"Layer {li+1}", fontsize=10, fontweight="bold")
            axes[li].set_yticks(range(n_chars))
            axes[li].set_yticklabels(
                [repr(c)[1:-1] for c in text[:n_chars]],
                fontsize=7, fontfamily="monospace",
            )

            # Metrics
            diag = compute_diagonality(positions[:n_chars], char_attn.shape[1])
            ent = np.mean([compute_entropy(row) for row in char_attn])
            layer_diags.append(diag)
            layer_ents.append(ent)

            axes[li].text(
                1.01, 0.5, f"ρ={diag:.2f}\nH={ent:.1f}",
                transform=axes[li].transAxes, fontsize=7, va="center",
            )

        axes[-1].set_xlabel("Patch Position", fontsize=10)
        fig.suptitle(
            f"Layer-wise Attention — Reg-{n_reg} (run_{run_num}) — '{text[:n_chars]}'",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 0.96, 0.96])

        out_path = os.path.join(
            args.save_dir,
            f"layers_run{run_num}_reg{n_reg}_{Path(args.image).stem}.png",
        )
        fig.savefig(out_path, bbox_inches="tight", dpi=args.dpi, facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")

        # ── Layer progression metrics plot ──────────────────────────────
        fig2, (ax_d, ax_e) = plt.subplots(1, 2, figsize=(12, 4), dpi=args.dpi)

        layers_x = list(range(1, num_layers + 1))
        ax_d.plot(layers_x, layer_diags, "o-", color="#4CAF50", linewidth=2)
        ax_d.set_xlabel("Layer", fontsize=11)
        ax_d.set_ylabel("Diagonality (ρ)", fontsize=11)
        ax_d.set_title(f"Diagonality Progression — Reg-{n_reg}", fontsize=12, fontweight="bold")
        ax_d.set_xticks(layers_x)
        ax_d.grid(True, alpha=0.3)

        ax_e.plot(layers_x, layer_ents, "s-", color="#9C27B0", linewidth=2)
        ax_e.set_xlabel("Layer", fontsize=11)
        ax_e.set_ylabel("Entropy (bits)", fontsize=11)
        ax_e.set_title(f"Entropy Progression — Reg-{n_reg}", fontsize=12, fontweight="bold")
        ax_e.set_xticks(layers_x)
        ax_e.grid(True, alpha=0.3)

        plt.tight_layout()
        out2 = os.path.join(
            args.save_dir,
            f"progression_run{run_num}_reg{n_reg}_{Path(args.image).stem}.png",
        )
        fig2.savefig(out2, bbox_inches="tight", dpi=args.dpi, facecolor="white")
        plt.close(fig2)
        print(f"Saved: {out2}")

        del net

    # ── Cross-model layer comparison ─────────────────────────────────────
    if len(args.runs) > 1:
        fig3, (ax3_d, ax3_e) = plt.subplots(1, 2, figsize=(14, 5), dpi=args.dpi)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_runs))

        for ri, run_num in enumerate(args.runs):
            n_reg = RUN_MAP.get(run_num, 0)
            model_path = os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")
            cfg_run = cfg.copy()
            cfg_run.arch.num_registers = n_reg

            try:
                net, classes, i2c = load_model(cfg_run, model_path, args.device)
            except Exception:
                continue
            result = extract_attention(net, image_tensor, args.device)
            text, positions = ctc_decode(result["logits"], i2c)
            n_chars = min(len(text), args.max_chars, len(positions))

            if n_chars < 2:
                del net
                continue

            num_layers = len(result["attn_maps"])
            diags, ents = [], []
            for li in range(num_layers):
                ca = get_char_attention(
                    result["attn_maps"], positions[:n_chars], result["num_registers"],
                    layer=str(li), gamma=1.0,
                )
                diags.append(compute_diagonality(positions[:n_chars], ca.shape[1]))
                ents.append(np.mean([compute_entropy(r) for r in ca]))

            layers_x = list(range(1, num_layers + 1))
            ax3_d.plot(layers_x, diags, "o-", color=colors[ri], label=f"Reg-{n_reg}", linewidth=2)
            ax3_e.plot(layers_x, ents, "s-", color=colors[ri], label=f"Reg-{n_reg}", linewidth=2)
            del net

        ax3_d.set_xlabel("Layer", fontsize=11)
        ax3_d.set_ylabel("Diagonality (ρ)", fontsize=11)
        ax3_d.set_title("Diagonality Across Layers", fontsize=12, fontweight="bold")
        ax3_d.legend(fontsize=9)
        ax3_d.grid(True, alpha=0.3)

        ax3_e.set_xlabel("Layer", fontsize=11)
        ax3_e.set_ylabel("Entropy (bits)", fontsize=11)
        ax3_e.set_title("Entropy Across Layers", fontsize=12, fontweight="bold")
        ax3_e.legend(fontsize=9)
        ax3_e.grid(True, alpha=0.3)

        plt.tight_layout()
        out3 = os.path.join(
            args.save_dir,
            f"cross_model_layers_{Path(args.image).stem}.png",
        )
        fig3.savefig(out3, bbox_inches="tight", dpi=args.dpi, facecolor="white")
        plt.close(fig3)
        print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
