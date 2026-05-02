#!/usr/bin/env python3
"""
Raw self-attention heatmap visualization for ViT-RGTS models.

Shows the full token×token attention matrices per layer and per head,
in the style of the classic "Attention Is All You Need" visualization.
Red dashed lines mark the register/patch boundary so the effect of
register tokens on attention patterns is immediately visible.

Outputs:
  per_model/   — one figure per model per layer (2×4 grid of 8 heads)
  comparison/  — head-averaged last-layer heatmaps side-by-side
  register_attention_fraction.png — bar chart of patch→register absorption

Usage:
    python scripts/postprocessing/visualize_attention_maps.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_55/model.pt \\
        --model-path saved_models/experiments/run_64/model.pt \\
        --image notebook/sample_images/a06-110-08.png \\
        --save-dir visualizations/attention_maps_run55_71
"""

import argparse, json, math, os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models import HTRNet
from utils.preprocessing import preprocess
from omegaconf import OmegaConf


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Raw self-attention heatmap visualization")
    p.add_argument("configs", nargs="+", help="YAML config files")
    p.add_argument("--model-path", action="append", dest="model_paths",
                   required=True, help="Path(s) to model checkpoints")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--save-dir", default="visualizations/attention_maps")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--layers", type=str, default="all",
                   choices=["all", "first", "middle", "last"],
                   help="Which layers to plot per-model")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def get_run_info(model_path):
    """Extract run name and register count from model path."""
    run_dir = Path(model_path).parent
    run_name = run_dir.name
    cfg_path = run_dir / "config.json"
    n_reg = 0
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Try nested arch.num_registers first, then top-level variants
        n_reg = (cfg.get("arch", {}).get("num_registers", 0)
                 or cfg.get("num_register_tokens", 0)
                 or cfg.get("register_tokens", 0))
    return run_name, n_reg


def build_model(configs, model_path, device):
    """Load model with correct register count."""
    cfg = OmegaConf.merge(*[OmegaConf.load(c) for c in configs])
    _, n_reg = get_run_info(model_path)

    OmegaConf.set_struct(cfg, False)
    cfg.arch.num_registers = n_reg

    # Number of classes from dataset
    dataset_folder = cfg.data.path
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    num_classes = len(classes) + 1

    from copy import deepcopy
    arch_cfg = deepcopy(cfg.arch)
    net = HTRNet(arch_cfg, num_classes)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    return net, cfg, n_reg


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, falling back to CPU.")

    # Load base config for image size
    base_cfg = OmegaConf.merge(*[OmegaConf.load(c) for c in args.configs])
    fixed_size = (base_cfg.preproc.image_height, base_cfg.preproc.image_width)

    # Load and preprocess image
    img_name = Path(args.image).stem
    from utils.preprocessing import load_image
    img_np = load_image(args.image)            # grayscale numpy, 0-1 range
    inp_np = preprocess(img_np, fixed_size, border_size=8)
    inp = torch.from_numpy(inp_np).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    x = inp.to(device)

    per_model_dir = os.path.join(args.save_dir, "per_model")
    comp_dir = os.path.join(args.save_dir, "comparison")
    os.makedirs(per_model_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    print("=" * 70)
    print("Raw Self-Attention Map Visualization")
    print("=" * 70)
    print(f"Image : {args.image}")
    print(f"Models: {len(args.model_paths)}")
    print(f"Output: {args.save_dir}")
    print()

    # Collect data for comparison
    all_data = []  # (label, n_reg, R, T, S, last_layer_attn)

    for mi, mp in enumerate(args.model_paths):
        run_name, n_reg = get_run_info(mp)
        label = f"Reg-{n_reg}"
        print(f"[{mi+1}/{len(args.model_paths)}] {label} ({run_name})")

        net, cfg, n_reg = build_model(args.configs, mp, device)

        with torch.no_grad():
            logits, reg_tokens, attn_maps, token_norms, (Hp, Wp) = \
                net.forward_explain(x)

        n_layers = len(attn_maps)
        n_heads = attn_maps[0].shape[1]
        S = attn_maps[0].shape[2]
        T = Wp   # patch tokens
        R = S - T  # register tokens

        print(f"  S={S} ({R} reg + {T} patch), {n_layers} layers, {n_heads} heads")

        # Select layers to plot
        if args.layers == "all":
            layers = list(range(n_layers))
        elif args.layers == "first":
            layers = [0]
        elif args.layers == "middle":
            layers = [n_layers // 2]
        else:
            layers = [n_layers - 1]

        # ── Per-model: 1 figure per layer, 2×4 grid of 8 heads ──────────
        for li in layers:
            attn = attn_maps[li][0].cpu().numpy()  # [H, S, S]

            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            fig.suptitle(
                f"{label} ({run_name}) — {img_name} — Layer {li+1}/{n_layers}\n"
                f"Self-Attention: {S}×{S}  ({R} registers + {T} patches)",
                fontsize=14, fontweight="bold")

            for h in range(n_heads):
                ax = axes[h // 4][h % 4]
                im = ax.imshow(attn[h], cmap="viridis", aspect="auto",
                               interpolation="nearest")
                ax.set_title(f"Head {h+1}", fontsize=11)

                # Register / patch boundary
                if R > 0:
                    ax.axhline(y=R - 0.5, color="red", lw=1.5,
                               linestyle="--", alpha=0.8)
                    ax.axvline(x=R - 0.5, color="red", lw=1.5,
                               linestyle="--", alpha=0.8)

                # Sparse tick labels
                step = max(1, S // 10)
                ticks = list(range(0, S, step))
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
                ax.set_yticklabels([str(t) for t in ticks], fontsize=7)

                if h % 4 == 0:
                    ax.set_ylabel("Query token", fontsize=9)
                if h // 4 == 1:
                    ax.set_xlabel("Key token", fontsize=9)

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if R > 0:
                fig.text(0.5, 0.01,
                         f"Red dashed line = register/patch boundary  "
                         f"(0–{R-1}: registers,  {R}–{S-1}: patches)",
                         ha="center", fontsize=10, color="red")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = f"{label}_{run_name}_layer{li+1}_{img_name}.png"
            fig.savefig(os.path.join(per_model_dir, fname),
                        dpi=args.dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        # Store last-layer head-averaged attention for comparison
        last_attn = attn_maps[-1][0].mean(dim=0).cpu().numpy()  # [S, S]
        all_data.append((label, n_reg, R, T, S, last_attn))

    # ── Comparison: head-averaged last-layer across register counts ──────
    print("\nGenerating comparison figure …")
    target_regs = [0, 1, 2, 4, 8, 12, 16]
    subset = [(l, nr, R, T, S, a) for (l, nr, R, T, S, a) in all_data
              if nr in target_regs]
    if not subset:
        subset = all_data

    n_sub = len(subset)
    ncols = min(4, n_sub)
    nrows = math.ceil(n_sub / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_2d(axes)
    if axes.ndim == 1:
        axes = axes[None, :]

    fig.suptitle(
        f"Self-Attention Comparison — Last Layer, Head-Averaged — {img_name}\n"
        f"Effect of Register Tokens on Attention Patterns",
        fontsize=16, fontweight="bold")

    for idx, (label, nr, R, T, S, attn) in enumerate(subset):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        im = ax.imshow(attn, cmap="viridis", aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"{label}  ({S} = {R} reg + {T} patch)",
                     fontsize=12, fontweight="bold")

        if R > 0:
            ax.axhline(y=R - 0.5, color="red", lw=2, ls="--", alpha=0.9)
            ax.axvline(x=R - 0.5, color="red", lw=2, ls="--", alpha=0.9)

        step = max(1, S // 8)
        ticks = list(range(0, S, step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
        ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
        ax.set_xlabel("Key token", fontsize=9)
        ax.set_ylabel("Query token", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_sub, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(comp_dir,
                             f"comparison_last_layer_{img_name}.png"),
                dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── Also: comparison for EACH layer ──────────────────────────────────
    # Re-run the subset models to get per-layer data
    print("Generating per-layer comparison figures …")
    for target_layer_idx in range(6):  # 6 layers
        fig, axes_l = plt.subplots(nrows, ncols,
                                   figsize=(7 * ncols, 6 * nrows))
        axes_l = np.atleast_2d(axes_l)
        if axes_l.ndim == 1:
            axes_l = axes_l[None, :]

        fig.suptitle(
            f"Self-Attention — Layer {target_layer_idx+1}/6, Head-Averaged — {img_name}",
            fontsize=16, fontweight="bold")

        plotted = 0
        for mi, mp in enumerate(args.model_paths):
            run_name, n_reg = get_run_info(mp)
            if n_reg not in target_regs:
                continue
            label = f"Reg-{n_reg}"

            net, cfg, _ = build_model(args.configs, mp, device)
            with torch.no_grad():
                _, _, attn_maps, _, _ = net.forward_explain(x)

            S = attn_maps[0].shape[2]
            T = 128
            R = S - T
            attn = attn_maps[target_layer_idx][0].mean(dim=0).cpu().numpy()

            r, c = divmod(plotted, ncols)
            ax = axes_l[r][c]
            im = ax.imshow(attn, cmap="viridis", aspect="auto",
                           interpolation="nearest")
            ax.set_title(f"{label} ({S}={R}r+{T}p)", fontsize=12,
                         fontweight="bold")
            if R > 0:
                ax.axhline(y=R - 0.5, color="red", lw=2, ls="--", alpha=0.9)
                ax.axvline(x=R - 0.5, color="red", lw=2, ls="--", alpha=0.9)
            step = max(1, S // 8)
            ticks = list(range(0, S, step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
            ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
            ax.set_xlabel("Key", fontsize=9)
            ax.set_ylabel("Query", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plotted += 1

        for idx in range(plotted, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes_l[r][c].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(comp_dir,
                                 f"comparison_layer{target_layer_idx+1}_{img_name}.png"),
                    dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Layer {target_layer_idx+1} done")

    # ── Register attention fraction bar chart ────────────────────────────
    if any(nr > 0 for _, nr, _, _, _, _ in all_data):
        print("Generating register absorption bar chart …")
        fig, ax = plt.subplots(figsize=(12, 5))
        reg_counts, fracs, labels_list = [], [], []
        for label, nr, R, T, S, attn in all_data:
            if R > 0:
                p2r = attn[R:, :R].mean()
                p2p = attn[R:, R:].mean()
                frac = p2r / (p2r + p2p) * 100 if (p2r + p2p) > 0 else 0
            else:
                frac = 0.0
            reg_counts.append(nr)
            fracs.append(frac)
            labels_list.append(label)

        bars = ax.bar(range(len(fracs)), fracs,
                      color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Patch → Register Attention (%)", fontsize=11)
        ax.set_xlabel("Model (register count)", fontsize=11)
        ax.set_title(
            f"Register Attention Absorption — Last Layer, Head-Averaged — {img_name}",
            fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Value labels on bars
        for bar, v in zip(bars, fracs):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fig.savefig(os.path.join(args.save_dir,
                                 f"register_attention_fraction_{img_name}.png"),
                    dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"\nDone! All figures → {args.save_dir}/")


if __name__ == "__main__":
    main()
