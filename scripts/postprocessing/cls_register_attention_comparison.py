#!/usr/bin/env python3
"""
CLS & Register Token Attention Comparison across Register Sweep (run_55–71)
===========================================================================

For each model variant (0–16 register tokens), this script:
  1. Runs forward_explain on the input image
  2. Extracts the **middle-layer** attention map (layer index = depth//2)
  3. Visualises how the CLS-like token (register 0, or avg-patch if no regs)
     and each register token (reg0 … regN) attend over the patch tokens
  4. Overlays these heatmaps on the original image in a single comparison grid

Output
------
  visualizations/cls_register_attention_comparison/
    per_model/run_XX_reg<N>.png      – one figure per model
    comparison_grid.png              – grand summary grid across all models
    cls_vs_registers_summary.png     – side-by-side CLS vs mean-register heatmap

Usage
-----
  python scripts/postprocessing/cls_register_attention_comparison.py \
      --image /path/to/image.png \
      [--runs 55 56 57 ... 71] \
      [--device cuda:0]
"""

import os, sys, json, argparse, math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── project root on sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from models import HTRNet
from utils.preprocessing import load_image, preprocess

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_model(run_dir: Path, device: str):
    """Build HTRNet from run config, load weights, return (model, cfg, num_reg)."""
    config_path = run_dir / "config.json"
    model_path  = run_dir / "model.pt"

    with open(config_path) as f:
        run_cfg = json.load(f)

    # Build OmegaConf from the flat dict stored in config.json
    cfg = OmegaConf.create(run_cfg)
    OmegaConf.set_struct(cfg, False)

    num_reg = cfg.arch.get("num_registers", 0)

    # Load classes
    classes_path = PROJECT_ROOT / "data" / "IAM" / "processed_lines" / "classes.npy"
    if not classes_path.exists():
        classes_path = run_dir.parent / "classes.npy"
    classes = np.load(str(classes_path), allow_pickle=True)
    num_classes = len(classes) + 1  # +1 for CTC blank

    # Build & load
    net = HTRNet(cfg.arch, num_classes)
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    net.load_state_dict(ckpt, strict=True)
    net.to(device).eval()

    # Build i2c mapping
    i2c = {(i + 1): str(c) for i, c in enumerate(classes)}

    return net, cfg, num_reg, i2c


def prepare_image(image_path: str, cfg):
    """Load & preprocess image, return (tensor [1,1,H,W], raw_image)."""
    raw = load_image(image_path)
    h = cfg.preproc.image_height
    w = cfg.preproc.image_width
    processed = preprocess(raw, (h, w), border_size=8)
    tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
    return tensor, raw


def ctc_greedy_decode(logits_np, i2c, blank_id=0):
    """Greedy CTC decode → (text, [(char, time_step), ...])."""
    pred = logits_np.argmax(2).squeeze()  # [T]
    chars, positions = [], []
    prev = -1
    for t, v in enumerate(pred):
        v = int(v)
        if v != blank_id and v != prev:
            chars.append(i2c.get(v, "?"))
            positions.append(t)
        prev = v
    return "".join(chars), list(zip(chars, positions))


def token_attention_heatmap(attn_layer, token_idx, num_reg, num_patches, grid_h, grid_w):
    """
    Given a single-layer attention [H, S, S] (head-averaged → [S, S]),
    return the 1-D heatmap of *token_idx* attending to **patch tokens**.

    Parameters
    ----------
    attn_layer : np.ndarray [S, S]  (head-averaged attention of one layer)
    token_idx  : int   Which token's row to extract (0 = first reg / CLS, etc.)
    num_reg    : int   Number of register tokens prepended
    num_patches: int   Number of patch tokens
    grid_h, grid_w : spatial layout of patches (Hp, Wp)

    Returns
    -------
    heatmap_1d : np.ndarray [num_patches]
    """
    # Row = what token_idx attends to; columns num_reg: are patch tokens
    row = attn_layer[token_idx, num_reg: num_reg + num_patches]
    return row


# ═══════════════════════════════════════════════════════════════════════════
# Main visualisation routine
# ═══════════════════════════════════════════════════════════════════════════

def visualise_single_model(
    net, cfg, num_reg, i2c, image_tensor, raw_image, device,
    run_name, out_dir, image_path,
):
    """
    Produce a per-model figure:
      Row 0: original image + predicted text
      Row 1: CLS / reg-0 attention heatmap overlay
      Row 2–N: reg-1 … reg-(N-1) heatmap overlays

    Returns
    -------
    result_dict with keys: run_name, num_reg, predicted_text,
        cls_heatmap_1d, reg_heatmaps_1d (list), grid_size
    """
    x = image_tensor.to(device)

    # Forward explain
    logits, reg_tokens, attn_maps, token_norms, (Hp, Wp) = net.forward_explain(x)

    logits_np = logits.cpu().numpy()
    pred_text, char_positions = ctc_greedy_decode(logits_np, i2c)

    # Number of layers & pick middle
    num_layers = len(attn_maps)
    mid_layer_idx = num_layers // 2
    # attn_maps[layer] shape: [B, H, S, S]
    attn_mid = attn_maps[mid_layer_idx][0]  # [H, S, S]  (batch 0)
    attn_avg = attn_mid.mean(dim=0).numpy()  # [S, S]  head-averaged

    num_patches = Hp * Wp

    # ---- heatmaps for CLS (token 0, which is first register if regs>0) ----
    # With registers: token 0 = reg-0 acts as "CLS-like"
    # Without registers: we take the mean of all patch self-attention
    heatmaps = {}

    if num_reg > 0:
        # CLS = reg-0
        cls_heat = token_attention_heatmap(attn_avg, 0, num_reg, num_patches, Hp, Wp)
        heatmaps["CLS (reg-0)"] = cls_heat
        # reg-1 … reg-(N-1)
        for r in range(1, num_reg):
            h = token_attention_heatmap(attn_avg, r, num_reg, num_patches, Hp, Wp)
            heatmaps[f"reg-{r}"] = h
    else:
        # No registers → use mean patch self-attention as a "global" view
        patch_self_attn = attn_avg[num_reg:, num_reg:]  # [T, T]
        cls_heat = patch_self_attn.mean(axis=0)          # mean column attention
        heatmaps["Mean patch attn (no regs)"] = cls_heat

    # ---- Build figure ----
    n_rows = len(heatmaps) + 1  # +1 for original image row
    fig_height = max(3 * n_rows, 6)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, fig_height))
    if n_rows == 1:
        axes = [axes]

    # Prepare image for overlay — use preprocessed image (matches patch grid)
    proc_img = image_tensor[0, 0].numpy()  # [H, W]

    # Row 0: original image + text
    axes[0].imshow(1 - proc_img, cmap="gray", aspect="auto")
    axes[0].set_title(f"{run_name}  |  regs={num_reg}  |  pred: \"{pred_text}\"  |  layer {mid_layer_idx}/{num_layers}",
                      fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Overlay character positions as vertical lines
    if char_positions:
        h_img, w_img = proc_img.shape
        for ch, t in char_positions:
            x_pos = (t + 0.5) / num_patches * w_img
            axes[0].axvline(x_pos, color="cyan", alpha=0.3, linewidth=0.5)

    # Rows 1+: heatmap overlays
    for idx, (label, heat_1d) in enumerate(heatmaps.items()):
        ax = axes[idx + 1]
        # Show greyscale image
        ax.imshow(1 - proc_img, cmap="gray", aspect="auto", alpha=0.5)

        # Resize heatmap to image width
        h_img, w_img = proc_img.shape
        # heat_1d is [num_patches] – 1D horizontal for CNN-stem models (Hp=1)
        if Hp == 1:
            heat_2d = heat_1d.reshape(1, Wp)
        else:
            heat_2d = heat_1d.reshape(Hp, Wp)

        ax.imshow(
            heat_2d,
            cmap="inferno",
            aspect="auto",
            alpha=0.65,
            extent=[0, w_img, h_img, 0],
            interpolation="bilinear",
        )
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(out_dir / "per_model", exist_ok=True)
    save_path = out_dir / "per_model" / f"{run_name}_reg{num_reg}.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")

    return {
        "run_name": run_name,
        "num_reg": num_reg,
        "predicted_text": pred_text,
        "heatmaps": heatmaps,          # dict label → 1d array
        "grid_size": (Hp, Wp),
        "proc_img": proc_img,
        "mid_layer_idx": mid_layer_idx,
        "num_layers": num_layers,
        "char_positions": char_positions,
        "num_patches": num_patches,
    }


def build_comparison_grid(results, out_dir):
    """
    Grand comparison grid: columns = models (sorted by num_reg),
    rows = CLS, reg-0 … reg-max.

    For models with fewer registers, those cells are left blank.
    """
    results_sorted = sorted(results, key=lambda r: r["num_reg"])

    # Determine max number of registers across all models
    max_reg = max(r["num_reg"] for r in results_sorted)

    # Row labels: "Original Image", "CLS / reg-0", "reg-1", …, "reg-{max_reg-1}"
    row_labels = ["Input image", "CLS / reg-0"]
    for r in range(1, max_reg):
        row_labels.append(f"reg-{r}")

    n_cols = len(results_sorted)
    n_rows = len(row_labels)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )

    for col_idx, res in enumerate(results_sorted):
        proc_img = res["proc_img"]
        h_img, w_img = proc_img.shape
        Hp, Wp = res["grid_size"]
        num_reg = res["num_reg"]
        heatmaps = res["heatmaps"]
        pred_text = res["predicted_text"]

        # Column title
        axes[0, col_idx].set_title(
            f"reg={num_reg}\n\"{pred_text[:20]}{'…' if len(pred_text)>20 else ''}\"",
            fontsize=7, fontweight="bold",
        )

        # Row 0: input image
        axes[0, col_idx].imshow(1 - proc_img, cmap="gray", aspect="auto")
        axes[0, col_idx].axis("off")

        # Rows 1+: heatmaps
        heat_items = list(heatmaps.items())
        for row_idx in range(1, n_rows):
            ax = axes[row_idx, col_idx]
            heat_label_idx = row_idx - 1  # 0 → CLS, 1 → reg-1, etc.
            if heat_label_idx < len(heat_items):
                label, heat_1d = heat_items[heat_label_idx]
                ax.imshow(1 - proc_img, cmap="gray", aspect="auto", alpha=0.4)
                if Hp == 1:
                    heat_2d = heat_1d.reshape(1, Wp)
                else:
                    heat_2d = heat_1d.reshape(Hp, Wp)
                ax.imshow(
                    heat_2d, cmap="inferno", aspect="auto", alpha=0.7,
                    extent=[0, w_img, h_img, 0], interpolation="bilinear",
                )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=8, color="gray", transform=ax.transAxes)
            ax.axis("off")

        # Row labels (left column only)
        if col_idx == 0:
            for row_idx, lbl in enumerate(row_labels):
                axes[row_idx, 0].set_ylabel(lbl, fontsize=7, rotation=0,
                                             labelpad=60, va="center")

    plt.suptitle(
        "CLS & Register Token Attention — Middle Layer — across Register Sweep",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save_path = out_dir / "comparison_grid.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Comparison grid saved → {save_path}")


def build_cls_vs_mean_reg_summary(results, out_dir):
    """
    Side-by-side: CLS (reg-0) heatmap vs mean-of-all-registers heatmap
    for every model that has ≥1 register.
    """
    filtered = [r for r in results if r["num_reg"] >= 1]
    filtered = sorted(filtered, key=lambda r: r["num_reg"])

    if not filtered:
        print("  No models with registers — skipping CLS-vs-mean-reg summary.")
        return

    n_cols = len(filtered)
    n_rows = 3  # input | CLS (reg-0) | mean(reg-1…N)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.2 * n_rows), squeeze=False)

    for col_idx, res in enumerate(filtered):
        proc_img = res["proc_img"]
        h_img, w_img = proc_img.shape
        Hp, Wp = res["grid_size"]
        num_reg = res["num_reg"]
        heatmaps = res["heatmaps"]
        heat_items = list(heatmaps.items())

        axes[0, col_idx].set_title(f"reg={num_reg}", fontsize=8, fontweight="bold")

        # Row 0: image
        axes[0, col_idx].imshow(1 - proc_img, cmap="gray", aspect="auto")
        axes[0, col_idx].axis("off")

        # Row 1: CLS (reg-0) heatmap
        cls_heat = heat_items[0][1]
        axes[1, col_idx].imshow(1 - proc_img, cmap="gray", aspect="auto", alpha=0.4)
        h2d = cls_heat.reshape(1, Wp) if Hp == 1 else cls_heat.reshape(Hp, Wp)
        axes[1, col_idx].imshow(h2d, cmap="inferno", aspect="auto", alpha=0.7,
                                extent=[0, w_img, h_img, 0], interpolation="bilinear")
        axes[1, col_idx].axis("off")

        # Row 2: mean of reg-1…reg-N
        if len(heat_items) > 1:
            mean_heat = np.mean([h for _, h in heat_items[1:]], axis=0)
        else:
            mean_heat = cls_heat  # only one register
        axes[2, col_idx].imshow(1 - proc_img, cmap="gray", aspect="auto", alpha=0.4)
        h2d = mean_heat.reshape(1, Wp) if Hp == 1 else mean_heat.reshape(Hp, Wp)
        axes[2, col_idx].imshow(h2d, cmap="inferno", aspect="auto", alpha=0.7,
                                extent=[0, w_img, h_img, 0], interpolation="bilinear")
        axes[2, col_idx].axis("off")

    if n_cols > 0:
        axes[0, 0].set_ylabel("Input", fontsize=7, rotation=0, labelpad=50, va="center")
        axes[1, 0].set_ylabel("CLS\n(reg-0)", fontsize=7, rotation=0, labelpad=50, va="center")
        axes[2, 0].set_ylabel("Mean\n(reg-1…N)", fontsize=7, rotation=0, labelpad=50, va="center")

    plt.suptitle("CLS vs Mean-Register Attention — Middle Layer", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = out_dir / "cls_vs_registers_summary.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ CLS vs Registers summary saved → {save_path}")


def build_character_specific_register_analysis(results, out_dir):
    """
    For models with registers, show how each register attends to
    character-specific positions (CTC-decoded character locations).

    Produces one figure per model showing a heatmap matrix:
      rows = registers (CLS/reg-0, reg-1, …)
      cols = predicted characters
      cell value = attention weight at that character's patch position
    """
    for res in sorted(results, key=lambda r: r["num_reg"]):
        num_reg = res["num_reg"]
        if num_reg == 0:
            continue

        char_positions = res["char_positions"]
        if not char_positions:
            continue

        heatmaps = res["heatmaps"]
        heat_items = list(heatmaps.items())
        run_name = res["run_name"]

        chars = [cp[0] for cp in char_positions]
        time_steps = [cp[1] for cp in char_positions]
        n_chars = len(chars)
        n_regs = len(heat_items)

        # Build matrix: [n_regs, n_chars]
        matrix = np.zeros((n_regs, n_chars))
        for reg_idx, (label, heat_1d) in enumerate(heat_items):
            for char_idx, t in enumerate(time_steps):
                if t < len(heat_1d):
                    matrix[reg_idx, char_idx] = heat_1d[t]

        # Normalise per row for better visibility
        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        matrix_norm = matrix / row_max

        fig, ax = plt.subplots(figsize=(max(8, 0.35 * n_chars), max(3, 0.4 * n_regs)))
        im = ax.imshow(matrix_norm, cmap="YlOrRd", aspect="auto", interpolation="nearest")

        # Axis labels
        ax.set_xticks(range(n_chars))
        ax.set_xticklabels(chars, fontsize=6, rotation=0)
        ax.set_yticks(range(n_regs))
        ax.set_yticklabels([lbl for lbl, _ in heat_items], fontsize=7)
        ax.set_xlabel("Predicted characters", fontsize=9)
        ax.set_ylabel("Token", fontsize=9)
        ax.set_title(
            f"{run_name} (reg={num_reg}) — Character-Specific Register Attention",
            fontsize=10, fontweight="bold",
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        os.makedirs(out_dir / "character_register_matrix", exist_ok=True)
        save_path = out_dir / "character_register_matrix" / f"{run_name}_reg{num_reg}_char_matrix.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved char-register matrix → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLS & Register Token Attention Comparison (middle layer)"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--runs", type=int, nargs="+",
        default=list(range(55, 72)),
        help="Run numbers to compare (default: 55–71)",
    )
    parser.add_argument(
        "--experiments-dir", type=str,
        default=str(PROJECT_ROOT / "saved_models" / "experiments"),
        help="Directory containing run_* folders",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "visualizations" / "cls_register_attention_comparison"),
        help="Output directory for visualisations",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir / "per_model", exist_ok=True)
    os.makedirs(out_dir / "character_register_matrix", exist_ok=True)

    exp_dir = Path(args.experiments_dir)
    device = args.device
    image_path = args.image

    print(f"Device      : {device}")
    print(f"Image       : {image_path}")
    print(f"Runs        : {args.runs}")
    print(f"Output dir  : {out_dir}")
    print()

    # ── Process each model ────────────────────────────────────────────────
    all_results = []
    image_tensor = None  # computed once from first valid config

    for run_num in sorted(args.runs):
        run_dir = exp_dir / f"run_{run_num}"
        if not (run_dir / "model.pt").exists():
            print(f"  ⚠ Skipping run_{run_num} — model.pt not found")
            continue

        run_name = f"run_{run_num}"
        print(f"─── {run_name} ───")

        try:
            net, cfg, num_reg, i2c = load_model(run_dir, device)
            print(f"  Loaded: num_registers={num_reg}, depth={cfg.arch.depth}")

            # Prepare image (only once — all runs share same preproc config)
            if image_tensor is None:
                image_tensor, raw_image = prepare_image(image_path, cfg)
                print(f"  Image preprocessed: {image_tensor.shape}")

            res = visualise_single_model(
                net, cfg, num_reg, i2c,
                image_tensor, raw_image, device,
                run_name, out_dir, image_path,
            )
            all_results.append(res)

        except Exception as e:
            print(f"  ✗ Error processing {run_name}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Free GPU memory
        del net
        torch.cuda.empty_cache() if "cuda" in device else None

    if not all_results:
        print("No models processed successfully. Exiting.")
        return

    # ── Summary visualisations ────────────────────────────────────────────
    print("\n═══ Building comparison grid ═══")
    build_comparison_grid(all_results, out_dir)

    print("\n═══ Building CLS vs Mean-Register summary ═══")
    build_cls_vs_mean_reg_summary(all_results, out_dir)

    print("\n═══ Building character-specific register analysis ═══")
    build_character_specific_register_analysis(all_results, out_dir)

    print(f"\n{'='*60}")
    print(f"All visualisations saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
