#!/usr/bin/env python3
"""
Comprehensive Register Impact Figure
=====================================
Produces a single multi-panel figure that tells the complete register-token story:

  Panel A — CER/WER vs Register Count (line plot, 0–16)
  Panel B — Attention Quality vs Register Count (diagonality + entropy)
  Panel C — Character Attention Grid (Fig.5-style, rows = Reg-0/4/8/16)
  Panel D — Register Token Behavior (what registers attend to)

This is the KEY DELIVERABLE for the project report.

Usage:
    python scripts/postprocessing/attention_viz/comprehensive_comparison.py \\
        --image notebook/sample_images/a01-038-12.png \\
        --runs-dir saved_models/experiments \\
        --device cpu \\
        --save-dir visualizations/comprehensive_comparison \\
        --dpi 200

Does NOT modify any existing file.
"""

import sys, os, argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Tuple

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.postprocessing.attention_viz.core import (
    load_configs, load_model, extract_attention, ctc_decode,
    load_htr_image, get_char_attention, get_register_attention,
    compute_diagonality, compute_entropy, compute_peak_sharpness,
    infer_num_registers, pure_heatmap, render_heatmap_on_image,
)

# ── Run → register-count mapping (complete 0–16 sweep) ──────────────────────
RUN_MAP = {
    55: 0, 64: 1, 56: 2, 66: 3, 58: 4, 65: 5, 67: 6, 68: 7,
    63: 8, 69: 9, 61: 10, 57: 11, 59: 12, 60: 13, 62: 14, 70: 15, 71: 16,
}

# Key runs for qualitative comparison (4 representative models)
KEY_RUNS = [55, 58, 63, 71]  # Reg-0, 4, 8, 16


def load_run_metrics(run_dir: str) -> Dict:
    """Load CER/WER from results.csv in a run directory."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return {}
    import csv
    best = {"cer": 999, "wer": 999}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cer = float(row.get("test/cer", 999))
                wer = float(row.get("test/wer", 999))
                if cer < best["cer"]:
                    best["cer"] = cer
                    best["wer"] = wer
            except (ValueError, TypeError):
                continue
    return best


def panel_a_performance(ax_cer, ax_wer, runs_dir: str):
    """Panel A: CER/WER vs Register Count (0–16)."""
    regs, cers, wers = [], [], []
    for run_num, n_reg in sorted(RUN_MAP.items(), key=lambda x: x[1]):
        run_dir = os.path.join(runs_dir, f"run_{run_num}")
        metrics = load_run_metrics(run_dir)
        if metrics:
            regs.append(n_reg)
            cers.append(metrics["cer"] * 100)
            wers.append(metrics["wer"] * 100)

    # CER subplot
    ax_cer.plot(regs, cers, "o-", color="#2196F3", linewidth=2, markersize=5)
    ax_cer.set_ylabel("CER (%)", fontsize=10)
    ax_cer.set_xlabel("Number of Register Tokens", fontsize=10)
    ax_cer.set_title("(A) Recognition Performance", fontsize=11, fontweight="bold")
    ax_cer.set_xticks(range(0, 17, 2))
    ax_cer.grid(True, alpha=0.3)
    if cers:
        ax_cer.axhline(min(cers), color="#2196F3", ls="--", alpha=0.4, lw=1)

    # WER subplot
    ax_wer.plot(regs, wers, "s-", color="#FF5722", linewidth=2, markersize=5)
    ax_wer.set_ylabel("WER (%)", fontsize=10)
    ax_wer.set_xlabel("Number of Register Tokens", fontsize=10)
    ax_wer.set_xticks(range(0, 17, 2))
    ax_wer.grid(True, alpha=0.3)
    if wers:
        ax_wer.axhline(min(wers), color="#FF5722", ls="--", alpha=0.4, lw=1)


def panel_b_quality(
    ax_diag, ax_ent,
    runs_dir: str, image_tensor, raw_image, cfg, device: str,
):
    """Panel B: Attention quality metrics vs register count."""
    regs, diags, entropies = [], [], []

    for run_num, n_reg in sorted(RUN_MAP.items(), key=lambda x: x[1]):
        model_path = os.path.join(runs_dir, f"run_{run_num}", "model.pt")
        if not os.path.exists(model_path):
            continue
        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg
        try:
            net, classes, i2c = load_model(cfg_run, model_path, device)
        except Exception:
            continue
        result = extract_attention(net, image_tensor, device)
        text, positions = ctc_decode(result["logits"], i2c)
        if len(positions) < 2:
            continue

        char_attn = get_char_attention(
            result["attn_maps"], positions, result["num_registers"],
            layer="last", gamma=1.0,
        )
        diag = compute_diagonality(positions, char_attn.shape[1])
        ent = np.mean([compute_entropy(row) for row in char_attn])

        regs.append(n_reg)
        diags.append(diag)
        entropies.append(ent)
        del net

    # Diagonality
    ax_diag.plot(regs, diags, "o-", color="#4CAF50", linewidth=2, markersize=5)
    ax_diag.set_ylabel("Diagonality (ρ)", fontsize=10)
    ax_diag.set_xlabel("Number of Register Tokens", fontsize=10)
    ax_diag.set_title("(B) Attention Quality", fontsize=11, fontweight="bold")
    ax_diag.set_xticks(range(0, 17, 2))
    ax_diag.grid(True, alpha=0.3)

    # Entropy on twin axis
    ax_ent.plot(regs, entropies, "^--", color="#9C27B0", linewidth=2, markersize=5)
    ax_ent.set_ylabel("Entropy (bits)", fontsize=10, color="#9C27B0")
    ax_ent.tick_params(axis="y", labelcolor="#9C27B0")


def panel_c_char_grid(
    ax, runs_dir: str, image_tensor, raw_image, cfg, device: str,
    gamma: float = 3.0, max_chars: int = 15,
):
    """Panel C: Fig.5-style character attention heatmaps for Reg-0, 4, 8, 16."""
    grid_data = []  # list of (label, char_labels, heatmap_rows)

    for run_num in KEY_RUNS:
        n_reg = RUN_MAP[run_num]
        model_path = os.path.join(runs_dir, f"run_{run_num}", "model.pt")
        if not os.path.exists(model_path):
            continue

        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg
        net, classes, i2c = load_model(cfg_run, model_path, device)
        result = extract_attention(net, image_tensor, device)
        text, positions = ctc_decode(result["logits"], i2c)

        if len(positions) == 0:
            continue

        char_attn = get_char_attention(
            result["attn_maps"], positions, result["num_registers"],
            layer="last", gamma=gamma,
        )

        # Truncate to max_chars
        n = min(len(text), max_chars, char_attn.shape[0])
        grid_data.append((f"Reg-{n_reg}", list(text[:n]), char_attn[:n]))
        del net

    if not grid_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Determine common number of chars (pad shorter ones)
    max_n = max(len(d[1]) for d in grid_data)
    num_patches = grid_data[0][2].shape[1]

    # Build composite image: rows = models, columns = characters
    cell_h, cell_w = 50, max(200, num_patches * 3)
    cmap_fn = plt.get_cmap("inferno")

    composite_rows = []
    for label, chars, char_attn in grid_data:
        row_cells = []
        for ci in range(max_n):
            if ci < len(chars):
                h = char_attn[ci]
                # Upsample to cell_w
                h_up = np.interp(np.linspace(0, len(h) - 1, cell_w), np.arange(len(h)), h)
                cell = cmap_fn(h_up)[..., :3]  # [cell_w, 3]
                cell = np.tile(cell, (cell_h, 1, 1))
            else:
                cell = np.zeros((cell_h, cell_w, 3))
            row_cells.append(cell)
        # Add label column (rendered as text later)
        composite_rows.append(np.concatenate(row_cells, axis=1))

    composite = np.concatenate(composite_rows, axis=0)
    ax.imshow(composite)

    # Add character labels at top
    for ci in range(max_n):
        # Use chars from first model that has this char
        ch = ""
        for _, chars, _ in grid_data:
            if ci < len(chars):
                ch = chars[ci]
                break
        ax.text(
            ci * cell_w + cell_w / 2, -5, repr(ch)[1:-1],
            ha="center", va="bottom", fontsize=7, fontfamily="monospace",
        )

    # Add model labels on left
    for ri, (label, _, _) in enumerate(grid_data):
        ax.text(
            -10, ri * cell_h + cell_h / 2, label,
            ha="right", va="center", fontsize=8, fontweight="bold",
        )

    ax.set_title("(C) Character Attention Maps (Fig.5 style)", fontsize=11, fontweight="bold")
    ax.axis("off")


def panel_d_register_behavior(
    ax, runs_dir: str, image_tensor, raw_image, cfg, device: str,
):
    """Panel D: What register tokens attend to (Reg-4 and Reg-16)."""
    import matplotlib.cm as cm

    panel_runs = [58, 71]  # Reg-4, Reg-16
    strips = []
    labels = []

    for run_num in panel_runs:
        n_reg = RUN_MAP[run_num]
        model_path = os.path.join(runs_dir, f"run_{run_num}", "model.pt")
        if not os.path.exists(model_path):
            continue

        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg
        net, classes, i2c = load_model(cfg_run, model_path, device)
        result = extract_attention(net, image_tensor, device)

        reg_attn = get_register_attention(
            result["attn_maps"], result["num_registers"],
            layer="last", direction="reg_to_patch",
        )  # [R, P]

        for ri in range(min(n_reg, 4)):  # Show max 4 registers per model
            row = reg_attn[ri]
            mn, mx = row.min(), row.max()
            if mx - mn > 1e-8:
                row = (row - mn) / (mx - mn)
            strips.append(row)
            labels.append(f"Reg-{n_reg} R{ri}")
        del net

    if not strips:
        ax.text(0.5, 0.5, "No register data", ha="center", va="center", transform=ax.transAxes)
        return

    # Stack as image
    num_patches = strips[0].shape[0]
    data = np.stack(strips)  # [num_strips, P]

    ax.imshow(data, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Patch Position (left → right)", fontsize=9)
    ax.set_title("(D) Register Token Attention", fontsize=11, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive register impact figure")
    parser.add_argument("--image", required=True, help="Path to sample image")
    parser.add_argument("--runs-dir", default="saved_models/experiments",
                        help="Directory containing run_XX folders")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="visualizations/comprehensive_comparison")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=3.0, help="Heatmap contrast")
    parser.add_argument("--max-chars", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load base config (ViT-RGTS v2)
    project_root = Path(__file__).resolve().parents[3]
    cfg = load_configs(
        str(project_root / "configs" / "config.yaml"),
        str(project_root / "configs" / "baseline_vit_rgts_v2.yaml"),
    )
    cfg.device = args.device

    # Load image
    h = getattr(cfg.preproc, "image_height", 128)
    w = getattr(cfg.preproc, "image_width", 1024)
    image_tensor, raw_image = load_htr_image(args.image, h, w)

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 16), dpi=args.dpi)
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1.5, 1], hspace=0.35, wspace=0.3)

    # Panel A: Performance (top-left and top-right)
    ax_cer = fig.add_subplot(gs[0, 0])
    ax_wer = fig.add_subplot(gs[0, 1])
    panel_a_performance(ax_cer, ax_wer, args.runs_dir)

    # Panel B: Attention quality (second row)
    ax_diag = fig.add_subplot(gs[1, 0])
    ax_ent = ax_diag.twinx()
    panel_b_quality(ax_diag, ax_ent, args.runs_dir, image_tensor, raw_image, cfg, args.device)

    # Panel B right: Original image for reference
    ax_img = fig.add_subplot(gs[1, 1])
    ax_img.imshow(1.0 - raw_image, cmap="gray", aspect="auto")
    ax_img.set_title("Input Image", fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # Panel C: Character attention grid (third row, full width)
    ax_grid = fig.add_subplot(gs[2, :])
    panel_c_char_grid(
        ax_grid, args.runs_dir, image_tensor, raw_image, cfg, args.device,
        gamma=args.gamma, max_chars=args.max_chars,
    )

    # Panel D: Register behavior (bottom row, full width)
    ax_reg = fig.add_subplot(gs[3, :])
    panel_d_register_behavior(
        ax_reg, args.runs_dir, image_tensor, raw_image, cfg, args.device,
    )

    # Save
    img_stem = Path(args.image).stem
    out_path = os.path.join(args.save_dir, f"comprehensive_{img_stem}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
