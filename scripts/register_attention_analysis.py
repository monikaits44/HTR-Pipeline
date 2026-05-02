#!/usr/bin/env python3
"""
Register Effect on Character-Level Attention — Definitive Analysis
==================================================================

Approach: CTC Self-Attention (Method 1 from paper_fig5_ctc)
  For character c decoded at CTC timestep t_c:
    A_c = head_avg(attn_last_layer)[R + t_c, R : R+Wp]
  This is the ViT self-attention analogue of the paper's cross-attention A_c.

Why this method (not GradCAM):
  1. Registers directly modify self-attention → this shows the effect directly
  2. Deterministic — no gradient noise
  3. Self-attention at timestep t_c shows WHERE the model looks for char c
  4. GradCAM shows what CAUSES the prediction — complementary but indirect

Outputs:
  register_char_attention_{name}.{pdf,png}  — comparison grid per image
  register_metrics_summary.{pdf,png}        — quantitative bar charts
  register_metrics.csv                      — raw numerical data
  Console: results table + conclusions
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.attention_visualization import (
    load_charset, load_model, prepare_image, ctc_greedy_decode,
    run_explain, attn_to_image_crop,
    IMAGE_H, IMAGE_W, DEFAULT_RUNS
)

SAMPLE_DIR = ROOT / "notebook" / "sample_images"
OUTPUT_DIR = ROOT / "output" / "attention_visualizations"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================================
# Core: Raw Attention Extraction (no gamma — fair cross-model comparison)
# ============================================================================

def extract_char_attention(attn_maps, num_registers, peak_cols, Wp,
                           layer_idx=-1):
    """
    Extract per-character self-attention profiles.

    A_c = head_avg(attn)[R + t_c, R:R+Wp]

    Uses percentile normalization only (no gamma correction) so that
    attention shape differences between register configurations are
    preserved for fair comparison.

    Returns:
        char_maps:     list of [Wp] arrays — normalized attention per char
        xmax_positions: list of float — weighted centroid X^(c)_max
        raw_maps:      list of [Wp] arrays — unnormalized (for metrics)
    """
    attn = attn_maps[layer_idx].mean(dim=1)  # [B, S, S]
    R = num_registers

    char_maps = []
    xmax_positions = []
    raw_maps = []

    for c in peak_cols:
        idx = R + c
        if idx >= attn.shape[1]:
            continue
        row = attn[0, idx, R:R + Wp].numpy().copy()
        raw_maps.append(row.copy())

        # Percentile normalization
        p1, p99 = np.percentile(row, 1), np.percentile(row, 99)
        if p99 > p1:
            row = np.clip((row - p1) / (p99 - p1), 0, 1)
        else:
            row = (row - row.min()) / (row.max() - row.min() + 1e-8)

        char_maps.append(row)

        # Xmax: weighted centroid
        positions = np.arange(Wp, dtype=np.float64)
        w_sum = row.sum()
        xmax = (float(np.sum(positions * row) / w_sum)
                if w_sum > 0 else float(np.argmax(row)))
        xmax_positions.append(xmax)

    return char_maps, xmax_positions, raw_maps


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(raw_maps, xmax_positions):
    """
    Compute attention quality metrics from RAW (unnormalized) attention.

    Returns dict with:
      entropy      — Shannon entropy in bits (lower = more focused)
      localization — peak / mean ratio (higher = sharper peak)
      xmax_rho     — Spearman ρ of Xmax with reading order (1.0 = perfect)
    """
    entropies = []
    localizations = []

    for attn in raw_maps:
        p = attn / (attn.sum() + 1e-12)
        p = np.clip(p, 1e-12, None)
        H = -np.sum(p * np.log2(p))
        entropies.append(H)

        L = attn.max() / (attn.mean() + 1e-12)
        localizations.append(L)

    if len(xmax_positions) >= 3:
        rho, _ = spearmanr(np.arange(len(xmax_positions)), xmax_positions)
    else:
        rho = 1.0

    return {
        "entropy": np.mean(entropies),
        "localization": np.mean(localizations),
        "xmax_rho": rho,
    }


# ============================================================================
# Figure: Register × Character Grid
# ============================================================================

def plot_register_char_grid(image_path, models_info, charset, device,
                            output_path, max_chars=10):
    """
    THE definitive figure: one grid per word.
      Rows  = register configurations (0, 4, 8, 16)
      Col 0 = input image (cropped to text)
      Cols 1..N = per-character attention heatmap
      Xmax centroid as white dashed line
      Metrics annotation per row
    """
    n_models = len(models_info)

    # Run all models on the same image
    results = []
    for run_id, model, n_regs in models_info:
        img_tensor, img_np, text_bbox, _ = prepare_image(image_path, device)
        res = run_explain(model, img_tensor)
        decoded, peak_cols = ctc_greedy_decode(res["logits"], charset)[0]
        Hp, Wp = res["grid_size"]

        char_maps, xmax_pos, raw_maps = extract_char_attention(
            res["attn_maps"], n_regs, peak_cols, Wp
        )
        metrics = compute_metrics(raw_maps, xmax_pos)
        n_show = min(len(decoded), max_chars, len(char_maps))

        results.append({
            "run_id": run_id, "n_regs": n_regs,
            "decoded": decoded.strip(), "char_maps": char_maps,
            "xmax_pos": xmax_pos, "metrics": metrics,
            "Wp": Wp, "n_show": n_show,
        })

    x_start, x_end = text_bbox
    img_crop = img_np[:, x_start:x_end]
    crop_w = x_end - x_start
    actual_max = max(r["n_show"] for r in results)
    n_cols = actual_max + 1

    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(2.0 * n_cols, 2.2 * n_models),
        gridspec_kw={"wspace": 0.06, "hspace": 0.40},
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        n_regs = r["n_regs"]
        decoded = r["decoded"]
        char_maps = r["char_maps"]
        xmax_pos = r["xmax_pos"]
        Wp = r["Wp"]
        n_show = r["n_show"]
        m = r["metrics"]

        # Col 0: image + register label
        axes[row, 0].imshow(img_crop, cmap="gray", aspect="auto")
        reg_label = f"{n_regs} reg" if n_regs > 0 else "baseline"
        axes[row, 0].set_ylabel(
            reg_label, fontsize=11, fontweight="bold",
            rotation=0, labelpad=50, va="center"
        )
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("Input", fontsize=10, fontweight="bold")
        # Metrics below image
        axes[row, 0].set_xlabel(
            f'H={m["entropy"]:.1f}  L={m["localization"]:.1f}  '
            f'\u03c1={m["xmax_rho"]:.2f}',
            fontsize=7.5, color="dimgray"
        )

        # Cols 1..n_show: character heatmaps
        for c_idx in range(n_show):
            ax = axes[row, c_idx + 1]
            char = decoded[c_idx] if c_idx < len(decoded) else ""
            display_char = char if char != " " else "SPC"

            if c_idx < len(char_maps):
                attn_1d = char_maps[c_idx]
                attn_2d, _ = attn_to_image_crop(
                    attn_1d, Wp, text_bbox, smooth_sigma=2
                )
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.25)
                ax.imshow(attn_2d, cmap="inferno", aspect="auto",
                          alpha=0.75, vmin=0, vmax=1)

                # Xmax centroid line
                if c_idx < len(xmax_pos):
                    xmax_px = xmax_pos[c_idx] / Wp * IMAGE_W - x_start
                    xmax_px = np.clip(xmax_px, 0, crop_w - 1)
                    ax.axvline(x=xmax_px, color="white", lw=1.0,
                               ls="--", alpha=0.85)
            else:
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)

            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"'{display_char}'", fontsize=10,
                             fontweight="bold", color="darkblue")

        # Hide unused columns
        for c_idx in range(n_show, actual_max):
            axes[row, c_idx + 1].axis("off")

    word = results[0]["decoded"]
    fig.suptitle(
        f'Character-Level Attention vs Register Count — "{word}"\n'
        r'$A_c = \mathrm{avg\_heads}(\mathrm{attn}_{\mathrm{last}})'
        r'[R{+}t_c,\; R{:}R{+}W_p]$'
        '    H=entropy  L=localization  \u03c1=Xmax monotonicity',
        fontsize=11, fontweight="bold", y=1.04,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return results


# ============================================================================
# Figure: Metrics Summary
# ============================================================================

def plot_metrics_summary(all_results, output_path):
    """
    Bar chart: entropy / localization / Xmax ρ vs register count.
    Averaged across all test images.
    """
    reg_counts = sorted(set(
        r["n_regs"] for results in all_results for r in results
    ))

    avg = {n: {"ent": [], "loc": [], "rho": []} for n in reg_counts}
    for results in all_results:
        for r in results:
            n = r["n_regs"]
            avg[n]["ent"].append(r["metrics"]["entropy"])
            avg[n]["loc"].append(r["metrics"]["localization"])
            avg[n]["rho"].append(r["metrics"]["xmax_rho"])

    ent_means = [np.mean(avg[n]["ent"]) for n in reg_counts]
    loc_means = [np.mean(avg[n]["loc"]) for n in reg_counts]
    rho_means = [np.mean(avg[n]["rho"]) for n in reg_counts]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(reg_counts))
    labels = [str(r) for r in reg_counts]
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']

    # Entropy (lower = better)
    bars = ax1.bar(x, ent_means, color=colors[:len(reg_counts)],
                   edgecolor="black", lw=0.5)
    ax1.set_xlabel("Register Count")
    ax1.set_ylabel("Attention Entropy (bits)")
    ax1.set_title("Entropy \u2193 = More Focused", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    for bar, val in zip(bars, ent_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    # Localization (higher = better)
    bars = ax2.bar(x, loc_means, color=colors[:len(reg_counts)],
                   edgecolor="black", lw=0.5)
    ax2.set_xlabel("Register Count")
    ax2.set_ylabel("Peak / Mean Ratio")
    ax2.set_title("Localization \u2191 = Sharper Peak", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    for bar, val in zip(bars, loc_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    # Xmax monotonicity (higher = better)
    bars = ax3.bar(x, rho_means, color=colors[:len(reg_counts)],
                   edgecolor="black", lw=0.5)
    ax3.set_xlabel("Register Count")
    ax3.set_ylabel("Spearman \u03c1")
    ax3.set_title("Xmax Monotonicity \u2191 = Correct L\u2192R Order",
                  fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1.15)
    for bar, val in zip(bars, rho_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    fig.suptitle(
        "Quantitative: Register Effect on Character Attention Quality\n"
        f"(averaged over {len(all_results)} images)",
        fontsize=13, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
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
        description="Register Effect on Character-Level Attention"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--max-chars", type=int, default=10)
    args = parser.parse_args()

    device = args.device
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default: two short words for clean grids
    if args.images:
        image_paths = [str(Path(p).resolve()) for p in args.images]
    else:
        image_paths = [
            str(SAMPLE_DIR / "a01-038-12.png"),   # "talks."
            str(SAMPLE_DIR / "a06-095-10.png"),   # "he said."
        ]

    print("=" * 60)
    print("Register Effect on Character-Level Attention")
    print("Method: CTC Self-Attention (paper_fig5_ctc)")
    print("=" * 60)

    # Load all 4 register configurations
    charset = load_charset()
    models_info = []
    for run_id in DEFAULT_RUNS:
        try:
            model, _, n_regs = load_model(run_id, device)
            models_info.append((run_id, model, n_regs))
            print(f"  {run_id}: {n_regs} registers")
        except Exception as e:
            print(f"  {run_id}: FAILED — {e}")

    models_info.sort(key=lambda x: x[2])
    print()

    # Generate per-image comparison grids
    all_results = []
    for img_path in image_paths:
        name = Path(img_path).stem
        print(f"Analyzing: {name}")
        results = plot_register_char_grid(
            img_path, models_info, charset, device,
            str(out_dir / f"register_char_attention_{name}.pdf"),
            max_chars=args.max_chars,
        )
        all_results.append(results)

    # Aggregate metrics chart
    print("\nGenerating metrics summary...")
    plot_metrics_summary(
        all_results,
        str(out_dir / "register_metrics_summary.pdf"),
    )

    # ----------------------------------------------------------------
    # Results table
    # ----------------------------------------------------------------
    reg_counts = sorted(set(r["n_regs"] for r in all_results[0]))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Regs':>5s}  {'Entropy\u2193':>10s}  {'Localiz\u2191':>10s}"
          f"  {'Xmax \u03c1\u2191':>10s}")
    print("-" * 42)

    summary = {}
    for n_regs in reg_counts:
        ent = np.mean([r["metrics"]["entropy"]
                       for res in all_results for r in res
                       if r["n_regs"] == n_regs])
        loc = np.mean([r["metrics"]["localization"]
                       for res in all_results for r in res
                       if r["n_regs"] == n_regs])
        rho = np.mean([r["metrics"]["xmax_rho"]
                       for res in all_results for r in res
                       if r["n_regs"] == n_regs])
        summary[n_regs] = {"entropy": ent, "localization": loc, "xmax_rho": rho}
        print(f"{n_regs:>5d}  {ent:>10.3f}  {loc:>10.3f}  {rho:>10.4f}")

    # Save CSV
    csv_path = out_dir / "register_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("image,run_id,registers,entropy,localization,xmax_rho,decoded\n")
        for img_path, results in zip(image_paths, all_results):
            name = Path(img_path).stem
            for r in results:
                m = r["metrics"]
                f.write(
                    f'{name},{r["run_id"]},{r["n_regs"]},'
                    f'{m["entropy"]:.4f},{m["localization"]:.4f},'
                    f'{m["xmax_rho"]:.4f},"{r["decoded"]}"\n'
                )
    print(f"\nCSV: {csv_path}")

    # ----------------------------------------------------------------
    # Conclusions
    # ----------------------------------------------------------------
    ent_0 = summary[0]["entropy"]
    loc_0 = summary[0]["localization"]

    best_ent_regs = min(
        (n for n in reg_counts if n > 0),
        key=lambda n: summary[n]["entropy"]
    )
    best_loc_regs = max(
        (n for n in reg_counts if n > 0),
        key=lambda n: summary[n]["localization"]
    )
    ent_best = summary[best_ent_regs]["entropy"]
    loc_best = summary[best_loc_regs]["localization"]

    ent_delta = (ent_0 - ent_best) / ent_0 * 100
    loc_delta = (loc_best - loc_0) / loc_0 * 100

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    if ent_delta > 1:
        print(f"1. Register tokens REDUCE attention entropy by {ent_delta:.1f}%"
              f" (best: {best_ent_regs} reg)")
        print("   → Character attention is more spatially focused")
    elif ent_delta < -1:
        print(f"1. Attention entropy INCREASES by {abs(ent_delta):.1f}%"
              " with registers")
        print("   → Attention spreads more broadly (registers may distribute"
              " information)")
    else:
        print(f"1. Attention entropy is STABLE across register counts"
              f" (Δ={ent_delta:+.1f}%)")
        print("   → Register tokens do not significantly change attention focus")

    if loc_delta > 1:
        print(f"2. Localization IMPROVES by {loc_delta:.1f}%"
              f" (best: {best_loc_regs} reg)")
        print("   → Sharper spatial peaks per character")
    elif loc_delta < -1:
        print(f"2. Localization DECREASES by {abs(loc_delta):.1f}%"
              " with registers")
        print("   → Attention peaks are less sharp but may be more"
              " distributed")
    else:
        print(f"2. Localization is STABLE (Δ={loc_delta:+.1f}%)")

    all_rho = [summary[n]["xmax_rho"] for n in reg_counts]
    if all(r > 0.85 for r in all_rho):
        print("3. All configurations maintain correct left→right character"
              " ordering")
        print(f"   (Xmax ρ > 0.85 across all register counts)")
    else:
        min_rho_n = min(reg_counts, key=lambda n: summary[n]["xmax_rho"])
        print(f"3. Xmax ordering varies: worst ρ={summary[min_rho_n]['xmax_rho']:.3f}"
              f" at {min_rho_n} registers")

    print()
    print("Primary finding: Register tokens' main effect is artifact")
    print("reduction (see Fig 1 & Fig 3), which cleans the attention")
    print("maps by absorbing global information into dedicated tokens,")
    print("leaving patch tokens with spatially coherent local attention.")


if __name__ == "__main__":
    main()
