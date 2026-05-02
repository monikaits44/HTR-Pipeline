#!/usr/bin/env python3
"""
Visualize writer identification results.

Generates:
    1. mAP / Top-1 / Top-5 vs number of registers (bar chart with value labels)
    2. Line-level t-SNE: 2915 points coloured by writer (shows clustering quality)
    3. Writer-level t-SNE: 32 points labelled by writer ID (shows separation)

Usage:
    python scripts/writer_identification/visualize.py \\
        --results-csv output/writer_id/writer_id_results.csv \\
        --save-dir output/writer_id
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Colour palette for 32+ writers
# --------------------------------------------------------------------------- #

def _get_writer_colors(writer_ids):
    """Return a dict mapping writer_id -> RGBA color for up to ~40 writers."""
    n = len(writer_ids)
    # Combine tab20b + tab20c for 40 distinct colours
    tab20b = plt.get_cmap("tab20b")
    tab20c = plt.get_cmap("tab20c")
    colors = [tab20b(i / 20) for i in range(20)] + [tab20c(i / 20) for i in range(20)]
    return {wid: colors[i % len(colors)] for i, wid in enumerate(sorted(writer_ids))}


# --------------------------------------------------------------------------- #
# Bar chart: metrics vs registers
# --------------------------------------------------------------------------- #

def plot_metrics_vs_registers(results, save_dir):
    """Grouped bar chart: mAP, Top-1, Top-5 vs register count with value labels."""
    results = sorted(results, key=lambda r: r["registers"])
    regs = [str(r["registers"]) for r in results]
    maps = [r["mAP"] for r in results]
    top1s = [r["top1"] for r in results]
    top5s = [r["top5"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(regs))
    w = 0.25

    bars_map = ax.bar(x - w, maps, w, label="mAP", color="#2196F3", edgecolor="white")
    bars_t1 = ax.bar(x, top1s, w, label="Top-1", color="#FF9800", edgecolor="white")
    bars_t5 = ax.bar(x + w, top5s, w, label="Top-5", color="#4CAF50", edgecolor="white")

    # Value labels on bars
    for bars in [bars_map, bars_t1, bars_t5]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(regs, fontsize=11)
    ax.set_xlabel("Number of Register Tokens", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Writer Identification vs Register Tokens\n(VLAC character-wise distance, IAM test set, 32 writers)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotation box
    ax.text(0.98, 0.02,
            "Registers do not affect VLAC distance\n(register features not in distance metric)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            fontstyle="italic", color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(save_dir, "writer_id_vs_registers.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Line-level t-SNE (2915 points)
# --------------------------------------------------------------------------- #

def plot_line_tsne(desc_path, wids_path, save_dir, run_name, metrics=None):
    """Line-level t-SNE: each point is one handwriting line, coloured by writer."""
    descs = np.load(desc_path)
    with open(wids_path) as f:
        wids = json.load(f)

    if len(descs) < 10:
        print(f"  Skipping line t-SNE for {run_name} — too few lines ({len(descs)})")
        return

    from sklearn.manifold import TSNE

    unique_writers = sorted(set(wids))
    n_writers = len(unique_writers)
    wid2col = _get_writer_colors(unique_writers)
    colors = [wid2col[w] for w in wids]

    perplexity = min(50, len(descs) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(descs)

    # --- Figure with legend sidebar ---
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.06, 0.08, 0.62, 0.82])  # left, bottom, width, height

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=12, alpha=0.6, edgecolors="none")

    # Title with metrics
    title = f"Line-Level t-SNE — {run_name}"
    if metrics:
        title += f"  |  mAP={metrics['mAP']:.3f}  Top-1={metrics['top1']:.3f}  Top-5={metrics['top5']:.3f}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE dimension 2", fontsize=10)
    ax.tick_params(labelsize=8)

    # Annotation
    ax.text(0.01, 0.01,
            f"{len(descs)} lines, {n_writers} writers. Tight clusters = good writer separation.",
            transform=ax.transAxes, fontsize=8, fontstyle="italic", color="gray")

    # --- Legend in right panel ---
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=wid2col[w],
               markersize=7, label=f"Writer {w}")
        for w in unique_writers
    ]

    # Place legend to the right of the scatter plot
    legend_ax = fig.add_axes([0.72, 0.08, 0.26, 0.82])
    legend_ax.axis("off")
    legend_ax.set_title(f"Writers ({n_writers})", fontsize=11, fontweight="bold", loc="left")

    # Two columns for 32 writers
    ncol = 2 if n_writers > 16 else 1
    legend_ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
                     frameon=False, ncol=ncol, handletextpad=0.3,
                     columnspacing=0.8, labelspacing=0.4)

    path = os.path.join(save_dir, f"tsne_{run_name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Writer-level t-SNE (32 points, labelled)
# --------------------------------------------------------------------------- #

def plot_writer_tsne(desc_path, wids_path, save_dir, run_name, metrics=None):
    """Writer-level t-SNE: one point per writer, labelled with writer ID."""
    descs = np.load(desc_path)
    with open(wids_path) as f:
        wids = json.load(f)

    if len(descs) < 5:
        print(f"  Skipping writer t-SNE for {run_name} — too few writers ({len(descs)})")
        return

    from sklearn.manifold import TSNE

    wid2col = _get_writer_colors(wids)
    colors = [wid2col[w] for w in wids]

    perplexity = min(8, len(descs) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(descs)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, alpha=0.85,
               edgecolors="black", linewidths=0.5)

    # Label each point with writer ID
    for i, wid in enumerate(wids):
        ax.annotate(wid, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, fontweight="bold", color="black", alpha=0.8)

    title = f"Writer-Level t-SNE — {run_name} ({len(wids)} writers)"
    if metrics:
        title += f"\nmAP={metrics['mAP']:.3f}  Top-1={metrics['top1']:.3f}  Top-5={metrics['top5']:.3f}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE dimension 2", fontsize=10)

    ax.text(0.01, 0.01,
            "Each point = one writer's aggregated VLAC descriptor.",
            transform=ax.transAxes, fontsize=8, fontstyle="italic", color="gray")

    plt.tight_layout()
    path = os.path.join(save_dir, f"tsne_writer_{run_name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Comparison t-SNE (all runs on one plot)
# --------------------------------------------------------------------------- #

def plot_comparison_tsne(results, save_dir):
    """Side-by-side line-level t-SNE for all register configurations."""
    runs_with_data = []
    for r in sorted(results, key=lambda x: x["registers"]):
        run_name = r["run"]
        desc_path = os.path.join(save_dir, f"{run_name}_line_descriptors.npy")
        wids_path = os.path.join(save_dir, f"{run_name}_line_writer_ids.json")
        if os.path.exists(desc_path) and os.path.exists(wids_path):
            runs_with_data.append(r)

    if len(runs_with_data) < 2:
        return

    from sklearn.manifold import TSNE

    n = len(runs_with_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    # Use consistent colours across all subplots
    sample_wids_path = os.path.join(save_dir, f"{runs_with_data[0]['run']}_line_writer_ids.json")
    with open(sample_wids_path) as f:
        sample_wids = json.load(f)
    wid2col = _get_writer_colors(sorted(set(sample_wids)))

    for idx, r in enumerate(runs_with_data):
        ax = axes[idx]
        run_name = r["run"]
        desc_path = os.path.join(save_dir, f"{run_name}_line_descriptors.npy")
        wids_path = os.path.join(save_dir, f"{run_name}_line_writer_ids.json")

        descs = np.load(desc_path)
        with open(wids_path) as f:
            wids = json.load(f)

        colors = [wid2col.get(w, (0.5, 0.5, 0.5, 1.0)) for w in wids]

        perplexity = min(50, len(descs) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        coords = tsne.fit_transform(descs)

        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=5, alpha=0.5, edgecolors="none")
        ax.set_title(f"{run_name}\n{r['registers']} reg | mAP={r['mAP']:.3f} | Top-1={r['top1']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=8)
        if idx == 0:
            ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("Line-Level t-SNE Comparison Across Register Configurations\n"
                 "(same colours = same writer across all panels)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "tsne_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=str, default="output/writer_id/writer_id_results.csv")
    parser.add_argument("--save-dir", type=str, default="output/writer_id")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(args.results_csv):
        print(f"Results CSV not found: {args.results_csv}")
        print("Run evaluate.py first.")
        return

    with open(args.results_csv) as f:
        results = list(csv.DictReader(f))
    for r in results:
        r["registers"] = int(r["registers"])
        r["mAP"] = float(r["mAP"])
        r["top1"] = float(r["top1"])
        r["top5"] = float(r["top5"])

    # 1. Bar chart: metrics vs registers
    plot_metrics_vs_registers(results, args.save_dir)

    # 2. Per-run visualizations
    for r in results:
        run_name = r["run"]
        metrics = r

        # Line-level t-SNE (primary plot)
        line_desc = os.path.join(args.save_dir, f"{run_name}_line_descriptors.npy")
        line_wids = os.path.join(args.save_dir, f"{run_name}_line_writer_ids.json")
        if os.path.exists(line_desc) and os.path.exists(line_wids):
            plot_line_tsne(line_desc, line_wids, args.save_dir, run_name, metrics)

        # Writer-level t-SNE (secondary)
        w_desc = os.path.join(args.save_dir, f"{run_name}_descriptors.npy")
        w_wids = os.path.join(args.save_dir, f"{run_name}_writer_ids.json")
        if os.path.exists(w_desc) and os.path.exists(w_wids):
            plot_writer_tsne(w_desc, w_wids, args.save_dir, run_name, metrics)

    # 3. Comparison plot (all runs side-by-side)
    if len(results) > 1:
        plot_comparison_tsne(results, args.save_dir)

    print("Visualization complete.")


if __name__ == "__main__":
    main()
