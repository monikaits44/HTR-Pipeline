#!/usr/bin/env python3
"""
t-SNE Visualization of Register Token Embeddings

Runs t-SNE dimensionality reduction on mean-pooled register token embeddings
and creates a scatter plot colored by CER.  Useful for understanding whether
register content correlates with prediction quality.

Each sample's 16 register tokens (each 256-dim) are mean-pooled into a single
256-dim "style vector".  t-SNE projects all style vectors to 2D.  Points are
colored by CER: low-CER (good) samples cluster differently from high-CER (bad)
ones if register tokens encode recognition-relevant information.

Prerequisite: Run extract_vit_rgts_features.py first to generate vit_rgts_explain/

Supported Architectures: vit_rgts only (requires register tokens, num_registers > 0)
Input: vit_rgts_explain/ directory from extract_vit_rgts_features.py
Output: t-SNE scatter plot PNG (auto-saved to vit_rgts_explain/)

Supports three execution modes:
  1. All samples (default) — use every sample in vit_rgts_explain/
  2. Single / multiple     — specify sample indices with --sample-idx
  3. By image name         — specify image stems with --image-name

Usage:
    # ── All samples (default) ─────────────────────────────────────────
    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_54/vit_rgts_explain

    # ── Single sample highlighted among all ───────────────────────────
    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_54/vit_rgts_explain --sample-idx 5

    # ── Multiple samples highlighted ──────────────────────────────────
    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_54/vit_rgts_explain --sample-idx 0 3 5 10

    # ── Single image by name ──────────────────────────────────────────
    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_54/vit_rgts_explain --image-name a01-038-12

    # ── Multiple images by name ───────────────────────────────────────
    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_54/vit_rgts_explain \
        --image-name a01-038-12 c04-110-00 r06-137-10

    # ── Limit number of samples (for large datasets) ──────────────────
    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_54/vit_rgts_explain --max-samples 500

    # ── Save to custom directory (default: saves inside explain_dir) ──
    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_54/vit_rgts_explain \
        --save-dir output/tsne/

    # ── Adjust t-SNE perplexity ───────────────────────────────────────
    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_54/vit_rgts_explain --perplexity 15

Arguments:
    explain_dir         Path to vit_rgts_explain/ directory (required)
    --sample-idx N ...  Highlight specific sample indices (still plots all)
    --image-name S ...  Highlight specific image stems (still plots all)
    --max-samples N     Maximum samples for t-SNE (default: 2000)
    --perplexity N      t-SNE perplexity (default: 30; lower for fewer samples)
    --save-dir PATH     Directory to save PNGs (default: explain_dir)
    --dpi INT           Output resolution (default: 150)
"""
import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_meta(csv_path):
    """Load vit_rgts_features.csv into a list of dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def resolve_npy_path(explain_dir, suffix, image_name):
    """Find a .npy file by image name."""
    p = os.path.join(explain_dir, f"{image_name}_{suffix}.npy")
    if os.path.exists(p):
        return p
    return None  # caller decides how to handle


def load_embeddings(rows, explain_dir):
    """
    Load mean-pooled register embeddings for all valid rows.

    Returns
    -------
    embeddings : ndarray [N, D]   mean-pooled register vectors
    cer_vals   : ndarray [N]      CER per sample
    wer_vals   : ndarray [N]      WER per sample
    names      : list[str]        image stems
    row_indices: list[int]        original row indices (for highlight lookup)
    """
    embeddings = []
    cer_vals = []
    wer_vals = []
    names = []
    row_indices = []

    for i, row in enumerate(rows):
        sample_idx = int(row["sample_idx"])
        image_name = Path(row["img_path"]).stem

        # Parse CER — skip if missing/invalid
        cer_str = row.get("cer", "")
        wer_str = row.get("wer", "")
        try:
            cer = float(cer_str) if cer_str else None
        except (TypeError, ValueError):
            cer = None
        try:
            wer = float(wer_str) if wer_str else None
        except (TypeError, ValueError):
            wer = None

        # Load register tokens
        reg_path = resolve_npy_path(explain_dir, "reg_tokens", image_name)
        if reg_path is None:
            continue

        reg_tokens = np.load(reg_path)
        if reg_tokens.ndim == 3:
            reg_tokens = np.squeeze(reg_tokens, axis=0)  # [R, D]
        if reg_tokens.ndim != 2:
            continue

        # Mean-pool across register tokens → style embedding [D]
        style_vec = reg_tokens.mean(axis=0)

        embeddings.append(style_vec)
        cer_vals.append(cer if cer is not None else 0.0)
        wer_vals.append(wer if wer is not None else 0.0)
        names.append(image_name)
        row_indices.append(i)

    if not embeddings:
        return None, None, None, None, None

    return (np.stack(embeddings), np.array(cer_vals), np.array(wer_vals),
            names, row_indices)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE of register-token style embeddings "
                    "from vit_rgts_explain/")
    parser.add_argument("explain_dir", type=str,
                        help="Path to vit_rgts_explain/ directory")
    parser.add_argument("--sample-idx", type=int, nargs="+", default=None,
                        help="Highlight specific sample indices on the plot")
    parser.add_argument("--image-name", type=str, nargs="+", default=None,
                        help="Highlight specific image stems on the plot")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Max samples for t-SNE (default: 2000)")
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity (default: 30)")
    parser.add_argument("--save-dir", type=str, default="",
                        help="Directory for output PNG (default: explain_dir)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")
    args = parser.parse_args()

    explain_dir = args.explain_dir
    csv_path = os.path.join(explain_dir, "vit_rgts_features.csv")

    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        print("Run extract_vit_rgts_features.py first.")
        sys.exit(1)

    save_dir = args.save_dir if args.save_dir else explain_dir
    os.makedirs(save_dir, exist_ok=True)

    # ── Load CSV metadata ────────────────────────────────────────────
    rows = load_meta(csv_path)
    if not rows:
        print("Error: CSV is empty.")
        sys.exit(1)

    # Limit samples if needed
    if len(rows) > args.max_samples:
        rows = rows[:args.max_samples]

    # Build image_name -> row index map
    name_to_idx = {}
    for i, r in enumerate(rows):
        stem = Path(r["img_path"]).stem
        name_to_idx[stem] = i

    # ── Determine highlighted samples ────────────────────────────────
    highlight_indices = set()
    if args.image_name:
        for name in args.image_name:
            if name in name_to_idx:
                highlight_indices.add(name_to_idx[name])
            else:
                print(f"  Warning: '{name}' not found in CSV, skipping.")
        mode_label = f"all samples, highlighting {len(highlight_indices)} by name"
    elif args.sample_idx is not None:
        for idx in args.sample_idx:
            if 0 <= idx < len(rows):
                highlight_indices.add(idx)
            else:
                print(f"  Warning: index {idx} out of range (max {len(rows)-1})")
        mode_label = f"all samples, highlighting {len(highlight_indices)} by index"
    else:
        mode_label = f"all {len(rows)} samples"

    print(f"{'=' * 60}")
    print(f"t-SNE of Register-Token Style Embeddings")
    print(f"{'=' * 60}")
    print(f"Input        : {explain_dir}")
    print(f"Samples      : {mode_label}")
    print(f"Max samples  : {args.max_samples}")
    print(f"Perplexity   : {args.perplexity}")
    print(f"Save to      : {save_dir}")
    print()

    # ── Load embeddings ──────────────────────────────────────────────
    print("Loading register-token embeddings...")
    embeddings, cer_vals, wer_vals, names, row_indices = \
        load_embeddings(rows, explain_dir)

    if embeddings is None:
        print("Error: no valid samples found (check CER field and .npy files).")
        sys.exit(1)

    N, D = embeddings.shape
    print(f"  Loaded {N} samples, embedding dim = {D}")

    # ── Adjust perplexity if too high for sample count ───────────────
    perplexity = min(args.perplexity, max(2, N - 1))
    if perplexity != args.perplexity:
        print(f"  Adjusted perplexity to {perplexity} (N={N} too small)")

    # ── Run t-SNE ────────────────────────────────────────────────────
    print(f"Running t-SNE (perplexity={perplexity})...")

    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": 42,
    }
    # Compatibility: n_iter may not exist in newer sklearn
    if "n_iter" in TSNE.__init__.__code__.co_varnames:
        tsne_kwargs["n_iter"] = 2000

    tsne = TSNE(**tsne_kwargs)
    emb_2d = tsne.fit_transform(embeddings)
    print(f"  t-SNE complete.")

    # ── Map highlight set to embedding indices ───────────────────────
    # row_indices[j] is the original CSV row for embedding j
    highlight_emb = set()
    for j, ri in enumerate(row_indices):
        if ri in highlight_indices:
            highlight_emb.add(j)

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    # All points
    sc = ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=cer_vals, cmap="viridis", s=20, alpha=0.8,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="CER", fraction=0.046, pad=0.04)

    # Highlighted points — larger, red-edged, labelled
    if highlight_emb:
        for j in highlight_emb:
            ax.scatter(
                emb_2d[j, 0], emb_2d[j, 1],
                s=120, facecolors="none", edgecolors="red", linewidths=2,
                zorder=5,
            )
            ax.annotate(
                names[j],
                (emb_2d[j, 0], emb_2d[j, 1]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=7, color="red", fontweight="bold",
            )

    ax.set_title(
        f"t-SNE of Register-Token Style Embeddings (N={N})\n"
        f"Color = CER  |  Perplexity = {perplexity}",
        fontsize=10,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    # ── Summary stats in corner ──────────────────────────────────────
    stats_text = (
        f"CER: mean={cer_vals.mean():.4f}, "
        f"min={cer_vals.min():.4f}, max={cer_vals.max():.4f}"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes, fontsize=7,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(save_dir, "tsne_register_tokens.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print(f"{'=' * 60}")

    # ── Print highlighted sample details ─────────────────────────────
    if highlight_emb:
        print("\nHighlighted samples:")
        for j in sorted(highlight_emb):
            ri = row_indices[j]
            row = rows[ri]
            print(f"  {names[j]:>40s}  CER={cer_vals[j]:.4f}  "
                  f"WER={wer_vals[j]:.4f}  "
                  f"t-SNE=({emb_2d[j, 0]:.1f}, {emb_2d[j, 1]:.1f})")


if __name__ == "__main__":
    main()
