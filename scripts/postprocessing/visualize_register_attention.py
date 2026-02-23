#!/usr/bin/env python3
"""
Visualize Register-to-Patch Attention Similarity

Creates heatmaps showing cosine similarity between each register token
and every patch token.  Helps understand what spatial regions each
register token attends to.

Multi-panel output per sample:
  Panel 0: Original grayscale image
  Panel 1..R: Patch-aligned heatmap overlay for each of the R register tokens

Prerequisite: Run extract_vit_rgts_features.py first to generate vit_rgts_explain/

Supported Architectures: vit_rgts only (requires register tokens, num_registers > 0)
Input: vit_rgts_explain/ directory from extract_vit_rgts_features.py
Output: Per-register heatmap overlay PNGs (auto-saved to vit_rgts_explain/)

Supports three execution modes:
  1. All samples (default) — visualize every sample in vit_rgts_explain/
  2. Single / multiple     — specify sample indices with --sample-idx
  3. By image name         — specify image stems with --image-name

Usage:
    # ── All samples (default) ─────────────────────────────────────────
    python scripts/postprocessing/visualize_register_attention.py \\
        saved_models/experiments/run_54/vit_rgts_explain

    # ── Single sample by index ────────────────────────────────────────
    python scripts/postprocessing/visualize_register_attention.py \\
        saved_models/experiments/run_54/vit_rgts_explain --sample-idx 5

    # ── Multiple samples by index ─────────────────────────────────────
    python scripts/postprocessing/visualize_register_attention.py \\
        saved_models/experiments/run_54/vit_rgts_explain --sample-idx 0 3 5 10

    # ── Single image by name ──────────────────────────────────────────
    python scripts/postprocessing/visualize_register_attention.py \\
        saved_models/experiments/run_54/vit_rgts_explain --image-name a01-038-12

    # ── Multiple images by name ───────────────────────────────────────
    python scripts/postprocessing/visualize_register_attention.py \\
        saved_models/experiments/run_54/vit_rgts_explain \\
        --image-name a01-038-12 c04-110-00 r06-137-10

    # ── Save to custom directory (default: saves inside explain_dir) ──
    python scripts/postprocessing/visualize_register_attention.py \\
        saved_models/experiments/run_54/vit_rgts_explain \\
        --save-dir output/register_attention/

    # ── Adjust overlay transparency and resolution ────────────────────
    python scripts/postprocessing/visualize_register_attention.py \\
        saved_models/experiments/run_54/vit_rgts_explain --alpha 0.5 --dpi 200

Arguments:
    explain_dir         Path to vit_rgts_explain/ directory (required)
    --sample-idx N ...  One or more sample indices to visualize
    --image-name S ...  One or more image stems (e.g. a01-038-12)
    --save-dir PATH     Directory to save PNGs (default: explain_dir)
    --alpha FLOAT       Heatmap overlay opacity 0.0–1.0 (default: 0.40)
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
from PIL import Image

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


def resolve_npy_path(explain_dir, sample_idx, suffix, image_name=None):
    """
    Find a .npy file — prefer image-name prefix, fall back to sample index.

    Parameters
    ----------
    explain_dir : str   Path to vit_rgts_explain/
    sample_idx  : int   Numeric sample index (from CSV)
    suffix      : str   e.g. 'reg_tokens', 'seq_tokens'
    image_name  : str   Optional image stem for human-friendly naming
    """
    if image_name:
        p = os.path.join(explain_dir, f"{image_name}_{suffix}.npy")
        if os.path.exists(p):
            return p
    p = os.path.join(explain_dir, f"sample_{sample_idx:05d}_{suffix}.npy")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(
        f"No {suffix}.npy for sample {sample_idx} "
        f"(image_name={image_name}) in {explain_dir}"
    )


def upscale_patch_aligned(patch_map, H_img, W_img):
    """
    Patch-aligned nearest-neighbour upscaling.

    Each value in the (Hp, Wp) grid is expanded into a block whose size
    matches the original patch footprint in image space, then cropped to
    (H_img, W_img).  This avoids the vertical-smear artefact caused by
    bilinear resize on a 1xWp grid.
    """
    Hp, Wp = patch_map.shape
    H_patch = max(1, H_img // Hp)
    W_patch = max(1, W_img // Wp)
    heat_up = np.kron(patch_map, np.ones((H_patch, W_patch)))
    return heat_up[:H_img, :W_img]


def compute_cosine_similarity(reg_tokens, seq_tokens):
    """
    Compute cosine similarity between register and patch tokens.

    Parameters
    ----------
    reg_tokens : ndarray  [R, D]  register token embeddings
    seq_tokens : ndarray  [T, D]  patch token embeddings

    Returns
    -------
    sim : ndarray  [R, T]  cosine similarity scores in [-1, 1]
    """
    reg_norm = np.linalg.norm(reg_tokens, axis=1, keepdims=True) + 1e-8
    seq_norm = np.linalg.norm(seq_tokens, axis=1, keepdims=True) + 1e-8
    return np.matmul(reg_tokens / reg_norm, (seq_tokens / seq_norm).T)


# ── Single-sample visualisation ─────────────────────────────────────────────

def visualize_sample(row, explain_dir, save_dir, alpha=0.40, dpi=150):
    """
    Create a multi-panel register-attention visualisation for one sample.

    Panel 0: Original grayscale image
    Panel 1..R: Patch-aligned heatmap overlay for each register token

    Returns the path of the saved PNG, or None on skip.
    """
    # ── Metadata from CSV row ────────────────────────────────────────
    img_path = row["img_path"]
    Hp, Wp = int(row["Hp"]), int(row["Wp"])
    num_registers = int(float(row["num_registers"]))
    sample_idx = int(row["sample_idx"])
    gt_text = row.get("gt_text", "")
    pred_text = row.get("pred_text", "")

    cer_str = row.get("cer", "")
    wer_str = row.get("wer", "")
    cer = float(cer_str) if cer_str else None
    wer = float(wer_str) if wer_str else None

    image_name = Path(img_path).stem

    if num_registers == 0:
        print(f"  [SKIP] {image_name}: no register tokens (num_registers=0)")
        return None

    # ── Load original image ──────────────────────────────────────────
    if not os.path.isfile(img_path):
        print(f"  [SKIP] Image not found: {img_path}")
        return None
    img = Image.open(img_path).convert("L")
    img_np = np.array(img)
    H_img, W_img = img_np.shape

    # ── Load token embeddings ────────────────────────────────────────
    try:
        reg_path = resolve_npy_path(explain_dir, sample_idx, "reg_tokens",
                                    image_name)
        seq_path = resolve_npy_path(explain_dir, sample_idx, "seq_tokens",
                                    image_name)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None

    reg_tokens = np.load(reg_path)   # [1, R, D] or [R, D]
    seq_tokens = np.load(seq_path)   # [T, 1, D] or [T, D]

    # Squeeze batch dim
    if reg_tokens.ndim == 3:
        reg_tokens = np.squeeze(reg_tokens, axis=0)   # [R, D]
    if seq_tokens.ndim == 3:
        seq_tokens = np.squeeze(seq_tokens, axis=1)    # [T, D]

    R, D = reg_tokens.shape
    T = seq_tokens.shape[0]

    if seq_tokens.shape[1] != D:
        print(f"  [SKIP] {image_name}: dim mismatch reg D={D}, "
              f"seq D={seq_tokens.shape[1]}")
        return None

    expected_T = Hp * Wp
    if T != expected_T:
        print(f"  [WARN] {image_name}: T={T} != Hp*Wp={expected_T}; "
              "reshaping may be incorrect.")

    # ── Compute register->patch cosine similarity ────────────────────
    sim = compute_cosine_similarity(reg_tokens, seq_tokens)  # [R, T]

    # Normalise each register's similarity map to [0, 1]
    sim_maps = []
    for k in range(R):
        s = sim[k]
        s_min, s_max = s.min(), s.max()
        s_norm = (s - s_min) / (s_max - s_min + 1e-8)
        sim_maps.append(s_norm.reshape(Hp, Wp))

    # ── Upscale each map to image dimensions (patch-aligned) ─────────
    heatmaps = []
    for k in range(R):
        heat_up = upscale_patch_aligned(sim_maps[k], H_img, W_img)
        heatmaps.append(heat_up)

    # ── Multi-panel figure ───────────────────────────────────────────
    n_cols = R + 1
    fig_w = min(4 * n_cols, 64)  # cap width for very large R
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 4))

    # Ensure axes is always iterable (edge case: R == 0 handled above)
    if n_cols == 1:
        axes = [axes]

    # Panel 0: Original image
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # Panels 1..R: Each register's similarity heatmap
    for k in range(R):
        ax = axes[k + 1]
        ax.imshow(img_np, cmap="gray")
        ax.imshow(heatmaps[k], cmap="jet", alpha=alpha)
        ax.set_title(f"Reg {k} (cosine)", fontsize=9)
        ax.axis("off")

    # ── Title with metadata ──────────────────────────────────────────
    title_parts = [f"Sample {sample_idx} -- {image_name}"]
    if cer is not None and wer is not None:
        title_parts.append(f"CER={cer:.4f}  WER={wer:.4f}")
    if gt_text:
        title_parts.append(f"GT: {gt_text}")
    if pred_text:
        title_parts.append(f"Pred: {pred_text}")

    fig.suptitle("\n".join(title_parts), fontsize=8, y=1.02)
    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────
    out_name = f"{image_name}_register_attention.png"
    out_path = os.path.join(save_dir, out_name)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize register->patch attention similarity "
                    "from vit_rgts_explain/")
    parser.add_argument("explain_dir", type=str,
                        help="Path to vit_rgts_explain/ directory")
    parser.add_argument("--sample-idx", type=int, nargs="+", default=None,
                        help="One or more sample indices to visualize")
    parser.add_argument("--image-name", type=str, nargs="+", default=None,
                        help="One or more image stems (e.g. a01-038-12)")
    parser.add_argument("--save-dir", type=str, default="",
                        help="Directory for output PNGs (default: explain_dir)")
    parser.add_argument("--alpha", type=float, default=0.40,
                        help="Overlay opacity 0.0-1.0 (default: 0.40)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")
    args = parser.parse_args()

    explain_dir = args.explain_dir
    csv_path = os.path.join(explain_dir, "vit_rgts_features.csv")

    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        print("Run extract_vit_rgts_features.py first.")
        sys.exit(1)

    # Output directory (default: inside explain_dir)
    save_dir = args.save_dir if args.save_dir else explain_dir
    os.makedirs(save_dir, exist_ok=True)

    # ── Load CSV metadata ────────────────────────────────────────────
    rows = load_meta(csv_path)
    if not rows:
        print("Error: CSV is empty.")
        sys.exit(1)

    # Build image_name -> row index for --image-name lookup
    name_to_idx = {}
    for i, r in enumerate(rows):
        stem = Path(r["img_path"]).stem
        name_to_idx[stem] = i

    # ── Determine which samples to visualize ─────────────────────────
    if args.image_name:
        # --image-name mode
        indices = []
        for name in args.image_name:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
            else:
                print(f"  Warning: '{name}' not found in CSV, skipping.")
        mode_label = f"{len(indices)} image(s) by name"

    elif args.sample_idx is not None:
        # --sample-idx mode
        indices = [i for i in args.sample_idx if 0 <= i < len(rows)]
        bad = [i for i in args.sample_idx if i < 0 or i >= len(rows)]
        if bad:
            print(f"  Warning: indices out of range (max {len(rows)-1}): {bad}")
        mode_label = f"{len(indices)} sample(s) by index"

    else:
        # Default: all samples
        indices = list(range(len(rows)))
        mode_label = f"all {len(indices)} samples"

    if not indices:
        print("Error: no valid samples to visualize.")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"Register->Patch Attention Visualization")
    print(f"{'=' * 60}")
    print(f"Input        : {explain_dir}")
    print(f"Similarity   : cosine")
    print(f"Samples      : {mode_label}")
    print(f"Save to      : {save_dir}")
    print(f"Alpha        : {args.alpha}")
    print()

    # ── Visualize each sample ────────────────────────────────────────
    saved = 0
    for idx in indices:
        row = rows[idx]
        image_name = Path(row["img_path"]).stem
        out_path = visualize_sample(
            row, explain_dir, save_dir,
            alpha=args.alpha, dpi=args.dpi,
        )
        if out_path:
            print(f"  [{idx:>5d}] {image_name:>40s}  ->  "
                  f"{os.path.basename(out_path)}")
            saved += 1

    print(f"\nDone -- {saved}/{len(indices)} visualizations saved to: {save_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
