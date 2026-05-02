#!/usr/bin/env python3
"""
Visualize Token Norm Heatmaps

Displays token norm magnitudes as heatmaps overlaid on the input image.
Shows register token norms and patch token norms with correct
patch-aligned upscaling to match original image dimensions.

Three-panel output per sample:
  Panel 1: Original grayscale image
  Panel 2: Raw token norm grid (Hp × Wp) with colorbar
  Panel 3: Patch-aligned heatmap overlay on the original image

Prerequisite: Run extract_vit_rgts_features.py first to generate vit_rgts_explain/

Supported Architectures: vit_rgts only (reads from vit_rgts_explain/)
Input: vit_rgts_explain/ directory from extract_vit_rgts_features.py
Output: Token norm heatmap PNGs (auto-saved to vit_rgts_explain/)

Supports three execution modes:
  1. All samples (default) — visualize every sample in vit_rgts_explain/
  2. Single / multiple     — specify sample indices with --sample-idx
  3. By image name         — specify image stems with --image-name

Usage:
    # ── All samples (default) ─────────────────────────────────────────
    python scripts/postprocessing/visualize_token_norms.py \\
        saved_models/experiments/run_54/vit_rgts_explain

    # ── Single sample by index ────────────────────────────────────────
    python scripts/postprocessing/visualize_token_norms.py \\
        saved_models/experiments/run_54/vit_rgts_explain --sample-idx 5

    # ── Multiple samples by index ─────────────────────────────────────
    python scripts/postprocessing/visualize_token_norms.py \\
        saved_models/experiments/run_54/vit_rgts_explain --sample-idx 0 3 5 10

    # ── Single image by name ──────────────────────────────────────────
    python scripts/postprocessing/visualize_token_norms.py \\
        saved_models/experiments/run_54/vit_rgts_explain --image-name a01-038-12

    # ── Multiple images by name ───────────────────────────────────────
    python scripts/postprocessing/visualize_token_norms.py \\
        saved_models/experiments/run_54/vit_rgts_explain \\
        --image-name a01-038-12 c04-110-00 r06-137-10

    # ── Save to custom directory (default: saves inside explain_dir) ──
    python scripts/postprocessing/visualize_token_norms.py \\
        saved_models/experiments/run_54/vit_rgts_explain \\
        --save-dir output/token_norms/

    # ── Adjust overlay transparency ──────────────────────────────────
    python scripts/postprocessing/visualize_token_norms.py \\
        saved_models/experiments/run_54/vit_rgts_explain --alpha 0.5

Arguments:
    explain_dir         Path to vit_rgts_explain/ directory (required)
    --sample-idx N ...  One or more sample indices to visualize
    --image-name S ...  One or more image stems (e.g. a01-038-12)
    --save-dir PATH     Directory to save PNGs (default: explain_dir)
    --alpha FLOAT       Heatmap overlay opacity 0.0–1.0 (default: 0.45)
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


def upscale_patch_aligned(patch_map, H_img, W_img):
    """
    Patch-aligned nearest-neighbour upscaling.

    Each value in the (Hp, Wp) grid is expanded into a block whose size
    matches the original patch footprint in image space, then cropped to
    (H_img, W_img).  This avoids the vertical-smear artefact caused by
    bilinear resize on a 1×Wp grid.
    """
    Hp, Wp = patch_map.shape
    H_patch = max(1, H_img // Hp)
    W_patch = max(1, W_img // Wp)
    heat_up = np.kron(patch_map, np.ones((H_patch, W_patch)))
    return heat_up[:H_img, :W_img]


def resolve_norms_path(explain_dir, image_name):
    """Find the token_norms .npy file by image name."""
    p = os.path.join(explain_dir, f"{image_name}_token_norms.npy")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(
        f"No token_norms .npy for '{image_name}' in {explain_dir}"
    )


# ── Single-sample visualisation ─────────────────────────────────────────────

def visualize_sample(row, explain_dir, save_dir, alpha=0.45, dpi=150):
    """
    Create a 3-panel token-norm visualisation for one sample.

    Panel 1: Original grayscale image
    Panel 2: Raw norm grid (Hp × Wp) with colorbar + value annotations
    Panel 3: Patch-aligned heatmap overlay on the original image

    Returns the path of the saved PNG.
    """
    # ── Metadata from CSV row ────────────────────────────────────────
    img_path = row["img_path"]
    Hp, Wp = int(row["Hp"]), int(row["Wp"])
    num_registers = int(float(row["num_registers"]))
    sample_idx = int(row["sample_idx"])
    gt_text = row.get("gt_text", "")
    pred_text = row.get("pred_text", "")

    # CER / WER — may be empty for ad-hoc images without ground truth
    cer_str = row.get("cer", "")
    wer_str = row.get("wer", "")
    cer = float(cer_str) if cer_str else None
    wer = float(wer_str) if wer_str else None

    image_name = Path(img_path).stem

    # ── Load original image ──────────────────────────────────────────
    if not os.path.isfile(img_path):
        print(f"  [SKIP] Image not found: {img_path}")
        return None
    img = Image.open(img_path).convert("L")
    img_np = np.array(img)
    H_img, W_img = img_np.shape

    # ── Load token norms ─────────────────────────────────────────────
    norms_path = resolve_norms_path(explain_dir, image_name)
    token_norms = np.load(norms_path).reshape(-1)

    reg_norms = token_norms[:num_registers]
    patch_norms = token_norms[num_registers:]

    if patch_norms.shape[0] != Hp * Wp:
        print(f"  [SKIP] Hp*Wp mismatch for sample {sample_idx}: "
              f"expected {Hp * Wp}, got {patch_norms.shape[0]}")
        return None

    # ── Normalise patch norms to [0, 1] for visualisation ────────────
    patch_map = patch_norms.reshape(Hp, Wp)
    pmin, pmax = patch_map.min(), patch_map.max()
    patch_map_vis = (patch_map - pmin) / (pmax - pmin + 1e-8)

    # ── Patch-aligned upscaling ──────────────────────────────────────
    heat_up = upscale_patch_aligned(patch_map_vis, H_img, W_img)

    # ── 3-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 1: Original image
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # Panel 2: Raw token norm grid with colorbar
    im1 = axes[1].imshow(patch_map, cmap="viridis", aspect="auto")
    axes[1].set_title(f"Token Norm Grid ({Hp}×{Wp})", fontsize=10)
    axes[1].set_xlabel("Width patches")
    axes[1].set_ylabel("Height patches")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="L2 norm")

    # Panel 3: Patch-aligned heatmap overlay
    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(heat_up, cmap="jet", alpha=alpha)
    axes[2].set_title("Heatmap Overlay (Patch-Aligned)", fontsize=10)
    axes[2].axis("off")

    # ── Title with metadata ──────────────────────────────────────────
    title_parts = [f"Sample {sample_idx} — {image_name}"]
    if cer is not None and wer is not None:
        title_parts.append(f"CER={cer:.4f}  WER={wer:.4f}")
    if gt_text:
        title_parts.append(f"GT: {gt_text}")
    if pred_text:
        title_parts.append(f"Pred: {pred_text}")
    if num_registers > 0:
        reg_str = ", ".join(f"{v:.1f}" for v in reg_norms)
        title_parts.append(f"Register norms: [{reg_str}]")

    fig.suptitle("\n".join(title_parts), fontsize=8, y=1.02)
    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(save_dir, f"{image_name}_token_norms_viz.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize token norm heatmaps from vit_rgts_explain/")
    parser.add_argument("explain_dir", type=str,
                        help="Path to vit_rgts_explain/ directory")
    parser.add_argument("--sample-idx", type=int, nargs="+", default=None,
                        help="One or more sample indices to visualize")
    parser.add_argument("--image-name", type=str, nargs="+", default=None,
                        help="One or more image stems (e.g. a01-038-12)")
    parser.add_argument("--save-dir", type=str, default="",
                        help="Directory for output PNGs (default: explain_dir)")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Overlay opacity 0.0–1.0 (default: 0.45)")
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

    # Build image_name → row index map for --image-name lookup
    name_to_idx = {}
    for i, r in enumerate(rows):
        stem = Path(r["img_path"]).stem
        name_to_idx[stem] = i

    # ── Determine which samples to visualize ─────────────────────────
    if args.image_name:
        # --image-name mode: look up by stem
        indices = []
        for name in args.image_name:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
            else:
                print(f"  Warning: '{name}' not found in CSV, skipping.")
        mode_label = f"{len(indices)} image(s) by name"

    elif args.sample_idx is not None:
        # --sample-idx mode: one or more indices
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
    print(f"Token Norm Visualization")
    print(f"{'=' * 60}")
    print(f"Input        : {explain_dir}")
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
            print(f"  [{idx:>5d}] {image_name:>40s}  ->  {os.path.basename(out_path)}")
            saved += 1

    print(f"\nDone — {saved}/{len(indices)} visualizations saved to: {save_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
