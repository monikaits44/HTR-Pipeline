#!/usr/bin/env python3
import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Make sure project root is on path (not strictly needed here, but safe)
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_meta(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    """
    Usage (from project root):

    python scripts/postprocessing/visualize_token_norms.py \
        saved_models/experiments/run_24/vit_rgts_explain \
        --sample-idx 0
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "explain_dir",
        type=str,
        help="Path to vit_rgts_explain folder in run_XX",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Which sample index (row in vit_rgts_features.csv) to visualize",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save the figure (PNG). If empty, show interactively.",
    )
    args = parser.parse_args()

    explain_dir = args.explain_dir
    csv_path = os.path.join(explain_dir, "vit_rgts_features.csv")

    # -------------------------------
    # Load metadata row
    # -------------------------------
    rows = load_meta(csv_path)
    if args.sample_idx < 0 or args.sample_idx >= len(rows):
        raise IndexError(
            f"sample_idx {args.sample_idx} out of range (0..{len(rows)-1})"
        )

    row = rows[args.sample_idx]
    img_path = row["img_path"]
    Hp = int(row["Hp"])
    Wp = int(row["Wp"])
    num_registers = int(float(row["num_registers"]))
    sample_idx = int(row["sample_idx"])
    cer = float(row["cer"])
    wer = float(row["wer"])
    gt_text = row["gt_text"]
    pred_text = row["pred_text"]

    print(f"[INFO] Visualizing sample_idx={sample_idx}")
    print(f"[INFO] Image: {img_path}")
    print(f"[INFO] GT   : {gt_text}")
    print(f"[INFO] Pred : {pred_text}")
    print(f"[INFO] CER  : {cer:.4f}, WER: {wer:.4f}")
    print(f"[INFO] Hp={Hp}, Wp={Wp}, num_registers={num_registers}")

    # -------------------------------
    # Load original image
    # -------------------------------
    img = Image.open(img_path).convert("L")
    img_np = np.array(img)  # [H_img, W_img]

    # -------------------------------
    # Load token norms
    # -------------------------------
    norms_path = os.path.join(
        explain_dir, f"sample_{sample_idx:05d}_token_norms.npy"
    )
    if not os.path.exists(norms_path):
        raise FileNotFoundError(f"Token norms file not found: {norms_path}")

    token_norms = np.load(norms_path)  # [R + T]
    if token_norms.ndim != 1:
        token_norms = token_norms.reshape(-1)

    # Split registers vs patches
    reg_norms = token_norms[:num_registers]        # [R]
    patch_norms = token_norms[num_registers:]      # [T = Hp*Wp]

    if patch_norms.shape[0] != Hp * Wp:
        raise ValueError(
            f"patch_norms length {patch_norms.shape[0]} != Hp*Wp={Hp*Wp}"
        )

    # -------------------------------
    # Reshape to patch grid
    # -------------------------------
    patch_map = patch_norms.reshape(Hp, Wp)

    # Normalize for visualization
    pmin, pmax = patch_map.min(), patch_map.max()
    patch_map_vis = (patch_map - pmin) / (pmax - pmin + 1e-8)

    # -------------------------------
    # Upsample heatmap to image size
    # -------------------------------
    H_img, W_img = img_np.shape
    heat_img = Image.fromarray((patch_map_vis * 255).astype(np.uint8))
    heat_img = heat_img.resize((W_img, H_img), resample=Image.BILINEAR)
    heat_up = np.array(heat_img) / 255.0  # [0,1] normalized

    # -------------------------------
    # Plot
    # -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Original
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original line image")
    axes[0].axis("off")

    # Patch grid heatmap
    im1 = axes[1].imshow(patch_map_vis, cmap="viridis", aspect="auto")
    axes[1].set_title(f"Token norms (patch grid {Hp}×{Wp})")
    axes[1].set_xlabel("W_patches")
    axes[1].set_ylabel("H_patches")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay on full image
    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(heat_up, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay: token norms on image")
    axes[2].axis("off")

    fig.suptitle(
        f"Sample {sample_idx} | CER={cer:.4f}, WER={wer:.4f}\n"
        f"GT: {gt_text}\nPred: {pred_text}\n"
        f"Register norms: {np.round(reg_norms, 3)}",
        fontsize=9,
    )

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
