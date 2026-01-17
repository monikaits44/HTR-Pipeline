#!/usr/bin/env python3
import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


def upscale_correct(patch_map_vis, H_img, W_img):
    """
    Correct patch-aligned upscaling.
    Uses nearest-neighbor expansion per patch to avoid vertical distortion.
    """
    Hp, Wp = patch_map_vis.shape

    # Approx size of each patch in image space
    H_patch = max(1, H_img // Hp)
    W_patch = max(1, W_img // Wp)

    # Expand each patch into an H_patch x W_patch block
    heat_up = np.kron(patch_map_vis, np.ones((H_patch, W_patch)))

    # Crop to image size
    heat_up = heat_up[:H_img, :W_img]

    return heat_up


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("explain_dir", type=str,
                        help="Path to vit_rgts_explain folder")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    explain_dir = args.explain_dir
    csv_path = os.path.join(explain_dir, "vit_rgts_features.csv")

    # -------------------
    # Load CSV + pick row
    # -------------------
    rows = load_meta(csv_path)
    if args.sample_idx < 0 or args.sample_idx >= len(rows):
        raise IndexError("sample_idx out of range")

    row = rows[args.sample_idx]
    img_path = row["img_path"]
    Hp, Wp = int(row["Hp"]), int(row["Wp"])
    num_registers = int(float(row["num_registers"]))
    sample_idx = int(row["sample_idx"])
    cer, wer = float(row["cer"]), float(row["wer"])
    gt_text = row["gt_text"]
    pred_text = row["pred_text"]

    print(f"[INFO] Visualizing sample_idx={sample_idx}")
    print(f"[INFO] GT   : {gt_text}")
    print(f"[INFO] Pred : {pred_text}")
    print(f"[INFO] CER={cer:.4f} WER={wer:.4f}")
    print(f"[INFO] Hp={Hp}, Wp={Wp}, Registers={num_registers}")

    # -------------------
    # Load original image
    # -------------------
    img = Image.open(img_path).convert("L")
    img_np = np.array(img)
    H_img, W_img = img_np.shape

    # -------------------
    # Load token norms
    # -------------------
    norms_path = os.path.join(explain_dir,
                              f"sample_{sample_idx:05d}_token_norms.npy")

    if not os.path.exists(norms_path):
        raise FileNotFoundError(norms_path)

    token_norms = np.load(norms_path).reshape(-1)

    reg_norms = token_norms[:num_registers]
    patch_norms = token_norms[num_registers:]

    if patch_norms.shape[0] != Hp * Wp:
        raise ValueError("Mismatch: Hp*Wp != number of patch tokens")

    # Patch grid
    patch_map = patch_norms.reshape(Hp, Wp)
    pmin, pmax = patch_map.min(), patch_map.max()
    patch_map_vis = (patch_map - pmin) / (pmax - pmin + 1e-8)

    # -------------------
    # CURRENT (incorrect) upscaling
    # -------------------
    heat_img_bad = Image.fromarray((patch_map_vis * 255).astype(np.uint8))
    heat_img_bad = heat_img_bad.resize((W_img, H_img), resample=Image.BILINEAR)
    heat_up_bad = np.array(heat_img_bad) / 255.0

    # -------------------
    # CORRECT (patch-aligned) upscaling
    # -------------------
    heat_up_good = upscale_correct(patch_map_vis, H_img, W_img)

    # -------------------
    # Plot: ORIGINAL + CURRENT + CORRECTED
    # -------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # ORIGINAL
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # PATCH GRID
    im1 = axes[1].imshow(patch_map_vis, cmap="viridis", aspect="auto")
    axes[1].set_title(f"Token norms grid {Hp}×{Wp}")
    axes[1].set_xlabel("W_patches")
    axes[1].set_ylabel("H_patches")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # CURRENT (WRONG)
    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(heat_up_bad, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay (Current / Incorrect)")
    axes[2].axis("off")

    # CORRECTED (GOOD)
    axes[3].imshow(img_np, cmap="gray")
    axes[3].imshow(heat_up_good, cmap="jet", alpha=0.4)
    axes[3].set_title("Overlay (Corrected Patch-Aligned)")
    axes[3].axis("off")

    fig.suptitle(
        f"Sample {sample_idx} | CER={cer:.4f}, WER={wer:.4f}\n"
        f"GT: {gt_text}\nPred: {pred_text}\n"
        f"Register norms: {np.round(reg_norms, 3)}",
        fontsize=9
    )

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
