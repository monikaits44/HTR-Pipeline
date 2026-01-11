#!/usr/bin/env python3
import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Make sure project root is on sys.path
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_metadata(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    """
    Usage (from project root):

    python scripts/postprocessing/visualize_register_attention.py ^
        saved_models/experiments/run_24/vit_rgts_explain ^
        --sample-idx 0
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "explain_dir",
        type=str,
        help="Path to vit_rgts_explain folder",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Row index in vit_rgts_features.csv to visualize",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save the figure; if empty, show",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity measure for register→patch",
    )
    args = parser.parse_args()

    explain_dir = args.explain_dir
    csv_path = os.path.join(explain_dir, "vit_rgts_features.csv")

    # ------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------
    rows = load_metadata(csv_path)
    if args.sample_idx < 0 or args.sample_idx >= len(rows):
        raise IndexError(
            f"sample_idx {args.sample_idx} out of range (0..{len(rows)-1})"
        )

    row = rows[args.sample_idx]
    sample_idx = int(row["sample_idx"])
    img_path = row["img_path"]
    Hp = int(row["Hp"])
    Wp = int(row["Wp"])
    num_registers = int(row["num_registers"])
    gt_text = row["gt_text"]
    pred_text = row["pred_text"]
    cer = float(row["cer"])
    wer = float(row["wer"])

    print(f"[INFO] Visualizing register→patch similarity for sample_idx={sample_idx}")
    print(f"[INFO] Image path: {img_path}")
    print(f"[INFO] GT   : {gt_text}")
    print(f"[INFO] Pred : {pred_text}")
    print(f"[INFO] CER  : {cer:.4f}, WER: {wer:.4f}")
    print(f"[INFO] Hp={Hp}, Wp={Wp}, num_registers={num_registers}")

    # ------------------------------------------------------
    # Load original image
    # ------------------------------------------------------
    img = Image.open(img_path).convert("L")
    img_np = np.array(img)  # [H_img, W_img]

    # ------------------------------------------------------
    # Load tokens
    # ------------------------------------------------------
    reg_path = os.path.join(
        explain_dir, f"sample_{sample_idx:05d}_reg_tokens.npy"
    )
    seq_path = os.path.join(
        explain_dir, f"sample_{sample_idx:05d}_seq_tokens.npy"
    )

    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Missing reg_tokens file: {reg_path}")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Missing seq_tokens file: {seq_path}")

    reg_tokens = np.load(reg_path)  # [1, R, D]
    seq_tokens = np.load(seq_path)  # [T, 1, D]

    # Squeeze batch dimension
    reg_tokens = np.squeeze(reg_tokens, axis=0)   # [R, D]
    seq_tokens = np.squeeze(seq_tokens, axis=1)   # [T, D]

    R, D = reg_tokens.shape
    T, D2 = seq_tokens.shape
    if D != D2:
        raise ValueError(f"Dim mismatch: reg D={D}, seq D={D2}")
    if T != Hp * Wp:
        print(
            f"[WARN] T={T} != Hp*Wp={Hp*Wp}; "
            "reshaping may be incorrect. Proceeding anyway."
        )

    # ------------------------------------------------------
    # Compute similarity: each register vs all patches
    # ------------------------------------------------------
    # Patches as [T, D]
    P = seq_tokens

    # Optionally normalize for cosine similarity
    if args.similarity == "cosine":
        reg_norm = np.linalg.norm(reg_tokens, axis=1, keepdims=True) + 1e-8  # [R,1]
        P_norm = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8             # [T,1]
        reg_unit = reg_tokens / reg_norm                                     # [R,D]
        P_unit = P / P_norm                                                  # [T,D]
        # sim[k, j] = dot(reg_unit[k], P_unit[j])
        sim = np.matmul(reg_unit, P_unit.T)                                  # [R, T]
    else:
        # plain dot product
        sim = np.matmul(reg_tokens, P.T)                                     # [R, T]

    # Normalize similarity maps per register for visualization
    sims_vis = []
    for k in range(R):
        s = sim[k]
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        sims_vis.append(s.reshape(Hp, Wp))

    # ------------------------------------------------------
    # Upsample each register map to image size
    # ------------------------------------------------------
    H_img, W_img = img_np.shape
    upsampled = []
    for k in range(R):
        heat = (sims_vis[k] * 255).astype(np.uint8)
        heat_img = Image.fromarray(heat)
        heat_img = heat_img.resize((W_img, H_img), resample=Image.BILINEAR)
        upsampled.append(np.array(heat_img) / 255.0)

    # ------------------------------------------------------
    # Plot: original + one subplot per register
    # ------------------------------------------------------
    n_cols = R + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Column 0: original image
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original line")
    axes[0].axis("off")

    # Columns 1..R: each register map
    for k in range(R):
        ax = axes[k + 1]
        ax.imshow(img_np, cmap="gray")
        ax.imshow(upsampled[k], cmap="jet", alpha=0.4)
        ax.set_title(f"Register {k} ({args.similarity})")
        ax.axis("off")

    fig.suptitle(
        f"Sample {sample_idx} | CER={cer:.4f}, WER={wer:.4f}\n"
        f"GT: {gt_text}\nPred: {pred_text}",
        fontsize=9,
    )
    plt.tight_layout()

    if args.save:
        out_path = args.save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved figure → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
