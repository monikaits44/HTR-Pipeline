#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Ensure project root is on path
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
    Usage:

    python scripts/postprocessing/tsne_register_tokens.py \
        saved_models/experiments/run_24/vit_rgts_explain \
        --max-samples 800 \
        --save plots/tsne_registers.png
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("explain_dir", type=str,
                        help="Path to vit_rgts_explain folder")
    parser.add_argument("--max-samples", type=int, default=800,
                        help="Optional limit (for speed)")
    parser.add_argument("--save", type=str, default="",
                        help="Where to save figure; leave blank to show")
    args = parser.parse_args()

    explain_dir = args.explain_dir
    csv_path = os.path.join(explain_dir, "vit_rgts_features.csv")

    print("[INFO] Loading metadata …")
    rows = load_metadata(csv_path)

    # Optional: limit samples for speed (by row order)
    if len(rows) > args.max_samples:
        rows = rows[:args.max_samples]

    embeddings = []
    colors = []
    sample_ids = []

    print("[INFO] Loading register-token embeddings …")
    for row in rows:
        sample_idx = int(row["sample_idx"])

        # Robust CER parsing (skip if missing/invalid)
        cer_str = row.get("cer", "")
        try:
            cer = float(cer_str)
        except (TypeError, ValueError):
            continue

        reg_path = os.path.join(
            explain_dir,
            f"sample_{sample_idx:05d}_reg_tokens.npy"
        )
        if not os.path.exists(reg_path):
            # Skip if npy missing
            continue

        reg_tokens = np.load(reg_path)  # expected [1, R, D]
        reg_tokens = np.squeeze(reg_tokens, axis=0)  # [R, D]

        if reg_tokens.ndim != 2:
            # Unexpected shape, skip
            continue

        # Mean-pool across register tokens → style embedding
        style_vec = reg_tokens.mean(axis=0)  # [D]

        embeddings.append(style_vec)
        colors.append(cer)
        sample_ids.append(sample_idx)

    if len(embeddings) == 0:
        print("[ERROR] No valid samples found (check CER field and npy files).")
        return

    embeddings = np.stack(embeddings, axis=0)  # [N, D]
    colors = np.array(colors)

    print(f"[INFO] Running t-SNE on {embeddings.shape[0]} samples …")

    # Build kwargs compatible with older and newer scikit-learn
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": 30,
        "init": "pca",
        "learning_rate": 200,  # safe default for all versions
    }

    # Only set n_iter if the parameter exists in this sklearn version
    if "n_iter" in TSNE.__init__.__code__.co_varnames:
        tsne_kwargs["n_iter"] = 2000

    tsne = TSNE(**tsne_kwargs)
    emb_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=colors,
        cmap="viridis",
        s=14,
        alpha=0.9,
    )
    plt.colorbar(sc, label="CER")
    plt.title("t-SNE of Register-Token Style Embeddings\n(Color = CER)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    if args.save:
        out_path = args.save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved t-SNE plot → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
