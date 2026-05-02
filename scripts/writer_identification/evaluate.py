#!/usr/bin/env python3
"""
Writer Identification evaluation pipeline.

Loads a trained ViT-RGTS model, extracts VLAC + register features from IAM
lines, groups by writer (prefix of line ID, e.g. 'c04'), and computes
retrieval metrics using character-wise distance (Eq. 8-9).

Usage:
    python scripts/writer_identification/evaluate.py \\
        --run-dir saved_models/experiments/run_58 \\
        --data-dir data/IAM/processed_lines \\
        --split test \\
        --device cpu \\
        --save-dir output/writer_id

    # Compare multiple register configs:
    python scripts/writer_identification/evaluate.py \\
        --run-dirs saved_models/experiments/run_55 \\
                   saved_models/experiments/run_58 \\
                   saved_models/experiments/run_71 \\
        --data-dir data/IAM/processed_lines \\
        --split test --device cpu --save-dir output/writer_id
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from omegaconf import OmegaConf

from scripts.writer_identification.vlac import (
    extract_line_features,
    vlac_from_lines,
    compute_global_prototypes,
    select_characters,
)
from scripts.writer_identification.embeddings import (
    build_writer_descriptor,
    build_writer_descriptors,
)  # used for t-SNE descriptor export
from scripts.writer_identification.retrieval import (
    cosine_distance_matrix,
    compute_retrieval_metrics,
    character_wise_distance,
    vlac_distance_matrix,
)
from scripts.postprocessing.attention_viz.core import load_model, load_htr_image


# --------------------------------------------------------------------------- #
# IAM writer / form parsing
# --------------------------------------------------------------------------- #

def parse_writer_id(line_id):
    """Extract writer ID from IAM line ID:  'a01-000u-03' → 'a01'.

    In IAM, the writer is identified by the prefix before the first hyphen
    (e.g. 'a01', 'c04', 'd01').  Multiple forms belong to the same writer.
    """
    return line_id.split("-")[0]


def parse_form_id(line_id):
    """Extract form ID from IAM line ID:  'a01-000u-03' → 'a01-000u'."""
    parts = line_id.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return line_id


def load_line_ids(data_dir, split):
    """Read gt.txt and return list of (line_id, image_path, transcription)."""
    gt_path = os.path.join(data_dir, split, "gt.txt")
    lines = []
    with open(gt_path) as f:
        for row in f:
            parts = row.strip().split(" ", 1)
            line_id = parts[0]
            transcr = parts[1] if len(parts) > 1 else ""
            img_path = os.path.join(data_dir, split, line_id + ".png")
            lines.append((line_id, img_path, transcr))
    return lines


# --------------------------------------------------------------------------- #
# Single-run evaluation
# --------------------------------------------------------------------------- #

def evaluate_run(run_dir, data_dir, split, device, max_lines=None):
    """
    Full writer identification evaluation for one trained model.

    Pipeline (Raven et al.):
      1. Extract per-line features (patch tokens at CTC positions)
      2. Compute global character prototypes μ_c (Eq. 4)
      3. Residual VLAC aggregation per line (Eq. 5)
      4. Character selection A* on writer-level VLAC (Eq. 6)
      5. Line-level retrieval with character-wise distance (Eq. 8-9)

    Writer identity = IAM prefix (e.g. 'c04'), NOT form ID.

    Returns
    -------
    metrics : dict with mAP, top1, top5
    writer_vlacs : dict  writer_id → (vlac, counts, reg_mean)
    charset : list[str]
    selected_chars : list[int]
    """
    # Load model
    config_path = os.path.join(run_dir, "config.json")
    model_path = os.path.join(run_dir, "model.pt")
    with open(config_path) as f:
        run_cfg = json.load(f)
    cfg = OmegaConf.create(run_cfg)
    net, classes, i2c = load_model(cfg, model_path, device)
    charset = list(classes)

    num_registers = getattr(net.backbone, "num_registers", 0)
    embed_dim = getattr(net.backbone, "embed_dim", 256)

    # Load line metadata
    lines = load_line_ids(data_dir, split)
    if max_lines:
        lines = lines[:max_lines]

    # --- Step 1: Extract raw features per line ---
    print(f"  Step 1/4: Extracting features from {len(lines)} lines ...")
    all_line_feats = []   # raw per-line features (before VLAC)
    line_writer_ids = []  # writer ID per line (prefix, e.g. 'c04')
    writer_line_indices = defaultdict(list)  # writer → indices

    for idx, (line_id, img_path, transcr) in enumerate(tqdm(lines, desc="  Lines")):
        writer_id = parse_writer_id(line_id)
        if not os.path.exists(img_path):
            continue
        img_tensor, _ = load_htr_image(img_path)
        lf = extract_line_features(net, img_tensor, i2c, device)
        all_line_feats.append(lf)
        line_writer_ids.append(writer_id)
        writer_line_indices[writer_id].append(len(all_line_feats) - 1)

    if not all_line_feats:
        empty = {"mAP": 0, "top1": 0, "top5": 0, "n_queries": 0, "per_query": []}
        return empty, {}, charset, [], {"vlacs": [], "counts": [], "writer_ids": []}

    n_writers = len(writer_line_indices)
    print(f"    {len(all_line_feats)} lines, {n_writers} writers")

    # --- Step 2: Compute global character prototypes (Eq. 4) ---
    print("  Step 2/4: Computing global character prototypes ...")
    prototypes = compute_global_prototypes(all_line_feats, charset)

    # --- Step 3: VLAC aggregation with residuals (Eq. 5) ---
    print("  Step 3/4: VLAC residual aggregation ...")
    line_vlacs = []
    line_counts = []

    for lf in all_line_feats:
        vlac, counts, reg_mean = vlac_from_lines([lf], charset, prototypes)
        line_vlacs.append(vlac)
        line_counts.append(counts)

    # Writer-level VLAC for character selection + interpretability
    writer_vlacs = {}
    for wid, indices in writer_line_indices.items():
        feats = [all_line_feats[i] for i in indices]
        vlac, counts, reg_mean = vlac_from_lines(feats, charset, prototypes)
        writer_vlacs[wid] = (vlac, counts, reg_mean)

    # --- Step 4: Character selection A* (Eq. 6) ---
    print("  Step 4/4: Character selection (τ=0.8) ...")
    selected_chars, char_maps = select_characters(writer_vlacs, charset, tau=0.8)

    if char_maps:
        top_chars = sorted(char_maps.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"    Selected {len(selected_chars)}/{len(charset)} characters.  Top by mAP:")
        for ci, m in top_chars:
            print(f"      '{charset[ci]}' mAP={m:.4f}")

    # --- Retrieval via character-wise distance (Eq. 8-9) ---
    dist_matrix = vlac_distance_matrix(line_vlacs, line_counts, selected_chars)
    metrics = compute_retrieval_metrics(dist_matrix, line_writer_ids)

    # Bundle line-level data for visualization
    line_data = {
        "vlacs": line_vlacs,
        "counts": line_counts,
        "writer_ids": line_writer_ids,
    }

    return metrics, writer_vlacs, charset, selected_chars, line_data


# --------------------------------------------------------------------------- #
# Interpretable character-distance example
# --------------------------------------------------------------------------- #

def print_char_distances(vlac_a, vlac_b, counts_a, counts_b, charset, top_k=10):
    """Print most discriminative characters between two writers."""
    char_dists, mean_dist = character_wise_distance(vlac_a, vlac_b, counts_a, counts_b)
    print(f"  Mean character distance: {mean_dist:.4f}")
    sorted_chars = sorted(char_dists.items(), key=lambda x: x[1], reverse=True)
    print(f"  Most different characters (top {top_k}):")
    for idx, d in sorted_chars[:top_k]:
        c = charset[idx] if idx < len(charset) else "?"
        print(f"    '{c}' : {d:.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Writer Identification via VLAC + Register Tokens")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=str, help="Single experiment run directory")
    group.add_argument("--run-dirs", nargs="+", type=str, help="Multiple run directories to compare")
    parser.add_argument("--data-dir", type=str, default="data/IAM/processed_lines")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-lines", type=int, default=None, help="Limit lines for quick testing")
    parser.add_argument("--save-dir", type=str, default="output/writer_id")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    run_dirs = [args.run_dir] if args.run_dir else args.run_dirs

    all_results = []

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        n_reg = cfg.get("arch", {}).get("num_registers", 0)

        print(f"\n{'='*60}")
        print(f"  {run_name}  |  registers={n_reg}")
        print(f"{'='*60}")

        metrics, writer_vlacs, charset, selected_chars, line_data = evaluate_run(
            run_dir, args.data_dir, args.split, args.device, args.max_lines
        )

        n_writers = len(writer_vlacs)
        print(f"\n  Results ({args.split} set, {metrics['n_queries']} queries, {n_writers} writers):")
        print(f"    mAP   : {metrics['mAP']:.4f}")
        print(f"    Top-1 : {metrics['top1']:.4f}")
        print(f"    Top-5 : {metrics['top5']:.4f}")
        print(f"    A* chars: {len(selected_chars)}/{len(charset)}")

        all_results.append({
            "run": run_name,
            "registers": n_reg,
            "mAP": metrics["mAP"],
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "n_queries": metrics["n_queries"],
            "n_writers": n_writers,
        })

        # Save line-level descriptors (flattened VLAC) + writer IDs for t-SNE
        line_descs = np.array([v.flatten() for v in line_data["vlacs"]])
        np.save(os.path.join(args.save_dir, f"{run_name}_line_descriptors.npy"), line_descs)
        with open(os.path.join(args.save_dir, f"{run_name}_line_writer_ids.json"), "w") as f:
            json.dump(line_data["writer_ids"], f)

        # Save writer-level descriptors
        wids_sorted = sorted(writer_vlacs.keys())
        writer_descs = np.array([
            build_writer_descriptor(writer_vlacs[w][0], writer_vlacs[w][2])
            for w in wids_sorted
        ])
        np.save(os.path.join(args.save_dir, f"{run_name}_descriptors.npy"), writer_descs)
        with open(os.path.join(args.save_dir, f"{run_name}_writer_ids.json"), "w") as f:
            json.dump(wids_sorted, f)

        # Save selected characters
        sel_chars_info = [charset[ci] for ci in selected_chars] if selected_chars else []
        with open(os.path.join(args.save_dir, f"{run_name}_selected_chars.json"), "w") as f:
            json.dump(sel_chars_info, f, indent=2)

        # Interpretable example: show char distances between first two writers
        wids = sorted(writer_vlacs.keys())
        if len(wids) >= 2:
            w1, w2 = wids[0], wids[1]
            print(f"\n  Interpretable distance: writer '{w1}' vs '{w2}'")
            print_char_distances(
                writer_vlacs[w1][0], writer_vlacs[w2][0],
                writer_vlacs[w1][1], writer_vlacs[w2][1],
                charset,
            )

    # Summary CSV
    csv_path = os.path.join(args.save_dir, "writer_id_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "registers", "mAP", "top1", "top5", "n_queries", "n_writers"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  Comparison Summary")
        print(f"{'='*60}")
        print(f"  {'Run':<12} {'Reg':>4} {'mAP':>8} {'Top-1':>8} {'Top-5':>8}")
        print(f"  {'-'*44}")
        for r in sorted(all_results, key=lambda x: x["registers"]):
            print(f"  {r['run']:<12} {r['registers']:>4} {r['mAP']:>8.4f} {r['top1']:>8.4f} {r['top5']:>8.4f}")


if __name__ == "__main__":
    main()
