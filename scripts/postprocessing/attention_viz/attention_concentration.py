#!/usr/bin/env python3
"""
Attention Concentration & Focus Analysis
==========================================
Quantitative analysis across the full register sweep (0–16).

Produces:
  1. Metrics vs Register Count: diagonality, entropy, peak sharpness (averaged
     over multiple sample images for statistical robustness)
  2. Correlation scatter: CER vs attention quality
  3. CSV with all per-image, per-model metrics

Usage:
    python scripts/postprocessing/attention_viz/attention_concentration.py \\
        --images-dir notebook/sample_images \\
        --runs-dir saved_models/experiments \\
        --device cpu \\
        --save-dir visualizations/attention_concentration
"""

import sys, os, argparse, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.postprocessing.attention_viz.core import (
    load_configs, load_model, extract_attention, ctc_decode,
    load_htr_image, get_char_attention, infer_num_registers,
    compute_diagonality, compute_entropy, compute_peak_sharpness,
    collect_image_paths,
)

RUN_MAP = {
    55: 0, 64: 1, 56: 2, 66: 3, 58: 4, 65: 5, 67: 6, 68: 7,
    63: 8, 69: 9, 61: 10, 57: 11, 59: 12, 60: 13, 62: 14, 70: 15, 71: 16,
}


def load_run_cer(run_dir: str) -> float:
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return float("nan")
    import csv as csv_mod
    best = 999.0
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            try:
                cer = float(row.get("test/cer", 999))
                if cer < best:
                    best = cer
            except (ValueError, TypeError):
                continue
    return best


def main():
    parser = argparse.ArgumentParser(description="Attention concentration analysis")
    parser.add_argument("--images-dir", default="notebook/sample_images",
                        help="Directory with sample images")
    parser.add_argument("--images", nargs="*", help="Specific image paths (overrides --images-dir)")
    parser.add_argument("--runs-dir", default="saved_models/experiments")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="visualizations/attention_concentration")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    project_root = Path(__file__).resolve().parents[3]
    cfg = load_configs(
        str(project_root / "configs" / "config.yaml"),
        str(project_root / "configs" / "baseline_vit_rgts_v2.yaml"),
    )
    cfg.device = args.device

    h = getattr(cfg.preproc, "image_height", 128)
    w = getattr(cfg.preproc, "image_width", 1024)

    # Collect images
    if args.images:
        image_paths = args.images
    else:
        image_paths = collect_image_paths([args.images_dir])
    image_paths = image_paths[: args.max_images]

    print(f"Analyzing {len(image_paths)} images across {len(RUN_MAP)} register configurations")

    # ── Collect metrics ──────────────────────────────────────────────────
    all_records = []  # (n_reg, img_name, diag, entropy, sharpness, cer)

    for run_num, n_reg in sorted(RUN_MAP.items(), key=lambda x: x[1]):
        model_path = os.path.join(args.runs_dir, f"run_{run_num}", "model.pt")
        if not os.path.exists(model_path):
            continue

        cfg_run = cfg.copy()
        cfg_run.arch.num_registers = n_reg

        try:
            net, classes, i2c = load_model(cfg_run, model_path, args.device)
        except Exception as e:
            print(f"  Skip run_{run_num}: {e}")
            continue

        cer = load_run_cer(os.path.join(args.runs_dir, f"run_{run_num}"))

        for img_path in image_paths:
            image_tensor, raw_image = load_htr_image(img_path, h, w)
            result = extract_attention(net, image_tensor, args.device)
            text, positions = ctc_decode(result["logits"], i2c)

            if len(positions) < 2:
                continue

            char_attn = get_char_attention(
                result["attn_maps"], positions, result["num_registers"],
                layer="last", gamma=1.0,
            )

            diag = compute_diagonality(positions, char_attn.shape[1])
            ent = np.mean([compute_entropy(row) for row in char_attn])
            sharp = np.mean([compute_peak_sharpness(row) for row in char_attn])

            all_records.append({
                "n_reg": n_reg,
                "run": run_num,
                "image": os.path.basename(img_path),
                "diagonality": diag,
                "entropy": ent,
                "sharpness": sharp,
                "cer": cer,
                "text": text,
            })

        del net
        print(f"  Done: run_{run_num} (Reg-{n_reg})")

    if not all_records:
        print("No records collected. Check model paths.")
        return

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(args.save_dir, "attention_metrics_full.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_reg", "run", "image", "diagonality", "entropy", "sharpness", "cer", "text"
        ])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Saved CSV: {csv_path}")

    # ── Aggregate per register count ─────────────────────────────────────
    from collections import defaultdict
    agg = defaultdict(lambda: {"diag": [], "ent": [], "sharp": [], "cer": []})
    for r in all_records:
        k = r["n_reg"]
        agg[k]["diag"].append(r["diagonality"])
        agg[k]["ent"].append(r["entropy"])
        agg[k]["sharp"].append(r["sharpness"])
        agg[k]["cer"].append(r["cer"])

    regs = sorted(agg.keys())
    mean_diag = [np.nanmean(agg[k]["diag"]) for k in regs]
    std_diag = [np.nanstd(agg[k]["diag"]) for k in regs]
    mean_ent = [np.nanmean(agg[k]["ent"]) for k in regs]
    std_ent = [np.nanstd(agg[k]["ent"]) for k in regs]
    mean_sharp = [np.nanmean(agg[k]["sharp"]) for k in regs]
    std_sharp = [np.nanstd(agg[k]["sharp"]) for k in regs]
    mean_cer = [np.nanmean(agg[k]["cer"]) for k in regs]

    # ── Plot 1: Metrics vs Register Count (with error bars) ──────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=args.dpi)

    axes[0].errorbar(regs, mean_diag, yerr=std_diag, fmt="o-", color="#4CAF50",
                     linewidth=2, capsize=3, markersize=5)
    axes[0].set_xlabel("Register Tokens", fontsize=11)
    axes[0].set_ylabel("Diagonality (ρ)", fontsize=11)
    axes[0].set_title("Alignment Quality", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(0, 17, 2))
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(regs, mean_ent, yerr=std_ent, fmt="s-", color="#9C27B0",
                     linewidth=2, capsize=3, markersize=5)
    axes[1].set_xlabel("Register Tokens", fontsize=11)
    axes[1].set_ylabel("Entropy (bits)", fontsize=11)
    axes[1].set_title("Attention Focus (↓ = more focused)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(0, 17, 2))
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(regs, mean_sharp, yerr=std_sharp, fmt="^-", color="#FF5722",
                     linewidth=2, capsize=3, markersize=5)
    axes[2].set_xlabel("Register Tokens", fontsize=11)
    axes[2].set_ylabel("Peak Sharpness", fontsize=11)
    axes[2].set_title("Attention Sharpness (↑ = sharper)", fontsize=12, fontweight="bold")
    axes[2].set_xticks(range(0, 17, 2))
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Attention Quality vs Register Count (avg over {len(image_paths)} images)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out1 = os.path.join(args.save_dir, "metrics_vs_registers.png")
    fig.savefig(out1, bbox_inches="tight", dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Plot 2: CER vs Attention Quality (correlation) ───────────────────
    fig2, (ax_cd, ax_ce) = plt.subplots(1, 2, figsize=(12, 5), dpi=args.dpi)

    ax_cd.scatter(mean_cer, mean_diag, c=regs, cmap="viridis", s=80, edgecolors="black", zorder=3)
    for i, k in enumerate(regs):
        ax_cd.annotate(str(k), (mean_cer[i], mean_diag[i]), fontsize=7, ha="left")
    ax_cd.set_xlabel("CER", fontsize=11)
    ax_cd.set_ylabel("Diagonality (ρ)", fontsize=11)
    ax_cd.set_title("CER vs Attention Alignment", fontsize=12, fontweight="bold")
    ax_cd.grid(True, alpha=0.3)

    ax_ce.scatter(mean_cer, mean_ent, c=regs, cmap="viridis", s=80, edgecolors="black", zorder=3)
    for i, k in enumerate(regs):
        ax_ce.annotate(str(k), (mean_cer[i], mean_ent[i]), fontsize=7, ha="left")
    ax_ce.set_xlabel("CER", fontsize=11)
    ax_ce.set_ylabel("Entropy (bits)", fontsize=11)
    ax_ce.set_title("CER vs Attention Focus", fontsize=12, fontweight="bold")
    ax_ce.grid(True, alpha=0.3)

    fig2.suptitle("Recognition Quality vs Attention Quality",
                  fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out2 = os.path.join(args.save_dir, "cer_vs_attention.png")
    fig2.savefig(out2, bbox_inches="tight", dpi=args.dpi, facecolor="white")
    plt.close(fig2)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
