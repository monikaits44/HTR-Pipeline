#!/usr/bin/env python3
"""
Publication Figure 3: Quantitative Register Effect Analysis
=============================================================
Four-panel figure comparing attention quality metrics across
register configurations (0/4/8/16 registers).

Panel (a): Per-character attention entropy (lower = more focused)
Panel (b): Spatial ordering Spearman ρ (higher = better L→R alignment)
Panel (c): Aligned head count per register config
Panel (d): Token norm distribution — register vs patch tokens

Computed over multiple test images for statistical robustness.

Usage:
  python scripts/pub_register_quantitative.py
  python scripts/pub_register_quantitative.py --device cuda:0
  python scripts/pub_register_quantitative.py --n_images 20
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models import HTRNet
from utils.preprocessing import load_image, preprocess

EXPERIMENTS_DIR = ROOT / "saved_models" / "experiments"
CLASSES_PATH = EXPERIMENTS_DIR / "classes.npy"
TEST_DIR = ROOT / "data" / "IAM" / "processed_lines" / "test"
SAMPLE_DIR = ROOT / "notebook" / "sample_images"
IMAGE_H, IMAGE_W = 128, 1024
SEQ_LEN = 5

REGISTER_RUNS = {0: "run_144", 4: "run_146", 8: "run_151", 16: "run_152"}
REG_COLORS = {0: "#1976D2", 4: "#388E3C", 8: "#F57C00", 16: "#D32F2F"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_charset():
    return list(np.load(str(CLASSES_PATH), allow_pickle=True))


def load_model(run_id, device="cpu"):
    run_dir = EXPERIMENTS_DIR / run_id
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    arch_ns = SimpleNamespace(**cfg["arch"])
    charset = load_charset()
    model = HTRNet(arch_ns, len(charset) + 1)
    state = torch.load(str(run_dir / "model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, cfg["arch"].get("num_registers", 0)


def prepare_image(image_path, device="cpu"):
    img_np = load_image(str(image_path))
    img_proc = preprocess(img_np, (IMAGE_H, IMAGE_W))
    tensor = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).float().to(device)
    return tensor


def ctc_decode(logits, charset):
    probs = torch.softmax(logits, dim=-1)[:, 0, :]
    seq = probs.argmax(dim=-1).cpu().numpy()
    chars, peaks = [], []
    prev = -1
    for t, idx in enumerate(seq):
        if idx != 0 and idx != prev:
            ci = idx - 1
            if 0 <= ci < len(charset):
                chars.append(charset[ci])
                peaks.append(t)
        prev = idx
    full = "".join(chars)
    start = 0
    while start < len(full) and full[start] == " ":
        start += 1
    end = len(full)
    while end > start and full[end - 1] == " ":
        end -= 1
    return full[start:end], peaks[start:end]


def compute_entropy(attn_1d):
    """Shannon entropy in bits."""
    p = attn_1d / (attn_1d.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_spearman(char_attns, R, Wp, peaks, n_chars=5):
    """Spearman ρ between character index and attention centroid position."""
    positions = []
    for ci in range(min(n_chars, len(char_attns), len(peaks))):
        attn = char_attns[ci]
        w = attn.sum()
        pos = np.arange(Wp, dtype=np.float64)
        positions.append(float(np.sum(pos * attn) / w) if w > 0 else 0.0)
    if len(positions) < 3:
        return float("nan")
    rho, _ = spearmanr(range(len(positions)), positions)
    return float(rho) if not np.isnan(rho) else 0.0


def count_aligned_heads(attn_maps, R, Wp, peaks, n_chars=5, rho_thresh=0.5):
    """Count heads with Spearman ρ > threshold for character ordering."""
    count = 0
    best_rho = -2.0
    for attn in attn_maps:
        for h in range(attn.shape[1]):
            h_peaks = []
            for ci in range(min(n_chars, len(peaks))):
                t_c = peaks[ci]
                qi = R + t_c
                if qi < attn.shape[2]:
                    row = attn[0, h, qi, R:R + Wp].numpy()
                    h_peaks.append(int(np.argmax(row)))
            if len(h_peaks) >= 3:
                if len(set(h_peaks)) < 2:
                    continue  # constant input
                rho, _ = spearmanr(range(len(h_peaks)), h_peaks)
                if not np.isnan(rho):
                    if rho > best_rho:
                        best_rho = rho
                    if rho > rho_thresh:
                        count += 1
    return count, best_rho


def get_test_images(n_images=10):
    """Get test images with short-to-medium length ground truth."""
    gt_path = TEST_DIR / "gt.txt"
    if not gt_path.exists():
        # Fallback to sample images
        gt_path = SAMPLE_DIR / "gt.txt"
        img_dir = SAMPLE_DIR
    else:
        img_dir = TEST_DIR

    candidates = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            img_id, text = parts
            img_file = img_dir / f"{img_id}.png"
            if img_file.exists() and 3 <= len(text.strip()) <= 30:
                candidates.append((img_file, text.strip()))

    # Sample diverse lengths
    candidates.sort(key=lambda x: len(x[1]))
    step = max(1, len(candidates) // n_images)
    return candidates[::step][:n_images]


def analyze_model(model, num_reg, charset, image_list, device):
    """Run attention analysis on all images for one model."""
    R = num_reg
    all_entropies = []
    all_rhos = []
    all_aligned_heads = []
    all_token_norms_reg = []
    all_token_norms_patch = []

    for img_path, gt_text in image_list:
        tensor = prepare_image(img_path, device)
        with torch.no_grad():
            logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(tensor)
        decoded, peaks = ctc_decode(logits, charset)
        Wp = grid[1]

        if len(peaks) < 2:
            continue

        # Extract per-char attention from last layer using BEST HEAD
        attn_last = attn_maps[-1]  # [B, H, S, S]
        n_heads = attn_last.shape[1]
        n_chars_use = min(SEQ_LEN, len(peaks))

        # Find best-aligned head (highest Spearman ρ)
        best_h, best_h_rho = 0, -2.0
        for h in range(n_heads):
            positions = []
            for ci in range(n_chars_use):
                qi = R + peaks[ci]
                if qi < attn_last.shape[2]:
                    row = attn_last[0, h, qi, R:R + Wp].numpy()
                    pos = np.arange(Wp, dtype=np.float64)
                    w = row.sum()
                    positions.append(float(np.sum(pos * row) / w) if w > 0 else 0.0)
            if len(positions) >= 3 and len(set([round(p) for p in positions])) >= 2:
                rho_h, _ = spearmanr(range(len(positions)), positions)
                if not np.isnan(rho_h) and rho_h > best_h_rho:
                    best_h_rho = rho_h
                    best_h = h

        # Use best head for character attention
        attn_2d = attn_last[0, best_h].numpy()
        char_attns = []
        for ci in range(n_chars_use):
            qi = R + peaks[ci]
            if qi < attn_2d.shape[0]:
                row = attn_2d[qi, R:R + Wp].copy()
                char_attns.append(row)
                all_entropies.append(compute_entropy(row))

        # Best-head Spearman ρ
        if best_h_rho > -2.0:
            all_rhos.append(best_h_rho)

        # Aligned heads
        aligned, _ = count_aligned_heads(attn_maps, R, Wp, peaks, n_chars=SEQ_LEN)
        all_aligned_heads.append(aligned)

        # Token norms
        norms = token_norms[0].numpy()  # [S]
        if R > 0:
            all_token_norms_reg.extend(norms[:R].tolist())
        all_token_norms_patch.extend(norms[R:R+Wp].tolist())

    return {
        "entropies": all_entropies,
        "rhos": all_rhos,
        "aligned_heads": all_aligned_heads,
        "token_norms_reg": all_token_norms_reg,
        "token_norms_patch": all_token_norms_patch,
    }


# ── Figure Generation ─────────────────────────────────────────────────────

def generate_quantitative_figure(results, output_dir):
    """Four-panel quantitative comparison figure."""
    reg_counts = sorted(results.keys())

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    # ── Panel (a): Entropy distribution ──
    ax = axes[0]
    data_ent = [results[nr]["entropies"] for nr in reg_counts]
    bp = ax.boxplot(data_ent, positions=range(len(reg_counts)),
                     patch_artist=True, widths=0.6,
                     medianprops=dict(color="black", linewidth=1.5))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(REG_COLORS[reg_counts[i]])
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(reg_counts)))
    ax.set_xticklabels([f"{nr} reg" for nr in reg_counts])
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("(a) Character Attention Entropy\nlower = more focused", fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    # Add median labels
    for i, nr in enumerate(reg_counts):
        med = np.median(results[nr]["entropies"]) if results[nr]["entropies"] else 0
        ax.text(i, ax.get_ylim()[1] * 0.95, f"med={med:.1f}",
                ha="center", va="top", fontsize=7, fontweight="bold")

    # ── Panel (b): Spatial ordering ρ ──
    ax = axes[1]
    rho_means = [np.mean(results[nr]["rhos"]) if results[nr]["rhos"] else 0
                 for nr in reg_counts]
    rho_stds = [np.std(results[nr]["rhos"]) if results[nr]["rhos"] else 0
                for nr in reg_counts]
    bars = ax.bar(range(len(reg_counts)), rho_means,
                   yerr=rho_stds, capsize=4,
                   color=[REG_COLORS[nr] for nr in reg_counts],
                   edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(reg_counts)))
    ax.set_xticklabels([f"{nr} reg" for nr in reg_counts])
    ax.set_ylabel("Spearman ρ")
    ax.set_title("(b) Spatial Ordering Quality\nhigher = better L→R alignment", fontsize=9)
    ax.set_ylim(-0.3, 1.15)
    ax.axhline(y=1.0, color="gray", ls=":", alpha=0.3)
    ax.grid(axis="y", alpha=0.15)
    for bar, rho_m in zip(bars, rho_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rho_m:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Panel (c): Aligned heads ──
    ax = axes[2]
    head_means = [np.mean(results[nr]["aligned_heads"]) if results[nr]["aligned_heads"] else 0
                  for nr in reg_counts]
    head_stds = [np.std(results[nr]["aligned_heads"]) if results[nr]["aligned_heads"] else 0
                 for nr in reg_counts]
    bars = ax.bar(range(len(reg_counts)), head_means,
                   yerr=head_stds, capsize=4,
                   color=[REG_COLORS[nr] for nr in reg_counts],
                   edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(reg_counts)))
    ax.set_xticklabels([f"{nr} reg" for nr in reg_counts])
    ax.set_ylabel("Count")
    ax.set_title("(c) Aligned Attention Heads\n(ρ > 0.5 per head)", fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    for bar, cnt in zip(bars, head_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{cnt:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Panel (d): Token norm distribution ──
    ax = axes[3]
    positions = []
    data_norms = []
    labels = []
    x_pos = 0
    for nr in reg_counts:
        if results[nr]["token_norms_reg"]:
            positions.append(x_pos)
            data_norms.append(results[nr]["token_norms_reg"])
            labels.append(f"{nr}r\nreg")
            x_pos += 1
        positions.append(x_pos)
        data_norms.append(results[nr]["token_norms_patch"])
        labels.append(f"{nr}r\npatch")
        x_pos += 1
        x_pos += 0.3  # gap between register configs

    bp = ax.boxplot(data_norms, positions=range(len(data_norms)),
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=1.5))
    for i, patch in enumerate(bp["boxes"]):
        # Color by register count
        for nr in reg_counts:
            if f"{nr}r" in labels[i]:
                patch.set_facecolor(REG_COLORS[nr])
                break
        patch.set_alpha(0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("L2 Norm")
    ax.set_title("(d) Token Norm Distribution\nregister vs patch tokens", fontsize=9)
    ax.grid(axis="y", alpha=0.15)

    fig.suptitle("Quantitative Attention Quality — Register Effect",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()

    out = output_dir / "fig3_register_quantitative"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(out) + ext, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}.png / .pdf")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative register effect analysis")
    parser.add_argument("--n_images", type=int, default=10,
                        help="Number of test images to analyze")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT / "outputs" / "pub_figures"))
    parser.add_argument("--runs", nargs=4,
                        help="Run IDs for 0/4/8/16 registers")
    args = parser.parse_args()

    if args.runs:
        for i, nr in enumerate([0, 4, 8, 16]):
            REGISTER_RUNS[nr] = args.runs[i]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charset = load_charset()
    image_list = get_test_images(args.n_images)
    print(f"Analyzing {len(image_list)} images across {len(REGISTER_RUNS)} register configs")

    results = {}
    for nr in sorted(REGISTER_RUNS.keys()):
        print(f"\n  Loading {REGISTER_RUNS[nr]} ({nr} registers)...")
        model, _ = load_model(REGISTER_RUNS[nr], args.device)
        results[nr] = analyze_model(model, nr, charset, image_list, args.device)
        n_ent = len(results[nr]["entropies"])
        n_rho = len(results[nr]["rhos"])
        avg_rho = np.mean(results[nr]["rhos"]) if results[nr]["rhos"] else 0
        print(f"    {n_ent} entropy samples, {n_rho} ρ samples, avg ρ={avg_rho:.3f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    generate_quantitative_figure(results, output_dir)


if __name__ == "__main__":
    main()
