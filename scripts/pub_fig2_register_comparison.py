#!/usr/bin/env python3
"""
Figure 2: Register Effect on Character Attention (Comparison Grid)
==================================================================

Rows = register configurations (0, 4, 8, 16)
Columns = [Input | Char_1 | Char_2 | ... | Char_5]

Directly shows how register tokens sharpen character-level attention.
Each model was trained independently with different num_registers.

Quantitative annotations:
  - Entropy (H) per cell: lower = more focused attention
  - Spearman ρ per row: spatial ordering quality (1.0 = perfect L→R)

Usage:
    python scripts/pub_fig2_register_comparison.py
    python scripts/pub_fig2_register_comparison.py --image path/to/img.png --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models import HTRNet
from utils.preprocessing import load_image, preprocess

EXPERIMENTS = ROOT / "saved_models" / "experiments"
CLASSES_PATH = EXPERIMENTS / "classes.npy"
IAM_TEST = ROOT / "data" / "IAM" / "processed_lines" / "test"
IMAGE_H, IMAGE_W = 128, 1024
SEQ_LEN = 5

REGISTER_RUNS = {0: "run_144", 4: "run_146", 8: "run_151", 16: "run_152"}
REG_COLORS = {0: "#1976D2", 4: "#388E3C", 8: "#F57C00", 16: "#D32F2F"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def load_charset():
    return list(np.load(str(CLASSES_PATH), allow_pickle=True))


def load_model(run_id, device="cpu"):
    run_dir = EXPERIMENTS / run_id
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    arch_ns = SimpleNamespace(**cfg["arch"])
    charset = load_charset()
    model = HTRNet(arch_ns, len(charset) + 1)
    state = torch.load(str(run_dir / "model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, cfg["arch"].get("num_registers", 0)


def prepare_image(img_path, device="cpu"):
    img_np = load_image(str(img_path))
    img_proc = preprocess(img_np, (IMAGE_H, IMAGE_W))
    tensor = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).float().to(device)
    col_var = np.var(img_proc, axis=0)
    thr = np.median(col_var) * 0.1 + 1e-6
    active = np.where(col_var > thr)[0]
    xs = max(0, active[0] - 4) if len(active) > 0 else 0
    xe = min(IMAGE_W, active[-1] + 4) if len(active) > 0 else IMAGE_W
    return tensor, img_proc, (xs, xe)


def ctc_decode(logits, charset):
    seq = torch.softmax(logits, dim=-1)[:, 0, :].argmax(dim=-1).cpu().numpy()
    chars, peaks = [], []
    prev = -1
    for t, idx in enumerate(seq):
        if idx != 0 and idx != prev:
            ci = idx - 1
            if 0 <= ci < len(charset):
                chars.append(charset[ci])
                peaks.append(t)
        prev = idx
    return "".join(chars), peaks


def shallow_rollout(attn_maps, n_layers=2):
    layers = attn_maps[-n_layers:]
    rollout = None
    for attn in layers:
        a = attn.mean(dim=1).numpy()[0]
        a = 0.5 * a + 0.5 * np.eye(a.shape[0])
        rollout = a if rollout is None else rollout @ a
    rollout /= rollout.sum(axis=-1, keepdims=True)
    return rollout


def extract_char_attention(attn_maps, R, Wp, peaks, seq_len=SEQ_LEN):
    rollout = shallow_rollout(attn_maps)
    char_attns = []
    for i in range(seq_len):
        if i < len(peaks):
            qi = R + peaks[i]
            row = rollout[qi, R:R + Wp].copy() if qi < rollout.shape[0] else np.zeros(Wp)
        else:
            row = np.zeros(Wp)
        char_attns.append(row)
    return char_attns


def attn_to_heatmap(attn_1d, Wp, text_bbox, sigma=1.5, gamma=0.5):
    xs, xe = text_bbox
    attn_full = np.interp(np.linspace(0, 1, IMAGE_W), np.linspace(0, 1, Wp), attn_1d)
    attn_crop = attn_full[xs:xe].astype(np.float64)
    if sigma > 0 and len(attn_crop) > 1:
        attn_crop = gaussian_filter1d(attn_crop, sigma=sigma)
    p2, p98 = np.percentile(attn_crop, 2), np.percentile(attn_crop, 98)
    if p98 > p2:
        attn_crop = np.clip((attn_crop - p2) / (p98 - p2), 0, 1)
    else:
        attn_crop = (attn_crop - attn_crop.min()) / (attn_crop.max() - attn_crop.min() + 1e-8)
    attn_crop = np.power(attn_crop, gamma)
    attn_crop = (attn_crop - attn_crop.min()) / (attn_crop.max() - attn_crop.min() + 1e-8)
    return np.tile(attn_crop, (IMAGE_H, 1))


def compute_entropy(attn_1d):
    p = attn_1d / (attn_1d.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_spearman(char_attns, Wp):
    positions = []
    for ca in char_attns:
        if ca.sum() > 1e-10:
            positions.append(float(np.sum(np.arange(Wp) * ca) / ca.sum()))
        else:
            positions.append(0.0)
    valid = [i for i, ca in enumerate(char_attns) if ca.sum() > 1e-10]
    if len(valid) < 3:
        return 0.0
    rho, _ = spearmanr([i for i in valid], [positions[i] for i in valid])
    return float(rho) if not np.isnan(rho) else 0.0


def find_good_image():
    """Find a test image with 4-6 characters."""
    gt_path = IAM_TEST / "gt.txt"
    if not gt_path.exists():
        return None, None
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                img_id, text = parts[0], parts[1].strip()
                if 4 <= len(text) <= 6 and (IAM_TEST / f"{img_id}.png").exists():
                    return IAM_TEST / f"{img_id}.png", text
    return None, None


def generate_figure(models, charset, img_path, gt_text, device, output_path):
    """Generate the register comparison figure."""
    tensor, img_proc, text_bbox = prepare_image(img_path, device)
    xs, xe = text_bbox
    img_crop = img_proc[:, xs:xe]

    reg_counts = sorted(models.keys())
    n_rows = len(reg_counts)
    n_cols = SEQ_LEN + 1  # Input + characters

    # Extract attention for all models
    all_data = {}
    for nr in reg_counts:
        model = models[nr]
        with torch.no_grad():
            logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(tensor)
        decoded, peaks = ctc_decode(logits, charset)
        R = 0 if reg_tokens is None else reg_tokens.shape[1]
        Wp = grid[1]
        char_attns = extract_char_attention(attn_maps, R, Wp, peaks, SEQ_LEN)
        rho = compute_spearman(char_attns, Wp)
        all_data[nr] = {
            "decoded": decoded, "peaks": peaks, "char_attns": char_attns,
            "Wp": Wp, "R": R, "rho": rho,
        }
        print(f"  {nr:>2} reg: \"{decoded[:SEQ_LEN]}\" ρ={rho:.2f}")

    # Figure
    aspect = (xe - xs) / IMAGE_H
    cell_w = max(1.8, aspect * 1.0)
    cell_h = 1.2
    fig_w = cell_w * n_cols + 1.8
    fig_h = cell_h * n_rows + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        wspace=0.03, hspace=0.20,
        left=0.10, right=0.90, top=0.86, bottom=0.04,
    )

    cmap = plt.cm.inferno

    for ri, nr in enumerate(reg_counts):
        d = all_data[nr]
        decoded = d["decoded"]
        chars = list(decoded[:SEQ_LEN])
        while len(chars) < SEQ_LEN:
            chars.append("—")

        # Column 0: input image
        ax = fig.add_subplot(gs[ri, 0])
        ax.imshow(img_crop, cmap="gray", aspect="auto")
        ax.set_ylabel(f"R={nr}", fontsize=10, fontweight="bold",
                       rotation=0, labelpad=28, va="center")
        ax.set_xticks([]); ax.set_yticks([])
        # ρ annotation
        ax.text(0.02, 0.05, f"ρ={d['rho']:.2f}", transform=ax.transAxes,
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc=REG_COLORS[nr], alpha=0.8))
        if ri == 0:
            ax.set_title("Input", fontsize=10, fontweight="bold", pad=4)

        # Character columns
        for ci in range(SEQ_LEN):
            ax = fig.add_subplot(gs[ri, ci + 1])
            ca = d["char_attns"][ci]
            is_padding = ci >= len(d["peaks"]) or ca.sum() < 1e-10

            if is_padding:
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.15)
                ax.text(0.5, 0.5, "pad", transform=ax.transAxes,
                        fontsize=7, color="gray", ha="center", va="center", style="italic")
            else:
                hm = attn_to_heatmap(ca, d["Wp"], text_bbox)
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)
                rgba = cmap(hm)
                rgba[..., 3] = hm * 0.85
                ax.imshow(rgba, aspect="auto")
                # Entropy
                ent = compute_entropy(ca)
                ax.text(0.02, 0.95, f"H={ent:.1f}", transform=ax.transAxes,
                        fontsize=6, color="white", va="top",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                label = chars[ci] if chars[ci] != " " else "⎵"
                ax.set_title(f"'{label}'", fontsize=10, fontweight="bold", pad=4)

    # Title
    fig.suptitle(
        f"Register Effect on Character Attention\n\"{gt_text}\"",
        fontsize=12, fontweight="bold", y=0.95,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.65])
    fig.colorbar(sm, cax=cax, label="Attention")

    for ext in [".png", ".pdf"]:
        out = output_path.with_suffix(ext)
        fig.savefig(str(out), facecolor="white")
        print(f"  Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fig 2: Register comparison on character attention")
    parser.add_argument("--image", help="Specific image path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "pub_figures"))
    parser.add_argument("--runs", nargs=4, help="Run IDs for 0,4,8,16 registers")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.runs:
        for i, nr in enumerate([0, 4, 8, 16]):
            REGISTER_RUNS[nr] = args.runs[i]

    charset = load_charset()
    models = {}
    for nr, run_id in REGISTER_RUNS.items():
        models[nr], _ = load_model(run_id, args.device)
        print(f"Loaded {run_id} ({nr} registers)")

    if args.image:
        img_path, gt_text = Path(args.image), "?"
    else:
        img_path, gt_text = find_good_image()
        if img_path is None:
            print("No suitable image found. Use --image.")
            return
    print(f"Image: {img_path.name}, GT: \"{gt_text}\"")

    generate_figure(
        models, charset, img_path, gt_text,
        args.device, output_dir / "fig2_register_comparison",
    )
    print("Done.")


if __name__ == "__main__":
    main()
