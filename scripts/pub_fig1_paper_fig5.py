#!/usr/bin/env python3
"""
Figure 1: Character-Level Attention Maps (Paper Fig. 5 Style)
=============================================================

Reproduces the style of Fig. 5 from "Beyond Memorization: Training-Free Style
Mixing for Variability in Handwritten Text Generation" (ICDAR 2025).

Layout:
  - Rows = different words (3-4 test images)
  - Columns = [Full Word | Char_1 | Char_2 | ... | Char_L]
  - L fixed to 5 (pad shorter words, truncate longer)
  - Each cell: handwriting image with character-specific attention heatmap overlay
  - Attention: shallow rollout (last 2 transformer layers) → interpolated to image

Key difference from paper:
  - Paper uses cross-attention from a diffusion model
  - We use self-attention rollout from ViT-RGTS with CTC peak decoding
  - Same visual effect: each character's attention highlights its spatial position

Usage:
    python scripts/pub_fig1_paper_fig5.py
    python scripts/pub_fig1_paper_fig5.py --run run_68 --device cuda:0
    python scripts/pub_fig1_paper_fig5.py --output ./my_figures
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models import HTRNet
from utils.preprocessing import load_image, preprocess

EXPERIMENTS = ROOT / "saved_models" / "experiments"
CLASSES_PATH = EXPERIMENTS / "classes.npy"
IAM_TEST = ROOT / "data" / "IAM" / "processed_lines" / "test"
IMAGE_H, IMAGE_W = 128, 1024
SEQ_LEN = 5  # Fixed number of character columns (pad if shorter)

# Publication style
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
    "text.usetex": False,
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
    # Detect text bounding box (active columns)
    col_var = np.var(img_proc, axis=0)
    thr = np.median(col_var) * 0.1 + 1e-6
    active = np.where(col_var > thr)[0]
    if len(active) > 0:
        xs, xe = max(0, active[0] - 4), min(IMAGE_W, active[-1] + 4)
    else:
        xs, xe = 0, IMAGE_W
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
    """Attention rollout over last n layers with residual."""
    layers = attn_maps[-n_layers:]
    rollout = None
    for attn in layers:
        a = attn.mean(dim=1).numpy()[0]  # [S, S] head-averaged
        a = 0.5 * a + 0.5 * np.eye(a.shape[0])  # residual
        rollout = a if rollout is None else rollout @ a
    rollout /= rollout.sum(axis=-1, keepdims=True)
    return rollout


def extract_char_attention(attn_maps, R, Wp, peaks, seq_len=SEQ_LEN):
    """Extract per-character 1D attention, padded to seq_len."""
    rollout = shallow_rollout(attn_maps)
    char_attns = []
    for i in range(seq_len):
        if i < len(peaks):
            t_c = peaks[i]
            qi = R + t_c
            if qi < rollout.shape[0]:
                row = rollout[qi, R:R + Wp].copy()
            else:
                row = np.zeros(Wp)
        else:
            # Padding: zeros (no attention for padding positions)
            row = np.zeros(Wp)
        char_attns.append(row)
    return char_attns


def attn_to_heatmap(attn_1d, Wp, text_bbox, sigma=1.5, gamma=0.5):
    """Convert 1D attention → 2D heatmap cropped to text region."""
    xs, xe = text_bbox
    attn_full = np.interp(
        np.linspace(0, 1, IMAGE_W), np.linspace(0, 1, Wp), attn_1d
    )
    attn_crop = attn_full[xs:xe].astype(np.float64)
    if sigma > 0 and len(attn_crop) > 1:
        attn_crop = gaussian_filter1d(attn_crop, sigma=sigma)
    p2, p98 = np.percentile(attn_crop, 2), np.percentile(attn_crop, 98)
    if p98 > p2:
        attn_crop = np.clip((attn_crop - p2) / (p98 - p2), 0, 1)
    else:
        mn = attn_crop.min()
        attn_crop = (attn_crop - mn) / (attn_crop.max() - mn + 1e-8)
    attn_crop = np.power(attn_crop, gamma)
    attn_crop = (attn_crop - attn_crop.min()) / (attn_crop.max() - attn_crop.min() + 1e-8)
    return np.tile(attn_crop, (IMAGE_H, 1))


def find_test_images(n=4, min_len=3, max_len=7):
    """Find n test images with short ground truth text."""
    gt_path = IAM_TEST / "gt.txt"
    if not gt_path.exists():
        return []
    candidates = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            img_id, text = parts[0], parts[1].strip()
            img_file = IAM_TEST / f"{img_id}.png"
            if min_len <= len(text) <= max_len and img_file.exists():
                candidates.append((img_file, text))
    # Sort by length closest to SEQ_LEN, then alphabetically for reproducibility
    candidates.sort(key=lambda x: (abs(len(x[1]) - SEQ_LEN), x[1]))
    return candidates[:n]


def generate_figure(model, charset, image_paths_texts, R, Wp_ref, device, output_path):
    """
    Generate Paper Fig. 5 style figure.
    
    Rows = words, Cols = [A (full word) | A_c1 | A_c2 | ... | A_c5]
    Where A is the complete word-level attention map and A_c is per-character.
    """
    n_rows = len(image_paths_texts)
    n_cols = SEQ_LEN + 1  # Full word attention + SEQ_LEN character columns

    # Compute all attention data first
    all_data = []
    for img_path, gt_text in image_paths_texts:
        tensor, img_proc, text_bbox = prepare_image(img_path, device)
        with torch.no_grad():
            logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(tensor)
        decoded, peaks = ctc_decode(logits, charset)
        Wp = grid[1]
        R_actual = 0 if reg_tokens is None else reg_tokens.shape[1]
        char_attns = extract_char_attention(attn_maps, R_actual, Wp, peaks, SEQ_LEN)

        # Full word attention: average over all character positions
        rollout = shallow_rollout(attn_maps)
        word_attn = np.zeros(Wp)
        for t_c in peaks[:SEQ_LEN]:
            qi = R_actual + t_c
            if qi < rollout.shape[0]:
                word_attn += rollout[qi, R_actual:R_actual + Wp]
        if len(peaks) > 0:
            word_attn /= min(len(peaks), SEQ_LEN)

        all_data.append({
            "img_proc": img_proc, "text_bbox": text_bbox,
            "decoded": decoded, "gt": gt_text, "peaks": peaks,
            "char_attns": char_attns, "word_attn": word_attn,
            "Wp": Wp, "R": R_actual,
        })

    # Figure layout
    xs_min = min(d["text_bbox"][0] for d in all_data)
    xe_max = max(d["text_bbox"][1] for d in all_data)
    aspect = (xe_max - xs_min) / IMAGE_H
    cell_w = max(1.5, aspect * 0.9)
    cell_h = 1.1
    fig_w = cell_w * n_cols + 0.8
    fig_h = cell_h * n_rows + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        wspace=0.03, hspace=0.15,
        left=0.02, right=0.93, top=0.88, bottom=0.03,
    )

    cmap = plt.cm.inferno

    for ri, d in enumerate(all_data):
        img_crop = d["img_proc"][:, d["text_bbox"][0]:d["text_bbox"][1]]
        decoded = d["decoded"]
        chars = list(decoded[:SEQ_LEN])
        # Pad character labels
        while len(chars) < SEQ_LEN:
            chars.append("—")

        # Column 0: Full word attention (A)
        ax = fig.add_subplot(gs[ri, 0])
        hm = attn_to_heatmap(d["word_attn"], d["Wp"], d["text_bbox"])
        ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)
        rgba = cmap(hm)
        rgba[..., 3] = hm * 0.8
        ax.imshow(rgba, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(r"$\mathbf{A}$", fontsize=11, pad=4)
        # Word label on left
        ax.text(-0.05, 0.5, f'"{decoded[:SEQ_LEN]}"', transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="center", ha="right", rotation=90)

        # Columns 1..SEQ_LEN: Per-character attention (A_c)
        for ci in range(SEQ_LEN):
            ax = fig.add_subplot(gs[ri, ci + 1])
            ca = d["char_attns"][ci]
            is_padding = ci >= len(d["peaks"])

            if is_padding or ca.sum() < 1e-10:
                # Padding position: show faded image
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.2)
                ax.text(0.5, 0.5, "pad", transform=ax.transAxes,
                        fontsize=7, color="gray", ha="center", va="center", style="italic")
            else:
                hm = attn_to_heatmap(ca, d["Wp"], d["text_bbox"])
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)
                rgba = cmap(hm)
                rgba[..., 3] = hm * 0.85
                ax.imshow(rgba, aspect="auto")

            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                label = chars[ci] if chars[ci] != " " else "⎵"
                ax.set_title(f"$\\mathbf{{A}}_{{\\mathrm{{{label}}}}}$",
                             fontsize=11, pad=4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.945, 0.12, 0.012, 0.7])
    fig.colorbar(sm, cax=cax, label="Attention")

    # Save
    for ext in [".png", ".pdf"]:
        out = output_path.with_suffix(ext)
        fig.savefig(str(out), facecolor="white")
        print(f"  Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fig 1: Paper Fig.5 style character attention")
    parser.add_argument("--run", default="run_68", help="Experiment run ID (best model)")
    parser.add_argument("--images", nargs="+", help="Specific image paths")
    parser.add_argument("--n_words", type=int, default=4, help="Number of word rows")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "pub_figures"))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    charset = load_charset()
    model, num_reg = load_model(args.run, args.device)
    print(f"Model: {args.run} ({num_reg} registers)")

    if args.images:
        image_paths_texts = [(Path(p), "?") for p in args.images]
    else:
        image_paths_texts = find_test_images(n=args.n_words)
        if not image_paths_texts:
            print("No suitable test images found. Use --images.")
            return
        print(f"Selected {len(image_paths_texts)} images:")
        for p, t in image_paths_texts:
            print(f"  {p.name}: '{t}'")

    # Get model info for figure
    with torch.no_grad():
        dummy_tensor = torch.zeros(1, 1, IMAGE_H, IMAGE_W).to(args.device)
        _, reg, _, _, grid = model.forward_explain(dummy_tensor)
    R = 0 if reg is None else reg.shape[1]
    Wp = grid[1]

    generate_figure(
        model, charset, image_paths_texts, R, Wp,
        args.device, output_dir / "fig1_char_attention_maps",
    )
    print("Done.")


if __name__ == "__main__":
    main()
