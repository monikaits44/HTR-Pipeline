#!/usr/bin/env python3
"""
Publication Figure 1: Character-Level Attention Grid (Paper Fig. 5 Style)
=========================================================================
Reproduces the visualization from "Beyond Memorization" Fig. 5, adapted
for CTC-based ViT-RGTS with self-attention.

Layout:
  Rows = different input words (3-4 words)
  Columns = [Input Image, Char_1, Char_2, ..., Char_L]  (L=5, padded)

Each cell = handwriting image overlaid with character-specific attention heatmap.
Words with < L characters have grayed-out padding columns.

Method:
  1. Forward pass with forward_explain → per-layer attention [B, H, S, S]
  2. CTC greedy decode → character string + peak timesteps
  3. Per-character attention = last-layer attention row at CTC peak, patch columns only
  4. Select best-aligned head (highest Spearman ρ) instead of head-average
  5. Interpolate 1D attention → 2D heatmap overlay

Usage:
  python scripts/pub_fig5_character_attention.py
  python scripts/pub_fig5_character_attention.py --device cuda:0
  python scripts/pub_fig5_character_attention.py --run_id run_146 --num_registers 4
  python scripts/pub_fig5_character_attention.py --images path1.png path2.png
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

# ── Constants ─────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = ROOT / "saved_models" / "experiments"
SAMPLE_DIR = ROOT / "notebook" / "sample_images"
CLASSES_PATH = EXPERIMENTS_DIR / "classes.npy"
IMAGE_H, IMAGE_W = 128, 1024
SEQ_LEN = 5  # Fixed character display length (padded)

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


# ── Utilities ─────────────────────────────────────────────────────────────

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
    """Load and preprocess image. Returns tensor, numpy array, and text bounding box."""
    img_np = load_image(str(image_path))
    img_proc = preprocess(img_np, (IMAGE_H, IMAGE_W))
    tensor = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).float().to(device)
    # Detect active (non-background) region
    col_var = np.var(img_proc, axis=0)
    thr = np.median(col_var) * 0.1 + 1e-6
    active = np.where(col_var > thr)[0]
    if len(active) > 0:
        xs, xe = max(0, active[0] - 4), min(IMAGE_W, active[-1] + 4)
    else:
        xs, xe = 0, IMAGE_W
    return tensor, img_proc, (xs, xe)


def ctc_decode(logits, charset):
    """Greedy CTC decode → (string, list of peak timesteps). Strips leading/trailing spaces."""
    probs = torch.softmax(logits, dim=-1)[:, 0, :]  # [T, nclasses]
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
    # Strip leading/trailing spaces and adjust peaks
    start = 0
    while start < len(full) and full[start] == " ":
        start += 1
    end = len(full)
    while end > start and full[end - 1] == " ":
        end -= 1
    return full[start:end], peaks[start:end]


def find_best_head(attn_layer, R, Wp, peaks, n_chars):
    """Find the attention head with best spatial ordering (highest Spearman ρ)."""
    n_heads = attn_layer.shape[1]
    best_rho, best_h = -2.0, 0
    for h in range(n_heads):
        positions = []
        for ci in range(min(n_chars, len(peaks))):
            qi = R + peaks[ci]
            if qi >= attn_layer.shape[2]:
                continue
            row = attn_layer[0, h, qi, R:R + Wp].numpy()
            pos = np.arange(Wp, dtype=np.float64)
            w = row.sum()
            positions.append(float(np.sum(pos * row) / w) if w > 0 else 0.0)
        if len(positions) >= 2:
            rho, _ = spearmanr(range(len(positions)), positions)
            if not np.isnan(rho) and rho > best_rho:
                best_rho = rho
                best_h = h
    return best_h, best_rho


def extract_character_attention(attn_maps, R, Wp, peaks, head_idx=None):
    """
    Extract per-character 1D attention from the last transformer layer.

    If head_idx is specified, use that head only (sharpest maps).
    Otherwise, average across all heads.

    Returns: list of 1D numpy arrays, one per CTC peak.
    """
    attn = attn_maps[-1]  # Last layer: [B, H, S, S]
    if head_idx is not None:
        attn_2d = attn[0, head_idx].numpy()  # [S, S]
    else:
        attn_2d = attn[0].mean(dim=0).numpy()  # [S, S] head-averaged

    results = []
    for t_c in peaks:
        qi = R + t_c
        if qi >= attn_2d.shape[0]:
            results.append(np.zeros(Wp))
            continue
        row = attn_2d[qi, R:R + Wp].copy()
        results.append(row)
    return results


def attn_to_2d_heatmap(attn_1d, Wp, img_h=IMAGE_H, img_w=IMAGE_W,
                        text_bbox=None, sigma=1.5, gamma=0.5):
    """Convert 1D patch attention to a 2D image-sized heatmap."""
    # Interpolate to full image width
    attn_full = np.interp(np.linspace(0, 1, img_w),
                           np.linspace(0, 1, Wp), attn_1d)
    # Crop to text region if provided
    if text_bbox:
        xs, xe = text_bbox
        attn_crop = attn_full[xs:xe].astype(np.float64)
    else:
        attn_crop = attn_full.astype(np.float64)
        xs, xe = 0, img_w

    # Gaussian smooth
    if sigma > 0 and len(attn_crop) > 1:
        attn_crop = gaussian_filter1d(attn_crop, sigma=sigma)

    # Percentile normalization
    p2, p98 = np.percentile(attn_crop, 2), np.percentile(attn_crop, 98)
    if p98 > p2:
        attn_crop = np.clip((attn_crop - p2) / (p98 - p2), 0, 1)
    else:
        mn, mx = attn_crop.min(), attn_crop.max()
        attn_crop = (attn_crop - mn) / (mx - mn + 1e-8)

    # Gamma correction
    attn_crop = np.power(attn_crop, gamma)
    attn_crop = (attn_crop - attn_crop.min()) / (attn_crop.max() - attn_crop.min() + 1e-8)

    # Tile to 2D
    return np.tile(attn_crop, (img_h, 1))


def find_sample_words(gt_path, img_dir, min_len=3, max_len=7, n=4):
    """Find short words from ground truth for visualization."""
    if not gt_path.exists():
        return []
    candidates = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            img_id, text = parts
            t = text.strip()
            img_file = img_dir / f"{img_id}.png"
            if min_len <= len(t) <= max_len and img_file.exists():
                candidates.append((img_file, t))
    # Prefer words close to SEQ_LEN in length
    candidates.sort(key=lambda x: abs(len(x[1]) - SEQ_LEN))
    return candidates[:n]


# ── FIGURE: Paper Fig. 5 Style ───────────────────────────────────────────

def generate_fig5(model, num_reg, charset, image_list, device, output_path):
    """
    Generate Paper Fig. 5 style character attention grid.

    Rows = words, Columns = [Input, c_1, c_2, ..., c_L, PAD...]
    L = SEQ_LEN (fixed at 5, padded for shorter words)
    """
    R = num_reg
    n_rows = len(image_list)
    n_cols = SEQ_LEN + 1  # input + L character columns
    cmap = plt.cm.inferno

    # Collect data for all images
    all_data = []
    for img_path, gt_text in image_list:
        tensor, img_proc, text_bbox = prepare_image(img_path, device)
        with torch.no_grad():
            logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(tensor)
        decoded, peaks = ctc_decode(logits, charset)
        Wp = grid[1]

        # Find best-aligned head
        n_chars = min(len(decoded), SEQ_LEN)
        best_head, best_rho = find_best_head(attn_maps[-1], R, Wp, peaks[:n_chars], n_chars)

        # Extract character attention using best head
        char_attns = extract_character_attention(attn_maps, R, Wp, peaks[:SEQ_LEN], head_idx=best_head)

        all_data.append({
            "img_path": img_path,
            "gt_text": gt_text,
            "decoded": decoded,
            "img_proc": img_proc,
            "text_bbox": text_bbox,
            "char_attns": char_attns,
            "Wp": Wp,
            "best_head": best_head,
            "best_rho": best_rho,
        })

    # ── Layout ────────────────────────────────────────────────────────
    cell_w, cell_h = 2.2, 1.4
    fig_w = cell_w * n_cols + 1.5
    fig_h = cell_h * n_rows + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           wspace=0.04, hspace=0.30,
                           left=0.08, right=0.90, top=0.88, bottom=0.04)

    for ri, d in enumerate(all_data):
        xs, xe = d["text_bbox"]
        img_crop = d["img_proc"][:, xs:xe]
        n_actual_chars = min(len(d["decoded"]), SEQ_LEN)

        # Column 0: Input image
        ax = fig.add_subplot(gs[ri, 0])
        ax.imshow(img_crop, cmap="gray", aspect="auto")
        # Show decoded text below
        display_text = d["decoded"][:SEQ_LEN]
        ax.set_xlabel(f'"{display_text}"', fontsize=7, labelpad=2)
        ax.set_ylabel(f"Word {ri+1}", fontsize=9, fontweight="bold",
                       rotation=0, labelpad=35, va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        if ri == 0:
            ax.set_title("Input", fontsize=10, fontweight="bold", pad=4)

        # Character columns
        for ci in range(SEQ_LEN):
            ax = fig.add_subplot(gs[ri, 1 + ci])

            if ci < n_actual_chars and ci < len(d["char_attns"]):
                # Active character — show attention overlay
                hm = attn_to_2d_heatmap(d["char_attns"][ci], d["Wp"],
                                         text_bbox=d["text_bbox"])
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)
                rgba = cmap(hm)
                rgba[..., 3] = hm * 0.85
                ax.imshow(rgba, aspect="auto")

                # Character label on top row
                if ri == 0:
                    ch = d["decoded"][ci] if ci < len(d["decoded"]) else "?"
                    label = ch if ch != " " else "⎵"
                    ax.set_title(f"$A_{{{ci+1}}}$  '{label}'", fontsize=9,
                                 fontweight="bold", pad=4)
            else:
                # Padded position — gray out
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.15)
                ax.text(0.5, 0.5, "PAD", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8,
                        color="gray", fontstyle="italic")
                if ri == 0:
                    ax.set_title(f"$A_{{{ci+1}}}$  (pad)", fontsize=9,
                                 color="gray", pad=4)

            ax.set_xticks([])
            ax.set_yticks([])

    # Suptitle
    fig.suptitle(
        f"Character Attention Maps — Fixed Sequence Length L={SEQ_LEN}\n"
        f"(CTC-aligned, best-head selection, {R} register tokens)",
        fontsize=11, fontweight="bold", y=0.97)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.65])
    fig.colorbar(sm, cax=cax, label="Attention")

    for ext in [".png", ".pdf"]:
        fig.savefig(str(output_path) + ext, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}.png / .pdf")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fig. 5 style character attention maps (fixed L=5, padded)")
    parser.add_argument("--run_id", type=str, default="run_146",
                        help="Experiment run directory name")
    parser.add_argument("--num_registers", type=int, default=None,
                        help="Override register count (auto-detected from config)")
    parser.add_argument("--images", nargs="+", help="Image paths to visualize")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT / "outputs" / "pub_figures"))
    parser.add_argument("--seq_len", type=int, default=5,
                        help="Fixed sequence display length L")
    args = parser.parse_args()

    global SEQ_LEN
    SEQ_LEN = args.seq_len

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charset = load_charset()
    model, auto_reg = load_model(args.run_id, args.device)
    num_reg = args.num_registers if args.num_registers is not None else auto_reg
    print(f"Model: {args.run_id}, registers: {num_reg}, device: {args.device}")

    # Resolve images
    if args.images:
        image_list = [(Path(p), "") for p in args.images]
    else:
        # Use sample images (short words)
        gt_path = SAMPLE_DIR / "gt.txt"
        image_list = find_sample_words(gt_path, SAMPLE_DIR, min_len=3, max_len=8, n=4)
        if not image_list:
            # Fallback to test set
            gt_path = ROOT / "data" / "IAM" / "processed_lines" / "test" / "gt.txt"
            test_dir = ROOT / "data" / "IAM" / "processed_lines" / "test"
            image_list = find_sample_words(gt_path, test_dir, min_len=3, max_len=8, n=4)

    if not image_list:
        print("ERROR: No images found. Provide --images or ensure sample_images/ exists.")
        sys.exit(1)

    print(f"Images: {[str(p.name) for p, _ in image_list]}")

    generate_fig5(model, num_reg, charset, image_list, args.device,
                  output_dir / f"fig5_char_attention_{args.run_id}")


if __name__ == "__main__":
    main()
