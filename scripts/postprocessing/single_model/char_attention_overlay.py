#!/usr/bin/env python3
"""
Per-character attention overlay grid — exactly like Fig. 5 reference.

Rows  = images (different handwritten words/lines)
Cols  = Char 1, Char 2, …, Char N
Cells = original image with spatial attention heatmap overlay

Usage:
    python scripts/postprocessing/char_attention_overlay.py \
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
        --model-path saved_models/experiments/run_55/model.pt \
        --save-dir visualizations/char_attention_overlay \
        -- notebook/sample_images/a06-110-08.png notebook/sample_images/a02-057-08.png
"""

import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models import HTRNet
from utils.preprocessing import load_image, preprocess
from omegaconf import OmegaConf


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+", help="YAML config files")
    p.add_argument("--model-path", required=True)
    p.add_argument("--save-dir", default="visualizations/char_attention_overlay")
    p.add_argument("--max-chars", type=int, default=0, help="0 = all")
    p.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    p.add_argument("--dpi", type=int, default=150)

    # split on '--'
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        main_args = argv[:sep]
        image_args = argv[sep + 1:]
    else:
        main_args, image_args = argv, []

    args = p.parse_args(main_args)
    args.images = [p for p in image_args if os.path.isfile(p)]
    if not args.images:
        p.error("Provide image paths after '--'")
    return args


# ─────────────────────────────────────────────────────────────────────────────
def ctc_decode(logits_np, i2c, blank=0):
    """Greedy CTC decode → (chars, positions)."""
    preds = logits_np.argmax(2).squeeze()  # [T]
    chars, pos = [], []
    prev = -1
    for t, v in enumerate(preds):
        v = int(v)
        if v != blank and v != prev:
            chars.append(str(i2c.get(v, "?")))
            pos.append(t)
        prev = v
    return chars, pos


def upscale(patch_attn_1d, Hp, Wp, fixed_h, fixed_w, border=8):
    """1-D patch attention [Wp] → image-sized heatmap [H_content, W_content]."""
    grid = patch_attn_1d.reshape(Hp, Wp)            # (1, 128) typically
    ph, pw = fixed_h // Hp, fixed_w // Wp
    heat = np.kron(grid, np.ones((ph, pw)))          # (fixed_h, fixed_w)
    content = heat[border:fixed_h - border, border:fixed_w - border]
    return content


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Config ───────────────────────────────────────────────────────────
    conf = OmegaConf.merge(*[OmegaConf.load(c) for c in args.configs])
    OmegaConf.set_struct(conf, False)

    run_dir = Path(args.model_path).parent
    with open(run_dir / "config.json") as f:
        run_cfg = json.load(f)
    n_reg = run_cfg.get("arch", {}).get("num_registers", 0)
    conf.arch.num_registers = n_reg

    classes = np.load(os.path.join(conf.data.path, "classes.npy"))
    num_classes = len(classes) + 1
    i2c = {i + 1: c for i, c in enumerate(classes)}
    fixed_h, fixed_w = conf.preproc.image_height, conf.preproc.image_width

    # ── Model ────────────────────────────────────────────────────────────
    net = HTRNet(deepcopy(conf.arch), num_classes)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt, strict=True)
    net.to(device).eval()

    run_name = run_dir.name
    label = f"Reg-{n_reg}"
    print(f"Model: {label} ({run_name})")

    # ── Per-image: extract chars + per-char spatial attention ────────────
    rows = []  # list of (image_name, raw_img, chars, heatmaps)
    for img_path in args.images:
        img_name = Path(img_path).stem
        raw = load_image(img_path)
        processed = preprocess(raw, (fixed_h, fixed_w))
        x = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, reg_out, attn_maps, _, (Hp, Wp) = net.forward_explain(x)

        chars, positions = ctc_decode(logits.cpu().numpy(), i2c)
        if not chars:
            print(f"  {img_name}: no characters decoded, skipping")
            continue

        # Middle layer, head-averaged attention
        mid = len(attn_maps) // 2
        attn = attn_maps[mid][0].mean(dim=0).cpu().numpy()  # [S, S]
        R = n_reg
        T = Wp  # patch tokens

        # Per-character spatial heatmaps
        heatmaps = []
        for t in positions:
            # attention row for patch token t over all patch tokens
            row = attn[R + t, R:R + T]  # [T]
            # normalize to [0, 1]
            mn, mx = row.min(), row.max()
            if mx > mn:
                row = (row - mn) / (mx - mn)
            heat = upscale(row, Hp, Wp, fixed_h, fixed_w)
            heatmaps.append(heat)

        mc = args.max_chars if args.max_chars > 0 else len(chars)
        chars = chars[:mc]
        heatmaps = heatmaps[:mc]

        # raw image for display (crop to content region matching heatmap)
        h_content = fixed_h - 2 * 8
        w_content = fixed_w - 2 * 8
        # Resize raw to content size for overlay
        from skimage.transform import resize as sk_resize
        disp_img = sk_resize(raw, (h_content, w_content), anti_aliasing=True)

        print(f"  {img_name} → \"{''.join(chars)}\"  ({len(chars)} chars)")
        rows.append((img_name, disp_img, chars, heatmaps))

    if not rows:
        print("No images decoded. Exiting.")
        return

    # ── Build the grid figure ────────────────────────────────────────────
    n_cols = max(len(r[2]) for r in rows)
    n_rows = len(rows)

    cell_w, cell_h = 2.2, 1.4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell_w * n_cols, cell_h * n_rows + 0.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Column headers
    for c in range(n_cols):
        axes[0, c].set_title(f"Char {c + 1}", fontsize=10, fontweight="bold")

    for ri, (img_name, disp_img, chars, heatmaps) in enumerate(rows):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            if ci < len(chars):
                # Show image with attention overlay
                ax.imshow(disp_img, cmap="gray", aspect="auto")
                ax.imshow(heatmaps[ci], cmap="inferno", alpha=args.alpha,
                          aspect="auto")
                # Character label in bottom-left
                ax.text(2, disp_img.shape[0] - 3, chars[ci],
                        fontsize=9, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1",
                                  facecolor="black", alpha=0.6))
            ax.set_xticks([])
            ax.set_yticks([])
            if ci >= len(chars):
                ax.set_visible(False)

    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.15)

    os.makedirs(args.save_dir, exist_ok=True)
    fname = f"char_overlay_{label}_{run_name}.png"
    save_path = os.path.join(args.save_dir, fname)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
