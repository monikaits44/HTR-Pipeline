#!/usr/bin/env python3
"""
HTR-VT (+R registers, 2-D grid) — Beyond-Memorization-style 2-D Attention Maps
==============================================================================

Companion to documents/most_most_latest/HTRVT_REGISTERS_BM_ATTENTION_FEASIBILITY.md.

Unlike scripts/register_attention_bm_style.py (which assumes a fully height-
collapsed 1-D patch row and therefore can only ever produce horizontal strips),
this script exploits the HTR-VT backbone's thin 2-D token grid (Hp x Wp, "mode B")
to extract GENUINE 2-D per-character attention blobs.

Pipeline (per test image):
  1. forward_explain()  -> self-attn [L][B,H,S,S], grid=(Hp,Wp), R registers
                           token layout: [R registers | Hp*Wp patches row-major]
  2. CTC greedy decode  -> character peak columns t_c in [0, Wp)
  3. For each character:
       - queries  = the Hp tokens of column t_c (one per grid row)
       - attn_to_patches = mean over those queries of A[:, query, R:]  -> [H, Hp*Wp]
       - reshape -> [H, Hp, Wp], pick most-localized head (or mean)
       - upsample [Hp, Wp] -> [IMAGE_H, IMAGE_W]  (genuine 2-D blob)
  4. Overlay inferno heatmap (alpha=0.5) on the input image, BM-style.

Quantitative companion:
  Reads results.csv from the run and prints best-val CER/WER + matching test
  CER/WER so qualitative maps sit next to the numbers.

Usage:
  python scripts/htrvt_bm_attention.py                 # latest htrvt run, READ2016 test
  python scripts/htrvt_bm_attention.py --run run_140
  python scripts/htrvt_bm_attention.py --device cuda:0 --num-images 8
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
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter, zoom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import HTRNet
from utils.preprocessing import load_image, preprocess

EXPERIMENTS_DIR = ROOT / "saved_models" / "experiments"
READ2016_TEST = ROOT / "data" / "READ2016" / "processed" / "test"
OUT_DIR = ROOT / "outputs" / "htrvt_bm_attention"
IMAGE_H, IMAGE_W = 128, 1024

# BM-exact output dimensions for per-character maps
# (matches Beyond-Memorization save_Attention2_with_blobs: figsize=(2.56,0.64), dpi=100)
BM_W, BM_H = 256, 64

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 150,      # used only by save_char_grid overview
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def find_latest_htrvt_run():
    runs = sorted(EXPERIMENTS_DIR.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    for run in reversed(runs):
        cfg_path = run / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if cfg.get("arch", {}).get("type") == "htrvt":
            return run
    raise SystemExit("No htrvt run found under saved_models/experiments/. Train first.")


def load_charset(run_cfg=None):
    """Load the charset the model was trained with.

    The trainer saves classes.npy next to the dataset (config.data.path), which
    is authoritative per-run. The shared experiments/classes.npy can be stale.
    """
    candidates = []
    if run_cfg is not None:
        data_path = run_cfg.get("data", {}).get("path")
        if data_path:
            candidates.append(Path(data_path) / "classes.npy")
            candidates.append(ROOT / str(data_path).lstrip("./") / "classes.npy")
    candidates.append(EXPERIMENTS_DIR / "classes.npy")
    for c in candidates:
        if c.exists():
            classes = np.load(c, allow_pickle=True)
            return [str(x) for x in classes]
    raise SystemExit("Could not locate classes.npy for this run.")


def load_model(run_dir, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    arch = SimpleNamespace(**cfg["arch"])
    classes = load_charset(cfg)
    nclasses = len(classes) + 1  # + CTC blank

    model = HTRNet(arch, nclasses)
    state = torch.load(run_dir / "model.pt", map_location=device)
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, classes, cfg


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def prepare_image(image_path, device):
    img = load_image(str(image_path))
    img = preprocess(img, (IMAGE_H, IMAGE_W))
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    return tensor, img


def ctc_greedy_decode(logits, charset):
    probs = torch.softmax(logits, dim=-1)
    seq = probs[:, 0, :].argmax(dim=-1).cpu().numpy()
    chars, cols, prev = [], [], -1
    for t, idx in enumerate(seq):
        if idx != 0 and idx != prev and 0 <= idx - 1 < len(charset):
            chars.append(charset[idx - 1])
            cols.append(t)
        prev = idx
    return "".join(chars), chars, cols


def extract_char_attention_2d(attn_maps, num_registers, grid, peak_cols,
                              layer_idx=-1, method="best", anchor_win=4.0):
    """Return list of [Hp, Wp] attention maps, one per character peak column.

    The plain self-attention of a CTC query token is contaminated by a few
    high-norm "sink" patch columns that every token attends to (the artifact
    tokens registers only partially absorb). To get an interpretable, character-
    local map we ANCHOR the read-out to the CTC alignment column t_c: we weight
    the per-patch attention by a Gaussian column prior centred on t_c
    (std = anchor_win columns). Set anchor_win<=0 to disable (raw attention).
    """
    A = attn_maps[layer_idx][0]          # [H, S, S]
    H = A.shape[0]
    Hp, Wp = grid
    R = num_registers
    P = Hp * Wp
    cc = np.arange(Wp)

    maps = []
    for t_c in peak_cols:
        # The Hp query tokens that map to CTC column t_c (one per grid row).
        q_idx = [R + h * Wp + t_c for h in range(Hp) if (R + h * Wp + t_c) < A.shape[1]]
        if not q_idx:
            maps.append(None)
            continue
        # mean over the column's query rows -> attention to all patch tokens
        patch_attn = A[:, q_idx, R:R + P].mean(dim=1).cpu().numpy()  # [H, P]
        patch_attn = patch_attn.reshape(H, Hp, Wp)                   # [H, Hp, Wp]

        # CTC-column anchoring: suppress global sink columns far from t_c
        if anchor_win and anchor_win > 0:
            prior = np.exp(-0.5 * ((cc - t_c) / anchor_win) ** 2)    # [Wp]
            patch_attn = patch_attn * prior[None, None, :]

        if method == "best":
            flat = patch_attn.reshape(H, -1)
            loc = flat.max(axis=1) / (flat.mean(axis=1) + 1e-8)
            attn_2d = patch_attn[int(np.argmax(loc))]               # [Hp, Wp]
        else:
            attn_2d = patch_attn.mean(axis=0)                       # [Hp, Wp]

        attn_2d = attn_2d - attn_2d.min()
        attn_2d = attn_2d / (attn_2d.max() + 1e-8)
        maps.append(attn_2d)
    return maps


def upsample_to_image(attn_2d, out_h=IMAGE_H, out_w=IMAGE_W, sigma=2.0):
    Hp, Wp = attn_2d.shape
    up = zoom(attn_2d, (out_h / Hp, out_w / Wp), order=1)
    up = gaussian_filter(up, sigma=sigma)
    up = up - up.min()
    up = up / (up.max() + 1e-8)
    return up


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def save_char_grid(img, chars, attn_2d_list, out_path, title):
    valid = [(c, a) for c, a in zip(chars, attn_2d_list) if a is not None]
    if not valid:
        return
    n = min(len(valid), 16)
    valid = valid[:n]
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.3 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < n:
            ch, a = valid[i]
            up = upsample_to_image(a)
            ax.imshow(1.0 - img, cmap="gray")
            ax.imshow(up, cmap="inferno", alpha=0.5)
            ax.set_title(repr(ch), fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cols_per_char(cols, Wp, fallback=6.0):
    """Estimate grid-columns per character from the CTC peak spacing."""
    if len(cols) >= 2:
        diffs = np.diff(np.asarray(cols, dtype=float))
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.median(diffs))
    return fallback


def save_bm_style_per_char(img, text, chars, cols, attn_2d_list, grid, out_dir,
                           window_chars=10, show_grid=False):
    """One 256x64 PNG per character -- exactly matching BM's save_Attention2_with_blobs.

    BM pipeline (reference):
      input  = PIL word image already 256x64 (single word crop from IAM words dataset)
      attn   = float32 [64, 256] from U-Net cross-attention at native image resolution
      render = figsize(2.56, 0.64) @ dpi=100 -> 256x64 px RGBA output
               ax.imshow(word_rgb) + ax.imshow(attn, cmap='inferno', alpha=0.5)
               ax.set_title(...), ax.axis('off'), savefig with no bbox_inches

    Our pipeline (adapted for HTR-VT CTC model):
      input  = float32 [128, 1024] full text line (1=ink from preprocessing)
      attn   = float32 [Hp=4, Wp=128] CTC-column-anchored self-attention grid
      adapt  = (a) horizontal: ~window_chars-wide grid-column window around CTC column
               (b) image: full 128px height -> PIL BILINEAR resize to 256x64
                          (compresses the line vertically to word-like proportions)
               (c) attn:  [4, crop_cols] -> scipy zoom (order=1) to [64, 256]
                          -> Gaussian smooth -> normalise
               (d) render: IDENTICAL to BM -- rc_context forces dpi=100, bbox=None
                           so output is exactly 256x64 px RGBA

    The pastel purple/yellow look is a natural consequence of alpha-blending
    inferno (dark-at-0, yellow-at-1) at alpha=0.5 over white paper: no manual
    floor or gamma tricks are needed -- this is exactly how BM achieves it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = img.shape   # 128, 1024
    Hp, Wp = grid      # 4, 128
    col_px = W / Wp    # 8.0 px per grid column

    cpc = _cols_per_char(cols, Wp)                          # grid-cols per char
    half_cols = max(int(round(0.5 * window_chars * cpc)), 3)
    # sigma in BM_W-pixel space: smooth over ~0.35 character widths
    char_w_out = BM_W / max(2 * half_cols, 1) * max(cpc, 1.0)
    sigma = max(char_w_out * 0.35, 3.0)

    # rc_context: forces this figure to be exactly BM_W x BM_H pixels,
    # overriding the script-level figure.dpi=150 / savefig.dpi=150 / bbox='tight'
    bm_ctx = {
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    }

    n = 0
    for ci, (ch, t_c, attn_2d) in enumerate(zip(chars, cols, attn_2d_list)):
        if attn_2d is None:
            continue

        # ---- (a) horizontal window in grid-column and pixel space ----
        c0 = max(0, t_c - half_cols)
        c1 = min(Wp, t_c + half_cols + 1)
        x0 = max(0, int(round(c0 * col_px)))
        x1 = min(W, int(round(c1 * col_px)))
        if x1 <= x0:
            continue

        # ---- (b) word image: full 128px height -> resize to 256x64 ----
        img_crop = img[:, x0:x1]             # [128, crop_w],  1=ink, 0=paper
        lo, hi = np.percentile(img_crop, 2), np.percentile(img_crop, 98)
        img_crop = np.clip((img_crop - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        img_crop = 1.0 - img_crop            # un-invert: 0=white paper, 1=dark ink
        img_pil = PILImage.fromarray((img_crop * 255).astype(np.uint8), mode="L").convert("RGB")
        img_pil = img_pil.resize((BM_W, BM_H), PILImage.BILINEAR)
        img_np = np.array(img_pil)           # [64, 256, 3] uint8 -- same as BM's currImg_np

        # ---- (c) attention: [Hp, crop_cols] -> [BM_H, BM_W] ----
        attn_crop = attn_2d[:, c0:c1].astype(np.float32)
        if attn_crop.size == 0:
            continue
        attn_crop = (attn_crop - attn_crop.min()) / (attn_crop.max() - attn_crop.min() + 1e-8)
        # bilinear zoom to [64, 256] -- same interpolation order as BM's imshow bilinear
        attn_up = zoom(attn_crop,
                       (BM_H / attn_crop.shape[0], BM_W / attn_crop.shape[1]),
                       order=1)
        attn_up = gaussian_filter(attn_up, sigma=sigma)
        attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
        # attn_up is [64, 256] float32 in [0,1] -- same shape/range as BM's currAttnImg

        # ---- (d) BM-exact rendering ----
        with plt.rc_context(bm_ctx):
            fig, ax = plt.subplots(figsize=(BM_W / 100, BM_H / 100))  # 2.56 x 0.64
            ax.imshow(img_np)                              # RGB word image (no cmap)
            ax.imshow(attn_up, cmap="inferno", alpha=0.5) # inferno overlay at alpha=0.5
            try:
                ax.set_title(f"Attention Map for Character {ch}")
            except Exception:
                pass
            if show_grid:
                ctc_x = (t_c - c0) / max(c1 - c0, 1) * BM_W
                ax.axvline(ctc_x, color="cyan", lw=0.5, alpha=0.7)
            ax.axis("off")
            safe = ch if ch.isalnum() else f"u{ord(ch)}"
            fname = f"{text_to_slug(text)}_{ci}_{safe}_char_att0.png"
            fig.savefig(out_dir / fname)  # no bbox_inches -> uses rc_context's None -> 256x64
            plt.close(fig)

        n += 1
    return n


def text_to_slug(text, maxlen=24):
    keep = "".join(c if c.isalnum() else "-" for c in text).strip("-")
    return (keep[:maxlen] or "line")


# ---------------------------------------------------------------------------
# Quantitative
# ---------------------------------------------------------------------------
def report_metrics(run_dir):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print("  (no results.csv yet)")
        return
    import csv as _csv
    rows = list(_csv.DictReader(csv_path.open()))
    if not rows:
        return
    def fnum(r, k):
        try:
            return float(r[k])
        except Exception:
            return float("inf")
    best = min(rows, key=lambda r: fnum(r, "val/cer"))
    print(f"  epochs trained:   {len(rows)}")
    print(f"  best epoch:       {best.get('epoch')}")
    print(f"  val  CER / WER:   {fnum(best,'val/cer'):.4f} / {fnum(best,'val/wer'):.4f}")
    print(f"  test CER / WER:   {fnum(best,'test/cer'):.4f} / {fnum(best,'test/wer'):.4f}")
    print(f"  params:           {best.get('model/params')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run_XX dir name (default: latest htrvt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-images", type=int, default=6)
    ap.add_argument("--method", choices=["best", "mean"], default="best")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--window-chars", type=float, default=10.0,
                    help="width of the per-character word crop, in characters (BM-style)")
    ap.add_argument("--grid-lines", action="store_true",
                    help="draw a cyan marker on the character's CTC column")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()

    run_dir = (EXPERIMENTS_DIR / args.run) if args.run else find_latest_htrvt_run()
    print(f"Run: {run_dir.name}")

    print("\n=== Quantitative (CER / WER) ===")
    report_metrics(run_dir)

    model, charset, cfg = load_model(run_dir, args.device)
    R = cfg["arch"].get("num_registers", 0)
    print(f"\nModel: htrvt | registers={R} | "
          f"grid={cfg['arch'].get('grid_height')}x{cfg['arch'].get('grid_width')}")

    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        image_paths = sorted(READ2016_TEST.glob("*.png"))[: args.num_images]
    if not image_paths:
        raise SystemExit(f"No images found in {READ2016_TEST}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Qualitative (2-D attention maps) -> {OUT_DIR} ===")
    for img_path in image_paths:
        tensor, img = prepare_image(img_path, args.device)
        with torch.no_grad():
            logits, _reg, attn_maps, _norms, grid = model.forward_explain(tensor)
        text, chars, cols = ctc_greedy_decode(logits, charset)
        attn_list = extract_char_attention_2d(
            attn_maps, R, grid, cols, layer_idx=args.layer, method=args.method
        )
        # (1) overview grid (all chars in one figure)
        out_path = OUT_DIR / f"{img_path.stem}_R{R}_chars.png"
        save_char_grid(img, chars, attn_list, out_path,
                       title=f"{img_path.stem}  |  pred: {text[:60]}")
        # (2) Beyond-Memorization style: one PNG per character
        per_char_dir = OUT_DIR / img_path.stem / "attentionMaps"
        n = save_bm_style_per_char(img, text, chars, cols, attn_list, grid, per_char_dir,
                                   window_chars=args.window_chars,
                                   show_grid=args.grid_lines)
        print(f"  {img_path.name:28s} pred='{text[:42]}'  -> grid + {n} per-char maps")

    print("\nDone.")


if __name__ == "__main__":
    main()
