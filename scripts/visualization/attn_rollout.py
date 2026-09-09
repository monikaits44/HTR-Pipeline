"""
Attention Rollout for ViT-RGTS
===============================
Propagates attention weights through ALL transformer layers
(Abnar & Zuidema 2020), giving a better account of information flow
than any single-layer attention map.

  A_rollout = A_L @ A_{L-1} @ ... @ A_1
  where each A_l = head-averaged attention + identity (residual), row-normalised

For each decoded character the rollout row at that token's position shows
which other patch tokens it "sees" through the entire ViT stack.

NOTE: The ViT in ViT-RGTS operates on a 1D token sequence (height was
collapsed by AdaptiveMaxPool2d before the ViT), so rollout attention is
inherently 1D.  Each character cell broadcasts the 1D slice to 2D for
display.  For a true 2D heatmap see char_gradcam.py instead.

Run:
  python scripts/visualization/attn_rollout.py
"""

import sys
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as spzoom
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_utils import (
    IMG_W, SCALE, RUNS, IMAGES, OUT, charset,
    load_model, prep, ctc_decode, char_window, height_bounds, uniform_win,
)


# -- Attention Rollout -------------------------------------------------------

def attention_rollout(attn_maps, discard_ratio=0.9):
    """
    Accumulate attention across all L layers.

    attn_maps    : list of L tensors, each [B, H, S, S]
    discard_ratio: fraction of lowest-weight entries zeroed per row (noise filter)

    Returns: rollout matrix [S, S]
    """
    result = None
    for am in attn_maps:
        a      = am[0].mean(0).numpy()              # head-average -> [S, S]
        thresh = np.quantile(a.flatten(), discard_ratio)
        a      = np.where(a >= thresh, a, 0.0)
        a      = a + np.eye(a.shape[0])             # residual connection
        a      = a / (a.sum(-1, keepdims=True) + 1e-8)
        result = a if result is None else result @ a
    return result


# -- Per-word register-comparison figure ------------------------------------

def make_rollout_figure(img_file, gt_text, img_dir, cmap="inferno", dpi=200):
    """
    Grid: rows = {0, 4, 8, 16} registers, cols = decoded characters.
    Each cell shows the rollout attention slice for that character token,
    broadcast uniformly across the crop height.
    """
    tensor, img_np = prep(img_file, img_dir)
    y0, y1  = height_bounds(img_np)
    crop_h  = y1 - y0

    ref_mdl = load_model(RUNS[8])
    with torch.no_grad():
        ref_lg, _, _, _, (_, Wp) = ref_mdl.forward_explain(tensor)
    ref_text, ref_peaks = ctc_decode(ref_lg)
    half_w, _ = char_window(ref_peaks)
    crop_w    = 2 * half_w
    del ref_mdl

    reg_keys       = sorted(RUNS.keys())
    nrows, n_chars = len(reg_keys), len(ref_peaks)

    cell_w = 1.6
    cell_h = cell_w * crop_h / crop_w
    fig, axes = plt.subplots(nrows, n_chars,
                              figsize=(cell_w * n_chars + 0.4,
                                       cell_h * nrows + 1.6))
    if nrows   == 1: axes = axes[np.newaxis, :]
    if n_chars == 1: axes = axes[:, np.newaxis]

    for row, nr in enumerate(reg_keys):
        mdl = load_model(RUNS[nr])
        R   = mdl.backbone.num_registers
        with torch.no_grad():
            lg, _, attn_maps, _, (_, Wp) = mdl.forward_explain(tensor)
        txt, peaks_r = ctc_decode(lg)
        peak_lut     = {ch: t for ch, t in peaks_r}
        rollout      = attention_rollout(attn_maps)       # [S, S]

        for ci, (ref_ch, ref_t) in enumerate(ref_peaks):
            ax  = axes[row, ci]
            t_i = peak_lut.get(ref_ch, ref_t)
            cx  = t_i * SCALE
            qi  = R + t_i                               # query index (skip registers)

            row_attn = rollout[qi, R:]                  # patch-token slice [Wp]
            p_cx     = int(cx / SCALE)
            p_half   = max(int(half_w / SCALE), 2)
            p_l      = max(0,  p_cx - p_half)
            p_r      = min(Wp, p_cx + p_half)
            row_c    = row_attn[p_l:p_r]

            lo, hi   = row_c.min(), row_c.max()
            row_n    = (row_c - lo) / (hi - lo) if hi > lo else row_c * 0

            # Upsample 1D -> 2D broadcast
            a_up = spzoom(row_n, crop_w / max(len(row_n), 1), order=1)[:crop_w]
            a_2d = np.tile(a_up, (crop_h, 1))

            img_win = uniform_win(img_np, cx, half_w, y0, y1)
            ax.imshow(img_win, cmap="gray_r", aspect="equal")
            ax.imshow(a_2d,    cmap=cmap,     aspect="equal", alpha=0.75, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if row == 0:
                ax.set_title(f'"{ref_ch}"', fontsize=13, fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(f"{nr} reg", fontsize=10, fontweight="bold", labelpad=6)

        del mdl
        print(f"    {nr:>2} reg  ->  \"{txt}\"")

    n_layers = len(attn_maps) if "attn_maps" in dir() else "?"
    fig.suptitle(
        f'Attention Rollout per Character  .  GT: "{gt_text}"\n'
        f"All {n_layers} layers accumulated (Abnar & Zuidema 2020)",
        fontsize=11, fontweight="bold", y=1.02)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)

    stem    = Path(img_file).stem
    out_png = OUT / f"attn_rollout_{stem}.png"
    fig.savefig(str(out_png),                           dpi=dpi, bbox_inches="tight")
    fig.savefig(str(OUT / f"attn_rollout_{stem}.pdf"),          bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_png}")


# -- Entry point ------------------------------------------------------------

def main():
    print("=== Attention Rollout - ViT-RGTS ===")
    for img_file, gt, img_dir in IMAGES:
        print(f"\n[{gt}]  {img_file}")
        make_rollout_figure(img_file, gt, img_dir)
    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
