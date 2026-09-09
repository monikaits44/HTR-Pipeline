"""
Character-Level GradCAM Attention Maps for ViT-RGTS
====================================================
GradCAM on the last GELU of the CNN stem produces a genuine 2D heatmap
[B, 256, 4, 128] -- the only point in the pipeline with spatial height
before AdaptiveMaxPool2d collapses height to 1 prior to the ViT.

Layout of output figures
------------------------
  char_gradcam_{id}.png     4 rows (register configs) x N cols (characters)
  char_gradcam_summary.png  4 cols (words) x 6 rows (input + 5 chars), 8-reg model

Reference: Selvaraju et al. "Grad-CAM" (ICCV 2017)

Run:
  python scripts/visualization/char_gradcam.py
"""

import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_utils import (
    DEVICE, IMG_W, SCALE, RUNS, IMAGES, OUT, charset,
    load_model, prep, ctc_decode, char_window, height_bounds,
    uniform_win, crop_cam,
)


# -- GradCAM ---------------------------------------------------------------

class GradCAM:
    """
    Hooks cnn_stem[11] (last GELU) to capture activations and gradients.
    Feature map shape at that layer: [B, 256, 4, 128] -- true 2D before
    height is collapsed by AdaptiveMaxPool2d.

    compute() returns a [crop_h, IMG_W] heatmap normalised to [0, 1].
    Use crop_cam() from viz_utils to extract the character window.
    """

    def __init__(self, model):
        self.model = model
        self._act  = None
        self._grad = None
        layer      = model.backbone.cnn_stem[11]
        self._fh   = layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_act", o))
        self._bh   = layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grad", go[0]))

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def compute(self, tensor, char_t, class_idx, full_h):
        """
        GradCAM for a single (character time-step, class) pair.
        Returns heatmap [full_h, IMG_W], values in [0, 1].
        """
        inp = tensor.clone().requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(inp)
        logits[char_t, 0, class_idx].backward()

        weights = self._grad.mean(dim=[2, 3], keepdim=True)
        cam     = torch.relu((weights * self._act).sum(dim=1))
        cam     = cam.squeeze(0).detach().numpy()

        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam   = F.interpolate(cam_t, size=(full_h, IMG_W),
                              mode="bilinear", align_corners=False).squeeze().numpy()
        cam   = gaussian_filter(cam, sigma=1.5)
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


# -- Per-word register-comparison figure -----------------------------------

def make_word_figure(img_file, gt_text, img_dir, cmap="inferno", dpi=200):
    """
    Grid: rows = {0, 4, 8, 16} registers, cols = decoded characters.
    Returns (ref_text, ref_peaks, half_w, y0, y1) for the summary panel.
    """
    tensor, img_np = prep(img_file, img_dir)
    y0, y1  = height_bounds(img_np)
    crop_h  = y1 - y0

    ref_mdl = load_model(RUNS[8])
    with torch.no_grad():
        ref_lg = ref_mdl(tensor)
    ref_text, ref_peaks = ctc_decode(ref_lg)
    half_w, _ = char_window(ref_peaks)
    del ref_mdl

    reg_keys       = sorted(RUNS.keys())
    nrows, n_chars = len(reg_keys), len(ref_peaks)

    cell_w = 1.6
    cell_h = cell_w * crop_h / (2 * half_w)
    fig, axes = plt.subplots(nrows, n_chars,
                              figsize=(cell_w * n_chars + 0.4,
                                       cell_h * nrows + 1.6))
    if nrows   == 1: axes = axes[np.newaxis, :]
    if n_chars == 1: axes = axes[:, np.newaxis]

    for row, nr in enumerate(reg_keys):
        mdl = load_model(RUNS[nr])
        with torch.no_grad():
            lg = mdl(tensor)
        txt, peaks_r = ctc_decode(lg)
        peak_lut = {ch: t for ch, t in peaks_r}
        gcam     = GradCAM(mdl)

        for ci, (ref_ch, ref_t) in enumerate(ref_peaks):
            ax        = axes[row, ci]
            t_i       = peak_lut.get(ref_ch, ref_t)
            cx        = t_i * SCALE
            class_idx = charset.index(ref_ch) + 1
            cam_win   = crop_cam(gcam.compute(tensor, t_i, class_idx, crop_h), cx, half_w)
            img_win   = uniform_win(img_np, cx, half_w, y0, y1)

            ax.imshow(img_win, cmap="gray_r", aspect="equal")
            ax.imshow(cam_win, cmap=cmap, aspect="equal", alpha=0.75, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if row == 0:
                ax.set_title(f'"{ref_ch}"', fontsize=13, fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(f"{nr} reg", fontsize=10, fontweight="bold", labelpad=6)

        gcam.remove()
        del mdl
        print(f"    {nr:>2} reg  ->  \"{txt}\"")

    fig.suptitle(
        f'GradCAM per Character  .  GT: "{gt_text}"  .  decoded: "{ref_text}"\n'
        f'CNN stem last layer  .  cell: {2*half_w}x{crop_h}px  .  rows = register count',
        fontsize=11, fontweight="bold", y=1.02)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)

    stem    = Path(img_file).stem
    out_png = OUT / f"char_gradcam_{stem}.png"
    fig.savefig(str(out_png),                              dpi=dpi, bbox_inches="tight")
    fig.savefig(str(OUT / f"char_gradcam_{stem}.pdf"),             bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_png}")
    return ref_text, ref_peaks, half_w, y0, y1


# -- 4-word summary panel --------------------------------------------------

def make_summary_panel(results, dpi=200, cmap="inferno"):
    """
    Publication-ready summary: 4 cols (words) x 6 rows (input + 5 chars).
    Uses the 8-register model only.
    """
    MAX_CHARS = 5
    n_imgs    = len(results)

    fig, axes = plt.subplots(MAX_CHARS + 1, n_imgs,
                              figsize=(2.2 * n_imgs, 2.2 * (MAX_CHARS + 1) + 0.8))
    if n_imgs == 1: axes = axes[:, np.newaxis]

    for col, (img_file, gt, img_dir, ref_text, ref_peaks, half_w, y0, y1) in enumerate(results):
        tensor, img_np = prep(img_file, img_dir)
        crop_h = y1 - y0

        mdl  = load_model(RUNS[8])
        gcam = GradCAM(mdl)
        with torch.no_grad():
            lg = mdl(tensor)
        _, peaks_r = ctc_decode(lg)
        peak_lut = {ch: t for ch, t in peaks_r}

        all_px  = [t * SCALE for _, t in ref_peaks]
        x0_word = max(0,     int(all_px[0]  - half_w))
        x1_word = min(IMG_W, int(all_px[-1] + half_w))
        ax = axes[0, col]
        ax.imshow(img_np[y0:y1, x0_word:x1_word], cmap="gray_r", aspect="auto")
        ax.set_title(f'"{gt}"', fontsize=12, fontweight="bold", pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if col == 0: ax.set_ylabel("Input", fontsize=10, fontweight="bold")

        for ri in range(MAX_CHARS):
            ax = axes[ri + 1, col]
            if ri < len(ref_peaks):
                ref_ch, ref_t = ref_peaks[ri]
                t_i       = peak_lut.get(ref_ch, ref_t)
                cx        = t_i * SCALE
                class_idx = charset.index(ref_ch) + 1
                cam_win   = crop_cam(gcam.compute(tensor, t_i, class_idx, crop_h), cx, half_w)
                img_win   = uniform_win(img_np, cx, half_w, y0, y1)
                ax.imshow(img_win, cmap="gray_r", aspect="equal")
                ax.imshow(cam_win, cmap=cmap, aspect="equal", alpha=0.75, vmin=0, vmax=1)
                if col == 0:
                    ax.set_ylabel(f'"{ref_ch}"', fontsize=11, fontweight="bold")
            else:
                ax.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)

        gcam.remove()
        del mdl

    fig.suptitle(
        "GradCAM Character Attention  .  8-register model\n"
        "CNN stem last layer  .  4 IAM test words  .  rows = character",
        fontsize=12, fontweight="bold", y=1.02)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    out_png = OUT / "char_gradcam_summary.png"
    fig.savefig(str(out_png),                          dpi=dpi, bbox_inches="tight")
    fig.savefig(str(OUT / "char_gradcam_summary.pdf"),         bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_png}")


# -- Entry point ------------------------------------------------------------

def main():
    print("=== Character GradCAM - ViT-RGTS ===")
    summary_data = []
    for img_file, gt, img_dir in IMAGES:
        print(f"\n[{gt}]  {img_file}")
        ret = make_word_figure(img_file, gt, img_dir)
        summary_data.append((img_file, gt, img_dir) + ret)

    print("\n[Summary panel]")
    make_summary_panel(summary_data)
    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
