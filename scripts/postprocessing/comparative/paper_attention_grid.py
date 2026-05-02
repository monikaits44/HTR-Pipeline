#!/usr/bin/env python3
"""
Paper-Style Character Attention Grid  (Fig. 5 equivalent for ViT-RGTS)
=======================================================================

Reproduces the exact figure style from the referenced paper (Fig. 5):

    Figure layout
    ─────────────
    ┌────────┬──────────┬──────────┬──────────┬──────────┬─────────┐
    │  label │   ch₀    │   ch₁    │   ch₂    │   …      │ charₙ₋₁│
    ├────────┼──────────┼──────────┼──────────┼──────────┼─────────┤
    │ word_1 │  heatmap │  heatmap │  heatmap │  heatmap │  [pad]  │
    ├────────┼──────────┼──────────┼──────────┼──────────┼─────────┤
    │ word_2 │  heatmap │  heatmap │  heatmap │  heatmap │ heatmap │
    └────────┴──────────┴──────────┴──────────┴──────────┴─────────┘

Key properties (to match the paper figure):
  • PURE heatmaps — no original image underneath, attention values only
  • Colormap: inferno  (dark→purple→orange→bright yellow)
  • White background with thin separator borders between cells
  • Each ROW   = one input image / word
  • Each COLUMN = one character position (fixed width = max_chars = L)
  • Padded positions (shorter words) shown as solid black cells
  • Character label header at top of every column
  • Row label (filename / Reg-N) on the left of each row
  • Colorbar per row at right edge showing intensity scale
  • Relative per-character scaling (p1–p99 percentile clip)
  • Adaptive gamma: γ_eff = γ + 0.05·max(n_chars-6, 0), capped at γ+3

Modes
─────
Single model + multiple images  → rows = images  (like paper Fig. 5)
Multiple models + single image  → rows = models  (register-sweep comparison)
Multiple models + multiple images → one figure per image (rows = models)

Output
──────
  <save-dir>/paper_attention_grid/<name>.png

Usage
─────
  # Single model, multiple images (paper Fig. 5 style)
  python scripts/postprocessing/paper_attention_grid.py \\
      configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
      --model-path saved_models/experiments/run_54/model.pt \\
      -- notebook/sample_images/

  # Register sweep (rows = models) for one image
  python scripts/postprocessing/paper_attention_grid.py \\
      configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
      --model-path saved_models/experiments/run_50/model.pt \\
      --model-path saved_models/experiments/run_51/model.pt \\
      --model-path saved_models/experiments/run_52/model.pt \\
      --model-path saved_models/experiments/run_53/model.pt \\
      --model-path saved_models/experiments/run_54/model.pt \\
      -- notebook/sample_images/a01-038-12.png

  # Custom options
  python scripts/postprocessing/paper_attention_grid.py \\
      configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
      --model-path saved_models/experiments/run_54/model.pt \\
      --layer last --max-chars 20 --gamma 3.5 --dpi 200 --cell-px 96 \\
      -- notebook/sample_images/

Arguments
─────────
  config.yaml         Base config file(s), merged in order
  key=value           Override params (e.g. device=cuda)
  --model-path PATH   Checkpoint path — repeat once per model
  --                  Separator; image path(s) follow
  --layer LAYER       first | middle | last | all  (default: last)
  --max-chars INT     Max character columns shown (default: 20)
  --gamma FLOAT       Contrast exponent. <1 boosts dim patches (good for overlay
                      on white-paper images); >1 sharpens focus on peak attention.
                      (default: 0.4)
  --dpi INT           Output DPI (default: 150)
  --cell-px INT       Cell height in pixels for the output figure (default: 80)
  --pad-to-max        Pad all rows to the same number of columns (default: True)
  --save-dir PATH     Output directory (default: visualizations/paper_attention_grid)
"""

import os
import sys
import gc
import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from PIL import Image

# ── Project root ──────────────────────────────────────────────────────────────
THIS_DIR    = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import HTRNet
from utils.metrics import CER, WER
from utils.preprocessing import load_image, preprocess

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

# ─────────────────────────────────────────────────────────────────────────────
# Style constants  (to match the paper figure)
# ─────────────────────────────────────────────────────────────────────────────
CMAP          = 'inferno'        # perceptually uniform; dark→purple→orange→bright
BG_COLOR      = '#FFFFFF'        # figure background (white)
CELL_BORDER   = '#E0E0E0'        # thin cell border colour
PAD_COLOR     = '#0B0B0F'        # solid black for padded/empty positions
LABEL_BG      = '#F7F5F0'        # warm off-white for label cells (like reference)
LABEL_FG      = '#1A1A1A'        # near-black text on label cells
HEADER_FG     = '#333333'        # character label colour
CER_GOOD      = '#2E7D32'        # green for low CER
CER_BAD       = '#C62828'        # red  for high CER
HEATMAP_ALPHA = 0.75             # inferno overlay opacity (0=image only, 1=heatmap only)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    argv    = sys.argv[1:]
    has_sep = '--' in argv
    if has_sep:
        sep      = argv.index('--')
        cfg_argv = argv[:sep]
        img_argv = argv[sep + 1:]
    else:
        cfg_argv = argv
        img_argv = []

    # Defaults
    model_paths = []
    save_dir    = ''
    layer       = 'last'
    max_chars   = 20
    gamma       = 0.4   # overlay mode: <1 boosts dim patches so inferno colors are vivid over white paper
    dpi         = 150
    cell_px     = 80      # cell height in figure pixels (controls cell aspect)
    pad_to_max  = True

    filtered, i = [], 0
    while i < len(cfg_argv):
        a = cfg_argv[i]
        if   a == '--model-path' and i + 1 < len(cfg_argv): model_paths.append(cfg_argv[i+1]); i += 2
        elif a == '--save-dir'   and i + 1 < len(cfg_argv): save_dir  = cfg_argv[i+1]; i += 2
        elif a == '--layer'      and i + 1 < len(cfg_argv): layer     = cfg_argv[i+1]; i += 2
        elif a == '--max-chars'  and i + 1 < len(cfg_argv): max_chars = int(cfg_argv[i+1]); i += 2
        elif a == '--gamma'      and i + 1 < len(cfg_argv): gamma     = float(cfg_argv[i+1]); i += 2
        elif a == '--dpi'        and i + 1 < len(cfg_argv): dpi       = int(cfg_argv[i+1]); i += 2
        elif a == '--cell-px'    and i + 1 < len(cfg_argv): cell_px   = int(cfg_argv[i+1]); i += 2
        elif a == '--no-pad':    pad_to_max = False; i += 1
        else: filtered.append(a); i += 1

    yaml_files = [a for a in filtered if a.endswith('.yaml')]
    overrides  = [a for a in filtered if not a.endswith('.yaml')]

    if not yaml_files:  sys.exit('Error: at least one .yaml config required.')
    if not model_paths: sys.exit('Error: at least one --model-path required.')
    if layer not in {'first', 'middle', 'last', 'all'}:
        sys.exit(f'Error: --layer must be first|middle|last|all (got: {layer})')

    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))
    OmegaConf.set_struct(conf, False)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))

    image_paths = _collect_images(img_argv)
    if has_sep and not image_paths:
        sys.exit('Error: "--" used but no valid images found.')

    return (conf, model_paths, image_paths, save_dir,
            layer, max_chars, gamma, dpi, cell_px, pad_to_max)


def _collect_images(paths):
    result = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    result.append(os.path.join(p, f))
        elif os.path.isfile(p):
            result.append(p)
        else:
            print(f'Warning: skipping {p}')
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Ground truth
# ═════════════════════════════════════════════════════════════════════════════

def load_gt(image_paths):
    gt, seen = {}, set()
    for p in image_paths:
        d = os.path.dirname(os.path.abspath(p))
        if d in seen: continue
        seen.add(d)
        cand = os.path.join(d, 'gt.txt')
        if os.path.isfile(cand):
            with open(cand, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            gt[parts[0]] = parts[1]
    return gt


# ═════════════════════════════════════════════════════════════════════════════
# Model utilities
# ═════════════════════════════════════════════════════════════════════════════

def run_info(model_path):
    run_dir  = os.path.dirname(os.path.abspath(model_path))
    run_name = os.path.basename(run_dir)
    cfg_json = os.path.join(run_dir, 'config.json')
    if not os.path.isfile(cfg_json):
        raise FileNotFoundError(f'config.json not found in {run_dir}')
    with open(cfg_json) as f:
        cfg = json.load(f)
    n_reg = cfg.get('arch', {}).get('num_registers', 0)
    return {'run_dir': run_dir, 'run_name': run_name,
            'num_registers': n_reg, 'label': f'Reg-{n_reg}'}


def build_model(base_conf, model_path, num_classes, num_reg, device):
    conf = deepcopy(base_conf)
    OmegaConf.set_struct(conf, False)
    conf.arch.num_registers = num_reg
    net = HTRNet(conf.arch, num_classes)
    ckpt = torch.load(model_path, map_location=device)
    net.load_state_dict(ckpt, strict=True)
    net.to(device).train()   # train mode needed for Grad-CAM compatibility
    return net


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

def select_layer(attn_maps_list, layer):
    """Pick a layer from attn_maps_list and head-average → [S,S] numpy."""
    L    = len(attn_maps_list)
    pick = {'first': 0, 'middle': L // 2, 'last': -1, 'all': None}[layer]
    attn = (torch.stack(attn_maps_list, 0).mean(0)
            if pick is None else attn_maps_list[pick])
    if attn.dim() == 4: attn = attn[0].mean(0)   # [H,S,S]→[S,S]
    elif attn.dim() == 3: attn = attn[0]
    return attn.cpu().numpy()


def run_inference(net, img_tensor, device, layer):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits, reg_tokens, attn_list, _, (Hp, Wp) = \
            net.forward_explain(img_tensor)
    if isinstance(logits, tuple): logits = logits[0]
    num_reg = reg_tokens.shape[1] if reg_tokens is not None else 0
    return {
        'attn':    select_layer(attn_list, layer),
        'logits':  logits.cpu().numpy(),
        'grid':    (Hp, Wp),
        'num_reg': num_reg,
    }


def ctc_decode(logits_np, i2c, blank=0):
    tdec = logits_np.argmax(2).squeeze()
    chars, positions, prev = [], [], -1
    for t, v in enumerate(tdec):
        v = int(v)
        if v != blank and v != prev:
            chars.append(i2c.get(v, '?'))
            positions.append(t)
        prev = v
    return chars, positions


# ═════════════════════════════════════════════════════════════════════════════
# Spatial upscaling:  patch-space → image-pixel-space
# ═════════════════════════════════════════════════════════════════════════════

def upscale(patch_map, H_img, W_img, fixed_size, border=8):
    """
    Upscale (Hp,Wp) patch heatmap → (H_img, W_img) pixel heatmap.

    Uses the preprocessed canvas coordinates (fixed_size) then crops to
    the original image region [border:border+H, border:border+W].
    """
    Hp, Wp = patch_map.shape
    if fixed_size is None:
        h = max(1, H_img // Hp)
        w = max(1, W_img // Wp)
        return np.kron(patch_map, np.ones((h, w)))[:H_img, :W_img]

    ph = fixed_size[0] // Hp    # preproc pixels per patch (height)
    pw = fixed_size[1] // Wp    # preproc pixels per patch (width)
    full = np.kron(patch_map, np.ones((ph, pw)))
    crop = full[border: border + H_img, border: border + W_img]
    if crop.shape != (H_img, W_img):
        padded = np.zeros((H_img, W_img), dtype=crop.dtype)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    return crop


# ═════════════════════════════════════════════════════════════════════════════
# Per-character heatmap builder
# ═════════════════════════════════════════════════════════════════════════════

def build_heatmaps(attn, grid, num_reg, chars, positions,
                   H_img, W_img, max_chars, gamma, fixed_size):
    """
    Build list of (H_img, W_img) heatmaps — one per PREDICTED character.

    Normalisation: percentile clip (p1–p99) then adaptive power contrast.
    Adaptive gamma scales up for long texts so each character stays crisp.

    Returns
    -------
    heatmaps : list[ndarray]  length = min(n_chars, max_chars)
    chars_out: list[str]      matching characters
    xmax_px  : list[int]      horizontal attention centroid in image pixels
    """
    Hp, Wp = grid
    T      = Hp * Wp
    n_show = min(len(chars), max_chars)

    # Adaptive gamma: for overlay mode (gamma<1) longer text needs slightly
    # more brightening (lower gamma), so we decrease by 0.02/char above 6.
    # For pure-heatmap mode (gamma>=1) the old increasing logic applied.
    n_ch  = max(1, len(chars))
    extra = 0.02 * max(n_ch - 6, 0)
    if gamma < 1.0:
        g_eff = float(max(gamma - extra, 0.15))   # lower = more boost
    else:
        g_eff = float(min(gamma + extra, gamma + 3.0))  # higher = more contrast

    heatmaps, xmax_px = [], []

    for ci in range(n_show):
        t       = positions[ci]
        row_idx = num_reg + t

        if row_idx >= attn.shape[0]:
            patch_attn = np.ones(T) / T
        else:
            patch_attn = attn[row_idx, num_reg:]
            if len(patch_attn) > T:
                patch_attn = patch_attn[:T]

        # ── Relative (percentile) normalisation ──────────────────────
        p_lo = max(0.0, np.percentile(patch_attn, 1.0))
        p_hi = np.percentile(patch_attn, 99.0)
        if p_hi - p_lo < 1e-8:
            p_lo, p_hi = patch_attn.min(), patch_attn.max()
        patch_attn = np.clip(patch_attn, p_lo, p_hi)
        patch_attn = (patch_attn - p_lo) / (p_hi - p_lo + 1e-8)

        # ── Adaptive power contrast ───────────────────────────────────
        patch_attn = patch_attn ** g_eff

        # ── Xmax_c: horizontal attention centroid ─────────────────────
        patch_grid = patch_attn.reshape(Hp, Wp)
        col_w      = patch_grid.sum(axis=0)
        col_w      = col_w / (col_w.sum() + 1e-12)
        x_patch    = float(np.dot(col_w, np.arange(Wp)))
        if fixed_size is not None:
            pw = fixed_size[1] // Wp
            xp = int(x_patch * pw + pw // 2) - 8   # subtract border
        else:
            xp = int(x_patch * W_img / Wp)
        xmax_px.append(max(0, min(xp, W_img - 1)))

        heatmaps.append(upscale(patch_grid, H_img, W_img, fixed_size))

    return heatmaps, chars[:n_show], xmax_px


# ═════════════════════════════════════════════════════════════════════════════
# Full per-image inference pipeline
# ═════════════════════════════════════════════════════════════════════════════

def process_image(net, img_path, fixed_size, device, layer,
                  i2c, gamma, max_chars, gt_dict, wer_mode, row_label):
    """Run inference on one image and return a row-descriptor dict."""
    name   = Path(img_path).stem
    raw    = load_image(img_path)
    proc   = preprocess(raw, fixed_size)
    tensor = torch.from_numpy(proc).float().unsqueeze(0).unsqueeze(0)

    res              = run_inference(net, tensor, device, layer)
    chars, positions = ctc_decode(res['logits'], i2c)
    pred_text        = ''.join(chars).strip()

    img_np = np.array(Image.open(img_path).convert('L'))
    H, W   = img_np.shape

    heatmaps, chars_shown, xmax_px = build_heatmaps(
        res['attn'], res['grid'], res['num_reg'],
        chars, positions, H, W, max_chars, gamma, fixed_size,
    )

    # ── Spatial crop boundary: right edge of last shown character ────
    # CTC position t → patch column t; each patch covers pw pixels.
    # crop_x = (t_last + 2) * pw - border  (+2 = one-char-width buffer)
    Wp = res['grid'][1]
    pw = (fixed_size[1] // Wp) if fixed_size is not None else (W // Wp)
    n_show = len(chars_shown)
    if n_show > 0:
        t_last  = positions[n_show - 1]
        crop_x  = max(1, (t_last + 2) * pw - 8)
    else:
        crop_x  = W
    crop_x = min(crop_x, W)

    gt_text = gt_dict.get(name, '')
    cer_val = wer_val = None
    if gt_text:
        try:
            cs = CER(); ws = WER(mode=wer_mode)
            cs.update(pred_text, gt_text); ws.update(pred_text, gt_text)
            cer_val = cs.score(); wer_val = ws.score()
        except Exception:
            pass

    return {
        'name':       name,
        'label':      row_label,
        'img_np':     img_np,
        'heatmaps':   heatmaps,      # list[ndarray H×W]  normalised [0,1]
        'chars':      chars_shown,   # list[str]
        'xmax_px':    xmax_px,       # list[int]
        'crop_x':     crop_x,        # pixel x to crop to (right edge of last char)
        'pred_text':  pred_text,
        'gt_text':    gt_text,
        'cer':        cer_val,
        'wer':        wer_val,
        'layer':      layer,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ─── PAPER-STYLE FIGURE RENDERER ────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def render_paper_grid(rows, save_path, dpi=150, cell_px=80, pad_to_max=True):
    """
    Build the paper Fig. 5 style grid and save as PNG.

    Parameters
    ----------
    rows : list[dict]
        Each dict has: label, img_np, heatmaps, chars, xmax_px, cer, pred_text
    cell_px : int
        Approximate rendered pixel height of one heatmap cell in the output.
    pad_to_max : bool
        If True, all rows are padded with black cells to the same column count.
    """
    if not rows:
        return

    # ── Uniform spatial crop: all rows share the same image width ─────
    # Take the maximum crop_x across rows so no character is cut off.
    uniform_crop_x = max(r.get('crop_x', r['img_np'].shape[1]) for r in rows)
    for r in rows:
        W_full = r['img_np'].shape[1]
        cx     = min(uniform_crop_x, W_full)
        r['img_np']   = r['img_np'][:, :cx]
        r['heatmaps'] = [h[:, :cx] for h in r['heatmaps']]
        # Clip xmax_px to new width
        r['xmax_px']  = [min(x, cx - 1) for x in r['xmax_px']]

    # ── Determine grid dimensions ────────────────────────────────────
    n_rows   = len(rows)
    col_lens = [len(r['chars']) for r in rows]
    L        = max(col_lens) if col_lens else 1   # number of char columns

    if not pad_to_max:
        # Each row only gets as many columns as it has characters
        pass   # L is still the max, padded cells are just black
    # L is always the same across all rows for alignment

    # ── Figure sizing ────────────────────────────────────────────────
    # Cell aspect ratio: use uniform_crop_x (the consistent cropped width)
    # so cells look proportional for both short words and cropped sentences.
    H0  = int(np.mean([r['img_np'].shape[0] for r in rows]))
    ar  = uniform_crop_x / max(H0, 1)   # width / height after crop
    # Limit cell width between 1.0 and 4.0 inches
    cell_h_in = max(0.8, cell_px / dpi)                   # inches
    cell_w_in = max(0.8, min(cell_h_in * ar, 4.0))        # inches

    LABEL_W_IN = 0.90   # row-label column width (inches)
    CBAR_W_IN  = 0.22   # colorbar column width  (inches)
    HPAD_IN    = 0.06   # header row height       (inches)
    SEP_IN     = 0.04   # gap between cells       (inches)

    fig_w = LABEL_W_IN + L * cell_w_in + CBAR_W_IN + SEP_IN * (L + 2)
    fig_h = HPAD_IN + n_rows * cell_h_in + SEP_IN * (n_rows + 2)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)

    # ── GridSpec ─────────────────────────────────────────────────────
    # rows: 0=header chars, 1..n_rows=data rows
    # cols: 0=label, 1..L=heatmaps, L+1=colorbar
    gs = gridspec.GridSpec(
        n_rows + 1,          # +1 for header
        L + 2,               # label + L char cols + colorbar
        figure=fig,
        hspace=SEP_IN / cell_h_in,
        wspace=SEP_IN / cell_w_in,
        left  = LABEL_W_IN / fig_w,
        right = 1.0 - (CBAR_W_IN + SEP_IN) / fig_w,
        top   = 1.0 - (HPAD_IN * 0.3) / fig_h,
        bottom= SEP_IN / fig_h,
    )

    # ── Row-label column widths via GridSpec ratios ───────────────────
    # (set via figure-level text instead, labels drawn as fig.text())

    cmap = plt.get_cmap(CMAP)

    # ── Header row  (character labels) ───────────────────────────────
    # Collect the union of characters at each position across all rows
    col_chars = []
    for ci in range(L):
        candidates = []
        for r in rows:
            if ci < len(r['chars']):
                candidates.append(r['chars'][ci])
        # Most common character at this column position
        if candidates:
            from collections import Counter
            col_chars.append(Counter(candidates).most_common(1)[0][0])
        else:
            col_chars.append('–')

    for ci in range(L):
        ax = fig.add_subplot(gs[0, ci])
        ch = col_chars[ci]
        display = repr(ch) if ch in (' ', '\t', '\n') else ch
        ax.text(0.5, 0.5, display,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=max(6, min(10, int(cell_w_in * 8))),
                fontweight='bold', color=HEADER_FG,
                fontfamily='monospace')
        ax.set_facecolor(LABEL_BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(CELL_BORDER); sp.set_linewidth(0.6)

    # Corner cells (header): label column and colorbar column
    for ci in [0, L + 1]:
        ax = fig.add_subplot(gs[0, ci if ci < L + 1 else L + 1])
        ax.set_facecolor(LABEL_BG)
        if ci == 0:
            ax.text(0.5, 0.5, 'sample', ha='center', va='center',
                    fontsize=6, color='#888888', fontstyle='italic')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(CELL_BORDER); sp.set_linewidth(0.6)

    # ── Data rows ─────────────────────────────────────────────────────
    for ri, row in enumerate(rows):
        n_ch     = len(row['chars'])
        last_im  = None

        # ── Row label cell ────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[ri + 1, 0])
        ax_lbl.set_facecolor(LABEL_BG)
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for sp in ax_lbl.spines.values():
            sp.set_edgecolor(CELL_BORDER); sp.set_linewidth(0.6)

        # Row label: name + CER (coloured) + short prediction
        lbl_parts = [row['label']]
        if row.get('cer') is not None:
            cer_col = CER_GOOD if row['cer'] < 0.05 else (
                      '#E65100' if row['cer'] < 0.15 else CER_BAD)
            ax_lbl.text(
                0.5, 0.62, row['label'],
                transform=ax_lbl.transAxes, ha='center', va='center',
                fontsize=min(8, max(5, int(cell_h_in * 6))),
                fontweight='bold', color=LABEL_FG, fontfamily='monospace',
            )
            ax_lbl.text(
                0.5, 0.38, f'CER={row["cer"]:.3f}',
                transform=ax_lbl.transAxes, ha='center', va='center',
                fontsize=min(7, max(5, int(cell_h_in * 5.5))),
                color=cer_col, fontfamily='monospace',
            )
        else:
            ax_lbl.text(
                0.5, 0.5, row['label'],
                transform=ax_lbl.transAxes, ha='center', va='center',
                fontsize=min(8, max(5, int(cell_h_in * 6))),
                fontweight='bold', color=LABEL_FG, fontfamily='monospace',
            )

        # Prediction text below (clipped)
        if row.get('pred_text'):
            pt = row['pred_text']
            # wrap to ~12 chars per line
            lines = [pt[i:i+12] for i in range(0, min(len(pt), 36), 12)]
            ax_lbl.text(
                0.5, 0.10, '\n'.join(lines),
                transform=ax_lbl.transAxes, ha='center', va='bottom',
                fontsize=4.5, color='#888888', fontfamily='monospace',
            )

        # ── Heatmap cells ─────────────────────────────────────────────
        for ci in range(L):
            ax = fig.add_subplot(gs[ri + 1, ci + 1])
            ax.set_xticks([]); ax.set_yticks([])

            if ci < n_ch:
                # Overlay: inferno heatmap blended over greyscale original image
                hm = row['heatmaps'][ci]      # float32 [0,1], shape (H,W)
                H_hm, W_hm = hm.shape

                # ── Resize original image to match heatmap shape ──────
                img_grey = row['img_np']
                if img_grey.shape != (H_hm, W_hm):
                    from PIL import Image as PILImage
                    img_grey = np.array(
                        PILImage.fromarray(img_grey.astype(np.uint8))
                              .resize((W_hm, H_hm), PILImage.BILINEAR)
                    )
                # Normalise to [0,1]
                img_f = img_grey.astype(np.float32) / 255.0

                # ── Inferno colourmap → RGB [0,1] ─────────────────────
                cmap_fn = plt.get_cmap(CMAP)
                heat_rgb = cmap_fn(hm)[:, :, :3]   # (H,W,3) float

                # ── Alpha blend: alpha*heat + (1-alpha)*grey_as_rgb ───
                grey_rgb = np.stack([img_f, img_f, img_f], axis=2)
                blended  = HEATMAP_ALPHA * heat_rgb + (1.0 - HEATMAP_ALPHA) * grey_rgb
                blended  = np.clip(blended, 0, 1)

                im = ax.imshow(blended, aspect='auto',
                               interpolation='bilinear', origin='upper')
                last_im = im

                # Xmax_c  — vertical dashed line showing horizontal centroid
                xmax = row['xmax_px'][ci]
                ax.axvline(x=xmax, color='#FFFFFF', linewidth=0.9,
                           linestyle='--', alpha=0.75)

                ax.set_facecolor('#000000')
            else:
                # Padded position: solid black cell
                ax.set_facecolor(PAD_COLOR)
                ax.imshow(np.zeros((4, 4)), cmap=CMAP, vmin=0, vmax=1,
                          aspect='auto')

            for sp in ax.spines.values():
                sp.set_edgecolor(CELL_BORDER); sp.set_linewidth(0.5)

        # ── Per-row colorbar ──────────────────────────────────────────
        ax_cb = fig.add_subplot(gs[ri + 1, L + 1])
        ax_cb.set_facecolor(LABEL_BG)
        ax_cb.set_xticks([]); ax_cb.set_yticks([])
        for sp in ax_cb.spines.values():
            sp.set_edgecolor(CELL_BORDER); sp.set_linewidth(0.4)

        if n_ch > 0:
            sm = plt.cm.ScalarMappable(
                cmap=plt.get_cmap(CMAP),
                norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
            )
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax_cb,
                              fraction=0.85, pad=0.0,
                              aspect=12,
                              orientation='vertical')
            cb.ax.tick_params(labelsize=4.5, colors='#555555')
            cb.set_ticks([0.0, 0.5, 1.0])
            cb.set_ticklabels(['low', 'mid', 'high'])
            cb.outline.set_edgecolor(CELL_BORDER)

    # ── Row-label column: draw outside GridSpec as figure text ────────
    # (the label column is handled inside the loop via ax_lbl above)

    # ── Figure title and caption ──────────────────────────────────────
    layer_str = rows[0].get('layer', 'last') if rows else 'last'
    title = (f'Character-Level Attention Maps  |  '
             f'Layer: {layer_str}  |  Colormap: {CMAP}  |  '
             f'{n_rows} sample(s) · {L} char columns')
    fig.suptitle(title, fontsize=7, color='#444444', y=1.0,
                 ha='center', fontfamily='sans-serif')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    print(f'  Saved → {save_path}')


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    (conf, model_paths, image_paths, save_dir,
     layer, max_chars, gamma, dpi, cell_px, pad_to_max) = parse_args()

    # ── Setup ─────────────────────────────────────────────────────────
    device = getattr(conf, 'device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA unavailable, using CPU.')
        device = 'cpu'

    dataset_folder = conf.data.path
    classes     = np.load(os.path.join(dataset_folder, 'classes.npy'))
    num_classes = len(classes) + 1
    i2c         = {(i + 1): c for i, c in enumerate(classes)}
    fixed_size  = (conf.preproc.image_height, conf.preproc.image_width)
    wer_mode    = conf.eval.wer_mode

    out_root = save_dir or os.path.join('visualizations', 'paper_attention_grid')
    os.makedirs(out_root, exist_ok=True)

    gt_dict = load_gt(image_paths) if image_paths else {}

    # ── Sort models by register count ─────────────────────────────────
    infos = [run_info(mp) for mp in model_paths]
    order = sorted(range(len(infos)), key=lambda i: infos[i]['num_registers'])
    infos       = [infos[i] for i in order]
    model_paths = [model_paths[i] for i in order]

    # ── Header ────────────────────────────────────────────────────────
    print('=' * 64)
    print('Paper Attention Grid Generator')
    print('=' * 64)
    print(f'Models     : {len(model_paths)}')
    for inf in infos:
        print(f'  {inf["label"]:>8s}  ({inf["run_name"]})')
    print(f'Images     : {len(image_paths)}')
    print(f'Layer      : {layer}')
    print(f'Max chars  : {max_chars}')
    print(f'Gamma base : {gamma}')
    print(f'DPI        : {dpi}')
    print(f'Cell px    : {cell_px}')
    print(f'Colormap   : {CMAP}')
    print(f'Output     : {out_root}')
    print(f'Timestamp  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    multi_model = len(model_paths) > 1
    multi_image = len(image_paths) > 1

    if multi_model:
        # ── rows = models, one figure per image ────────────────────────
        for img_path in image_paths:
            name = Path(img_path).stem
            print(f'Image: {name}')
            rows = []
            for info, mp in zip(infos, model_paths):
                print(f'  [{info["label"]}] …', end=' ', flush=True)
                net = build_model(conf, mp, num_classes,
                                  info['num_registers'], device)
                row = process_image(net, img_path, fixed_size, device, layer,
                                    i2c, gamma, max_chars, gt_dict, wer_mode,
                                    row_label=info['label'])
                del net; gc.collect()
                cer_s = f'CER={row["cer"]:.4f}' if row['cer'] is not None else ''
                print(f'"{row["pred_text"][:40]}"  {cer_s}')
                rows.append(row)

            out_path = os.path.join(out_root, f'{name}_{layer}.png')
            render_paper_grid(rows, out_path, dpi=dpi,
                              cell_px=cell_px, pad_to_max=pad_to_max)
            print()

    else:
        # ── single model, rows = images ────────────────────────────────
        info = infos[0]; mp = model_paths[0]
        print(f'Loading {info["label"]} ({info["run_name"]}) …')
        net = build_model(conf, mp, num_classes, info['num_registers'], device)

        rows = []
        for img_path in image_paths:
            name = Path(img_path).stem
            row  = process_image(net, img_path, fixed_size, device, layer,
                                 i2c, gamma, max_chars, gt_dict, wer_mode,
                                 row_label=name)
            cer_s = f'CER={row["cer"]:.4f}' if row['cer'] is not None else ''
            print(f'  {name:>30s}  →  "{row["pred_text"][:40]}"  {cer_s}')
            rows.append(row)

        del net; gc.collect()

        if multi_image:
            # All images in one combined figure (like paper Fig. 5)
            out_path = os.path.join(out_root,
                                    f'{info["run_name"]}_{layer}_grid.png')
            render_paper_grid(rows, out_path, dpi=dpi,
                              cell_px=cell_px, pad_to_max=pad_to_max)
        else:
            # Single image → single row
            name     = rows[0]['name']
            out_path = os.path.join(out_root,
                                    f'{name}_{info["run_name"]}_{layer}.png')
            render_paper_grid(rows, out_path, dpi=dpi,
                              cell_px=cell_px, pad_to_max=pad_to_max)

    print()
    print(f'Done!  Output → {out_root}/')
    print('=' * 64)


if __name__ == '__main__':
    main()
