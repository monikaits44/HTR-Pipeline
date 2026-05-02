#!/usr/bin/env python3
"""
Character-Level Attention Mapping for ViT-RGTS

Produces a publication-ready figure showing exactly WHERE in the input image
the model looks when predicting each character — matching the classic
character-attention-carpet layout.

Figure layout (for a single model):
  ┌────────────────────────────────────────────────────────────────┐
  │  Row 1: Original line image (full width, with GT & prediction) │
  ├────┬────┬────┬────┬────┬────┬────┬────────────────────────────┤
  │ "p"│ "e"│ "o"│ "p"│ "l"│ "e"│ "."│  ← character labels        │
  │ img│ img│ img│ img│ img│ img│ img│  ← same image repeated with │
  │+hm │+hm │+hm │+hm │+hm │+hm │+hm │     per-char heatmap      │
  ├────┴────┴────┴────┴────┴────┴────┴────────────────────────────┤
  │  Row 3: Character-Patch attention matrix (alignment carpet)    │
  │         X = patch positions (128 columns, aligned with image)  │
  │         Y = characters                                         │
  │         → diagonal pattern = correct spatial localisation       │
  └────────────────────────────────────────────────────────────────┘

Usage:
    python scripts/postprocessing/character_attention_mapping.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_54/model.pt \\
        -- notebook/sample_images/a01-038-12.png

    # Multiple models (one figure per model)
    python scripts/postprocessing/character_attention_mapping.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        -- notebook/sample_images/a01-038-12.png

    # All sample images, custom settings
    python scripts/postprocessing/character_attention_mapping.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --layer last --alpha 0.5 --dpi 200 --gamma 3.0 \\
        -- notebook/sample_images/

Arguments:
    config.yaml        Base config file(s), merged in order
    --model-path P     Checkpoint — repeat per model
    --                 Separator; image paths follow
    --layer LAYER      first|middle|last|all|rollout (default: last)
    --alpha FLOAT      Heatmap overlay opacity (default: 0.45)
    --dpi INT          Output resolution (default: 150)
    --gamma FLOAT      Contrast exponent (default: 3.0)
    --max-chars INT    Max characters to show (default: 0 = all)
    --save-dir DIR     Output directory (default: visualizations/character_attention_mapping/)
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
from matplotlib.patches import ConnectionPatch
from PIL import Image

# ── Project root ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import HTRNet
from utils.preprocessing import load_image, preprocess

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


# ═════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    argv = sys.argv[1:]
    has_sep = '--' in argv
    if has_sep:
        sep = argv.index('--')
        config_args, image_args = argv[:sep], argv[sep + 1:]
    else:
        config_args, image_args = argv, []

    model_paths, save_dir = [], ''
    alpha, dpi, gamma, max_chars = 0.45, 150, 3.0, 0
    layer = 'last'

    filtered = []
    i = 0
    while i < len(config_args):
        a = config_args[i]
        if a == '--model-path' and i + 1 < len(config_args):
            model_paths.append(config_args[i + 1]); i += 2
        elif a == '--save-dir' and i + 1 < len(config_args):
            save_dir = config_args[i + 1]; i += 2
        elif a == '--alpha' and i + 1 < len(config_args):
            alpha = float(config_args[i + 1]); i += 2
        elif a == '--dpi' and i + 1 < len(config_args):
            dpi = int(config_args[i + 1]); i += 2
        elif a == '--gamma' and i + 1 < len(config_args):
            gamma = float(config_args[i + 1]); i += 2
        elif a == '--max-chars' and i + 1 < len(config_args):
            max_chars = int(config_args[i + 1]); i += 2
        elif a == '--layer' and i + 1 < len(config_args):
            layer = config_args[i + 1]; i += 2
        else:
            filtered.append(a); i += 1

    yaml_files = [a for a in filtered if a.endswith('.yaml')]
    overrides = [a for a in filtered if not a.endswith('.yaml')]

    if not yaml_files:
        print('Error: at least one .yaml config.'); sys.exit(1)
    if not model_paths:
        print('Error: at least one --model-path.'); sys.exit(1)

    valid = {'first', 'middle', 'last', 'all', 'rollout'}
    if layer not in valid:
        print(f'Error: --layer must be one of {valid} (got: {layer})'); sys.exit(1)

    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))
    OmegaConf.set_struct(conf, False)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))

    image_paths = []
    for p in image_args:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(p, f))
        elif os.path.isfile(p):
            image_paths.append(p)
    if has_sep and not image_paths:
        print('Error: "--" used but no images found.'); sys.exit(1)
    if not image_paths:
        print('Error: at least one image required after --.'); sys.exit(1)

    return conf, model_paths, image_paths, save_dir, alpha, dpi, gamma, max_chars, layer


# ═════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════════════════════

def get_run_info(model_path):
    run_dir = os.path.dirname(os.path.abspath(model_path))
    run_name = os.path.basename(run_dir)
    config_json = os.path.join(run_dir, 'config.json')
    if not os.path.isfile(config_json):
        raise FileNotFoundError(f'config.json not found in {run_dir}')
    with open(config_json) as f:
        cfg = json.load(f)
    num_reg = cfg.get('arch', {}).get('num_registers', 0)
    return {'run_dir': run_dir, 'run_name': run_name,
            'num_registers': num_reg, 'label': f'Reg-{num_reg}'}


def build_model(base_config, model_path, num_classes, num_registers, device):
    config = deepcopy(base_config)
    OmegaConf.set_struct(config, False)
    config.arch.num_registers = num_registers
    net = HTRNet(config.arch, num_classes)
    ckpt = torch.load(model_path, map_location=device)
    net.load_state_dict(ckpt, strict=True)
    net.to(device).eval()
    return net


# ═════════════════════════════════════════════════════════════════════════════
# CTC Decoding
# ═════════════════════════════════════════════════════════════════════════════

def ctc_char_positions(logits_np, i2c, blank_id=0):
    tdec = logits_np.argmax(2).squeeze()
    chars, positions = [], []
    prev = -1
    for t, v in enumerate(tdec):
        v_int = int(v)
        if v_int != blank_id and v_int != prev:
            chars.append(str(i2c.get(v_int, '?')))
            positions.append(t)
        prev = v_int
    return chars, positions


def _char_display(c):
    """Safe display label for a character (works with numpy strings too)."""
    c = str(c)
    if c in (' ', '\t', '\n'):
        return repr(c)
    return c


# ═════════════════════════════════════════════════════════════════════════════
# Attention Extraction
# ═════════════════════════════════════════════════════════════════════════════

def attention_rollout(attn_maps_list):
    result = None
    for attn in attn_maps_list:
        attn_avg = attn.mean(dim=1)
        S = attn_avg.size(-1)
        I = torch.eye(S, device=attn_avg.device).unsqueeze(0)
        attn_res = 0.5 * attn_avg + 0.5 * I
        attn_res = attn_res / (attn_res.sum(dim=-1, keepdim=True) + 1e-8)
        result = attn_res if result is None else torch.bmm(attn_res, result)
    return result[0].cpu().numpy()


def select_layer(attn_maps_list, layer='last'):
    L = len(attn_maps_list)
    if layer == 'rollout':
        return attention_rollout(attn_maps_list)
    idx = {'first': 0, 'middle': L // 2, 'last': -1}.get(layer)
    if layer == 'all':
        attn = torch.stack(attn_maps_list, dim=0).mean(dim=0)
    else:
        attn = attn_maps_list[idx if idx is not None else -1]
    if attn.dim() == 4:
        attn = attn[0].mean(dim=0)
    elif attn.dim() == 3:
        attn = attn[0]
    return attn.cpu().numpy()


def extract_attention(net, img_tensor, device, layer):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits, reg_tokens, attn_maps_list, token_norms, (Hp, Wp) = \
            net.forward_explain(img_tensor)
    if isinstance(logits, tuple):
        logits = logits[0]
    num_reg = reg_tokens.shape[1] if reg_tokens is not None else 0
    attn = select_layer(attn_maps_list, layer)
    return {
        'attn': attn,
        'grid_size': (Hp, Wp),
        'logits': logits.cpu().numpy(),
        'num_reg': num_reg,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Ground Truth
# ═════════════════════════════════════════════════════════════════════════════

def load_gt(image_paths):
    gt = {}
    seen = set()
    for p in image_paths:
        d = os.path.dirname(os.path.abspath(p))
        if d in seen:
            continue
        seen.add(d)
        gf = os.path.join(d, 'gt.txt')
        if os.path.isfile(gf):
            with open(gf, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        gt[parts[0]] = parts[1]
    return gt


# ═════════════════════════════════════════════════════════════════════════════
# Upscale patch map → image coordinates
# ═════════════════════════════════════════════════════════════════════════════

def upscale_patch_aligned(patch_map, H_img, W_img, fixed_size, border_size=8):
    Hp, Wp = patch_map.shape
    H_pre, W_pre = fixed_size
    ph, pw = H_pre // Hp, W_pre // Wp
    heat_full = np.kron(patch_map, np.ones((ph, pw)))
    r0, c0 = border_size, border_size
    crop = heat_full[r0:r0 + H_img, c0:c0 + W_img]
    if crop.shape[0] < H_img or crop.shape[1] < W_img:
        padded = np.zeros((H_img, W_img), dtype=crop.dtype)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    return crop


def compute_content_geometry(H_img, W_img, grid_size, fixed_size, border_size=8):
    """Compute how the original image maps into the preprocessed canvas
    and into the patch grid.

    Returns dict with:
        scale     – resize factor (original → preprocessed)
        n_width   – content width in preprocessed pixels
        pw        – single-patch width in preprocessed pixels
        p_start   – first patch overlapping actual image content
        p_end     – last patch overlapping content (exclusive)
        img_left  – left edge of image in fractional patch coords
        img_right – right edge of image in fractional patch coords
    """
    H_pre, W_pre = fixed_size
    Hp, Wp = grid_size
    pw = W_pre // Wp
    n_height = min(H_pre - 2 * border_size, H_img)
    scale = n_height / H_img
    n_width = min(W_pre - 2 * border_size, int(scale * W_img))
    p_start = border_size // pw
    p_end = min(Wp, (border_size + n_width + pw - 1) // pw)
    img_left = border_size / pw
    img_right = (border_size + n_width) / pw
    return dict(scale=scale, n_width=n_width, pw=pw,
                p_start=p_start, p_end=p_end,
                img_left=img_left, img_right=img_right)


# ═════════════════════════════════════════════════════════════════════════════
# Per-character attention extraction
# ═════════════════════════════════════════════════════════════════════════════

def get_char_attention(attn_matrix, num_reg, positions, grid_size, gamma=3.0,
                       percentile_clip=99.0):
    """
    Extract per-character attention vectors and build the alignment matrix.

    Returns:
        char_vectors : list of [T] raw attention vectors (one per character)
        char_normed  : list of [T] normalised+gamma-boosted vectors
        alignment    : [n_chars, T] matrix
    """
    Hp, Wp = grid_size
    T = Hp * Wp
    n_chars = len(positions)
    gamma_adaptive = float(np.clip(gamma + 0.05 * (n_chars - 6), gamma, gamma + 3.0))

    char_vectors = []
    char_normed = []

    for ci in range(n_chars):
        t = positions[ci]
        row_idx = num_reg + t
        if row_idx >= attn_matrix.shape[0]:
            vec = np.ones(T) / T
        else:
            vec = attn_matrix[row_idx, num_reg:num_reg + T].copy()
            if vec.shape[0] < T:
                vec = np.pad(vec, (0, T - vec.shape[0]))

        char_vectors.append(vec)

        # Normalise + gamma contrast
        p_lo = max(0.0, np.percentile(vec, 100.0 - percentile_clip))
        p_hi = np.percentile(vec, percentile_clip)
        if p_hi - p_lo < 1e-8:
            p_lo, p_hi = vec.min(), vec.max()
        normed = np.clip(vec, p_lo, p_hi)
        normed = (normed - p_lo) / (p_hi - p_lo + 1e-8)
        normed = normed ** gamma_adaptive
        char_normed.append(normed)

    alignment = np.stack(char_normed, axis=0)  # [n_chars, T]
    return char_vectors, char_normed, alignment


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Character Attention Carpet (matching sample_attention.png style)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_char_attention_carpet(img_path, chars, char_normed, grid_size,
                                    fixed_size, model_label, gt_text,
                                    pred_text, save_path, alpha, dpi,
                                    max_chars, border_size=8):
    """
    Side-by-side character attention panels: each character gets a copy
    of the original image with its attention heatmap overlaid.

    Layout:
      Top   : character labels
      Main  : [img+hm₁ | img+hm₂ | … | img+hmₙ]
    """
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape
    image_name = Path(img_path).stem
    Hp, Wp = grid_size

    n_show = min(len(chars), max_chars) if max_chars > 0 else len(chars)
    if n_show == 0:
        return None

    # ── Build per-character heatmaps ─────────────────────────────────
    heatmaps = []
    for ci in range(n_show):
        grid = char_normed[ci].reshape(Hp, Wp)
        heat_up = upscale_patch_aligned(grid, H_img, W_img, fixed_size,
                                        border_size)
        heatmaps.append(heat_up)

    # ── Figure dimensions (scale for many characters) ────────────────
    panel_w_base = max(1.5, min(W_img / 70, 4.0))
    panel_h = max(1.8, min(H_img / 25, 5.0))
    # Cap total width: shrink panels if too many characters
    max_fig_w = 50.0  # inches
    panel_w = min(panel_w_base, (max_fig_w - 0.5) / max(n_show, 1))
    panel_w = max(0.35, panel_w)  # minimum readable panel width
    fig_w = panel_w * n_show + 0.5
    fig_h = panel_h + 1.2

    title_fs = max(5, min(9, int(9 - (n_show - 15) * 0.1)))

    fig, axes = plt.subplots(1, n_show, figsize=(fig_w, fig_h), squeeze=False)

    for ci in range(n_show):
        ax = axes[0, ci]
        ax.imshow(img_np, cmap="gray")
        im = ax.imshow(heatmaps[ci], cmap="inferno", alpha=alpha,
                       vmin=0.0, vmax=1.0)
        ch_disp = f'"{_char_display(chars[ci])}"'
        ax.set_title(ch_disp, fontsize=max(5, min(9, int(9 - (n_show - 15) * 0.08))),
                     fontweight='bold', color='darkblue')
        ax.axis("off")

        # Peak position marker (pixel-accurate from upscaled heatmap)
        peak_x = int(np.argmax(heatmaps[ci].max(axis=0)))
        ax.axvline(peak_x, color='cyan', linewidth=0.5, alpha=0.6,
                   linestyle='--')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04,
                        orientation='vertical')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('rel. attn.', fontsize=6)

    # Title
    title_parts = [f'{image_name}  |  {model_label}']
    if gt_text:
        title_parts.append(f'GT: "{gt_text}"')
    title_parts.append(f'Pred: "{pred_text}"')
    fig.suptitle('  —  '.join(title_parts), fontsize=9, fontweight='bold',
                 y=1.02)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Alignment Matrix + Original Image + Connecting Lines
# ═════════════════════════════════════════════════════════════════════════════

def visualize_alignment_matrix(img_path, chars, alignment, grid_size,
                               fixed_size, model_label, gt_text, pred_text,
                               save_path, dpi, max_chars, border_size=8):
    """
    Two-panel figure:
      Top   : Original line image (aligned with patch positions)
      Bottom: Character × Patch alignment matrix

    A diagonal pattern in the bottom panel = the model reads left-to-right
    and spatially localises each character correctly.

    Connecting lines link each character's peak attention to the
    corresponding position in the image above.
    """
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape
    image_name = Path(img_path).stem
    Hp, Wp = grid_size

    n_show = min(len(chars), max_chars) if max_chars > 0 else len(chars)
    if n_show == 0:
        return None

    alignment_vis = alignment[:n_show, :]   # [n_show, T]
    T = alignment_vis.shape[1]

    # ── Content geometry: which patches hold actual image content ─────
    geo = compute_content_geometry(H_img, W_img, grid_size, fixed_size,
                                   border_size)
    margin = 2
    p_lo = max(0, geo['p_start'] - margin)
    p_hi = min(T, geo['p_end'] + margin)
    alignment_crop = alignment_vis[:, p_lo:p_hi]

    # ── Figure (scale for many characters) ───────────────────────────
    n_vis = p_hi - p_lo
    fig_w = max(8, min(n_vis / 3, 18))
    fig_h = max(3.5, n_show * 0.35 + 3.5)

    # Scale font sizes for many characters
    ytick_fs = max(4, min(7, int(7 - (n_show - 20) * 0.06)))
    label_fs = max(5, min(8, int(8 - (n_show - 20) * 0.05)))

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, max(2, n_show * 0.30)],
                           hspace=0.08)

    # ── Top: Original image mapped to patch coordinates ──────────────
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_np, cmap="gray", aspect='auto',
                  extent=[geo['img_left'], geo['img_right'], H_img, 0])
    ax_img.set_xlim(p_lo, p_hi)
    ax_img.set_xticks([])
    ax_img.set_ylabel('Image', fontsize=8)
    ax_img.tick_params(axis='y', labelsize=6)

    # Mark peak attention positions on the image
    peak_positions = []
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_show, 1)))
    for ci in range(n_show):
        peak = int(alignment_vis[ci].argmax())
        peak_positions.append(peak)
        if p_lo <= peak < p_hi:
            ax_img.axvline(peak + 0.5, color=colors[ci % len(colors)],
                           linewidth=1.0, alpha=0.7, linestyle='--')
            ch_disp = _char_display(chars[ci])
            ax_img.text(peak + 0.5, -2, ch_disp, fontsize=max(5, ytick_fs),
                        ha='center', va='bottom',
                        color=colors[ci % len(colors)],
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  alpha=0.8, edgecolor='none'))

    # ── Bottom: Alignment matrix (cropped to active region) ──────────
    ax_mat = fig.add_subplot(gs[1])
    im = ax_mat.imshow(alignment_crop, cmap="inferno", aspect='auto',
                       vmin=0.0, vmax=1.0, interpolation='nearest',
                       extent=[p_lo, p_hi, n_show - 0.5, -0.5])

    # Y-axis: character labels
    char_labels = [_char_display(c) for c in chars[:n_show]]
    ax_mat.set_yticks(range(n_show))
    ax_mat.set_yticklabels(char_labels, fontsize=ytick_fs, fontfamily='monospace')
    ax_mat.set_xlabel('Patch position (left → right in image)', fontsize=label_fs)
    ax_mat.set_ylabel('Character', fontsize=label_fs)
    ax_mat.tick_params(axis='x', labelsize=max(4, ytick_fs - 1))

    # Connecting lines from alignment peak to image
    for ci in range(n_show):
        peak = peak_positions[ci]
        if p_lo <= peak < p_hi:
            con = ConnectionPatch(
                xyA=(peak + 0.5, -0.5), coordsA=ax_mat.transData,
                xyB=(peak + 0.5, H_img), coordsB=ax_img.transData,
                color=colors[ci % len(colors)], linewidth=0.8, alpha=0.5,
                linestyle=':')
            fig.add_artist(con)

    # Diagonal guide
    if n_show >= 2:
        peaks = [peak_positions[ci] + 0.5 for ci in range(n_show)]
        ax_mat.plot(peaks, range(n_show), 'w--', linewidth=0.8, alpha=0.6,
                    label='peak path')
        ax_mat.legend(fontsize=5, loc='lower right',
                      framealpha=0.7, edgecolor='none')

    # Colorbar attached to BOTH axes so widths stay aligned
    cbar = fig.colorbar(im, ax=[ax_img, ax_mat], fraction=0.02, pad=0.02,
                        orientation='vertical')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('normalised attention', fontsize=6)

    # Title
    title_parts = [f'Character–Patch Alignment: {image_name}  |  {model_label}']
    if gt_text:
        gt_disp = gt_text if len(gt_text) <= 50 else gt_text[:47] + '…'
        title_parts.append(f'GT: "{gt_disp}"')
    pred_disp = pred_text if len(pred_text) <= 50 else pred_text[:47] + '…'
    title_parts.append(f'Pred: "{pred_disp}"')
    fig.suptitle('\n'.join(title_parts), fontsize=9, fontweight='bold', y=1.04)

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Combined (Original + Carpet + Alignment in one figure)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_combined(img_path, chars, char_normed, alignment, grid_size,
                       fixed_size, model_label, gt_text, pred_text,
                       save_path, alpha, dpi, max_chars, border_size=8):
    """
    Three-panel combined figure:
      Row 0: Original image with peak markers and text labels
      Row 1: Side-by-side character attention panels
      Row 2: Character × Patch alignment matrix with connecting lines
    """
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape
    image_name = Path(img_path).stem
    Hp, Wp = grid_size
    T = Hp * Wp

    n_show = min(len(chars), max_chars) if max_chars > 0 else len(chars)
    if n_show == 0:
        return None

    alignment_vis = alignment[:n_show, :]

    # ── Content geometry ─────────────────────────────────────────────
    geo = compute_content_geometry(H_img, W_img, grid_size, fixed_size,
                                   border_size)
    margin = 2
    p_lo = max(0, geo['p_start'] - margin)
    p_hi = min(T, geo['p_end'] + margin)
    alignment_crop = alignment_vis[:, p_lo:p_hi]

    # Build heatmaps for carpet
    heatmaps = []
    for ci in range(n_show):
        grid = char_normed[ci].reshape(Hp, Wp)
        heat_up = upscale_patch_aligned(grid, H_img, W_img, fixed_size,
                                        border_size)
        heatmaps.append(heat_up)

    # ── Dimensions (scale for many characters) ───────────────────────
    panel_w_base = max(1.5, min(W_img / 70, 3.5))
    max_fig_w = 50.0
    panel_w = min(panel_w_base, (max_fig_w - 1.5) / max(n_show, 1))
    panel_w = max(0.35, panel_w)
    fig_w = max(10, panel_w * n_show + 1.5)
    carpet_h = max(1.8, min(H_img / 30, 4.0))
    mat_h = max(2.0, n_show * 0.25)
    orig_h = max(1.2, carpet_h * 0.6)

    # Scale font sizes
    ytick_fs = max(3, min(6, int(6 - (n_show - 20) * 0.05)))
    carpet_title_fs = max(4, min(7, int(7 - (n_show - 15) * 0.06)))

    fig = plt.figure(figsize=(fig_w, orig_h + carpet_h + mat_h + 1.5))
    gs = gridspec.GridSpec(3, 1,
                           height_ratios=[orig_h, carpet_h, mat_h],
                           hspace=0.15)

    # ── Row 0: Original image in patch-aligned coordinates ───────────
    ax_orig = fig.add_subplot(gs[0])
    ax_orig.imshow(img_np, cmap="gray", aspect='auto',
                   extent=[geo['img_left'], geo['img_right'], H_img, 0])
    ax_orig.set_xlim(p_lo, p_hi)
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])

    # Peak position markers on reference image
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_show, 1)))
    for ci in range(n_show):
        peak = int(alignment_vis[ci].argmax())
        if p_lo <= peak < p_hi:
            ax_orig.axvline(peak + 0.5, color=colors[ci % len(colors)],
                            linewidth=0.8, alpha=0.5, linestyle='--')

    # Text annotation
    info_parts = [model_label]
    if gt_text:
        gt_d = gt_text if len(gt_text) <= 40 else gt_text[:37] + '…'
        info_parts.append(f'GT: "{gt_d}"')
    pred_d = pred_text if len(pred_text) <= 40 else pred_text[:37] + '…'
    info_parts.append(f'Pred: "{pred_d}"')
    ax_orig.set_title('  |  '.join(info_parts), fontsize=8, fontweight='bold',
                      pad=4)
    ax_orig.axis("off")

    # ── Row 1: Character attention carpet ────────────────────────────
    gs_carpet = gs[1].subgridspec(1, n_show, wspace=0.05)
    carpet_axes = []
    for ci in range(n_show):
        ax = fig.add_subplot(gs_carpet[0, ci])
        carpet_axes.append(ax)

        ax.imshow(img_np, cmap="gray")
        im = ax.imshow(heatmaps[ci], cmap="inferno", alpha=alpha,
                       vmin=0.0, vmax=1.0)

        ch_disp = f'"{_char_display(chars[ci])}"'
        ax.set_title(ch_disp, fontsize=carpet_title_fs, fontweight='bold',
                     color='#1a237e', pad=2)
        ax.axis("off")

        # Peak marker (pixel-accurate from upscaled heatmap)
        peak_x = int(np.argmax(heatmaps[ci].max(axis=0)))
        ax.axvline(peak_x, color='cyan', linewidth=0.6, alpha=0.5,
                   linestyle='--')

    # Colorbar
    cbar = fig.colorbar(im, ax=carpet_axes[-1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('attn', fontsize=5)

    # ── Row 2: Alignment matrix (cropped to active region) ───────────
    ax_mat = fig.add_subplot(gs[2])
    im2 = ax_mat.imshow(alignment_crop, cmap="inferno", aspect='auto',
                        vmin=0.0, vmax=1.0, interpolation='nearest',
                        extent=[p_lo, p_hi, n_show - 0.5, -0.5])

    char_labels = [_char_display(c) for c in chars[:n_show]]
    ax_mat.set_yticks(range(n_show))
    ax_mat.set_yticklabels(char_labels, fontsize=ytick_fs, fontfamily='monospace')
    ax_mat.set_xlabel('Patch position (left → right)', fontsize=max(5, ytick_fs + 1))
    ax_mat.set_ylabel('Char', fontsize=max(5, ytick_fs + 1))
    ax_mat.tick_params(axis='both', labelsize=max(3, ytick_fs - 1))

    # Diagonal guide
    if n_show >= 2:
        peaks = [int(alignment_vis[ci].argmax()) + 0.5 for ci in range(n_show)]
        ax_mat.plot(peaks, range(n_show), 'w--', linewidth=0.8, alpha=0.6,
                    label='peak path')
        ax_mat.legend(fontsize=5, loc='lower right',
                      framealpha=0.7, edgecolor='none')

    cbar2 = fig.colorbar(im2, ax=ax_mat, fraction=0.02, pad=0.02)
    cbar2.ax.tick_params(labelsize=5)
    cbar2.set_label('attn', fontsize=5)

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    (conf, model_paths, image_paths, save_dir,
     alpha, dpi, gamma, max_chars, layer) = parse_args()

    device = getattr(conf, 'device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA not available, falling back to CPU.')
        device = 'cpu'

    dataset_folder = conf.data.path
    classes = np.load(os.path.join(dataset_folder, 'classes.npy'))
    num_classes = len(classes) + 1
    i2c = {(i + 1): c for i, c in enumerate(classes)}
    fixed_size = (conf.preproc.image_height, conf.preproc.image_width)

    out_dir = (save_dir if save_dir
               else os.path.join('visualizations', 'character_attention_mapping'))
    carpet_dir = os.path.join(out_dir, 'carpet')
    alignment_dir = os.path.join(out_dir, 'alignment')
    combined_dir = os.path.join(out_dir, 'combined')

    import shutil
    for sub in (carpet_dir, alignment_dir, combined_dir):
        if os.path.isdir(sub):
            shutil.rmtree(sub)
        os.makedirs(sub, exist_ok=True)

    model_infos = [get_run_info(mp) for mp in model_paths]
    sorted_idx = sorted(range(len(model_infos)),
                        key=lambda i: model_infos[i]['num_registers'])
    model_infos = [model_infos[i] for i in sorted_idx]
    model_paths = [model_paths[i] for i in sorted_idx]

    gt_dict = load_gt(image_paths)

    # ── Header ───────────────────────────────────────────────────────
    print('=' * 70)
    print('Character-Level Attention Mapping')
    print('=' * 70)
    print(f'Images      : {len(image_paths)}')
    print(f'Models      : {len(model_paths)}')
    for info in model_infos:
        print(f'  {info["label"]:>8s}  ({info["run_name"]})')
    print(f'Layer       : {layer}')
    print(f'Gamma       : {gamma}')
    print(f'Alpha       : {alpha}')
    print(f'Max chars   : {"all" if max_chars == 0 else max_chars}')
    print(f'Output      : {out_dir}')
    print(f'  └─ carpet/     character attention panels (sample_attention style)')
    print(f'  └─ alignment/  character × patch matrix')
    print(f'  └─ combined/   all-in-one (image + carpet + matrix)')
    print(f'Timestamp   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    saved = 0

    for mi, (info, mp) in enumerate(zip(model_infos, model_paths)):
        print(f'[{mi + 1}/{len(model_paths)}] {info["label"]} ({info["run_name"]})')
        net = build_model(conf, mp, num_classes, info['num_registers'], device)

        for img_path in image_paths:
            image_name = Path(img_path).stem
            raw_img = load_image(img_path)
            processed = preprocess(raw_img, fixed_size)
            img_tensor = (torch.from_numpy(processed).float()
                          .unsqueeze(0).unsqueeze(0))

            result = extract_attention(net, img_tensor, device, layer)
            chars, positions = ctc_char_positions(result['logits'], i2c)
            pred_text = ''.join(chars).strip()

            # Trim leading/trailing whitespace (CTC boundary artefacts)
            while chars and chars[0] in (' ', '\t'):
                chars.pop(0); positions.pop(0)
            while chars and chars[-1] in (' ', '\t'):
                chars.pop(); positions.pop()

            gt_text = gt_dict.get(image_name, '')

            char_vectors, char_normed, alignment = get_char_attention(
                result['attn'], result['num_reg'], positions,
                result['grid_size'], gamma)

            suffix = f'_{info["run_name"]}'

            # Carpet (sample_attention.png style)
            sp = os.path.join(carpet_dir, f'{image_name}{suffix}.png')
            out = visualize_char_attention_carpet(
                img_path, chars, char_normed, result['grid_size'],
                fixed_size, info['label'], gt_text, pred_text,
                sp, alpha, dpi, max_chars)
            if out:
                saved += 1

            # Alignment matrix
            sp = os.path.join(alignment_dir, f'{image_name}{suffix}.png')
            out = visualize_alignment_matrix(
                img_path, chars, alignment, result['grid_size'],
                fixed_size, info['label'], gt_text, pred_text,
                sp, dpi, max_chars)
            if out:
                saved += 1

            # Combined
            sp = os.path.join(combined_dir, f'{image_name}{suffix}.png')
            out = visualize_combined(
                img_path, chars, char_normed, alignment, result['grid_size'],
                fixed_size, info['label'], gt_text, pred_text,
                sp, alpha, dpi, max_chars)
            if out:
                saved += 1

            print(f'  {image_name:>30s}  →  "{pred_text}"'
                  f'  (GT: "{gt_text}")')

        del net
        gc.collect()
        print()

    print(f'Done! {saved} figures saved → {out_dir}/')
    print(f'  carpet/    : {carpet_dir}/')
    print(f'  alignment/ : {alignment_dir}/')
    print(f'  combined/  : {combined_dir}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
