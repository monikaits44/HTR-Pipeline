#!/usr/bin/env python3
"""
Character-Level Attention Grid Visualization for ViT-RGTS HTR

Produces publication-ready grid figures showing WHERE in the input image the
model looks when predicting each character — adapted for line-level HTR from
the character-localization grids used in diffusion-based HTR models.

═══════════════════════════════════════════════════════════════════════════════
 Scientific Adaptation: Cross-Attention → Self-Attention + CTC
═══════════════════════════════════════════════════════════════════════════════

In U-Net / diffusion HTR models (e.g., WordStylist), cross-attention between
predefined text embeddings T and image features I directly produces attention
maps  A ∈ ℝ^{B × L × H × W} ,  where L is the max character sequence length.
Each  Aₗ  is a spatial map showing where character l is localized in the image.

In our ViT-RGTS + CTC architecture there is no cross-attention with text.
Instead we *construct equivalent per-character spatial maps* from the encoder's
self-attention and the CTC decode:

  1. Self-attention maps:  A^(ℓ) ∈ ℝ^{B × H × S × S}
     where  S = R + T  (R register tokens + T patch tokens),
     H = number of attention heads, ℓ = encoder layer index.

  2. CTC greedy decode yields characters  c₁ … cₙ  at CTC output positions
     t₁ … tₙ   (positions where the most-likely class changes).

  3. Per-character attention:
       Attention(cᵢ) = A^(ℓ)[0, :, R+tᵢ, R : R+T]   (head-averaged)
     This extracts the attention distribution FROM the encoder token that
     produced character cᵢ TO all spatial patch tokens.

  4. Reshape [T] → [Hp, Wp] → upscale to image size via nearest-neighbor,
     accounting for the preprocessing border (8 px).

This adaptation yields the same character-localization grid as cross-attention
models, but uses self-attention alignment instead of text–image interaction.

For **line-level** data the sequence length L can reach 50+ characters.
The mechanism is identical — CTC naturally handles variable-length outputs.
The resulting grids simply have more columns.

═══════════════════════════════════════════════════════════════════════════════
 Visualization Modes
═══════════════════════════════════════════════════════════════════════════════

  char-grid      Rows = sample images,  Cols = character positions.
                 Multi-image grid matching the reference Fig. 5 layout.

  layer-grid     Rows = encoder layers (1–depth), Cols = characters.
                 Shows how character attention evolves across depth.
                 (Ref: https://alessiodevoto.github.io/vit-attention/)

  head-grid      Rows = attention heads (1–H),  Cols = characters.
                 Reveals which heads specialize in character localization.
                 (Ref: https://alessiodevoto.github.io/vit-attention/)

  gradcam-grid   Rows = sample images,  Cols = character positions.
                 Per-character Grad-CAM (Selvaraju et al. 2017) showing
                 which spatial regions are *causally* important for each
                 character's prediction.

Usage:
    # Character grid across multiple sample images (like reference Fig. 5)
    python scripts/postprocessing/character_attention_grid.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_55/model.pt \\
        --mode char-grid \\
        -- notebook/sample_images/

    # Per-layer attention evolution for one image
    python scripts/postprocessing/character_attention_grid.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_55/model.pt \\
        --mode layer-grid \\
        -- notebook/sample_images/a01-091-10.png

    # Per-head specialization for one image
    python scripts/postprocessing/character_attention_grid.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_55/model.pt \\
        --mode head-grid --layer last \\
        -- notebook/sample_images/a01-091-10.png

    # Per-character Grad-CAM
    python scripts/postprocessing/character_attention_grid.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_55/model.pt \\
        --mode gradcam-grid \\
        -- notebook/sample_images/

Arguments:
    config.yaml        Base config file(s), merged in order
    --model-path P     Checkpoint — repeat once per model
    --                 Separator; image paths/directories follow
    --mode MODE        char-grid | layer-grid | head-grid | gradcam-grid
    --layer LAYER      first | middle | last  (default: last)
    --alpha FLOAT      Heatmap overlay opacity (default: 0.45)
    --dpi INT          Output resolution (default: 150)
    --gamma FLOAT      Contrast exponent (default: 3.0)
    --max-chars INT    Max character columns (default: 0 = all)
    --save-dir DIR     Output directory
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
    mode = 'char-grid'

    filtered = []
    i = 0
    while i < len(config_args):
        a = config_args[i]
        if a == '--model-path' and i + 1 < len(config_args):
            model_paths.append(config_args[i + 1]); i += 2
        elif a == '--save-dir' and i + 1 < len(config_args):
            save_dir = config_args[i + 1]; i += 2
        elif a == '--mode' and i + 1 < len(config_args):
            mode = config_args[i + 1]; i += 2
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

    valid_modes = {'char-grid', 'layer-grid', 'head-grid', 'gradcam-grid'}
    if mode not in valid_modes:
        print(f'Error: --mode must be one of {valid_modes} (got: {mode})')
        sys.exit(1)

    valid_layers = {'first', 'middle', 'last'}
    if layer not in valid_layers:
        print(f'Error: --layer must be one of {valid_layers} (got: {layer})')
        sys.exit(1)

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

    return (conf, model_paths, image_paths, save_dir, mode,
            alpha, dpi, gamma, max_chars, layer)


# ═════════════════════════════════════════════════════════════════════════════
# Model Utilities
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
    """Greedy CTC decode → (characters, CTC-output-positions)."""
    tdec = logits_np.argmax(2).squeeze()
    chars, positions = [], []
    prev = -1
    for t, v in enumerate(tdec):
        v = int(v)
        if v != blank_id and v != prev:
            chars.append(str(i2c.get(v, '?')))
            positions.append(t)
        prev = v
    return chars, positions


def _char_display(c):
    """Display label for a character (spaces → SPC)."""
    c = str(c)
    if c == ' ':
        return 'SPC'
    if c in ('\t', '\n'):
        return repr(c)
    return c


# ═════════════════════════════════════════════════════════════════════════════
# Normalization
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_gamma(vec, gamma=3.0, percentile_clip=99.0):
    """Percentile-clip → [0,1] → γ contrast boost."""
    p_lo = max(0.0, np.percentile(vec, 100.0 - percentile_clip))
    p_hi = np.percentile(vec, percentile_clip)
    if p_hi - p_lo < 1e-8:
        p_lo, p_hi = vec.min(), vec.max()
    if p_hi - p_lo < 1e-8:
        return np.zeros_like(vec)
    normed = np.clip(vec, p_lo, p_hi)
    normed = (normed - p_lo) / (p_hi - p_lo + 1e-8)
    return normed ** gamma


# ═════════════════════════════════════════════════════════════════════════════
# Upscaling: patch grid → image coordinates
# ═════════════════════════════════════════════════════════════════════════════

def upscale_to_image(patch_map, H_img, W_img, fixed_size, border_size=8):
    """Upscale (Hp, Wp) attention grid to (H_img, W_img) image size."""
    Hp, Wp = patch_map.shape
    H_pre, W_pre = fixed_size
    ph = H_pre // max(Hp, 1)
    pw = W_pre // max(Wp, 1)
    heat_full = np.kron(patch_map, np.ones((ph, pw)))
    r0, c0 = border_size, border_size
    crop = heat_full[r0:r0 + H_img, c0:c0 + W_img]
    if crop.shape[0] < H_img or crop.shape[1] < W_img:
        padded = np.zeros((H_img, W_img), dtype=crop.dtype)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    return crop


# ═════════════════════════════════════════════════════════════════════════════
# Attention Extraction — Per-character (single layer, head-averaged)
# ═════════════════════════════════════════════════════════════════════════════

def _layer_index(layer_name, n_layers):
    return {'first': 0, 'middle': n_layers // 2, 'last': n_layers - 1}[layer_name]


def extract_char_attention(net, img_tensor, device, i2c,
                           layer='last', gamma=3.0):
    """
    Head-averaged per-character attention at a chosen layer.

    For character cᵢ at CTC position tᵢ:
        vec = A^(ℓ)[:, :, R+tᵢ, R:R+T].mean(heads)   →  [T]

    Returns dict: chars, positions, char_heatmaps [(Hp,Wp)], grid_size,
                  logits, num_reg, n_layers, n_heads, pred_text.
    """
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits, reg_out, attn_maps, token_norms, (Hp, Wp) = \
            net.forward_explain(img_tensor)

    if isinstance(logits, tuple):
        logits = logits[0]

    chars, positions = ctc_char_positions(logits.cpu().numpy(), i2c)
    num_reg = reg_out.shape[1] if reg_out is not None else 0
    T = Hp * Wp
    L = len(attn_maps)
    li = _layer_index(layer, L)

    attn = attn_maps[li][0].mean(dim=0).numpy()          # [S, S]

    char_heatmaps = []
    for ci, t in enumerate(positions):
        row_idx = num_reg + t
        if row_idx < attn.shape[0]:
            vec = attn[row_idx, num_reg:num_reg + T].copy()
        else:
            vec = np.ones(T) / T
        char_heatmaps.append(_normalize_gamma(vec, gamma).reshape(Hp, Wp))

    pred_text = ''.join(chars).strip()
    return {
        'chars': chars, 'positions': positions,
        'char_heatmaps': char_heatmaps,
        'grid_size': (Hp, Wp),
        'logits': logits.cpu().numpy(),
        'num_reg': num_reg,
        'n_layers': L,
        'n_heads': attn_maps[0].shape[1],
        'pred_text': pred_text,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Attention Extraction — Per-layer (all layers, head-averaged)
# ═════════════════════════════════════════════════════════════════════════════

def extract_layer_char_attention(net, img_tensor, device, i2c, gamma=3.0):
    """
    Per-layer, per-character attention maps (head-averaged).

    Returns dict with layer_heatmaps: [n_layers][n_chars] of (Hp, Wp).
    """
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits, reg_out, attn_maps, token_norms, (Hp, Wp) = \
            net.forward_explain(img_tensor)

    if isinstance(logits, tuple):
        logits = logits[0]

    chars, positions = ctc_char_positions(logits.cpu().numpy(), i2c)
    num_reg = reg_out.shape[1] if reg_out is not None else 0
    T = Hp * Wp

    layer_heatmaps = []
    for li in range(len(attn_maps)):
        attn = attn_maps[li][0].mean(dim=0).numpy()      # [S, S]
        per_char = []
        for ci, t in enumerate(positions):
            row_idx = num_reg + t
            if row_idx < attn.shape[0]:
                vec = attn[row_idx, num_reg:num_reg + T].copy()
            else:
                vec = np.ones(T) / T
            per_char.append(_normalize_gamma(vec, gamma).reshape(Hp, Wp))
        layer_heatmaps.append(per_char)

    pred_text = ''.join(chars).strip()
    return {
        'chars': chars, 'positions': positions,
        'layer_heatmaps': layer_heatmaps,
        'grid_size': (Hp, Wp),
        'logits': logits.cpu().numpy(),
        'num_reg': num_reg,
        'n_layers': len(attn_maps),
        'pred_text': pred_text,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Attention Extraction — Per-head (single layer, all heads)
# ═════════════════════════════════════════════════════════════════════════════

def extract_head_char_attention(net, img_tensor, device, i2c,
                                layer='last', gamma=3.0):
    """
    Per-head, per-character attention at a chosen layer.

    Returns dict with head_heatmaps: [n_heads][n_chars] of (Hp, Wp).
    """
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits, reg_out, attn_maps, token_norms, (Hp, Wp) = \
            net.forward_explain(img_tensor)

    if isinstance(logits, tuple):
        logits = logits[0]

    chars, positions = ctc_char_positions(logits.cpu().numpy(), i2c)
    num_reg = reg_out.shape[1] if reg_out is not None else 0
    T = Hp * Wp
    L = len(attn_maps)
    li = _layer_index(layer, L)

    attn_layer = attn_maps[li][0].numpy()                 # [H, S, S]
    n_heads = attn_layer.shape[0]

    head_heatmaps = []
    for h in range(n_heads):
        per_char = []
        for ci, t in enumerate(positions):
            row_idx = num_reg + t
            if row_idx < attn_layer.shape[1]:
                vec = attn_layer[h, row_idx, num_reg:num_reg + T].copy()
            else:
                vec = np.ones(T) / T
            per_char.append(_normalize_gamma(vec, gamma).reshape(Hp, Wp))
        head_heatmaps.append(per_char)

    pred_text = ''.join(chars).strip()
    return {
        'chars': chars, 'positions': positions,
        'head_heatmaps': head_heatmaps,
        'grid_size': (Hp, Wp),
        'logits': logits.cpu().numpy(),
        'num_reg': num_reg,
        'n_heads': n_heads,
        'pred_text': pred_text,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Per-Character Grad-CAM  (Selvaraju et al. 2017)
# ═════════════════════════════════════════════════════════════════════════════

def compute_char_gradcam(net, img_tensor, device, i2c, gamma=3.0):
    """
    Per-character Grad-CAM via individual backward passes.

    For each decoded character cᵢ at CTC position tᵢ:
      1. target = logits[tᵢ, 0, predicted_class]
      2. Backward through CTC head → backbone activations
      3. α_d = (1/T) Σ_t  ∂target/∂A_{t,d}       (global avg pool of grads)
      4. cam_t = ReLU( Σ_d  α_d · A_{t,d} )       (weighted combination)

    This produces a spatial importance map specific to each character's
    prediction, showing which image regions are *causally* important.

    Returns dict: chars, positions, char_cams [(Hp,Wp)], grid_size, pred_text.
    """
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(False)

    # Backbone forward (WITHOUT torch.no_grad — need gradient graph)
    seq_tokens, reg_tokens, (Hp, Wp) = net.backbone(img_tensor)
    seq_tokens.retain_grad()

    # CTC head forward
    seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)    # [1, D, 1, T]
    logits = net.top(seq_4d)
    if isinstance(logits, tuple):
        logits = logits[0]                                 # [T, 1, C]

    chars, positions = ctc_char_positions(
        logits.detach().cpu().numpy(), i2c)

    T = Hp * Wp
    char_cams = []
    n_chars = len(positions)

    for ci, t in enumerate(positions):
        net.zero_grad()
        if seq_tokens.grad is not None:
            seq_tokens.grad.zero_()

        pred_class = logits[t, 0].argmax()
        target = logits[t, 0, pred_class]
        target.backward(retain_graph=(ci < n_chars - 1))

        grads = seq_tokens.grad.detach()                  # [T, 1, D]
        acts = seq_tokens.detach()                         # [T, 1, D]
        alpha = grads.mean(dim=0, keepdim=True)            # [1, 1, D]  GAP
        cam = (alpha * acts).squeeze(1).sum(dim=-1)        # [T]
        cam = torch.relu(cam).cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Gamma contrast
        cam = cam ** gamma
        char_cams.append(cam.reshape(Hp, Wp))

    pred_text = ''.join(chars).strip()
    return {
        'chars': chars, 'positions': positions,
        'char_cams': char_cams,
        'grid_size': (Hp, Wp),
        'logits': logits.detach().cpu().numpy(),
        'pred_text': pred_text,
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
# Visualization: Character Grid  (rows=images, cols=characters)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_char_grid(image_results, model_label, save_path,
                        alpha, dpi, max_chars, layer_name):
    """
    Publication-ready multi-image character attention grid (Fig. 5 style).

    Each cell shows the original image overlaid with per-character attention.
    Column headers: "Char 1", "Char 2", …
    Character identity shown as a small label in each cell's top-left corner.
    """
    n_rows = len(image_results)
    n_cols = max((len(r['chars']) for r in image_results), default=0)
    if max_chars > 0:
        n_cols = min(n_cols, max_chars)
    if n_cols == 0 or n_rows == 0:
        return None

    # ── Cell sizing ──────────────────────────────────────────────────
    img0 = np.array(Image.open(image_results[0]['img_path']).convert('L'))
    h_img, w_img = img0.shape
    img_aspect = w_img / max(h_img, 1)

    cell_h = max(0.6, min(1.8, 18.0 / max(n_rows, 1)))
    cell_w = cell_h * min(img_aspect, 4.0)
    cell_w = max(0.6, min(cell_w, 45.0 / max(n_cols, 1)))

    fig_w = n_cols * cell_w + 1.2
    fig_h = n_rows * cell_h + 1.0
    hdr_fs = max(5, min(10, int(100 / max(n_cols, 1))))
    lbl_fs = max(4, min(7, int(80 / max(n_cols, 1))))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              squeeze=False)

    for ri, res in enumerate(image_results):
        img_np = np.array(Image.open(res['img_path']).convert('L'))
        chars = res['chars']
        heatmaps = res['heatmaps']

        for ci in range(n_cols):
            ax = axes[ri, ci]
            if ci < len(chars) and ci < len(heatmaps):
                ax.imshow(img_np, cmap='gray', aspect='auto')
                ax.imshow(heatmaps[ci], cmap='inferno', alpha=alpha,
                          vmin=0, vmax=1, aspect='auto')

                # Character identity label
                ch = _char_display(chars[ci])
                ax.text(0.03, 0.95, f'"{ch}"', transform=ax.transAxes,
                        fontsize=lbl_fs, fontweight='bold', color='white',
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.12',
                                  facecolor='black', alpha=0.6,
                                  edgecolor='none'))

                if ri == 0:
                    ax.set_title(f'Char {ci + 1}', fontsize=hdr_fs,
                                 fontweight='bold', pad=3)
            else:
                ax.set_facecolor('#f5f5f5')

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)

        # Row label
        pred = res.get('pred_text', '')
        if len(pred) > 25:
            pred = pred[:22] + '…'
        axes[ri, 0].set_ylabel(f'"{pred}"',
                                fontsize=max(5, min(8, int(90 / max(n_rows, 1)))),
                                rotation=0, ha='right', va='center',
                                labelpad=10, fontweight='bold')

    fig.suptitle(f'Character Attention Grid  |  {model_label}  |  layer={layer_name}',
                 fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout(pad=0.3, h_pad=0.15, w_pad=0.15)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Visualization: Layer Grid  (rows=layers, cols=characters)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_layer_grid(result, img_path, model_label, save_path,
                         alpha, dpi, max_chars):
    """
    Single-image grid showing how per-character attention evolves across
    encoder layers — inspired by the per-layer ViT attention visualizations
    at https://alessiodevoto.github.io/vit-attention/.
    """
    chars = result['chars']
    layer_heatmaps = result['layer_heatmaps']
    n_layers = result['n_layers']

    n_cols = len(chars)
    if max_chars > 0:
        n_cols = min(n_cols, max_chars)
    if n_cols == 0:
        return None

    img_np = np.array(Image.open(img_path).convert('L'))
    h_img, w_img = img_np.shape
    img_aspect = w_img / max(h_img, 1)

    cell_h = max(0.7, min(1.6, 14.0 / max(n_layers, 1)))
    cell_w = cell_h * min(img_aspect, 4.0)
    cell_w = max(0.6, min(cell_w, 42.0 / max(n_cols, 1)))

    fig_w = n_cols * cell_w + 1.5
    fig_h = n_layers * cell_h + 1.2
    hdr_fs = max(5, min(9, int(90 / max(n_cols, 1))))
    row_fs = max(5, min(8, int(80 / max(n_layers, 1))))

    fig, axes = plt.subplots(n_layers, n_cols, figsize=(fig_w, fig_h),
                              squeeze=False)

    fixed_size = (result.get('fixed_size_h', 64),
                  result.get('fixed_size_w', 1024))

    for li in range(n_layers):
        for ci in range(n_cols):
            ax = axes[li, ci]
            if ci < len(chars) and ci < len(layer_heatmaps[li]):
                hm = upscale_to_image(layer_heatmaps[li][ci],
                                      h_img, w_img, fixed_size)
                ax.imshow(img_np, cmap='gray', aspect='auto')
                ax.imshow(hm, cmap='inferno', alpha=alpha,
                          vmin=0, vmax=1, aspect='auto')

                if li == 0:
                    ch = _char_display(chars[ci])
                    ax.set_title(f'"{ch}"', fontsize=hdr_fs,
                                 fontweight='bold', color='#1a237e', pad=3)
            else:
                ax.set_facecolor('#f5f5f5')

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)

        # Row label
        axes[li, 0].set_ylabel(f'Layer {li + 1}', fontsize=row_fs,
                                rotation=0, ha='right', va='center',
                                labelpad=10, fontweight='bold')

    image_name = Path(img_path).stem
    pred = result.get('pred_text', '')
    fig.suptitle(f'Per-Layer Character Attention  |  {image_name}'
                 f'  |  {model_label}  |  pred: "{pred}"',
                 fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout(pad=0.3, h_pad=0.15, w_pad=0.15)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Visualization: Head Grid  (rows=heads, cols=characters)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_head_grid(result, img_path, model_label, layer_name,
                        save_path, alpha, dpi, max_chars):
    """
    Single-image grid showing per-head, per-character attention at one layer.
    Reveals which heads specialize in spatial character localization.
    """
    chars = result['chars']
    head_heatmaps = result['head_heatmaps']
    n_heads = result['n_heads']

    n_cols = len(chars)
    if max_chars > 0:
        n_cols = min(n_cols, max_chars)
    if n_cols == 0:
        return None

    img_np = np.array(Image.open(img_path).convert('L'))
    h_img, w_img = img_np.shape
    img_aspect = w_img / max(h_img, 1)

    cell_h = max(0.7, min(1.5, 14.0 / max(n_heads, 1)))
    cell_w = cell_h * min(img_aspect, 4.0)
    cell_w = max(0.6, min(cell_w, 42.0 / max(n_cols, 1)))

    fig_w = n_cols * cell_w + 1.5
    fig_h = n_heads * cell_h + 1.2
    hdr_fs = max(5, min(9, int(90 / max(n_cols, 1))))
    row_fs = max(5, min(8, int(80 / max(n_heads, 1))))

    fig, axes = plt.subplots(n_heads, n_cols, figsize=(fig_w, fig_h),
                              squeeze=False)

    fixed_size = (result.get('fixed_size_h', 64),
                  result.get('fixed_size_w', 1024))

    for hi in range(n_heads):
        for ci in range(n_cols):
            ax = axes[hi, ci]
            if ci < len(chars) and ci < len(head_heatmaps[hi]):
                hm = upscale_to_image(head_heatmaps[hi][ci],
                                      h_img, w_img, fixed_size)
                ax.imshow(img_np, cmap='gray', aspect='auto')
                ax.imshow(hm, cmap='inferno', alpha=alpha,
                          vmin=0, vmax=1, aspect='auto')

                if hi == 0:
                    ch = _char_display(chars[ci])
                    ax.set_title(f'"{ch}"', fontsize=hdr_fs,
                                 fontweight='bold', color='#1a237e', pad=3)
            else:
                ax.set_facecolor('#f5f5f5')

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)

        axes[hi, 0].set_ylabel(f'Head {hi + 1}', fontsize=row_fs,
                                rotation=0, ha='right', va='center',
                                labelpad=10, fontweight='bold')

    image_name = Path(img_path).stem
    pred = result.get('pred_text', '')
    fig.suptitle(f'Per-Head Character Attention  |  {image_name}'
                 f'  |  {model_label}  |  layer={layer_name}'
                 f'  |  pred: "{pred}"',
                 fontsize=9, fontweight='bold', y=1.02)

    plt.tight_layout(pad=0.3, h_pad=0.15, w_pad=0.15)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Visualization: Grad-CAM Grid  (rows=images, cols=characters)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_gradcam_grid(image_results, model_label, save_path,
                           alpha, dpi, max_chars):
    """
    Multi-image per-character Grad-CAM grid.
    Same layout as char-grid but uses gradient-weighted class activation maps
    instead of raw attention.
    """
    n_rows = len(image_results)
    n_cols = max((len(r['chars']) for r in image_results), default=0)
    if max_chars > 0:
        n_cols = min(n_cols, max_chars)
    if n_cols == 0 or n_rows == 0:
        return None

    img0 = np.array(Image.open(image_results[0]['img_path']).convert('L'))
    h_img, w_img = img0.shape
    img_aspect = w_img / max(h_img, 1)

    cell_h = max(0.6, min(1.8, 18.0 / max(n_rows, 1)))
    cell_w = cell_h * min(img_aspect, 4.0)
    cell_w = max(0.6, min(cell_w, 45.0 / max(n_cols, 1)))

    fig_w = n_cols * cell_w + 1.2
    fig_h = n_rows * cell_h + 1.0
    hdr_fs = max(5, min(10, int(100 / max(n_cols, 1))))
    lbl_fs = max(4, min(7, int(80 / max(n_cols, 1))))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              squeeze=False)

    for ri, res in enumerate(image_results):
        img_np = np.array(Image.open(res['img_path']).convert('L'))
        chars = res['chars']
        heatmaps = res['heatmaps']

        for ci in range(n_cols):
            ax = axes[ri, ci]
            if ci < len(chars) and ci < len(heatmaps):
                ax.imshow(img_np, cmap='gray', aspect='auto')
                ax.imshow(heatmaps[ci], cmap='magma', alpha=alpha,
                          vmin=0, vmax=1, aspect='auto')

                ch = _char_display(chars[ci])
                ax.text(0.03, 0.95, f'"{ch}"', transform=ax.transAxes,
                        fontsize=lbl_fs, fontweight='bold', color='white',
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.12',
                                  facecolor='black', alpha=0.6,
                                  edgecolor='none'))

                if ri == 0:
                    ax.set_title(f'Char {ci + 1}', fontsize=hdr_fs,
                                 fontweight='bold', pad=3)
            else:
                ax.set_facecolor('#f5f5f5')

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)

        pred = res.get('pred_text', '')
        if len(pred) > 25:
            pred = pred[:22] + '…'
        axes[ri, 0].set_ylabel(f'"{pred}"',
                                fontsize=max(5, min(8, int(90 / max(n_rows, 1)))),
                                rotation=0, ha='right', va='center',
                                labelpad=10, fontweight='bold')

    fig.suptitle(f'Per-Character Grad-CAM Grid  |  {model_label}',
                 fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout(pad=0.3, h_pad=0.15, w_pad=0.15)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    (conf, model_paths, image_paths, save_dir, mode,
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
               else os.path.join('visualizations', 'character_attention_grid'))
    mode_dir = os.path.join(out_dir, mode)

    import shutil
    if os.path.isdir(mode_dir):
        shutil.rmtree(mode_dir)
    os.makedirs(mode_dir, exist_ok=True)

    model_infos = [get_run_info(mp) for mp in model_paths]
    sorted_idx = sorted(range(len(model_infos)),
                        key=lambda i: model_infos[i]['num_registers'])
    model_infos = [model_infos[i] for i in sorted_idx]
    model_paths = [model_paths[i] for i in sorted_idx]

    gt_dict = load_gt(image_paths)

    # ── Header ───────────────────────────────────────────────────────
    print('=' * 70)
    print('Character-Level Attention Grid Visualization')
    print('=' * 70)
    print(f'Mode        : {mode}')
    print(f'Images      : {len(image_paths)}')
    print(f'Models      : {len(model_paths)}')
    for info in model_infos:
        print(f'  {info["label"]:>8s}  ({info["run_name"]})')
    print(f'Layer       : {layer}')
    print(f'Gamma       : {gamma}')
    print(f'Alpha       : {alpha}')
    print(f'Max chars   : {"all" if max_chars == 0 else max_chars}')
    print(f'Output      : {mode_dir}/')
    print(f'Timestamp   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    saved = 0

    for mi, (info, mp) in enumerate(zip(model_infos, model_paths)):
        print(f'[{mi + 1}/{len(model_paths)}] {info["label"]} ({info["run_name"]})')
        net = build_model(conf, mp, num_classes, info['num_registers'], device)

        # ──────────────────────────────────────────────────────────────
        # char-grid: one figure per model, all images as rows
        # ──────────────────────────────────────────────────────────────
        if mode == 'char-grid':
            image_results = []
            for img_path in image_paths:
                image_name = Path(img_path).stem
                raw = load_image(img_path)
                processed = preprocess(raw, fixed_size)
                img_tensor = (torch.from_numpy(processed).float()
                              .unsqueeze(0).unsqueeze(0))

                res = extract_char_attention(net, img_tensor, device, i2c,
                                             layer=layer, gamma=gamma)
                # Trim whitespace
                chars, positions = res['chars'], res['positions']
                heatmaps = res['char_heatmaps']
                while chars and chars[0] in (' ', '\t'):
                    chars.pop(0); positions.pop(0); heatmaps.pop(0)
                while chars and chars[-1] in (' ', '\t'):
                    chars.pop(); positions.pop(); heatmaps.pop()

                img_np = np.array(Image.open(img_path).convert('L'))
                h_img, w_img = img_np.shape
                heatmaps_up = [upscale_to_image(hm, h_img, w_img, fixed_size)
                               for hm in heatmaps]

                gt_text = gt_dict.get(image_name, '')
                image_results.append({
                    'img_path': img_path,
                    'chars': chars,
                    'heatmaps': heatmaps_up,
                    'pred_text': res['pred_text'],
                    'gt_text': gt_text,
                })
                print(f'    {image_name:>25s}  →  "{res["pred_text"]}"')

            fname = f'{info["label"]}_{info["run_name"]}_char_grid.png'
            sp = os.path.join(mode_dir, fname)
            out = visualize_char_grid(image_results, info['label'], sp,
                                      alpha, dpi, max_chars, layer)
            if out:
                saved += 1

        # ──────────────────────────────────────────────────────────────
        # layer-grid: one figure per model per image
        # ──────────────────────────────────────────────────────────────
        elif mode == 'layer-grid':
            for img_path in image_paths:
                image_name = Path(img_path).stem
                raw = load_image(img_path)
                processed = preprocess(raw, fixed_size)
                img_tensor = (torch.from_numpy(processed).float()
                              .unsqueeze(0).unsqueeze(0))

                res = extract_layer_char_attention(
                    net, img_tensor, device, i2c, gamma=gamma)
                res['fixed_size_h'] = fixed_size[0]
                res['fixed_size_w'] = fixed_size[1]

                fname = (f'{image_name}_{info["label"]}'
                         f'_{info["run_name"]}_layer_grid.png')
                sp = os.path.join(mode_dir, fname)
                out = visualize_layer_grid(res, img_path, info['label'],
                                           sp, alpha, dpi, max_chars)
                if out:
                    saved += 1
                print(f'    {image_name:>25s}  →  "{res["pred_text"]}"')

        # ──────────────────────────────────────────────────────────────
        # head-grid: one figure per model per image
        # ──────────────────────────────────────────────────────────────
        elif mode == 'head-grid':
            for img_path in image_paths:
                image_name = Path(img_path).stem
                raw = load_image(img_path)
                processed = preprocess(raw, fixed_size)
                img_tensor = (torch.from_numpy(processed).float()
                              .unsqueeze(0).unsqueeze(0))

                res = extract_head_char_attention(
                    net, img_tensor, device, i2c, layer=layer, gamma=gamma)
                res['fixed_size_h'] = fixed_size[0]
                res['fixed_size_w'] = fixed_size[1]

                fname = (f'{image_name}_{info["label"]}'
                         f'_{info["run_name"]}_head_grid.png')
                sp = os.path.join(mode_dir, fname)
                out = visualize_head_grid(res, img_path, info['label'],
                                          layer, sp, alpha, dpi, max_chars)
                if out:
                    saved += 1
                print(f'    {image_name:>25s}  →  "{res["pred_text"]}"')

        # ──────────────────────────────────────────────────────────────
        # gradcam-grid: one figure per model, all images as rows
        # ──────────────────────────────────────────────────────────────
        elif mode == 'gradcam-grid':
            image_results = []
            for img_path in image_paths:
                image_name = Path(img_path).stem
                raw = load_image(img_path)
                processed = preprocess(raw, fixed_size)
                img_tensor = (torch.from_numpy(processed).float()
                              .unsqueeze(0).unsqueeze(0))

                res = compute_char_gradcam(net, img_tensor, device, i2c,
                                           gamma=gamma)

                chars, positions = res['chars'], res['positions']
                cams = res['char_cams']
                while chars and chars[0] in (' ', '\t'):
                    chars.pop(0); positions.pop(0); cams.pop(0)
                while chars and chars[-1] in (' ', '\t'):
                    chars.pop(); positions.pop(); cams.pop()

                img_np = np.array(Image.open(img_path).convert('L'))
                h_img, w_img = img_np.shape
                cams_up = [upscale_to_image(cm, h_img, w_img, fixed_size)
                           for cm in cams]

                gt_text = gt_dict.get(image_name, '')
                image_results.append({
                    'img_path': img_path,
                    'chars': chars,
                    'heatmaps': cams_up,
                    'pred_text': res['pred_text'],
                    'gt_text': gt_text,
                })
                print(f'    {image_name:>25s}  →  "{res["pred_text"]}"'
                      f'  ({len(chars)} chars, Grad-CAM)')

            fname = f'{info["label"]}_{info["run_name"]}_gradcam_grid.png'
            sp = os.path.join(mode_dir, fname)
            out = visualize_gradcam_grid(image_results, info['label'], sp,
                                         alpha, dpi, max_chars)
            if out:
                saved += 1

        del net
        gc.collect()
        print()

    print(f'Done! {saved} figures saved → {mode_dir}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
