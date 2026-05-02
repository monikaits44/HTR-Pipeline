#!/usr/bin/env python3
"""
Cross-Model Attention & Grad-CAM Comparison for ViT-RGTS

Compares character-level attention maps AND Grad-CAM patch importance
across multiple ViT-RGTS models with different register counts.
Designed for the register-sweep experiments (run_50 … run_54).

Generates TWO separate figures per image:
  1. *_attention.png  — Character-level attention comparison
     Rows = models (Reg-0 → Reg-16), Cols = [Original | char₁ | … | charₙ]
  2. *_gradcam.png    — Grad-CAM patch importance comparison
     Rows = models (Reg-0 → Reg-16), Cols = [Original | Grad-CAM]

Attention method — last-layer head-averaged attention:
  1. Take the last transformer layer's attention A_L ∈ [B, H, S, S]
  2. Average across heads → [S, S]
  3. For each predicted character c at CTC time step t:
     row (num_reg + t) → [T] patch importance → reshape (Hp, Wp)
  4. Power contrast (γ=3) amplifies peaks for clean heatmaps

Grad-CAM method — gradient-based patch importance:
  backbone → retain grad on seq_tokens → CTC head → backward from sum of
  predicted-class logits → importance = mean(|grad|) per token

Register count is auto-detected from each run's config.json.
Models are loaded one at a time to conserve memory.

Supports four execution modes:
  1. Test set  (default) — every sample in data/IAM/test
  2. Single image        — one image after the -- separator
  3. Multiple images     — several images after --
  4. Directory           — a folder of images after --

Output: visualizations/cross_model_comparison/

Usage:
    # ── Test set mode (default — no images after --) ───────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_51/model.pt \\
        --model-path saved_models/experiments/run_52/model.pt \\
        --model-path saved_models/experiments/run_53/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt

    # ── Single image ───────────────────────────────────────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_51/model.pt \\
        --model-path saved_models/experiments/run_52/model.pt \\
        --model-path saved_models/experiments/run_53/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        -- notebook/sample_images/a01-038-12.png

    # ── Multiple images ────────────────────────────────────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        -- img1.png img2.png img3.png

    # ── Directory of images ────────────────────────────────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        -- notebook/sample_images/

    # ── Attention maps only (skip Grad-CAM) ────────────────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --mode attn \\
        -- notebook/sample_images/a01-038-12.png

    # ── Grad-CAM only ──────────────────────────────────────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --mode gradcam \\
        -- notebook/sample_images/a01-038-12.png

    # ── Custom settings ────────────────────────────────────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --save-dir output/comparison/ --alpha 0.5 --dpi 200 --max-chars 15 \\
        -- notebook/sample_images/a01-038-12.png

    # ── GPU acceleration ───────────────────────────────────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        device=cuda \\
        -- notebook/sample_images/

    # ── With explicit ground truth ─────────────────────────────────
    python scripts/postprocessing/compare_attention_across_models.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        gt=notebook/sample_images/gt.txt \\
        -- notebook/sample_images/

Arguments:
    config.yaml        Base config file(s), merged in order
    key=value          Override params applied to ALL models (e.g., device=cuda)
    --model-path PATH  Checkpoint path — repeat once per model (min 1)
    --                 Separator; image paths follow (omit for test set mode)
    --mode MODE        attn | gradcam | both (default: both)
    --save-dir PATH    Output directory (default: visualizations/cross_model_comparison/)
    --alpha FLOAT      Heatmap overlay opacity 0.0–1.0 (default: 0.40)
    --dpi INT          Output resolution (default: 150)
    --max-chars INT    Max characters shown per model row (default: 20)
    --layer LAYER      Attention layer(s): first|middle|last|all, comma-separated
                       for multiple (e.g. last,middle). Default: last
    --gamma FLOAT      Contrast exponent for attention maps (default: 3.0)
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
from torch.utils.data import DataLoader
import tqdm
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Project root on sys.path ─────────────────────────────────────────────────
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import HTRNet
from utils.htr_dataset import HTRDataset
from utils.metrics import CER, WER
from utils.preprocessing import load_image, preprocess

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


# ═════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """
    Parse CLI arguments.

    Before '--': config YAMLs + key=value overrides + flags
    After  '--': image path(s); omit for test-set mode

    Returns (config, model_paths, image_paths, save_dir, mode, alpha, dpi,
             max_chars, gamma, layers)
    """
    argv = sys.argv[1:]

    has_sep = '--' in argv
    if has_sep:
        sep = argv.index('--')
        config_args = argv[:sep]
        image_args = argv[sep + 1:]
    else:
        config_args = argv
        image_args = []

    # Defaults
    model_paths = []
    save_dir = ''
    mode = 'both'
    alpha = 0.40
    dpi = 150
    max_chars = 20
    gamma = 3.0
    layer = 'last'

    filtered = []
    i = 0
    while i < len(config_args):
        if config_args[i] == '--model-path' and i + 1 < len(config_args):
            model_paths.append(config_args[i + 1])
            i += 2
        elif config_args[i] == '--save-dir' and i + 1 < len(config_args):
            save_dir = config_args[i + 1]
            i += 2
        elif config_args[i] == '--mode' and i + 1 < len(config_args):
            mode = config_args[i + 1]
            i += 2
        elif config_args[i] == '--alpha' and i + 1 < len(config_args):
            alpha = float(config_args[i + 1])
            i += 2
        elif config_args[i] == '--dpi' and i + 1 < len(config_args):
            dpi = int(config_args[i + 1])
            i += 2
        elif config_args[i] == '--max-chars' and i + 1 < len(config_args):
            max_chars = int(config_args[i + 1])
            i += 2
        elif config_args[i] == '--gamma' and i + 1 < len(config_args):
            gamma = float(config_args[i + 1])
            i += 2
        elif config_args[i] == '--layer' and i + 1 < len(config_args):
            layer = config_args[i + 1]
            i += 2
        else:
            filtered.append(config_args[i])
            i += 1

    yaml_files = [a for a in filtered if a.endswith('.yaml')]
    overrides = [a for a in filtered if not a.endswith('.yaml')]

    if not yaml_files:
        print('Error: at least one .yaml config file is required.')
        sys.exit(1)
    if not model_paths:
        print('Error: at least one --model-path is required.')
        sys.exit(1)
    if mode not in ('attn', 'gradcam', 'both'):
        print(f'Error: --mode must be attn, gradcam, or both (got: {mode})')
        sys.exit(1)
    # Parse layer(s) — supports comma-separated, e.g. "last,middle"
    valid_layers = {'first', 'middle', 'last', 'all', 'rollout'}
    layers = tuple(l.strip() for l in layer.split(','))
    for l in layers:
        if l not in valid_layers:
            print(f'Error: each --layer value must be one of '
                  f'{valid_layers} (got: {l})')
            sys.exit(1)

    # Merge YAMLs
    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))
    OmegaConf.set_struct(conf, False)
    cli_conf = OmegaConf.from_dotlist(overrides)
    conf = OmegaConf.merge(conf, cli_conf)

    # Resolve images
    image_paths = collect_image_paths(image_args)
    if has_sep and not image_paths:
        print('Error: "--" separator used but no valid images found.')
        sys.exit(1)

    return (conf, model_paths, image_paths, save_dir, mode,
            alpha, dpi, max_chars, gamma, layers)


def collect_image_paths(paths):
    """Expand directories into individual image files, keep files as-is."""
    result = []
    for p in paths:
        if os.path.isdir(p):
            for fname in sorted(os.listdir(p)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    result.append(os.path.join(p, fname))
        elif os.path.isfile(p):
            result.append(p)
        else:
            print(f'Warning: skipping {p} (not a file or directory)')
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Ground Truth
# ═════════════════════════════════════════════════════════════════════════════

def load_gt_file(path):
    """Load a ground truth file: {image_stem: gt_text}."""
    gt = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                gt[parts[0]] = parts[1]
            elif len(parts) == 1:
                gt[parts[0]] = ''
    return gt


def find_gt_for_images(image_paths, explicit_gt_path=None):
    """Locate and load GT: explicit gt= path > auto-detect gt.txt."""
    if explicit_gt_path:
        if not os.path.isfile(explicit_gt_path):
            print(f'Warning: specified gt file not found: {explicit_gt_path}')
            return {}
        print(f'Ground truth : {explicit_gt_path}')
        return load_gt_file(explicit_gt_path)

    gt = {}
    seen_dirs = set()
    for img_path in image_paths:
        d = os.path.dirname(os.path.abspath(img_path))
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        candidate = os.path.join(d, 'gt.txt')
        if os.path.isfile(candidate):
            print(f'Ground truth : {candidate} (auto-detected)')
            gt.update(load_gt_file(candidate))
    if not gt:
        print('Ground truth : none (CER / WER will be empty)')
    return gt


# ═════════════════════════════════════════════════════════════════════════════
# Run Info & Model Loading
# ═════════════════════════════════════════════════════════════════════════════

def get_run_info(model_path):
    """Read config.json to discover num_registers and run name."""
    run_dir = os.path.dirname(os.path.abspath(model_path))
    run_name = os.path.basename(run_dir)

    config_json = os.path.join(run_dir, 'config.json')
    if not os.path.isfile(config_json):
        raise FileNotFoundError(
            f'config.json not found in {run_dir}. '
            'Each model directory must contain config.json.'
        )
    with open(config_json) as f:
        cfg = json.load(f)

    num_reg = cfg.get('arch', {}).get('num_registers', 0)
    return {
        'run_dir': run_dir,
        'run_name': run_name,
        'num_registers': num_reg,
        'label': f'Reg-{num_reg}',
    }


def build_model_for_run(base_config, model_path, num_classes,
                        num_registers, device):
    """Build HTRNet with the correct num_registers, load checkpoint."""
    config = deepcopy(base_config)
    OmegaConf.set_struct(config, False)
    config.arch.num_registers = num_registers
    config.resume = model_path

    net = HTRNet(config.arch, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    load_info = net.load_state_dict(checkpoint, strict=True)
    net.to(device)
    # Don't set global train/eval here — process_single_image() sets
    # the correct mode for each operation (eval for attention, selective
    # train for Grad-CAM GRU backward).
    net.eval()

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return net, n_params, load_info


# ═════════════════════════════════════════════════════════════════════════════
# CTC Decoding
# ═════════════════════════════════════════════════════════════════════════════

def ctc_char_positions(logits_np, i2c, blank_id=0):
    """Map each predicted character to its CTC time step."""
    tdec = logits_np.argmax(2).squeeze()
    chars, positions = [], []
    prev = -1
    for t, v in enumerate(tdec):
        v_int = int(v)
        if v_int != blank_id and v_int != prev:
            chars.append(i2c.get(v_int, '?'))
            positions.append(t)
        prev = v_int
    return chars, positions


# ═════════════════════════════════════════════════════════════════════════════
# Grad-CAM Computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_gradcam(net, img_tensor, device):
    """
    Grad-CAM (Selvaraju et al. 2017) adapted for ViT patch-token sequences.

    For each feature channel d, the gradient is globally averaged over all
    token positions to produce a single importance weight α_d.  The per-token
    CAM score is then:

        α_d    = (1/T) Σ_t  ∂y/∂A_{t,d}           — feature importance
        cam_t  = ReLU( Σ_d  α_d · A_{t,d} )        — per-token importance

    Key design choices:
      - Global average pooling of gradients over the spatial (token) axis
        follows the original paper (GAP before channel weighting).
      - ReLU keeps only positive-contribution patches.
      - Backward target excludes blank (CTC padding) time steps.
      - Backbone runs in eval mode (no dropout noise in features).
    """
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(False)

    # Backbone forward → activations A [T, 1, D]
    seq_tokens, reg_tokens, (Hp, Wp) = net.backbone(img_tensor)
    seq_tokens.retain_grad()

    # CTC head forward
    seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)    # [1, D, 1, T]
    logits = net.top(seq_4d)
    if isinstance(logits, tuple):
        logits = logits[0]                                # [T, 1, C]

    # Backward from NON-BLANK predicted classes only
    pred_classes = logits.argmax(dim=2)                   # [T, 1]
    non_blank = pred_classes.squeeze(1) != 0              # [T] bool
    if non_blank.any():
        target = logits[non_blank].gather(
            2, pred_classes[non_blank].unsqueeze(2)).sum()
    else:
        target = logits.gather(2, pred_classes.unsqueeze(2)).sum()

    net.zero_grad()
    target.backward()

    # Grad-CAM (Selvaraju et al. 2017): global-average-pool gradients
    # over the spatial (token) dimension to obtain one importance weight
    # α_d per feature channel, then weighted-sum with activations per token.
    #   α_d    = (1/T) Σ_t  ∂y/∂A_{t,d}           — feature importance
    #   cam_t  = ReLU( Σ_d  α_d · A_{t,d} )        — per-token importance
    grads = seq_tokens.grad.detach()                      # [T, 1, D]
    acts  = seq_tokens.detach()                           # [T, 1, D]
    alpha = grads.mean(dim=0, keepdim=True)               # [1, 1, D]
    cam = (alpha * acts).squeeze(1).sum(dim=-1)           # [T]
    cam = torch.relu(cam).cpu().numpy()                   # positive only

    # Normalise to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return {
        'importance': cam,
        'grid_size': (Hp, Wp),
        'logits': logits.detach().cpu().numpy(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Attention Extraction (layer-based, matching version_1 approach)
# ═════════════════════════════════════════════════════════════════════════════

def attention_rollout(attn_maps_list):
    """
    Attention Rollout (Abnar & Zuidema, 2020).

    Multiplies attention matrices across all layers, accounting for
    residual connections:  A' = 0.5·A + 0.5·I  (identity for skip).
    Rows are renormalised after each multiplication.

    Returns [S, S] rolled-out attention (numpy).
    """
    result = None
    for attn in attn_maps_list:
        # [B, H, S, S] → [B, S, S]  head-average
        attn_avg = attn.mean(dim=1)           # [B, S, S]
        S = attn_avg.size(-1)

        # Add identity for residual connection
        I = torch.eye(S, device=attn_avg.device).unsqueeze(0)
        attn_res = 0.5 * attn_avg + 0.5 * I

        # Row-normalise
        attn_res = attn_res / (attn_res.sum(dim=-1, keepdim=True) + 1e-8)

        if result is None:
            result = attn_res
        else:
            result = torch.bmm(attn_res, result)

    return result[0].cpu().numpy()            # [S, S]


def select_attention_layer(attn_maps_list, layer='last'):
    """
    Select and aggregate attention from the specified layer(s).

    Parameters
    ----------
    attn_maps_list : list of [B, H, S, S] tensors — one per layer
    layer          : 'first' | 'middle' | 'last' | 'all' | 'rollout'

    Returns [S, S] head-averaged attention matrix (numpy).
    """
    L = len(attn_maps_list)
    if layer == 'rollout':
        return attention_rollout(attn_maps_list)
    elif layer == 'first':
        attn = attn_maps_list[0]
    elif layer == 'middle':
        attn = attn_maps_list[L // 2]
    elif layer == 'last':
        attn = attn_maps_list[-1]
    elif layer == 'all':
        attn = torch.stack(attn_maps_list, dim=0).mean(dim=0)
    else:
        attn = attn_maps_list[-1]

    # Average across heads → [S, S]
    if attn.dim() == 4:
        attn = attn[0].mean(dim=0)            # [S, S]
    elif attn.dim() == 3:
        attn = attn[0]                         # [S, S]
    return attn.cpu().numpy()


def extract_attention(net, img_tensor, device, layers=('last',)):
    """
    Run forward_explain() once and extract attention from specified layer(s).

    Parameters
    ----------
    layers : tuple of str
        One or more of 'first', 'middle', 'last', 'all'.

    Returns dict with attn_<layer> [S,S] for each layer, plus
    grid_size, logits, num_reg.
    """
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits, reg_tokens, attn_maps_list, token_norms, (Hp, Wp) = \
            net.forward_explain(img_tensor)

    if isinstance(logits, tuple):
        logits = logits[0]
    logits_np = logits.cpu().numpy()

    num_reg = reg_tokens.shape[1] if reg_tokens is not None else 0

    result = {
        'grid_size': (Hp, Wp),
        'logits': logits_np,
        'num_reg': num_reg,
    }
    for layer in layers:
        result[f'attn_{layer}'] = select_attention_layer(attn_maps_list, layer)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Patch-aligned Upscaling
# ═════════════════════════════════════════════════════════════════════════════

def upscale_patch_aligned(patch_map, H_img, W_img, fixed_size=None,
                          border_size=8):
    """
    Map patch-level heatmap to original image coordinates.

    The model operates on a *preprocessed* image of ``fixed_size`` (e.g.
    128×1024), not the raw image (e.g. 42×162).  Within the preprocessed
    canvas the original image sits at an offset of ``border_size`` pixels
    (top, left).  Each patch covers (H_preproc // Hp, W_preproc // Wp)
    pixels in the preprocessed image.

    Steps:
      1. Upscale to preprocessed resolution with nearest-neighbour (kron)
      2. Crop the region that corresponds to the original image

    When ``fixed_size`` is ``None`` the legacy (incorrect) 1:1 mapping is
    used as a fallback to keep backward compatibility.
    """
    Hp, Wp = patch_map.shape

    if fixed_size is None:
        # Legacy fallback — 1:1 mapping (kept for backward compat)
        H_patch = max(1, H_img // Hp)
        W_patch = max(1, W_img // Wp)
        heat_up = np.kron(patch_map, np.ones((H_patch, W_patch)))
        return heat_up[:H_img, :W_img]

    H_preproc, W_preproc = fixed_size

    # Pixels per patch in the preprocessed coordinate frame
    ph = H_preproc // Hp          # e.g. 128 for Hp=1
    pw = W_preproc // Wp          # e.g. 8   for Wp=128

    # Upscale to full preprocessed resolution
    heat_full = np.kron(patch_map, np.ones((ph, pw)))   # (H_preproc, W_preproc)

    # Crop to the region occupied by the original image
    r0 = border_size
    c0 = border_size
    heat_crop = heat_full[r0 : r0 + H_img, c0 : c0 + W_img]

    # Safety: if original image is larger than the crop window (shouldn't
    # happen with correct configs), pad with zeros.
    if heat_crop.shape[0] < H_img or heat_crop.shape[1] < W_img:
        padded = np.zeros((H_img, W_img), dtype=heat_crop.dtype)
        padded[:heat_crop.shape[0], :heat_crop.shape[1]] = heat_crop
        heat_crop = padded

    return heat_crop


# ═════════════════════════════════════════════════════════════════════════════
# Per-character Heatmap Builder
# ═════════════════════════════════════════════════════════════════════════════

def build_char_heatmaps(attn_matrix, grid_size, num_reg, chars, positions,
                        H_img, W_img, max_chars, gamma=3.0,
                        fixed_size=None, border_size=8,
                        percentile_clip=99.0):
    """
    Build per-character attention heatmaps from the attention matrix.

    For each character c at CTC time step t:
      row (num_reg + t) of attn → [T] patch importance
      Percentile-clip normalisation: clip at ``percentile_clip``-th percentile
        then scale to [0, 1].  This gives relative scaling that works for both
        short words (sharp, concentrated attention) and long sentences
        (diffuse attention spread over many patches).
      Adaptive gamma: increased for longer texts so character boundaries
        remain crisp despite diffuse raw attention.
      Upscale to (H_img, W_img) with correct spatial mapping.

    Returns (heatmaps: list[ndarray], n_show: int).
    """
    Hp, Wp = grid_size
    T = Hp * Wp
    n_show = min(len(chars), max_chars)

    # Adaptive gamma: longer text → more aggressive contrast to pull out
    # each character from the diffuse background.
    n_chars = max(1, len(chars))
    gamma_adaptive = float(np.clip(gamma + 0.05 * (n_chars - 6), gamma, gamma + 3.0))

    heatmaps = []
    for ci in range(n_show):
        t = positions[ci]
        row_idx = num_reg + t
        if row_idx >= attn_matrix.shape[0]:
            patch_attn = np.ones(T) / T
        else:
            patch_attn = attn_matrix[row_idx, num_reg:]
            if patch_attn.shape[0] > T:
                patch_attn = patch_attn[:T]

        # Percentile-based relative normalisation to [0, 1].
        # Clips extreme outlier patches so that the full colour range is
        # used for the relevant attention region — critical for long text
        # where a single outlier patch would otherwise compress all other
        # values towards zero.
        p_low  = max(0.0, np.percentile(patch_attn, 100.0 - percentile_clip))
        p_high = np.percentile(patch_attn, percentile_clip)
        if p_high - p_low < 1e-8:
            p_low, p_high = patch_attn.min(), patch_attn.max()
        patch_attn = np.clip(patch_attn, p_low, p_high)
        patch_attn = (patch_attn - p_low) / (p_high - p_low + 1e-8)

        # Power contrast: amplify peaks, suppress diffuse background.
        patch_attn = patch_attn ** gamma_adaptive

        grid = patch_attn.reshape(Hp, Wp)
        heat_up = upscale_patch_aligned(grid, H_img, W_img, fixed_size,
                                        border_size)
        heatmaps.append(heat_up)

    return heatmaps, n_show


# ═════════════════════════════════════════════════════════════════════════════
# Process One Model on One Image
# ═════════════════════════════════════════════════════════════════════════════

def process_single_image(net, device, img_tensor, mode, layers=('last',)):
    """
    Compute Grad-CAM and/or attention for one image tensor.

    Sets the correct train/eval mode for each operation:
      - Attention: net.eval()  → no dropout noise in attention weights
      - Grad-CAM:  backbone.eval() + top.train()  → clean features,
                   but GRU backward requires train mode

    Parameters
    ----------
    layers : tuple of str
        Attention layers to extract (e.g. ('last',) or ('last', 'rollout')).

    Returns dict with gc_* and/or attn_<layer> keys + logits.
    """
    result = {}

    gc_logits = None
    if mode in ('gradcam', 'both'):
        # Full eval mode: no dropout noise, deterministic results.
        # GRU backward works fine in eval mode on CPU and modern PyTorch.
        net.eval()
        gc_result = compute_gradcam(net, img_tensor, device)
        result['gc_importance'] = gc_result['importance']
        result['gc_grid'] = gc_result['grid_size']
        gc_logits = gc_result['logits']

    attn_logits = None
    if mode in ('attn', 'both'):
        # Attention: full eval (no dropout noise in attention weights)
        net.eval()
        attn_result = extract_attention(net, img_tensor, device, layers=layers)
        for layer in layers:
            result[f'attn_{layer}'] = attn_result[f'attn_{layer}']
        result['attn_grid'] = attn_result['grid_size']
        result['num_reg'] = attn_result['num_reg']
        attn_logits = attn_result['logits']

    result['logits'] = gc_logits if gc_logits is not None else attn_logits
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Visualization — Attention comparison figure
# ═════════════════════════════════════════════════════════════════════════════

def visualize_attention_comparison(img_path, model_infos, all_results,
                                   save_path, alpha, dpi, max_chars, gamma,
                                   fixed_size=None, border_size=8,
                                   layer_name='last'):
    """
    Generate attention comparison grid (one figure).

    Rows = models (sorted by register count)
    Cols = [Original | char₁ | char₂ | … | charₙ]

    Each row labelled with Reg-N, CER, predicted text.
    """
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape
    image_name = Path(img_path).stem

    n_models = len(all_results)

    # Max characters across all models
    max_n_chars = max(
        (min(len(r['chars']), max_chars) for r in all_results),
        default=0
    )
    max_n_chars = max(max_n_chars, 1)

    n_cols = 1 + max_n_chars   # Original + characters

    col_w = max(2.5, min(W_img / 80, 5.0))
    row_h = max(1.8, min(H_img / 30, 4.0))
    fig_w = min(col_w * n_cols + 2.5, 80)
    fig_h = row_h * n_models + 1.5

    fig, axes = plt.subplots(n_models, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    for row_idx, (info, result) in enumerate(zip(model_infos, all_results)):
        # Column 0: Original image + label
        ax = axes[row_idx, 0]
        ax.imshow(img_np, cmap="gray")
        if row_idx == 0:
            ax.set_title("Original", fontsize=9, fontweight='bold')
        ax.axis("off")

        pred_disp = result['pred_text']
        if len(pred_disp) > 30:
            pred_disp = pred_disp[:27] + '…'
        label_lines = [info['label']]
        if result.get('cer') is not None:
            label_lines.append(f"CER={result['cer']:.4f}")
        label_lines.append(f'"{pred_disp}"')
        ax.text(0.02, 0.98, '\n'.join(label_lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=7, color='white', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          alpha=0.75, edgecolor='none'))

        # Character attention columns
        attn_matrix = result.get(f'attn_{layer_name}')
        if attn_matrix is None:
            for ci in range(max_n_chars):
                axes[row_idx, 1 + ci].set_visible(False)
            continue

        grid_size = result['attn_grid']
        num_reg = result.get('num_reg', 0)
        chars = result['chars']
        positions = result['positions']

        heatmaps, n_show = build_char_heatmaps(
            attn_matrix, grid_size, num_reg, chars, positions,
            H_img, W_img, max_chars, gamma,
            fixed_size=fixed_size, border_size=border_size,
        )

        for ci in range(max_n_chars):
            ax = axes[row_idx, 1 + ci]
            if ci < n_show:
                ax.imshow(img_np, cmap="gray")
                # inferno: perceptually-uniform, high-contrast, print-safe
                im = ax.imshow(heatmaps[ci], cmap="inferno", alpha=alpha,
                               vmin=0.0, vmax=1.0)
                ch = chars[ci]
                ch_disp = repr(ch) if ch in (' ', '\t', '\n') else f'"{ch}"'
                if row_idx == 0:
                    ax.set_title(ch_disp, fontsize=8, fontweight='bold')
                # Thin colorbar on the right of each attention panel
                if ci == n_show - 1:
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                        orientation='vertical')
                    cbar.ax.tick_params(labelsize=5)
                    cbar.set_label('rel. attn.', fontsize=5)
            else:
                ax.set_visible(False)
            ax.axis("off")

    # Suptitle
    gt_text = all_results[0].get('gt_text', '')
    title_parts = [f'Attention Comparison: {image_name}']
    if gt_text:
        gt_disp = gt_text if len(gt_text) <= 60 else gt_text[:57] + '…'
        title_parts.append(f'GT: "{gt_disp}"')
    # Adaptive gamma used: recompute for display
    n_chars_example = max(1, max((len(r['chars']) for r in all_results), default=1))
    gamma_display = float(np.clip(gamma + 0.05 * (n_chars_example - 6), gamma, gamma + 3.0))
    title_parts.append(
        f'Method: {layer_name}-layer attention (γ≈{gamma_display:.1f}, p99-clip)  |  Models: {n_models}')
    fig.suptitle('\n'.join(title_parts), fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Visualization — Grad-CAM comparison figure
# ═════════════════════════════════════════════════════════════════════════════

def visualize_gradcam_comparison(img_path, model_infos, all_results,
                                  save_path, alpha, dpi,
                                  fixed_size=None, border_size=8):
    """
    Generate Grad-CAM comparison grid (one figure).

    Rows = models (sorted by register count)
    Cols = [Original | Grad-CAM]

    Each row labelled with Reg-N, CER, predicted text.
    """
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape
    image_name = Path(img_path).stem

    n_models = len(all_results)
    n_cols = 2   # Original + Grad-CAM

    fig_w = max(8, min(12, W_img / 40))
    row_h = max(1.8, min(H_img / 30, 4.0))
    fig_h = row_h * n_models + 1.5

    fig, axes = plt.subplots(n_models, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    for row_idx, (info, result) in enumerate(zip(model_infos, all_results)):
        # Column 0: Original + label
        ax = axes[row_idx, 0]
        ax.imshow(img_np, cmap="gray")
        if row_idx == 0:
            ax.set_title("Original", fontsize=9, fontweight='bold')
        ax.axis("off")

        pred_disp = result['pred_text']
        if len(pred_disp) > 30:
            pred_disp = pred_disp[:27] + '…'
        label_lines = [info['label']]
        if result.get('cer') is not None:
            label_lines.append(f"CER={result['cer']:.4f}")
        label_lines.append(f'"{pred_disp}"')
        ax.text(0.02, 0.98, '\n'.join(label_lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=7, color='white', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          alpha=0.75, edgecolor='none'))

        # Column 1: Grad-CAM
        ax = axes[row_idx, 1]
        importance = result.get('gc_importance')
        grid_size = result.get('gc_grid')
        if importance is not None and grid_size is not None:
            Hp, Wp = grid_size
            T = importance.shape[0]
            if T == Hp * Wp:
                heat_grid = importance.reshape(Hp, Wp)
                heat_up = upscale_patch_aligned(heat_grid, H_img, W_img,
                                                fixed_size, border_size)
                # Percentile-based relative scaling: ensures both short
                # (concentrated importance) and long (diffuse importance)
                # texts fill the full colour range.
                p99 = np.percentile(heat_up, 99)
                p01 = np.percentile(heat_up,  1)
                if p99 - p01 < 1e-8:
                    p01, p99 = heat_up.min(), heat_up.max()
                heat_scaled = np.clip((heat_up - p01) / (p99 - p01 + 1e-8),
                                      0.0, 1.0)
                ax.imshow(img_np, cmap="gray")
                im = ax.imshow(heat_scaled, cmap="inferno", alpha=alpha,
                               vmin=0.0, vmax=1.0)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                    orientation='vertical')
                cbar.ax.tick_params(labelsize=5)
                cbar.set_label('rel. imp.', fontsize=5)
            else:
                ax.imshow(img_np, cmap="gray")
                ax.text(0.5, 0.5, 'T≠Hp·Wp', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='red')
        else:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10)
        if row_idx == 0:
            ax.set_title("Grad-CAM", fontsize=9, fontweight='bold')
        ax.axis("off")

    # Suptitle
    gt_text = all_results[0].get('gt_text', '')
    title_parts = [f'Grad-CAM Comparison: {image_name}']
    if gt_text:
        gt_disp = gt_text if len(gt_text) <= 60 else gt_text[:57] + '…'
        title_parts.append(f'GT: "{gt_disp}"')
    title_parts.append(f'Method: Grad-CAM (Selvaraju 2017) ReLU(grad·act)  |  Models: {n_models}')
    fig.suptitle('\n'.join(title_parts), fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    (config, model_paths, image_paths, save_dir, mode,
     alpha, dpi, max_chars, gamma, layers) = parse_args()

    # ── Device ───────────────────────────────────────────────────────
    device = getattr(config, 'device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA not available, falling back to CPU.')
        device = 'cpu'

    # ── Character classes ────────────────────────────────────────────
    dataset_folder = config.data.path
    classes = np.load(os.path.join(dataset_folder, 'classes.npy'))
    num_classes = len(classes) + 1
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    fixed_size = (config.preproc.image_height, config.preproc.image_width)
    wer_mode = config.eval.wer_mode

    # ── Output directory ─────────────────────────────────────────────
    out_dir = (save_dir if save_dir
               else os.path.join('visualizations', 'cross_model_comparison'))
    attn_dir = os.path.join(out_dir, 'attention')
    gc_dir = os.path.join(out_dir, 'gradcam')
    rollout_dir = os.path.join(out_dir, 'rollout')

    # Clean previous results for a clear output
    import shutil
    for sub in (attn_dir, gc_dir, rollout_dir):
        if os.path.isdir(sub):
            shutil.rmtree(sub)
    os.makedirs(attn_dir, exist_ok=True)
    os.makedirs(gc_dir, exist_ok=True)
    if 'rollout' in layers:
        os.makedirs(rollout_dir, exist_ok=True)

    # ── Run info for each model ──────────────────────────────────────
    model_infos = []
    for mp in model_paths:
        model_infos.append(get_run_info(mp))

    # Sort by register count
    sorted_idx = sorted(range(len(model_infos)),
                        key=lambda i: model_infos[i]['num_registers'])
    model_infos = [model_infos[i] for i in sorted_idx]
    model_paths = [model_paths[i] for i in sorted_idx]

    # ── Determine execution mode ─────────────────────────────────────
    is_test_set = len(image_paths) == 0
    if is_test_set:
        test_set = HTRDataset(dataset_folder, "test", fixed_size=fixed_size,
                              transforms=None)
        image_paths_all = [test_set.data[i][0] for i in range(len(test_set))]
        gt_dict = {Path(test_set.data[i][0]).stem: test_set.data[i][1].strip()
                   for i in range(len(test_set))}
        loader = DataLoader(test_set, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False)
        n_samples = len(test_set)
    else:
        image_paths_all = image_paths
        explicit_gt = getattr(config, 'gt', None)
        gt_dict = find_gt_for_images(image_paths, explicit_gt_path=explicit_gt)
        loader = None
        n_samples = len(image_paths)

    # ── Header ───────────────────────────────────────────────────────
    print('=' * 70)
    print('Cross-Model Attention & Grad-CAM Comparison')
    print('=' * 70)
    if is_test_set:
        print(f'Mode         : test_set ({n_samples} samples)')
    else:
        print(f'Mode         : image ({n_samples} file(s))')
    print(f'Vis. mode    : {mode}')
    print(f'Models       : {len(model_paths)}')
    for info in model_infos:
        print(f'  {info["label"]:>8s}  ({info["run_name"]})')
    print(f'Output dir   : {out_dir}')
    if mode in ('attn', 'both'):
        print(f'  └─ attention/ : {attn_dir}')
        if 'rollout' in layers:
            print(f'  └─ rollout/   : {rollout_dir}')
    if mode in ('gradcam', 'both'):
        print(f'  └─ gradcam/   : {gc_dir}')
    print(f'Alpha        : {alpha}')
    print(f'DPI          : {dpi}')
    print(f'Max chars    : {max_chars}')
    print(f'Gamma        : {gamma}')
    print(f'Attn layer   : {", ".join(layers)}')
    print(f'Timestamp    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    # ══════════════════════════════════════════════════════════════════
    # Process each model sequentially (memory-efficient)
    # ══════════════════════════════════════════════════════════════════
    all_model_data = []   # [model_idx][image_idx] = result dict

    for mi, (info, mp) in enumerate(zip(model_infos, model_paths)):
        print(f'[{mi + 1}/{len(model_paths)}] Loading {info["label"]} '
              f'({info["run_name"]}) …')

        net, n_params, load_info = build_model_for_run(
            config, mp, num_classes, info['num_registers'], device,
        )
        print(f'  Parameters : {n_params:,}  |  {load_info}')

        results_this_model = []

        if is_test_set:
            for sample_idx, (imgs, transcrs) in enumerate(
                    tqdm.tqdm(loader, desc=f'  {info["label"]}')):
                gt_text = transcrs[0].strip()
                img_path = test_set.data[sample_idx][0]
                image_name = Path(img_path).stem

                r = process_single_image(net, device, imgs, mode, layers)

                chars, positions = ctc_char_positions(r['logits'], i2c)
                pred_text = ''.join(chars).strip()

                cer_val = wer_val = None
                if gt_text:
                    cer_s = CER()
                    wer_s = WER(mode=wer_mode)
                    cer_s.update(pred_text, gt_text)
                    wer_s.update(pred_text, gt_text)
                    cer_val = cer_s.score()
                    wer_val = wer_s.score()

                r.update({
                    'image_name': image_name,
                    'img_path': img_path,
                    'chars': chars,
                    'positions': positions,
                    'pred_text': pred_text,
                    'gt_text': gt_text,
                    'cer': cer_val,
                    'wer': wer_val,
                })
                results_this_model.append(r)

                if sample_idx % 50 == 0:
                    gc.collect()
        else:
            for img_path in tqdm.tqdm(image_paths_all,
                                      desc=f'  {info["label"]}'):
                image_name = Path(img_path).stem

                raw_img = load_image(img_path)
                processed = preprocess(raw_img, fixed_size)
                img_tensor = torch.from_numpy(processed).float() \
                    .unsqueeze(0).unsqueeze(0)

                r = process_single_image(net, device, img_tensor, mode, layers)

                chars, positions = ctc_char_positions(r['logits'], i2c)
                pred_text = ''.join(chars).strip()

                gt_text = gt_dict.get(image_name, '')
                cer_val = wer_val = None
                if gt_text:
                    cer_s = CER()
                    wer_s = WER(mode=wer_mode)
                    cer_s.update(pred_text, gt_text)
                    wer_s.update(pred_text, gt_text)
                    cer_val = cer_s.score()
                    wer_val = wer_s.score()

                r.update({
                    'image_name': image_name,
                    'img_path': img_path,
                    'chars': chars,
                    'positions': positions,
                    'pred_text': pred_text,
                    'gt_text': gt_text,
                    'cer': cer_val,
                    'wer': wer_val,
                })
                results_this_model.append(r)

        all_model_data.append(results_this_model)

        # Report per-model CER
        cer_vals = [r['cer'] for r in results_this_model
                    if r['cer'] is not None]
        if cer_vals:
            avg_cer = sum(cer_vals) / len(cer_vals)
            print(f'  Avg CER    : {avg_cer:.4f}')

        # Sample predictions
        for r in results_this_model[:5]:
            cer_str = f'  CER={r["cer"]:.4f}' if r['cer'] is not None else ''
            tqdm.tqdm.write(
                f'    {r["image_name"]:>30s}  →  "{r["pred_text"]}"{cer_str}')
        if len(results_this_model) > 5:
            tqdm.tqdm.write(
                f'    … and {len(results_this_model) - 5} more')

        del net
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        print()

    # ══════════════════════════════════════════════════════════════════
    # Generate comparison figures (SEPARATE attention & grad-cam)
    # ══════════════════════════════════════════════════════════════════
    print(f'Generating comparison figures …')

    saved_attn = 0
    saved_gc = 0

    for img_idx in tqdm.tqdm(range(n_samples), desc='Figures'):
        img_results = [all_model_data[mi][img_idx]
                       for mi in range(len(model_infos))]
        img_path = img_results[0]['img_path']
        image_name = img_results[0]['image_name']

        # ── Attention figure(s) → attention/ or rollout/ subfolder ────
        if mode in ('attn', 'both'):
            for layer_name in layers:
                suffix = f'_{layer_name}' if len(layers) > 1 else ''
                # Route rollout figures to dedicated rollout/ folder
                if layer_name == 'rollout':
                    save_path = os.path.join(rollout_dir,
                                             f'{image_name}{suffix}.png')
                else:
                    save_path = os.path.join(attn_dir,
                                             f'{image_name}{suffix}.png')
                out = visualize_attention_comparison(
                    img_path, model_infos, img_results,
                    save_path, alpha, dpi, max_chars, gamma,
                    fixed_size=fixed_size,
                    layer_name=layer_name,
                )
                if out:
                    saved_attn += 1

        # ── Grad-CAM figure → gradcam/ subfolder ─────────────────────
        if mode in ('gradcam', 'both'):
            save_path = os.path.join(gc_dir, f'{image_name}.png')
            out = visualize_gradcam_comparison(
                img_path, model_infos, img_results,
                save_path, alpha, dpi,
                fixed_size=fixed_size,
            )
            if out:
                saved_gc += 1

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print()
    print(f'Comparison complete!')
    if mode in ('attn', 'both'):
        n_expected = n_samples * len(layers)
        print(f'  Attention figures : {saved_attn}/{n_expected}  → {attn_dir}/')
        if 'rollout' in layers:
            print(f'  Rollout figures   : → {rollout_dir}/')
    if mode in ('gradcam', 'both'):
        print(f'  Grad-CAM figures  : {saved_gc}/{n_samples}  → {gc_dir}/')
    print(f'  Output root       : {out_dir}')

    # Per-model CER summary
    has_gt = any(r['cer'] is not None
                 for model_data in all_model_data for r in model_data)
    if has_gt:
        print()
        print(f'  {"Model":>10s}  {"CER":>8s}  {"WER":>8s}')
        print(f'  {"─" * 10}  {"─" * 8}  {"─" * 8}')
        for mi, info in enumerate(model_infos):
            cer_vals = [r['cer'] for r in all_model_data[mi]
                        if r['cer'] is not None]
            wer_vals = [r['wer'] for r in all_model_data[mi]
                        if r['wer'] is not None]
            if cer_vals:
                avg_cer = sum(cer_vals) / len(cer_vals)
                avg_wer = sum(wer_vals) / len(wer_vals)
                print(f'  {info["label"]:>10s}  {avg_cer:>8.4f}  {avg_wer:>8.4f}')

    print('=' * 70)


if __name__ == '__main__':
    main()
