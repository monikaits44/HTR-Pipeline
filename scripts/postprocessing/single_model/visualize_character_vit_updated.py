#!/usr/bin/env python3
"""
Character-level Attention Visualization for ViT-RGTS

Visualises which image patches the model attends to when predicting each
character.  Uses REAL attention weights from the transformer's
forward_explain() path — not gradient-based surrogates.

How it works:
  1. Runs forward_explain() → gets per-layer attention maps [B, H, S, S]
  2. Selects a layer (first / middle / last / all-averaged)
  3. Averages across heads → [S, S]   (S = R + T where R = registers, T = patches)
  4. For each predicted character *c* at CTC time step *t*:
     - Looks up which patch token *t* the CTC decoder used
     - Extracts row t from the attention matrix → [T] importance per patch
     - Reshapes to (Hp, Wp) and upscales with patch-aligned nearest-neighbour
  5. Overlays the heatmap on the original image (one panel per character)

Multi-panel output per sample:
  Panel 0:      Original grayscale image
  Panel 1..N:   Per-character attention heatmap overlaid on the image

Supports four execution modes:
  1. Test set  (default) — every sample in data/IAM/test
  2. Single image        — one image after the -- separator
  3. Multiple images     — several images after --
  4. Directory           — a folder of images after --

Supported Architectures: vit_rgts only (requires forward_explain)
Input: config YAML(s) + model checkpoint + optional image paths
Output: Per-sample character attention PNGs (auto-saved to visualizations/character_attention/)

Usage:
    # ── Test set mode (default — no image paths after --) ───────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt

    # ── Single image ───────────────────────────────────────────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- notebook/sample_images/a01-038-12.png

    # ── Multiple images ────────────────────────────────────────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- img1.png img2.png img3.png

    # ── Directory of images ────────────────────────────────────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- notebook/sample_images/

    # ── Choose attention layer ─────────────────────────────────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        --layer last \\
        -- notebook/sample_images/a01-038-12.png

    # ── Save to custom directory ───────────────────────────────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        --save-dir output/char_attention/

    # ── Adjust overlay / resolution / max characters ───────────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        --alpha 0.5 --dpi 200 --max-chars 20

    # ── GPU acceleration ──────────────────────────────────────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt device=cuda

    # ── With ground truth file ─────────────────────────────────────
    python scripts/postprocessing/visualize_character_vit_updated.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        gt=notebook/sample_images/gt.txt \\
        -- notebook/sample_images/

Arguments:
    config.yaml       Base config file (required)
    extra.yaml        Additional config files merged in order
    key=value         Override config params (e.g., resume=..., device=cuda)
    --                Separator; image paths follow
    --save-dir PATH   Directory for PNGs (default: visualizations/character_attention/)
    --layer LAYER     Attention layer: first, middle, last, all (default: last)
    --alpha FLOAT     Heatmap overlay opacity 0.0–1.0 (default: 0.45)
    --dpi INT         Output resolution (default: 150)
    --max-chars INT   Maximum characters to show per image (default: 30)
"""
import os
import sys
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
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

# Image extensions accepted in directory mode
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    """
    Parse config YAML(s), key=value overrides, visualization flags,
    and optional image paths.

    Before '--': config YAMLs + key=value overrides + --save-dir/--layer/etc.
    After  '--': image paths (single, multiple, or directory)
    If no '--' separator, runs in test-set mode.

    Returns: (config, image_paths, save_dir, layer, alpha, dpi, max_chars)
    """
    argv = sys.argv[1:]

    has_separator = '--' in argv
    if has_separator:
        sep = argv.index('--')
        config_args = argv[:sep]
        image_args = argv[sep + 1:]
    else:
        config_args = argv
        image_args = []

    # Extract visualization flags from config_args
    save_dir = ''
    layer = 'last'
    alpha = 0.45
    dpi = 150
    max_chars = 30

    filtered = []
    i = 0
    while i < len(config_args):
        if config_args[i] == '--save-dir' and i + 1 < len(config_args):
            save_dir = config_args[i + 1]
            i += 2
        elif config_args[i] == '--layer' and i + 1 < len(config_args):
            layer = config_args[i + 1]
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
        else:
            filtered.append(config_args[i])
            i += 1

    yaml_files = [a for a in filtered if a.endswith('.yaml')]
    overrides = [a for a in filtered if not a.endswith('.yaml')]

    if not yaml_files:
        print('Error: at least one .yaml config file is required.')
        sys.exit(1)

    # Merge YAMLs in order
    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))

    # Apply CLI key=value overrides safely via OmegaConf.from_dotlist
    OmegaConf.set_struct(conf, False)
    cli_conf = OmegaConf.from_dotlist(overrides)
    conf = OmegaConf.merge(conf, cli_conf)

    # Validate layer
    if layer not in ('first', 'middle', 'last', 'all'):
        print(f'Error: --layer must be first, middle, last, or all (got: {layer})')
        sys.exit(1)

    # Resolve image paths (files and directories)
    image_paths = collect_image_paths(image_args)

    if has_separator and not image_paths:
        print('Error: "--" separator was used but no valid images found.')
        print(f'  Provided paths: {image_args}')
        sys.exit(1)

    return conf, image_paths, save_dir, layer, alpha, dpi, max_chars


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


def load_gt_file(path):
    """Load a ground truth file into a dict {image_stem: gt_text}."""
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
    """Locate and load ground truth for image paths (explicit > auto-detect)."""
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
        print('Ground truth : none (gt_text, cer, wer will be empty)')
    return gt


# ── Model loading ────────────────────────────────────────────────────────────

def build_model(config, num_classes):
    """Instantiate HTRNet and load checkpoint weights (eval mode)."""
    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        device = "cpu"
        config.device = "cpu"

    print(f'Architecture : {config.arch.type}')

    net = HTRNet(config.arch, num_classes)

    if config.resume is None:
        raise ValueError(
            "config.resume is None — provide a trained checkpoint "
            "(e.g. resume=saved_models/experiments/run_54/model.pt)"
        )

    print(f'Checkpoint   : {config.resume}')
    checkpoint = torch.load(config.resume, map_location=device)
    print(net.load_state_dict(checkpoint, strict=True))

    net.to(device)
    net.eval()  # forward_explain() uses @torch.no_grad — eval mode is fine

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Parameters   : {n_params:,}')

    arch_type = getattr(config.arch, "type", "cnn_rnn")
    if arch_type != "vit_rgts":
        raise ValueError(
            f"Expected arch.type == 'vit_rgts', got '{arch_type}'. "
            "This script requires forward_explain() from the ViT-RGTS backbone."
        )

    return net, device


# ── CTC decode ───────────────────────────────────────────────────────────────

def ctc_decode(tdec, i2c, blank_id=0):
    """Greedy CTC decode: collapse repeats, strip blank (index 0)."""
    tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
    return "".join([i2c[t] for t in tt if t != blank_id])


def ctc_char_positions(logits_np, i2c, blank_id=0):
    """
    Map each predicted character to its dominant CTC time step.

    CTC decoding collapses consecutive identical labels and removes blanks.
    This function records which time step *t* produced each character so
    we can look up that row in the attention matrix.

    Parameters
    ----------
    logits_np : ndarray [T, 1, C]   CTC logits
    i2c       : dict                 {class_idx: char}
    blank_id  : int                  CTC blank index (default 0)

    Returns
    -------
    chars     : list of str          predicted characters
    positions : list of int          time-step index per character
    """
    tdec = logits_np.argmax(2).squeeze()       # [T]
    chars = []
    positions = []
    prev = -1
    for t, v in enumerate(tdec):
        v_int = int(v)
        if v_int != blank_id and v_int != prev:
            chars.append(i2c.get(v_int, '?'))
            positions.append(t)
        prev = v_int
    return chars, positions


# ── Attention extraction ─────────────────────────────────────────────────────

def extract_character_attention(net, img_tensor, device, layer='last'):
    """
    Run forward_explain() and extract per-character attention.

    Returns
    -------
    dict with:
        chars      : list of str — predicted characters
        positions  : list of int — CTC time step per character
        attn_maps  : list of [T] ndarray — per-character patch importance
        grid_size  : (Hp, Wp)
        logits     : [T, 1, C] ndarray
        num_reg    : int — number of register tokens
    """
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits, reg_tokens, attn_maps_list, token_norms, (Hp, Wp) = \
            net.forward_explain(img_tensor)

    # logits: [T, B, C] → [T, 1, C] numpy
    logits_np = logits.cpu().numpy()

    # ── Select attention layer ───────────────────────────────────────
    # attn_maps_list: list of L tensors, each [B, H, S, S]
    # S = R + T  (register + patch tokens)
    L = len(attn_maps_list)
    if layer == 'first':
        attn = attn_maps_list[0]
    elif layer == 'middle':
        attn = attn_maps_list[L // 2]
    elif layer == 'last':
        attn = attn_maps_list[-1]
    elif layer == 'all':
        attn = torch.stack(attn_maps_list, dim=0).mean(dim=0)
    else:
        attn = attn_maps_list[-1]

    # attn: [B, H, S, S] → average across heads → [S, S]
    if attn.dim() == 4:
        attn = attn[0].mean(dim=0)              # [S, S]
    elif attn.dim() == 3:
        attn = attn[0]                            # [S, S]
    attn = attn.numpy()

    # Number of registers & patches
    num_reg = reg_tokens.shape[1] if reg_tokens is not None else 0
    T = Hp * Wp

    # ── Map characters to time steps via CTC ─────────────────────────
    # Build i2c from config classes (use the parent caller's i2c)
    # We'll return logits and let the caller decode
    # But we also need i2c here, so accept it from caller — OR we can
    # compute positions from raw logits and return them.
    # Decision: return all raw data, let the caller pass i2c.

    return {
        'attn': attn,           # [S, S] — full attention matrix (selected layer)
        'grid_size': (Hp, Wp),
        'logits': logits_np,    # [T, 1, C]
        'num_reg': num_reg,
    }


# ── Patch-aligned upscaling ──────────────────────────────────────────────────

def upscale_patch_aligned(patch_map, H_img, W_img, fixed_size=None,
                          border_size=8):
    """
    Map patch-level heatmap to original image coordinates.

    The model operates on a *preprocessed* image of ``fixed_size``.
    Within the preprocessed canvas the original image sits at an offset
    of ``border_size`` pixels (top, left).  Each patch covers
    (H_preproc // Hp, W_preproc // Wp) preprocessed pixels.

    Steps:
      1. Upscale to preprocessed resolution with nearest-neighbour (kron)
      2. Crop the region that corresponds to the original image
    """
    Hp, Wp = patch_map.shape

    if fixed_size is None:
        # Legacy fallback
        H_patch = max(1, H_img // Hp)
        W_patch = max(1, W_img // Wp)
        heat_up = np.kron(patch_map, np.ones((H_patch, W_patch)))
        return heat_up[:H_img, :W_img]

    H_preproc, W_preproc = fixed_size
    ph = H_preproc // Hp
    pw = W_preproc // Wp
    heat_full = np.kron(patch_map, np.ones((ph, pw)))
    r0, c0 = border_size, border_size
    heat_crop = heat_full[r0 : r0 + H_img, c0 : c0 + W_img]
    if heat_crop.shape[0] < H_img or heat_crop.shape[1] < W_img:
        padded = np.zeros((H_img, W_img), dtype=heat_crop.dtype)
        padded[:heat_crop.shape[0], :heat_crop.shape[1]] = heat_crop
        heat_crop = padded
    return heat_crop


# ── Visualisation ────────────────────────────────────────────────────────────

def visualize_character_attention(img_path, attn, grid_size, num_reg,
                                  chars, positions, pred_text,
                                  gt_text='', cer=None, wer=None,
                                  save_path=None, alpha=0.45, dpi=150,
                                  max_chars=30, fixed_size=None,
                                  border_size=8, gamma=3.0):
    """
    Create a multi-panel figure with one heatmap per predicted character.

    Panel 0:    Original image
    Panel 1..N: Per-character attention heatmap overlay

    Parameters
    ----------
    img_path   : str          path to original image
    attn       : [S, S]       attention matrix (selected layer, head-averaged)
    grid_size  : (Hp, Wp)
    num_reg    : int          number of register tokens
    chars      : list[str]    predicted characters
    positions  : list[int]    CTC time step per character
    pred_text  : str
    gt_text    : str
    cer, wer   : float | None
    save_path  : str
    alpha      : float
    dpi        : int
    max_chars  : int

    Returns
    -------
    save_path or None
    """
    Hp, Wp = grid_size
    T = Hp * Wp

    if not chars:
        print(f'  [SKIP] No predicted characters for {Path(img_path).stem}')
        return None

    # Load original image
    if not os.path.isfile(img_path):
        print(f'  [SKIP] Image not found: {img_path}')
        return None
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape

    # Limit displayed characters
    n_show = min(len(chars), max_chars)
    image_name = Path(img_path).stem

    # Adaptive gamma: longer text gets more contrast to keep characters crisp
    n_chars = max(1, len(chars))
    gamma_adaptive = float(np.clip(gamma + 0.05 * (n_chars - 6), gamma, gamma + 3.0))

    # ── Build per-character heatmaps ─────────────────────────────────
    heatmaps = []
    for ci in range(n_show):
        t = positions[ci]       # CTC time step for this character
        # The attention matrix is [S, S] where S = num_reg + T.
        # Row (num_reg + t) is the query from patch token t.
        # Columns num_reg: are the patch key positions.
        row_idx = num_reg + t
        if row_idx >= attn.shape[0]:
            # Fallback: uniform
            patch_attn = np.ones(T) / T
        else:
            patch_attn = attn[row_idx, num_reg:]    # [T] — attention to patches
            if patch_attn.shape[0] != T:
                patch_attn = patch_attn[:T]

        # Percentile-based relative normalisation: clip 1st–99th percentile
        # so the full colour range maps to the relevant attention region.
        # Handles both short text (concentrated) and long text (diffuse).
        p_low  = max(0.0, np.percentile(patch_attn, 1.0))
        p_high = np.percentile(patch_attn, 99.0)
        if p_high - p_low < 1e-8:
            p_low, p_high = patch_attn.min(), patch_attn.max()
        patch_attn = np.clip(patch_attn, p_low, p_high)
        patch_attn = (patch_attn - p_low) / (p_high - p_low + 1e-8)

        # Adaptive power contrast
        patch_attn = patch_attn ** gamma_adaptive

        grid = patch_attn.reshape(Hp, Wp)
        heat_up = upscale_patch_aligned(grid, H_img, W_img, fixed_size,
                                        border_size)
        heatmaps.append(heat_up)

    # ── Multi-panel figure ───────────────────────────────────────────
    n_cols = n_show + 1   # +1 for original image
    fig_w = min(3 * n_cols, 64)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 4))
    if n_cols == 1:
        axes = [axes]

    # Panel 0: Original image
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # Panels 1..N: Per-character attention
    for ci in range(n_show):
        ax = axes[ci + 1]
        ax.imshow(img_np, cmap="gray")
        # inferno: perceptually uniform, high-contrast, better than jet
        im = ax.imshow(heatmaps[ci], cmap="inferno", alpha=alpha,
                       vmin=0.0, vmax=1.0)
        ch = chars[ci]
        # Use repr for whitespace chars so they are visible in title
        ch_label = repr(ch) if ch in (' ', '\t', '\n') else f'"{ch}"'
        ax.set_title(f'{ch_label}', fontsize=9)
        ax.axis("off")
        # Colorbar on the last displayed character panel
        if ci == n_show - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                orientation='vertical')
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label('rel. attn.', fontsize=5)

    # ── Suptitle with metadata ───────────────────────────────────────
    title_parts = [image_name]
    if cer is not None and wer is not None:
        title_parts.append(f"CER={cer:.4f}  WER={wer:.4f}")
    if gt_text:
        gt_disp = gt_text if len(gt_text) <= 60 else gt_text[:57] + '...'
        title_parts.append(f"GT: {gt_disp}")
    if pred_text:
        pr_disp = pred_text if len(pred_text) <= 60 else pred_text[:57] + '...'
        title_parts.append(f"Pred: {pr_disp}")
    title_parts.append(f"Layer: attention, Chars shown: {n_show}/{len(chars)}")

    fig.suptitle("\n".join(title_parts), fontsize=7, y=1.04)
    plt.tight_layout()

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── Test-set mode ────────────────────────────────────────────────────────────

def run_test_set_mode(net, config, device, out_dir, layer, alpha, dpi,
                      max_chars):
    """Run character attention for every sample in the test set."""
    dataset_folder = config.data.path
    fixed_size = (config.preproc.image_height, config.preproc.image_width)

    test_set = HTRDataset(dataset_folder, "test", fixed_size=fixed_size,
                          transforms=None)
    print(f'Test samples : {len(test_set)}')

    loader = DataLoader(test_set, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=False)

    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    cer_meter = CER()
    wer_meter = WER(mode=config.eval.wer_mode)

    print(f'\nGenerating character attention visualizations on TEST set...')

    saved = 0
    for sample_idx, (imgs, transcrs) in enumerate(
            tqdm.tqdm(loader, desc='CharAttn')):
        gt_text = transcrs[0].strip()
        img_path = test_set.data[sample_idx][0]
        image_name = Path(img_path).stem

        result = extract_character_attention(net, imgs, device, layer)

        # Map characters to time steps
        chars, positions = ctc_char_positions(result['logits'], i2c)
        pred_text = ''.join(chars).strip()

        # CER / WER
        cer_s = CER()
        wer_s = WER(mode=config.eval.wer_mode)
        cer_s.update(pred_text, gt_text)
        wer_s.update(pred_text, gt_text)
        cer_meter.update(pred_text, gt_text)
        wer_meter.update(pred_text, gt_text)

        save_path = os.path.join(out_dir, f"{image_name}_char_attention.png")
        out = visualize_character_attention(
            img_path, result['attn'], result['grid_size'], result['num_reg'],
            chars, positions, pred_text,
            gt_text=gt_text, cer=cer_s.score(), wer=wer_s.score(),
            save_path=save_path, alpha=alpha, dpi=dpi, max_chars=max_chars,
            fixed_size=fixed_size,
        )
        if out:
            print(f'  [{sample_idx:>5d}] {image_name:>40s}  ->  '
                  f'{os.path.basename(out)}')
            saved += 1

        if sample_idx % 50 == 0:
            gc.collect()

    print(f'\nTest set complete!')
    print(f'  Visualized : {saved}/{len(test_set)}')
    print(f'  CER        : {cer_meter.score():.4f}')
    print(f'  WER        : {wer_meter.score():.4f}')


# ── Image mode (single / multiple / directory) ──────────────────────────────

def run_image_mode(net, config, device, out_dir, image_paths,
                   layer, alpha, dpi, max_chars):
    """Run character attention for user-specified image(s)."""
    fixed_size = (config.preproc.image_height, config.preproc.image_width)

    dataset_folder = config.data.path
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    explicit_gt = getattr(config, 'gt', None)
    gt_dict = find_gt_for_images(image_paths, explicit_gt_path=explicit_gt)

    cer_meter = CER()
    wer_meter = WER(mode=config.eval.wer_mode)
    has_gt = bool(gt_dict)

    print(f'\nGenerating character attention for {len(image_paths)} image(s)...')

    saved = 0
    for idx, img_path in enumerate(
            tqdm.tqdm(image_paths, desc='CharAttn')):
        image_name = Path(img_path).stem

        raw_img = load_image(img_path)
        processed = preprocess(raw_img, fixed_size)
        img_tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)

        result = extract_character_attention(net, img_tensor, device, layer)

        chars, positions = ctc_char_positions(result['logits'], i2c)
        pred_text = ''.join(chars).strip()

        gt_text = gt_dict.get(image_name, '')
        cer_val = wer_val = None
        if gt_text:
            cer_s = CER()
            wer_s = WER(mode=config.eval.wer_mode)
            cer_s.update(pred_text, gt_text)
            wer_s.update(pred_text, gt_text)
            cer_val = cer_s.score()
            wer_val = wer_s.score()
            cer_meter.update(pred_text, gt_text)
            wer_meter.update(pred_text, gt_text)

        save_path = os.path.join(out_dir, f"{image_name}_char_attention.png")
        out = visualize_character_attention(
            img_path, result['attn'], result['grid_size'], result['num_reg'],
            chars, positions, pred_text,
            gt_text=gt_text, cer=cer_val, wer=wer_val,
            save_path=save_path, alpha=alpha, dpi=dpi, max_chars=max_chars,
            fixed_size=fixed_size,
        )
        if out:
            tqdm.tqdm.write(
                f'  {image_name:>40s}  ->  "{pred_text}"'
                + (f'  (GT: "{gt_text}")' if gt_text else ''))
            saved += 1

    print(f'\nImage mode complete!')
    print(f'  Visualized : {saved}/{len(image_paths)}')
    if has_gt and cer_meter.total_len > 0:
        print(f'  CER        : {cer_meter.score():.4f}')
        print(f'  WER        : {wer_meter.score():.4f}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config, image_paths, save_dir, layer, alpha, dpi, max_chars = parse_args()

    # Resolve device
    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        device = "cpu"
        config.device = "cpu"

    # Load character classes
    dataset_folder = config.data.path
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    num_classes = len(classes) + 1

    net, device = build_model(config, num_classes)

    run_dir = os.path.dirname(os.path.abspath(config.resume))
    out_dir = save_dir if save_dir else os.path.join(run_dir, "visualizations", "character_attention")
    os.makedirs(out_dir, exist_ok=True)

    mode = 'images' if image_paths else 'test_set'

    print('=' * 60)
    print('Character-level Attention Visualization')
    print('=' * 60)
    if image_paths:
        print(f'Mode         : image ({len(image_paths)} file(s))')
    else:
        print(f'Mode         : test_set')
    print(f'Attn layer   : {layer}')
    print(f'Output dir   : {out_dir}')
    print(f'Alpha        : {alpha}')
    print(f'DPI          : {dpi}')
    print(f'Max chars    : {max_chars}')
    gt_spec = getattr(config, 'gt', None)
    if gt_spec:
        print(f'GT file      : {gt_spec}')
    print(f'Timestamp    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    if mode == 'test_set':
        run_test_set_mode(net, config, device, out_dir, layer, alpha, dpi,
                          max_chars)
    else:
        run_image_mode(net, config, device, out_dir, image_paths,
                       layer, alpha, dpi, max_chars)

    print(f'\nAll character attention PNGs saved to: {out_dir}')
    print('=' * 60)


if __name__ == "__main__":
    main()
