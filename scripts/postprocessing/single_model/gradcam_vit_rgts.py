#!/usr/bin/env python3
"""
write git commit msg -
update: Grad-CAM visualization for ViT-RGTS backbone
Grad-CAM Patch Importance Visualization for ViT-RGTS

Generates Grad-CAM style heatmaps showing which image patches contribute
most to the model's predictions.  Overlays importance on the original image.

Two-panel output per sample:
  Panel 1: Original grayscale image with GT and predicted text
  Panel 2: Grad-CAM heatmap overlay on the original image

How it works (Selvaraju et al. 2017, adapted for ViT tokens):
  The backbone produces T patch token activations A [T, 1, D].
  We retain gradients on A, run through the CTC head, and backpropagate
  from the sum of NON-BLANK predicted-class logits.  Importance per token:
      importance[t] = ReLU( Σ_d  grad_{t,d} · A_{t,d} )
  i.e. gradient × activation (not just |gradient|), with ReLU to keep
  only positive contributions.  Scores are reshaped to (Hp, Wp) and
  overlaid using patch-aligned nearest-neighbour upscaling.

Supports four execution modes:
  1. Test set  (default) — Grad-CAM for every sample in data/IAM/test
  2. Single image        — one image after the -- separator
  3. Multiple images     — several images after --
  4. Directory           — a folder of images after --

Supported Architectures: vit_rgts only (accesses backbone patch embeddings)
Input: config YAML(s) + model checkpoint + optional image path(s)
Output: Per-sample Grad-CAM PNG (auto-saved to <run_dir>/vit_rgts_explain/)

Usage:
    # ── Test set mode (default — no image paths after --) ───────────
    python scripts/postprocessing/gradcam_vit_rgts.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt

    # ── Single image ───────────────────────────────────────────────
    python scripts/postprocessing/gradcam_vit_rgts.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- path/to/image.png

    # ── Multiple images ────────────────────────────────────────────
    python scripts/postprocessing/gradcam_vit_rgts.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- img1.png img2.png img3.png

    # ── Directory of images ────────────────────────────────────────
    python scripts/postprocessing/gradcam_vit_rgts.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- path/to/image_folder/

    # ── Save to custom directory ───────────────────────────────────
    python scripts/postprocessing/gradcam_vit_rgts.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        --save-dir output/gradcam/

    # ── Adjust overlay and resolution ──────────────────────────────
    python scripts/postprocessing/gradcam_vit_rgts.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        --alpha 0.5 --dpi 200

    # ── GPU acceleration ──────────────────────────────────────────
    python scripts/postprocessing/gradcam_vit_rgts.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt device=cuda

    # ── With ground truth file (fills GT / CER / WER in title) ────
    python scripts/postprocessing/gradcam_vit_rgts.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        gt=notebook/sample_images/gt.txt \\
        -- notebook/sample_images/

Arguments:
    config.yaml      Base config file (required)
    extra.yaml       Additional config files merged in order
    key=value        Override config params (e.g., resume=..., device=cuda)
    --               Separator; image paths follow
    --save-dir PATH  Directory for PNGs (default: <run_dir>/vit_rgts_explain/)
    --alpha FLOAT    Heatmap overlay opacity 0.0–1.0 (default: 0.40)
    --dpi INT        Output resolution (default: 150)
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

    Before '--': config YAMLs + key=value overrides + --save-dir/--alpha/--dpi
    After  '--': image paths (single, multiple, or directory)
    If no '--' separator, runs in test-set mode.

    Returns: (config, image_paths, save_dir, alpha, dpi)
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
    alpha = 0.40
    dpi = 150

    filtered = []
    i = 0
    while i < len(config_args):
        if config_args[i] == '--save-dir' and i + 1 < len(config_args):
            save_dir = config_args[i + 1]
            i += 2
        elif config_args[i] == '--alpha' and i + 1 < len(config_args):
            alpha = float(config_args[i + 1])
            i += 2
        elif config_args[i] == '--dpi' and i + 1 < len(config_args):
            dpi = int(config_args[i + 1])
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

    # Resolve image paths (files and directories)
    image_paths = collect_image_paths(image_args)

    # If user explicitly used '--' but no valid images found, error out
    if has_separator and not image_paths:
        print('Error: "--" separator was used but no valid images found.')
        print(f'  Provided paths: {image_args}')
        sys.exit(1)

    return conf, image_paths, save_dir, alpha, dpi


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
    """
    Load a ground truth file into a dict {image_stem: gt_text}.

    Format: <image_stem><whitespace><ground_truth_text>  (one per line)
    Lines starting with '#' are ignored.
    """
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
    """
    Locate and load ground truth for a list of image paths.

    Priority: explicit gt= override > gt.txt auto-detected in image directory.
    Returns a dict {image_stem: gt_text}, may be empty.
    """
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
    """Instantiate HTRNet and load checkpoint weights (kept in train mode)."""
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

    # Full eval mode: disables dropout in both backbone and CTC head (GRU).
    # GRU backward works fine in eval mode on CPU and modern PyTorch.
    # This ensures deterministic, reproducible Grad-CAM importance maps.
    net.eval()

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Parameters   : {n_params:,}')

    # Validate architecture
    arch_type = getattr(config.arch, "type", "cnn_rnn")
    if arch_type != "vit_rgts":
        raise ValueError(
            f"Expected arch.type == 'vit_rgts', got '{arch_type}'. "
            "This script is specific to the ViT-RGTS backbone."
        )

    if not hasattr(net, "backbone"):
        raise AttributeError(
            "HTRNet has no 'backbone' attribute. "
            "Check that models.py contains ViTRGTSBackbone."
        )

    return net, device


# ── CTC decode ───────────────────────────────────────────────────────────────

def ctc_decode(tdec, i2c, blank_id=0):
    """Greedy CTC decode: collapse repeats, strip blank (index 0)."""
    tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
    return "".join([i2c[t] for t in tt if t != blank_id])


# ── Grad-CAM computation ─────────────────────────────────────────────────────

def compute_gradcam(net, img_tensor, device):
    """
    Proper Grad-CAM (Selvaraju et al. 2017) for ViT patch tokens.

    Steps:
      1. Forward backbone → activations A [T, 1, D]
      2. Retain gradients on A
      3. Forward CTC head → logits [T, 1, C]
      4. Backward from sum of NON-BLANK predicted-class logits
         (excludes CTC blank tokens to avoid diluting the signal)
      5. importance[t] = ReLU( Σ_d grad_{t,d} · A_{t,d} )
         Grad × activation weighting + ReLU (positive contributions only)

    Parameters
    ----------
    net         : HTRNet (backbone.eval, top.train)
    img_tensor  : [1, 1, H, W] tensor
    device      : torch device string

    Returns
    -------
    dict with:
        importance  : [T] numpy, per-patch importance (normalised 0-1)
        grid_size   : (Hp, Wp) tuple
        logits      : [T, 1, C] numpy
    """
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(False)

    # 1. Backbone forward → activations A [T, 1, D]
    seq_tokens, reg_tokens, (Hp, Wp) = net.backbone(img_tensor)
    seq_tokens.retain_grad()

    # 2. CTC head forward
    seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)   # [1, D, 1, T]
    logits = net.top(seq_4d)
    if isinstance(logits, tuple):
        logits = logits[0]                                # [T, 1, C]

    # 3. Backward from NON-BLANK predicted-class logits only
    pred_classes = logits.argmax(dim=2)                   # [T, 1]
    non_blank = pred_classes.squeeze(1) != 0              # [T] bool
    if non_blank.any():
        target = logits[non_blank].gather(
            2, pred_classes[non_blank].unsqueeze(2)).sum()
    else:
        # Fallback: all predictions blank (unlikely)
        target = logits.gather(2, pred_classes.unsqueeze(2)).sum()

    net.zero_grad()
    target.backward()

    # 4. Grad-CAM: grad × activation, sum over D, ReLU
    grads = seq_tokens.grad.detach()                      # [T, 1, D]
    acts  = seq_tokens.detach()                           # [T, 1, D]
    cam = (grads * acts).squeeze(1).sum(dim=-1)           # [T]
    cam = torch.relu(cam).cpu().numpy()                   # positive only

    # 5. Normalise to [0, 1]
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


# ── Patch-aligned upscaling ──────────────────────────────────────────────────

def upscale_patch_aligned(patch_map, H_img, W_img, fixed_size=None,
                          border_size=8):
    """
    Map patch-level heatmap to original image coordinates.

    The model operates on a *preprocessed* image of ``fixed_size`` (e.g.
    128x1024), not the raw image (e.g. 42x162).  Within the preprocessed
    canvas the original image sits at an offset of ``border_size`` pixels
    (top, left).  Each patch covers (H_preproc//Hp, W_preproc//Wp) pixels
    in the preprocessed image.

    Steps:
      1. Upscale to preprocessed resolution with nearest-neighbour (kron)
      2. Crop the region that corresponds to the original image

    When ``fixed_size`` is ``None`` the legacy 1:1 mapping is used as a
    fallback to keep backward compatibility.
    """
    Hp, Wp = patch_map.shape

    if fixed_size is None:
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

def visualize_gradcam(img_path, importance, grid_size,
                      pred_text, gt_text='', cer=None, wer=None,
                      save_path=None, alpha=0.40, dpi=150,
                      fixed_size=None, border_size=8):
    """
    Create a 2-panel Grad-CAM visualisation and save as PNG.

    Panel 1: Original grayscale image
    Panel 2: Grad-CAM heatmap overlay (patch-aligned upscaling)

    Returns the save path, or None on failure.
    """
    Hp, Wp = grid_size
    T = importance.shape[0]

    if T != Hp * Wp:
        print(f'  [WARN] T={T} != Hp*Wp={Hp * Wp}; cannot reshape. Skipping.')
        return None

    heat_grid = importance.reshape(Hp, Wp)

    # Load the original image
    if not os.path.isfile(img_path):
        print(f'  [SKIP] Image not found: {img_path}')
        return None
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape

    # Patch-aligned upscaling — correct preprocessed-space coordinates
    heat_up = upscale_patch_aligned(heat_grid, H_img, W_img, fixed_size,
                                    border_size)

    # Percentile-based relative scaling: clip 1st–99th percentile so
    # the full colour range covers the relevant importance region.
    # Works for both short text (concentrated) and long text (diffuse).
    p01 = np.percentile(heat_up,  1)
    p99 = np.percentile(heat_up, 99)
    if p99 - p01 < 1e-8:
        p01, p99 = heat_up.min(), heat_up.max()
    heat_scaled = np.clip((heat_up - p01) / (p99 - p01 + 1e-8), 0.0, 1.0)

    image_name = Path(img_path).stem

    # ── 2-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Original image
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # Panel 2: Grad-CAM overlay with inferno colormap (perceptually uniform)
    axes[1].imshow(img_np, cmap="gray")
    im = axes[1].imshow(heat_scaled, cmap="inferno", alpha=alpha,
                        vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Patch Importance", fontsize=10)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                        orientation='vertical')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('relative importance', fontsize=6)

    # ── Title with metadata ──────────────────────────────────────────
    title_parts = [image_name]
    if cer is not None and wer is not None:
        title_parts.append(f"CER={cer:.4f}  WER={wer:.4f}")
    if gt_text:
        title_parts.append(f"GT: {gt_text}")
    if pred_text:
        title_parts.append(f"Pred: {pred_text}")
    title_parts.append("Method: Grad-CAM (Selvaraju 2017)  |  ReLU(grad·act)  |  p1–p99 scaling")

    fig.suptitle("\n".join(title_parts), fontsize=8, y=1.02)
    plt.tight_layout()

    # Save
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return save_path


# ── Test-set mode ────────────────────────────────────────────────────────────

def run_test_set_mode(net, config, device, out_dir, alpha, dpi):
    """Run Grad-CAM for every sample in the test set."""
    dataset_folder = config.data.path
    fixed_size = (config.preproc.image_height, config.preproc.image_width)

    test_set = HTRDataset(dataset_folder, "test", fixed_size=fixed_size,
                          transforms=None)
    print(f'Test samples : {len(test_set)}')

    loader = DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=0,   # single-threaded: avoids edge-cases with grad tracking
        pin_memory=False,
    )

    # Character classes
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    cer_meter = CER()
    wer_meter = WER(mode=config.eval.wer_mode)

    print(f'\nGenerating Grad-CAM visualizations on TEST set...')

    saved = 0
    for sample_idx, (imgs, transcrs) in enumerate(
            tqdm.tqdm(loader, desc='Grad-CAM')):
        gt_text = transcrs[0].strip()
        img_path = test_set.data[sample_idx][0]
        image_name = Path(img_path).stem

        result = compute_gradcam(net, imgs, device)

        # CTC decode
        tdec = result['logits'].argmax(2).transpose(1, 0).squeeze()
        pred_text = ctc_decode(tdec, i2c).strip()

        # CER / WER
        cer_s = CER()
        wer_s = WER(mode=config.eval.wer_mode)
        cer_s.update(pred_text, gt_text)
        wer_s.update(pred_text, gt_text)
        cer_meter.update(pred_text, gt_text)
        wer_meter.update(pred_text, gt_text)

        # Visualize + save
        save_path = os.path.join(out_dir, f"{image_name}_gradcam.png")
        out = visualize_gradcam(
            img_path, result['importance'], result['grid_size'],
            pred_text, gt_text=gt_text,
            cer=cer_s.score(), wer=wer_s.score(),
            save_path=save_path, alpha=alpha, dpi=dpi,
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

def run_image_mode(net, config, device, out_dir, image_paths, alpha, dpi):
    """Run Grad-CAM for user-specified image(s) or a directory."""
    fixed_size = (config.preproc.image_height, config.preproc.image_width)

    # Character classes
    dataset_folder = config.data.path
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    # Ground truth: explicit gt= override or auto-detect gt.txt in image dirs
    explicit_gt = getattr(config, 'gt', None)
    gt_dict = find_gt_for_images(image_paths, explicit_gt_path=explicit_gt)

    cer_meter = CER()
    wer_meter = WER(mode=config.eval.wer_mode)
    has_gt = bool(gt_dict)

    print(f'\nGenerating Grad-CAM for {len(image_paths)} image(s)...')

    saved = 0
    for idx, img_path in enumerate(
            tqdm.tqdm(image_paths, desc='Grad-CAM')):
        image_name = Path(img_path).stem

        # Load & preprocess (same pipeline as training)
        raw_img = load_image(img_path)
        processed = preprocess(raw_img, fixed_size)
        img_tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)

        result = compute_gradcam(net, img_tensor, device)

        # CTC decode
        tdec = result['logits'].argmax(2).transpose(1, 0).squeeze()
        pred_text = ctc_decode(tdec, i2c).strip()

        # Ground truth + metrics
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

        # Visualize + save
        save_path = os.path.join(out_dir, f"{image_name}_gradcam.png")
        out = visualize_gradcam(
            img_path, result['importance'], result['grid_size'],
            pred_text, gt_text=gt_text,
            cer=cer_val, wer=wer_val,
            save_path=save_path, alpha=alpha, dpi=dpi,
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
    config, image_paths, save_dir, alpha, dpi = parse_args()

    # Resolve device
    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        device = "cpu"
        config.device = "cpu"

    # Load character classes
    dataset_folder = config.data.path
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    num_classes = len(classes) + 1  # +1 for CTC blank

    # Build model (train mode for gradient backprop)
    net, device = build_model(config, num_classes)

    # Output directory: default = <run_dir>/vit_rgts_explain/
    run_dir = os.path.dirname(os.path.abspath(config.resume))
    out_dir = save_dir if save_dir else os.path.join(run_dir, "vit_rgts_explain")
    os.makedirs(out_dir, exist_ok=True)

    mode = 'images' if image_paths else 'test_set'

    print('=' * 60)
    print('Grad-CAM Patch Importance Visualization')
    print('=' * 60)
    if image_paths:
        print(f'Mode         : image ({len(image_paths)} file(s))')
    else:
        print(f'Mode         : test_set')
    print(f'Output dir   : {out_dir}')
    print(f'Alpha        : {alpha}')
    print(f'DPI          : {dpi}')
    gt_spec = getattr(config, 'gt', None)
    if gt_spec:
        print(f'GT file      : {gt_spec}')
    print(f'Timestamp    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    if mode == 'test_set':
        run_test_set_mode(net, config, device, out_dir, alpha, dpi)
    else:
        run_image_mode(net, config, device, out_dir, image_paths,
                       alpha, dpi)

    print(f'\nAll Grad-CAM PNGs saved to: {out_dir}')
    print('=' * 60)


if __name__ == "__main__":
    main()
