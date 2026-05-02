#!/usr/bin/env python3
"""
write git commit msg -
update: Extract ViT-RGTS features for post-hoc analysis
Extract ViT-RGTS Features for Post-hoc Analysis

Runs the trained ViT-RGTS model and extracts per-sample:
  - Register token embeddings  (*_reg_tokens.npy)
  - Sequence/patch token embeddings (*_seq_tokens.npy)
  - CTC logits (*_logits.npy)
  - Token norms (*_token_norms.npy)
  - Attention maps via forward_explain() (*_attn_maps.npy)  [optional]
  - Metadata CSV with predictions, ground truth, CER, WER

Supports four execution modes:
  1. Test set  (default) — extract features for every sample in data/IAM/test
  2. Single image        — one image after the -- separator
  3. Multiple images     — several images after --
  4. Directory           — a folder of images after --

The extracted features are consumed by downstream visualization scripts:
  - visualize_register_attention.py (register-to-patch similarity)
  - visualize_token_norms.py (token norm heatmaps)
  - tsne_register_tokens.py (t-SNE of register embeddings)

Supported Architectures: vit_rgts only
Input: config YAML(s) + model checkpoint + optional image path(s)
Output: <run_dir>/vit_rgts_explain/ with .npy files and metadata CSV

Usage:
    # ── Test set mode (default — no image paths after --) ───────────────
    python scripts/postprocessing/extract_vit_rgts_features.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt

    # ── Single image ───────────────────────────────────────────────────
    python scripts/postprocessing/extract_vit_rgts_features.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- path/to/image.png

    # ── Multiple images ────────────────────────────────────────────────
    python scripts/postprocessing/extract_vit_rgts_features.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- img1.png img2.png img3.png

    # ── Directory of images ────────────────────────────────────────────
    python scripts/postprocessing/extract_vit_rgts_features.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        -- path/to/image_folder/

    # ── Also extract attention maps (adds *_attn_maps.npy per sample) ──
    python scripts/postprocessing/extract_vit_rgts_features.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        extract_attention=true

    # ── Run_50 (0 registers) on test set ──────────────────────────────
    python scripts/postprocessing/extract_vit_rgts_features.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=0 \\
        resume=saved_models/experiments/run_50/model.pt

    # ── GPU acceleration ──────────────────────────────────────────────
    python scripts/postprocessing/extract_vit_rgts_features.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt device=cuda

    # ── With explicit ground truth file (fills gt_text, cer, wer) ────
    python scripts/postprocessing/extract_vit_rgts_features.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        arch.num_registers=16 \\
        resume=saved_models/experiments/run_54/model.pt \\
        gt=notebook/sample_images/gt.txt \\
        -- notebook/sample_images/

    # NOTE: A gt.txt in the same folder as the images is found automatically.
    # gt.txt format (one line per image, tab or space separated):
    #   <image_stem>  <ground_truth_text>
    # Example:
    #   a01-038-12 talks.
"""
import os
import sys
import csv
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
from omegaconf import OmegaConf

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
    Parse config YAML(s), key=value overrides, and optional image paths.

    Everything before '--' is config.  Everything after '--' is image paths.
    If no '--' separator, all arguments are treated as config (test-set mode).

    Returns: (config, image_paths)
        image_paths is empty list for test-set mode.
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

    yaml_files = [a for a in config_args if a.endswith('.yaml')]
    overrides = [a for a in config_args if not a.endswith('.yaml')]

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
        print('  Check that the paths exist and are files or directories.')
        sys.exit(1)

    return conf, image_paths


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

    Supported format (one entry per line):
        <image_stem><whitespace><ground_truth_text>
    Example:
        a01-038-12 talks.
        c04-110-00 Become a success...
    Lines starting with '#' are ignored.
    """
    gt = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)   # split on first whitespace
            if len(parts) == 2:
                gt[parts[0]] = parts[1]
            elif len(parts) == 1:
                gt[parts[0]] = ''         # image with empty GT
    return gt


def find_gt_for_images(image_paths, explicit_gt_path=None):
    """
    Locate and load ground truth for a list of image paths.

    Priority:
        1. explicit_gt_path (from gt= CLI override)
        2. gt.txt auto-detected in the same directory as the images

    Returns a dict {image_stem: gt_text}, may be empty if no GT found.
    """
    # Explicit path takes top priority
    if explicit_gt_path:
        if not os.path.isfile(explicit_gt_path):
            print(f'Warning: specified gt file not found: {explicit_gt_path}')
            return {}
        print(f'Ground truth : {explicit_gt_path}')
        return load_gt_file(explicit_gt_path)

    # Auto-detect gt.txt in each unique image directory
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
    """Instantiate HTRNet and load checkpoint weights."""
    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        device = "cpu"
        config.device = "cpu"

    print(f'Architecture : {config.arch.type}')

    net = HTRNet(config.arch, num_classes)

    if config.resume is None:
        raise ValueError(
            "config.resume is None — you must point to a trained checkpoint "
            "(e.g. resume=saved_models/experiments/run_54/model.pt)"
        )

    print(f'Checkpoint   : {config.resume}')
    checkpoint = torch.load(config.resume, map_location=device)
    print(net.load_state_dict(checkpoint, strict=True))

    net.to(device)
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

def ctc_decode(tdec, tdict, blank_id=0):
    """Greedy CTC decode: collapse repeats, strip blank (index 0)."""
    tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
    return "".join([tdict[t] for t in tt if t != blank_id])


# ── Feature extraction (single sample) ───────────────────────────────────────

def extract_sample_features(net, img_tensor, device, extract_attention=False):
    """
    Extract all features from one image tensor [1, 1, H, W].

    Returns dict with:
        reg_tokens   : [1, R, D] numpy
        seq_tokens   : [T, 1, D] numpy
        logits       : [T, 1, C] numpy
        token_norms  : [R+T] numpy
        grid_size    : (Hp, Wp)
        attn_maps    : list of [1, H, S, S] numpy  (only if extract_attention)
    """
    img_tensor = img_tensor.to(device)
    result = {}

    with torch.no_grad():
        # ── Backbone: get register + sequence tokens + grid ──────────
        seq_tokens, reg_tokens, (Hp, Wp) = net.backbone(img_tensor)
        grid_size = (Hp, Wp)

        result['reg_tokens'] = reg_tokens.cpu().numpy()    # [1, R, D]
        result['seq_tokens'] = seq_tokens.cpu().numpy()    # [T, 1, D]

        # ── Token norms: concat registers + patches → L2 norm ───────
        seq_bt = seq_tokens.permute(1, 0, 2)               # [1, T, D]
        tokens_all = torch.cat([reg_tokens, seq_bt], dim=1) # [1, R+T, D]
        result['token_norms'] = tokens_all.norm(dim=-1).cpu().numpy()[0]  # [R+T]

        # ── CTC logits via the head ──────────────────────────────────
        seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)  # [B, D, 1, T]
        logits = net.top(seq_4d)
        if isinstance(logits, tuple):
            logits = logits[0]
        result['logits'] = logits.cpu().numpy()             # [T, 1, C]
        result['grid_size'] = grid_size

        # ── Optional: attention maps via forward_explain() ───────────
        if extract_attention:
            _, _, attn_maps, _, _ = net.forward_explain(img_tensor)
            result['attn_maps'] = [a.cpu().numpy() for a in attn_maps]

    return result


# ── Save features for one sample ─────────────────────────────────────────────

def save_sample_features(out_dir, image_name, features):
    """
    Save .npy files for one sample using the image name as prefix.

    Files saved:
      {image_name}_reg_tokens.npy   — register token embeddings [1, R, D]
      {image_name}_seq_tokens.npy   — patch token embeddings    [T, 1, D]
      {image_name}_token_norms.npy  — L2 norms per token        [R+T]
      {image_name}_logits.npy       — CTC logits                [T, 1, C]
      {image_name}_attn_maps.npy    — attention maps (optional)
    """
    np.save(os.path.join(out_dir, f"{image_name}_reg_tokens.npy"),
            features['reg_tokens'])
    np.save(os.path.join(out_dir, f"{image_name}_seq_tokens.npy"),
            features['seq_tokens'])
    np.save(os.path.join(out_dir, f"{image_name}_token_norms.npy"),
            features['token_norms'])
    np.save(os.path.join(out_dir, f"{image_name}_logits.npy"),
            features['logits'])
    if 'attn_maps' in features:
        np.save(os.path.join(out_dir, f"{image_name}_attn_maps.npy"),
                features['attn_maps'], allow_pickle=True)


# ── Test-set mode ────────────────────────────────────────────────────────────

def run_test_set_mode(net, config, device, out_dir, extract_attention):
    """Extract features for every sample in the test set."""
    dataset_folder = config.data.path
    fixed_size = (config.preproc.image_height, config.preproc.image_width)

    test_set = HTRDataset(dataset_folder, "test", fixed_size=fixed_size,
                          transforms=None)
    print(f'Test samples : {len(test_set)}')

    loader = DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=config.eval.num_workers, pin_memory=False,
    )

    # Character classes
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    # CSV metadata (same schema used by downstream scripts)
    csv_path = os.path.join(out_dir, "vit_rgts_features.csv")
    csv_f = open(csv_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(csv_f)
    writer.writerow([
        "sample_idx", "img_path", "gt_text", "pred_text",
        "cer", "wer", "num_registers", "num_patches", "Hp", "Wp",
    ])
    csv_f.flush()

    cer_meter = CER()
    wer_meter = WER(mode=config.eval.wer_mode)

    print(f'\nExtracting features on TEST set '
          f'{"(with attention maps)" if extract_attention else "(backbone only)"}...')

    for sample_idx, (imgs, transcrs) in enumerate(
            tqdm.tqdm(loader, desc='Extracting')):
        gt_text = transcrs[0].strip()
        img_path = test_set.data[sample_idx][0]

        features = extract_sample_features(
            net, imgs, device, extract_attention)

        # CTC decode
        tdec = features['logits'].argmax(2).transpose(1, 0).squeeze()
        pred_text = ctc_decode(tdec, i2c).strip()

        # Per-sample metrics
        cer_s = CER()
        wer_s = WER(mode=config.eval.wer_mode)
        cer_s.update(pred_text, gt_text)
        wer_s.update(pred_text, gt_text)
        cer_meter.update(pred_text, gt_text)
        wer_meter.update(pred_text, gt_text)

        # Save .npy files using image name as prefix
        image_name = Path(img_path).stem
        save_sample_features(out_dir, image_name, features)

        # CSV row
        Hp, Wp = features['grid_size']
        writer.writerow([
            int(sample_idx), img_path, gt_text, pred_text,
            f"{cer_s.score():.6f}", f"{wer_s.score():.6f}",
            int(features['reg_tokens'].shape[1]),
            int(features['seq_tokens'].shape[0]),
            int(Hp), int(Wp),
        ])

        # Periodic flush + cleanup
        if sample_idx % 50 == 0:
            csv_f.flush()
            gc.collect()

        # Progress checkpoint
        if sample_idx > 0 and sample_idx % 500 == 0:
            print(f'\n  [Checkpoint {sample_idx}/{len(loader)}] '
                  f'CER: {cer_meter.score():.4f}, WER: {wer_meter.score():.4f}')

    csv_f.close()

    print(f'\nTest set extraction complete!')
    print(f'  Samples  : {sample_idx + 1}')
    print(f'  TEST CER : {cer_meter.score():.4f}')
    print(f'  TEST WER : {wer_meter.score():.4f}')
    print(f'  CSV      : {csv_path}')


# ── Image mode (single / multiple / directory) ──────────────────────────────

def run_image_mode(net, config, device, out_dir, image_paths, extract_attention):
    """Extract features for user-specified image(s) or a directory."""
    fixed_size = (config.preproc.image_height, config.preproc.image_width)

    # Character classes for CTC decode
    dataset_folder = config.data.path
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    # Ground truth: explicit gt= override or auto-detect gt.txt in image dirs
    explicit_gt = getattr(config, 'gt', None)
    gt_dict = find_gt_for_images(image_paths, explicit_gt_path=explicit_gt)

    cer_meter = CER()
    wer_meter = WER(mode=config.eval.wer_mode)
    has_gt = bool(gt_dict)

    # CSV metadata (same schema for downstream compatibility)
    csv_path = os.path.join(out_dir, "vit_rgts_features.csv")
    csv_f = open(csv_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(csv_f)
    writer.writerow([
        "sample_idx", "img_path", "gt_text", "pred_text",
        "cer", "wer", "num_registers", "num_patches", "Hp", "Wp",
    ])
    csv_f.flush()

    print(f'\nExtracting features for {len(image_paths)} image(s) '
          f'{"(with attention maps)" if extract_attention else "(backbone only)"}...')

    for sample_idx, img_path in enumerate(
            tqdm.tqdm(image_paths, desc='Extracting')):
        image_name = Path(img_path).stem

        # Load & preprocess (same pipeline as training)
        raw_img = load_image(img_path)
        processed = preprocess(raw_img, fixed_size)
        img_tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)

        features = extract_sample_features(
            net, img_tensor, device, extract_attention)

        # CTC decode
        tdec = features['logits'].argmax(2).transpose(1, 0).squeeze()
        pred_text = ctc_decode(tdec, i2c).strip()

        # Ground truth + metrics (if available)
        gt_text = gt_dict.get(image_name, '')
        if gt_text:
            cer_s = CER()
            wer_s = WER(mode=config.eval.wer_mode)
            cer_s.update(pred_text, gt_text)
            wer_s.update(pred_text, gt_text)
            cer_val = f'{cer_s.score():.6f}'
            wer_val = f'{wer_s.score():.6f}'
            cer_meter.update(pred_text, gt_text)
            wer_meter.update(pred_text, gt_text)
        else:
            cer_val = ''
            wer_val = ''

        # Save .npy files using image name as prefix
        save_sample_features(out_dir, image_name, features)

        # CSV row
        Hp, Wp = features['grid_size']
        writer.writerow([
            int(sample_idx), img_path, gt_text, pred_text,
            cer_val, wer_val,
            int(features['reg_tokens'].shape[1]),
            int(features['seq_tokens'].shape[0]),
            int(Hp), int(Wp),
        ])
        csv_f.flush()

        tqdm.tqdm.write(f'  {image_name:>40s}  ->  "{pred_text}"'
                        + (f'  (GT: "{gt_text}")' if gt_text else ''))

    csv_f.close()

    print(f'\nImage extraction complete!')
    print(f'  Images : {len(image_paths)}')
    if has_gt and cer_meter.total_len > 0:
        print(f'  CER    : {cer_meter.score():.4f}')
        print(f'  WER    : {wer_meter.score():.4f}')
    print(f'  CSV    : {csv_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config, image_paths = parse_args()

    # Resolve device
    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        device = "cpu"
        config.device = "cpu"

    # Load character classes for model construction
    dataset_folder = config.data.path
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))
    num_classes = len(classes) + 1  # +1 for CTC blank

    # Build model
    net, device = build_model(config, num_classes)

    # Output directory: <run_dir>/vit_rgts_explain/
    run_dir = os.path.dirname(os.path.abspath(config.resume))
    out_dir = os.path.join(run_dir, "vit_rgts_explain")
    os.makedirs(out_dir, exist_ok=True)

    # Check if attention extraction is requested
    extract_attention = getattr(config, 'extract_attention', False)
    if isinstance(extract_attention, str):
        extract_attention = extract_attention.lower() in ('true', '1', 'yes')

    # Determine execution mode
    mode = 'images' if image_paths else 'test_set'

    print('=' * 70)
    print('ViT-RGTS Feature Extraction')
    print('=' * 70)
    if image_paths:
        print(f'Mode         : image ({len(image_paths)} file(s))')
    else:
        print(f'Mode         : test_set')
    print(f'Output dir   : {out_dir}')
    print(f'Attention    : '
          f'{"yes (forward_explain)" if extract_attention else "no (backbone only)"}')
    gt_spec = getattr(config, 'gt', None)
    if gt_spec:
        print(f'GT file      : {gt_spec}')
    print(f'Timestamp    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    if mode == 'test_set':
        run_test_set_mode(net, config, device, out_dir, extract_attention)
    else:
        run_image_mode (net, config, device, out_dir, image_paths,
                       extract_attention)

    print(f'\nAll features saved to: {out_dir}')
    print('=' * 70)


if __name__ == "__main__":
    main()
