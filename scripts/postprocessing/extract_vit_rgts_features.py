#!/usr/bin/env python3
"""
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

def save_sample_features(out_dir, sample_idx, features, image_name=None):
    """
    Save .npy files for one sample.

    Always saves with sample_{XXXXX}_ prefix (numeric index) for downstream
    script compatibility.  If image_name is provided, also saves with the
    image name as prefix for easy human lookup.
    """
    tag = f"sample_{sample_idx:05d}"

    np.save(os.path.join(out_dir, f"{tag}_reg_tokens.npy"),
            features['reg_tokens'])
    np.save(os.path.join(out_dir, f"{tag}_seq_tokens.npy"),
            features['seq_tokens'])
    np.save(os.path.join(out_dir, f"{tag}_token_norms.npy"),
            features['token_norms'])
    np.save(os.path.join(out_dir, f"{tag}_logits.npy"),
            features['logits'])
    if 'attn_maps' in features:
        np.save(os.path.join(out_dir, f"{tag}_attn_maps.npy"),
                features['attn_maps'], allow_pickle=True)

    # Also save with image name (symlink-friendly lookup)
    if image_name is not None:
        for suffix in ['reg_tokens', 'seq_tokens', 'token_norms', 'logits']:
            src = os.path.join(out_dir, f"{tag}_{suffix}.npy")
            dst = os.path.join(out_dir, f"{image_name}_{suffix}.npy")
            if not os.path.exists(dst):
                # Copy instead of symlink for portability
                np.save(dst, np.load(src, allow_pickle=True))
        if 'attn_maps' in features:
            src = os.path.join(out_dir, f"{tag}_attn_maps.npy")
            dst = os.path.join(out_dir, f"{image_name}_attn_maps.npy")
            if not os.path.exists(dst):
                np.save(dst, np.load(src, allow_pickle=True), allow_pickle=True)


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

        # Save .npy files (numeric index only — image name from dataset path)
        save_sample_features(out_dir, sample_idx, features)

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

        # Save .npy files — numeric index + image name for easy lookup
        save_sample_features(out_dir, sample_idx, features,
                             image_name=image_name)

        # CSV row (gt_text empty — no ground truth for ad-hoc images)
        Hp, Wp = features['grid_size']
        writer.writerow([
            int(sample_idx), img_path, "", pred_text,
            "", "",
            int(features['reg_tokens'].shape[1]),
            int(features['seq_tokens'].shape[0]),
            int(Hp), int(Wp),
        ])
        csv_f.flush()

        tqdm.tqdm.write(f'  {image_name:>40s}  ->  "{pred_text}"')

    csv_f.close()

    print(f'\nImage extraction complete!')
    print(f'  Images : {len(image_paths)}')
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
    print(f'Timestamp    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    if mode == 'test_set':
        run_test_set_mode(net, config, device, out_dir, extract_attention)
    else:
        run_image_mode(net, config, device, out_dir, image_paths,
                       extract_attention)

    print(f'\nAll features saved to: {out_dir}')
    print('=' * 70)


if __name__ == "__main__":
    main()
