#!/usr/bin/env python3
"""
Explainable HTR Demo — Predict + Visualize Attention & Register Tokens

Loads a trained ViT-based HTR model and runs explainability analysis on one or
more handwriting images.  For each image it:
  1. Predicts the transcription (CTC decode)
  2. Extracts per-layer, per-head self-attention maps via forward_explain()
  3. Extracts register-token embeddings and L2 norms (ViT-RGTS / TorchVision ViT)
  4. Generates visualizations (attention heatmaps, register embeddings, token norms)
  5. Saves raw .npy arrays for downstream analysis

Supports single image, multiple images, or a directory of images.
Saves everything to <run_folder>/explainability/.

Supported Architectures: vit_rgts, torchvision_vit, trocr
  (cnn_rnn does NOT have forward_explain — use demo.py instead)

Output files (per image):
    <image_name>_attention_maps.png   — multi-head attention heatmaps
    <image_name>_register_tokens.png  — register embedding heatmap + L2 norms
    <image_name>_token_norms.png      — per-token L2 norms (register vs patch)
    <image_name>_input.png            — preprocessed input image
    attention_maps/  *.npy            — raw attention arrays
    register_tokens/ *.npy            — raw register embeddings
    token_norms/     *.npy            — raw token norm arrays
    predictions.txt                   — tab-separated filename → prediction

Usage:
    # Single image  (ViT-RGTS v2, run_54, 16 registers)
    python scripts/postprocessing/demo_explainability.py \
        configs/config.yaml configs/baseline.yaml \
        configs/baseline_vit_rgts_v2.yaml arch.num_registers=16 \
        resume=saved_models/experiments/run_54/model.pt \
        -- path/to/image.png

    # Multiple images
    python scripts/postprocessing/demo_explainability.py \
        configs/config.yaml configs/baseline.yaml \
        configs/baseline_vit_rgts_v2.yaml arch.num_registers=16 \
        resume=saved_models/experiments/run_54/model.pt \
        -- img1.png img2.png img3.png

    # Directory of images
    python scripts/postprocessing/demo_explainability.py \
        configs/config.yaml configs/baseline.yaml \
        configs/baseline_vit_rgts_v2.yaml arch.num_registers=16 \
        resume=saved_models/experiments/run_54/model.pt \
        -- path/to/image_folder/

    # TorchVision ViT (run_39)
    python scripts/postprocessing/demo_explainability.py \
        configs/config.yaml configs/baseline.yaml configs/torchvision_vit.yaml \
        resume=saved_models/experiments/run_39/model.pt \
        -- path/to/image.png

    # TrOCR (run_40)  — no register tokens, attention only
    python scripts/postprocessing/demo_explainability.py \
        configs/config.yaml configs/baseline.yaml configs/trocr.yaml \
        resume=saved_models/experiments/run_40/model.pt \
        -- path/to/image.png
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tqdm
from models import HTRNet
from utils.preprocessing import load_image, preprocess

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


# ── Attention extraction via hooks ───────────────────────────────────────────

class AttentionExtractor:
    """Register hooks on transformer self_attn layers to capture attention weights
    during the regular (correct) forward pass."""

    def __init__(self, model, arch_type):
        self.attn_maps = []
        self.hooks = []
        self._register_hooks(model, arch_type)

    def _register_hooks(self, model, arch_type):
        if arch_type == 'vit_rgts':
            encoder = model.backbone.encoder
            for layer in encoder.layers:
                self.hooks.append(
                    layer.self_attn.register_forward_hook(self._make_hook())
                )
        elif arch_type == 'torchvision_vit':
            encoder = model.backbone.vit.encoder
            for layer in encoder.layers:
                self.hooks.append(
                    layer.self_attention.register_forward_hook(self._make_hook())
                )
        elif arch_type == 'trocr':
            encoder = model.backbone.vit.encoder
            for layer in encoder.layer:
                self.hooks.append(
                    layer.attention.attention.register_forward_hook(self._make_hook())
                )

    def _make_hook(self):
        extractor = self
        def hook_fn(module, input, output):
            # Manually compute attention weights from Q, K
            # input[0] is the input tensor [B, S, D] (or [S, B, D] for nn.MultiheadAttention)
            x = input[0]

            if hasattr(module, 'in_proj_weight'):
                # nn.MultiheadAttention (vit_rgts)
                embed_dim = module.embed_dim
                num_heads = module.num_heads
                head_dim = embed_dim // num_heads

                # x may be [S, B, D] for batch_first=False
                if x.dim() == 3 and x.shape[0] != x.shape[1]:
                    if not getattr(module, 'batch_first', False):
                        x = x.transpose(0, 1)  # -> [B, S, D]

                qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
                q, k, _ = qkv.chunk(3, dim=-1)

                B, S, _ = q.shape
                q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
                k = k.view(B, S, num_heads, head_dim).transpose(1, 2)

                attn_w = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_w = torch.softmax(attn_w, dim=-1)  # [B, H, S, S]
                extractor.attn_maps.append(attn_w.detach().cpu())

            elif hasattr(module, 'qkv'):
                # timm / torchvision style (single qkv projection)
                x_in = input[0]
                if x_in.dim() == 3:
                    B, S, D = x_in.shape
                    num_heads = module.num_heads
                    head_dim = D // num_heads
                    qkv = module.qkv(x_in).reshape(B, S, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv.unbind(0)
                    attn_w = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
                    attn_w = torch.softmax(attn_w, dim=-1)
                    extractor.attn_maps.append(attn_w.detach().cpu())

            elif hasattr(module, 'query') and hasattr(module, 'key'):
                # HuggingFace style (separate query/key projections)
                x_in = input[0]
                B, S, D = x_in.shape
                num_heads = module.num_attention_heads
                head_dim = module.attention_head_size
                q = module.query(x_in).view(B, S, num_heads, head_dim).transpose(1, 2)
                k = module.key(x_in).view(B, S, num_heads, head_dim).transpose(1, 2)
                attn_w = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_w = torch.softmax(attn_w, dim=-1)
                extractor.attn_maps.append(attn_w.detach().cpu())

        return hook_fn

    def clear(self):
        self.attn_maps = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(config, device):
    """Load trained HTR model and character classes."""
    classes = np.load(os.path.join(config.data.path, 'classes.npy'), allow_pickle=True)
    nclasses = len(classes) + 1  # +1 for CTC blank

    print(f'Architecture : {config.arch.type}')
    print(f'Classes      : {nclasses}')

    model = HTRNet(config.arch, nclasses)

    if config.resume:
        print(f'Checkpoint   : {config.resume}')
        ckpt = torch.load(config.resume, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        print(model.load_state_dict(ckpt, strict=True))

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters   : {n_params:,}')
    return model, classes


def load_and_preprocess(image_path, config):
    """Load an image and preprocess to model input tensor [1,1,H,W]."""
    img = load_image(image_path)
    img = preprocess(img, (config.preproc.image_height, config.preproc.image_width))
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    return tensor, img


# ── CTC decode ───────────────────────────────────────────────────────────────

def decode_prediction(logits, classes):
    """Greedy CTC decode: collapse repeats, strip blank (index 0)."""
    if isinstance(logits, tuple):
        logits = logits[0]
    pred = logits.squeeze(1).argmax(dim=-1).cpu().numpy()  # [T]
    decoded = []
    prev = None
    for idx in pred:
        if idx != 0 and idx != prev and idx <= len(classes):
            decoded.append(classes[idx - 1])
        prev = idx
    return ''.join(decoded).strip()


# ── Visualization helpers ────────────────────────────────────────────────────

def visualize_attention_maps(attn_maps, raw_img, image_name, save_dir):
    """Save multi-head attention heatmap for middle layer."""
    num_layers = len(attn_maps)
    mid = num_layers // 2
    attn = attn_maps[mid][0].cpu().numpy()  # [H, S, S]
    num_heads = attn.shape[0]
    heads_to_show = min(5, num_heads)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image in top-left
    axes[0, 0].imshow(raw_img, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    for idx in range(heads_to_show):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        head_attn = attn[idx]  # [S, S]
        attn_from_first = head_attn[0, :]
        im = axes[row, col].imshow(attn_from_first.reshape(-1, 1),
                                   cmap='hot', aspect='auto')
        axes[row, col].set_title(f'Layer {mid}, Head {idx}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    plt.suptitle(f'Attention Maps — {image_name}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{image_name}_attention_maps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def visualize_register_tokens(reg_tokens, image_name, save_dir):
    """Save register-token embedding heatmap + L2 norms."""
    if reg_tokens is None:
        return None
    regs = reg_tokens[0].cpu().numpy()  # [R, D]
    num_regs, dim = regs.shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im0 = axes[0].imshow(regs, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Register Embeddings ({num_regs} x {dim})')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Register Index')
    plt.colorbar(im0, ax=axes[0])

    norms = np.linalg.norm(regs, axis=1)
    axes[1].bar(range(num_regs), norms, color='#1f77b4')
    axes[1].set_title('Register Token L2 Norms')
    axes[1].set_xlabel('Register Index')
    axes[1].set_ylabel('L2 Norm')

    plt.suptitle(f'Register Tokens — {image_name}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{image_name}_register_tokens.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def visualize_token_norms(token_norms, num_registers, image_name, save_dir):
    """Save per-token L2 norm bar chart (register tokens in red, patch in blue)."""
    norms = token_norms[0].cpu().numpy()  # [S]
    colors = ['red' if i < num_registers else '#1f77b4'
              for i in range(len(norms))]

    plt.figure(figsize=(14, 4))
    plt.bar(range(len(norms)), norms, color=colors, alpha=0.7)
    if num_registers > 0:
        plt.axvline(x=num_registers - 0.5, color='black', linestyle='--',
                    label=f'Register / Patch boundary ({num_registers} registers)')
        plt.legend()
    plt.title(f'Token L2 Norms — {image_name}  (Red=Register, Blue=Patch)')
    plt.xlabel('Token Index')
    plt.ylabel('L2 Norm')
    plt.tight_layout()
    path = os.path.join(save_dir, f'{image_name}_token_norms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── Core processing ──────────────────────────────────────────────────────────

def process_image(model, classes, config, img_path, output_dir, device, attn_extractor):
    """Run explainability analysis on one image. Returns (filename, prediction)."""
    image_name = Path(img_path).stem

    # Load & preprocess
    tensor, raw_img = load_and_preprocess(img_path, config)
    tensor = tensor.to(device)

    # Save preprocessed input
    plt.imsave(os.path.join(output_dir, f'{image_name}_input.png'),
               raw_img, cmap='gray')

    # Regular forward (correct logits) — hooks capture attention
    attn_extractor.clear()
    with torch.no_grad():
        output = model(tensor)
        # In eval mode, head_type='both' returns single tensor
        logits = output[0] if isinstance(output, tuple) else output

    prediction = decode_prediction(logits, classes)
    attn_maps = attn_extractor.attn_maps  # List[L] of [B, H, S, S]

    # Extract register tokens and token norms from backbone
    arch_type = config.arch.type
    reg_tokens = None
    token_norms = None

    if arch_type == 'vit_rgts':
        # Re-run backbone forward to get register tokens & encoded state
        with torch.no_grad():
            seq_tokens, reg_tokens, grid = model.backbone(tensor)
            # Reconstruct full encoded for norms: [B, S, D]
            encoded = torch.cat([reg_tokens,
                                 seq_tokens.permute(1, 0, 2)], dim=1)
            token_norms = encoded.norm(dim=-1).cpu()  # [B, S]
    elif arch_type == 'torchvision_vit':
        with torch.no_grad():
            seq_tokens, cls_token, reg_tokens, grid = model.backbone(tensor)
            encoded = torch.cat([reg_tokens,
                                 seq_tokens.permute(1, 0, 2)], dim=1)
            token_norms = encoded.norm(dim=-1).cpu()
    elif arch_type == 'trocr':
        with torch.no_grad():
            seq_tokens, cls_token, grid = model.backbone(tensor)
            token_norms = seq_tokens.permute(1, 0, 2).norm(dim=-1).cpu()

    # Determine number of register tokens
    num_regs = reg_tokens.shape[1] if reg_tokens is not None else 0

    # ── Visualizations ──
    if attn_maps:
        visualize_attention_maps(attn_maps, raw_img, image_name, output_dir)
    visualize_register_tokens(reg_tokens, image_name, output_dir)
    if token_norms is not None:
        visualize_token_norms(token_norms, num_regs, image_name, output_dir)

    # ── Save raw arrays ──
    npy_attn = os.path.join(output_dir, 'attention_maps')
    npy_regs = os.path.join(output_dir, 'register_tokens')
    npy_norms = os.path.join(output_dir, 'token_norms')

    if attn_maps:
        os.makedirs(npy_attn, exist_ok=True)
        np.save(os.path.join(npy_attn, f'{image_name}.npy'),
                [a.cpu().numpy() for a in attn_maps], allow_pickle=True)

    if token_norms is not None:
        os.makedirs(npy_norms, exist_ok=True)
        np.save(os.path.join(npy_norms, f'{image_name}.npy'),
                token_norms.cpu().numpy())

    if reg_tokens is not None:
        os.makedirs(npy_regs, exist_ok=True)
        np.save(os.path.join(npy_regs, f'{image_name}.npy'),
                reg_tokens.cpu().numpy())

    return image_name, prediction


def collect_image_paths(paths):
    """Resolve paths into individual image files (files and/or directories)."""
    image_paths = []
    for p in paths:
        if os.path.isdir(p):
            for fname in sorted(os.listdir(p)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(p, fname))
        elif os.path.isfile(p):
            image_paths.append(p)
        else:
            print(f'Warning: skipping {p} (not a file or directory)')
    return image_paths


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    """Parse config YAML(s), key=value overrides, and image paths separated by --."""
    argv = sys.argv[1:]
    if '--' in argv:
        sep = argv.index('--')
        config_args = argv[:sep]
        image_args = argv[sep + 1:]
    else:
        config_args = argv[:-1]
        image_args = [argv[-1]] if argv else []

    yaml_files = [a for a in config_args if a.endswith('.yaml')]
    overrides  = [a for a in config_args if not a.endswith('.yaml')]

    if not yaml_files:
        print('Error: at least one .yaml config file is required.')
        sys.exit(1)

    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))

    OmegaConf.set_struct(conf, False)
    cli_conf = OmegaConf.from_dotlist(overrides)
    conf = OmegaConf.merge(conf, cli_conf)

    return conf, image_args


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config, image_args = parse_args()

    if not image_args:
        print('Error: no image path(s) provided. Use -- followed by image paths.')
        sys.exit(1)

    image_paths = collect_image_paths(image_args)
    if not image_paths:
        print('Error: no valid image files found.')
        sys.exit(1)

    # Resolve device
    device = config.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA not available, falling back to CPU.')
        device = 'cpu'
        config.device = 'cpu'

    # Output dir: <run_folder>/explainability/
    if hasattr(config, 'resume') and config.resume:
        run_dir = os.path.dirname(os.path.abspath(config.resume))
    else:
        run_dir = 'output'
    output_dir = os.path.join(run_dir, 'explainability')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Explainable HTR Demo')
    print('=' * 70)
    print(f'Images       : {len(image_paths)}')
    print(f'Output dir   : {output_dir}')
    print()

    # Validate architecture supports explainability
    arch_type = config.arch.type
    if arch_type not in ('vit_rgts', 'torchvision_vit', 'trocr'):
        print(f'Error: forward_explain not supported for architecture "{arch_type}".')
        print('Supported: vit_rgts, torchvision_vit, trocr')
        sys.exit(1)

    model, classes = load_model(config, device)

    # Set up attention hooks
    attn_extractor = AttentionExtractor(model, arch_type)

    # Process each image
    results = []
    for img_path in tqdm.tqdm(image_paths, desc='Processing'):
        try:
            name, pred = process_image(model, classes, config, img_path,
                                       output_dir, device, attn_extractor)
            results.append((name, pred))
            tqdm.tqdm.write(f'  {name:>40s}  ->  {pred}')
        except Exception as e:
            name = Path(img_path).stem
            results.append((name, f'[ERROR: {e}]'))
            tqdm.tqdm.write(f'  {name:>40s}  ->  [ERROR: {e}]')

    # Clean up hooks
    attn_extractor.remove_hooks()

    # Save predictions.txt
    pred_path = os.path.join(output_dir, 'predictions.txt')
    with open(pred_path, 'w', encoding='utf-8') as f:
        f.write(f'# Explainability Predictions  —  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Model: {config.resume}\n')
        f.write(f'# Architecture: {arch_type}\n')
        f.write(f'# Images: {len(results)}\n')
        f.write('#' + '-' * 59 + '\n')
        for name, pred in results:
            f.write(f'{name}\t{pred}\n')

    print()
    print('=' * 70)
    print(f'Done — {len(results)} images processed')
    print(f'Predictions  : {pred_path}')
    print(f'Output dir   : {output_dir}')
    print('=' * 70)
