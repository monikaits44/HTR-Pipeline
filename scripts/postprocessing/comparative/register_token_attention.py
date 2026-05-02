#!/usr/bin/env python3
"""
Register & CLS Token Attention Maps for ViT-RGTS

Visualises how register tokens (the CLS-like special tokens in ViT-RGTS)
interact with input patch tokens across models with varying register counts.

ViT-RGTS has NO explicit CLS token.  Register tokens (Reg-Tok 0 … Reg-Tok N-1)
are prepended to the patch sequence and serve a similar role: they absorb
global / background attention (the "attention sink" phenomenon) so that patch
tokens retain clean, spatially-localised features for CTC decoding.

Generates TWO figures per image per layer:
  1. reg_to_patch/  — Register → Patch attention
     "What does each register token look at in the image?"
     Row r of the attention matrix for register token r.

  2. patch_to_reg/  — Patch → Register attention (attention sink)
     "Which patches dump the most attention INTO register tokens?"
     Column r of the attention matrix for register token r.
     For Reg-0 (no registers): shows patch self-sink — which patches
     absorb the most attention from other patches (the known ViT
     attention-collapse artifact that registers are designed to fix).

Layout per figure:
  Rows  = models (sorted by register count: Reg-0 → Reg-16)
  Cols  = [Original | token₀ | token₁ | … | tokenN | (Aggregate)]

Usage:
    # Single image, last layer
    python scripts/postprocessing/register_token_attention.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_51/model.pt \\
        --model-path saved_models/experiments/run_52/model.pt \\
        --model-path saved_models/experiments/run_53/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --layer last \\
        -- notebook/sample_images/a01-038-12.png

    # Multiple layers
    python scripts/postprocessing/register_token_attention.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --layer first,middle,last \\
        -- notebook/sample_images/

Arguments:
    config.yaml        Base config file(s), merged in order
    key=value          Override params (e.g., device=cuda)
    --model-path PATH  Checkpoint — repeat per model (min 1)
    --                 Separator; image paths / directory follow
    --save-dir DIR     Output directory (default: visualizations/register_token_attention/)
    --alpha FLOAT      Heatmap overlay opacity (default: 0.50)
    --dpi INT          Output resolution (default: 150)
    --max-regs INT     Max register tokens to display (default: 16)
    --layer LAYER      first|middle|last|all, comma-separated (default: last)
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

# ── Project root on sys.path ─────────────────────────────────────────────────
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
    """Parse CLI: configs + flags before '--', image paths after '--'."""
    argv = sys.argv[1:]

    has_sep = '--' in argv
    if has_sep:
        sep = argv.index('--')
        config_args = argv[:sep]
        image_args = argv[sep + 1:]
    else:
        config_args = argv
        image_args = []

    model_paths = []
    save_dir = ''
    alpha = 0.50
    dpi = 150
    max_regs = 16
    layer = 'last'

    filtered = []
    i = 0
    while i < len(config_args):
        if config_args[i] == '--model-path' and i + 1 < len(config_args):
            model_paths.append(config_args[i + 1]); i += 2
        elif config_args[i] == '--save-dir' and i + 1 < len(config_args):
            save_dir = config_args[i + 1]; i += 2
        elif config_args[i] == '--alpha' and i + 1 < len(config_args):
            alpha = float(config_args[i + 1]); i += 2
        elif config_args[i] == '--dpi' and i + 1 < len(config_args):
            dpi = int(config_args[i + 1]); i += 2
        elif config_args[i] == '--max-regs' and i + 1 < len(config_args):
            max_regs = int(config_args[i + 1]); i += 2
        elif config_args[i] == '--layer' and i + 1 < len(config_args):
            layer = config_args[i + 1]; i += 2
        else:
            filtered.append(config_args[i]); i += 1

    yaml_files = [a for a in filtered if a.endswith('.yaml')]
    overrides = [a for a in filtered if not a.endswith('.yaml')]

    if not yaml_files:
        print('Error: at least one .yaml config file is required.'); sys.exit(1)
    if not model_paths:
        print('Error: at least one --model-path is required.'); sys.exit(1)

    valid_layers = {'first', 'middle', 'last', 'all'}
    layers = tuple(l.strip() for l in layer.split(','))
    for l in layers:
        if l not in valid_layers:
            print(f'Error: --layer must be one of {valid_layers} (got: {l})')
            sys.exit(1)

    # Merge YAMLs
    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))
    OmegaConf.set_struct(conf, False)
    cli_conf = OmegaConf.from_dotlist(overrides)
    conf = OmegaConf.merge(conf, cli_conf)

    # Resolve images
    image_paths = []
    for p in image_args:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(p, f))
        elif os.path.isfile(p):
            image_paths.append(p)
        else:
            print(f'Warning: skipping {p} (not found)')

    if has_sep and not image_paths:
        print('Error: "--" used but no valid images found.'); sys.exit(1)
    if not image_paths:
        print('Error: at least one image is required (after --).'); sys.exit(1)

    return conf, model_paths, image_paths, save_dir, alpha, dpi, max_regs, layers


# ═════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════════════════════

def get_run_info(model_path):
    """Read config.json to discover num_registers and run name."""
    run_dir = os.path.dirname(os.path.abspath(model_path))
    run_name = os.path.basename(run_dir)
    config_json = os.path.join(run_dir, 'config.json')
    if not os.path.isfile(config_json):
        raise FileNotFoundError(
            f'config.json not found in {run_dir}. '
            'Each model directory must contain config.json.')
    with open(config_json) as f:
        cfg = json.load(f)
    num_reg = cfg.get('arch', {}).get('num_registers', 0)
    return {
        'run_dir': run_dir,
        'run_name': run_name,
        'num_registers': num_reg,
        'label': f'Reg-{num_reg}',
    }


def build_model(base_config, model_path, num_classes, num_registers, device):
    """Build HTRNet with correct num_registers, load checkpoint."""
    config = deepcopy(base_config)
    OmegaConf.set_struct(config, False)
    config.arch.num_registers = num_registers
    config.resume = model_path
    net = HTRNet(config.arch, num_classes)
    ckpt = torch.load(model_path, map_location=device)
    load_info = net.load_state_dict(ckpt, strict=True)
    net.to(device).eval()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return net, n_params, load_info


# ═════════════════════════════════════════════════════════════════════════════
# Attention Extraction
# ═════════════════════════════════════════════════════════════════════════════

def select_layer_attn(attn_maps_list, layer='last'):
    """Select one layer's attention and average across heads → [S, S]."""
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
    # [B, H, S, S] → [S, S]
    if attn.dim() == 4:
        attn = attn[0].mean(dim=0)
    elif attn.dim() == 3:
        attn = attn[0]
    return attn.cpu().numpy()


def select_layer_per_head(attn_maps_list, layer='last'):
    """Select one layer's attention, return per-head: [H, S, S]."""
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
    # [B, H, S, S] → [H, S, S]  (batch=0)
    if attn.dim() == 4:
        return attn[0].cpu().numpy()      # [H, S, S]
    elif attn.dim() == 3:
        return attn.unsqueeze(0).cpu().numpy()
    return attn.cpu().numpy()


def extract_attention(net, img_tensor, device, layers):
    """Run forward_explain() once, return per-layer head-averaged + per-head attention."""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits, reg_tokens, attn_maps_list, token_norms, (Hp, Wp) = \
            net.forward_explain(img_tensor)
    num_reg = reg_tokens.shape[1] if reg_tokens is not None else 0
    result = {'grid_size': (Hp, Wp), 'num_reg': num_reg}
    for layer in layers:
        result[f'attn_{layer}'] = select_layer_attn(attn_maps_list, layer)
        result[f'attn_{layer}_perhead'] = select_layer_per_head(attn_maps_list, layer)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Patch-aligned Upscaling
# ═════════════════════════════════════════════════════════════════════════════

def upscale_patch_aligned(patch_map, H_img, W_img, fixed_size=None,
                          border_size=8):
    """Map patch-level heatmap to original image coordinates."""
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
    heat_crop = heat_full[r0:r0 + H_img, c0:c0 + W_img]
    if heat_crop.shape[0] < H_img or heat_crop.shape[1] < W_img:
        padded = np.zeros((H_img, W_img), dtype=heat_crop.dtype)
        padded[:heat_crop.shape[0], :heat_crop.shape[1]] = heat_crop
        heat_crop = padded
    return heat_crop


def normalise_heatmap(vec):
    """Percentile-clip normalise a 1-D vector to [0, 1]."""
    p_low = np.percentile(vec, 1)
    p_high = np.percentile(vec, 99)
    if p_high - p_low < 1e-8:
        p_low, p_high = vec.min(), vec.max()
    if p_high - p_low < 1e-8:
        return np.zeros_like(vec)
    return np.clip((vec - p_low) / (p_high - p_low), 0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# Visualization — Register → Patch Attention
# ═════════════════════════════════════════════════════════════════════════════

def visualize_reg_to_patch(img_path, model_infos, all_results, save_path,
                           alpha, dpi, max_regs, layer_name,
                           fixed_size, border_size=8):
    """
    What each register token looks at in the image.

    For register r: A[r, R:R+T] = how much register r attends to each patch.
    For Reg-0 models (no registers): shows "No registers" placeholder.
    """
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape
    image_name = Path(img_path).stem

    n_models = len(all_results)
    max_R = min(max_regs, max((r['num_reg'] for r in all_results), default=0))
    max_R = max(max_R, 1)  # at least 1 column for Reg-0 placeholder

    n_cols = 1 + max_R   # Original + register columns

    col_w = max(2.5, min(W_img / 80, 5.0))
    row_h = max(1.8, min(H_img / 30, 4.0))
    fig_w = min(col_w * n_cols + 2.5, 80)
    fig_h = row_h * n_models + 1.5

    fig, axes = plt.subplots(n_models, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    for row_idx, (info, result) in enumerate(zip(model_infos, all_results)):
        # Col 0: Original image + label
        ax = axes[row_idx, 0]
        ax.imshow(img_np, cmap="gray")
        if row_idx == 0:
            ax.set_title("Original", fontsize=9, fontweight='bold')
        ax.axis("off")

        label = f'{info["label"]}\n({info["run_name"]})'
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va='top', ha='left',
                fontsize=7, color='white', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          alpha=0.75, edgecolor='none'))

        attn = result.get(f'attn_{layer_name}')
        num_reg = result['num_reg']
        Hp, Wp = result['grid_size']
        T = Hp * Wp

        for ri in range(max_R):
            ax = axes[row_idx, 1 + ri]
            if ri < num_reg and attn is not None:
                # Register ri → all patch tokens
                reg_attn = attn[ri, num_reg:num_reg + T]
                reg_norm = normalise_heatmap(reg_attn)

                grid = reg_norm.reshape(Hp, Wp)
                heat_up = upscale_patch_aligned(grid, H_img, W_img,
                                                fixed_size, border_size)

                ax.imshow(img_np, cmap="gray")
                im = ax.imshow(heat_up, cmap="inferno", alpha=alpha,
                               vmin=0.0, vmax=1.0)

                # Stats overlay
                mean_a = float(attn[ri, num_reg:num_reg + T].mean())
                max_a = float(attn[ri, num_reg:num_reg + T].max())
                # How much of this register's total attention goes to patches
                # vs. other registers
                total_to_patch = float(attn[ri, num_reg:].sum())
                total_to_reg = float(attn[ri, :num_reg].sum())
                ax.text(0.02, 0.02,
                        f'μ={mean_a:.4f}  max={max_a:.3f}\n'
                        f'→patch={total_to_patch:.2f}  →reg={total_to_reg:.2f}',
                        transform=ax.transAxes, va='bottom', ha='left',
                        fontsize=5, color='white', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                  alpha=0.6, edgecolor='none'))

                # Colorbar on last visible register column
                if ri == min(num_reg, max_R) - 1:
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=5)
                    cbar.set_label('attn weight', fontsize=5)
            else:
                if num_reg == 0:
                    ax.text(0.5, 0.5, 'No\nregisters',
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=9, color='gray', style='italic')
                else:
                    ax.set_visible(False)

            if row_idx == 0:
                lbl = (f'Reg-Tok 0\n(CLS-like)' if ri == 0
                       else f'Reg-Tok {ri}')
                ax.set_title(lbl, fontsize=8, fontweight='bold')
            ax.axis("off")

    title = (f'Register → Patch Attention: {image_name}\n'
             f'{layer_name}-layer, head-averaged  |  '
             f'Models: {n_models}')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Visualization — Patch → Register Attention (Sink View)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_patch_to_reg(img_path, model_infos, all_results, save_path,
                           alpha, dpi, max_regs, layer_name,
                           fixed_size, border_size=8):
    """
    Attention sink view: which patches dump the most attention INTO registers.

    For register r: A[R:R+T, r] = how much each patch sends to register r.
    Aggregate: mean across all registers → total sink per patch.
    For Reg-0: shows "patch self-sink" — which patches absorb the most
    attention from OTHER patches (demonstrates the ViT attention-collapse
    problem that registers are designed to fix).
    """
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape
    image_name = Path(img_path).stem

    n_models = len(all_results)
    max_R = min(max_regs, max((r['num_reg'] for r in all_results), default=0))
    max_R = max(max_R, 1)

    # Cols: Original | Aggregate | Per-register sinks
    n_cols = 1 + 1 + max_R

    col_w = max(2.5, min(W_img / 80, 5.0))
    row_h = max(1.8, min(H_img / 30, 4.0))
    fig_w = min(col_w * n_cols + 2.5, 80)
    fig_h = row_h * n_models + 1.5

    fig, axes = plt.subplots(n_models, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    for row_idx, (info, result) in enumerate(zip(model_infos, all_results)):
        # Col 0: Original
        ax = axes[row_idx, 0]
        ax.imshow(img_np, cmap="gray")
        if row_idx == 0:
            ax.set_title("Original", fontsize=9, fontweight='bold')
        ax.axis("off")

        label = f'{info["label"]}\n({info["run_name"]})'
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va='top', ha='left',
                fontsize=7, color='white', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          alpha=0.75, edgecolor='none'))

        attn = result.get(f'attn_{layer_name}')
        num_reg = result['num_reg']
        Hp, Wp = result['grid_size']
        T = Hp * Wp

        # ── Col 1: Aggregate sink ────────────────────────────────────
        ax = axes[row_idx, 1]
        if num_reg > 0 and attn is not None:
            # Mean attention FROM patches TO all registers
            sink_agg = attn[num_reg:num_reg + T, :num_reg].mean(axis=1)  # [T]
            sink_norm = normalise_heatmap(sink_agg)

            grid = sink_norm.reshape(Hp, Wp)
            heat_up = upscale_patch_aligned(grid, H_img, W_img,
                                            fixed_size, border_size)

            ax.imshow(img_np, cmap="gray")
            im = ax.imshow(heat_up, cmap="inferno", alpha=alpha,
                           vmin=0.0, vmax=1.0)

            total_sink = float(attn[num_reg:num_reg + T, :num_reg].sum() / T)
            ax.text(0.02, 0.02, f'avg sink/patch={total_sink:.3f}',
                    transform=ax.transAxes, va='bottom', ha='left',
                    fontsize=5, color='white', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                              alpha=0.6, edgecolor='none'))

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label('sink mag.', fontsize=5)

        elif num_reg == 0 and attn is not None:
            # Reg-0: show patch SELF-SINK — which patches absorb the most
            # attention from OTHER patches (the known ViT attention-collapse
            # artifact).  Column sum with diagonal zeroed.
            patch_attn = attn.copy()
            np.fill_diagonal(patch_attn, 0.0)
            self_sink = patch_attn.sum(axis=0)   # [T] — absorption per patch
            sink_norm = normalise_heatmap(self_sink)

            grid = sink_norm.reshape(Hp, Wp)
            heat_up = upscale_patch_aligned(grid, H_img, W_img,
                                            fixed_size, border_size)

            ax.imshow(img_np, cmap="gray")
            im = ax.imshow(heat_up, cmap="magma", alpha=alpha,
                           vmin=0.0, vmax=1.0)

            top_sink_frac = float(self_sink.max() / (self_sink.sum() + 1e-8))
            ax.text(0.02, 0.02,
                    f'Patch self-sink\ntop patch={top_sink_frac:.1%} of total',
                    transform=ax.transAxes, va='bottom', ha='left',
                    fontsize=5, color='white', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                              alpha=0.6, edgecolor='none'))
        else:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')

        if row_idx == 0:
            ax.set_title("Agg. Sink\n(Patch→Regs)", fontsize=8,
                         fontweight='bold')
        ax.axis("off")

        # ── Cols 2+: Per-register sink ───────────────────────────────
        for ri in range(max_R):
            ax = axes[row_idx, 2 + ri]
            if ri < num_reg and attn is not None:
                sink_ri = attn[num_reg:num_reg + T, ri]   # [T]
                sink_norm = normalise_heatmap(sink_ri)

                grid = sink_norm.reshape(Hp, Wp)
                heat_up = upscale_patch_aligned(grid, H_img, W_img,
                                                fixed_size, border_size)

                ax.imshow(img_np, cmap="gray")
                im = ax.imshow(heat_up, cmap="inferno", alpha=alpha,
                               vmin=0.0, vmax=1.0)

                mag = float(sink_ri.mean())
                ax.text(0.02, 0.02, f'μ={mag:.4f}',
                        transform=ax.transAxes, va='bottom', ha='left',
                        fontsize=5, color='white', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                  alpha=0.6, edgecolor='none'))
            else:
                if num_reg == 0:
                    ax.text(0.5, 0.5, 'No regs', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8,
                            color='gray', style='italic')
                else:
                    ax.set_visible(False)

            if row_idx == 0:
                ax.set_title(f'Sink→Reg-{ri}', fontsize=8, fontweight='bold')
            ax.axis("off")

    title = (f'Patch → Register Attention (Sink): {image_name}\n'
             f'{layer_name}-layer, head-averaged  |  '
             f'Models: {n_models}')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Visualization — Per-Head Register Attention (detail view)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_per_head_reg(img_path, model_infos, all_results, save_path,
                           alpha, dpi, layer_name, reg_idx,
                           fixed_size, border_size=8):
    """
    Per-head attention for a single register token across models.

    Rows = models, Cols = [Original | Head 0 | Head 1 | … | Head H-1]
    Shows how different attention heads specialise for a given register.
    """
    img_np = np.array(Image.open(img_path).convert("L"))
    H_img, W_img = img_np.shape
    image_name = Path(img_path).stem

    n_models = len(all_results)

    # Determine number of heads from first model that has per-head data
    n_heads = 0
    for r in all_results:
        ph = r.get(f'attn_{layer_name}_perhead')
        if ph is not None:
            n_heads = ph.shape[0]
            break
    if n_heads == 0:
        return None

    n_cols = 1 + n_heads   # Original + per-head

    col_w = max(2.0, min(W_img / 90, 4.0))
    row_h = max(1.5, min(H_img / 35, 3.5))
    fig_w = min(col_w * n_cols + 2.0, 80)
    fig_h = row_h * n_models + 1.5

    fig, axes = plt.subplots(n_models, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    for row_idx, (info, result) in enumerate(zip(model_infos, all_results)):
        ax = axes[row_idx, 0]
        ax.imshow(img_np, cmap="gray")
        if row_idx == 0:
            ax.set_title("Original", fontsize=8, fontweight='bold')
        ax.axis("off")

        label = f'{info["label"]}\n({info["run_name"]})'
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va='top', ha='left',
                fontsize=6, color='white', family='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          alpha=0.75, edgecolor='none'))

        attn_ph = result.get(f'attn_{layer_name}_perhead')  # [H, S, S]
        num_reg = result['num_reg']
        Hp, Wp = result['grid_size']
        T = Hp * Wp

        for hi in range(n_heads):
            ax = axes[row_idx, 1 + hi]
            if reg_idx < num_reg and attn_ph is not None:
                head_attn = attn_ph[hi, reg_idx, num_reg:num_reg + T]  # [T]
                head_norm = normalise_heatmap(head_attn)

                grid = head_norm.reshape(Hp, Wp)
                heat_up = upscale_patch_aligned(grid, H_img, W_img,
                                                fixed_size, border_size)

                ax.imshow(img_np, cmap="gray")
                ax.imshow(heat_up, cmap="inferno", alpha=alpha,
                          vmin=0.0, vmax=1.0)
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='gray')

            if row_idx == 0:
                ax.set_title(f'Head {hi}', fontsize=7, fontweight='bold')
            ax.axis("off")

    reg_lbl = f'Reg-Tok {reg_idx}' + (' (CLS-like)' if reg_idx == 0 else '')
    title = (f'Per-Head Register Attention: {image_name}\n'
             f'{reg_lbl}, {layer_name}-layer  |  Models: {n_models}')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    (conf, model_paths, image_paths, save_dir,
     alpha, dpi, max_regs, layers) = parse_args()

    # ── Device ───────────────────────────────────────────────────────
    device = getattr(conf, 'device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA not available, falling back to CPU.')
        device = 'cpu'

    # ── Character classes (needed for model construction) ────────────
    dataset_folder = conf.data.path
    classes = np.load(os.path.join(dataset_folder, 'classes.npy'))
    num_classes = len(classes) + 1

    fixed_size = (conf.preproc.image_height, conf.preproc.image_width)

    # ── Output directories ───────────────────────────────────────────
    out_dir = (save_dir if save_dir
               else os.path.join('visualizations', 'register_token_attention'))
    reg2patch_dir = os.path.join(out_dir, 'reg_to_patch')
    patch2reg_dir = os.path.join(out_dir, 'patch_to_reg')
    perhead_dir = os.path.join(out_dir, 'per_head')

    import shutil
    for sub in (reg2patch_dir, patch2reg_dir, perhead_dir):
        if os.path.isdir(sub):
            shutil.rmtree(sub)
    os.makedirs(reg2patch_dir, exist_ok=True)
    os.makedirs(patch2reg_dir, exist_ok=True)
    os.makedirs(perhead_dir, exist_ok=True)

    # ── Run info ─────────────────────────────────────────────────────
    model_infos = [get_run_info(mp) for mp in model_paths]
    sorted_idx = sorted(range(len(model_infos)),
                        key=lambda i: model_infos[i]['num_registers'])
    model_infos = [model_infos[i] for i in sorted_idx]
    model_paths = [model_paths[i] for i in sorted_idx]

    # ── Header ───────────────────────────────────────────────────────
    print('=' * 70)
    print('Register & CLS Token Attention Visualization')
    print('=' * 70)
    print(f'Images       : {len(image_paths)}')
    print(f'Models       : {len(model_paths)}')
    for info in model_infos:
        print(f'  {info["label"]:>8s}  ({info["run_name"]})')
    print(f'Output       : {out_dir}')
    print(f'  └─ reg_to_patch/ : register → patch attention')
    print(f'  └─ patch_to_reg/ : patch → register attention (sink)')
    print(f'  └─ per_head/     : per-head detail for Reg-Tok 0')
    print(f'Layers       : {", ".join(layers)}')
    print(f'Max regs     : {max_regs}')
    print(f'Alpha        : {alpha}')
    print(f'DPI          : {dpi}')
    print(f'Timestamp    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    # ══════════════════════════════════════════════════════════════════
    # Process each model sequentially (memory-efficient)
    # ══════════════════════════════════════════════════════════════════
    all_data = []   # [model_idx][image_idx] = result dict

    for mi, (info, mp) in enumerate(zip(model_infos, model_paths)):
        print(f'[{mi + 1}/{len(model_paths)}] Loading {info["label"]} '
              f'({info["run_name"]}) …')
        net, n_params, load_info = build_model(
            conf, mp, num_classes, info['num_registers'], device)
        print(f'  Params: {n_params:,}  |  {load_info}')

        results = []
        for img_path in image_paths:
            image_name = Path(img_path).stem
            raw_img = load_image(img_path)
            processed = preprocess(raw_img, fixed_size)
            img_tensor = (torch.from_numpy(processed).float()
                          .unsqueeze(0).unsqueeze(0))

            r = extract_attention(net, img_tensor, device, layers)
            r['img_path'] = img_path
            r['image_name'] = image_name
            results.append(r)
            print(f'    {image_name:>30s}  registers={r["num_reg"]}  '
                  f'grid={r["grid_size"]}')

        all_data.append(results)
        del net
        gc.collect()
        print()

    # ══════════════════════════════════════════════════════════════════
    # Generate figures
    # ══════════════════════════════════════════════════════════════════
    print('Generating figures …')
    saved = 0

    for img_idx in range(len(image_paths)):
        img_results = [all_data[mi][img_idx]
                       for mi in range(len(model_infos))]
        img_path = img_results[0]['img_path']
        image_name = img_results[0]['image_name']

        for layer_name in layers:
            suffix = f'_{layer_name}' if len(layers) > 1 else ''

            # ── Register → Patch ─────────────────────────────────────
            sp = os.path.join(reg2patch_dir, f'{image_name}{suffix}.png')
            out = visualize_reg_to_patch(
                img_path, model_infos, img_results, sp,
                alpha, dpi, max_regs, layer_name, fixed_size)
            if out:
                saved += 1
                print(f'  ✓ reg_to_patch/{image_name}{suffix}.png')

            # ── Patch → Register (Sink) ──────────────────────────────
            sp = os.path.join(patch2reg_dir, f'{image_name}{suffix}.png')
            out = visualize_patch_to_reg(
                img_path, model_infos, img_results, sp,
                alpha, dpi, max_regs, layer_name, fixed_size)
            if out:
                saved += 1
                print(f'  ✓ patch_to_reg/{image_name}{suffix}.png')

            # ── Per-Head detail for Reg-Tok 0 (CLS-like) ────────────
            sp = os.path.join(perhead_dir,
                              f'{image_name}_reg0{suffix}.png')
            out = visualize_per_head_reg(
                img_path, model_infos, img_results, sp,
                alpha, dpi, layer_name, reg_idx=0, fixed_size=fixed_size)
            if out:
                saved += 1
                print(f'  ✓ per_head/{image_name}_reg0{suffix}.png')

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print()
    print(f'Done! {saved} figures saved.')
    print(f'  reg_to_patch/ : {reg2patch_dir}/')
    print(f'  patch_to_reg/ : {patch2reg_dir}/')
    print(f'  per_head/     : {perhead_dir}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
