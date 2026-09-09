#!/usr/bin/env python3
"""
Attention Quality Metrics: Diagonality, Peak Sharpness, Entropy

Computes three quantitative metrics per (model, image) pair and produces
two publication-ready figures:

  1. Diagonality vs Text Length  — scatter plot, one series per register count
     (reveals the ~15-char regime shift)
  2. Peak Sharpness vs Register Count — grouped bar/box plot by length bucket
     (quantifies why Reg-2 attention "looks best")

Metrics:
  • Diagonality (ρ)      Spearman rank correlation between character index
                          and peak attention patch position.  ρ=1 ⇒ perfect
                          left-to-right reading.
  • Peak Sharpness       max(attn) / mean(attn) for each character's raw
                          attention vector, averaged over all characters.
  • Entropy              −Σ p·log₂(p) over patch positions, averaged over
                          all characters.  Lower ⇒ more focused.

Usage:
    python scripts/postprocessing/attention_quality_metrics.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_51/model.pt \\
        --model-path saved_models/experiments/run_52/model.pt \\
        --model-path saved_models/experiments/run_53/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        --layer last \\
        --save-dir visualizations/attention_quality_metrics \\
        -- notebook/sample_images/

    # With images from dataset splits (use paths directly)
    python scripts/postprocessing/attention_quality_metrics.py \\
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \\
        --model-path saved_models/experiments/run_50/model.pt \\
        --model-path saved_models/experiments/run_51/model.pt \\
        --model-path saved_models/experiments/run_52/model.pt \\
        --model-path saved_models/experiments/run_53/model.pt \\
        --model-path saved_models/experiments/run_54/model.pt \\
        -- data/IAM/processed_lines/test/a01-000u-00.png \\
           data/IAM/processed_lines/test/c04-110-00.png \\
           notebook/sample_images/

Arguments:
    config.yaml        Base config file(s), merged in order
    --model-path P     Checkpoint – repeat per model (run_50 to run_54)
    --                 Separator; image paths / directories follow
    --layer LAYER      first|middle|last|all|rollout  (default: last)
    --save-dir DIR     Output directory
                       (default: visualizations/attention_quality_metrics/)
    --dpi INT          Figure resolution (default: 200)
"""

from __future__ import annotations

import os
import sys
import gc
import json
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── project root ──────────────────────────────────────────────────────────────
# This file is 3 levels below root: scripts/postprocessing/comparative/
THIS_DIR  = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import HTRNet
from utils.preprocessing import load_image, preprocess

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


# ═════════════════════════════════════════════════════════════════════════════
#  Argument parsing  (mirrors character_attention_mapping.py)
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    argv = sys.argv[1:]
    has_sep = '--' in argv
    if has_sep:
        sep = argv.index('--')
        config_args, image_args = argv[:sep], argv[sep + 1:]
    else:
        config_args, image_args = argv, []

    model_paths: list[str] = []
    save_dir = ''
    dpi  = 200
    layer = 'last'

    filtered: list[str] = []
    i = 0
    while i < len(config_args):
        a = config_args[i]
        if a == '--model-path' and i + 1 < len(config_args):
            model_paths.append(config_args[i + 1]); i += 2
        elif a == '--save-dir' and i + 1 < len(config_args):
            save_dir = config_args[i + 1]; i += 2
        elif a == '--dpi' and i + 1 < len(config_args):
            dpi = int(config_args[i + 1]); i += 2
        elif a == '--layer' and i + 1 < len(config_args):
            layer = config_args[i + 1]; i += 2
        else:
            filtered.append(a); i += 1

    yaml_files = [a for a in filtered if a.endswith('.yaml')]
    overrides  = [a for a in filtered if not a.endswith('.yaml')]

    if not yaml_files:
        print('Error: at least one .yaml config file required.'); sys.exit(1)
    if not model_paths:
        print('Error: at least one --model-path required.'); sys.exit(1)

    valid_layers = {'first', 'middle', 'last', 'all', 'rollout'}
    if layer not in valid_layers:
        print(f'Error: --layer must be one of {valid_layers}'); sys.exit(1)

    conf = OmegaConf.load(yaml_files[0])
    for yf in yaml_files[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(yf))
    OmegaConf.set_struct(conf, False)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))

    image_paths: list[str] = []
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

    return conf, model_paths, image_paths, save_dir, dpi, layer


# ═════════════════════════════════════════════════════════════════════════════
#  Model helpers  (reused from character_attention_mapping.py)
# ═════════════════════════════════════════════════════════════════════════════

def get_run_info(model_path: str) -> dict:
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
#  CTC decode
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


# ═════════════════════════════════════════════════════════════════════════════
#  Attention extraction
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
        'attn':      attn,
        'grid_size': (Hp, Wp),
        'logits':    logits.cpu().numpy(),
        'num_reg':   num_reg,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Ground truth loading
# ═════════════════════════════════════════════════════════════════════════════

def load_gt(image_paths):
    gt: dict[str, str] = {}
    seen: set[str] = set()
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
#  Per-character RAW attention vectors  (no gamma / normalisation)
# ═════════════════════════════════════════════════════════════════════════════

def get_raw_char_attention(attn_matrix, num_reg, positions, grid_size):
    """Return list of RAW (un-normalised) per-character attention vectors."""
    Hp, Wp = grid_size
    T = Hp * Wp
    vectors: list[np.ndarray] = []
    for t in positions:
        row_idx = num_reg + t
        if row_idx >= attn_matrix.shape[0]:
            vectors.append(np.ones(T) / T)
        else:
            vec = attn_matrix[row_idx, num_reg:num_reg + T].copy()
            if vec.shape[0] < T:
                vec = np.pad(vec, (0, T - vec.shape[0]))
            vectors.append(vec)
    return vectors


# ═════════════════════════════════════════════════════════════════════════════
#  Metric computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_diagonality(positions: list[int],
                        char_vectors: list[np.ndarray]) -> float:
    """Spearman ρ between character index and peak attention patch position."""
    n = len(positions)
    if n < 3:
        # With < 3 points Spearman is degenerate; return NaN
        return float('nan')
    peaks = [int(np.argmax(v)) for v in char_vectors]
    rho, _ = sp_stats.spearmanr(list(range(n)), peaks)
    return float(rho)


def compute_peak_sharpness(char_vectors: list[np.ndarray]) -> float:
    """Mean per-character  max(attn) / mean(attn)."""
    vals = []
    for v in char_vectors:
        m = v.mean()
        if m < 1e-12:
            vals.append(0.0)
        else:
            vals.append(float(v.max() / m))
    return float(np.mean(vals)) if vals else 0.0


def compute_entropy(char_vectors: list[np.ndarray]) -> float:
    """Mean per-character Shannon entropy  −Σ p·log₂(p)."""
    vals = []
    for v in char_vectors:
        p = v / (v.sum() + 1e-12)
        p = p[p > 0]
        vals.append(float(-np.sum(p * np.log2(p))))
    return float(np.mean(vals)) if vals else 0.0


def compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate via Levenshtein distance (pure-Python)."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    n, m = len(pred), len(gt)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m] / m


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

# colour palette: one colour per register count, colour-blind friendly
REG_COLORS = {
    0:  '#1b9e77',   # teal
    2:  '#d95f02',   # orange
    4:  '#7570b3',   # purple
    8:  '#e7298a',   # pink
    16: '#66a61e',   # green
}
REG_MARKERS = {0: 'o', 2: 's', 4: 'D', 8: '^', 16: 'v'}

LENGTH_BUCKETS = [
    ('Short\n(≤10)',    0, 10),
    ('Medium\n(11–30)', 11, 30),
    ('Long\n(31–60)',   31, 60),
    ('V.Long\n(61+)',   61, 9999),
]


def _bucket_label(text_len: int) -> str:
    for label, lo, hi in LENGTH_BUCKETS:
        if lo <= text_len <= hi:
            return label
    return LENGTH_BUCKETS[-1][0]


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Diagonality vs Text Length  (scatter + trend lines)
# ═════════════════════════════════════════════════════════════════════════════

def plot_diagonality_vs_length(records: list[dict], save_path: str, dpi: int):
    """
    Scatter plot: X = text length (GT chars), Y = diagonality (ρ).
    One series per register count, with LOWESS-like trend line.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by register count
    reg_counts = sorted(set(r['num_registers'] for r in records))

    for reg in reg_counts:
        subset = [r for r in records if r['num_registers'] == reg
                  and not np.isnan(r['diagonality'])]
        if not subset:
            continue
        xs = [r['text_length'] for r in subset]
        ys = [r['diagonality'] for r in subset]
        color  = REG_COLORS.get(reg, '#333333')
        marker = REG_MARKERS.get(reg, 'o')

        ax.scatter(xs, ys, c=color, marker=marker, s=50, alpha=0.7,
                   edgecolors='white', linewidths=0.5,
                   label=f'Reg-{reg}', zorder=3)

        # Trend line (polynomial fit, degree 2, or linear if few points)
        if len(xs) >= 4:
            order = sorted(zip(xs, ys))
            xs_s = np.array([o[0] for o in order], dtype=float)
            ys_s = np.array([o[1] for o in order], dtype=float)
            deg = 2 if len(xs) >= 6 else 1
            try:
                coeffs = np.polyfit(xs_s, ys_s, deg)
                xfit = np.linspace(xs_s.min(), xs_s.max(), 200)
                yfit = np.polyval(coeffs, xfit)
                ax.plot(xfit, yfit, color=color, linewidth=1.8, alpha=0.6,
                        linestyle='--')
            except Exception:
                pass

    # Reference lines
    ax.axhline(0.0, color='gray', linewidth=0.6, linestyle=':',  alpha=0.5)
    ax.axhline(1.0, color='gray', linewidth=0.6, linestyle=':',  alpha=0.3)

    # Mark the ~15-char regime-shift boundary
    ax.axvline(15, color='red', linewidth=1.0, linestyle='--', alpha=0.4,
               label='~15-char regime shift')
    ax.annotate('regime\nshift', xy=(15, 0.85), fontsize=8, color='red',
                alpha=0.6, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                          ec='red', lw=0.5))

    ax.set_xlabel('Text Length (number of characters)', fontsize=12)
    ax.set_ylabel('Diagonality  (Spearman ρ)', fontsize=12)
    ax.set_title('Diagonality vs Text Length — Register Token Ablation\n'
                 '(ρ = 1.0: perfect left-to-right reading order)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85, edgecolor='gray')
    ax.set_ylim(-0.4, 1.15)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Plot 1] Diagonality vs Length → {save_path}')
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Peak Sharpness vs Register Count  (grouped box / bar by bucket)
# ═════════════════════════════════════════════════════════════════════════════

def plot_sharpness_vs_registers(records: list[dict], save_path: str, dpi: int):
    """
    Grouped bar chart: X = register count, Y = mean peak sharpness,
    with one group per text-length bucket.  Error bars = ±1 std.
    """
    reg_counts = sorted(set(r['num_registers'] for r in records))
    n_reg = len(reg_counts)
    n_bkt = len(LENGTH_BUCKETS)

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_w = 0.15
    x_base = np.arange(n_reg)

    # bucket colours
    bkt_colors = ['#4daf4a', '#377eb8', '#ff7f00', '#e41a1c']

    for bi, (bkt_label, lo, hi) in enumerate(LENGTH_BUCKETS):
        means, stds, counts = [], [], []
        for reg in reg_counts:
            vals = [r['peak_sharpness'] for r in records
                    if r['num_registers'] == reg and lo <= r['text_length'] <= hi]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                counts.append(len(vals))
            else:
                means.append(0)
                stds.append(0)
                counts.append(0)

        offset = (bi - (n_bkt - 1) / 2) * bar_w
        bars = ax.bar(x_base + offset, means, bar_w, yerr=stds,
                      color=bkt_colors[bi % len(bkt_colors)], alpha=0.8,
                      edgecolor='white', linewidth=0.5,
                      label=bkt_label, capsize=3, error_kw={'linewidth': 0.8})

        # Annotate counts on top
        for xi, (m, c) in enumerate(zip(means, counts)):
            if c > 0:
                ax.text(x_base[xi] + offset, m + stds[xi] + 0.3,
                        f'n={c}', ha='center', va='bottom', fontsize=6,
                        color='gray')

    ax.set_xticks(x_base)
    ax.set_xticklabels([f'Reg-{r}' for r in reg_counts], fontsize=11)
    ax.set_xlabel('Register Token Count', fontsize=12)
    ax.set_ylabel('Peak Sharpness   (max / mean)', fontsize=12)
    ax.set_title('Attention Peak Sharpness vs Register Count\n'
                 'Higher = more focused per-character attention',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85, edgecolor='gray',
              title='Text Length', title_fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Plot 2] Peak Sharpness vs Registers → {save_path}')
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
#  Summary table (terminal + CSV)
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(records: list[dict]):
    """Print aggregated metrics to stdout."""
    reg_counts = sorted(set(r['num_registers'] for r in records))

    print()
    print('=' * 100)
    print(f'{"":>10} {"":>8}  {"Diagonality (ρ)":>17}  {"Peak Sharpness":>15}  '
          f'{"Entropy":>10}  {"Mean CER":>10}  {"N":>4}')
    print('-' * 100)

    for bkt_label, lo, hi in LENGTH_BUCKETS:
        clean_label = bkt_label.replace('\n', ' ')
        for reg in reg_counts:
            sub = [r for r in records
                   if r['num_registers'] == reg and lo <= r['text_length'] <= hi]
            if not sub:
                continue
            diag_vals = [r['diagonality'] for r in sub if not np.isnan(r['diagonality'])]
            sharp_vals = [r['peak_sharpness'] for r in sub]
            ent_vals   = [r['entropy'] for r in sub]
            cer_vals   = [r['cer'] for r in sub]
            n = len(sub)
            d_str = f'{np.mean(diag_vals):.3f}±{np.std(diag_vals):.3f}' if diag_vals else '  N/A'
            s_str = f'{np.mean(sharp_vals):.1f}±{np.std(sharp_vals):.1f}'
            e_str = f'{np.mean(ent_vals):.2f}±{np.std(ent_vals):.2f}'
            c_str = f'{np.mean(cer_vals)*100:.1f}%'
            print(f'{clean_label:>10} Reg-{reg:<3d} {d_str:>17s}  {s_str:>15s}  '
                  f'{e_str:>10s}  {c_str:>10s}  {n:>4d}')
        print()

    # Overall per register count
    print('-' * 100)
    print(f'{"OVERALL":>10}')
    for reg in reg_counts:
        sub = [r for r in records if r['num_registers'] == reg]
        diag_vals = [r['diagonality'] for r in sub if not np.isnan(r['diagonality'])]
        sharp_vals = [r['peak_sharpness'] for r in sub]
        ent_vals   = [r['entropy'] for r in sub]
        cer_vals   = [r['cer'] for r in sub]
        n = len(sub)
        d_str = f'{np.mean(diag_vals):.3f}±{np.std(diag_vals):.3f}' if diag_vals else '  N/A'
        s_str = f'{np.mean(sharp_vals):.1f}±{np.std(sharp_vals):.1f}'
        e_str = f'{np.mean(ent_vals):.2f}±{np.std(ent_vals):.2f}'
        c_str = f'{np.mean(cer_vals)*100:.1f}%'
        print(f'{"":>10} Reg-{reg:<3d} {d_str:>17s}  {s_str:>15s}  '
              f'{e_str:>10s}  {c_str:>10s}  {n:>4d}')
    print('=' * 100)


def save_csv(records: list[dict], csv_path: str):
    """Write all per-(model, image) records to CSV."""
    fieldnames = ['image', 'run_name', 'num_registers', 'label',
                  'text_length', 'n_chars_pred', 'gt_text', 'pred_text',
                  'cer', 'diagonality', 'peak_sharpness', 'entropy',
                  'length_bucket']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, '') for k in fieldnames}
            row['cer'] = f'{r["cer"]:.4f}'
            row['diagonality'] = f'{r["diagonality"]:.4f}' if not np.isnan(r['diagonality']) else 'NaN'
            row['peak_sharpness'] = f'{r["peak_sharpness"]:.2f}'
            row['entropy'] = f'{r["entropy"]:.4f}'
            writer.writerow(row)
    print(f'  [CSV] Metrics → {csv_path}')


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    conf, model_paths, image_paths, save_dir, dpi, layer = parse_args()

    device = getattr(conf, 'device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA not available, falling back to CPU.')
        device = 'cpu'

    dataset_folder = conf.data.path
    classes = np.load(os.path.join(dataset_folder, 'classes.npy'))
    num_classes = len(classes) + 1
    i2c = {(i + 1): c for i, c in enumerate(classes)}
    fixed_size = (conf.preproc.image_height, conf.preproc.image_width)

    out_dir = save_dir or os.path.join('visualizations', 'attention_quality_metrics')
    os.makedirs(out_dir, exist_ok=True)

    # ── Resolve model info and sort by register count ────────────────
    model_infos = [get_run_info(mp) for mp in model_paths]
    sorted_idx  = sorted(range(len(model_infos)),
                         key=lambda i: model_infos[i]['num_registers'])
    model_infos = [model_infos[i] for i in sorted_idx]
    model_paths = [model_paths[i] for i in sorted_idx]

    gt_dict = load_gt(image_paths)

    # ── Header ───────────────────────────────────────────────────────
    print('=' * 70)
    print('Attention Quality Metrics')
    print('=' * 70)
    print(f'Images      : {len(image_paths)}')
    print(f'Models      : {len(model_paths)}')
    for info in model_infos:
        print(f'  {info["label"]:>8s}  ({info["run_name"]})')
    print(f'Layer       : {layer}')
    print(f'Output      : {out_dir}')
    print(f'Timestamp   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    # ── Collect records ──────────────────────────────────────────────
    records: list[dict] = []

    for mi, (info, mp) in enumerate(zip(model_infos, model_paths)):
        print(f'[{mi + 1}/{len(model_paths)}] Loading {info["label"]} '
              f'({info["run_name"]}) ...')
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

            # Trim whitespace at boundaries (CTC artefacts)
            while chars and chars[0] in (' ', '\t'):
                chars.pop(0); positions.pop(0)
            while chars and chars[-1] in (' ', '\t'):
                chars.pop(); positions.pop()

            gt_text = gt_dict.get(image_name, '')
            text_length = len(gt_text) if gt_text else len(pred_text)

            # Raw attention vectors (no gamma)
            char_vectors = get_raw_char_attention(
                result['attn'], result['num_reg'], positions,
                result['grid_size'])

            # Metrics
            diag  = compute_diagonality(positions, char_vectors)
            sharp = compute_peak_sharpness(char_vectors)
            ent   = compute_entropy(char_vectors)
            cer   = compute_cer(pred_text, gt_text) if gt_text else float('nan')

            rec = {
                'image':          image_name,
                'run_name':       info['run_name'],
                'num_registers':  info['num_registers'],
                'label':          info['label'],
                'text_length':    text_length,
                'n_chars_pred':   len(chars),
                'gt_text':        gt_text,
                'pred_text':      pred_text,
                'cer':            cer,
                'diagonality':    diag,
                'peak_sharpness': sharp,
                'entropy':        ent,
                'length_bucket':  _bucket_label(text_length),
            }
            records.append(rec)

            cer_s = f'{cer*100:.1f}%' if not np.isnan(cer) else 'N/A'
            diag_s = f'{diag:.3f}' if not np.isnan(diag) else 'N/A'
            print(f'  {image_name:>25s}  len={text_length:>3d}  '
                  f'ρ={diag_s:>6s}  sharp={sharp:>5.1f}  '
                  f'H={ent:.2f}  CER={cer_s:>6s}  '
                  f'pred="{pred_text[:40]}"')

        del net
        gc.collect()
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print_summary_table(records)

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, 'attention_quality_metrics.csv')
    save_csv(records, csv_path)

    # ── Plot 1: Diagonality vs Text Length ───────────────────────────
    p1 = os.path.join(out_dir, 'diagonality_vs_text_length.png')
    plot_diagonality_vs_length(records, p1, dpi)

    # ── Plot 2: Peak Sharpness vs Register Count ─────────────────────
    p2 = os.path.join(out_dir, 'peak_sharpness_vs_registers.png')
    plot_sharpness_vs_registers(records, p2, dpi)

    print()
    print(f'Done! All outputs in: {out_dir}/')
    print(f'  attention_quality_metrics.csv')
    print(f'  diagonality_vs_text_length.png')
    print(f'  peak_sharpness_vs_registers.png')
    print('=' * 70)


if __name__ == '__main__':
    main()
