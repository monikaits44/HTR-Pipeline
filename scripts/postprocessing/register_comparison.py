#!/usr/bin/env python3
"""
Register Comparison: Systematic attention map quality comparison.
Answers: "Do attention maps become cleaner with registers?"

Loads models with different register counts, evaluates on the same test images,
computes quantitative metrics, and produces comparison visualizations.

Usage:
    python scripts/postprocessing/register_comparison.py \
        --models run_76:0 run_84:4 run_68:7 run_71:16 \
        --config configs/baseline_vit_rgts_v2.yaml \
        --num-samples 50 \
        --save-dir visualizations/register_comparison

    Model format: "run_name:num_registers" or "path/to/model.pt:num_registers"
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import HTRNet
from utils.htr_dataset import HTRDataset
from utils.attention_metrics import compute_sample_metrics
from omegaconf import OmegaConf


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare attention map quality across register counts")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model specs as 'run_name:num_registers' or 'path:num_registers'")
    p.add_argument("--config", default="configs/baseline_vit_rgts_v2.yaml",
                   help="Base config file (register count will be overridden per model)")
    p.add_argument("--data-path", default="data/IAM/processed_lines",
                   help="Path to IAM dataset (must contain train/val/test subdirs and classes.npy)")
    p.add_argument("--dataset", default="test", choices=["val", "test"],
                   help="Which split to evaluate on")
    p.add_argument("--num-samples", type=int, default=50,
                   help="Number of samples to evaluate")
    p.add_argument("--save-dir", default="visualizations/register_comparison")
    p.add_argument("--layer", type=int, default=-1,
                   help="Transformer layer to analyze (-1=last)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--device", default=None,
                   help="Device (auto-detected if not specified)")
    return p.parse_args()


def resolve_model_path(model_spec: str, experiments_dir: str = "saved_models/experiments"):
    """
    Parse model specification string.
    Format: "run_76:0" or "/path/to/model.pt:4"
    Returns: (model_path, num_registers)
    """
    parts = model_spec.rsplit(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model spec '{model_spec}'. Use format 'run_name:num_registers'")

    model_ref, num_reg_str = parts
    num_registers = int(num_reg_str)

    # Check if it's a direct path
    if os.path.isfile(model_ref):
        return model_ref, num_registers

    # Check if it's a run name
    run_path = os.path.join(experiments_dir, model_ref, "model.pt")
    if os.path.isfile(run_path):
        return run_path, num_registers

    raise FileNotFoundError(
        f"Cannot find model for '{model_ref}'. "
        f"Checked: {model_ref} and {run_path}"
    )


def load_model(config_path: str, model_path: str, num_registers: int, device: torch.device):
    """Load model with specified register count."""
    cfg = OmegaConf.load(config_path)

    # Override register count
    cfg.arch.num_registers = num_registers

    # Load character classes from dataset (NOT letter2index.json — that has only A-Z/a-z).
    # The trainer saves classes.npy to the data folder; use it to build the correct i2l.
    data_path = getattr(cfg, 'data', None)
    data_path = getattr(data_path, 'path', 'data/IAM/processed_lines') if data_path else 'data/IAM/processed_lines'
    classes_path = os.path.join(data_path, 'classes.npy')
    classes = list(np.load(classes_path, allow_pickle=True))
    i2l = {str(i): str(c) for i, c in enumerate(classes)}  # 0-indexed: model idx - 1 → char
    num_classes = len(classes) + 1  # +1 for CTC blank at index 0

    model = HTRNet(cfg.arch, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, cfg, i2l


def ctc_decode_with_positions(logits: torch.Tensor, i2l: dict, blank_idx: int = 0):
    """
    CTC greedy decode with character-to-position mapping.

    Args:
        logits: [T, C] logits for one sample
        i2l: index-to-letter dictionary

    Returns:
        decoded_text: str
        char_positions: dict mapping char_idx → list of time positions
    """
    pred_indices = logits.argmax(dim=-1).cpu().numpy()  # [T]

    char_positions = {}
    decoded_chars = []
    current_char_idx = 0
    prev_idx = -1

    for t in range(len(pred_indices)):
        idx = pred_indices[t]
        if idx == blank_idx:
            prev_idx = idx
            continue
        if idx == prev_idx:
            if current_char_idx - 1 in char_positions:
                char_positions[current_char_idx - 1].append(t)
            prev_idx = idx
            continue

        char = i2l.get(str(idx - 1), '?')
        decoded_chars.append(char)
        char_positions[current_char_idx] = [t]
        current_char_idx += 1
        prev_idx = idx

    return ''.join(decoded_chars), char_positions


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, dataloader, i2l, num_registers, layer_idx, device, num_samples):
    """
    Run model on samples and compute attention quality metrics.

    Returns:
        List of per-sample metric dictionaries
    """
    all_metrics = []
    sample_count = 0

    for imgs, transcrs in dataloader:
        if sample_count >= num_samples:
            break

        imgs = imgs.to(device)
        batch_size = imgs.size(0)

        with torch.no_grad():
            logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(imgs)

        for i in range(min(batch_size, num_samples - sample_count)):
            # Get attention for this sample at specified layer
            attn = attn_maps[layer_idx][i]  # [H, S, S]
            attn_avg = attn.mean(dim=0).numpy()  # [S, S]

            R = num_registers
            patch_attn = attn_avg[R:, R:]  # [P, P]

            # CTC decode
            sample_logits = logits[:, i, :]  # [T, C]
            decoded_text, char_positions = ctc_decode_with_positions(sample_logits, i2l)

            if not char_positions:
                sample_count += 1
                continue

            # Compute metrics
            metrics = compute_sample_metrics(
                patch_attn=patch_attn,
                char_positions=char_positions,
                full_attn=attn_avg,
                num_registers=R
            )
            metrics['decoded_text'] = decoded_text
            metrics['ground_truth'] = transcrs[i].strip()
            metrics['sample_idx'] = sample_count
            all_metrics.append(metrics)

            sample_count += 1

    return all_metrics


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_metrics_comparison(results: dict, save_dir: str, dpi: int = 150):
    """
    Generate comparison plots across register counts.

    Args:
        results: Dict mapping num_registers → list of metric dicts
    """
    os.makedirs(save_dir, exist_ok=True)

    reg_counts = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(reg_counts)))

    # Metrics to plot
    metric_keys = [
        ('entropy_mean', 'Attention Entropy (lower = cleaner)', True),
        ('sparsity_gini_mean', 'Sparsity (Gini, higher = cleaner)', False),
        ('peak_sharpness_mean', 'Peak Sharpness (higher = cleaner)', False),
        ('localization_accuracy_mean', 'Character Localization (higher = cleaner)', False),
        ('cross_char_overlap', 'Cross-Character Overlap (lower = cleaner)', True),
        ('locality_mean', 'Attention Locality (lower = more local)', True),
    ]

    # ─── Box plots ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, (key, label, lower_is_better) in enumerate(metric_keys):
        ax = axes[ax_idx]
        data = []
        labels = []
        for rc in reg_counts:
            values = [m[key] for m in results[rc] if key in m]
            if values:
                data.append(values)
                labels.append(f"R={rc}")

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Register Count')
        ax.grid(axis='y', alpha=0.3)

        # Add arrow indicating "better" direction
        direction = "↓ better" if lower_is_better else "↑ better"
        ax.annotate(direction, xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=8, color='green', fontweight='bold')

    plt.suptitle("Attention Quality Metrics vs Register Count", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_boxplots.png"), dpi=dpi, bbox_inches='tight')
    plt.close()

    # ─── Line plot (mean ± std) ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, (key, label, lower_is_better) in enumerate(metric_keys):
        ax = axes[ax_idx]
        means = []
        stds = []
        for rc in reg_counts:
            values = [m[key] for m in results[rc] if key in m]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(0)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(reg_counts, means, yerr=stds, marker='o', capsize=5,
                    linewidth=2, markersize=8, color='steelblue')
        ax.fill_between(reg_counts, means - stds, means + stds, alpha=0.2, color='steelblue')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Number of Registers')
        ax.set_xticks(reg_counts)
        ax.grid(alpha=0.3)

    plt.suptitle("Attention Quality: Mean ± Std vs Register Count", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_lineplot.png"), dpi=dpi, bbox_inches='tight')
    plt.close()

    # ─── Register-specific metrics ───────────────────────────────────────────
    reg_metrics = ['register_specialization', 'register_utilization']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_idx, key in enumerate(reg_metrics):
        ax = axes[ax_idx]
        valid_counts = []
        means = []
        stds = []
        for rc in reg_counts:
            if rc == 0:
                continue
            values = [m[key] for m in results[rc] if key in m]
            if values:
                valid_counts.append(rc)
                means.append(np.mean(values))
                stds.append(np.std(values))

        if valid_counts:
            ax.bar(valid_counts, means, yerr=stds, capsize=5,
                   color='coral', alpha=0.7, edgecolor='darkred')
        ax.set_title(key.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Number of Registers')
        ax.set_xticks(valid_counts if valid_counts else reg_counts)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Register Token Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "register_metrics.png"), dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Plots saved to {save_dir}")


def plot_example_comparison(
    models_data: dict, dataloader, i2l, layer_idx, device,
    save_dir: str, num_examples: int = 3, dpi: int = 150
):
    """
    Side-by-side character attention heatmaps for the same image across models.

    Args:
        models_data: Dict of num_registers → (model, i2l)
    """
    os.makedirs(save_dir, exist_ok=True)

    reg_counts = sorted(models_data.keys())
    example_count = 0

    for imgs, transcrs in dataloader:
        if example_count >= num_examples:
            break

        img = imgs[0:1]  # single image
        img_np = imgs[0, 0].numpy()  # [H, W]

        fig, axes = plt.subplots(len(reg_counts) + 1, 1,
                                  figsize=(12, 2.5 * (len(reg_counts) + 1)))

        # Row 0: input image
        axes[0].imshow(img_np, cmap='gray', aspect='auto')
        axes[0].set_title(f"Input: '{transcrs[0].strip()}'", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        for row_idx, rc in enumerate(reg_counts):
            model, _ = models_data[rc]
            model.eval()

            img_device = img.to(device)
            with torch.no_grad():
                logits, _, attn_maps, _, _ = model.forward_explain(img_device)

            attn = attn_maps[layer_idx][0].mean(dim=0).numpy()  # [S, S]
            R = rc
            patch_attn = attn[R:, R:]  # [P, P]
            P = patch_attn.shape[0]

            sample_logits = logits[:, 0, :]
            decoded_text, char_positions = ctc_decode_with_positions(sample_logits, i2l)

            # Create average attention heatmap over all characters
            if char_positions:
                all_positions = []
                for positions in char_positions.values():
                    all_positions.extend(positions)
                avg_char_attn = patch_attn[all_positions, :].mean(axis=0)
            else:
                avg_char_attn = patch_attn.mean(axis=0)

            # Upsample to image width
            H_img, W_img = img_np.shape
            pixels_per_patch = W_img / P
            heatmap = np.repeat(avg_char_attn, max(1, int(pixels_per_patch)))[:W_img]
            if len(heatmap) < W_img:
                heatmap = np.pad(heatmap, (0, W_img - len(heatmap)))
            heatmap_2d = np.tile(heatmap, (H_img, 1))
            heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)

            ax = axes[row_idx + 1]
            ax.imshow(img_np, cmap='gray', aspect='auto')
            ax.imshow(heatmap_2d, cmap='inferno', alpha=0.5, aspect='auto')
            ax.set_title(f"R={rc} registers — pred: '{decoded_text}'", fontsize=11)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"example_{example_count:02d}.png"),
                    dpi=dpi, bbox_inches='tight')
        plt.close()

        example_count += 1

    print(f"  Saved {example_count} example comparisons to {save_dir}")


def save_results_csv(results: dict, save_dir: str):
    """Save metrics to CSV for further analysis."""
    import csv
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "metrics_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        metric_keys = [
            'entropy_mean', 'sparsity_gini_mean', 'peak_sharpness_mean',
            'localization_accuracy_mean', 'cross_char_overlap', 'locality_mean',
            'register_specialization', 'register_utilization', 'num_characters'
        ]
        writer.writerow(['num_registers', 'sample_idx'] + metric_keys +
                        ['decoded_text', 'ground_truth'])

        for rc in sorted(results.keys()):
            for m in results[rc]:
                row = [rc, m.get('sample_idx', '')]
                row += [m.get(k, '') for k in metric_keys]
                row += [m.get('decoded_text', ''), m.get('ground_truth', '')]
                writer.writerow(row)

    print(f"  CSV saved to {csv_path}")

    # Also save summary statistics
    summary_path = os.path.join(save_dir, "summary_statistics.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REGISTER COMPARISON — SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        for rc in sorted(results.keys()):
            f.write(f"{'─' * 40}\n")
            f.write(f"Registers: {rc}  (N={len(results[rc])} samples)\n")
            f.write(f"{'─' * 40}\n")
            for key in metric_keys:
                values = [m[key] for m in results[rc] if key in m and m[key] != '']
                if values:
                    f.write(f"  {key:35s}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
            f.write("\n")

    print(f"  Summary saved to {summary_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Layer: {args.layer}")
    print(f"Samples: {args.num_samples}")
    print()

    # Parse model specifications
    model_specs = []
    for spec in args.models:
        model_path, num_registers = resolve_model_path(spec)
        model_specs.append((model_path, num_registers))
        print(f"  Model: {model_path} (R={num_registers})")

    print()

    # Load dataset
    cfg = OmegaConf.load(args.config)
    fixed_size = (cfg.preproc.image_height, cfg.preproc.image_width)
    dataset = HTRDataset(args.data_path, args.dataset, fixed_size=fixed_size, transforms=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset: {args.data_path}/{args.dataset} ({len(dataset)} samples)")
    print()

    # Evaluate each model
    results = {}
    models_data = {}

    for model_path, num_registers in model_specs:
        print(f"{'═' * 60}")
        print(f"Evaluating: R={num_registers} ({model_path})")
        print(f"{'═' * 60}")

        t0 = time.time()
        model, model_cfg, i2l = load_model(args.config, model_path, num_registers, device)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        metrics = evaluate_model(
            model, dataloader, i2l, num_registers,
            args.layer, device, args.num_samples
        )
        print(f"  Evaluated {len(metrics)} samples in {time.time() - t0:.1f}s")

        # Print summary
        if metrics:
            for key in ['entropy_mean', 'sparsity_gini_mean', 'localization_accuracy_mean']:
                values = [m[key] for m in metrics if key in m]
                if values:
                    print(f"    {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        results[num_registers] = metrics
        models_data[num_registers] = (model, i2l)
        print()

    # Generate visualizations
    print("Generating plots...")
    plot_metrics_comparison(results, args.save_dir, args.dpi)

    print("Generating example comparisons...")
    plot_example_comparison(
        models_data, dataloader, i2l, args.layer, device,
        os.path.join(args.save_dir, "examples"),
        num_examples=3, dpi=args.dpi
    )

    # Save CSV results
    print("Saving results...")
    save_results_csv(results, args.save_dir)

    print(f"\n{'═' * 60}")
    print(f"DONE — All results saved to: {args.save_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
