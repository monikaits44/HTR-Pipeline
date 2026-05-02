#!/usr/bin/env python3
"""
Simple attention map visualization.

Takes an image and model, shows head-averaged attention at the middle layer
as a token×token heatmap (like "Attention Is All You Need" style).

Usage:
    python scripts/postprocessing/simple_attention_map.py \
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
        --model-path saved_models/experiments/run_55/model.pt \
        --image notebook/sample_images/a06-110-08.png
"""

import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import torch
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models import HTRNet
from utils.preprocessing import load_image, preprocess
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+", help="YAML config files")
    p.add_argument("--model-path", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--save-dir", default="visualizations/simple_attention_maps")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load config ──────────────────────────────────────────────────────
    conf = OmegaConf.merge(*[OmegaConf.load(c) for c in args.configs])
    OmegaConf.set_struct(conf, False)

    # Register count from checkpoint's config.json
    run_dir = Path(args.model_path).parent
    with open(run_dir / "config.json") as f:
        run_cfg = json.load(f)
    n_reg = run_cfg.get("arch", {}).get("num_registers", 0)
    conf.arch.num_registers = n_reg

    # Classes
    classes = np.load(os.path.join(conf.data.path, "classes.npy"))
    num_classes = len(classes) + 1

    # ── Build model ──────────────────────────────────────────────────────
    net = HTRNet(conf.arch, num_classes)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt, strict=True)
    net.to(device).eval()

    # ── Preprocess image ─────────────────────────────────────────────────
    fixed_size = (conf.preproc.image_height, conf.preproc.image_width)
    img = load_image(args.image)
    processed = preprocess(img, fixed_size)
    x = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0).to(device)

    # ── Forward pass ─────────────────────────────────────────────────────
    with torch.no_grad():
        logits, reg_tokens, attn_maps, token_norms, (Hp, Wp) = net.forward_explain(x)

    n_layers = len(attn_maps)
    mid = n_layers // 2  # middle layer index
    n_heads = attn_maps[mid].shape[1]
    S = attn_maps[mid].shape[2]
    R = n_reg
    T = S - R

    img_name = Path(args.image).stem
    run_name = run_dir.name

    print(f"Image  : {args.image}")
    print(f"Model  : {args.model_path}  (Reg-{n_reg}, {run_name})")
    print(f"Layer  : {mid+1}/{n_layers} (middle)")
    print(f"Tokens : S={S}  ({R} register + {T} patch),  {n_heads} heads")

    # ── Head-averaged attention at middle layer ──────────────────────────
    attn = attn_maps[mid][0].mean(dim=0).cpu().numpy()  # [S, S]

    os.makedirs(args.save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    seaborn.heatmap(attn, square=True, vmin=0.0, vmax=attn.max(),
                    cbar=True, ax=ax, cmap="viridis")

    if R > 0:
        ax.axhline(y=R, color="red", lw=2, ls="--", alpha=0.8)
        ax.axvline(x=R, color="red", lw=2, ls="--", alpha=0.8)

    ax.set_title(f"Reg-{n_reg} ({run_name}) — {img_name}\n"
                 f"Head-averaged attention, Layer {mid+1}/{n_layers}\n"
                 f"{S}×{S}  ({R} reg + {T} patch)",
                 fontsize=13)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")

    save_path = os.path.join(args.save_dir,
                             f"attn_Reg-{n_reg}_{run_name}_{img_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved  : {save_path}")


if __name__ == "__main__":
    main()
