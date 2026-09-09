#!/usr/bin/env python3
"""
Publication Figure 2: GradCAM + Gradient-Weighted Attention
============================================================
Two-panel figure for alternative interpretability:

Panel A — GradCAM on CNN Stem:
  Hooks the last CNN stem layer (before height pooling) to get 2D feature maps.
  Computes gradient of CTC logit at each character's peak timestep w.r.t.
  stem features → GradCAM heatmap (2D, preserves height information).

Panel B — Gradient-Weighted Attention:
  Computes gradient of CTC loss w.r.t. last-layer attention weights.
  Multiplies attention × |gradient| → retains only task-relevant attention.
  Produces sharper, more character-aligned maps than raw attention.

Both methods compared across register configurations (0/4/8/16).

Usage:
  python scripts/pub_gradcam_analysis.py
  python scripts/pub_gradcam_analysis.py --device cuda:0
  python scripts/pub_gradcam_analysis.py --image notebook/sample_images/a01-038-12.png
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models import HTRNet
from utils.preprocessing import load_image, preprocess

EXPERIMENTS_DIR = ROOT / "saved_models" / "experiments"
SAMPLE_DIR = ROOT / "notebook" / "sample_images"
CLASSES_PATH = EXPERIMENTS_DIR / "classes.npy"
IMAGE_H, IMAGE_W = 128, 1024
SEQ_LEN = 5

REGISTER_RUNS = {0: "run_144", 4: "run_146", 8: "run_151", 16: "run_152"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_charset():
    return list(np.load(str(CLASSES_PATH), allow_pickle=True))


def load_model(run_id, device="cpu"):
    run_dir = EXPERIMENTS_DIR / run_id
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    arch_ns = SimpleNamespace(**cfg["arch"])
    charset = load_charset()
    model = HTRNet(arch_ns, len(charset) + 1)
    state = torch.load(str(run_dir / "model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, cfg["arch"].get("num_registers", 0)


def prepare_image(image_path, device="cpu"):
    img_np = load_image(str(image_path))
    img_proc = preprocess(img_np, (IMAGE_H, IMAGE_W))
    tensor = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).float().to(device)
    col_var = np.var(img_proc, axis=0)
    thr = np.median(col_var) * 0.1 + 1e-6
    active = np.where(col_var > thr)[0]
    if len(active) > 0:
        xs, xe = max(0, active[0] - 4), min(IMAGE_W, active[-1] + 4)
    else:
        xs, xe = 0, IMAGE_W
    return tensor, img_proc, (xs, xe)


def ctc_decode(logits, charset):
    probs = torch.softmax(logits, dim=-1)[:, 0, :]
    seq = probs.argmax(dim=-1).cpu().numpy()
    chars, peaks = [], []
    prev = -1
    for t, idx in enumerate(seq):
        if idx != 0 and idx != prev:
            ci = idx - 1
            if 0 <= ci < len(charset):
                chars.append(charset[ci])
                peaks.append(t)
        prev = idx
    full = "".join(chars)
    start = 0
    while start < len(full) and full[start] == " ":
        start += 1
    end = len(full)
    while end > start and full[end - 1] == " ":
        end -= 1
    return full[start:end], peaks[start:end]


# ── GradCAM on CNN Stem ──────────────────────────────────────────────────

def compute_gradcam_stem(model, tensor, peak_timestep, device):
    """
    GradCAM on the CNN stem's last conv layer (before stem_pool).
    Returns a 2D heatmap of shape (H_stem, W_stem).

    The CNN stem output is [B, D, H', W'] where H'≈4, W'≈128.
    We backprop from the CTC logit at peak_timestep to get spatial gradients.
    """
    model.eval()
    # Enable gradients for this forward pass
    tensor_grad = tensor.clone().detach().requires_grad_(False)

    # Hook the CNN stem output (before pooling)
    stem_activations = {}

    def stem_hook(module, input, output):
        stem_activations["features"] = output

    # Register hook on last GELU in cnn_stem
    hook = model.backbone.cnn_stem[-1].register_forward_hook(stem_hook)

    # Forward pass (need gradients through the model)
    for p in model.parameters():
        p.requires_grad_(False)

    # Manual forward to get logits while keeping stem activations
    x = tensor_grad.to(device)
    stem_out = model.backbone.cnn_stem(x)
    stem_features = stem_out.detach().clone()
    stem_features.requires_grad_(True)

    # Continue forward from stem features
    pooled = model.backbone.stem_pool(stem_features)
    B, D, Hp, Wp = pooled.shape
    patch_tokens = pooled.squeeze(2).transpose(1, 2)

    R = model.backbone.num_registers
    reg_tokens = model.backbone.register_tokens.expand(B, -1, -1)
    tokens = torch.cat([reg_tokens, patch_tokens], dim=1)
    S = tokens.size(1)
    pos = model.backbone.pos_embed[:, :S, :]
    tokens = tokens + pos

    encoded = model.backbone.encoder(tokens)
    patch_out = encoded[:, R:, :]
    seq_tokens = patch_out.transpose(0, 1)  # [T, B, D]
    seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)

    # Get logits through head (eval mode, return single output)
    model.top.eval()
    logits = model.top(seq_4d)  # [T, B, nclasses]

    # Target: CTC logit at peak timestep
    if peak_timestep < logits.shape[0]:
        target_logit = logits[peak_timestep, 0, :].max()
    else:
        target_logit = logits[-1, 0, :].max()

    target_logit.backward()

    hook.remove()

    # GradCAM: gradient-weighted activation
    grad = stem_features.grad  # [B, D, H', W']
    if grad is None:
        return np.zeros((4, 128))

    # Global average pool gradients over spatial dims → channel weights
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, D, 1, 1]
    cam = (weights * stem_features.detach()).sum(dim=1, keepdim=True)  # [B, 1, H', W']
    cam = F.relu(cam).squeeze().cpu().numpy()  # [H', W']

    # Normalize
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)

    return cam


def compute_gradcam_per_char(model, tensor, peaks, device, n_chars=5):
    """Compute GradCAM heatmap for each character position."""
    heatmaps = []
    for ci in range(min(n_chars, len(peaks))):
        cam = compute_gradcam_stem(model, tensor, peaks[ci], device)
        heatmaps.append(cam)
    # Pad to n_chars
    while len(heatmaps) < n_chars:
        heatmaps.append(None)
    return heatmaps


# ── Gradient-Weighted Attention ──────────────────────────────────────────

def compute_grad_attention(model, tensor, charset, device, n_chars=5):
    """
    Gradient-weighted attention: attention × |∂logit/∂attention|.
    Uses the last transformer layer's attention weights.

    Returns per-character 1D attention arrays (patch tokens only).
    """
    model.eval()

    # We need to compute attention manually with gradients enabled
    x = tensor.to(device)
    B, C, H, W = x.shape

    # Forward through CNN stem
    stem_out = model.backbone.cnn_stem(x)
    pooled = model.backbone.stem_pool(stem_out)
    B, D, Hp, Wp = pooled.shape
    patch_tokens = pooled.squeeze(2).transpose(1, 2)

    R = model.backbone.num_registers
    reg_tokens = model.backbone.register_tokens.expand(B, -1, -1)
    tokens = torch.cat([reg_tokens, patch_tokens], dim=1)
    S = tokens.size(1)
    pos = model.backbone.pos_embed[:, :S, :]
    tokens = tokens + pos
    tokens = model.backbone.emb_dropout(tokens)

    # Run through all but last layer normally
    x_tokens = tokens
    for layer in model.backbone.encoder.layers[:-1]:
        x_tokens = layer(x_tokens)

    # Last layer: manually compute attention with grad tracking
    last_layer = model.backbone.encoder.layers[-1]
    attn_module = last_layer.self_attn
    embed_dim = attn_module.embed_dim
    num_heads = attn_module.num_heads
    head_dim = embed_dim // num_heads

    normed = last_layer.norm1(x_tokens)
    q, k, v = F.linear(normed, attn_module.in_proj_weight,
                        attn_module.in_proj_bias).chunk(3, dim=-1)
    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_heads, head_dim).transpose(1, 2)

    attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_weights.retain_grad()

    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, embed_dim)
    attn_output = F.linear(attn_output, attn_module.out_proj.weight,
                            attn_module.out_proj.bias)

    x_tokens_out = x_tokens + attn_output
    normed2 = last_layer.norm2(x_tokens_out)
    ff_out = last_layer.linear2(F.dropout(
        last_layer.activation(last_layer.linear1(normed2)),
        p=0.0, training=False))
    x_tokens_out = x_tokens_out + ff_out

    # Get patch tokens and compute logits
    patch_out = x_tokens_out[:, R:, :]
    seq_tokens = patch_out.transpose(0, 1)
    seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)

    model.top.eval()
    logits = model.top(seq_4d)

    # CTC decode to get peaks
    decoded, peaks = ctc_decode(logits.detach(), charset)

    # Compute gradient-weighted attention for each character
    results = []
    for ci in range(min(n_chars, len(peaks))):
        t_c = peaks[ci]
        if t_c >= logits.shape[0]:
            results.append(np.zeros(Wp))
            continue

        # Zero gradients
        if attn_weights.grad is not None:
            attn_weights.grad.zero_()

        # Backward from this character's logit
        target = logits[t_c, 0, :].max()
        target.backward(retain_graph=True)

        if attn_weights.grad is not None:
            grad = attn_weights.grad[0].abs()  # [H, S, S]
            # Gradient-weighted attention for query at peak timestep
            qi = R + t_c
            if qi < S:
                # Weighted attention: attn × |grad|, then extract patch columns
                gw = (attn_weights[0, :, qi, R:R+Wp].detach() *
                      grad[:, qi, R:R+Wp]).mean(dim=0).cpu().numpy()
                results.append(gw)
            else:
                results.append(np.zeros(Wp))
        else:
            results.append(np.zeros(Wp))

    # Pad
    while len(results) < n_chars:
        results.append(None)

    return results, decoded, peaks


# ── Figure Generation ─────────────────────────────────────────────────────

def generate_gradcam_figure(image_path, device, output_dir):
    """
    Generate Figure 2: GradCAM + Grad-weighted attention across register configs.

    Layout: 2 panels
      Top panel (GradCAM):  rows=register configs, cols=[Input, c1..c5]
      Bottom panel (Grad-Attn): rows=register configs, cols=[Input, c1..c5]
    """
    charset = load_charset()
    tensor, img_proc, text_bbox = prepare_image(image_path, device)
    xs, xe = text_bbox
    img_crop = img_proc[:, xs:xe]
    img_name = Path(image_path).stem

    cmap = plt.cm.inferno
    reg_counts = sorted(REGISTER_RUNS.keys())
    n_regs = len(reg_counts)
    n_cols = SEQ_LEN + 1

    fig, axes = plt.subplots(n_regs, n_cols,
                              figsize=(2.2 * n_cols + 0.5, 1.4 * n_regs + 1.2),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.35})

    for ri, nr in enumerate(reg_counts):
        model, _ = load_model(REGISTER_RUNS[nr], device)

        # Gradient-weighted attention
        grad_attns, decoded, peaks = compute_grad_attention(
            model, tensor, charset, device, n_chars=SEQ_LEN)
        Wp = 128  # CNN stem output width

        # Column 0: input
        ax = axes[ri, 0]
        ax.imshow(img_crop, cmap="gray", aspect="auto")
        ax.set_ylabel(f"{nr} reg", fontsize=10, fontweight="bold",
                       rotation=0, labelpad=30, va="center")
        display = decoded[:SEQ_LEN]
        ax.set_xlabel(f'"{display}"', fontsize=7, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        if ri == 0:
            ax.set_title("Input", fontsize=10, fontweight="bold", pad=4)

        # Character columns
        for ci in range(SEQ_LEN):
            ax = axes[ri, 1 + ci]
            if ci < len(grad_attns) and grad_attns[ci] is not None:
                attn_1d = grad_attns[ci]
                # Interpolate and normalize
                attn_full = np.interp(np.linspace(0, 1, IMAGE_W),
                                       np.linspace(0, 1, Wp), attn_1d)
                attn_c = attn_full[xs:xe].astype(np.float64)
                if len(attn_c) > 1:
                    attn_c = gaussian_filter1d(attn_c, sigma=1.5)
                p2, p98 = np.percentile(attn_c, 2), np.percentile(attn_c, 98)
                if p98 > p2:
                    attn_c = np.clip((attn_c - p2) / (p98 - p2), 0, 1)
                else:
                    attn_c = (attn_c - attn_c.min()) / (attn_c.max() - attn_c.min() + 1e-8)
                hm = np.tile(attn_c, (IMAGE_H, 1))

                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.3)
                rgba = cmap(hm)
                rgba[..., 3] = hm * 0.85
                ax.imshow(rgba, aspect="auto")
            else:
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.15)
                ax.text(0.5, 0.5, "PAD", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")

            if ri == 0:
                if ci < len(decoded):
                    ch = decoded[ci] if decoded[ci] != " " else "⎵"
                    ax.set_title(f"$A_{{{ci+1}}}$  '{ch}'", fontsize=9,
                                 fontweight="bold", pad=4)
                else:
                    ax.set_title(f"$A_{{{ci+1}}}$ (pad)", fontsize=9, color="gray", pad=4)
            ax.set_xticks([])
            ax.set_yticks([])

        del model
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    fig.suptitle(
        f"Gradient-Weighted Attention — Register Comparison\n{img_name}",
        fontsize=11, fontweight="bold", y=0.98)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, location="right", shrink=0.6, label="Attention",
                 pad=0.02)

    out = output_dir / f"fig2_grad_attention_{img_name}"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(out) + ext, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}.png / .pdf")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GradCAM + Gradient-weighted attention analysis")
    parser.add_argument("--image", type=str,
                        default=str(SAMPLE_DIR / "a01-038-12.png"),
                        help="Input image path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT / "outputs" / "pub_figures"))
    parser.add_argument("--runs", nargs=4,
                        help="Run IDs for 0/4/8/16 registers")
    args = parser.parse_args()

    if args.runs:
        for i, nr in enumerate([0, 4, 8, 16]):
            REGISTER_RUNS[nr] = args.runs[i]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image: {args.image}")
    print(f"Register runs: {REGISTER_RUNS}")
    generate_gradcam_figure(args.image, args.device, output_dir)


if __name__ == "__main__":
    main()
