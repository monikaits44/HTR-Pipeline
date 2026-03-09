#!/usr/bin/env python3
"""
ViT Attention Explorer – Streamlit Dashboard
=============================================

Interactive visualization of Attention Maps, Attention Rollout, and Grad-CAM
for ViT-RGTS models with register tokens (0 → 16).

Usage:
    cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
    streamlit run dashboard/attention_explorer.py

Features:
    - Upload any handwritten line image (or pick from samples)
    - Select one or more model runs (runs 50–54, registers 0–16)
    - Per-layer, per-head attention map visualization
    - Attention Rollout across all layers
    - Proper Grad-CAM (Selvaraju et al. 2017)
    - CTC decoding with predicted text
    - Side-by-side comparison across register counts
"""

import os
import sys
import json
import gc
import math
from pathlib import Path
from copy import deepcopy
from io import BytesIO

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import streamlit as st

# ── Project root ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import HTRNet
from utils.preprocessing import load_image, preprocess

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

RUNS = {
    "run_50": {"label": "Run 50", "registers": 0,  "color": "#00d4ff"},
    "run_51": {"label": "Run 51", "registers": 2,  "color": "#38bdf8"},
    "run_52": {"label": "Run 52", "registers": 4,  "color": "#818cf8"},
    "run_53": {"label": "Run 53", "registers": 8,  "color": "#a855f7"},
    "run_54": {"label": "Run 54", "registers": 16, "color": "#f472b6"},
}

FIXED_SIZE = (128, 1024)
BORDER_SIZE = 8
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "IAM", "processed_lines")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models", "experiments")
SAMPLE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "notebook", "sample_images")

REG_NOTES = {
    0:  "No registers → attention diffuse, heads share noise patterns",
    2:  "2 registers → slight focus gain, noise partially absorbed",
    4:  "4 registers → heads begin to specialise clearly",
    8:  "8 registers → distinct role per head, less scatter",
    16: "16 registers → highly focused, each head very specialised",
}


# ═════════════════════════════════════════════════════════════════════════════
# Model Loading (cached)
# ═════════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_classes():
    classes_path = os.path.join(DATASET_PATH, "classes.npy")
    if not os.path.exists(classes_path):
        st.error(f"classes.npy not found at {classes_path}")
        st.stop()
    classes = np.load(classes_path, allow_pickle=True)
    num_classes = len(classes) + 1  # +1 for CTC blank (index 0)
    i2c = {(i + 1): str(c) for i, c in enumerate(classes)}
    return classes, num_classes, i2c


@st.cache_resource
def load_model(model_path, num_registers, num_classes):
    """Load and cache a ViT-RGTS model."""
    device = get_device()

    # Load config from run directory
    run_dir = os.path.dirname(os.path.abspath(model_path))
    config_json = os.path.join(run_dir, "config.json")

    if os.path.isfile(config_json):
        with open(config_json) as f:
            cfg = json.load(f)
        from omegaconf import OmegaConf
        arch_cfg = OmegaConf.create(cfg["arch"])
    else:
        # Fallback: build config manually
        from omegaconf import OmegaConf
        arch_cfg = OmegaConf.create({
            "type": "vit_rgts",
            "image_height": 128,
            "image_width": 1024,
            "use_cnn_stem": True,
            "patch_height": 16,
            "patch_width": 16,
            "dim": 256,
            "depth": 6,
            "heads": 8,
            "mlp_dim": 1024,
            "num_registers": num_registers,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "head_type": "both",
            "rnn_type": "lstm",
            "rnn_layers": 3,
            "rnn_hidden_size": 256,
        })

    arch_cfg.num_registers = num_registers
    net = HTRNet(arch_cfg, num_classes)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt, strict=True)
    net.to(device).eval()
    return net


# ═════════════════════════════════════════════════════════════════════════════
# Image Preprocessing
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_array):
    """Preprocess a grayscale image for the model.
    image_array: numpy array (grayscale, values 0-255 or 0-1).
    Returns: tensor [1, 1, H, W]
    """
    if image_array.max() > 1.0:
        image_array = 1.0 - image_array / 255.0
    else:
        image_array = 1.0 - image_array

    processed = preprocess(image_array, FIXED_SIZE, BORDER_SIZE)
    tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
    return tensor


def load_and_preprocess(uploaded_file):
    """Load image from upload widget."""
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img).astype(np.float64)
    return img_np


# ═════════════════════════════════════════════════════════════════════════════
# CTC Decoding
# ═════════════════════════════════════════════════════════════════════════════

def ctc_decode(logits_np, i2c, blank_id=0):
    """Greedy CTC decode from logits [T, 1, C] → string."""
    tdec = logits_np.argmax(2).squeeze()
    chars = []
    prev = -1
    for v in tdec:
        v = int(v)
        if v != blank_id and v != prev:
            chars.append(i2c.get(v, "?"))
        prev = v
    return "".join(chars)


def ctc_char_positions(logits_np, i2c, blank_id=0):
    """Return characters and their time-step positions."""
    tdec = logits_np.argmax(2).squeeze()
    chars, positions = [], []
    prev = -1
    for t, v in enumerate(tdec):
        v_int = int(v)
        if v_int != blank_id and v_int != prev:
            chars.append(str(i2c.get(v_int, "?")))
            positions.append(t)
        prev = v_int
    return chars, positions


# ═════════════════════════════════════════════════════════════════════════════
# Attention Extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_attention_maps(net, img_tensor, device):
    """
    Extract per-layer attention maps using forward_explain.
    Returns dict with attn_maps, logits, grid_size, num_registers.
    """
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits, reg_tokens, attn_maps_list, token_norms, grid_size = \
            net.forward_explain(img_tensor)

    if isinstance(logits, tuple):
        logits = logits[0]

    num_reg = reg_tokens.shape[1] if reg_tokens is not None else 0

    return {
        "attn_maps": attn_maps_list,  # List[L] of [B, H, S, S]
        "logits": logits.cpu().numpy(),
        "grid_size": grid_size,
        "num_reg": num_reg,
        "token_norms": token_norms,
    }


def attention_rollout(attn_maps_list):
    """Compute attention rollout across all layers."""
    result = None
    for attn in attn_maps_list:
        attn_avg = attn.mean(dim=1)  # avg over heads → [B, S, S]
        S = attn_avg.size(-1)
        I = torch.eye(S, device=attn_avg.device).unsqueeze(0)
        attn_res = 0.5 * attn_avg + 0.5 * I  # residual connection
        attn_res = attn_res / (attn_res.sum(dim=-1, keepdim=True) + 1e-8)
        result = attn_res if result is None else torch.bmm(attn_res, result)
    return result[0].cpu().numpy()  # [S, S]


def get_patch_attention_from_rollout(rollout_matrix, num_reg, grid_size):
    """Extract patch→patch attention from rollout for spatial heatmap."""
    Hp, Wp = grid_size
    T = Hp * Wp
    # Average over patch rows (how patches attend to other patches)
    patch_attn = rollout_matrix[num_reg:num_reg + T, num_reg:num_reg + T]
    # Sum each column = total attention received by each patch
    spatial = patch_attn.mean(axis=0)
    return spatial.reshape(Hp, Wp)


def get_head_attention_map(attn_map_layer, head_idx, num_reg, grid_size):
    """
    Extract spatial attention map for a specific head from one layer.
    attn_map_layer: [B, H, S, S] (single layer)
    Returns: [Hp, Wp] heatmap
    """
    Hp, Wp = grid_size
    T = Hp * Wp

    # attn_map_layer: [B=1, H, S, S] → [H, S, S]
    attn = attn_map_layer[0]  # [H, S, S]

    if head_idx == "avg":
        head_attn = attn.mean(dim=0).numpy()  # [S, S]
    else:
        head_attn = attn[head_idx].numpy()  # [S, S]

    # Patch-to-patch attention (skip register rows/cols)
    patch_attn = head_attn[num_reg:num_reg + T, num_reg:num_reg + T]

    # Mean attention received per patch position
    spatial = patch_attn.mean(axis=0)
    return spatial.reshape(Hp, Wp)


# ═════════════════════════════════════════════════════════════════════════════
# Grad-CAM
# ═════════════════════════════════════════════════════════════════════════════

def compute_gradcam(net, img_tensor, device):
    """
    Standard Grad-CAM for ViT patch tokens (Selvaraju et al. 2017).
    importance[t] = ReLU( sum_d( grad[t,d] * activation[t,d] ) )
    """
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(False)

    # 1. Forward backbone → activations
    seq_tokens, reg_tokens, (Hp, Wp) = net.backbone(img_tensor)
    seq_tokens.retain_grad()

    # 2. Forward CTC head → logits
    seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)  # [1, D, 1, T]
    logits = net.top(seq_4d)
    if isinstance(logits, tuple):
        logits = logits[0]  # [T, 1, C]

    # 3. Backward target: non-blank predicted logits
    pred_classes = logits.argmax(dim=2)  # [T, 1]
    non_blank = pred_classes.squeeze(1) != 0

    if non_blank.any():
        target = logits[non_blank].gather(
            2, pred_classes[non_blank].unsqueeze(2)
        ).sum()
    else:
        target = logits.gather(2, pred_classes.unsqueeze(2)).sum()

    net.zero_grad()
    target.backward()

    # 4. Grad-CAM
    grads = seq_tokens.grad.detach()  # [T, 1, D]
    acts = seq_tokens.detach()         # [T, 1, D]

    cam = (grads * acts).squeeze(1).sum(dim=-1)  # [T]
    cam = torch.relu(cam).cpu().numpy()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return {
        "importance": cam,
        "grid_size": (Hp, Wp),
        "logits": logits.detach().cpu().numpy(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Upscale patch map → image coordinates
# ═════════════════════════════════════════════════════════════════════════════

def upscale_patch_aligned(patch_map, H_img, W_img, fixed_size=FIXED_SIZE, border_size=BORDER_SIZE):
    """Map [Hp, Wp] patch heatmap → [H_img, W_img] pixel heatmap."""
    Hp, Wp = patch_map.shape
    H_pre, W_pre = fixed_size
    ph = H_pre // Hp
    pw = W_pre // Wp

    heat_full = np.kron(patch_map, np.ones((ph, pw)))
    r0, c0 = border_size, border_size
    crop = heat_full[r0:r0 + H_img, c0:c0 + W_img]

    if crop.shape[0] < H_img or crop.shape[1] < W_img:
        padded = np.zeros((H_img, W_img), dtype=crop.dtype)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded

    return crop


# ═════════════════════════════════════════════════════════════════════════════
# Plotting Helpers
# ═════════════════════════════════════════════════════════════════════════════

def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1520", edgecolor="none")
    buf.seek(0)
    return Image.open(buf)


def plot_heatmap_overlay(img_np, heatmap, title="", alpha=0.55, cmap="inferno"):
    """Create overlay figure: grayscale image + heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5),
                              facecolor="#0d1520")

    for ax in axes:
        ax.set_facecolor("#080c12")
        ax.tick_params(colors="#475569", labelsize=6)

    # Original
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original", fontsize=10, color="#e2e8f0", pad=8)
    axes[0].axis("off")

    # Overlay
    axes[1].imshow(img_np, cmap="gray")
    im = axes[1].imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    axes[1].set_title(f"{title} Overlay", fontsize=10, color="#e2e8f0", pad=8)
    axes[1].axis("off")

    # Raw heatmap
    im2 = axes[2].imshow(heatmap, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f"{title} Raw", fontsize=10, color="#e2e8f0", pad=8)
    axes[2].axis("off")
    cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6, colors="#94a3b8")
    cbar.set_label("importance", fontsize=7, color="#94a3b8")

    plt.tight_layout()
    return fig


def plot_attention_heads(attn_maps_layer, num_reg, grid_size, img_np,
                         layer_idx, alpha=0.55):
    """Plot all heads + avg for one layer."""
    H = attn_maps_layer.shape[1]  # number of heads
    Hp, Wp = grid_size
    n_cols = min(4, H + 1)
    n_rows = math.ceil((H + 1) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              facecolor="#0d1520")
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#080c12")
            ax.axis("off")

    for hd in range(H):
        row, col = divmod(hd, n_cols)
        hmap = get_head_attention_map(attn_maps_layer, hd, num_reg, grid_size)
        hmap_up = upscale_patch_aligned(hmap, img_np.shape[0], img_np.shape[1])

        # Percentile scaling
        p01, p99 = np.percentile(hmap_up, 1), np.percentile(hmap_up, 99)
        if p99 - p01 < 1e-8:
            p01, p99 = hmap_up.min(), hmap_up.max()
        hmap_scaled = np.clip((hmap_up - p01) / (p99 - p01 + 1e-8), 0, 1)

        axes[row, col].imshow(img_np, cmap="gray")
        axes[row, col].imshow(hmap_scaled, cmap="inferno", alpha=alpha, vmin=0, vmax=1)
        axes[row, col].set_title(f"Head {hd + 1}", fontsize=9, color="#e2e8f0", pad=4)

    # Average head
    row_a, col_a = divmod(H, n_cols)
    avg_hmap = get_head_attention_map(attn_maps_layer, "avg", num_reg, grid_size)
    avg_up = upscale_patch_aligned(avg_hmap, img_np.shape[0], img_np.shape[1])
    p01, p99 = np.percentile(avg_up, 1), np.percentile(avg_up, 99)
    if p99 - p01 < 1e-8:
        p01, p99 = avg_up.min(), avg_up.max()
    avg_scaled = np.clip((avg_up - p01) / (p99 - p01 + 1e-8), 0, 1)

    axes[row_a, col_a].imshow(img_np, cmap="gray")
    axes[row_a, col_a].imshow(avg_scaled, cmap="inferno", alpha=alpha, vmin=0, vmax=1)
    axes[row_a, col_a].set_title("Avg Head", fontsize=9, color="#fbbf24", pad=4)

    # Hide unused axes
    for idx in range(H + 1, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Layer {layer_idx + 1} — Per-Head Attention Maps",
                 fontsize=12, color="#00d4ff", y=1.02)
    plt.tight_layout()
    return fig


def plot_attention_matrix(attn_matrix, num_reg, title="Attention Matrix"):
    """Plot raw attention matrix as heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="#0d1520")
    ax.set_facecolor("#080c12")

    im = ax.imshow(attn_matrix, cmap="inferno", aspect="auto")
    ax.set_title(title, fontsize=11, color="#e2e8f0", pad=8)
    ax.set_xlabel("Key position", fontsize=9, color="#94a3b8")
    ax.set_ylabel("Query position", fontsize=9, color="#94a3b8")
    ax.tick_params(colors="#475569", labelsize=7)

    # Draw register boundary
    if num_reg > 0:
        ax.axhline(y=num_reg - 0.5, color="#00d4ff", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(x=num_reg - 0.5, color="#00d4ff", linewidth=0.8, linestyle="--", alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6, colors="#94a3b8")

    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Auto-discover models
# ═════════════════════════════════════════════════════════════════════════════

def discover_models():
    """Find all available model runs."""
    models = {}
    if not os.path.isdir(SAVED_MODELS_DIR):
        return models

    for run_name in sorted(os.listdir(SAVED_MODELS_DIR)):
        run_dir = os.path.join(SAVED_MODELS_DIR, run_name)
        model_path = os.path.join(run_dir, "model.pt")
        config_path = os.path.join(run_dir, "config.json")

        if os.path.isfile(model_path) and os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            arch_type = cfg.get("arch", {}).get("type", "unknown")
            num_reg = cfg.get("arch", {}).get("num_registers", 0)

            if arch_type == "vit_rgts":
                models[run_name] = {
                    "path": model_path,
                    "num_registers": num_reg,
                    "label": f"{run_name} (reg={num_reg})",
                    "config": cfg,
                }
    return models


def discover_sample_images():
    """Find sample images."""
    images = []
    if os.path.isdir(SAMPLE_IMAGES_DIR):
        for f in sorted(os.listdir(SAMPLE_IMAGES_DIR)):
            ext = os.path.splitext(f)[1].lower()
            if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
                images.append(os.path.join(SAMPLE_IMAGES_DIR, f))
    return images


# ═════════════════════════════════════════════════════════════════════════════
# Ground Truth
# ═════════════════════════════════════════════════════════════════════════════

def load_gt():
    """Load ground truth from sample images directory."""
    gt = {}
    gt_file = os.path.join(SAMPLE_IMAGES_DIR, "gt.txt")
    if os.path.isfile(gt_file):
        with open(gt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    gt[parts[0]] = parts[1]
    return gt


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="ViT Attention Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .stApp { background-color: #0d1520; }
    .stSidebar { background-color: #080c12; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stMarkdown p { color: #94a3b8; }
    .metric-card {
        background: #0f1a2e;
        border: 1px solid #1e2d40;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 20px; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; }
    .reg-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    ## 🔍 ViT Attention Explorer
    **IAM HTR · Runs 50–54 · Register Token Impact · Attention Maps · Rollout · Grad-CAM**
    """)

    # ── Discover resources ────────────────────────────────────────────────
    available_models = discover_models()
    sample_images = discover_sample_images()
    gt_dict = load_gt()
    _, num_classes, i2c = load_classes()

    if not available_models:
        st.error("No ViT-RGTS models found in saved_models/experiments/")
        st.info("Expected model directories with model.pt and config.json")
        st.stop()

    # ══════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════

    with st.sidebar:
        st.markdown("### 📸 Image Input")

        image_source = st.radio(
            "Source", ["Upload Image", "Sample Images"],
            horizontal=True, label_visibility="collapsed"
        )

        img_np = None
        img_name = ""

        if image_source == "Upload Image":
            uploaded = st.file_uploader(
                "Upload a handwritten line image",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                help="IAM handwritten line image (grayscale preferred)"
            )
            if uploaded:
                img_np = load_and_preprocess(uploaded)
                img_name = Path(uploaded.name).stem
        else:
            if sample_images:
                selected_sample = st.selectbox(
                    "Select sample image",
                    sample_images,
                    format_func=lambda x: os.path.basename(x),
                )
                if selected_sample:
                    img_np = np.array(Image.open(selected_sample).convert("L")).astype(np.float64)
                    img_name = Path(selected_sample).stem
            else:
                st.warning("No sample images found.")

        if img_np is not None:
            st.image(img_np, caption=f"{img_name} ({img_np.shape[1]}×{img_np.shape[0]})",
                     use_container_width=True, clamp=True)
            gt_text = gt_dict.get(img_name, "")
            if gt_text:
                st.markdown(f"**GT:** `{gt_text}`")

        st.markdown("---")
        st.markdown("### 🧠 Model Selection")

        # Model selection with custom model path option
        use_custom_path = st.checkbox("Use custom model path", value=False)

        if use_custom_path:
            custom_model_path = st.text_input(
                "Model path (.pt file)",
                placeholder="/path/to/model.pt"
            )
            custom_num_registers = st.number_input(
                "Number of register tokens", min_value=0, max_value=64, value=4, step=1
            )
            selected_runs = []
            custom_model = None
            if custom_model_path and os.path.isfile(custom_model_path):
                custom_model = {
                    "path": custom_model_path,
                    "num_registers": custom_num_registers,
                    "label": f"Custom (reg={custom_num_registers})",
                }
            elif custom_model_path:
                st.warning("File not found")
        else:
            custom_model = None
            selected_runs = st.multiselect(
                "Select model runs",
                list(available_models.keys()),
                default=["run_50"] if "run_50" in available_models else [],
                format_func=lambda x: available_models[x]["label"],
            )

        st.markdown("---")
        st.markdown("### ⚙️ Visualization Settings")

        vis_modes = st.multiselect(
            "Visualization types",
            ["Attention Maps", "Attention Rollout", "Grad-CAM"],
            default=["Attention Maps", "Attention Rollout", "Grad-CAM"],
        )

        alpha = st.slider("Overlay opacity", 0.1, 1.0, 0.55, 0.05)

        if "Attention Maps" in vis_modes:
            layer_select = st.selectbox(
                "Attention layer",
                ["All Layers"] + [f"Layer {i + 1}" for i in range(6)],
                index=0,
            )
        else:
            layer_select = "Layer 3"

        show_attn_matrix = st.checkbox("Show raw attention matrix", value=False)

        st.markdown("---")
        run_button = st.button(
            "▶ Run Visualization",
            type="primary",
            use_container_width=True,
            disabled=img_np is None,
        )

    # ══════════════════════════════════════════════════════════════════════
    # MAIN CONTENT
    # ══════════════════════════════════════════════════════════════════════

    if img_np is None:
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px; color: #1e3a4a;">
            <div style="font-size: 48px; margin-bottom: 16px;">⬆</div>
            <div style="font-size: 16px; color: #334155; margin-bottom: 8px;">
                Load a handwritten line image
            </div>
            <div style="font-size: 13px; color: #1e3a4a;">
                Upload an image or select a sample from the sidebar
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not run_button and "results" not in st.session_state:
        st.info("📌 Image loaded. Click **▶ Run Visualization** to compute.")
        return

    # ── Build list of models to process ───────────────────────────────────
    models_to_run = []
    if use_custom_path and custom_model is not None:
        models_to_run.append(custom_model)
    elif not use_custom_path:
        for rn in selected_runs:
            models_to_run.append(available_models[rn])

    if run_button and not models_to_run:
        st.warning("Please select at least one model run.")
        return

    # ── Run inference ─────────────────────────────────────────────────────
    if run_button:
        device = get_device()
        all_results = {}
        progress = st.progress(0, text="Loading models...")

        img_tensor = preprocess_image(img_np)

        for mi, model_info in enumerate(models_to_run):
            progress.progress(
                (mi) / len(models_to_run),
                text=f"Processing {model_info['label']}..."
            )

            net = load_model(model_info["path"], model_info["num_registers"], num_classes)
            result = {}

            # Attention Maps + Rollout
            if "Attention Maps" in vis_modes or "Attention Rollout" in vis_modes:
                attn_result = extract_attention_maps(net, img_tensor, device)
                result["attn"] = attn_result

                pred_text = ctc_decode(attn_result["logits"], i2c)
                result["pred_text"] = pred_text

                if "Attention Rollout" in vis_modes:
                    rollout = attention_rollout(attn_result["attn_maps"])
                    result["rollout"] = rollout

            # Grad-CAM (requires gradients)
            if "Grad-CAM" in vis_modes:
                # Need to temporarily enable gradients
                with torch.enable_grad():
                    gc_result = compute_gradcam(net, img_tensor.clone(), device)
                result["gradcam"] = gc_result

                if "pred_text" not in result:
                    pred_text = ctc_decode(gc_result["logits"], i2c)
                    result["pred_text"] = pred_text

            result["model_info"] = model_info
            all_results[model_info["label"]] = result

        progress.progress(1.0, text="Done ✓")
        st.session_state["results"] = all_results
        st.session_state["img_np"] = img_np
        st.session_state["img_name"] = img_name
        st.session_state["vis_modes"] = vis_modes
        st.session_state["alpha"] = alpha
        st.session_state["layer_select"] = layer_select
        st.session_state["show_attn_matrix"] = show_attn_matrix

    # ── Display results ───────────────────────────────────────────────────
    if "results" not in st.session_state:
        return

    all_results = st.session_state["results"]
    img_np = st.session_state.get("img_np", img_np)
    img_name = st.session_state.get("img_name", img_name)
    vis_modes = st.session_state.get("vis_modes", vis_modes)
    alpha = st.session_state.get("alpha", alpha)
    layer_select = st.session_state.get("layer_select", layer_select)
    show_attn_matrix = st.session_state.get("show_attn_matrix", False)

    H_img, W_img = img_np.shape[:2]

    # ── Per-model results ─────────────────────────────────────────────────
    for model_label, result in all_results.items():
        model_info = result["model_info"]
        num_reg = model_info["num_registers"]
        reg_note = REG_NOTES.get(num_reg, f"{num_reg} register tokens")

        st.markdown(f"""
        ### 🏷️ {model_label}
        <div class="metric-card">
            <span class="reg-badge" style="background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff40;">
                {num_reg} registers
            </span>
            &nbsp;
            <span style="color: #475569; font-size: 12px;">{reg_note}</span>
        </div>
        """, unsafe_allow_html=True)

        # Prediction
        pred_text = result.get("pred_text", "")
        gt_text = gt_dict.get(img_name, "")

        col_pred, col_gt = st.columns(2)
        with col_pred:
            st.markdown(f"**Prediction:** `{pred_text}`")
        with col_gt:
            if gt_text:
                st.markdown(f"**Ground Truth:** `{gt_text}`")

        # ── TAB LAYOUT ────────────────────────────────────────────────────
        tab_names = []
        if "Attention Maps" in vis_modes:
            tab_names.append("🗺 Attention Maps")
        if "Attention Rollout" in vis_modes:
            tab_names.append("🔁 Attention Rollout")
        if "Grad-CAM" in vis_modes:
            tab_names.append("🌡 Grad-CAM")

        if not tab_names:
            continue

        tabs = st.tabs(tab_names)
        tab_idx = 0

        # ── ATTENTION MAPS ────────────────────────────────────────────────
        if "Attention Maps" in vis_modes:
            with tabs[tab_idx]:
                attn_data = result.get("attn")
                if attn_data:
                    attn_maps = attn_data["attn_maps"]
                    grid_size = attn_data["grid_size"]
                    nr = attn_data["num_reg"]
                    n_layers = len(attn_maps)
                    n_heads = attn_maps[0].shape[1]

                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">Layers: {n_layers}</span> &nbsp;|&nbsp;
                        <span class="metric-label">Heads: {n_heads}</span> &nbsp;|&nbsp;
                        <span class="metric-label">Grid: {grid_size[0]}×{grid_size[1]}</span> &nbsp;|&nbsp;
                        <span class="metric-label">Registers: {nr}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if layer_select == "All Layers":
                        layers_to_show = list(range(n_layers))
                    else:
                        layer_num = int(layer_select.split()[-1]) - 1
                        layers_to_show = [layer_num]

                    for li in layers_to_show:
                        st.markdown(f"#### Layer {li + 1}")
                        fig = plot_attention_heads(
                            attn_maps[li], nr, grid_size, img_np, li, alpha
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                        # Show raw attention matrix
                        if show_attn_matrix:
                            attn_avg = attn_maps[li][0].mean(dim=0).numpy()
                            fig_mat = plot_attention_matrix(
                                attn_avg, nr,
                                title=f"Layer {li + 1} — Avg Head Attention Matrix"
                            )
                            st.pyplot(fig_mat, use_container_width=True)
                            plt.close(fig_mat)

            tab_idx += 1

        # ── ATTENTION ROLLOUT ─────────────────────────────────────────────
        if "Attention Rollout" in vis_modes:
            with tabs[tab_idx]:
                rollout = result.get("rollout")
                attn_data = result.get("attn")
                if rollout is not None and attn_data is not None:
                    grid_size = attn_data["grid_size"]
                    nr = attn_data["num_reg"]

                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">Method: ∏(0.5·A_l + 0.5·I) across {len(attn_data['attn_maps'])} layers</span> &nbsp;|&nbsp;
                        <span class="metric-label">Register columns zeroed: {nr} tokens</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Spatial heatmap from rollout
                    spatial = get_patch_attention_from_rollout(rollout, nr, grid_size)
                    heatmap = upscale_patch_aligned(spatial, H_img, W_img)

                    # Normalization
                    p01, p99 = np.percentile(heatmap, 1), np.percentile(heatmap, 99)
                    if p99 - p01 < 1e-8:
                        p01, p99 = heatmap.min(), heatmap.max()
                    heatmap_scaled = np.clip((heatmap - p01) / (p99 - p01 + 1e-8), 0, 1)

                    fig = plot_heatmap_overlay(
                        img_np, heatmap_scaled,
                        title="Attention Rollout",
                        alpha=alpha,
                        cmap="inferno",
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # Show rollout matrix
                    if show_attn_matrix:
                        fig_mat = plot_attention_matrix(
                            rollout, nr,
                            title="Full Attention Rollout Matrix"
                        )
                        st.pyplot(fig_mat, use_container_width=True)
                        plt.close(fig_mat)

            tab_idx += 1

        # ── GRAD-CAM ─────────────────────────────────────────────────────
        if "Grad-CAM" in vis_modes:
            with tabs[tab_idx]:
                gc_data = result.get("gradcam")
                if gc_data:
                    importance = gc_data["importance"]
                    gc_grid = gc_data["grid_size"]
                    Hp, Wp = gc_grid
                    T = importance.shape[0]

                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">Method: Selvaraju et al. 2017</span> &nbsp;|&nbsp;
                        <span class="metric-label">importance = ReLU(Σ_d grad_d · act_d)</span> &nbsp;|&nbsp;
                        <span class="metric-label">Tokens: {T}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if T == Hp * Wp:
                        heat_grid = importance.reshape(Hp, Wp)
                        heatmap = upscale_patch_aligned(heat_grid, H_img, W_img)

                        p01, p99 = np.percentile(heatmap, 1), np.percentile(heatmap, 99)
                        if p99 - p01 < 1e-8:
                            p01, p99 = heatmap.min(), heatmap.max()
                        heatmap_scaled = np.clip(
                            (heatmap - p01) / (p99 - p01 + 1e-8), 0, 1
                        )

                        fig = plot_heatmap_overlay(
                            img_np, heatmap_scaled,
                            title="Grad-CAM",
                            alpha=alpha,
                            cmap="inferno",
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.warning(
                            f"Token count T={T} doesn't match grid "
                            f"Hp×Wp={Hp}×{Wp}={Hp * Wp}. Showing 1D importance."
                        )
                        fig, ax = plt.subplots(1, 1, figsize=(14, 3), facecolor="#0d1520")
                        ax.set_facecolor("#080c12")
                        ax.bar(range(T), importance, color="#ff6b35", width=1.0)
                        ax.set_xlabel("Token position", color="#94a3b8", fontsize=9)
                        ax.set_ylabel("Importance", color="#94a3b8", fontsize=9)
                        ax.set_title("Grad-CAM Token Importance", color="#e2e8f0", fontsize=11)
                        ax.tick_params(colors="#475569", labelsize=7)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            tab_idx += 1

        st.markdown("---")

    # ── Theory Section ────────────────────────────────────────────────────
    with st.expander("📖 Theory — Methods Comparison", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            #### 🗺 Attention Maps
            | Property | Value |
            |----------|-------|
            | **Source** | softmax(QKᵀ/√d) per layer |
            | **Depth** | ❌ layer-by-layer only |
            | **Task-aware** | ❌ attention ≠ importance |
            | **Registers** | Absorb diffuse/noise attn → heads specialise |
            """)

        with col2:
            st.markdown("""
            #### 🔁 Attention Rollout
            | Property | Value |
            |----------|-------|
            | **Source** | ∏ layers (A_l + I) |
            | **Depth** | ✅ full stack propagation |
            | **Task-aware** | ⚠ CLS-centric only |
            | **Registers** | Columns zeroed → purer patch→CLS flow |
            """)

        with col3:
            st.markdown("""
            #### 🌡 Grad-CAM
            | Property | Value |
            |----------|-------|
            | **Source** | ∂output/∂activations |
            | **Depth** | ✅ full network |
            | **Task-aware** | ✅ gradient-driven |
            | **Registers** | Minimal direct effect (acts on patch tokens) |
            """)

    # ── Register Impact Summary ───────────────────────────────────────────
    if len(all_results) > 1:
        with st.expander("📊 Register Impact Comparison", expanded=False):
            st.markdown("""
            **Key observations across register counts:**

            | Registers | Attention Pattern | Head Specialisation | Noise |
            |-----------|-------------------|---------------------|-------|
            | 0 | Diffuse, scattered | Heads overlap heavily | High |
            | 2 | Slight improvement | Marginal separation | Medium |
            | 4 | Clearer focus | Heads begin specialising | Low |
            | 8 | Distinct patterns | Clear role per head | Very low |
            | 16 | Highly focused | Maximum specialisation | Minimal |

            Register tokens act as **"attention sinks"** — they absorb the diffuse,
            low-information attention that would otherwise spread across all patch tokens.
            This frees the remaining heads to develop specialised roles (edge detection,
            stroke following, spacing, etc.).
            """)


if __name__ == "__main__":
    main()
