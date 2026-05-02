"""
Core utilities for attention visualization scripts.

Provides shared model loading, attention extraction, CTC decoding, and
heatmap rendering so that downstream visualization scripts stay DRY.

Usage:
    from core import load_model, extract_attention, ctc_decode, render_heatmap
"""

import sys, os, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from typing import List, Dict, Tuple, Optional

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models import HTRNet
from utils.preprocessing import load_image, preprocess


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_configs(*config_paths: str) -> OmegaConf:
    """Merge multiple YAML config files into one OmegaConf object."""
    cfg = OmegaConf.load(config_paths[0])
    for p in config_paths[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))
    return cfg


def load_model(
    cfg: OmegaConf,
    model_path: str,
    device: str = "cpu",
) -> Tuple[HTRNet, np.ndarray, Dict]:
    """
    Build HTRNet from *cfg*, load checkpoint from *model_path*.

    Returns
    -------
    net : HTRNet in eval mode, on *device*
    classes : numpy array of character classes
    i2c : dict mapping index -> character (0 = CTC blank)
    """
    # Load character classes
    data_path = cfg.data.path
    classes_path = os.path.join(data_path, "classes.npy")
    if os.path.exists(classes_path):
        classes = np.load(classes_path, allow_pickle=True)
    else:
        raise FileNotFoundError(
            f"Character classes not found at {classes_path}. "
            "Run training first to generate classes.npy."
        )

    nclasses = len(classes) + 1  # +1 for CTC blank
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    net = HTRNet(cfg.arch, nclasses)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    # Handle both raw state_dict and wrapped checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    net.load_state_dict(ckpt, strict=True)
    net.to(device).eval()

    return net, classes, i2c


def infer_num_registers(model_path: str) -> int:
    """Infer the number of register tokens from run directory config."""
    run_dir = os.path.dirname(model_path)
    config_json = os.path.join(run_dir, "config.json")
    if os.path.exists(config_json):
        import json
        with open(config_json) as f:
            run_cfg = json.load(f)
        return run_cfg.get("arch", {}).get("num_registers", 0)
    return 0


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention(
    net: HTRNet,
    image: torch.Tensor,
    device: str = "cpu",
) -> Dict:
    """
    Run forward_explain on a single image and return a structured dict.

    Parameters
    ----------
    net : HTRNet (eval mode)
    image : [1, 1, H, W]
    device : str

    Returns
    -------
    dict with keys:
        logits       : [T, 1, C]
        attn_maps    : List[L] of [1, H, S, S]  (on CPU)
        reg_tokens   : [1, R, D] or None
        token_norms  : [1, S]
        grid_size    : (Hp, Wp)
        num_registers: int
    """
    image = image.to(device)
    with torch.no_grad():
        result = net.forward_explain(image)

    # Unified unpacking across architecture types
    if net.arch_type == "vit_rgts":
        logits, reg_tokens, attn_maps, token_norms, grid = result
    elif net.arch_type == "torchvision_vit":
        logits, reg_tokens, attn_maps, token_norms, grid = result
    elif net.arch_type == "trocr":
        logits, _, attn_maps, token_norms, grid = result
        reg_tokens = None
    else:
        raise ValueError(f"forward_explain not supported for {net.arch_type}")

    num_reg = getattr(net.backbone, "num_registers", 0)

    return {
        "logits": logits.cpu(),
        "attn_maps": [a.cpu() if not a.is_cuda else a for a in attn_maps],
        "reg_tokens": reg_tokens.cpu() if reg_tokens is not None else None,
        "token_norms": token_norms.cpu() if token_norms is not None else None,
        "grid_size": grid,
        "num_registers": num_reg,
    }


# ---------------------------------------------------------------------------
# CTC decode
# ---------------------------------------------------------------------------

def ctc_decode(logits: torch.Tensor, i2c: Dict, blank_id: int = 0) -> Tuple[str, List[int]]:
    """
    Greedy CTC decode.

    Returns
    -------
    text : decoded string
    positions : list of CTC output positions (one per decoded char)
    """
    # logits: [T, 1, C]
    preds = logits[:, 0, :].argmax(dim=-1).cpu().numpy()  # [T]
    chars = []
    positions = []
    for t, idx in enumerate(preds):
        if idx == blank_id:
            continue
        if t > 0 and idx == preds[t - 1]:
            continue
        if idx in i2c:
            chars.append(i2c[idx])
            positions.append(t)
    return "".join(chars), positions


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_htr_image(
    image_path: str,
    height: int = 128,
    width: int = 1024,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess a single HTR line image.

    Returns
    -------
    tensor : [1, 1, H, W]  ready for the model
    raw    : [H, W] numpy (0-1, ink=1) for overlays
    """
    img = load_image(image_path)
    img = preprocess(img, (height, width))
    raw = img.copy()
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    return tensor, raw


def collect_image_paths(paths: List[str]) -> List[str]:
    """Expand a list of paths (files or dirs) into individual .png paths."""
    result = []
    for p in paths:
        if os.path.isdir(p):
            result.extend(sorted(glob.glob(os.path.join(p, "*.png"))))
        elif os.path.isfile(p):
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Attention processing helpers
# ---------------------------------------------------------------------------

def get_char_attention(
    attn_maps: List[torch.Tensor],
    positions: List[int],
    num_registers: int,
    layer: str = "last",
    head: Optional[int] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Extract per-character spatial attention vectors.

    Parameters
    ----------
    attn_maps : List[L] of [1, H, S, S]
    positions : CTC output positions (indices into patch dimension)
    num_registers : R
    layer : 'first', 'middle', 'last', or int
    head : None = mean across heads; int = specific head
    gamma : contrast exponent (>1 sharpens, <1 softens)

    Returns
    -------
    char_attn : [num_chars, num_patches] normalised attention per character
    """
    L = len(attn_maps)
    if layer == "last":
        lidx = L - 1
    elif layer == "first":
        lidx = 0
    elif layer == "middle":
        lidx = L // 2
    else:
        lidx = int(layer)

    attn = attn_maps[lidx][0]  # [H, S, S]

    if head is not None:
        attn = attn[head : head + 1]  # [1, S, S]

    attn = attn.mean(dim=0)  # [S, S]  — mean over heads

    R = num_registers
    num_patches = attn.shape[0] - R

    rows = []
    for pos in positions:
        # Row pos+R (CTC position offset by registers) → columns R: (attention on patches)
        row = attn[R + pos, R:].numpy().copy()  # [num_patches]
        # Min-max normalise per character
        mn, mx = row.min(), row.max()
        if mx - mn > 1e-8:
            row = (row - mn) / (mx - mn)
        row = np.power(row, gamma)
        rows.append(row)

    return np.array(rows)  # [num_chars, num_patches]


def attention_rollout(
    attn_maps: List[torch.Tensor],
    num_registers: int,
) -> np.ndarray:
    """
    Compute attention rollout (Abnar & Zuidema, 2020) across all layers.

    Returns
    -------
    rollout : [S, S] numpy array (accumulated attention from first to last layer)
    """
    result = None
    for attn in attn_maps:
        # attn: [1, H, S, S] — average over heads
        a = attn[0].mean(dim=0).numpy()  # [S, S]
        # Add residual connection (identity)
        a = 0.5 * a + 0.5 * np.eye(a.shape[0])
        # Re-normalise rows
        a = a / a.sum(axis=-1, keepdims=True)
        if result is None:
            result = a
        else:
            result = result @ a
    return result


def get_register_attention(
    attn_maps: List[torch.Tensor],
    num_registers: int,
    layer: str = "last",
    direction: str = "reg_to_patch",
) -> np.ndarray:
    """
    Extract register ↔ patch attention.

    direction: 'reg_to_patch' or 'patch_to_reg'

    Returns
    -------
    attn : [R, num_patches] or [num_patches, R]
    """
    L = len(attn_maps)
    if layer == "last":
        lidx = L - 1
    elif layer == "first":
        lidx = 0
    elif layer == "middle":
        lidx = L // 2
    else:
        lidx = int(layer)

    a = attn_maps[lidx][0].mean(dim=0).numpy()  # [S, S]
    R = num_registers

    if direction == "reg_to_patch":
        return a[:R, R:]  # [R, P]
    else:
        return a[R:, :R]  # [P, R]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_heatmap_on_image(
    heatmap_1d: np.ndarray,
    raw_image: np.ndarray,
    grid_w: int,
    alpha: float = 0.5,
    cmap_name: str = "inferno",
) -> np.ndarray:
    """
    Overlay a 1-D attention vector on a raw grayscale image.

    Parameters
    ----------
    heatmap_1d : [num_patches] normalised 0-1
    raw_image  : [H, W] grayscale 0-1
    grid_w     : number of horizontal patches
    alpha      : overlay opacity

    Returns
    -------
    overlay : [H, W, 3] uint8 RGB image
    """
    import matplotlib.pyplot as plt

    H, W = raw_image.shape
    # Truncate or pad heatmap to grid_w
    h = np.zeros(grid_w)
    n = min(len(heatmap_1d), grid_w)
    h[:n] = heatmap_1d[:n]

    # Upsample to image width (nearest neighbour — preserves patch boundaries)
    h_up = np.repeat(h, max(1, W // grid_w))
    if len(h_up) < W:
        h_up = np.concatenate([h_up, np.zeros(W - len(h_up))])
    h_up = h_up[:W]

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(h_up)[..., :3]  # [W, 3]
    heat_rgb = np.tile(heat_rgb, (H, 1, 1))  # [H, W, 3]

    # Grayscale to RGB
    bg = np.stack([1.0 - raw_image] * 3, axis=-1)  # ink=0 (white bg)

    blend = (1 - alpha) * bg + alpha * heat_rgb
    return (np.clip(blend, 0, 1) * 255).astype(np.uint8)


def pure_heatmap(
    heatmap_1d: np.ndarray,
    grid_w: int,
    cell_h: int = 80,
    target_w: int = 400,
    cmap_name: str = "inferno",
) -> np.ndarray:
    """
    Render a pure heatmap (no image background) as an RGB array.

    Returns [cell_h, target_w, 3] uint8.
    """
    import matplotlib.pyplot as plt

    h = np.zeros(grid_w)
    n = min(len(heatmap_1d), grid_w)
    h[:n] = heatmap_1d[:n]

    # Upsample
    h_up = np.repeat(h, max(1, target_w // grid_w))
    if len(h_up) < target_w:
        h_up = np.concatenate([h_up, np.zeros(target_w - len(h_up))])
    h_up = h_up[:target_w]

    cmap = plt.get_cmap(cmap_name)
    row_rgb = cmap(h_up)[..., :3]  # [W, 3]
    tile = np.tile(row_rgb, (cell_h, 1, 1))  # [cell_h, W, 3]
    return (tile * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_diagonality(positions: List[int], num_patches: int) -> float:
    """Spearman ρ between character index and CTC peak position."""
    if len(positions) < 2:
        return float("nan")
    from scipy.stats import spearmanr
    char_idx = list(range(len(positions)))
    rho, _ = spearmanr(char_idx, positions)
    return rho


def compute_entropy(attn_row: np.ndarray) -> float:
    """Shannon entropy of a single attention distribution (bits)."""
    p = attn_row + 1e-12
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def compute_peak_sharpness(attn_row: np.ndarray) -> float:
    """max / mean ratio — higher = more focused."""
    mean_val = attn_row.mean()
    if mean_val < 1e-12:
        return 0.0
    return float(attn_row.max() / mean_val)
