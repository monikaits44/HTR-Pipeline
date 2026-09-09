"""
Shared utilities for ViT-RGTS attention visualization scripts.

Used by:
  char_gradcam.py   — GradCAM on CNN stem last layer
  attn_rollout.py   — Attention Rollout through all ViT layers
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import warnings

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models import HTRNet
from utils.preprocessing import load_image, preprocess

# ── Constants ─────────────────────────────────────────────────────

DEVICE    = 'cpu'
IMG_H     = 128
IMG_W     = 1024
SCALE     = IMG_W / 128.0          # pixels per ViT patch column

EXP       = ROOT / 'saved_models' / 'experiments'
OUT       = ROOT / 'outputs' / 'viz_gradcam'
TESTDIR   = ROOT / 'data' / 'IAM' / 'processed_lines' / 'test'
SAMPLEDIR = ROOT / 'notebook' / 'sample_images'

# Register-count → run directory mapping
RUNS = {0: 'run_76', 4: 'run_84', 8: 'run_63', 16: 'run_71'}

# 4 IAM test words (~5 chars each, distinct writers)
IMAGES = [
    ('a01-038-12.png', 'talks.', SAMPLEDIR),
    ('d04-021-07.png', 'tell!',  TESTDIR),
    ('d04-086-06.png', 'back.',  TESTDIR),
    ('g03-000-09.png', 'ways.',  TESTDIR),
]

# Character set (index 0 = CTC blank, so charset[i-1] = class i)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    charset = list(np.load(str(EXP / 'classes.npy'), allow_pickle=True))

OUT.mkdir(parents=True, exist_ok=True)


# ── Model helpers ─────────────────────────────────────────────────

def load_model(run_id: str) -> HTRNet:
    cfg = json.load(open(EXP / run_id / 'config.json'))
    m   = HTRNet(SimpleNamespace(**cfg['arch']), len(charset) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.load_state_dict(
            torch.load(str(EXP / run_id / 'model.pt'),
                       map_location=DEVICE, weights_only=False),
            strict=False)
    return m.to(DEVICE).eval()


def prep(img_file: str, img_dir: Path):
    """Load and preprocess an image. Returns (tensor [1,1,H,W], numpy [H,W])."""
    raw = load_image(str(img_dir / img_file))
    img = preprocess(raw, (IMG_H, IMG_W))   # float32, inverted (0=bg, high=ink)
    t   = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return t, img


def ctc_decode(logits):
    """Greedy CTC decode. Returns (text, [(char, time_step), ...])."""
    ids  = logits[:, 0, :].softmax(-1).argmax(-1).cpu().numpy()
    out, prev = [], -1
    for t, idx in enumerate(ids):
        if idx != 0 and idx != prev:
            ci = idx - 1
            if 0 <= ci < len(charset):
                out.append((str(charset[ci]), t))
        prev = idx
    while out and out[0][0]  == ' ': out.pop(0)
    while out and out[-1][0] == ' ': out.pop()
    return ''.join(c for c, _ in out), out


# ── Image cropping helpers ────────────────────────────────────────

def char_window(char_peaks, scale=SCALE):
    """
    Compute a uniform half-width for per-character crop windows.

    half_w  = mean inter-character spacing in pixels (≥ 16 px)
    Returns (half_w, [center_px per char])
    """
    px = [t * scale for _, t in char_peaks]
    if len(px) < 2:
        return int(scale * 3), px
    mean_sp = (px[-1] - px[0]) / (len(px) - 1)
    return max(int(mean_sp), 16), px


def height_bounds(img):
    """
    Tight vertical bounds of the ink region.

    img is inverted (0=white background, high value=ink).
    Adds 4-pixel margin around the detected ink rows.
    """
    row_mean = img.mean(axis=1)
    ink_rows = np.where(row_mean > row_mean.max() * 0.05)[0]
    if len(ink_rows) == 0:
        return 0, img.shape[0]
    return max(0, ink_rows[0] - 4), min(img.shape[0], ink_rows[-1] + 4)


def uniform_win(img_np, cx_px, half_w, y0, y1):
    """
    Extract a 2*half_w × (y1-y0) crop centred on cx_px.
    Zero-pads if the window extends beyond the image boundary.
    """
    lx   = int(cx_px - half_w)
    rx   = lx + 2 * half_w
    lx_c = max(0, lx);  rx_c = min(IMG_W, rx)
    win  = img_np[y0:y1, lx_c:rx_c]
    pl   = lx_c - lx;   pr = rx - rx_c
    if pl > 0 or pr > 0:
        win = np.pad(win, ((0, 0), (pl, pr)), constant_values=0)
    return win


def crop_cam(cam_full, cx_px, half_w):
    """
    Crop a full-width GradCAM map [H, IMG_W] to the character window.
    Edge-pads if the window extends beyond the image boundary.
    """
    lx = max(0, int(cx_px - half_w))
    rx = lx + 2 * half_w
    if rx > IMG_W:
        rx = IMG_W
        lx = max(0, rx - 2 * half_w)
    win = cam_full[:, lx:rx]
    pl  = 2 * half_w - win.shape[1]
    if pl > 0:
        win = np.pad(win, ((0, 0), (0, pl)), mode='edge')
    return win
