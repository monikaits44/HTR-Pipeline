"""
VLAC — Vectors of Locally Aggregated Characters.

Implements the VLAC encoding from:
  Raven et al., "Interpretable Writer Recognition via Vectors of Locally
  Aggregated Characters" (2025)

Key concepts from the paper:
  1. Global character prototypes μ_c  (Eq. 4) — mean feature per char across
     the training set.
  2. Residual aggregation Φ_{v,c} = L2( Σ (x - μ_c) )  (Eq. 5) — captures
     document-specific deviations from the prototype.
  3. Character selection A* — keep only chars whose single-char mAP exceeds
     τ * max(mAP_c).  Blank and space are excluded.
  4. Distance = mean cosine over character-wise representations  (Eq. 8–9).

Our extension:
  - Register tokens from ViT-RGTS are appended as global style features,
    bridging the "Vision Transformers Need Registers" paper with VLAC.

Architecture bridge:
    ViT-RGTS backbone → forward_explain()
        ├── patch tokens [T, B, D]  →  per-CTC-position features
        ├── register tokens [B, R, D]  →  global style features
        └── attention maps [L × (B, H, S, S)]
    CTC decode → character labels + positions
    VLAC aggregation → Φ_{v,c} for each character class c  →  descriptor [C, D]
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.postprocessing.attention_viz.core import (
    load_model,
    extract_attention,
    ctc_decode,
    load_htr_image,
)


# --------------------------------------------------------------------------- #
# Feature extraction from a single image
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_line_features(net, image_tensor, i2c, device="cpu"):
    """
    Run a single line image through the model and return character-level
    patch-token features plus register-token embeddings.

    Parameters
    ----------
    net : HTRNet (eval mode)
    image_tensor : [1, 1, H, W]
    i2c : dict  index → character
    device : str

    Returns
    -------
    dict with keys:
        chars       : str   — decoded text
        positions   : list[int] — CTC output positions
        char_feats  : [N, D] numpy — patch-token feature at each char position
        reg_feats   : [R, D] numpy or None — register token embeddings
        char_labels : list[str] — per-position character label
    """
    image_tensor = image_tensor.to(device)

    # Forward explain to get both token embeddings and logits
    result = net.forward_explain(image_tensor)

    if net.arch_type == "vit_rgts":
        logits, reg_tokens, attn_maps, token_norms, grid = result
    else:
        logits, reg_tokens, attn_maps, token_norms, grid = result

    # CTC decode
    text, positions = ctc_decode(logits.cpu(), i2c)

    if not positions:
        return {
            "chars": text,
            "positions": positions,
            "char_feats": np.zeros((0, net.backbone.embed_dim), dtype=np.float32),
            "reg_feats": reg_tokens[0].cpu().numpy() if reg_tokens is not None else None,
            "char_labels": [],
        }

    # Extract patch-token embeddings at CTC-decoded positions.
    # After forward_explain the backbone stores final-layer tokens internally;
    # we re-derive them from the seq_tokens returned by forward_explain.
    # seq_tokens (before CTC head) = result's first element reshaped.
    # But forward_explain applies the CTC head already, so we need the raw
    # backbone output.  We do a second lightweight pass through backbone only.
    backbone_out = net.backbone.forward_explain(image_tensor)
    seq_tokens = backbone_out[0]  # [T, B, D]
    reg_out = backbone_out[1]     # [B, R, D]

    # seq_tokens is [T, 1, D] — pick positions
    seq_np = seq_tokens[:, 0, :].cpu().numpy()  # [T, D]
    char_feats = seq_np[positions]               # [N, D]

    # L2-normalise each feature vector
    norms = np.linalg.norm(char_feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    char_feats = char_feats / norms

    reg_feats = None
    if reg_out is not None and reg_out.shape[1] > 0:
        reg_feats = reg_out[0].cpu().numpy()  # [R, D]

    char_labels = list(text)

    return {
        "chars": text,
        "positions": positions,
        "char_feats": char_feats.astype(np.float32),
        "reg_feats": reg_feats,
        "char_labels": char_labels,
    }


# --------------------------------------------------------------------------- #
# VLAC aggregation  (Raven et al. Eq. 3–7)
# --------------------------------------------------------------------------- #

# Characters that are never useful for writer discrimination (paper Sec. 3.3)
_EXCLUDED_CHARS = {" ", "", "\t", "\n"}


def compute_global_prototypes(all_line_features, charset):
    """
    Compute global character prototypes μ_c  (Eq. 4).

    Averages all feature vectors annotated as character c across the entire
    dataset.

    Parameters
    ----------
    all_line_features : list of dicts from extract_line_features
    charset : list[str]

    Returns
    -------
    prototypes : [C, D] numpy — one prototype per character class
    """
    char2idx = {c: i for i, c in enumerate(charset)}
    D = None
    sums = None
    counts = None

    for lf in all_line_features:
        feats = lf["char_feats"]
        if feats.shape[0] == 0:
            continue
        if D is None:
            D = feats.shape[1]
            sums = np.zeros((len(charset), D), dtype=np.float64)
            counts = np.zeros(len(charset), dtype=np.int64)
        for feat, label in zip(feats, lf["char_labels"]):
            if label in char2idx and label not in _EXCLUDED_CHARS:
                idx = char2idx[label]
                sums[idx] += feat
                counts[idx] += 1

    if D is None:
        D = 256
        return np.zeros((len(charset), D), dtype=np.float32)

    mask = counts > 0
    sums[mask] /= counts[mask, None]
    return sums.astype(np.float32)


def vlac_aggregate(char_feats, char_labels, charset, prototypes=None):
    """
    Aggregate per-character features into a VLAC descriptor using residual
    encoding (Eq. 5 from the VLAC paper).

    For each character class c, aggregate the residuals (x - μ_c) of all
    occurrences and L2-normalise the result.  When prototypes are None,
    falls back to simple mean pooling (no residuals).

    Parameters
    ----------
    char_feats  : [N, D] numpy
    char_labels : list[str] of length N
    charset     : list[str] — ordered character vocabulary
    prototypes  : [C, D] numpy or None — global character prototypes μ_c

    Returns
    -------
    vlac : [C, D] numpy — one row per character class (L2-normalised)
    counts : [C] numpy int — occurrence count per class
    """
    if len(char_feats) == 0:
        D = 256  # fallback
        return np.zeros((len(charset), D), dtype=np.float32), np.zeros(len(charset), dtype=np.int32)

    D = char_feats.shape[1]
    C = len(charset)
    char2idx = {c: i for i, c in enumerate(charset)}

    vlac = np.zeros((C, D), dtype=np.float64)
    counts = np.zeros(C, dtype=np.int32)

    for feat, label in zip(char_feats, char_labels):
        if label in char2idx and label not in _EXCLUDED_CHARS:
            idx = char2idx[label]
            if prototypes is not None:
                # Residual aggregation (Eq. 5): sum of (x - μ_c)
                vlac[idx] += feat - prototypes[idx]
            else:
                vlac[idx] += feat
            counts[idx] += 1

    # L2-normalise each character representation (Eq. 5)
    mask = counts > 0
    row_norms = np.linalg.norm(vlac, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-8)
    vlac = vlac / row_norms

    # Zero out classes with no occurrences
    vlac[~mask] = 0.0

    return vlac.astype(np.float32), counts


def vlac_from_lines(line_features_list, charset, prototypes=None):
    """
    Aggregate VLAC across multiple lines (same writer/document).

    Parameters
    ----------
    line_features_list : list of dicts from extract_line_features
    charset : list[str]
    prototypes : [C, D] or None — global character prototypes

    Returns
    -------
    vlac : [C, D]
    counts : [C]
    reg_mean : [R, D] or None — mean register embedding across lines
    """
    all_feats = []
    all_labels = []
    reg_list = []

    for lf in line_features_list:
        if lf["char_feats"].shape[0] > 0:
            all_feats.append(lf["char_feats"])
            all_labels.extend(lf["char_labels"])
        if lf["reg_feats"] is not None:
            reg_list.append(lf["reg_feats"])

    if all_feats:
        all_feats = np.concatenate(all_feats, axis=0)
    else:
        all_feats = np.zeros((0, 256), dtype=np.float32)

    vlac, counts = vlac_aggregate(all_feats, all_labels, charset, prototypes)

    reg_mean = None
    if reg_list:
        reg_mean = np.mean(reg_list, axis=0)  # [R, D]

    return vlac, counts, reg_mean


# --------------------------------------------------------------------------- #
# Character selection  (Raven et al. Eq. 6–7)
# --------------------------------------------------------------------------- #

def select_characters(writer_vlacs, charset, tau=0.8):
    """
    Select discriminative characters via per-character retrieval mAP (Eq. 6).

    For each character c, compute mAP using only Φ_{v,c} as the descriptor.
    Keep characters where mAP_c >= τ * max(mAP).  Blank and space are
    always excluded.

    Parameters
    ----------
    writer_vlacs : dict  writer_id → (vlac [C,D], counts [C], reg_mean)
    charset      : list[str]
    tau          : float — relative threshold (default 0.8 from paper)

    Returns
    -------
    selected_indices : list[int] — indices into charset of selected chars
    char_maps : dict  char_idx → mAP_c
    """
    from scripts.writer_identification.retrieval import (
        cosine_distance_matrix,
        compute_retrieval_metrics,
    )

    wids = sorted(writer_vlacs.keys())
    C = len(charset)

    char_maps = {}
    for ci in range(C):
        if charset[ci] in _EXCLUDED_CHARS:
            continue

        # Check if at least 2 writers have this character
        has_char = [wid for wid in wids if writer_vlacs[wid][1][ci] > 0]
        if len(has_char) < 2:
            continue

        # Build single-character descriptors
        descs = []
        desc_wids = []
        for wid in wids:
            vlac_row = writer_vlacs[wid][0][ci]  # [D]
            if writer_vlacs[wid][1][ci] > 0:
                norm = np.linalg.norm(vlac_row)
                if norm > 1e-8:
                    descs.append(vlac_row / norm)
                    desc_wids.append(wid)

        if len(descs) < 2:
            continue

        desc_arr = np.stack(descs)
        dist = cosine_distance_matrix(desc_arr)
        metrics = compute_retrieval_metrics(dist, desc_wids)
        char_maps[ci] = metrics["mAP"]

    if not char_maps:
        # Fallback: use all non-excluded characters
        return [i for i, c in enumerate(charset) if c not in _EXCLUDED_CHARS], {}

    max_map = max(char_maps.values())
    threshold = tau * max_map
    selected = [ci for ci, m in char_maps.items() if m >= threshold]

    return sorted(selected), char_maps
