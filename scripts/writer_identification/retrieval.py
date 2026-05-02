"""
Distance computation, retrieval, and evaluation metrics for writer identification.

Implements the distance and evaluation framework from:
  Raven et al., "Interpretable Writer Recognition via Vectors of Locally
  Aggregated Characters" — Eq. 8–9, Sec. 4.1

Supports:
    - Global cosine distance on full descriptors
    - Character-wise interpretable distance (per-character cosine, Eq. 8–9)
    - Character-selected distance (only A* characters)
    - Retrieval: mAP, Top-1, Top-5, EER
"""

import numpy as np
from collections import defaultdict


# --------------------------------------------------------------------------- #
# Distance functions
# --------------------------------------------------------------------------- #

def cosine_distance_matrix(descriptors):
    """
    Pairwise cosine distance matrix.

    Parameters
    ----------
    descriptors : [N, D] numpy, assumed L2-normalised

    Returns
    -------
    dist : [N, N] numpy, 0 = identical, 2 = opposite
    """
    sim = descriptors @ descriptors.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def character_wise_distance(vlac_a, vlac_b, counts_a, counts_b):
    """
    Interpretable per-character cosine distance between two VLAC matrices.

    Only considers characters present in both documents.

    Parameters
    ----------
    vlac_a, vlac_b : [C, D]
    counts_a, counts_b : [C]

    Returns
    -------
    char_distances : dict  char_idx → cosine distance (0–2)
    mean_distance  : float — mean over shared characters
    """
    shared = (counts_a > 0) & (counts_b > 0)
    if not shared.any():
        return {}, float("nan")

    char_distances = {}
    dists = []
    for i in np.where(shared)[0]:
        sim = float(vlac_a[i] @ vlac_b[i])
        d = 1.0 - np.clip(sim, -1.0, 1.0)
        char_distances[int(i)] = d
        dists.append(d)

    return char_distances, float(np.mean(dists))


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #

def compute_retrieval_metrics(dist_matrix, writer_ids):
    """
    Compute mAP, Top-1, and Top-5 for writer retrieval.

    Each row is a query; all other rows with the same writer_id are relevant.

    Parameters
    ----------
    dist_matrix : [N, N]
    writer_ids : list[str] of length N

    Returns
    -------
    dict with 'mAP', 'top1', 'top5', 'per_query' details
    """
    N = len(writer_ids)

    # Build relevance mapping: for each sample, which other samples share
    # the same writer?
    wid_to_indices = defaultdict(list)
    for i, wid in enumerate(writer_ids):
        wid_to_indices[wid].append(i)

    aps = []
    top1_correct = 0
    top5_correct = 0
    per_query = []

    for q in range(N):
        wid = writer_ids[q]
        relevant = set(wid_to_indices[wid]) - {q}

        if not relevant:
            # Single-sample writer — skip (can't evaluate retrieval)
            continue

        # Sort gallery by distance (ascending), excluding self
        dists = dist_matrix[q].copy()
        dists[q] = np.inf  # exclude self
        ranking = np.argsort(dists)

        # Top-1
        if ranking[0] in relevant:
            top1_correct += 1

        # Top-5
        if relevant & set(ranking[:5].tolist()):
            top5_correct += 1

        # Average Precision
        num_relevant = len(relevant)
        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(ranking):
            if idx in relevant:
                hits += 1
                precision_sum += hits / (rank + 1)
        ap = precision_sum / num_relevant
        aps.append(ap)

        per_query.append({
            "query_idx": q,
            "writer_id": wid,
            "ap": ap,
            "top1_hit": ranking[0] in relevant,
        })

    n_queries = len(aps)
    if n_queries == 0:
        return {"mAP": 0.0, "top1": 0.0, "top5": 0.0, "n_queries": 0, "per_query": []}

    return {
        "mAP": float(np.mean(aps)),
        "top1": top1_correct / n_queries,
        "top5": top5_correct / n_queries,
        "n_queries": n_queries,
        "per_query": per_query,
    }


def vlac_distance_matrix(vlac_list, counts_list, selected_chars=None):
    """
    Compute pairwise character-wise cosine distance (Eq. 8–9 from VLAC paper).

    Uses only characters in selected_chars (A*) if provided.
    Vectorized implementation for efficiency.

    Parameters
    ----------
    vlac_list   : list of [C, D] arrays
    counts_list : list of [C] arrays
    selected_chars : list[int] or None — indices of A* characters

    Returns
    -------
    dist : [N, N] numpy
    """
    N = len(vlac_list)
    C = vlac_list[0].shape[0]

    # Stack into [N, C, D] and [N, C]
    V = np.stack(vlac_list)        # [N, C, D]
    Counts = np.stack(counts_list) # [N, C]

    chars = selected_chars if selected_chars is not None else list(range(C))
    if not chars:
        return np.ones((N, N), dtype=np.float32)

    # Subset to selected characters: [N, |A*|, D]
    V_sel = V[:, chars, :]
    C_sel = Counts[:, chars]  # [N, |A*|]

    # Presence mask: [N, |A*|]
    present = C_sel > 0

    # Pairwise cosine similarity per character using einsum: [N, N, |A*|]
    sim = np.einsum('icd,jcd->ijc', V_sel, V_sel)
    sim = np.clip(sim, -1.0, 1.0)

    # Shared character mask: both present [N, N, |A*|]
    shared = present[:, None, :] & present[None, :, :]

    # Distance = 1 - sim, averaged over shared characters
    char_dist = 1.0 - sim  # [N, N, |A*|]
    char_dist[~shared] = 0.0

    n_shared = shared.sum(axis=2).astype(np.float32)  # [N, N]
    n_shared = np.maximum(n_shared, 1.0)

    dist = char_dist.sum(axis=2) / n_shared
    # Where no shared chars, set distance to 1
    dist[shared.sum(axis=2) == 0] = 1.0

    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)
