"""
Quantitative metrics for attention map quality analysis.
Used to answer: "Do attention maps become cleaner with registers?"

Metrics:
    - Attention entropy (lower = more focused)
    - Sparsity via Gini coefficient (higher = more concentrated)
    - Peak sharpness (higher = sharper peak)
    - Character localization accuracy (higher = better spatial alignment)
    - Cross-character overlap (lower = more distinct characters)
    - Register specialization (higher = more distinct register roles)
"""

import numpy as np
from typing import Dict, List, Optional


def attention_entropy(attn_vector: np.ndarray) -> float:
    """
    Shannon entropy of attention distribution.
    Lower = more focused = "cleaner".
    
    Args:
        attn_vector: [P] attention weights (sums to ~1 after softmax)
    Returns:
        Entropy in nats. Max = log(P) for uniform distribution.
    """
    a = attn_vector.copy()
    a = a + 1e-10
    a = a / a.sum()
    return -np.sum(a * np.log(a))


def attention_sparsity_gini(attn_vector: np.ndarray) -> float:
    """
    Gini coefficient of attention distribution.
    Higher = more concentrated = "cleaner".
    Range: [0, 1] where 1 = all attention on single position.
    """
    a = np.sort(attn_vector.copy())
    n = len(a)
    if a.sum() < 1e-10:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * a) / (n * np.sum(a))) - (n + 1) / n


def peak_sharpness(attn_vector: np.ndarray) -> float:
    """
    Ratio of maximum attention to mean attention.
    Higher = sharper peak = "cleaner".
    """
    mean_val = attn_vector.mean()
    if mean_val < 1e-10:
        return 0.0
    return float(attn_vector.max() / mean_val)


def character_localization_accuracy(
    attn_vector: np.ndarray,
    char_positions: List[int],
    tolerance: int = 3
) -> float:
    """
    Fraction of attention mass within ±tolerance positions of the character.
    Higher = better localization = "cleaner".
    
    Args:
        attn_vector: [P] attention weights
        char_positions: Time positions assigned to this character
        tolerance: Number of positions around the character to consider "correct"
    """
    P = len(attn_vector)
    mask = np.zeros(P, dtype=bool)
    for pos in char_positions:
        start = max(0, pos - tolerance)
        end = min(P, pos + tolerance + 1)
        mask[start:end] = True

    total = attn_vector.sum()
    if total < 1e-10:
        return 0.0
    return float(attn_vector[mask].sum() / total)


def cross_character_overlap(
    attn_vectors: List[np.ndarray],
    top_k_fraction: float = 0.1
) -> float:
    """
    Average IoU between adjacent characters' top-k attention positions.
    Lower = more distinct character attention = "cleaner".
    
    Args:
        attn_vectors: List of [P] attention vectors, one per character
        top_k_fraction: Fraction of positions to consider as "attended"
    """
    if len(attn_vectors) < 2:
        return 0.0

    P = len(attn_vectors[0])
    k = max(1, int(P * top_k_fraction))

    overlaps = []
    for i in range(len(attn_vectors) - 1):
        top_a = set(np.argsort(attn_vectors[i])[-k:])
        top_b = set(np.argsort(attn_vectors[i + 1])[-k:])
        intersection = len(top_a & top_b)
        union = len(top_a | top_b)
        if union > 0:
            overlaps.append(intersection / union)

    return float(np.mean(overlaps)) if overlaps else 0.0


def register_specialization(
    full_attn: np.ndarray,
    num_registers: int
) -> float:
    """
    Measure how specialized registers are (dissimilarity of their attention patterns).
    Higher = more specialized = registers serving distinct roles.
    
    Args:
        full_attn: [S, S] attention map (averaged over heads)
        num_registers: R
    Returns:
        Score in [0, 1]. 1 = maximally different attention patterns.
    """
    if num_registers < 2:
        return 0.0

    # Register-to-patch attention: [R, P]
    reg_to_patch = full_attn[:num_registers, num_registers:]

    # Normalize each register's attention for cosine similarity
    norms = np.linalg.norm(reg_to_patch, axis=1, keepdims=True) + 1e-10
    normalized = reg_to_patch / norms
    similarity_matrix = normalized @ normalized.T  # [R, R]

    # Average off-diagonal similarity
    R = num_registers
    mask = ~np.eye(R, dtype=bool)
    off_diag = similarity_matrix[mask]
    return float(1.0 - np.mean(off_diag))


def register_utilization(
    full_attn: np.ndarray,
    num_registers: int
) -> float:
    """
    How much patch tokens attend to register tokens (use them as info sinks).
    
    Args:
        full_attn: [S, S] attention map
        num_registers: R
    Returns:
        Mean attention from patches to registers.
    """
    if num_registers == 0:
        return 0.0
    patch_to_reg = full_attn[num_registers:, :num_registers]  # [P, R]
    return float(patch_to_reg.mean())


def attention_locality(attn_vector: np.ndarray, query_position: int) -> float:
    """
    Measure how local the attention is relative to the query position.
    Computed as weighted average distance from query position.
    Lower = more local = "cleaner" for character recognition.
    
    Args:
        attn_vector: [P] attention weights
        query_position: The position of the query token
    """
    P = len(attn_vector)
    positions = np.arange(P)
    distances = np.abs(positions - query_position)
    
    total = attn_vector.sum()
    if total < 1e-10:
        return 0.0
    return float(np.sum(attn_vector * distances) / total)


def compute_sample_metrics(
    patch_attn: np.ndarray,
    char_positions: Dict[int, List[int]],
    full_attn: Optional[np.ndarray] = None,
    num_registers: int = 0
) -> Dict[str, float]:
    """
    Compute all attention quality metrics for a single sample.
    
    Args:
        patch_attn: [P, P] patch-to-patch attention (registers stripped)
        char_positions: Dict mapping char_idx → list of time positions
        full_attn: [S, S] full attention including registers
        num_registers: Number of register tokens
    
    Returns:
        Dictionary of metric_name → value
    """
    metrics = {}

    if not char_positions:
        return metrics

    entropies = []
    sparsities = []
    sharpnesses = []
    localizations = []
    localities = []
    char_attn_vectors = []

    for char_idx in sorted(char_positions.keys()):
        positions = char_positions[char_idx]
        # Average attention FROM this character's positions TO all patches
        char_attn = patch_attn[positions, :].mean(axis=0)  # [P]
        char_attn_vectors.append(char_attn)

        entropies.append(attention_entropy(char_attn))
        sparsities.append(attention_sparsity_gini(char_attn))
        sharpnesses.append(peak_sharpness(char_attn))
        localizations.append(
            character_localization_accuracy(char_attn, positions, tolerance=3)
        )
        # Locality: average distance from character's own position
        center_pos = positions[len(positions) // 2]
        localities.append(attention_locality(char_attn, center_pos))

    metrics['entropy_mean'] = float(np.mean(entropies))
    metrics['entropy_std'] = float(np.std(entropies))
    metrics['sparsity_gini_mean'] = float(np.mean(sparsities))
    metrics['sparsity_gini_std'] = float(np.std(sparsities))
    metrics['peak_sharpness_mean'] = float(np.mean(sharpnesses))
    metrics['localization_accuracy_mean'] = float(np.mean(localizations))
    metrics['localization_accuracy_std'] = float(np.std(localizations))
    metrics['locality_mean'] = float(np.mean(localities))
    metrics['cross_char_overlap'] = cross_character_overlap(char_attn_vectors)
    metrics['num_characters'] = len(char_positions)

    # Register-specific metrics
    if full_attn is not None and num_registers > 0:
        metrics['register_specialization'] = register_specialization(
            full_attn, num_registers
        )
        metrics['register_utilization'] = register_utilization(
            full_attn, num_registers
        )

    return metrics
