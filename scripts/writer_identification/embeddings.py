"""
Writer embeddings — build fixed-size descriptors per writer/document.

Combines VLAC character-level features with register-token global features
to produce a single embedding per writer.

Descriptor layout:
    [VLAC_flat (C*D) | register_mean_flat (R*D)]
    Total dimension: C*D + R*D

When registers are absent (R=0), the descriptor is purely VLAC.
"""

import numpy as np


def build_writer_descriptor(vlac, reg_mean=None):
    """
    Flatten VLAC matrix and optional register mean into a 1-D descriptor.

    Parameters
    ----------
    vlac : [C, D]
    reg_mean : [R, D] or None

    Returns
    -------
    descriptor : [C*D + R*D] numpy float32, L2-normalised
    """
    parts = [vlac.ravel()]
    if reg_mean is not None:
        parts.append(reg_mean.ravel())
    desc = np.concatenate(parts).astype(np.float32)

    # Global L2 normalisation
    norm = np.linalg.norm(desc)
    if norm > 1e-8:
        desc /= norm
    return desc


def build_writer_descriptors(writer_vlacs):
    """
    Build descriptors for multiple writers.

    Parameters
    ----------
    writer_vlacs : dict  writer_id → (vlac, counts, reg_mean)

    Returns
    -------
    writer_ids : list[str]
    descriptors : [W, dim] numpy
    """
    writer_ids = sorted(writer_vlacs.keys())
    descs = []
    for wid in writer_ids:
        vlac, counts, reg_mean = writer_vlacs[wid]
        descs.append(build_writer_descriptor(vlac, reg_mean))
    return writer_ids, np.stack(descs)
