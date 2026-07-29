"""Rigid-body superposition shared by the mobility feature and interface RMSF.

Kabsch least-squares alignment of every frame in a coordinate array onto the
ensemble mean. Module-private: an implementation detail of the per-residue
fluctuation feature (:mod:`confdelta.features`) and the interface RMSF
(:mod:`confdelta.interface`), not part of the public API.
"""

from __future__ import annotations

import numpy as np


def superpose_to_mean(coords: np.ndarray, align_cols: np.ndarray, *, passes: int = 2) -> np.ndarray:
    """Least-squares superpose every frame onto the mean using *align_cols* atoms.

    ``coords`` is ``(n_frames, n_atoms, 3)``; ``align_cols`` indexes the atoms the
    fit is computed on. Returns the aligned coordinates (all atoms). Uses the
    Kabsch algorithm with no mass weighting, iterating *passes* times: fit to the
    mean, recompute the mean, refit.
    """
    aligned = coords.copy()
    ref = coords[:, align_cols, :].mean(axis=0)
    ref_centroid = ref.mean(axis=0)
    ref_centered = ref - ref_centroid

    for _ in range(passes):
        for f in range(aligned.shape[0]):
            mobile = aligned[f, align_cols, :]
            mob_centroid = mobile.mean(axis=0)
            mob_centered = mobile - mob_centroid
            h = mob_centered.T @ ref_centered
            u, _, vt = np.linalg.svd(h)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
            aligned[f] = (aligned[f] - mob_centroid) @ rot.T + ref_centroid
        ref = aligned[:, align_cols, :].mean(axis=0)
        ref_centroid = ref.mean(axis=0)
        ref_centered = ref - ref_centroid
    return aligned
