"""Per-residue features and the ensemble-level statistical comparison entry point.

:mod:`confdelta.compare` is a pure numeric engine that knows nothing about
ensembles. This module bridges the two: it extracts per-residue features from an
:class:`~confdelta.ensemble.Ensemble`, and :func:`compare_ensemble_groups` feeds
them to the engine, choosing replicate or bootstrap mode from how many
replicates each condition has.

The one feature shipped so far is per-residue contact number (coordination): for
each frame, how many other residues have their centroid within a cutoff. It is
computed from internal distances only, so it needs no structural alignment, and
it is a standard proxy for local packing that a mutation perturbs. The engine is
feature-agnostic, so further features (RMSF, per-pair correlations) can be added
here without touching it.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Callable

import numpy as np

from .compare import ComparisonReport, compare_feature_matrices, compare_frame_matrices
from .ensemble import Ensemble, EnsembleGroup
from .stats import CorrectionMethod, RandomState

logger = logging.getLogger("confdelta")

__all__ = [
    "FeatureExtractor",
    "per_residue_contact_counts",
    "compare_ensemble_groups",
    "FEATURES",
]

# A feature extractor maps an ensemble to (residue labels, per-frame values),
# where per-frame values has shape (n_frames, n_residues).
FeatureExtractor = Callable[[Ensemble], tuple[list[str], np.ndarray]]


def _residue_chain_and_number(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Split ``CHAIN_RESID`` labels into a chain array and an integer resid array."""
    chains = np.empty(len(labels), dtype=object)
    resids = np.empty(len(labels), dtype=int)
    for i, key in enumerate(labels):
        chain, _, resid = str(key).rpartition("_")
        chains[i] = chain
        resids[i] = int(resid) if resid.isdigit() else i
    return chains, resids


def per_residue_contact_counts(
    ensemble: Ensemble,
    *,
    cutoff: float = 8.0,
    min_sequence_separation: int = 2,
) -> tuple[list[str], np.ndarray]:
    """Per-frame contact number for each residue.

    A residue's contact number in a frame is the number of other residues whose
    centroid lies within *cutoff* angstroms of its own, excluding residues fewer
    than *min_sequence_separation* apart in sequence within the same chain (which
    would otherwise count trivial backbone neighbours).

    Parameters
    ----------
    ensemble
        The ensemble to analyse.
    cutoff
        Contact distance between residue centroids, in angstroms.
    min_sequence_separation
        Same-chain residues closer than this in sequence are not counted.

    Returns
    -------
    (labels, values)
        ``labels`` are the residue identifiers (``"A_50"`` ...); ``values`` has
        shape ``(n_frames, n_residues)``.
    """
    # to_simulation() prints progress; silence it for this internal use.
    with contextlib.redirect_stdout(io.StringIO()):
        sim = ensemble.to_simulation()

    labels = [str(k) for k in sim.unique_residue_keys]
    n_res = sim.n_residues
    atom_res = np.asarray(sim._atom_to_residue_map)
    atoms = sim._atoms

    atoms_per_res = np.bincount(atom_res, minlength=n_res).astype(float)
    atoms_per_res[atoms_per_res == 0.0] = 1.0  # guard; every residue has >=1 atom

    # Eligibility mask: exclude self and near-in-sequence same-chain neighbours.
    chains, resids = _residue_chain_and_number(labels)
    same_chain = chains[:, None] == chains[None, :]
    seq_gap = np.abs(resids[:, None] - resids[None, :])
    eligible = ~(same_chain & (seq_gap < min_sequence_separation))

    n_frames = ensemble.n_frames
    values = np.empty((n_frames, n_res), dtype=float)
    for frame_index, _ in enumerate(sim.universe.trajectory):
        positions = atoms.positions
        centroids = np.zeros((n_res, 3), dtype=float)
        np.add.at(centroids, atom_res, positions)
        centroids /= atoms_per_res[:, None]

        deltas = centroids[:, None, :] - centroids[None, :, :]
        distances = np.sqrt((deltas**2).sum(axis=-1))
        in_contact = (distances < cutoff) & eligible
        values[frame_index] = in_contact.sum(axis=1)

    return labels, values


# Registry of named feature extractors, so the CLI/config can select by name.
FEATURES: dict[str, FeatureExtractor] = {
    "contacts": per_residue_contact_counts,
}


def _labels_for_group(
    group: EnsembleGroup, extractor: FeatureExtractor
) -> tuple[list[str], list[np.ndarray]]:
    """Extract features for every replicate, checking the residue labels agree."""
    reference: list[str] | None = None
    per_replicate: list[np.ndarray] = []
    for ensemble in group:
        labels, values = extractor(ensemble)
        if reference is None:
            reference = labels
        elif labels != reference:
            raise ValueError(
                f"Replicates in condition {group.label!r} describe different residues; "
                "every replicate must be the same system with the same selection."
            )
        per_replicate.append(values)
    assert reference is not None  # group is non-empty by construction
    return reference, per_replicate


def compare_ensemble_groups(
    group_a: EnsembleGroup,
    group_b: EnsembleGroup,
    *,
    feature: str = "contacts",
    correction: CorrectionMethod = "fdr_bh",
    alpha: float = 0.05,
    confidence: float = 0.95,
    rng: RandomState = None,
) -> ComparisonReport:
    """Compare two conditions per residue, with full statistics.

    Extracts *feature* from every ensemble and dispatches to the engine:

    * **replicate mode** when both conditions have two or more replicates: each
      replicate is reduced to its per-residue mean, giving one value per
      replicate, and the conditions are compared with a permutation test and
      Hedges' g per residue, FDR-corrected across residues.
    * **bootstrap mode** otherwise: the single run per condition is compared
      frame-by-frame with a block bootstrap. If a condition happens to carry
      extra replicates but the other does not, the extras are dropped with a
      warning, since a replicate-level test needs both sides.

    Parameters
    ----------
    group_a, group_b
        The two conditions. Both must describe the same residues.
    feature
        Name of a registered feature extractor; see :data:`FEATURES`.
    correction, alpha, confidence, rng
        Passed through to the engine.

    Returns
    -------
    ComparisonReport
        Per-residue effect sizes, confidence intervals and corrected q-values.
    """
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature {feature!r}. Available: {', '.join(sorted(FEATURES))}.")
    extractor = FEATURES[feature]

    labels_a, frames_a = _labels_for_group(group_a, extractor)
    labels_b, frames_b = _labels_for_group(group_b, extractor)
    if labels_a != labels_b:
        raise ValueError(
            "The two conditions describe different residues; they must be the same "
            "system with the same selection to be compared per residue."
        )

    both_replicated = group_a.has_replicates and group_b.has_replicates
    if both_replicated:
        matrix_a = np.stack([values.mean(axis=0) for values in frames_a])
        matrix_b = np.stack([values.mean(axis=0) for values in frames_b])
        return compare_feature_matrices(
            matrix_a,
            matrix_b,
            feature_names=labels_a,
            correction=correction,
            alpha=alpha,
            confidence=confidence,
            rng=rng,
        )

    if group_a.has_replicates or group_b.has_replicates:
        logger.warning(
            "Only one condition has replicates (%d vs %d). A replicate-level test "
            "needs both, so falling back to a single-run block bootstrap using the "
            "first replicate of each; the extra replicates are ignored.",
            group_a.n_replicates,
            group_b.n_replicates,
        )
    return compare_frame_matrices(
        frames_a[0],
        frames_b[0],
        feature_names=labels_a,
        correction=correction,
        alpha=alpha,
        confidence=confidence,
        rng=rng,
    )
