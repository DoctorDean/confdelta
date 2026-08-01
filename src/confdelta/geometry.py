"""User-defined geometric collective variables: inter-residue distances and angles.

The built-in per-residue features (``contacts``, ``rmsf``) are computed for every
residue automatically. Geometric CVs are different: you name the distances and
angles you care about, by residue, and get a per-frame feature matrix that feeds
the same comparison engine -- so a hand-picked reaction coordinate (a flap-opening
distance, a hinge angle) gets the same effect size, confidence interval and
corrected q-value as any other feature.

* A **distance** is measured between two residues.
* An **angle** is the angle at the *middle* residue of three (vertex in the centre).

Residues are named by number (``50``) or, where a number is ambiguous across
chains, by ``CHAIN_RESID`` label (``"A_50"``). Measurements use each residue's
alpha carbon by default (``atom="CA"``).

Example
-------
::

    from confdelta import compare_ensemble_groups
    from confdelta.geometry import geometric_features

    cvs = geometric_features(
        distances=[("A_25", "A_50"), ("B_25", "B_50")],  # flap opening, each monomer
        angles=[("A_48", "A_49", "A_50")],               # flap-tip angle
    )
    report = compare_ensemble_groups(wild_type, mutant, feature=cvs)
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence

import numpy as np

from .ensemble import Ensemble
from .features import FeatureExtractor

__all__ = ["ResidueRef", "geometric_features"]

ResidueRef = int | str
"""A residue: its number (``50``) or a ``CHAIN_RESID`` label (``"A_50"``)."""


def _resid_of(label: str) -> int | None:
    tail = label.rpartition("_")[2]
    return int(tail) if tail.isdigit() else None


def _resolve_atom(
    labels: list[str], atom_res: np.ndarray, names: np.ndarray, ref: ResidueRef, atom: str
) -> tuple[int, str]:
    """Return ``(atom index, residue label)`` for *ref*'s *atom*-named atom."""
    if isinstance(ref, str) and ref in labels:
        residue = labels.index(ref)
    else:
        try:
            target = int(ref)
        except (TypeError, ValueError):
            raise ValueError(
                f"Residue {ref!r} is neither a residue number nor a known CHAIN_RESID label."
            ) from None
        matches = [i for i, lab in enumerate(labels) if _resid_of(lab) == target]
        if not matches:
            raise ValueError(f"No residue numbered {target} in this ensemble.")
        if len(matches) > 1:
            options = ", ".join(labels[i] for i in matches)
            raise ValueError(
                f"Residue number {target} is ambiguous across chains ({options}); "
                f"use a CHAIN_RESID label such as {labels[matches[0]]!r}."
            )
        residue = matches[0]
    candidates = np.where((atom_res == residue) & (names == atom))[0]
    if candidates.size == 0:
        raise ValueError(f"Residue {labels[residue]} has no atom named {atom!r}.")
    return int(candidates[0]), labels[residue]


def geometric_features(
    *,
    distances: Sequence[tuple[ResidueRef, ResidueRef]] = (),
    angles: Sequence[tuple[ResidueRef, ResidueRef, ResidueRef]] = (),
    atom: str = "CA",
) -> FeatureExtractor:
    """Build a feature extractor for the given inter-residue distances and angles.

    Parameters
    ----------
    distances
        Pairs of residues; each yields their separation in angstroms.
    angles
        Triples of residues; each yields the angle in degrees at the middle one.
    atom
        Atom name used for every residue (default the alpha carbon, ``"CA"``).

    Returns
    -------
    FeatureExtractor
        Suitable for ``compare_ensemble_groups(..., feature=<this>)``. Feature
        labels are ``"dist:A_25-A_50"`` and ``"angle:A_48-A_49-A_50"``.
    """
    dist_specs = [tuple(d) for d in distances]
    angle_specs = [tuple(a) for a in angles]
    for d in dist_specs:
        if len(d) != 2:
            raise ValueError(f"A distance needs exactly two residues; got {d!r}.")
    for a in angle_specs:
        if len(a) != 3:
            raise ValueError(f"An angle needs exactly three residues; got {a!r}.")
    if not dist_specs and not angle_specs:
        raise ValueError("Specify at least one distance or angle to measure.")

    def extractor(ensemble: Ensemble) -> tuple[list[str], np.ndarray]:
        # to_simulation() prints progress; silence it for this internal use.
        with contextlib.redirect_stdout(io.StringIO()):
            sim = ensemble.to_simulation()
        labels = [str(k) for k in sim.unique_residue_keys]
        atom_res = np.asarray(sim._atom_to_residue_map)
        atoms = sim._atoms
        names = np.asarray(atoms.names)

        dist_idx: list[tuple[int, int]] = []
        angle_idx: list[tuple[int, int, int]] = []
        cv_labels: list[str] = []
        for a, b in dist_specs:
            ia, la = _resolve_atom(labels, atom_res, names, a, atom)
            ib, lb = _resolve_atom(labels, atom_res, names, b, atom)
            dist_idx.append((ia, ib))
            cv_labels.append(f"dist:{la}-{lb}")
        for a, b, c in angle_specs:
            ia, la = _resolve_atom(labels, atom_res, names, a, atom)
            ib, lb = _resolve_atom(labels, atom_res, names, b, atom)
            ic, lc = _resolve_atom(labels, atom_res, names, c, atom)
            angle_idx.append((ia, ib, ic))
            cv_labels.append(f"angle:{la}-{lb}-{lc}")

        n_frames = ensemble.n_frames
        values = np.empty((n_frames, len(cv_labels)), dtype=float)
        for f, _ in enumerate(sim.universe.trajectory):
            pos = atoms.positions
            col = 0
            for ia, ib in dist_idx:
                values[f, col] = float(np.linalg.norm(pos[ia] - pos[ib]))
                col += 1
            for ia, ib, ic in angle_idx:
                v1 = pos[ia] - pos[ib]
                v2 = pos[ic] - pos[ib]
                cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                values[f, col] = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                col += 1
        return cv_labels, values

    return extractor
