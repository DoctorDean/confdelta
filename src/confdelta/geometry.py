"""User-defined geometric collective variables: distances, angles and dihedrals.

The built-in per-residue features (``contacts``, ``rmsf``) are computed for every
residue automatically. Geometric CVs are different: you name the distances, angles
and dihedrals you care about, by residue, and get a per-frame feature matrix that
feeds the same comparison engine -- so a hand-picked reaction coordinate (a
flap-opening distance, a hinge angle, a backbone torsion) gets the same effect
size, confidence interval and corrected q-value as any other feature.

* A **distance** is measured between two points.
* An **angle** is the angle at the *middle* point of three (vertex in the centre).
* A **dihedral** is the torsion about the central bond of four points, in
  ``(-180, 180]`` degrees.

Each *point* is a residue, or a **group** of residues measured at its centroid
(pass a list, e.g. ``[64, 65, 66]``, to measure between domains rather than single
residues). Residues are named by number (``50``) or, where a number is ambiguous
across chains, by ``CHAIN_RESID`` label (``"A_50"``). Points use each residue's
alpha carbon by default (``atom="CA"``).

Example
-------
::

    from confdelta import compare_ensemble_groups, geometric_features

    cvs = geometric_features(
        distances=[("A_25", "A_50")],                    # flap opening
        angles=[("A_48", "A_49", "A_50")],               # flap-tip angle
        dihedrals=[("A_47", "A_48", "A_49", "A_50")],    # flap-tip torsion
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

__all__ = ["ResidueRef", "PointRef", "geometric_features"]

ResidueRef = int | str
"""A residue: its number (``50``) or a ``CHAIN_RESID`` label (``"A_50"``)."""

PointRef = ResidueRef | Sequence[ResidueRef]
"""A point: a single residue, or a list/tuple of residues (measured at the centroid)."""


def _resid_of(label: str) -> int | None:
    tail = label.rpartition("_")[2]
    return int(tail) if tail.isdigit() else None


def _resolve_residue(labels: list[str], ref: ResidueRef) -> int:
    """Return the residue index for a single residue *ref*."""
    if isinstance(ref, str) and ref in labels:
        return labels.index(ref)
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
    return matches[0]


def _resolve_point(
    labels: list[str], atom_res: np.ndarray, names: np.ndarray, ref: PointRef, atom: str
) -> tuple[np.ndarray, str]:
    """Return ``(atom indices, label)`` for a point: one residue or a group centroid."""
    members: list[ResidueRef]
    if isinstance(ref, Sequence) and not isinstance(ref, str):
        members = list(ref)
    else:
        members = [ref]
    indices: list[int] = []
    member_labels: list[str] = []
    for member in members:
        residue = _resolve_residue(labels, member)
        candidates = np.where((atom_res == residue) & (names == atom))[0]
        if candidates.size == 0:
            raise ValueError(f"Residue {labels[residue]} has no atom named {atom!r}.")
        indices.append(int(candidates[0]))
        member_labels.append(labels[residue])
    label = member_labels[0] if len(member_labels) == 1 else "[" + ",".join(member_labels) + "]"
    return np.array(indices, dtype=int), label


def _angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    v1, v2 = p1 - vertex, p2 - vertex
    cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x, y = float(np.dot(n1, n2)), float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def geometric_features(
    *,
    distances: Sequence[tuple[PointRef, PointRef]] = (),
    angles: Sequence[tuple[PointRef, PointRef, PointRef]] = (),
    dihedrals: Sequence[tuple[PointRef, PointRef, PointRef, PointRef]] = (),
    atom: str = "CA",
) -> FeatureExtractor:
    """Build a feature extractor for the given distances, angles and dihedrals.

    Parameters
    ----------
    distances
        Pairs of points; each yields their separation in angstroms.
    angles
        Triples of points; each yields the angle in degrees at the middle one.
    dihedrals
        Quadruples of points; each yields the torsion in degrees, ``(-180, 180]``.
    atom
        Atom name used for every residue (default the alpha carbon, ``"CA"``).

    Each point is a residue, or a list/tuple of residues measured at the centroid.

    Returns
    -------
    FeatureExtractor
        Suitable for ``compare_ensemble_groups(..., feature=<this>)``. Feature
        labels are ``"dist:A_25-A_50"``, ``"angle:A_48-A_49-A_50"`` and
        ``"dihedral:A_47-A_48-A_49-A_50"``; group points read ``"[A_64,A_65]"``.
    """
    dist_specs = [tuple(d) for d in distances]
    angle_specs = [tuple(a) for a in angles]
    dihedral_specs = [tuple(t) for t in dihedrals]
    for spec, arity, kind in (
        *((d, 2, "distance") for d in dist_specs),
        *((a, 3, "angle") for a in angle_specs),
        *((t, 4, "dihedral") for t in dihedral_specs),
    ):
        if len(spec) != arity:
            raise ValueError(f"{kind.capitalize()} CVs take exactly {arity} points; got {spec!r}.")
    if not dist_specs and not angle_specs and not dihedral_specs:
        raise ValueError("Specify at least one distance, angle or dihedral to measure.")

    def extractor(ensemble: Ensemble) -> tuple[list[str], np.ndarray]:
        # to_simulation() prints progress; silence it for this internal use.
        with contextlib.redirect_stdout(io.StringIO()):
            sim = ensemble.to_simulation()
        labels = [str(k) for k in sim.unique_residue_keys]
        atom_res = np.asarray(sim._atom_to_residue_map)
        atoms = sim._atoms
        names = np.asarray(atoms.names)

        def resolve(ref: PointRef) -> tuple[np.ndarray, str]:
            return _resolve_point(labels, atom_res, names, ref, atom)

        cv_labels: list[str] = []
        dist_pts: list[tuple[np.ndarray, np.ndarray]] = []
        angle_pts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        dihedral_pts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for spec in dist_specs:
            pts, labs = zip(*(resolve(r) for r in spec), strict=True)
            dist_pts.append(pts)
            cv_labels.append("dist:" + "-".join(labs))
        for spec in angle_specs:
            pts, labs = zip(*(resolve(r) for r in spec), strict=True)
            angle_pts.append(pts)
            cv_labels.append("angle:" + "-".join(labs))
        for spec in dihedral_specs:
            pts, labs = zip(*(resolve(r) for r in spec), strict=True)
            dihedral_pts.append(pts)
            cv_labels.append("dihedral:" + "-".join(labs))

        n_frames = ensemble.n_frames
        values = np.empty((n_frames, len(cv_labels)), dtype=float)
        for f, _ in enumerate(sim.universe.trajectory):
            pos = atoms.positions
            col = 0
            for ia, ib in dist_pts:
                values[f, col] = float(np.linalg.norm(pos[ia].mean(0) - pos[ib].mean(0)))
                col += 1
            for ia, ib, ic in angle_pts:
                values[f, col] = _angle(pos[ia].mean(0), pos[ib].mean(0), pos[ic].mean(0))
                col += 1
            for ia, ib, ic, idd in dihedral_pts:
                values[f, col] = _dihedral(
                    pos[ia].mean(0), pos[ib].mean(0), pos[ic].mean(0), pos[idd].mean(0)
                )
                col += 1
        return cv_labels, values

    return extractor
