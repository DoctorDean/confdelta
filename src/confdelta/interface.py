"""Interface persistence: how well a binding interface holds up across an ensemble.

Given an ensemble of a two-partner complex, this module measures how reliably the
interface is maintained: which inter-partner residue contacts form, what fraction
of the ensemble each one persists for, how mobile the interface residues are, and
how pre-organised the binder is. Two designs can then be compared over a shared
reference set of contacts, reusing :mod:`confdelta.compare`, so "does design X
hold its interface better than design Y" is answered with the same effect sizes,
confidence intervals and corrected q-values as the rest of confdelta.

This is confdelta's first downstream capability, aimed at design triage: a
designed binder should hold the contacts it was designed to make, and a
pre-organised binder pays less entropic cost on binding.

Definitions
-----------
An *interface contact* is a pair of residues, one from each partner, judged in
contact in a frame. Two contact definitions are offered (``method=``):

* ``"heavy_atom"`` (default, 4.5 A) -- in contact if any heavy-atom pair is within
  the cutoff. The standard, chemically-faithful definition.
* ``"centroid"`` (8 A) -- in contact if the residue centroids are within the
  cutoff. Coarser and cheaper, consistent with confdelta's contact-number feature.

*Persistence* of a contact is the fraction of frames it is present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .compare import ComparisonReport, compare_feature_matrices, compare_frame_matrices
from .ensemble import Ensemble, EnsembleGroup
from .stats import CorrectionMethod, RandomState

__all__ = [
    "ContactMethod",
    "DEFAULT_CUTOFFS",
    "InterfaceContact",
    "InterfacePersistence",
    "interface_persistence",
    "interface_rmsf",
    "binder_preorganisation",
    "compare_interface_persistence",
]

ContactMethod = str  # "heavy_atom" | "centroid"

# Default contact cutoffs, in angstroms, per method.
DEFAULT_CUTOFFS: dict[str, float] = {"heavy_atom": 4.5, "centroid": 8.0}


def _residue_labels_and_membership(atoms) -> tuple[list[str], np.ndarray]:
    """Return per-residue ``CHAIN_RESID`` labels and an (n_res, n_atoms) mask.

    The membership mask groups the selection's atoms by residue so atom-atom
    distances can be reduced to residue-residue contacts.
    """
    resindices = atoms.resindices
    unique, inverse = np.unique(resindices, return_inverse=True)
    n_res, n_atoms = len(unique), len(atoms)

    membership = np.zeros((n_res, n_atoms), dtype=bool)
    membership[inverse, np.arange(n_atoms)] = True

    # Label each residue as CHAIN_RESID, preferring chainID then segid.
    try:
        chains = [str(c).strip() or "X" for c in atoms.chainIDs]
    except Exception:  # noqa: BLE001 - topology may lack chainIDs
        try:
            chains = [str(s).strip() or "X" for s in atoms.segids]
        except Exception:  # noqa: BLE001
            chains = ["X"] * n_atoms
    resids = atoms.resids

    labels = []
    for r in range(n_res):
        atom_idx = int(np.argmax(membership[r]))
        labels.append(f"{chains[atom_idx]}_{resids[atom_idx]}")
    return labels, membership


def _resolve(method: ContactMethod, cutoff: float | None) -> tuple[str, float]:
    if method not in DEFAULT_CUTOFFS:
        raise ValueError(
            f"Unknown contact method {method!r}; choose from {sorted(DEFAULT_CUTOFFS)}."
        )
    return method, DEFAULT_CUTOFFS[method] if cutoff is None else cutoff


def _per_frame_contacts(
    ensemble: Ensemble, partner_a: str, partner_b: str, method: str, cutoff: float
) -> tuple[list[str], list[str], np.ndarray]:
    """Return partner-A residue labels, partner-B labels, and an

    ``(n_frames, n_res_a, n_res_b)`` boolean contact array.
    """
    from MDAnalysis.lib.distances import distance_array

    universe = ensemble.universe
    atoms_a = universe.select_atoms(partner_a)
    atoms_b = universe.select_atoms(partner_b)
    if len(atoms_a) == 0 or len(atoms_b) == 0:
        raise ValueError(
            f"Partner selection matched no atoms (A: {len(atoms_a)}, B: {len(atoms_b)}). "
            f"Check {partner_a!r} and {partner_b!r}."
        )

    labels_a, mem_a = _residue_labels_and_membership(atoms_a)
    labels_b, mem_b = _residue_labels_and_membership(atoms_b)
    mem_a_f = mem_a.astype(float)
    mem_b_f = mem_b.astype(float)

    n_frames = ensemble.n_frames
    contacts = np.zeros((n_frames, len(labels_a), len(labels_b)), dtype=bool)
    for f, _ in enumerate(universe.trajectory):
        pos_a = atoms_a.positions
        pos_b = atoms_b.positions
        if method == "centroid":
            cen_a = mem_a_f @ pos_a / mem_a_f.sum(axis=1, keepdims=True)
            cen_b = mem_b_f @ pos_b / mem_b_f.sum(axis=1, keepdims=True)
            res_dist = distance_array(cen_a.astype(np.float32), cen_b.astype(np.float32))
            contacts[f] = res_dist < cutoff
        else:  # heavy_atom: any atom pair within cutoff
            atom_close = distance_array(pos_a, pos_b) < cutoff
            # reduce atom-atom to residue-residue: mem_a (r_a, atoms_a) @ close @ mem_b.T
            res_close = mem_a @ atom_close @ mem_b.T
            contacts[f] = res_close > 0
    return labels_a, labels_b, contacts


@dataclass(frozen=True)
class InterfaceContact:
    """One inter-partner residue contact and how persistent it is."""

    residue_a: str
    residue_b: str
    persistence: float

    @property
    def label(self) -> str:
        return f"{self.residue_a}:{self.residue_b}"


@dataclass(frozen=True)
class InterfacePersistence:
    """The interface of one complex ensemble: its contacts and their persistence."""

    contacts: list[InterfaceContact]
    partner_a: str
    partner_b: str
    method: str
    cutoff: float
    n_frames: int
    # per-frame presence of each contact, aligned with `contacts`: (n_frames, n_contacts)
    _presence: np.ndarray

    def labels(self) -> list[str]:
        return [c.label for c in self.contacts]

    def persistence_array(self) -> np.ndarray:
        return np.array([c.persistence for c in self.contacts])

    def score(self) -> float:
        """Mean persistence over interface contacts: a single interface-stability number."""
        if not self.contacts:
            return 0.0
        return float(self.persistence_array().mean())

    def n_stable(self, threshold: float = 0.75) -> int:
        """Number of contacts present in at least *threshold* of frames."""
        return int((self.persistence_array() >= threshold).sum())

    def ranked(self) -> list[InterfaceContact]:
        """Contacts ordered from most to least persistent."""
        return sorted(self.contacts, key=lambda c: c.persistence, reverse=True)


def interface_persistence(
    ensemble: Ensemble,
    partner_a: str,
    partner_b: str,
    *,
    method: ContactMethod = "heavy_atom",
    cutoff: float | None = None,
    min_persistence: float = 0.0,
) -> InterfacePersistence:
    """Measure the interface between two partners across an ensemble.

    Parameters
    ----------
    ensemble
        The complex ensemble.
    partner_a, partner_b
        MDAnalysis selection strings for the two partners, e.g. ``"chainID A"``
        and ``"chainID B"``, or ``"segid NB"`` and ``"segid AG"``.
    method
        ``"heavy_atom"`` (default) or ``"centroid"``; see the module docstring.
    cutoff
        Contact cutoff in angstroms; defaults to the method's standard value.
    min_persistence
        Drop contacts present in fewer than this fraction of frames. The default
        keeps every contact seen at least once.

    Returns
    -------
    InterfacePersistence
    """
    method, cutoff = _resolve(method, cutoff)
    labels_a, labels_b, contacts = _per_frame_contacts(
        ensemble, partner_a, partner_b, method, cutoff
    )

    ever = contacts.any(axis=0)
    persistence = contacts.mean(axis=0)

    entries: list[InterfaceContact] = []
    presence_columns: list[np.ndarray] = []
    for i in range(len(labels_a)):
        for j in range(len(labels_b)):
            if ever[i, j] and persistence[i, j] >= min_persistence:
                entries.append(InterfaceContact(labels_a[i], labels_b[j], float(persistence[i, j])))
                presence_columns.append(contacts[:, i, j].astype(float))

    presence = (
        np.column_stack(presence_columns) if presence_columns else np.empty((ensemble.n_frames, 0))
    )
    return InterfacePersistence(
        contacts=entries,
        partner_a=partner_a,
        partner_b=partner_b,
        method=method,
        cutoff=cutoff,
        n_frames=ensemble.n_frames,
        _presence=presence,
    )


# ---------------------------------------------------------------------------
# Interface RMSF and binder pre-organisation
# ---------------------------------------------------------------------------


def _superpose_to_mean(coords: np.ndarray, align_cols: np.ndarray) -> np.ndarray:
    """Least-squares superpose every frame onto the mean using *align_cols* atoms.

    ``coords`` is ``(n_frames, n_atoms, 3)``; ``align_cols`` indexes the atoms the
    fit is computed on. Returns the aligned coordinates (all atoms). Uses the
    Kabsch algorithm; no mass weighting.
    """
    aligned = coords.copy()
    ref = coords[:, align_cols, :].mean(axis=0)
    ref_centroid = ref.mean(axis=0)
    ref_centered = ref - ref_centroid

    for _ in range(2):  # two passes: fit to mean, recompute mean, refit
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


def interface_rmsf(
    ensemble: Ensemble,
    selection: str,
    *,
    align_on: str | None = None,
) -> dict[str, float]:
    """Per-residue RMSF of *selection*, after removing global motion.

    Each frame is superposed onto the mean structure using the ``align_on`` atoms
    (default: the same *selection*), then the root-mean-square fluctuation of each
    residue about its mean position is returned, keyed by ``CHAIN_RESID`` label.

    Aligning the binder on the *target* (``align_on`` = the target selection) and
    reporting RMSF of the binder measures how much the binder moves relative to
    the target -- interface mobility. Aligning on the selection itself measures
    the selection's internal flexibility.
    """
    universe = ensemble.universe
    target = universe.select_atoms(selection)
    if len(target) == 0:
        raise ValueError(f"selection {selection!r} matched no atoms.")
    align_atoms = target if align_on is None else universe.select_atoms(align_on)
    if len(align_atoms) == 0:
        raise ValueError(f"align_on selection {align_on!r} matched no atoms.")

    # Collect coordinates for the union of target and align atoms.
    target_ix = target.ix
    align_ix = align_atoms.ix
    all_ix = np.union1d(target_ix, align_ix)
    index_of = {a: k for k, a in enumerate(all_ix)}
    align_cols = np.array([index_of[a] for a in align_ix])
    target_cols = np.array([index_of[a] for a in target_ix])

    frames = np.empty((ensemble.n_frames, len(all_ix), 3), dtype=float)
    ag = universe.atoms[all_ix]
    for f, _ in enumerate(universe.trajectory):
        frames[f] = ag.positions

    aligned = _superpose_to_mean(frames, align_cols)
    target_coords = aligned[:, target_cols, :]
    mean_pos = target_coords.mean(axis=0)
    sq = ((target_coords - mean_pos) ** 2).sum(axis=2)  # (n_frames, n_target)
    atom_rmsf = np.sqrt(sq.mean(axis=0))

    labels, membership = _residue_labels_and_membership(target)
    out: dict[str, float] = {}
    for r, label in enumerate(labels):
        out[label] = float(atom_rmsf[membership[r]].mean())
    return out


def binder_preorganisation(
    bound: Ensemble,
    unbound: Ensemble,
    binder_selection: str,
) -> dict[str, float]:
    """Compare the binder's internal flexibility bound vs unbound, per residue.

    Returns ``{residue: bound_rmsf - unbound_rmsf}``. A pre-organised binder
    changes little on binding, so values near zero indicate pre-organisation;
    large negative values mean the binder rigidifies when bound (it was
    disordered free and ordered on binding, an entropic cost). Both ensembles
    are aligned on the binder itself, so this is internal conformational spread,
    not rigid-body motion.
    """
    bound_rmsf = interface_rmsf(bound, binder_selection)
    unbound_rmsf = interface_rmsf(unbound, binder_selection)
    shared = set(bound_rmsf) & set(unbound_rmsf)
    if not shared:
        raise ValueError(
            "The bound and unbound binder selections share no residues; they must "
            "be the same binder with the same numbering."
        )
    return {res: bound_rmsf[res] - unbound_rmsf[res] for res in sorted(shared)}


# ---------------------------------------------------------------------------
# Two-design comparison over a reference contact set
# ---------------------------------------------------------------------------


def _presence_over_reference(
    ensemble: Ensemble,
    partner_a: str,
    partner_b: str,
    reference: Sequence[str],
    method: str,
    cutoff: float,
) -> np.ndarray:
    """Per-frame presence (n_frames, n_reference) for a fixed set of contacts.

    Contacts in *reference* not formed by this ensemble are all-zero columns
    (persistence 0), which is the point: a design that fails to make a designed
    contact should score zero on it, not be silently dropped.
    """
    labels_a, labels_b, contacts = _per_frame_contacts(
        ensemble, partner_a, partner_b, method, cutoff
    )
    index_a = {label: i for i, label in enumerate(labels_a)}
    index_b = {label: j for j, label in enumerate(labels_b)}

    out = np.zeros((ensemble.n_frames, len(reference)), dtype=float)
    for k, contact_label in enumerate(reference):
        res_a, _, res_b = contact_label.partition(":")
        i = index_a.get(res_a)
        j = index_b.get(res_b)
        if i is not None and j is not None:
            out[:, k] = contacts[:, i, j].astype(float)
    return out


def compare_interface_persistence(
    group_a: EnsembleGroup,
    group_b: EnsembleGroup,
    partner_a: str,
    partner_b: str,
    *,
    reference_contacts: Sequence[str],
    method: ContactMethod = "heavy_atom",
    cutoff: float | None = None,
    correction: CorrectionMethod = "fdr_bh",
    alpha: float = 0.05,
    rng: RandomState = None,
) -> ComparisonReport:
    """Compare per-contact persistence between two designs over a reference set.

    *reference_contacts* is the set of contacts to compare on -- typically the
    designed interface, as ``"CHAIN_RESID:CHAIN_RESID"`` labels (the labels from
    :meth:`InterfacePersistence.labels`). Each design's per-frame presence of
    those contacts is compared with the same engine as the rest of confdelta:
    replicate mode when both conditions have replicates, block bootstrap
    otherwise. The effect size per contact is the difference in persistence, with
    a confidence interval and an FDR-corrected q-value.

    Answering the design-triage question: over the contacts the binder is meant
    to make, does design A hold them better than design B, and where?
    """
    method, cutoff = _resolve(method, cutoff)
    reference = list(reference_contacts)
    if not reference:
        raise ValueError("reference_contacts is empty; nothing to compare.")

    def presence(ensemble: Ensemble) -> np.ndarray:
        return _presence_over_reference(ensemble, partner_a, partner_b, reference, method, cutoff)

    both_replicated = group_a.has_replicates and group_b.has_replicates
    if both_replicated:
        matrix_a = np.stack([presence(e).mean(axis=0) for e in group_a])
        matrix_b = np.stack([presence(e).mean(axis=0) for e in group_b])
        return compare_feature_matrices(
            matrix_a,
            matrix_b,
            feature_names=reference,
            correction=correction,
            alpha=alpha,
            rng=rng,
        )

    return compare_frame_matrices(
        presence(group_a[0]),
        presence(group_b[0]),
        feature_names=reference,
        correction=correction,
        alpha=alpha,
        rng=rng,
    )
