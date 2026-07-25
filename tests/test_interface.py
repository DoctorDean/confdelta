"""Tests for confdelta.interface — interface persistence, RMSF, pre-organisation."""

from __future__ import annotations

import numpy as np
import pytest

from confdelta import Ensemble, EnsembleGroup
from confdelta.interface import (
    DEFAULT_CUTOFFS,
    InterfacePersistence,
    binder_preorganisation,
    compare_interface_persistence,
    interface_persistence,
    interface_rmsf,
)


def _write_complex_pdb(path, n_target, n_binder, target_xyz, binder_xyz):
    """Write a 2-chain CA-only PDB: chain A (target), chain B (binder)."""
    lines = []
    for i in range(n_target):
        x, y, z = target_xyz[i]
        lines.append(
            f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    for j in range(n_binder):
        x, y, z = binder_xyz[j]
        lines.append(
            f"ATOM  {n_target + j + 1:>5}  CA  ALA B{j + 1:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _complex_base(n_target=10, n_binder=6, gap=6.0):
    target = np.zeros((n_target, 3), dtype=np.float32)
    target[:, 0] = np.arange(n_target) * 3.8
    binder = np.zeros((n_binder, 3), dtype=np.float32)
    binder[:, 0] = np.arange(n_binder) * 3.8 + 3.8  # offset so B_j sits over A_{j+1}
    binder[:, 1] = gap
    return target, binder


@pytest.fixture
def complex_topology(tmp_path):
    target, binder = _complex_base()
    return _write_complex_pdb(tmp_path / "complex.pdb", 10, 6, target, binder)


def _complex_ensemble(topology, *, n_frames=60, binder_rock=0.0, seed=0, name="cplx"):
    """A complex ensemble; larger binder_rock breaks interface contacts more."""
    target, binder = _complex_base()
    base = np.vstack([target, binder])
    rng = np.random.default_rng(seed)
    coords = np.empty((n_frames, len(base), 3), dtype=np.float32)
    for f in range(n_frames):
        c = base.copy()
        c[10:] += rng.normal(0, 0.3, (6, 3))  # binder internal wobble
        c[10:, 1] += rng.normal(0, binder_rock)  # binder rocks away from target
        coords[f] = c
    return Ensemble.from_coordinates(coords, topology, selection="all", name=name)


class TestInterfacePersistence:
    def test_finds_contacts_and_reports_persistence(self, complex_topology):
        ens = _complex_ensemble(complex_topology, binder_rock=0.0)
        result = interface_persistence(ens, "chainID A", "chainID B", method="centroid")
        assert isinstance(result, InterfacePersistence)
        assert len(result.contacts) > 0
        assert all(0.0 <= c.persistence <= 1.0 for c in result.contacts)

    def test_stable_interface_scores_high(self, complex_topology):
        # No rocking: the interface stays intact, so mean persistence is high.
        ens = _complex_ensemble(complex_topology, binder_rock=0.0)
        result = interface_persistence(ens, "chainID A", "chainID B", method="centroid")
        assert result.score() > 0.8

    def test_rocking_binder_lowers_persistence(self, complex_topology):
        stable = interface_persistence(
            _complex_ensemble(complex_topology, binder_rock=0.0, seed=1),
            "chainID A",
            "chainID B",
            method="centroid",
        )
        wobbly = interface_persistence(
            _complex_ensemble(complex_topology, binder_rock=4.0, seed=1),
            "chainID A",
            "chainID B",
            method="centroid",
        )
        assert wobbly.score() < stable.score()

    def test_heavy_atom_and_centroid_both_work(self, complex_topology):
        ens = _complex_ensemble(complex_topology, binder_rock=0.5)
        for method in ("heavy_atom", "centroid"):
            result = interface_persistence(ens, "chainID A", "chainID B", method=method)
            assert result.method == method
            assert result.cutoff == DEFAULT_CUTOFFS[method]

    def test_min_persistence_filters_transient_contacts(self, complex_topology):
        ens = _complex_ensemble(complex_topology, binder_rock=3.0, seed=2)
        everything = interface_persistence(ens, "chainID A", "chainID B", method="centroid")
        stable_only = interface_persistence(
            ens, "chainID A", "chainID B", method="centroid", min_persistence=0.9
        )
        assert len(stable_only.contacts) <= len(everything.contacts)
        assert all(c.persistence >= 0.9 for c in stable_only.contacts)

    def test_n_stable_counts_persistent_contacts(self, complex_topology):
        ens = _complex_ensemble(complex_topology, binder_rock=0.0)
        result = interface_persistence(ens, "chainID A", "chainID B", method="centroid")
        assert result.n_stable(threshold=0.75) <= len(result.contacts)
        assert result.n_stable(threshold=0.0) == len(result.contacts)

    def test_contact_labels_are_cross_partner(self, complex_topology):
        ens = _complex_ensemble(complex_topology, binder_rock=0.0)
        result = interface_persistence(ens, "chainID A", "chainID B", method="centroid")
        for c in result.contacts:
            assert c.residue_a.startswith("A_")
            assert c.residue_b.startswith("B_")
            assert c.label == f"{c.residue_a}:{c.residue_b}"

    def test_empty_partner_selection_rejected(self, complex_topology):
        ens = _complex_ensemble(complex_topology)
        with pytest.raises(ValueError, match="matched no atoms"):
            interface_persistence(ens, "chainID A", "chainID Z", method="centroid")

    def test_unknown_method_rejected(self, complex_topology):
        ens = _complex_ensemble(complex_topology)
        with pytest.raises(ValueError, match="Unknown contact method"):
            interface_persistence(ens, "chainID A", "chainID B", method="nonsense")


class TestInterfaceRMSF:
    def test_returns_per_residue_values(self, complex_topology):
        ens = _complex_ensemble(complex_topology, binder_rock=0.0)
        rmsf = interface_rmsf(ens, "chainID B")
        assert len(rmsf) == 6
        assert all(v >= 0.0 for v in rmsf.values())
        assert all(k.startswith("B_") for k in rmsf)

    def test_more_mobile_binder_has_higher_rmsf(self, complex_topology):
        # Align on the target; a rocking binder fluctuates more relative to it.
        calm = interface_rmsf(
            _complex_ensemble(complex_topology, binder_rock=0.0, seed=3),
            "chainID B",
            align_on="chainID A",
        )
        rocking = interface_rmsf(
            _complex_ensemble(complex_topology, binder_rock=4.0, seed=3),
            "chainID B",
            align_on="chainID A",
        )
        assert np.mean(list(rocking.values())) > np.mean(list(calm.values()))

    def test_alignment_removes_rigid_body_translation(self, complex_topology):
        # A binder that is internally rigid but drifts bodily should show ~0 RMSF
        # once aligned on itself.
        target, binder = _complex_base()
        base = np.vstack([target, binder])
        rng = np.random.default_rng(4)
        coords = np.empty((40, len(base), 3), dtype=np.float32)
        for f in range(40):
            c = base.copy()
            c += rng.normal(0, 3.0, 3)  # whole-complex rigid translation
            coords[f] = c
        ens = Ensemble.from_coordinates(coords, complex_topology, selection="all", name="rigid")
        rmsf = interface_rmsf(ens, "chainID B")
        assert max(rmsf.values()) < 0.1  # rigid body: no internal fluctuation


class TestBinderPreorganisation:
    def test_reports_per_residue_difference(self, complex_topology):
        bound = _complex_ensemble(complex_topology, binder_rock=0.0, seed=5)
        unbound = _complex_ensemble(complex_topology, binder_rock=0.0, seed=6)
        diff = binder_preorganisation(bound, unbound, "chainID B")
        assert len(diff) == 6
        assert all(k.startswith("B_") for k in diff)

    def test_rigidifying_on_binding_is_negative(self, complex_topology):
        # Unbound binder is floppy (internal), bound binder is calm -> bound minus
        # unbound RMSF is negative (it ordered on binding).
        bound = _complex_ensemble(complex_topology, binder_rock=0.0, seed=7)
        # make an unbound ensemble with large internal binder wobble
        target, binder = _complex_base()
        base = np.vstack([target, binder])
        rng = np.random.default_rng(8)
        coords = np.empty((60, len(base), 3), dtype=np.float32)
        for f in range(60):
            c = base.copy()
            c[10:] += rng.normal(0, 2.0, (6, 3))  # floppy free binder
            coords[f] = c
        unbound = Ensemble.from_coordinates(coords, complex_topology, selection="all", name="free")
        diff = binder_preorganisation(bound, unbound, "chainID B")
        assert np.mean(list(diff.values())) < 0.0


class TestCompareInterfacePersistence:
    def _group(self, complex_topology, rock, seeds, label):
        return EnsembleGroup(
            [
                _complex_ensemble(complex_topology, binder_rock=rock, seed=s, name=f"{label}{s}")
                for s in seeds
            ],
            label=label,
        )

    def _reference(self, complex_topology):
        ref = interface_persistence(
            _complex_ensemble(complex_topology, binder_rock=0.0),
            "chainID A",
            "chainID B",
            method="centroid",
            min_persistence=0.5,
        )
        return ref.labels()

    def test_compare_two_designs_over_reference(self, complex_topology):
        reference = self._reference(complex_topology)
        good = self._group(complex_topology, 0.0, range(4), "good")
        poor = self._group(complex_topology, 4.0, range(10, 14), "poor")
        report = compare_interface_persistence(
            good,
            poor,
            "chainID A",
            "chainID B",
            reference_contacts=reference,
            method="centroid",
            rng=0,
        )
        assert report.mode == "replicate"
        assert report.n_tests == len(reference)
        # The poor design loses interface contacts, so persistence differs.
        assert report.n_significant >= 1

    def test_empty_reference_rejected(self, complex_topology):
        good = self._group(complex_topology, 0.0, range(2), "good")
        with pytest.raises(ValueError, match="reference_contacts is empty"):
            compare_interface_persistence(
                good, good, "chainID A", "chainID B", reference_contacts=[]
            )

    def test_single_run_uses_bootstrap(self, complex_topology):
        reference = self._reference(complex_topology)
        good = EnsembleGroup.single(
            _complex_ensemble(complex_topology, binder_rock=0.0, seed=0), label="good"
        )
        poor = EnsembleGroup.single(
            _complex_ensemble(complex_topology, binder_rock=4.0, seed=1), label="poor"
        )
        report = compare_interface_persistence(
            good,
            poor,
            "chainID A",
            "chainID B",
            reference_contacts=reference,
            method="centroid",
            rng=0,
        )
        assert report.mode == "bootstrap"
