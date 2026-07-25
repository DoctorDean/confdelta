"""Tests for confdelta.features — per-residue extraction and the group entry point."""

from __future__ import annotations

import numpy as np
import pytest

from confdelta import Ensemble, EnsembleGroup
from confdelta.features import (
    compare_ensemble_groups,
    per_residue_contact_counts,
)
from tests.conftest import _base_ca_coords


def _ensemble(ca_topology_pdb, *, spread, n_frames, seed, name):
    """A CA ensemble whose residues fluctuate around a compact base structure."""
    base = _base_ca_coords(8)
    coords = base + np.random.default_rng(seed).normal(0, spread, size=(n_frames, 8, 3))
    return Ensemble.from_coordinates(
        coords.astype(np.float32), ca_topology_pdb, selection="name CA", name=name
    )


class TestPerResidueContactCounts:
    def test_shape_and_labels(self, ca_topology_pdb):
        ens = _ensemble(ca_topology_pdb, spread=0.3, n_frames=10, seed=0, name="e")
        labels, values = per_residue_contact_counts(ens)
        assert labels == [f"A_{i}" for i in range(1, 9)]
        assert values.shape == (10, 8)

    def test_counts_are_non_negative_integers(self, ca_topology_pdb):
        ens = _ensemble(ca_topology_pdb, spread=0.3, n_frames=5, seed=1, name="e")
        _, values = per_residue_contact_counts(ens)
        assert values.min() >= 0
        np.testing.assert_array_equal(values, np.round(values))

    def test_sequence_neighbours_are_excluded(self, ca_topology_pdb):
        """Adjacent residues in sequence must not count as contacts."""
        # A straight chain 3.6 A apart: with min_sequence_separation=2 and an
        # 8 A cutoff, only the excluded i,i+1 neighbours are within range, so
        # every residue should report zero contacts.
        base = _base_ca_coords(8)
        base[:, 1] = 0.0  # remove the wiggle -> pure straight line on x
        base[:, 2] = 0.0
        ens = Ensemble.from_coordinates(
            base[None, :, :].astype(np.float32),
            ca_topology_pdb,
            selection="name CA",
            name="line",
        )
        _, values = per_residue_contact_counts(ens, cutoff=8.0, min_sequence_separation=2)
        # i to i+2 is 7.2 A (< 8), i to i+1 is 3.6 (excluded). So each interior
        # residue sees its i-2 and i+2 neighbours: check the middle residue.
        assert values[0, 3] == 2  # residue A_4 sees A_2 and A_6
        # With a tighter cutoff below 7.2, no eligible contacts remain.
        _, tight = per_residue_contact_counts(ens, cutoff=5.0, min_sequence_separation=2)
        assert tight.sum() == 0

    def test_more_compact_structure_has_more_contacts(self, ca_topology_pdb):
        base = _base_ca_coords(8)
        compact = base * 0.5  # halve all coordinates -> residues closer together
        spread_out = base * 2.0
        e_compact = Ensemble.from_coordinates(
            compact[None].astype(np.float32), ca_topology_pdb, selection="name CA", name="c"
        )
        e_spread = Ensemble.from_coordinates(
            spread_out[None].astype(np.float32), ca_topology_pdb, selection="name CA", name="s"
        )
        _, compact_counts = per_residue_contact_counts(e_compact)
        _, spread_counts = per_residue_contact_counts(e_spread)
        assert compact_counts.sum() > spread_counts.sum()


class TestCompareEnsembleGroups:
    def test_replicate_mode_when_both_have_replicates(self, ca_topology_pdb):
        a = EnsembleGroup(
            [
                _ensemble(ca_topology_pdb, spread=0.3, n_frames=20, seed=s, name=f"a{s}")
                for s in range(3)
            ],
            label="A",
        )
        b = EnsembleGroup(
            [
                _ensemble(ca_topology_pdb, spread=0.6, n_frames=20, seed=10 + s, name=f"b{s}")
                for s in range(3)
            ],
            label="B",
        )
        report = compare_ensemble_groups(a, b, rng=0)
        assert report.mode == "replicate"
        assert report.n_tests == 8  # one per residue
        assert [f.feature for f in report.features] == [f"A_{i}" for i in range(1, 9)]

    def test_three_replicates_are_underpowered_but_report_effects(self, ca_topology_pdb):
        a = EnsembleGroup(
            [
                _ensemble(ca_topology_pdb, spread=0.2, n_frames=20, seed=s, name=f"a{s}")
                for s in range(3)
            ],
            label="A",
        )
        b = EnsembleGroup(
            [
                _ensemble(ca_topology_pdb, spread=1.5, n_frames=20, seed=10 + s, name=f"b{s}")
                for s in range(3)
            ],
            label="B",
        )
        report = compare_ensemble_groups(a, b, rng=0)
        assert report.underpowered  # 3 v 3 floor is 0.10
        assert report.min_pvalue == pytest.approx(0.10)

    def test_bootstrap_mode_for_single_runs(self, ca_topology_pdb):
        a = EnsembleGroup.single(
            _ensemble(ca_topology_pdb, spread=0.3, n_frames=200, seed=0, name="a"), label="A"
        )
        b = EnsembleGroup.single(
            _ensemble(ca_topology_pdb, spread=0.9, n_frames=200, seed=1, name="b"), label="B"
        )
        report = compare_ensemble_groups(a, b, rng=0)
        assert report.mode == "bootstrap"
        assert report.caveat is not None
        assert report.n_tests == 8

    def test_mismatched_systems_rejected(self, ca_topology_pdb, tmp_path):
        from tests.conftest import _write_ca_pdb

        big = EnsembleGroup.single(
            _ensemble(ca_topology_pdb, spread=0.3, n_frames=20, seed=0, name="big"), label="A"
        )
        # A 6-residue system: different residue labels.
        small_top = _write_ca_pdb(tmp_path / "small.pdb", _base_ca_coords(6))
        small_ens = Ensemble.from_coordinates(
            (_base_ca_coords(6) + np.random.default_rng(0).normal(0, 0.3, (20, 6, 3))).astype(
                np.float32
            ),
            small_top,
            selection="name CA",
            name="small",
        )
        small = EnsembleGroup.single(small_ens, label="B")
        with pytest.raises(ValueError, match="different residues"):
            compare_ensemble_groups(big, small)

    def test_unknown_feature_rejected(self, ca_topology_pdb):
        g = EnsembleGroup.single(
            _ensemble(ca_topology_pdb, spread=0.3, n_frames=20, seed=0, name="a")
        )
        with pytest.raises(ValueError, match="Unknown feature"):
            compare_ensemble_groups(g, g, feature="nonsense")

    def test_mixed_replication_warns_and_uses_bootstrap(self, ca_topology_pdb, caplog):
        a = EnsembleGroup(
            [
                _ensemble(ca_topology_pdb, spread=0.3, n_frames=150, seed=s, name=f"a{s}")
                for s in range(3)
            ],
            label="A",
        )
        b = EnsembleGroup.single(
            _ensemble(ca_topology_pdb, spread=0.6, n_frames=150, seed=9, name="b"), label="B"
        )
        with caplog.at_level("WARNING", logger="confdelta"):
            report = compare_ensemble_groups(a, b, rng=0)
        assert report.mode == "bootstrap"
        assert any("Only one condition has replicates" in r.message for r in caplog.records)
