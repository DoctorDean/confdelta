"""Tests for confdelta.ensemble — the source-agnostic input model.

The point of Ensemble is that the same analysis works whether the
conformations come from a trajectory, a multi-model PDB, a set of predicted
structures, or an in-memory array. These tests exercise all four sources and
confirm each reaches the existing analysis pipeline through the bridge.
"""

from __future__ import annotations

import numpy as np
import pytest

from confdelta.ensemble import (
    DEFAULT_SELECTION,
    Ensemble,
    EnsembleError,
    EnsembleGroup,
)


class TestFromTrajectory:
    def test_topology_only_is_one_frame(self, ca_topology_pdb):
        ens = Ensemble.from_trajectory(ca_topology_pdb, selection="name CA")
        assert ens.n_frames == 1
        assert ens.n_atoms == 8

    def test_missing_topology_raises_named_error(self, tmp_path):
        with pytest.raises(EnsembleError, match="Topology file not found"):
            Ensemble.from_trajectory(tmp_path / "nope.pdb")

    def test_missing_trajectory_raises_named_error(self, ca_topology_pdb, tmp_path):
        with pytest.raises(EnsembleError, match="Trajectory file not found"):
            Ensemble.from_trajectory(ca_topology_pdb, tmp_path / "nope.xtc")


class TestFromPdbModels:
    def test_each_model_is_a_frame(self, multi_model_pdb):
        ens = Ensemble.from_pdb_models(multi_model_pdb, selection="name CA")
        assert ens.n_frames == 3
        assert ens.n_atoms == 8

    def test_missing_file(self, tmp_path):
        with pytest.raises(EnsembleError, match="PDB file not found"):
            Ensemble.from_pdb_models(tmp_path / "nope.pdb")

    def test_source_records_model_count(self, multi_model_pdb):
        ens = Ensemble.from_pdb_models(multi_model_pdb, selection="name CA")
        assert "3 models" in ens.source


class TestFromStructures:
    def test_structures_are_stacked_as_frames(self, structure_files):
        ens = Ensemble.from_structures(structure_files, selection="name CA")
        assert ens.n_frames == len(structure_files)
        assert ens.n_atoms == 8

    def test_empty_list_raises(self):
        with pytest.raises(EnsembleError, match="at least one"):
            Ensemble.from_structures([])

    def test_missing_structure_named(self, structure_files, tmp_path):
        with pytest.raises(EnsembleError, match="not found"):
            Ensemble.from_structures([*structure_files, tmp_path / "nope.pdb"])

    def test_atom_count_mismatch_is_rejected(self, structure_files, tmp_path):
        """A structure with a different atom count must not be silently stacked."""
        odd = tmp_path / "odd.pdb"
        odd.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
        )
        with pytest.raises(EnsembleError, match="must have the same atoms"):
            Ensemble.from_structures([*structure_files, odd])

    def test_frames_differ_between_structures(self, structure_files):
        """Distinct input files must produce distinct frames, not copies."""
        ens = Ensemble.from_structures(structure_files, selection="name CA")
        coords = ens.coordinates()
        assert not np.allclose(coords[0], coords[1])


class TestFromCoordinates:
    def test_array_becomes_frames(self, ca_topology_pdb):
        coords = np.random.default_rng(0).normal(0, 1, size=(6, 8, 3))
        ens = Ensemble.from_coordinates(coords, ca_topology_pdb, selection="name CA")
        assert ens.n_frames == 6
        assert ens.n_atoms == 8

    def test_single_conformation_promoted_to_one_frame(self, ca_topology_pdb):
        coords = np.random.default_rng(1).normal(0, 1, size=(8, 3))
        ens = Ensemble.from_coordinates(coords, ca_topology_pdb, selection="name CA")
        assert ens.n_frames == 1

    def test_wrong_last_dimension_rejected(self, ca_topology_pdb):
        with pytest.raises(EnsembleError, match=r"n_frames, n_atoms, 3"):
            Ensemble.from_coordinates(np.zeros((4, 8, 2)), ca_topology_pdb)

    def test_atom_count_mismatch_rejected(self, ca_topology_pdb):
        with pytest.raises(EnsembleError, match="must match"):
            Ensemble.from_coordinates(np.zeros((4, 99, 3)), ca_topology_pdb)

    def test_topology_as_universe_is_not_mutated(self, ca_topology_pdb):
        import MDAnalysis as mda

        template = mda.Universe(str(ca_topology_pdb))
        original = template.atoms.positions.copy()
        coords = np.random.default_rng(2).normal(0, 1, size=(3, 8, 3))
        Ensemble.from_coordinates(coords, template, selection="name CA")
        # The caller's universe must be untouched by the ensemble build.
        np.testing.assert_array_equal(template.atoms.positions, original)

    def test_coordinates_round_trip(self, ca_topology_pdb):
        coords = np.random.default_rng(3).normal(0, 5, size=(4, 8, 3)).astype(np.float32)
        ens = Ensemble.from_coordinates(coords, ca_topology_pdb, selection="name CA")
        np.testing.assert_allclose(ens.coordinates(), coords, atol=1e-3)


class TestSelectionAndMetadata:
    def test_empty_selection_is_rejected(self, ca_topology_pdb):
        with pytest.raises(EnsembleError, match="matched no atoms"):
            Ensemble.from_trajectory(ca_topology_pdb, selection="name ZZ")

    def test_default_selection_used_when_unspecified(self, multi_model_pdb):
        # ALA CA atoms are protein heavy atoms, so the default selects them.
        ens = Ensemble.from_pdb_models(multi_model_pdb)
        assert ens.selection == DEFAULT_SELECTION
        assert ens.n_atoms == 8

    def test_metadata_is_carried(self, ca_topology_pdb):
        ens = Ensemble.from_trajectory(
            ca_topology_pdb, selection="name CA", metadata={"mutant": "V82A"}
        )
        assert ens.summary()["metadata"] == {"mutant": "V82A"}


class TestBridge:
    """Every source must reach the existing analysis pipeline unchanged."""

    def _run_network(self, ensemble):
        import contextlib
        import io

        from confdelta.core import AnalysisConfig, NetworkAnalyzer

        sim = ensemble.to_simulation()
        analyzer = NetworkAnalyzer(AnalysisConfig(compute_msm=False))
        with contextlib.redirect_stdout(io.StringIO()):
            analyzer.compute_contact_maps(sim)
            metrics = analyzer.compute_network_metrics(sim)
        return metrics

    def test_bridge_from_multi_model_pdb(self, multi_model_pdb):
        ens = Ensemble.from_pdb_models(multi_model_pdb, selection="name CA")
        metrics = self._run_network(ens)
        assert metrics.n_nodes == 8

    def test_bridge_from_structures(self, structure_files):
        ens = Ensemble.from_structures(structure_files, selection="name CA")
        metrics = self._run_network(ens)
        assert metrics.n_nodes == 8

    def test_bridge_from_coordinates(self, ca_topology_pdb):
        coords = np.random.default_rng(0).normal(0, 1, size=(20, 8, 3))
        # Keep sequential CAs within contact distance so edges form.
        coords[:, :, 0] += np.arange(8) * 3.6
        ens = Ensemble.from_coordinates(coords, ca_topology_pdb, selection="name CA")
        metrics = self._run_network(ens)
        assert metrics.n_nodes == 8

    def test_bridge_produces_a_working_simulation(self, multi_model_pdb):
        ens = Ensemble.from_pdb_models(multi_model_pdb, selection="name CA")
        sim = ens.to_simulation()
        assert sim.n_residues == 8
        assert sim.n_frames == 3
        assert sim.name == ens.name


class TestEnsembleGroup:
    def _ensemble(self, structure_files, name):
        return Ensemble.from_structures(structure_files, selection="name CA", name=name)

    def test_single_wraps_one_ensemble(self, structure_files):
        ens = self._ensemble(structure_files, "wt")
        group = EnsembleGroup.single(ens)
        assert group.n_replicates == 1
        assert group.has_replicates is False
        assert group.label == "wt"

    def test_replicates_are_iterable(self, structure_files):
        group = EnsembleGroup(
            [self._ensemble(structure_files, f"r{i}") for i in range(3)], label="wt"
        )
        assert group.n_replicates == 3
        assert group.has_replicates is True
        assert [e.name for e in group] == ["r0", "r1", "r2"]
        assert len(group) == 3

    def test_empty_group_rejected(self):
        with pytest.raises(EnsembleError, match="at least one"):
            EnsembleGroup([])

    def test_replicates_must_share_atom_count(self, structure_files, ca_topology_pdb):
        big = self._ensemble(structure_files, "big")  # 8 atoms
        small = Ensemble.from_coordinates(
            np.zeros((3, 4, 3)),
            # a 4-atom topology
            _four_atom_topology(ca_topology_pdb),
            selection="name CA",
            name="small",
        )
        with pytest.raises(EnsembleError, match="different selected atom counts"):
            EnsembleGroup([big, small], label="wt")

    def test_indexing(self, structure_files):
        group = EnsembleGroup(
            [self._ensemble(structure_files, f"r{i}") for i in range(2)], label="wt"
        )
        assert group[0].name == "r0"
        assert group[1].name == "r1"

    def test_summary_structure(self, structure_files):
        group = EnsembleGroup.single(self._ensemble(structure_files, "wt"))
        summary = group.summary()
        assert summary["label"] == "wt"
        assert summary["n_replicates"] == 1
        assert len(summary["replicates"]) == 1


def _four_atom_topology(reference_pdb):
    """Build a 4-atom topology Universe for the atom-count mismatch test."""
    import MDAnalysis as mda

    universe = mda.Universe(str(reference_pdb))
    return mda.Merge(universe.atoms[:4])
