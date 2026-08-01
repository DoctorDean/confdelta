"""Tests for confdelta.geometry — user-defined distance and angle collective variables."""

from __future__ import annotations

import numpy as np
import pytest

from confdelta import Ensemble, EnsembleGroup, compare_ensemble_groups, geometric_features


def _ca_ensemble(tmp_path, atoms, *, n_frames=20, jitter=0.0, seed=0, name="g"):
    """Build a CA-only ensemble from ``atoms`` = list of ``(chain, resid, (x, y, z))``."""
    lines = []
    for i, (chain, resid, (x, y, z)) in enumerate(atoms, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA {chain}{resid:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    topo = tmp_path / f"{name}.pdb"
    topo.write_text("\n".join(lines) + "\n")

    base = np.array([xyz for _, _, xyz in atoms], dtype=np.float32)
    rng = np.random.default_rng(seed)
    coords = base[None] + rng.normal(0.0, jitter, (n_frames, len(atoms), 3)).astype(np.float32)
    return Ensemble.from_coordinates(coords, str(topo), selection="name CA", name=name)


# A 3-4-5 right triangle: residues 1,2,3 of chain A at known positions.
TRIANGLE = [("A", 1, (0.0, 0.0, 0.0)), ("A", 2, (3.0, 0.0, 0.0)), ("A", 3, (3.0, 4.0, 0.0))]


class TestGeometricFeatureValues:
    def test_distances_and_angle_are_exact(self, tmp_path):
        ens = _ca_ensemble(tmp_path, TRIANGLE)  # static (jitter=0)
        ext = geometric_features(distances=[(1, 2), (1, 3)], angles=[(1, 2, 3)])
        labels, values = ext(ens)
        assert labels == ["dist:A_1-A_2", "dist:A_1-A_3", "angle:A_1-A_2-A_3"]
        assert values.shape == (20, 3)
        np.testing.assert_allclose(values[:, 0], 3.0, atol=1e-4)  # 1-2
        np.testing.assert_allclose(values[:, 1], 5.0, atol=1e-4)  # 1-3 (hypotenuse)
        np.testing.assert_allclose(values[:, 2], 90.0, atol=1e-3)  # angle at residue 2

    def test_chain_resid_labels_resolve(self, tmp_path):
        ens = _ca_ensemble(tmp_path, TRIANGLE)
        labels, values = geometric_features(distances=[("A_1", "A_3")])(ens)
        assert labels == ["dist:A_1-A_3"]
        np.testing.assert_allclose(values[:, 0], 5.0, atol=1e-4)

    def test_non_ca_atom_selection(self, tmp_path):
        # With a CA-only ensemble, asking for a non-existent atom fails clearly.
        ens = _ca_ensemble(tmp_path, TRIANGLE)
        with pytest.raises(ValueError, match="no atom named 'CB'"):
            geometric_features(distances=[(1, 2)], atom="CB")(ens)


class TestResidueResolution:
    def test_ambiguous_residue_number_rejected(self, tmp_path):
        atoms = [
            ("A", 1, (0.0, 0, 0)),
            ("A", 2, (3.0, 0, 0)),
            ("B", 1, (0, 5.0, 0)),
            ("B", 2, (3.0, 5.0, 0)),
        ]
        ens = _ca_ensemble(tmp_path, atoms, name="dimer")
        with pytest.raises(ValueError, match="ambiguous across chains"):
            geometric_features(distances=[(1, 2)])(ens)
        # the chain-qualified labels are unambiguous
        labels, _ = geometric_features(distances=[("A_1", "B_1")])(ens)
        assert labels == ["dist:A_1-B_1"]

    def test_missing_residue_rejected(self, tmp_path):
        ens = _ca_ensemble(tmp_path, TRIANGLE)
        with pytest.raises(ValueError, match="No residue numbered 99"):
            geometric_features(distances=[(1, 99)])(ens)


class TestSpecValidation:
    def test_empty_spec_rejected(self):
        with pytest.raises(ValueError, match="at least one distance or angle"):
            geometric_features()

    def test_wrong_distance_arity_rejected(self):
        with pytest.raises(ValueError, match="exactly two residues"):
            geometric_features(distances=[(1, 2, 3)])

    def test_wrong_angle_arity_rejected(self):
        with pytest.raises(ValueError, match="exactly three residues"):
            geometric_features(angles=[(1, 2)])


class TestComparisonIntegration:
    def test_geometric_cv_flows_through_compare(self, tmp_path):
        # Two conditions differing in the 1-3 distance; the CV comparison should run
        # and label the report by the CV, not by residue.
        a = _ca_ensemble(tmp_path, TRIANGLE, jitter=0.2, seed=1, name="a")
        moved = [("A", 1, (0.0, 0, 0)), ("A", 2, (3.0, 0, 0)), ("A", 3, (3.0, 8.0, 0))]
        b = _ca_ensemble(tmp_path, moved, jitter=0.2, seed=2, name="b")
        ext = geometric_features(distances=[(1, 3)], angles=[(1, 2, 3)])
        report = compare_ensemble_groups(
            EnsembleGroup([a], label="a"), EnsembleGroup([b], label="b"), feature=ext, rng=0
        )
        assert report.mode == "bootstrap"
        assert [f.feature for f in report.features] == ["dist:A_1-A_3", "angle:A_1-A_2-A_3"]
        dist_cv = next(f for f in report.features if f.feature == "dist:A_1-A_3")
        # 1-3 distance moved from 5 to ~9; the two conditions clearly differ.
        assert abs(dist_cv.difference) > 1.0
