"""Tests for the RMSD-to-reference feature and the custom-observable helper."""

from __future__ import annotations

import numpy as np
import pytest

from confdelta import (
    Ensemble,
    EnsembleGroup,
    compare_ensemble_groups,
    feature_from_positions,
    per_residue_rmsd,
)


def _ens(tmp_path, base, frames, name):
    """CA-only ensemble; *base* is (n_res, 3), *frames* is (n_frames, n_res, 3)."""
    lines = [
        f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        for i, (x, y, z) in enumerate(base)
    ]
    topo = tmp_path / f"{name}.pdb"
    topo.write_text("\n".join(lines) + "\nEND\n")
    return Ensemble.from_coordinates(
        np.asarray(frames, dtype=np.float32), str(topo), selection="name CA", name=name
    )


def _line(n=6, spacing=4.0):
    base = np.zeros((n, 3), dtype=np.float32)
    base[:, 0] = np.arange(n) * spacing
    return base


class TestPerResidueRmsd:
    def test_identical_to_reference_is_zero(self, tmp_path):
        base = _line()
        ref = _ens(tmp_path, base, np.stack([base] * 4), "ref")
        ens = _ens(tmp_path, base, np.stack([base] * 6), "ens")
        _, values = per_residue_rmsd(ref)(ens)
        assert np.sqrt(values.mean(0)).max() < 0.01

    def test_rigid_translation_is_removed(self, tmp_path):
        base = _line()
        ref = _ens(tmp_path, base, np.stack([base] * 4), "ref")
        rng = np.random.default_rng(0)
        frames = np.stack([base + rng.normal(0, 5.0, 3).astype(np.float32) for _ in range(8)])
        ens = _ens(tmp_path, base, frames, "shifted")
        _, values = per_residue_rmsd(ref)(ens)
        assert np.sqrt(values.mean(0)).max() < 0.01  # bodily drift removed by alignment

    def test_displaced_residue_has_largest_rmsd(self, tmp_path):
        base = _line()
        ref = _ens(tmp_path, base, np.stack([base] * 4), "ref")
        moved = np.stack([base] * 6).copy()
        moved[:, 3, 1] += 6.0  # residue 4 (index 3) pushed out in y
        ens = _ens(tmp_path, base, moved, "moved")
        _, values = per_residue_rmsd(ref)(ens)
        assert int(np.argmax(values.mean(0))) == 3

    def test_reference_mismatch_rejected(self, tmp_path):
        ref = _ens(tmp_path, _line(6), np.stack([_line(6)] * 3), "ref6")
        ens = _ens(tmp_path, _line(5), np.stack([_line(5)] * 3), "ens5")
        with pytest.raises(ValueError, match="different residues"):
            per_residue_rmsd(ref)(ens)


class TestFeatureFromPositions:
    def test_known_function(self, tmp_path):
        base = _line(4)
        ens = _ens(tmp_path, base, np.stack([base] * 5), "e")

        def mean_x(pos):
            return np.array([pos[:, 0].mean()])

        labels, values = feature_from_positions(mean_x, labels=["mean_x"])(ens)
        assert labels == ["mean_x"]
        np.testing.assert_allclose(values[:, 0], base[:, 0].mean(), atol=1e-3)

    def test_wrong_output_length_rejected(self, tmp_path):
        ens = _ens(tmp_path, _line(4), np.stack([_line(4)] * 3), "e")
        bad = feature_from_positions(lambda pos: np.array([1.0, 2.0]), labels=["only_one"])
        with pytest.raises(ValueError, match="returned 2 values but 1 labels"):
            bad(ens)

    def test_empty_labels_rejected(self):
        with pytest.raises(ValueError, match="at least one label"):
            feature_from_positions(lambda pos: np.array([]), labels=[])

    def test_flows_through_compare(self, tmp_path):
        rng = np.random.default_rng(1)
        a_frames = np.stack([_line(4) + rng.normal(0, 0.4, (4, 3)) for _ in range(12)])
        spread = _line(4).copy()
        spread[:, 1] += np.array([0, 5, 0, 5], dtype=np.float32)  # a wider y spread
        b_frames = np.stack([spread + rng.normal(0, 0.4, (4, 3)) for _ in range(12)])
        a = _ens(tmp_path, _line(4), a_frames, "a")
        b = _ens(tmp_path, spread, b_frames, "b")

        def spread_y(pos):
            return np.array([pos[:, 1].max() - pos[:, 1].min()])

        ext = feature_from_positions(spread_y, labels=["y_spread"])
        report = compare_ensemble_groups(
            EnsembleGroup([a], label="a"), EnsembleGroup([b], label="b"), feature=ext, rng=0
        )
        assert [f.feature for f in report.features] == ["y_spread"]
