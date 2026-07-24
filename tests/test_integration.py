"""End-to-end tests of the full comparison pipeline.

Nothing else exercises DifferentialAnalyzer from input to written output: the
differential tests hit the comparators with mock objects, and the audit found
the full runner returned the wrong type (a summary dict where an MDSimulation
was expected), so it had never actually run. These tests drive a complete
two-condition comparison built from in-memory ensembles and assert the
descriptive results and honest output files.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from confdelta import (
    AnalysisConfig,
    DifferentialAnalyzer,
    DifferentialConfig,
    Ensemble,
    EnsembleGroup,
)

# These drive the full analysis pipeline (contact maps, DCCM, PCA, energy
# landscapes, allosteric paths, figure generation) twice per test, so each
# takes a few seconds. Marked slow so `pytest -m "not slow"` skips them in the
# fast dev loop; CI runs the whole suite.
pytestmark = pytest.mark.slow


@pytest.fixture
def two_conditions(ca_topology_pdb):
    """Two ensembles of the same 8-CA system with different fluctuations."""
    from tests.conftest import _base_ca_coords  # local import; test-only helper

    base = _base_ca_coords(8)
    rng = np.random.default_rng(0)
    coords_a = base + rng.normal(0, 0.3, size=(40, 8, 3)).astype(np.float32)
    coords_b = base + rng.normal(0, 0.7, size=(40, 8, 3)).astype(np.float32)
    a = Ensemble.from_coordinates(coords_a, ca_topology_pdb, selection="name CA", name="wt")
    b = Ensemble.from_coordinates(coords_b, ca_topology_pdb, selection="name CA", name="mut")
    return EnsembleGroup.single(a), EnsembleGroup.single(b)


def _run(analyzer, group_a, group_b):
    config = AnalysisConfig(compute_msm=False, pca_components=4, segments=1)
    with contextlib.redirect_stdout(io.StringIO()):
        return analyzer.run_ensemble_comparison(group_a, group_b, config)


class TestEnsembleComparisonEndToEnd:
    def test_runs_and_returns_results(self, two_conditions, tmp_path):
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "out"))
        results = _run(analyzer, *two_conditions)
        assert results.simulation_names == ("wt", "mut")

    def test_network_comparison_is_populated(self, two_conditions, tmp_path):
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "out"))
        results = _run(analyzer, *two_conditions)
        net = results.network_comparison
        assert net is not None
        # Same system, so no nodes appear or vanish.
        assert net.nodes_added == []
        assert net.nodes_removed == []
        assert -1.0 <= net.modularity_change <= 1.0

    def test_dynamics_difference_matrix_has_right_shape(self, two_conditions, tmp_path):
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "out"))
        results = _run(analyzer, *two_conditions)
        dyn = results.dynamics_comparison
        assert dyn is not None
        assert dyn.dccm_difference_matrix.shape == (8, 8)

    def test_no_fabricated_summary_fields_exist(self, two_conditions, tmp_path):
        """The removed executive summary must not have crept back onto results."""
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "out"))
        results = _run(analyzer, *two_conditions)
        for gone in (
            "executive_summary",
            "overall_similarity_scores",
            "most_significant_changes",
        ):
            assert not hasattr(results, gone)

    def test_only_honest_csvs_are_written(self, two_conditions, tmp_path):
        """No fabricated-statistics CSVs, and the DCCM file uses the honest name."""
        out = tmp_path / "out"
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(out))
        _run(analyzer, *two_conditions)

        written = {p.name for p in out.rglob("*.csv")}
        # Phantom files the old docs promised but no code writes.
        for phantom in (
            "statistical_significance_tests.csv",
            "overall_similarity_scores.csv",
            "network_property_changes.csv",
            "significant_correlation_changes.csv",
        ):
            assert phantom not in written
        # The renamed, honest file is present.
        assert "large_correlation_changes.csv" in written

    def test_identical_conditions_give_zero_dccm_difference(self, ca_topology_pdb, tmp_path):
        from tests.conftest import _base_ca_coords

        coords = _base_ca_coords(8) + np.random.default_rng(1).normal(
            0, 0.3, size=(30, 8, 3)
        ).astype(np.float32)
        ens = Ensemble.from_coordinates(coords, ca_topology_pdb, selection="name CA", name="x")
        # Compare an ensemble against itself.
        a = EnsembleGroup.single(ens, label="a")
        b = EnsembleGroup.single(
            Ensemble.from_coordinates(coords, ca_topology_pdb, selection="name CA", name="y"),
            label="b",
        )
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "out"))
        results = _run(analyzer, a, b)
        np.testing.assert_allclose(
            results.dynamics_comparison.dccm_difference_matrix, 0.0, atol=1e-6
        )


class TestReplicateWarning:
    def test_extra_replicates_warn_and_use_the_first(self, two_conditions, ca_topology_pdb, caplog):
        from tests.conftest import _base_ca_coords

        group_a, group_b = two_conditions
        # Give condition A a second replicate.
        extra = Ensemble.from_coordinates(
            _base_ca_coords(8)
            + np.random.default_rng(9).normal(0, 0.3, size=(20, 8, 3)).astype(np.float32),
            ca_topology_pdb,
            selection="name CA",
            name="wt_rep2",
        )
        group_a = EnsembleGroup([group_a[0], extra], label="wt")
        assert group_a.has_replicates

        import tempfile

        analyzer = DifferentialAnalyzer(DifferentialConfig(), tempfile.mkdtemp())
        with caplog.at_level("WARNING", logger="confdelta"):
            with contextlib.redirect_stdout(io.StringIO()):
                analyzer.run_ensemble_comparison(
                    group_a, group_b, AnalysisConfig(compute_msm=False, segments=1)
                )
        assert any("only the first is used" in r.message for r in caplog.records)
