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


class TestStatisticalComparisonWired:
    """The inferential layer must be attached to the full comparison and written."""

    def _replicated_conditions(self, ca_topology_pdb):
        from tests.conftest import _base_ca_coords

        base = _base_ca_coords(8)

        def group(spread, seeds, label):
            ensembles = [
                Ensemble.from_coordinates(
                    (base + np.random.default_rng(s).normal(0, spread, size=(30, 8, 3))).astype(
                        np.float32
                    ),
                    ca_topology_pdb,
                    selection="name CA",
                    name=f"{label}{s}",
                )
                for s in seeds
            ]
            return EnsembleGroup(ensembles, label=label)

        return group(0.3, range(4), "wt"), group(0.9, range(10, 14), "mut")

    def test_statistical_comparison_is_attached(self, ca_topology_pdb, tmp_path):
        a, b = self._replicated_conditions(ca_topology_pdb)
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "out"))
        results = _run(analyzer, a, b)
        report = results.statistical_comparison
        assert report is not None
        assert report.mode == "replicate"
        assert report.n_tests == 8  # one per residue
        assert report.correction == "fdr_bh"

    def test_per_residue_csv_is_written(self, ca_topology_pdb, tmp_path):
        a, b = self._replicated_conditions(ca_topology_pdb)
        out = tmp_path / "out"
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(out))
        _run(analyzer, a, b)
        csv_path = out / "07_comprehensive_report" / "per_residue_statistics.csv"
        assert csv_path.is_file()
        text = csv_path.read_text()
        assert "effect_size" in text
        assert "qvalue" in text
        assert "# correction: fdr_bh" in text

    def test_correction_method_is_honoured(self, ca_topology_pdb, tmp_path):
        a, b = self._replicated_conditions(ca_topology_pdb)
        analyzer = DifferentialAnalyzer(
            DifferentialConfig(multiple_comparison_correction="bonferroni"), str(tmp_path / "out")
        )
        results = _run(analyzer, a, b)
        assert results.statistical_comparison.correction == "bonferroni"

    def test_single_run_comparison_uses_bootstrap_mode(self, two_conditions, tmp_path):
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "out"))
        results = _run(analyzer, *two_conditions)
        report = results.statistical_comparison
        assert report is not None
        assert report.mode == "bootstrap"
        assert report.caveat is not None


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
    def test_descriptive_views_warn_they_use_only_the_first_replicate(
        self, two_conditions, ca_topology_pdb, caplog
    ):
        """Extra replicates feed the statistics but not the descriptive views.

        The warning must make that split clear: the descriptive comparisons use
        only the first replicate, while the statistical comparison uses all.
        """
        from tests.conftest import _base_ca_coords

        group_a, group_b = two_conditions
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
        messages = " ".join(r.message for r in caplog.records)
        assert "descriptive comparisons" in messages
        assert "statistical comparison uses all" in messages
