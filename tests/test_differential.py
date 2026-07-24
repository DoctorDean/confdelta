"""Tests for confdelta.differential — comparators and differential analysis.

This replaces the original print-based test_differential_implementation.py
script with proper pytest assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from confdelta.differential import (
    AllostericComparator,
    DifferentialAnalyzer,
    DifferentialConfig,
    DynamicsComparator,
    NetworkComparator,
)

# ---------------------------------------------------------------------------
# Configuration & analyzer construction
# ---------------------------------------------------------------------------


class TestDifferentialConfig:
    def test_defaults(self):
        cfg = DifferentialConfig()
        assert cfg.compare_networks is True
        assert cfg.significance_threshold == pytest.approx(0.05)
        assert cfg.multiple_comparison_correction in {"fdr_bh", "bonferroni", "none"}

    def test_custom_values(self):
        cfg = DifferentialConfig(
            compare_kinetics=False,
            bootstrap_iterations=200,
            correlation_change_threshold=0.15,
        )
        assert cfg.compare_kinetics is False
        assert cfg.bootstrap_iterations == 200
        assert cfg.correlation_change_threshold == pytest.approx(0.15)


class TestDifferentialAnalyzer:
    def test_initialization_creates_subdirs(self, tmp_path):
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "diff_out"))
        assert analyzer.output_dir is not None
        assert isinstance(analyzer.subdirs, dict)
        assert len(analyzer.subdirs) > 0

    def test_comparators_are_constructed(self, tmp_path):
        analyzer = DifferentialAnalyzer(DifferentialConfig(), str(tmp_path / "diff_out"))
        assert isinstance(analyzer.network_comparator, NetworkComparator)
        assert isinstance(analyzer.dynamics_comparator, DynamicsComparator)
        assert isinstance(analyzer.allosteric_comparator, AllostericComparator)


# ---------------------------------------------------------------------------
# Network comparator
# ---------------------------------------------------------------------------


class TestNetworkComparator:
    @staticmethod
    def _metrics_with_graph(graph):
        """Wrap a graph in a minimal object exposing a .network attribute."""
        return type("M", (), {"network": graph})()

    def test_identical_networks_have_no_changes(self, small_graph):
        comparator = NetworkComparator(DifferentialConfig(perform_statistical_tests=False))
        m = self._metrics_with_graph(small_graph)
        result = comparator.compare_networks(m, m)
        assert result.nodes_added == []
        assert result.nodes_removed == []

    def test_added_and_removed_nodes_detected(self, small_graph):

        comparator = NetworkComparator(DifferentialConfig(perform_statistical_tests=False))
        modified = small_graph.copy()
        modified.add_node("C_1")
        modified.remove_node("A_1")

        result = comparator.compare_networks(
            self._metrics_with_graph(small_graph),
            self._metrics_with_graph(modified),
        )
        assert "C_1" in result.nodes_added
        assert "A_1" in result.nodes_removed

    def test_missing_network_returns_empty_comparison(self):
        comparator = NetworkComparator(DifferentialConfig())
        result = comparator.compare_networks(None, None)
        # Must degrade gracefully, not raise.
        assert result.nodes_added == []
        assert result.nodes_removed == []


# ---------------------------------------------------------------------------
# Dynamics comparator
# ---------------------------------------------------------------------------


class TestDynamicsComparator:
    @staticmethod
    def _dyn(n=6, seed=0):
        rng = np.random.default_rng(seed)
        dccm = rng.uniform(-1, 1, size=(n, n))
        dccm = (dccm + dccm.T) / 2
        np.fill_diagonal(dccm, 1.0)
        return {
            "dccm_matrix": dccm,
            "pca_eigenvalues": np.array([0.5, 0.25, 0.12, 0.08, 0.03, 0.02]),
            "pca_eigenvectors": rng.random((n, n)),
        }

    def test_dccm_difference_matrix_shape(self):
        comparator = DynamicsComparator(DifferentialConfig())
        dyn1 = self._dyn(seed=1)
        dyn2 = self._dyn(seed=2)
        result = comparator.compare_dynamics(dyn1, dyn2)
        assert result.dccm_difference_matrix.shape == (6, 6)

    def test_identical_dynamics_zero_difference(self):
        comparator = DynamicsComparator(DifferentialConfig())
        dyn = self._dyn(seed=3)
        result = comparator.compare_dynamics(dyn, dyn)
        np.testing.assert_allclose(result.dccm_difference_matrix, 0.0, atol=1e-9)

    def test_pca_variance_changes_present(self):
        comparator = DynamicsComparator(DifferentialConfig())
        result = comparator.compare_dynamics(self._dyn(seed=4), self._dyn(seed=5))
        assert result.pca_variance_changes is not None


# ---------------------------------------------------------------------------
# Allosteric comparator
# ---------------------------------------------------------------------------


class TestAllostericComparator:
    @staticmethod
    def _ap(eff_scale=1.0):
        return {
            "communication_efficiency": {
                "A_1_A_2": 0.8 * eff_scale,
                "A_1_B_1": 0.6 * eff_scale,
                "A_2_B_2": 0.4 * eff_scale,
            },
            "allosteric_hotspots": [
                {"node": "A_1", "hotspot_score": 0.9, "frequency": 10},
                {"node": "A_2", "hotspot_score": 0.7, "frequency": 8},
            ],
            "pathways": [
                {
                    "source": "A_1",
                    "target": "B_1",
                    "efficiency": 0.8,
                    "path": ["A_1", "A_2", "B_1"],
                },
            ],
        }

    def test_allosteric_comparison_runs(self):
        comparator = AllostericComparator(DifferentialConfig())
        m1 = type("M", (), {"allosteric_pathways": self._ap(1.0)})()
        m2 = type("M", (), {"allosteric_pathways": self._ap(0.75)})()
        result = comparator.compare_allosteric(m1, m2)
        assert result is not None
        assert hasattr(result, "pathway_disruption_analysis")

    def test_hotspot_ranking_changes_tracked(self):
        comparator = AllostericComparator(DifferentialConfig())
        m1 = type("M", (), {"allosteric_pathways": self._ap(1.0)})()
        m2 = type("M", (), {"allosteric_pathways": self._ap(0.5)})()
        result = comparator.compare_allosteric(m1, m2)
        assert result.hotspot_ranking_changes is not None


# ---------------------------------------------------------------------------
# CLI integration surface
# ---------------------------------------------------------------------------


def test_cli_exposes_differential_entrypoint():
    """The CLI module must expose the comprehensive differential runner."""
    from confdelta import cli

    assert hasattr(cli, "run_comprehensive_differential_analysis")
    assert callable(cli.main)
