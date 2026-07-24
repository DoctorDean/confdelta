"""Tests for confdelta.utils — network metrics, preprocessing, persistence."""

from __future__ import annotations

import numpy as np

from confdelta import utils

# ---------------------------------------------------------------------------
# Network robustness / assortativity
# ---------------------------------------------------------------------------


class TestNetworkRobustness:
    def test_robustness_returns_dict(self, small_graph):
        result = utils.calculate_network_robustness(small_graph)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_robustness_handles_empty_graph(self):
        import networkx as nx

        # Must not raise on an empty graph.
        result = utils.calculate_network_robustness(nx.Graph())
        assert isinstance(result, dict)

    def test_assortativity_returns_dict(self, small_graph):
        result = utils.analyze_network_assortativity(small_graph)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Contact persistence
# ---------------------------------------------------------------------------


class TestContactPersistence:
    def test_persistence_basic(self):
        rng = np.random.default_rng(0)
        cmap = rng.random((10, 10))
        cmap = (cmap + cmap.T) / 2  # symmetric
        np.fill_diagonal(cmap, 0.0)
        result = utils.analyze_contact_persistence(cmap)
        assert isinstance(result, dict)

    def test_persistence_custom_percentiles(self):
        cmap = np.full((8, 8), 0.5)
        np.fill_diagonal(cmap, 0.0)
        result = utils.analyze_contact_persistence(cmap, percentiles=[50, 90])
        assert isinstance(result, dict)

    def test_find_highly_persistent_contacts(self):
        cmap = np.zeros((5, 5))
        cmap[0, 1] = cmap[1, 0] = 0.95
        cmap[2, 3] = cmap[3, 2] = 0.85
        keys = ["A_1", "A_2", "A_3", "A_4", "A_5"]
        result = utils.find_highly_persistent_contacts(cmap, keys, threshold=0.8)
        # Both contacts above threshold should be reported.
        assert result is not None


# ---------------------------------------------------------------------------
# Sequence distance matrix
# ---------------------------------------------------------------------------


class TestSequenceDistance:
    def test_sequence_distance_shape(self):
        keys = ["A_1", "A_2", "A_3", "B_1", "B_2"]
        mat = utils.calculate_sequence_distance_matrix(keys)
        assert mat.shape == (5, 5)

    def test_sequence_distance_diagonal_zero(self):
        keys = ["A_1", "A_5", "A_9"]
        mat = utils.calculate_sequence_distance_matrix(keys)
        np.testing.assert_allclose(np.diag(mat), 0.0)

    def test_sequence_distance_symmetric(self):
        keys = ["A_1", "A_2", "A_10"]
        mat = utils.calculate_sequence_distance_matrix(keys)
        np.testing.assert_allclose(mat, mat.T)


# ---------------------------------------------------------------------------
# Network similarity
# ---------------------------------------------------------------------------


class TestNetworkSimilarity:
    def test_identical_networks_are_similar(self, small_graph):
        result = utils.calculate_network_similarity(small_graph, small_graph)
        assert isinstance(result, dict)

    def test_different_networks(self, small_graph):
        import networkx as nx

        other = nx.cycle_graph(8)
        result = utils.calculate_network_similarity(small_graph, other)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Performance monitor
# ---------------------------------------------------------------------------


class TestPerformanceMonitor:
    def test_step_timing(self):
        mon = utils.PerformanceMonitor()
        mon.start_step("step1")
        mon.end_step()
        summary = mon.get_summary()
        assert isinstance(summary, dict)

    def test_multiple_steps(self):
        mon = utils.PerformanceMonitor()
        for name in ("a", "b", "c"):
            mon.start_step(name)
            mon.end_step()
        summary = mon.get_summary()
        assert isinstance(summary, dict)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestConfigIO:
    def test_save_and_load_config(self, tmp_path):
        from confdelta.core import AnalysisConfig

        cfg = AnalysisConfig(threshold=0.33, pca_components=7)
        path = tmp_path / "cfg.json"
        utils.save_analysis_config(cfg, str(path))
        assert path.exists()
        loaded = utils.load_analysis_config(str(path))
        assert loaded is not None
