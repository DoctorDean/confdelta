"""Tests for mdcompare.core — configs, simulation loading, network analysis."""

from __future__ import annotations

import numpy as np
import pytest

from mdcompare.core import (
    AnalysisConfig,
    MDSimulation,
    NetworkAnalyzer,
    NetworkMetrics,
    SimulationConfig,
)

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


class TestConfigs:
    def test_analysis_config_defaults(self):
        cfg = AnalysisConfig()
        assert cfg.cutoffs is not None
        assert "all_atom" in cfg.cutoffs
        assert cfg.interaction_types == ["distance"]
        assert cfg.threshold == pytest.approx(0.2)

    def test_analysis_config_post_init_fills_none(self):
        cfg = AnalysisConfig(cutoffs=None, interaction_types=None)
        assert isinstance(cfg.cutoffs, dict)
        assert isinstance(cfg.interaction_types, list)

    def test_simulation_config_metadata_default(self):
        cfg = SimulationConfig(name="s", topology="t.pdb", trajectory="x.xtc")
        assert cfg.metadata == {}
        assert cfg.selection  # non-empty default selection

    def test_network_metrics_defaults(self):
        m = NetworkMetrics()
        assert m.n_nodes == 0
        assert m.n_edges == 0
        assert m.density == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Chain-label resolution (regression tests for the chainID load bug)
# ---------------------------------------------------------------------------


class TestChainResolution:
    def test_chainids_are_used_when_present(self, synthetic_universe):
        cfg = SimulationConfig(name="s", topology="m", trajectory="m", selection="all")
        sim = MDSimulation(cfg)
        sim._universe = synthetic_universe
        sim._atoms = synthetic_universe.select_atoms("all")
        labels = sim._resolve_chain_labels()
        assert len(labels) == len(sim._atoms)
        assert set(labels) == {"A", "B"}

    def test_chainless_universe_falls_back_to_single_chain(self, chainless_universe):
        """A Universe with only a segid must not crash; it maps to one chain."""
        cfg = SimulationConfig(name="s", topology="m", trajectory="m", selection="all")
        sim = MDSimulation(cfg)
        sim._universe = chainless_universe
        sim._atoms = chainless_universe.select_atoms("all")
        sim._setup_residue_mapping()  # previously raised NoDataError
        assert sim.n_chains == 1
        assert sim.n_residues == 12

    def test_residue_mapping_covers_all_atoms(self, synthetic_simulation):
        sim = synthetic_simulation
        assert len(sim._atom_to_residue_map) == len(sim._atoms)
        assert sim._atom_to_residue_map.max() < sim.n_residues
        assert sim._atom_to_residue_map.min() >= 0

    def test_residue_keys_have_chain_prefix(self, synthetic_simulation):
        for key in synthetic_simulation.unique_residue_keys:
            chain, _, resid = key.partition("_")
            assert chain in {"A", "B"}
            assert resid.isdigit()


# ---------------------------------------------------------------------------
# Simulation properties
# ---------------------------------------------------------------------------


class TestMDSimulation:
    def test_basic_properties(self, synthetic_simulation):
        sim = synthetic_simulation
        assert sim.n_chains == 2
        assert sim.n_residues == 12
        assert sim.n_frames == 25

    def test_chain_info_structure(self, synthetic_simulation):
        for _chain, info in synthetic_simulation.chain_info.items():
            assert info["n_residues"] > 0
            assert info["atom_count"] > 0
            assert len(info["residue_range"]) == 2

    def test_system_summary_returns_dict(self, synthetic_simulation):
        summary = synthetic_simulation.get_system_summary()
        assert isinstance(summary, dict)


# ---------------------------------------------------------------------------
# Network analysis pipeline
# ---------------------------------------------------------------------------


class TestNetworkAnalyzer:
    def test_contact_maps_shape(self, synthetic_simulation):
        analyzer = NetworkAnalyzer(AnalysisConfig())
        contact_maps = analyzer.compute_contact_maps(synthetic_simulation)
        assert "distance" in contact_maps
        n = synthetic_simulation.n_residues
        assert contact_maps["distance"].shape == (n, n)

    def test_contact_map_values_are_frequencies(self, synthetic_simulation):
        analyzer = NetworkAnalyzer(AnalysisConfig())
        cmap = analyzer.compute_contact_maps(synthetic_simulation)["distance"]
        assert cmap.min() >= 0.0
        assert cmap.max() <= 1.0

    def test_contact_map_is_symmetric(self, synthetic_simulation):
        analyzer = NetworkAnalyzer(AnalysisConfig())
        cmap = analyzer.compute_contact_maps(synthetic_simulation)["distance"]
        np.testing.assert_allclose(cmap, cmap.T, atol=1e-6)

    def test_network_metrics_computed(self, synthetic_simulation):
        analyzer = NetworkAnalyzer(AnalysisConfig())
        analyzer.compute_contact_maps(synthetic_simulation)
        metrics = analyzer.compute_network_metrics(synthetic_simulation)
        assert isinstance(metrics, NetworkMetrics)
        assert metrics.n_nodes > 0

    def test_safe_diameter_on_disconnected_graph(self):
        import networkx as nx

        analyzer = NetworkAnalyzer(AnalysisConfig())
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("c", "d")])  # disconnected
        # Must not raise — returns inf for disconnected graphs.
        result = analyzer._safe_diameter(g)
        assert result == float("inf") or result >= 0


# ---------------------------------------------------------------------------
# MSM backend integration (delegates to mdcompare.msm_backends)
# ---------------------------------------------------------------------------


class TestMSMIntegration:
    @staticmethod
    def _has_backend():
        import importlib.util

        return (
            importlib.util.find_spec("deeptime") is not None
            or importlib.util.find_spec("pyemma") is not None
        )

    def _features(self, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, size=(600, 4))

    def test_clustering_delegates_to_backend(self):
        if not self._has_backend():
            import pytest

            pytest.skip("no MSM backend installed")
        config = AnalysisConfig(msm_n_clusters=8, msm_clustering_method="kmeans")
        analyzer = NetworkAnalyzer(config)
        clustering = analyzer._perform_msm_clustering(self._features())
        assert hasattr(clustering, "dtrajs")
        assert hasattr(clustering, "clustercenters")
        assert len(clustering.dtrajs) == 1

    def test_build_msm_model_returns_triplet(self):
        if not self._has_backend():
            import pytest

            pytest.skip("no MSM backend installed")
        config = AnalysisConfig(msm_n_clusters=8, msm_lag_time=3, kinetic_timescales_count=4)
        analyzer = NetworkAnalyzer(config)
        clustering = analyzer._perform_msm_clustering(self._features(1))
        model, lag_times, timescales = analyzer._build_msm_model(clustering.dtrajs[0])
        assert hasattr(model, "transition_matrix")
        assert hasattr(model, "stationary_distribution")
        assert len(lag_times) >= 1

    def test_pairwise_distance_features_are_subsampled(self):
        """Large atom counts must be capped to keep the feature dim tractable."""
        config = AnalysisConfig(msm_max_distance_atoms=50)
        analyzer = NetworkAnalyzer(config)
        coords = np.random.default_rng(0).normal(0, 1, size=(20, 400, 3))
        features = analyzer._compute_pairwise_distances(coords)
        # 50 atoms -> 50*49/2 = 1225 pairwise distances, not 400*399/2.
        assert features.shape[0] == 20
        assert features.shape[1] == 50 * 49 // 2
