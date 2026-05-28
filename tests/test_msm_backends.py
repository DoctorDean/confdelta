"""Tests for mdcompare.msm_backends (PyEMMA / deeptime MSM abstraction)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mdcompare import msm_backends as mb

HAS_DEEPTIME = importlib.util.find_spec("deeptime") is not None
HAS_PYEMMA = importlib.util.find_spec("pyemma") is not None

requires_deeptime = pytest.mark.skipif(not HAS_DEEPTIME, reason="deeptime not installed")


# ---------------------------------------------------------------------------
# Backend selection (no heavy imports required)
# ---------------------------------------------------------------------------


class TestSelectBackend:
    def test_auto_prefers_pyemma_then_deeptime(self):
        chosen = mb.select_backend("auto")
        if HAS_PYEMMA:
            assert chosen == "pyemma"
        elif HAS_DEEPTIME:
            assert chosen == "deeptime"
        else:
            assert chosen is None

    def test_none_behaves_like_auto(self):
        assert mb.select_backend(None) == mb.select_backend("auto")

    def test_explicit_deeptime_when_available(self):
        if HAS_DEEPTIME:
            assert mb.select_backend("deeptime") == "deeptime"

    def test_unknown_preference_falls_back(self):
        # Should not raise; resolves via auto.
        assert mb.select_backend("nonsense") == mb.select_backend("auto")

    def test_unavailable_explicit_backend_falls_back(self):
        # Requesting a backend that isn't installed must not raise; it
        # should fall through to auto-selection.
        if not HAS_PYEMMA:
            assert mb.select_backend("pyemma") == mb.select_backend("auto")

    def test_backend_report_structure(self):
        report = mb.backend_report()
        assert set(report) == {"pyemma", "deeptime", "selected"}
        assert isinstance(report["pyemma"], bool)
        assert isinstance(report["deeptime"], bool)


# ---------------------------------------------------------------------------
# Lag schedule helper
# ---------------------------------------------------------------------------


class TestLagSchedule:
    def test_schedule_within_bounds(self):
        dtraj = np.zeros(1000, dtype=int)
        lags = mb._lag_schedule(dtraj)
        assert lags[0] >= 1
        assert lags[-1] <= 100

    def test_short_trajectory_has_valid_schedule(self):
        dtraj = np.zeros(15, dtype=int)
        lags = mb._lag_schedule(dtraj)
        assert len(lags) >= 1
        assert lags[0] >= 1


# ---------------------------------------------------------------------------
# Clustering (deeptime path)
# ---------------------------------------------------------------------------


@requires_deeptime
class TestClusterFeaturesDeeptime:
    @staticmethod
    def _features(seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, size=(400, 4))

    def test_kmeans_returns_clustering_result(self):
        result = mb.cluster_features(
            self._features(), n_clusters=8, method="kmeans", backend="deeptime"
        )
        assert isinstance(result, mb._ClusteringResult)
        assert result.clustercenters.shape[0] <= 8
        assert len(result.dtrajs) == 1

    def test_dtraj_indices_within_cluster_range(self):
        result = mb.cluster_features(
            self._features(), n_clusters=6, method="kmeans", backend="deeptime"
        )
        dtraj = result.dtrajs[0]
        assert dtraj.min() >= 0
        assert dtraj.max() < result.n_clusters

    def test_minibatch_kmeans(self):
        result = mb.cluster_features(
            self._features(1), n_clusters=5, method="minibatch_kmeans", backend="deeptime"
        )
        assert result.clustercenters.shape[0] <= 5

    def test_unknown_method_defaults_to_kmeans(self):
        # Should not raise; falls back to kmeans.
        result = mb.cluster_features(
            self._features(2), n_clusters=4, method="bogus", backend="deeptime"
        )
        assert isinstance(result, mb._ClusteringResult)


# ---------------------------------------------------------------------------
# MSM construction and the deeptime adapter surface
# ---------------------------------------------------------------------------


@requires_deeptime
class TestBuildMSMDeeptime:
    @staticmethod
    def _dtraj(seed=7):
        # Recurrent (equilibrium-like) discrete trajectory so the MSM has
        # multiple connected states.
        rng = np.random.default_rng(seed)
        feats = rng.normal(0, 1, size=(800, 4))
        result = mb.cluster_features(feats, n_clusters=10, method="kmeans", backend="deeptime")
        return result.dtrajs[0]

    def test_build_returns_model_and_timescales(self):
        model, lag_times, its = mb.build_msm(
            self._dtraj(), lag=3, n_timescales=4, backend="deeptime"
        )
        assert isinstance(model, mb._DeeptimeMSMAdapter)
        assert len(lag_times) >= 1
        assert len(its) == len(lag_times)

    def test_adapter_exposes_pyemma_surface(self):
        model, _, _ = mb.build_msm(self._dtraj(), lag=3, n_timescales=4, backend="deeptime")
        # PyEMMA-style attribute names that core.py relies on.
        assert isinstance(model.nstates, int)
        assert model.nstates_full >= model.nstates
        assert model.transition_matrix.shape == (model.nstates, model.nstates)
        assert model.stationary_distribution.shape == (model.nstates,)
        np.testing.assert_allclose(model.stationary_distribution.sum(), 1.0, atol=1e-6)

    def test_eigenvalues_and_timescales(self):
        model, _, _ = mb.build_msm(self._dtraj(), lag=3, n_timescales=4, backend="deeptime")
        eig = model.eigenvalues()
        assert eig[0] == pytest.approx(1.0, abs=1e-6)  # stationary eigenvalue
        ts = model.timescales()
        assert np.all(ts[np.isfinite(ts)] >= 0)

    def test_mfpt_matrix_form(self):
        model, _, _ = mb.build_msm(self._dtraj(), lag=3, n_timescales=4, backend="deeptime")
        mfpt = model.mfpt()  # no args -> full matrix
        assert mfpt.shape == (model.nstates, model.nstates)
        assert np.all(np.diag(mfpt) == 0)

    def test_mfpt_scalar_form(self):
        model, _, _ = mb.build_msm(self._dtraj(), lag=3, n_timescales=4, backend="deeptime")
        if model.nstates >= 2:
            val = model.mfpt(0, 1)
            assert isinstance(val, float)
            assert val >= 0

    def test_pcca_adapter(self):
        model, _, _ = mb.build_msm(self._dtraj(), lag=3, n_timescales=4, backend="deeptime")
        if model.nstates >= 3:
            pcca = model.pcca(3)
            assert isinstance(pcca, mb._PCCAAdapter)
            assert pcca.assignments.shape[0] == model.nstates
            # PyEMMA-name alias for coarse-grained stationary probability.
            np.testing.assert_allclose(pcca.stationary_distribution.sum(), 1.0, atol=1e-6)
            assert pcca.coarse_grained_transition_matrix.shape == (3, 3)


# ---------------------------------------------------------------------------
# No-backend behaviour
# ---------------------------------------------------------------------------


def test_build_msm_without_backend_raises(monkeypatch):
    """When no backend resolves, build/cluster raise a clear error."""
    monkeypatch.setattr(mb, "select_backend", lambda preference=None: None)
    with pytest.raises(RuntimeError, match="No MSM backend"):
        mb.build_msm(np.zeros(50, dtype=int), lag=2)
    with pytest.raises(RuntimeError, match="No MSM backend"):
        mb.cluster_features(np.zeros((50, 3)), n_clusters=4)
