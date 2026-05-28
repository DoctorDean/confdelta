#!/usr/bin/env python3

"""
Markov State Model backend abstraction.

MD-Compare's MSM analysis was originally written directly against the
PyEMMA API. PyEMMA is no longer actively maintained and is difficult to
install on Python 3.11+, so this module introduces a thin abstraction with
two interchangeable backends:

* **PyEMMA** -- the original backend, used as-is when available.
* **deeptime** -- PyEMMA's actively maintained successor.

Both backends are exposed through a single pair of entry points,
:func:`cluster_features` and :func:`build_msm`, which return objects whose
attributes match what the rest of ``core.py`` already expects (a PyEMMA-style
surface: ``clustercenters``/``dtrajs`` for clustering; ``nstates``,
``transition_matrix``, ``stationary_distribution``, ``eigenvalues()``,
``timescales()``, ``mfpt()``, ``pcca()`` for the model).

Backend selection
-----------------
``select_backend()`` honours an explicit preference, otherwise prefers
PyEMMA (for exact backward compatibility) and falls back to deeptime.
Neither library is imported until a backend is actually used.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mdcompare")


# ---------------------------------------------------------------------------
# Backend availability (no heavy imports at module load)
# ---------------------------------------------------------------------------


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


PYEMMA_AVAILABLE = _available("pyemma")
DEEPTIME_AVAILABLE = _available("deeptime")
MSM_AVAILABLE = PYEMMA_AVAILABLE or DEEPTIME_AVAILABLE


def select_backend(preference: str | None = None) -> str | None:
    """Return the name of the MSM backend to use.

    Parameters
    ----------
    preference
        Optional explicit choice: ``"pyemma"``, ``"deeptime"`` or ``"auto"``.
        ``None`` and ``"auto"`` behave identically.

    Returns
    -------
    str or None
        ``"pyemma"`` or ``"deeptime"`` if a usable backend is available,
        otherwise ``None``.
    """
    pref = (preference or "auto").lower()

    if pref == "pyemma":
        if PYEMMA_AVAILABLE:
            return "pyemma"
        logger.warning("PyEMMA requested but not installed; falling back to auto-selection.")
        pref = "auto"
    elif pref == "deeptime":
        if DEEPTIME_AVAILABLE:
            return "deeptime"
        logger.warning("deeptime requested but not installed; falling back to auto-selection.")
        pref = "auto"

    if pref == "auto":
        # Prefer PyEMMA for exact backward compatibility, then deeptime.
        if PYEMMA_AVAILABLE:
            return "pyemma"
        if DEEPTIME_AVAILABLE:
            return "deeptime"
        return None

    logger.warning("Unknown MSM backend preference '%s'; using auto-selection.", preference)
    return select_backend("auto")


# ===========================================================================
# Clustering adapters
# ===========================================================================


class _ClusteringResult:
    """Uniform clustering result with a PyEMMA-style surface.

    Exposes ``dtrajs`` (list of discrete trajectories) and ``clustercenters``
    regardless of which backend produced the clustering.
    """

    def __init__(self, dtrajs: list[np.ndarray], cluster_centers: np.ndarray):
        self.dtrajs = dtrajs
        self.clustercenters = cluster_centers

    @property
    def n_clusters(self) -> int:
        return len(self.clustercenters)


def _cluster_pyemma(features: np.ndarray, n_clusters: int, method: str) -> _ClusteringResult:
    import pyemma.coordinates as coor

    if method == "minibatch_kmeans":
        clustering = coor.cluster_mini_batch_kmeans(data=[features], k=n_clusters, max_iter=500)
    elif method == "regular_space":
        clustering = coor.cluster_regspace(data=[features], dmin=0.5)
    else:
        if method != "kmeans":
            logger.warning("Unknown clustering method '%s'; using kmeans.", method)
        clustering = coor.cluster_kmeans(data=[features], k=n_clusters, max_iter=500)

    return _ClusteringResult(
        dtrajs=[np.asarray(d) for d in clustering.dtrajs],
        cluster_centers=np.asarray(clustering.clustercenters),
    )


def _cluster_deeptime(features: np.ndarray, n_clusters: int, method: str) -> _ClusteringResult:
    from deeptime.clustering import KMeans, MiniBatchKMeans, RegularSpace

    features = np.asarray(features, dtype=float)

    if method == "minibatch_kmeans":
        estimator = MiniBatchKMeans(n_clusters=n_clusters, max_iter=500)
    elif method == "regular_space":
        # RegularSpace discretises by a minimum inter-centre distance rather
        # than a fixed cluster count; max_centers caps the result.
        estimator = RegularSpace(dmin=0.5, max_centers=max(n_clusters * 10, 100))
    else:
        if method != "kmeans":
            logger.warning("Unknown clustering method '%s'; using kmeans.", method)
        estimator = KMeans(n_clusters=n_clusters, max_iter=500)

    model = estimator.fit_fetch(features)
    dtraj = np.asarray(model.transform(features))

    return _ClusteringResult(
        dtrajs=[dtraj],
        cluster_centers=np.asarray(model.cluster_centers),
    )


def cluster_features(
    features: np.ndarray,
    n_clusters: int,
    method: str = "kmeans",
    backend: str | None = None,
) -> _ClusteringResult:
    """Discretise *features* into microstates.

    Parameters
    ----------
    features
        Feature matrix of shape ``(n_frames, n_features)``.
    n_clusters
        Target number of microstates.
    method
        ``"kmeans"``, ``"minibatch_kmeans"`` or ``"regular_space"``.
    backend
        Backend preference; resolved via :func:`select_backend`.

    Returns
    -------
    _ClusteringResult
        Object exposing ``dtrajs`` and ``clustercenters``.
    """
    chosen = select_backend(backend)
    if chosen is None:
        raise RuntimeError("No MSM backend available (install 'pyemma' or 'deeptime').")

    method = (method or "kmeans").lower()
    if chosen == "pyemma":
        return _cluster_pyemma(features, n_clusters, method)
    return _cluster_deeptime(features, n_clusters, method)


# ===========================================================================
# MSM model adapters
# ===========================================================================


class _DeeptimeMSMAdapter:
    """Wrap a deeptime MSM so it matches the PyEMMA model surface.

    ``core.py`` accesses the model through PyEMMA-style names (``nstates``,
    ``eigenvalues()``, ``stationary_distribution`` ...). deeptime uses
    slightly different names; this adapter bridges the two so the analysis
    code needs no per-backend branching.
    """

    def __init__(self, msm: Any):
        self._msm = msm

    # --- state counts -----------------------------------------------------
    @property
    def nstates(self) -> int:
        return int(self._msm.n_states)

    @property
    def nstates_full(self) -> int:
        # The count model knows the full (pre-connectivity) state count.
        count_model = getattr(self._msm, "count_model", None)
        if count_model is not None and hasattr(count_model, "n_states_full"):
            return int(count_model.n_states_full)
        return int(self._msm.n_states)

    @property
    def active_set(self) -> np.ndarray:
        count_model = getattr(self._msm, "count_model", None)
        if count_model is not None and hasattr(count_model, "state_symbols"):
            return np.asarray(count_model.state_symbols)
        return np.arange(self.nstates)

    @property
    def reversible(self) -> bool:
        return bool(getattr(self._msm, "reversible", True))

    @property
    def sparse(self) -> bool:
        return bool(getattr(self._msm, "sparse", False))

    # --- spectral / kinetic quantities -----------------------------------
    @property
    def transition_matrix(self) -> np.ndarray:
        return np.asarray(self._msm.transition_matrix)

    @property
    def stationary_distribution(self) -> np.ndarray:
        return np.asarray(self._msm.stationary_distribution)

    def eigenvalues(self, k: int | None = None) -> np.ndarray:
        vals = np.asarray(self._msm.eigenvalues(k) if k else self._msm.eigenvalues())
        return vals

    def timescales(self, k: int | None = None) -> np.ndarray:
        return np.asarray(self._msm.timescales(k) if k else self._msm.timescales())

    def mfpt(self, origin: int | None = None, target: int | None = None):
        """Mean first passage time.

        PyEMMA's ``mfpt()`` with no arguments returns the full MFPT matrix;
        deeptime requires explicit origin/target. This reproduces the
        matrix form when called without arguments.
        """
        if origin is not None and target is not None:
            return float(self._msm.mfpt(origin, target))

        n = self.nstates
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        matrix[i, j] = float(self._msm.mfpt(i, j))
                    except Exception:
                        matrix[i, j] = np.nan
        return matrix

    def pcca(self, n_metastable: int):
        """Run PCCA+ and return a PyEMMA-style metastable model.

        deeptime's PCCAModel exposes ``coarse_grained_stationary_probability``
        where PyEMMA exposes ``stationary_distribution``; the returned
        wrapper adds that alias so downstream code is backend-agnostic.
        """
        pcca_model = self._msm.pcca(n_metastable)
        return _PCCAAdapter(pcca_model)

    def __getattr__(self, item):
        # Anything not explicitly bridged falls through to the deeptime MSM.
        return getattr(self._msm, item)


class _PCCAAdapter:
    """Expose a deeptime ``PCCAModel`` with PyEMMA-compatible attributes."""

    def __init__(self, pcca_model: Any):
        self._pcca = pcca_model

    @property
    def assignments(self) -> np.ndarray:
        return np.asarray(self._pcca.assignments)

    @property
    def memberships(self) -> np.ndarray:
        return np.asarray(self._pcca.memberships)

    @property
    def stationary_distribution(self) -> np.ndarray:
        # PyEMMA name -> deeptime's coarse_grained_stationary_probability.
        return np.asarray(self._pcca.coarse_grained_stationary_probability)

    @property
    def coarse_grained_transition_matrix(self) -> np.ndarray:
        return np.asarray(self._pcca.coarse_grained_transition_matrix)

    def __getattr__(self, item):
        return getattr(self._pcca, item)


def _build_msm_pyemma(dtraj: np.ndarray, lag: int, n_timescales: int):
    import pyemma.msm as msm

    its = msm.its(dtraj, lags=_lag_schedule(dtraj), nits=n_timescales)
    msm_model = msm.estimate_markov_model(dtraj, lag=lag)
    return msm_model, its.timescales


def _build_msm_deeptime(dtraj: np.ndarray, lag: int, n_timescales: int):
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM

    def _estimate(lagtime: int):
        """Estimate one MSM at *lagtime*; returns a deeptime MSM model."""
        counts = TransitionCountEstimator(lagtime=lagtime, count_mode="sliding").fit_fetch(dtraj)
        # Use the *undirected* largest connected set. The directed (strongly
        # connected) set collapses to a single state on "sticky"
        # trajectories that visit microstates in near-sequential order --
        # exactly the regime smooth MD features produce. PyEMMA's default
        # connectivity behaviour is similarly lenient.
        counts = counts.submodel_largest(directed=False)
        try:
            return MaximumLikelihoodMSM(reversible=True).fit_fetch(counts)
        except Exception:
            # A reversible MSM needs a strongly connected count matrix; fall
            # back to a non-reversible estimate when that does not hold.
            logger.info(
                "Reversible MSM estimation failed at lag %d; "
                "falling back to a non-reversible MSM.",
                lagtime,
            )
            return MaximumLikelihoodMSM(reversible=False).fit_fetch(counts)

    msm_model = _estimate(lag)

    # Implied timescales across a lag schedule, for lag-time diagnostics.
    its = []
    for test_lag in _lag_schedule(dtraj):
        try:
            m = _estimate(int(test_lag))
            its.append(np.asarray(m.timescales())[:n_timescales])
        except Exception:
            its.append(np.full(n_timescales, np.nan))

    return _DeeptimeMSMAdapter(msm_model), its


def _lag_schedule(dtraj: np.ndarray) -> np.ndarray:
    """Lag times to probe for implied-timescale diagnostics."""
    max_lag = min(100, max(2, len(dtraj) // 10))
    return np.arange(1, max_lag, 2)


def build_msm(
    dtraj: np.ndarray,
    lag: int,
    n_timescales: int = 5,
    backend: str | None = None,
):
    """Estimate a Markov State Model from a discrete trajectory.

    Parameters
    ----------
    dtraj
        Discrete (microstate) trajectory.
    lag
        MSM lag time, in frames.
    n_timescales
        Number of implied timescales to report.
    backend
        Backend preference; resolved via :func:`select_backend`.

    Returns
    -------
    (model, lag_times, implied_timescales)
        ``model`` has a PyEMMA-style surface (native PyEMMA model, or a
        deeptime model wrapped in :class:`_DeeptimeMSMAdapter`).
        ``lag_times`` is the probed lag schedule; ``implied_timescales`` is a
        list of timescale arrays, one per probed lag.
    """
    chosen = select_backend(backend)
    if chosen is None:
        raise RuntimeError("No MSM backend available (install 'pyemma' or 'deeptime').")

    dtraj = np.asarray(dtraj)
    lag_times = _lag_schedule(dtraj)

    if chosen == "pyemma":
        model, timescales = _build_msm_pyemma(dtraj, lag, n_timescales)
    else:
        model, timescales = _build_msm_deeptime(dtraj, lag, n_timescales)

    return model, lag_times, timescales


def backend_report() -> dict:
    """Return a small dict describing MSM backend availability."""
    return {
        "pyemma": PYEMMA_AVAILABLE,
        "deeptime": DEEPTIME_AVAILABLE,
        "selected": select_backend("auto"),
    }
