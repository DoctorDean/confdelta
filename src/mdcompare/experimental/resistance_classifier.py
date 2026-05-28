"""Drug-resistance classification from MD-derived descriptors.

This module turns per-(sub-)trajectory analysis output from MD-Compare into
a feature matrix and trains supervised classifiers to distinguish
drug-*sensitive* from drug-*resistant* protein variants.

It is **experimental**: the feature set and API may change. The core
RandomForest / SVM path depends only on scikit-learn (already a core
MD-Compare dependency). The optional graph-neural-network path depends on
``torch`` + ``torch_geometric`` and is imported lazily, so importing this
module never requires a deep-learning stack.

Typical use
-----------
>>> from mdcompare.experimental.resistance_classifier import (
...     prepare_ml_features, train_resistance_classifier)
>>> X, y, names = prepare_ml_features(per_traj_results, labels=phenotypes)
>>> report = train_resistance_classifier(X, y)
>>> report.best_model_name
'RandomForest'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("mdcompare")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Order of scalar feature blocks produced for each trajectory. Kept explicit
# so feature-importance output downstream is interpretable.
_BASE_FEATURE_NAMES: list[str] = [
    "avg_betweenness_centrality",
    "pc1_cumulative_variance",
    "pc2_cumulative_variance",
    "pc3_cumulative_variance",
    "pc4_cumulative_variance",
    "pc5_cumulative_variance",
    "avg_pocket_volume",
    "pocket_volume_std",
    "configurational_entropy",
]


def _safe_mean(values) -> float:
    """Mean of *values* tolerating dict/list/array/empty input."""
    if values is None:
        return 0.0
    if isinstance(values, dict):
        values = list(values.values())
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()) if arr.size else 0.0


def _safe_std(values) -> float:
    """Standard deviation of *values* tolerating dict/list/array/empty input."""
    if values is None:
        return 0.0
    if isinstance(values, dict):
        values = list(values.values())
    arr = np.asarray(list(values), dtype=float)
    return float(arr.std()) if arr.size else 0.0


def _cumulative_variance(pca_result: dict[str, Any] | None, n: int = 5) -> np.ndarray:
    """Extract the first *n* cumulative-variance values from a PCA result.

    Accepts either a pre-computed ``cumulative_variance`` array or raw
    ``pca_variance_explained`` / ``pca_eigenvalues`` and derives it. Always
    returns a length-*n* array (zero-padded if necessary).
    """
    out = np.zeros(n, dtype=float)
    if not pca_result:
        return out

    cumulative = pca_result.get("cumulative_variance")
    if cumulative is None:
        explained = (
            pca_result.get("pca_variance_explained")
            or pca_result.get("variance_explained")
            or pca_result.get("pca_eigenvalues")
        )
        if explained is not None:
            explained = np.asarray(explained, dtype=float)
            if explained.sum() > 0:
                explained = explained / explained.sum()
            cumulative = np.cumsum(explained)
    if cumulative is None:
        return out

    cumulative = np.asarray(cumulative, dtype=float)
    take = min(n, cumulative.size)
    out[:take] = cumulative[:take]
    return out


def _feature_vector_for(
    network_result: dict[str, Any] | None,
    pca_result: dict[str, Any] | None,
    pocket_result: dict[str, Any] | None,
    thermo_result: dict[str, Any] | None,
) -> np.ndarray:
    """Build the fixed-length feature vector for one trajectory."""
    # Network: average betweenness centrality.
    centrality = {}
    if network_result:
        metrics = network_result.get("metrics", network_result)
        if isinstance(metrics, dict):
            centrality = metrics.get("betweenness_centrality", {}) or {}
        else:  # a NetworkMetrics dataclass
            centrality = getattr(metrics, "betweenness_centrality", {}) or {}
    avg_centrality = _safe_mean(centrality)

    pc_variance = _cumulative_variance(pca_result, n=5)

    pocket_volumes = (pocket_result or {}).get("volumes")
    avg_volume = _safe_mean(pocket_volumes)
    volume_std = _safe_std(pocket_volumes)

    entropy = 0.0
    if thermo_result:
        entropy = float(thermo_result.get("configurational_entropy", 0.0) or 0.0)

    return np.concatenate(
        [[avg_centrality], pc_variance, [avg_volume, volume_std], [entropy]]
    ).astype(float)


def prepare_ml_features(
    per_trajectory_results: dict[str, dict[str, Any]],
    labels: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble a feature matrix from per-trajectory MD-Compare results.

    Parameters
    ----------
    per_trajectory_results
        Mapping ``{trajectory_id: {...}}``. Each inner dict may contain any
        subset of the keys ``network``, ``pca``, ``pocket`` and ``thermo``;
        missing blocks are tolerated and contribute zero-valued features.
    labels
        Optional resistance phenotypes (0 = sensitive, 1 = resistant), one
        per trajectory in iteration order. When omitted, an all-zero label
        vector is returned (useful for inference-only feature extraction).

    Returns
    -------
    (X, y, feature_names)
        ``X`` has shape ``(n_trajectories, n_features)``; ``y`` has shape
        ``(n_trajectories,)``; ``feature_names`` labels the columns of X.

    Notes
    -----
    Unlike the original prototype this does **not** hard-code five
    sub-trajectories — it processes whatever trajectories are supplied.
    """
    if not per_trajectory_results:
        raise ValueError("per_trajectory_results is empty")

    traj_ids = list(per_trajectory_results.keys())

    if labels is not None and len(labels) != len(traj_ids):
        raise ValueError(
            f"labels length ({len(labels)}) does not match number of "
            f"trajectories ({len(traj_ids)})"
        )

    features = []
    for tid in traj_ids:
        block = per_trajectory_results[tid] or {}
        features.append(
            _feature_vector_for(
                block.get("network"),
                block.get("pca"),
                block.get("pocket"),
                block.get("thermo"),
            )
        )

    X = np.vstack(features)
    y = np.asarray(labels, dtype=int) if labels is not None else np.zeros(len(traj_ids), dtype=int)
    return X, y, list(_BASE_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass
class ClassifierReport:
    """Outcome of :func:`train_resistance_classifier`."""

    models: dict[str, Any] = field(default_factory=dict)
    cv_scores: dict[str, np.ndarray] = field(default_factory=dict)
    mean_accuracy: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, np.ndarray | None] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    scaler: Any = None

    @property
    def best_model_name(self) -> str | None:
        """Name of the model with the highest mean cross-validation accuracy."""
        if not self.mean_accuracy:
            return None
        return max(self.mean_accuracy, key=self.mean_accuracy.get)

    @property
    def best_model(self) -> Any:
        """The fitted estimator with the highest mean CV accuracy."""
        name = self.best_model_name
        return self.models.get(name) if name else None


def train_resistance_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> ClassifierReport:
    """Train RandomForest and SVM classifiers on resistance features.

    Parameters
    ----------
    features, labels
        Feature matrix and integer labels, as returned by
        :func:`prepare_ml_features`.
    feature_names
        Optional column labels, propagated into the report for
        interpretability.
    cv_folds
        Requested number of cross-validation folds. Automatically reduced
        when the data set or smallest class is too small to support it.
    random_state
        Seed for reproducible model fitting.

    Returns
    -------
    ClassifierReport
        Fitted models, per-model CV scores, mean accuracies, feature
        importances (RandomForest) and the fitted feature scaler.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if features.ndim != 2:
        raise ValueError("features must be a 2-D array")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels have mismatched lengths")

    n_samples = features.shape[0]
    unique, counts = np.unique(labels, return_counts=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    report = ClassifierReport(
        feature_names=list(feature_names or []),
        scaler=scaler,
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM": SVC(kernel="rbf", random_state=random_state),
    }

    # Cross-validation is only meaningful with >=2 classes and enough
    # samples per class; otherwise we still fit but skip scoring.
    can_cross_validate = len(unique) >= 2 and counts.min() >= 2
    effective_folds = max(2, min(cv_folds, int(counts.min()))) if can_cross_validate else 0
    if not can_cross_validate:
        logger.warning(
            "Skipping cross-validation: need >=2 classes with >=2 samples "
            "each (got classes=%s, counts=%s).",
            unique.tolist(),
            counts.tolist(),
        )

    for name, model in models.items():
        if not can_cross_validate and len(unique) < 2:
            # A classifier cannot be fitted on a single class at all.
            report.cv_scores[name] = np.array([])
            report.mean_accuracy[name] = float("nan")
            report.models[name] = model  # unfitted
            report.feature_importance[name] = None
            continue

        if can_cross_validate:
            scores = cross_val_score(model, features_scaled, labels, cv=effective_folds)
            report.cv_scores[name] = scores
            report.mean_accuracy[name] = float(scores.mean())
        else:
            report.cv_scores[name] = np.array([])
            report.mean_accuracy[name] = float("nan")

        model.fit(features_scaled, labels)
        report.models[name] = model
        report.feature_importance[name] = getattr(model, "feature_importances_", None)

    logger.info(
        "Trained resistance classifiers on %d samples; best=%s",
        n_samples,
        report.best_model_name,
    )
    return report


def predict_resistance(report: ClassifierReport, features: np.ndarray) -> np.ndarray:
    """Predict resistance labels for new feature vectors.

    Uses the best model from *report* and the scaler fitted during
    training, so callers pass raw (unscaled) features.
    """
    if report.best_model is None:
        raise ValueError("report contains no fitted model")
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    scaled = report.scaler.transform(features)
    return report.best_model.predict(scaled)


# ---------------------------------------------------------------------------
# Optional graph-neural-network path (heavy dependency, lazily imported)
# ---------------------------------------------------------------------------


def graph_neural_network_analysis(network_results: dict[str, dict[str, Any]]):
    """Convert residue networks into PyTorch-Geometric graphs.

    This is an **optional** feature requiring ``torch`` and
    ``torch_geometric`` (not MD-Compare dependencies). The imports happen
    inside the function so the rest of this module is usable without a
    deep-learning stack.

    Parameters
    ----------
    network_results
        Mapping ``{trajectory_id: {...}}`` where each value provides a
        ``metrics`` block with ``betweenness_centrality`` and a
        ``persistent_contacts`` iterable of ``(u, v)`` residue-index pairs.

    Returns
    -------
    list
        One ``torch_geometric.data.Data`` object per trajectory.

    Raises
    ------
    ImportError
        If ``torch`` / ``torch_geometric`` are not installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "graph_neural_network_analysis requires 'torch' and "
            "'torch_geometric'. Install them separately to use this feature."
        ) from exc

    graph_data = []
    for _tid, data in network_results.items():
        metrics = data.get("metrics", {})
        centrality = metrics.get("betweenness_centrality", {}) or {}
        n_nodes = len(centrality)

        node_features = torch.tensor(
            [[float(centrality.get(i, 0.0))] for i in range(n_nodes)],
            dtype=torch.float,
        )

        edges = list(data.get("persistent_contacts", []))
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        graph_data.append(Data(x=node_features, edge_index=edge_index))

    return graph_data
