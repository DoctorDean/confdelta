"""Tests for mdcompare.experimental.resistance_classifier."""

from __future__ import annotations

import numpy as np
import pytest

from mdcompare.experimental.resistance_classifier import (
    ClassifierReport,
    predict_resistance,
    prepare_ml_features,
    train_resistance_classifier,
)


def _mock_traj_block(centrality_level: float, entropy: float):
    """Build a single per-trajectory result dict."""
    return {
        "network": {
            "metrics": {
                "betweenness_centrality": {i: centrality_level + 0.01 * i for i in range(10)}
            }
        },
        "pca": {"cumulative_variance": [0.4, 0.65, 0.8, 0.9, 0.95]},
        "pocket": {"volumes": [100.0, 120.0, 110.0]},
        "thermo": {"configurational_entropy": entropy},
    }


def _mock_results(n: int):
    return {f"sub_traj_{i}": _mock_traj_block(0.1 * (i + 1), 5.0 + i) for i in range(n)}


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


class TestPrepareFeatures:
    def test_feature_matrix_shape(self):
        results = _mock_results(6)
        X, y, names = prepare_ml_features(results, labels=[0, 0, 0, 1, 1, 1])
        assert X.shape == (6, len(names))
        assert y.shape == (6,)

    def test_does_not_hardcode_five_trajectories(self):
        """Original prototype assumed exactly 5 sub-trajectories."""
        for n in (3, 7, 12):
            X, _, _ = prepare_ml_features(_mock_results(n))
            assert X.shape[0] == n

    def test_missing_blocks_tolerated(self):
        results = {"t0": {"network": None}, "t1": {}}
        X, y, names = prepare_ml_features(results)
        assert X.shape == (2, len(names))
        # Missing data contributes zero features, never NaN.
        assert not np.isnan(X).any()

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            prepare_ml_features({})

    def test_label_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            prepare_ml_features(_mock_results(4), labels=[0, 1])

    def test_labels_default_to_zeros(self):
        _, y, _ = prepare_ml_features(_mock_results(5))
        assert np.all(y == 0)

    def test_cumulative_variance_derived_from_eigenvalues(self):
        results = {"t0": {"pca": {"pca_eigenvalues": [4.0, 3.0, 2.0, 1.0]}}}
        X, _, names = prepare_ml_features(results)
        idx = names.index("pc1_cumulative_variance")
        # First cumulative-variance entry = 4 / 10 = 0.4.
        assert X[0, idx] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------


class TestTrainClassifier:
    def test_training_produces_report(self):
        X, y, names = prepare_ml_features(_mock_results(10), labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        report = train_resistance_classifier(X, y, feature_names=names)
        assert isinstance(report, ClassifierReport)
        assert "RandomForest" in report.models
        assert "SVM" in report.models

    def test_best_model_selected(self):
        X, y, names = prepare_ml_features(_mock_results(10), labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        report = train_resistance_classifier(X, y, feature_names=names)
        assert report.best_model_name in {"RandomForest", "SVM"}
        assert report.best_model is not None

    def test_feature_importance_present_for_random_forest(self):
        X, y, names = prepare_ml_features(_mock_results(8), labels=[0, 0, 0, 0, 1, 1, 1, 1])
        report = train_resistance_classifier(X, y, feature_names=names)
        importance = report.feature_importance["RandomForest"]
        assert importance is not None
        assert len(importance) == X.shape[1]

    def test_single_class_skips_cross_validation(self):
        """All-sensitive data must not crash; CV and fitting are skipped."""
        X, y, names = prepare_ml_features(_mock_results(6))  # all labels 0
        report = train_resistance_classifier(X, y, feature_names=names)
        assert np.isnan(report.mean_accuracy["RandomForest"])
        # Estimator is present but left unfitted (cannot fit on one class).
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        with pytest.raises(NotFittedError):
            check_is_fitted(report.models["RandomForest"])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            train_resistance_classifier(np.zeros((5, 9)), np.zeros(3))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_returns_labels(self):
        X, y, names = prepare_ml_features(_mock_results(10), labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        report = train_resistance_classifier(X, y, feature_names=names)
        preds = predict_resistance(report, X)
        assert preds.shape == (10,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_single_vector(self):
        X, y, names = prepare_ml_features(_mock_results(8), labels=[0, 0, 0, 0, 1, 1, 1, 1])
        report = train_resistance_classifier(X, y, feature_names=names)
        preds = predict_resistance(report, X[0])
        assert preds.shape == (1,)

    def test_predict_without_model_raises(self):
        with pytest.raises(ValueError):
            predict_resistance(ClassifierReport(), np.zeros((1, 9)))
