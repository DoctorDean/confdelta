"""Tests for confdelta.compare — the statistical comparison engine.

The engine composes the confdelta.stats primitives, which are validated
separately, so these tests focus on the composition: that effect sizes, p, and
q are wired correctly, that the two modes behave as designed, and that
underpowered designs and the single-run caveat are surfaced.
"""

from __future__ import annotations

import numpy as np
import pytest

from confdelta import stats
from confdelta.compare import (
    ComparisonReport,
    FeatureComparison,
    compare_feature_matrices,
    compare_frame_matrices,
)

# ---------------------------------------------------------------------------
# Replicate mode
# ---------------------------------------------------------------------------


class TestReplicateMode:
    def test_shapes_and_report_fields(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, size=(5, 4))
        b = rng.normal(0, 1, size=(6, 4))
        report = compare_feature_matrices(a, b, rng=0)
        assert report.mode == "replicate"
        assert report.n_tests == 4
        assert len(report.features) == 4
        assert report.n_a == 5 and report.n_b == 6
        assert report.caveat is None
        assert all(isinstance(f, FeatureComparison) for f in report.features)

    def test_effect_size_matches_hedges_g(self):
        rng = np.random.default_rng(1)
        a = rng.normal(1.0, 1, size=(6, 3))
        b = rng.normal(0.0, 1, size=(6, 3))
        report = compare_feature_matrices(a, b, rng=0)
        for j, f in enumerate(report.features):
            assert f.effect_size_name == "hedges_g"
            assert f.effect_size == pytest.approx(stats.hedges_g(a[:, j], b[:, j]))

    def test_qvalues_match_correcting_the_raw_pvalues(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.5, 1, size=(8, 6))
        b = rng.normal(0.0, 1, size=(8, 6))
        report = compare_feature_matrices(a, b, correction="fdr_bh", rng=0)
        raw = np.array([f.pvalue for f in report.features])
        expected = stats.correct_pvalues(raw, method="fdr_bh")
        got = np.array([f.qvalue for f in report.features])
        np.testing.assert_allclose(got, expected.qvalues)

    def test_large_separation_is_significant_with_adequate_replicates(self):
        rng = np.random.default_rng(3)
        a = rng.normal(5.0, 0.5, size=(8, 3))
        b = rng.normal(0.0, 0.5, size=(8, 3))
        report = compare_feature_matrices(a, b, rng=0)
        assert not report.underpowered
        assert report.n_significant == 3
        assert all(f.significant for f in report.features)

    def test_three_versus_three_is_underpowered(self):
        """The design constraint: 3 v 3 cannot reach 0.05, however large the effect."""
        rng = np.random.default_rng(4)
        a = rng.normal(50.0, 0.5, size=(3, 4))  # enormous separation
        b = rng.normal(0.0, 0.5, size=(3, 4))
        report = compare_feature_matrices(a, b, rng=0)
        assert report.underpowered
        assert report.min_pvalue == pytest.approx(0.10)
        assert report.n_significant == 0
        # But the effect sizes are huge and reported.
        assert all(abs(f.effect_size) > 2.0 for f in report.features)

    def test_no_true_difference_gives_few_significant(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0, 1, size=(10, 20))
        b = rng.normal(0, 1, size=(10, 20))
        report = compare_feature_matrices(a, b, rng=0)
        # With FDR at 0.05 and no real effect, expect essentially none.
        assert report.n_significant <= 1

    def test_ranked_by_effect_orders_by_magnitude(self):
        rng = np.random.default_rng(6)
        a = np.column_stack([rng.normal(m, 0.5, size=6) for m in (0.1, 3.0, 1.0)])
        b = rng.normal(0.0, 0.5, size=(6, 3))
        report = compare_feature_matrices(a, b, feature_names=["p", "q", "r"], rng=0)
        ranked = report.ranked_by_effect()
        magnitudes = [abs(f.effect_size) for f in ranked]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert ranked[0].feature == "q"  # the biggest shift

    def test_feature_names_are_used(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, size=(4, 2))
        b = rng.normal(0, 1, size=(4, 2))
        report = compare_feature_matrices(a, b, feature_names=["alpha", "beta"], rng=0)
        assert [f.feature for f in report.features] == ["alpha", "beta"]

    def test_single_replicate_rejected(self):
        with pytest.raises(ValueError, match="at least two replicates"):
            compare_feature_matrices(np.zeros((1, 3)), np.zeros((4, 3)))

    def test_mismatched_feature_count_rejected(self):
        with pytest.raises(ValueError, match="same number of features"):
            compare_feature_matrices(np.zeros((3, 4)), np.zeros((3, 5)))

    def test_wrong_feature_name_count_rejected(self):
        with pytest.raises(ValueError, match="feature_names"):
            compare_feature_matrices(np.zeros((3, 2)), np.zeros((3, 2)), feature_names=["only_one"])


# ---------------------------------------------------------------------------
# Bootstrap mode
# ---------------------------------------------------------------------------


class TestBootstrapMode:
    def test_report_is_bootstrap_mode_with_caveat(self):
        rng = np.random.default_rng(10)
        a = rng.normal(0, 1, size=(400, 3))
        b = rng.normal(0, 1, size=(400, 3))
        report = compare_frame_matrices(a, b, n_resamples=300, rng=0)
        assert report.mode == "bootstrap"
        assert report.caveat is not None
        assert "single run" in report.caveat.lower()
        assert report.min_pvalue is None

    def test_effect_size_is_cohens_d(self):
        rng = np.random.default_rng(11)
        a = rng.normal(1.0, 1, size=(500, 2))
        b = rng.normal(0.0, 1, size=(500, 2))
        report = compare_frame_matrices(a, b, n_resamples=300, rng=0)
        for j, f in enumerate(report.features):
            assert f.effect_size_name == "cohens_d"
            assert f.effect_size == pytest.approx(stats.cohens_d(a[:, j], b[:, j]))

    def test_clear_difference_is_detected(self):
        rng = np.random.default_rng(12)
        a = rng.normal(3.0, 1, size=(600, 2))
        b = rng.normal(0.0, 1, size=(600, 2))
        report = compare_frame_matrices(a, b, n_resamples=500, rng=0)
        assert report.n_significant == 2

    def test_no_difference_is_not_flagged(self):
        rng = np.random.default_rng(13)
        a = rng.normal(0, 1, size=(600, 4))
        b = rng.normal(0, 1, size=(600, 4))
        report = compare_frame_matrices(a, b, n_resamples=500, rng=0)
        assert report.n_significant == 0

    def test_effect_ci_brackets_estimate(self):
        rng = np.random.default_rng(14)
        a = rng.normal(1.0, 1, size=(400, 3))
        b = rng.normal(0.0, 1, size=(400, 3))
        report = compare_frame_matrices(a, b, n_resamples=400, rng=0)
        for f in report.features:
            assert f.effect_ci.low <= f.effect_size <= f.effect_ci.high
            assert f.effect_ci.method == "block-bootstrap-percentile"

    def test_mismatched_feature_count_rejected(self):
        with pytest.raises(ValueError, match="same number of features"):
            compare_frame_matrices(np.zeros((100, 3)), np.zeros((100, 4)))


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------


class TestReportHelpers:
    def _report(self):
        rng = np.random.default_rng(20)
        a = np.column_stack([rng.normal(5, 0.5, size=8), rng.normal(0, 0.5, size=8)])
        b = rng.normal(0, 0.5, size=(8, 2))
        return compare_feature_matrices(a, b, feature_names=["big", "null"], rng=0)

    def test_significant_features_filters(self):
        report = self._report()
        sig = report.significant_features()
        assert all(f.significant for f in sig)
        assert "big" in {f.feature for f in sig}

    def test_report_is_a_dataclass_instance(self):
        assert isinstance(self._report(), ComparisonReport)


class TestValidation:
    def test_non_finite_rejected(self):
        a = np.full((3, 2), np.nan)
        with pytest.raises(ValueError, match="non-finite"):
            compare_feature_matrices(a, np.zeros((3, 2)))

    def test_one_dimensional_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            compare_feature_matrices(np.zeros(5), np.zeros((3, 2)))
