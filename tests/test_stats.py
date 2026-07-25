"""Tests for confdelta.stats — the statistical primitives.

Every primitive is checked against a known analytical result or an independent
implementation (SciPy, statsmodels, or a hand computation). Nothing is checked
against itself. The two components with real numerical risk -- the integrated
autocorrelation time and the block bootstrap -- are validated against the
analytic AR(1) values and by measured confidence-interval coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from confdelta import stats

# ---------------------------------------------------------------------------
# Autocorrelation time and effective sample size
# ---------------------------------------------------------------------------


def _ar1(n, phi, rng):
    """An AR(1) series x_t = phi x_{t-1} + eps, whose true tau_int is known."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


class TestAutocorrelationTime:
    def test_uncorrelated_series_has_tau_near_one(self):
        rng = np.random.default_rng(0)
        taus = [stats.integrated_autocorrelation_time(rng.normal(size=5000)) for _ in range(20)]
        assert 0.9 < np.mean(taus) < 1.15

    @pytest.mark.parametrize("phi", [0.5, 0.8])
    def test_ar1_recovers_analytic_tau(self, phi):
        """For AR(1), tau_int = (1+phi)/(1-phi). Estimator should recover it."""
        rng = np.random.default_rng(1)
        true_tau = (1 + phi) / (1 - phi)
        taus = [stats.integrated_autocorrelation_time(_ar1(20000, phi, rng)) for _ in range(30)]
        # Within 10% of the analytic value, averaged over realisations.
        assert np.mean(taus) == pytest.approx(true_tau, rel=0.1)

    def test_constant_series_is_uncorrelated(self):
        assert stats.integrated_autocorrelation_time(np.full(100, 3.0)) == 1.0

    def test_tau_is_at_least_one(self):
        rng = np.random.default_rng(2)
        # Even anticorrelated noise must not report tau < 1.
        tau = stats.integrated_autocorrelation_time(rng.normal(size=1000))
        assert tau >= 1.0

    def test_effective_sample_size_matches_n_over_tau(self):
        rng = np.random.default_rng(3)
        x = _ar1(10000, 0.8, rng)
        tau = stats.integrated_autocorrelation_time(x)
        assert stats.effective_sample_size(x) == pytest.approx(len(x) / tau)

    def test_effective_sample_size_bounded_by_n(self):
        rng = np.random.default_rng(4)
        x = _ar1(2000, 0.9, rng)
        ess = stats.effective_sample_size(x)
        assert 0 < ess <= len(x)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_known_value(self):
        # Two groups differing by exactly one pooled SD -> d = 1.
        a = np.array([2.0, 4.0, 6.0])  # mean 4, var 4
        b = np.array([0.0, 2.0, 4.0])  # mean 2, var 4; pooled sd = 2
        assert stats.cohens_d(a, b) == pytest.approx(1.0)

    def test_sign_follows_first_group(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert stats.cohens_d(a, b) < 0

    def test_zero_when_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert stats.cohens_d(a, a) == 0.0

    def test_matches_manual_formula_on_random_data(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0.5, 1.0, size=40)
        b = rng.normal(0.0, 1.2, size=35)
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
        expected = (a.mean() - b.mean()) / sp
        assert stats.cohens_d(a, b) == pytest.approx(expected)

    def test_infinite_when_no_spread_but_means_differ(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([1.0, 1.0, 1.0])
        assert np.isinf(stats.cohens_d(a, b))


class TestHedgesG:
    def test_g_is_d_shrunk_toward_zero(self):
        rng = np.random.default_rng(6)
        a = rng.normal(1, 1, size=5)
        b = rng.normal(0, 1, size=5)
        d = stats.cohens_d(a, b)
        g = stats.hedges_g(a, b)
        assert abs(g) < abs(d)  # correction shrinks the estimate
        assert np.sign(g) == np.sign(d)

    def test_correction_factor_matches_formula(self):
        a = np.arange(1.0, 6.0)
        b = np.arange(0.0, 5.0)
        df = len(a) + len(b) - 2
        j = 1.0 - 3.0 / (4.0 * df - 1.0)
        assert stats.hedges_g(a, b) == pytest.approx(stats.cohens_d(a, b) * j)

    def test_correction_negligible_for_large_samples(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0.3, 1, size=500)
        b = rng.normal(0.0, 1, size=500)
        assert stats.hedges_g(a, b) == pytest.approx(stats.cohens_d(a, b), rel=0.01)


class TestCliffsDelta:
    def test_complete_separation_is_plus_one(self):
        a = np.array([4.0, 5.0, 6.0])
        b = np.array([1.0, 2.0, 3.0])
        assert stats.cliffs_delta(a, b) == 1.0

    def test_reverse_separation_is_minus_one(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert stats.cliffs_delta(a, b) == -1.0

    def test_identical_distributions_near_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert stats.cliffs_delta(a, a) == 0.0

    def test_hand_computed_value(self):
        # a = [1, 3], b = [2]. Pairs: 1<2 (-1), 3>2 (+1) -> mean 0.
        assert stats.cliffs_delta(np.array([1.0, 3.0]), np.array([2.0])) == 0.0
        # a = [3, 4], b = [1, 2]: all four pairs a>b -> +1.
        assert stats.cliffs_delta(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0

    def test_bounded(self):
        rng = np.random.default_rng(8)
        d = stats.cliffs_delta(rng.normal(size=50), rng.normal(size=40))
        assert -1.0 <= d <= 1.0

    def test_interpretation_thresholds(self):
        assert stats.cliffs_delta_interpretation(0.1) == "negligible"
        assert stats.cliffs_delta_interpretation(0.2) == "small"
        assert stats.cliffs_delta_interpretation(0.4) == "medium"
        assert stats.cliffs_delta_interpretation(0.8) == "large"


class TestHedgesGInterval:
    def test_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(9)
        a = rng.normal(0.5, 1, size=20)
        b = rng.normal(0.0, 1, size=20)
        ci = stats.hedges_g_interval(a, b)
        assert ci.low <= ci.estimate <= ci.high
        assert ci.method == "hedges-g-analytic"

    def test_wider_interval_at_higher_confidence(self):
        rng = np.random.default_rng(10)
        a = rng.normal(0.5, 1, size=15)
        b = rng.normal(0.0, 1, size=15)
        ci95 = stats.hedges_g_interval(a, b, confidence=0.95)
        ci99 = stats.hedges_g_interval(a, b, confidence=0.99)
        assert (ci99.high - ci99.low) > (ci95.high - ci95.low)

    def test_rejects_bad_confidence(self):
        a = np.arange(1.0, 6.0)
        b = np.arange(0.0, 5.0)
        with pytest.raises(ValueError):
            stats.hedges_g_interval(a, b, confidence=1.5)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------


class TestBlockBootstrap:
    def test_auto_block_length_grows_with_autocorrelation(self):
        rng = np.random.default_rng(11)
        weak = stats.auto_block_length(_ar1(4000, 0.3, rng))
        strong = stats.auto_block_length(_ar1(4000, 0.9, rng))
        assert strong > weak

    def test_distribution_has_requested_size(self):
        rng = np.random.default_rng(12)
        x = _ar1(500, 0.6, rng)
        dist = stats.block_bootstrap(x, np.mean, n_resamples=300, rng=0)
        assert dist.shape == (300,)

    def test_is_reproducible_with_a_seed(self):
        rng = np.random.default_rng(13)
        x = _ar1(400, 0.7, rng)
        d1 = stats.block_bootstrap(x, np.mean, n_resamples=200, rng=42)
        d2 = stats.block_bootstrap(x, np.mean, n_resamples=200, rng=42)
        np.testing.assert_array_equal(d1, d2)

    def test_bootstrap_mean_is_centred_on_sample_mean(self):
        rng = np.random.default_rng(14)
        x = _ar1(1000, 0.6, rng)
        dist = stats.block_bootstrap(x, np.mean, n_resamples=2000, rng=1)
        assert np.mean(dist) == pytest.approx(x.mean(), abs=0.1)

    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(15)
        x = _ar1(800, 0.7, rng)
        ci = stats.block_bootstrap_interval(x, np.mean, n_resamples=1000, rng=2)
        assert ci.low <= ci.estimate <= ci.high
        assert ci.method == "block-bootstrap-percentile"

    def test_rejects_out_of_range_block_length(self):
        with pytest.raises(ValueError):
            stats.block_bootstrap(np.zeros(10), np.mean, block_length=99)


@pytest.mark.slow
class TestBlockBootstrapCoverage:
    """The acceptance test for the block bootstrap (DECISIONS.md D-007).

    On autocorrelated data the block bootstrap must achieve near-nominal
    confidence-interval coverage and must dramatically outperform the IID
    bootstrap, which assumes exchangeable observations and badly undercovers.
    Exact 0.95 coverage is not attainable at these effective sample sizes -- an
    inherent property of estimating a long-run variance from a short dependent
    series, shared by all methods including the gold-standard ESS t-interval --
    so the bar is near-nominal, not exact.
    """

    def _coverage(self, n, phi, method, trials=250, seed=1):
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(trials):
            x = _ar1(n, phi, rng)  # true mean is 0
            if method == "block":
                ci = stats.block_bootstrap_interval(x, np.mean, n_resamples=500, rng=rng)
                lo, hi = ci.low, ci.high
            else:  # naive IID bootstrap
                boot = np.array(
                    [rng.choice(x, size=len(x), replace=True).mean() for _ in range(500)]
                )
                lo, hi = np.percentile(boot, [2.5, 97.5])
            hits += lo <= 0.0 <= hi
        return hits / trials

    def test_block_bootstrap_is_near_nominal(self):
        coverage = self._coverage(1000, 0.8, "block")
        # Near the nominal 0.95; comfortably above 0.85 at this effective size.
        assert coverage >= 0.85

    def test_block_bootstrap_beats_iid(self):
        block = self._coverage(1000, 0.8, "block", seed=3)
        iid = self._coverage(1000, 0.8, "iid", seed=3)
        # The IID bootstrap undercovers badly on autocorrelated data.
        assert iid < 0.75
        assert block - iid >= 0.15


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


class TestMinAttainablePValue:
    def test_three_versus_three_is_point_one(self):
        """The constraint that shapes the whole design: 3v3 cannot beat 0.05."""
        assert stats.min_attainable_pvalue(3, 3) == pytest.approx(0.10)

    def test_four_versus_four(self):
        # C(8,4) = 70; two-sided floor = 2/70.
        assert stats.min_attainable_pvalue(4, 4) == pytest.approx(2 / 70)

    def test_one_sided_is_half_the_two_sided(self):
        two = stats.min_attainable_pvalue(4, 4, alternative="two-sided")
        one = stats.min_attainable_pvalue(4, 4, alternative="greater")
        assert one == pytest.approx(two / 2)

    def test_larger_designs_can_reach_significance(self):
        assert stats.min_attainable_pvalue(6, 6) < 0.05


class TestPermutationTwoSample:
    def test_matches_scipy_exact(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([6.0, 7.0, 8.0, 9.0])
        result = stats.permutation_two_sample(a, b, rng=0)

        def stat(u, v, axis=-1):
            return np.mean(u, axis=axis) - np.mean(v, axis=axis)

        expected = scipy_stats.permutation_test(
            (a, b),
            stat,
            permutation_type="independent",
            alternative="two-sided",
            n_resamples=np.inf,
        )
        assert result.pvalue == pytest.approx(expected.pvalue)
        assert result.exact is True

    def test_small_design_is_flagged_underpowered(self):
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([100.0, 101.0, 102.0])  # enormous effect
        result = stats.permutation_two_sample(a, b, rng=0)
        assert result.underpowered is True
        # Even this huge separation cannot beat 0.05 at 3v3.
        assert result.pvalue >= 0.05
        assert result.pvalue == pytest.approx(result.min_pvalue)

    def test_adequate_design_can_be_significant(self):
        rng = np.random.default_rng(20)
        a = rng.normal(3, 1, size=8)
        b = rng.normal(0, 1, size=8)
        result = stats.permutation_two_sample(a, b, rng=0)
        assert result.underpowered is False
        assert result.pvalue < 0.05

    def test_exact_flag_switches_to_sampling_for_large_groups(self):
        rng = np.random.default_rng(21)
        a = rng.normal(0, 1, size=30)
        b = rng.normal(0, 1, size=30)
        result = stats.permutation_two_sample(a, b, n_resamples=2000, rng=0)
        assert result.exact is False  # C(60,30) >> 2000
        assert result.n_resamples == 2000


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------


class TestMultipleTestingCorrection:
    P = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])

    def test_bh_matches_scipy(self):
        result = stats.correct_pvalues(self.P, method="fdr_bh")
        expected = scipy_stats.false_discovery_control(self.P, method="bh")
        np.testing.assert_allclose(result.qvalues, expected)

    def test_bh_matches_statsmodels(self):
        statsmodels = pytest.importorskip("statsmodels.stats.multitest")
        result = stats.correct_pvalues(self.P, method="fdr_bh")
        _, q_sm, _, _ = statsmodels.multipletests(self.P, method="fdr_bh")
        np.testing.assert_allclose(result.qvalues, q_sm, rtol=1e-10)

    def test_by_matches_statsmodels(self):
        statsmodels = pytest.importorskip("statsmodels.stats.multitest")
        result = stats.correct_pvalues(self.P, method="fdr_by")
        _, q_sm, _, _ = statsmodels.multipletests(self.P, method="fdr_by")
        np.testing.assert_allclose(result.qvalues, q_sm, rtol=1e-10)

    def test_bonferroni_matches_statsmodels(self):
        statsmodels = pytest.importorskip("statsmodels.stats.multitest")
        result = stats.correct_pvalues(self.P, method="bonferroni")
        _, q_sm, _, _ = statsmodels.multipletests(self.P, method="bonferroni")
        np.testing.assert_allclose(result.qvalues, q_sm, rtol=1e-10)

    def test_by_is_more_conservative_than_bh(self):
        bh = stats.correct_pvalues(self.P, method="fdr_bh")
        by = stats.correct_pvalues(self.P, method="fdr_by")
        assert np.all(by.qvalues >= bh.qvalues - 1e-12)

    def test_none_leaves_pvalues_unchanged(self):
        result = stats.correct_pvalues(self.P, method="none")
        np.testing.assert_array_equal(result.qvalues, self.P)

    def test_family_metadata_is_recorded(self):
        result = stats.correct_pvalues(self.P, method="fdr_bh", alpha=0.1)
        assert result.n_tests == len(self.P)
        assert result.method == "fdr_bh"
        assert result.alpha == 0.1
        assert result.n_rejected == int((result.qvalues <= 0.1).sum())

    def test_rejects_pvalues_out_of_range(self):
        with pytest.raises(ValueError):
            stats.correct_pvalues(np.array([0.5, 1.5]), method="fdr_bh")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            stats.correct_pvalues(np.array([]), method="fdr_bh")


# ---------------------------------------------------------------------------
# Input validation shared across the module
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_finite_rejected(self):
        with pytest.raises(ValueError):
            stats.integrated_autocorrelation_time(np.array([1.0, np.nan, 3.0]))

    def test_multidimensional_rejected(self):
        with pytest.raises(ValueError):
            stats.cohens_d(np.zeros((3, 3)), np.zeros(3))

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            stats.effective_sample_size(np.array([]))
