"""Statistical primitives for two-ensemble comparison.

Pure functions, no I/O. Each is validated against a known analytical result or
an independent implementation in the test suite; none is validated against
itself.

What this module is for
-----------------------
confdelta compares two conformational ensembles. For any scalar derived from an
ensemble -- a contact frequency, a per-residue centrality, a correlation -- the
question is whether it differs between conditions, by how much, and whether the
difference is distinguishable from noise. This module provides the pieces to
answer that honestly:

* **effect sizes** (Cohen's d, Hedges' g, Cliff's delta) -- *how much* changed,
  the primary output. These stay informative even at the small replicate counts
  typical of MD (three runs per condition), where significance testing cannot.
* **integrated autocorrelation time** and **effective sample size** -- how much
  independent information a single trajectory actually carries. A 10,000-frame
  run with a correlation time of 100 frames is worth ~100 independent samples,
  not 10,000, and pretending otherwise is the field's most common error.
* a **block bootstrap** -- confidence intervals that respect that
  autocorrelation, unlike the IID bootstrap SciPy ships.
* a **permutation test** over replicate-level values, plus the *minimum
  attainable p-value* for a given replicate count, so the tool can say when a
  design simply cannot reach significance.
* **multiple-testing correction** (Benjamini-Hochberg, Benjamini-Yekutieli,
  Bonferroni) for the hundreds of simultaneous per-residue tests a comparison
  produces.

Design note
-----------
Effect sizes lead; p-values and q-values are secondary and are never the sole
basis for a reported change. See DECISIONS.md D-004 for why: an exact two-sided
permutation test on three replicates per condition has a minimum attainable
p-value of 0.10, so a p-value-led tool would report "nothing significant" for
most real, well-powered comparisons.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb
from typing import Literal

import numpy as np
from scipy import stats

__all__ = [
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "cohens_d",
    "hedges_g",
    "cliffs_delta",
    "hedges_g_interval",
    "cliffs_delta_interpretation",
    "ConfidenceInterval",
    "block_bootstrap",
    "block_bootstrap_interval",
    "block_resample_indices",
    "auto_block_length",
    "PermutationResult",
    "permutation_two_sample",
    "min_attainable_pvalue",
    "MultipleTestingResult",
    "correct_pvalues",
    "CorrectionMethod",
]

# A random source: a Generator, an int seed, or None (nondeterministic).
RandomState = np.random.Generator | int | None


def _as_generator(rng: RandomState) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _validate_1d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


# ===========================================================================
# Autocorrelation and effective sample size
# ===========================================================================


def integrated_autocorrelation_time(x: np.ndarray, *, c: float = 5.0) -> float:
    r"""Integrated autocorrelation time :math:`\tau_{int}` of a 1-D series.

    Defined as :math:`\tau_{int} = 1 + 2\sum_{k\ge1}\rho_k`, where
    :math:`\rho_k` is the lag-*k* autocorrelation. The sum is truncated with
    Sokal's adaptive window: it stops at the smallest lag *M* with
    ``M >= c * tau(M)``, which balances bias against the variance of the tail.

    For an AR(1) process with parameter :math:`\phi`, the true value is
    :math:`(1+\phi)/(1-\phi)`; the estimator recovers it to about 1% on long
    series (verified in the tests).

    Returns at least 1.0 (an uncorrelated series has :math:`\tau_{int}=1`).
    """
    arr = _validate_1d(x, "x")
    n = arr.size
    if n < 2:
        return 1.0
    centered = arr - arr.mean()
    if np.allclose(centered, 0.0):
        # A constant series has no fluctuations; treat as uncorrelated.
        return 1.0

    # Autocovariance via FFT (Wiener-Khinchin), normalised to autocorrelation.
    size = int(2 ** np.ceil(np.log2(2 * n)))
    freq = np.fft.fft(centered, n=size)
    acf = np.fft.ifft(freq * np.conjugate(freq))[:n].real
    acf /= acf[0]

    cumulative = np.cumsum(acf[1:])
    for window in range(1, n):
        tau = 1.0 + 2.0 * cumulative[window - 1]
        if window >= c * tau:
            return max(1.0, float(tau))
    return max(1.0, float(1.0 + 2.0 * cumulative[-1]))


def effective_sample_size(x: np.ndarray) -> float:
    r"""Effective sample size :math:`N_{eff} = N / \tau_{int}`.

    The number of independent samples a correlated series of length *N* is worth.
    Always in ``(0, N]``.
    """
    arr = _validate_1d(x, "x")
    tau = integrated_autocorrelation_time(arr)
    return arr.size / tau


# ===========================================================================
# Effect sizes
# ===========================================================================


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's *d*: standardised difference of means, ``(mean_a - mean_b)/s_p``.

    ``s_p`` is the pooled standard deviation with (n-1) weighting. Positive when
    *a* is larger. Returns 0.0 when both groups are constant with equal means,
    and ``inf`` when the pooled SD is zero but the means differ.
    """
    x = _validate_1d(a, "a")
    y = _validate_1d(b, "b")
    if x.size < 2 or y.size < 2:
        raise ValueError("Cohen's d needs at least two observations per group.")
    na, nb = x.size, y.size
    var_a, var_b = x.var(ddof=1), y.var(ddof=1)
    pooled = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    mean_diff = x.mean() - y.mean()
    if pooled == 0.0:
        if mean_diff == 0.0:
            return 0.0
        return float(np.sign(mean_diff) * np.inf)
    return float(mean_diff / np.sqrt(pooled))


def _hedges_correction(n_a: int, n_b: int) -> float:
    """Small-sample bias-correction factor J for Hedges' g."""
    df = n_a + n_b - 2
    # The standard approximation to the exact gamma-ratio J; accurate to
    # better than 0.01 for df >= 4 and asymptoting to 1 as df grows.
    return 1.0 - 3.0 / (4.0 * df - 1.0)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' *g*: Cohen's *d* with the small-sample bias correction.

    Cohen's *d* is biased upward for small samples; *g* multiplies it by a
    correction factor that shrinks it toward zero. The two agree to within a
    percent or so beyond a few dozen samples, but at MD replicate counts the
    correction matters, which is why *g* is the effect size confdelta reports.
    """
    x = _validate_1d(a, "a")
    y = _validate_1d(b, "b")
    return cohens_d(x, y) * _hedges_correction(x.size, y.size)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    r"""Cliff's delta: a nonparametric effect size in :math:`[-1, 1]`.

    :math:`\delta = P(a > b) - P(a < b)`, estimated over all pairs. It is +1
    when every value of *a* exceeds every value of *b*, -1 in the reverse case,
    and 0 under stochastic equality. Unlike *d*/*g* it makes no distributional
    assumption and is not distorted by outliers, which suits the skewed
    distributions many MD-derived quantities have.
    """
    x = _validate_1d(a, "a")
    y = _validate_1d(b, "b")
    # Pairwise sign comparison, vectorised. For the sizes confdelta handles
    # (replicate counts, or per-residue value sets) the na*nb matrix is small.
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.mean())


def cliffs_delta_interpretation(delta: float) -> str:
    """Map |Cliff's delta| to the conventional negligible/small/medium/large label.

    Thresholds are Romano et al.'s widely-used cutoffs (0.147 / 0.33 / 0.474).
    A label is a convenience for reading output, never a substitute for the
    number.
    """
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval for a point estimate.

    ``method`` records how it was produced (e.g. ``"hedges-g-analytic"``,
    ``"block-bootstrap-percentile"``) so a consumer can tell an analytic
    interval from a resampled one.
    """

    estimate: float
    low: float
    high: float
    confidence: float
    method: str

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def hedges_g_interval(
    a: np.ndarray, b: np.ndarray, *, confidence: float = 0.95
) -> ConfidenceInterval:
    """Hedges' *g* with an approximate large-sample confidence interval.

    The interval uses the standard normal approximation
    ``g ± z * SE(g)`` with
    ``SE(g) = sqrt((n_a+n_b)/(n_a n_b) + g^2 / (2(n_a+n_b)))``. It is a
    large-sample approximation: at three-versus-three it is wide and only
    indicative. For a distribution-free interval, bootstrap the estimate with
    :func:`block_bootstrap_interval` (or an ordinary bootstrap for independent
    replicates).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}.")
    x = _validate_1d(a, "a")
    y = _validate_1d(b, "b")
    g = hedges_g(x, y)
    na, nb = x.size, y.size
    se = np.sqrt((na + nb) / (na * nb) + g**2 / (2.0 * (na + nb)))
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    return ConfidenceInterval(
        estimate=g,
        low=g - z * se,
        high=g + z * se,
        confidence=confidence,
        method="hedges-g-analytic",
    )


# ===========================================================================
# Block bootstrap
# ===========================================================================


def auto_block_length(x: np.ndarray, *, multiplier: float = 2.0) -> int:
    r"""Choose a block length from the integrated autocorrelation time.

    Returns ``ceil(multiplier * tau_int)``, clamped to ``[1, n // 2]``. Blocks
    of roughly :math:`2\tau_{int}` preserve most of the series' dependence while
    leaving enough blocks to resample; this is the standard default.
    """
    arr = _validate_1d(x, "x")
    tau = integrated_autocorrelation_time(arr)
    length = int(np.ceil(multiplier * tau))
    return max(1, min(arr.size // 2 if arr.size >= 2 else 1, length))


def block_resample_indices(n: int, block_length: int, generator: np.random.Generator) -> np.ndarray:
    """Indices for one moving circular block resample of a length-*n* series.

    Draws ``ceil(n / block_length)`` blocks of consecutive indices from random
    wrap-around start points and returns the first *n* of the concatenation.
    Exposed so that a two-sample statistic (which must resample two series with
    one scheme) can share the exact resampling used by :func:`block_bootstrap`.
    """
    n_blocks = int(np.ceil(n / block_length))
    starts = generator.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length)
    return (starts[:, None] + offsets[None, :]).ravel()[:n] % n


def block_bootstrap(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 2000,
    block_length: int | None = None,
    rng: RandomState = None,
) -> np.ndarray:
    """Moving circular block bootstrap of *statistic* over an autocorrelated series.

    Resamples length-*n* series by drawing ``ceil(n / block_length)`` blocks of
    consecutive observations from random (wrap-around) start points and
    concatenating them. Circular wrapping keeps every observation equally likely
    to appear, avoiding the end-effect bias of the non-circular scheme.

    An IID bootstrap (SciPy's ``bootstrap``) assumes exchangeable observations
    and badly understates uncertainty for autocorrelated data; the block
    bootstrap resamples blocks precisely to keep the within-block dependence.

    Returns the bootstrap distribution of the statistic (length ``n_resamples``).
    """
    arr = _validate_1d(x, "x")
    n = arr.size
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1.")
    if block_length is None:
        block_length = auto_block_length(arr)
    if not 1 <= block_length <= n:
        raise ValueError(f"block_length must be in [1, {n}]; got {block_length}.")

    generator = _as_generator(rng)
    out = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        out[i] = statistic(arr[block_resample_indices(n, block_length, generator)])
    return out


def block_bootstrap_interval(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    block_length: int | None = None,
    rng: RandomState = None,
) -> ConfidenceInterval:
    """Percentile confidence interval for *statistic* via the block bootstrap.

    Coverage approaches the nominal level as the effective sample size grows and
    is far closer to nominal than an IID bootstrap on the same autocorrelated
    data (both are checked in the test suite). At small effective sample sizes
    it undercovers somewhat -- an inherent property of estimating a long-run
    variance from a short dependent series, shared by all methods -- so the
    interval should be read as near-nominal, not exact.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}.")
    arr = _validate_1d(x, "x")
    distribution = block_bootstrap(
        arr, statistic, n_resamples=n_resamples, block_length=block_length, rng=rng
    )
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(distribution, [100.0 * tail, 100.0 * (1.0 - tail)])
    return ConfidenceInterval(
        estimate=float(statistic(arr)),
        low=float(low),
        high=float(high),
        confidence=confidence,
        method="block-bootstrap-percentile",
    )


# ===========================================================================
# Permutation test
# ===========================================================================


Alternative = Literal["two-sided", "less", "greater"]


def min_attainable_pvalue(n_a: int, n_b: int, *, alternative: Alternative = "two-sided") -> float:
    r"""Smallest p-value an exact permutation test can return for these group sizes.

    An exact two-sample permutation test enumerates the :math:`\binom{n_a+n_b}{n_a}`
    ways of splitting the pooled data. The most extreme observed statistic still
    shares its bin with the identity permutation, so the one-sided p-value cannot
    fall below :math:`1/\binom{n_a+n_b}{n_a}` and the two-sided below twice that.

    This is the honesty function behind confdelta's stance on small designs:
    with three replicates per condition the two-sided floor is 0.10, so **no
    effect, however large, can reach significance at 0.05** -- let alone survive
    multiple-testing correction. Callers use it to warn rather than silently
    report "not significant". See DECISIONS.md D-004.
    """
    if n_a < 1 or n_b < 1:
        raise ValueError("Both groups need at least one observation.")
    total = comb(n_a + n_b, n_a)
    floor = 1.0 / total
    if alternative == "two-sided":
        return min(1.0, 2.0 * floor)
    return floor


@dataclass(frozen=True)
class PermutationResult:
    """Outcome of a two-sample permutation test."""

    statistic: float
    pvalue: float
    alternative: Alternative
    n_resamples: int
    exact: bool
    min_pvalue: float

    @property
    def underpowered(self) -> bool:
        """True when the design cannot reach p < 0.05 regardless of effect size.

        i.e. the minimum attainable p-value already exceeds 0.05. A caller
        should surface this rather than report a non-significant result as if
        the data were the limiting factor.
        """
        return self.min_pvalue > 0.05


def permutation_two_sample(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alternative: Alternative = "two-sided",
    n_resamples: int = 10000,
    rng: RandomState = None,
) -> PermutationResult:
    """Two-sample permutation test on the difference of means.

    Wraps :func:`scipy.stats.permutation_test`. When the number of distinct
    splits is small enough, the test is computed exactly (every permutation
    enumerated) and ``exact`` is True; otherwise ``n_resamples`` random
    permutations are used. The result carries the minimum attainable p-value for
    the group sizes so callers can distinguish "no effect" from "design too
    small to detect one".
    """
    x = _validate_1d(a, "a")
    y = _validate_1d(b, "b")

    def _statistic(u: np.ndarray, v: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.mean(u, axis=axis) - np.mean(v, axis=axis)

    total_splits = comb(x.size + y.size, x.size)
    exact = total_splits <= n_resamples
    result = stats.permutation_test(
        (x, y),
        _statistic,
        permutation_type="independent",
        alternative=alternative,
        n_resamples=np.inf if exact else n_resamples,
        rng=_as_generator(rng),
    )
    return PermutationResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        alternative=alternative,
        n_resamples=int(total_splits if exact else n_resamples),
        exact=exact,
        min_pvalue=min_attainable_pvalue(x.size, y.size, alternative=alternative),
    )


# ===========================================================================
# Multiple-testing correction
# ===========================================================================


CorrectionMethod = Literal["fdr_bh", "fdr_by", "bonferroni", "none"]


@dataclass(frozen=True)
class MultipleTestingResult:
    """Result of correcting a family of p-values.

    ``qvalues`` are adjusted p-values comparable directly against ``alpha``.
    ``method``, ``n_tests`` and ``alpha`` make the correction family explicit in
    the output, so a reader can see how many tests were corrected over and how.
    """

    pvalues: np.ndarray
    qvalues: np.ndarray
    rejected: np.ndarray
    method: CorrectionMethod
    alpha: float
    n_tests: int

    @property
    def n_rejected(self) -> int:
        return int(self.rejected.sum())


def correct_pvalues(
    pvalues: np.ndarray,
    *,
    method: CorrectionMethod = "fdr_bh",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Correct a family of p-values for multiple testing.

    A two-ensemble comparison produces one test per residue (or per pair), so a
    few hundred simultaneously. Reporting raw p-values would guarantee false
    positives; this applies a correction over the whole family and records which
    one.

    Methods:

    * ``"fdr_bh"`` -- Benjamini-Hochberg FDR (default). Controls the expected
      false-discovery proportion; assumes independence or positive dependence.
    * ``"fdr_by"`` -- Benjamini-Yekutieli FDR. Valid under arbitrary dependence,
      so more conservative.
    * ``"bonferroni"`` -- controls the family-wise error rate; the most
      conservative.
    * ``"none"`` -- no correction (q == p). Provided for completeness; using it
      across many tests is rarely defensible.

    BH and BY delegate to :func:`scipy.stats.false_discovery_control`, which the
    tests cross-check against ``statsmodels.multipletests``.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"pvalues must be one-dimensional; got shape {p.shape}.")
    if p.size == 0:
        raise ValueError("pvalues is empty.")
    if np.any((p < 0.0) | (p > 1.0)) or not np.all(np.isfinite(p)):
        raise ValueError("pvalues must all be finite and in [0, 1].")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")

    n = p.size
    if method == "none":
        q = p.copy()
    elif method == "bonferroni":
        q = np.minimum(p * n, 1.0)
    elif method == "fdr_bh":
        q = stats.false_discovery_control(p, method="bh")
    elif method == "fdr_by":
        q = stats.false_discovery_control(p, method="by")
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"Unknown correction method: {method!r}.")

    return MultipleTestingResult(
        pvalues=p,
        qvalues=q,
        rejected=q <= alpha,
        method=method,
        alpha=alpha,
        n_tests=n,
    )
