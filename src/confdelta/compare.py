"""Statistically rigorous comparison of many features between two conditions.

This is the layer that turns the primitives in :mod:`confdelta.stats` into the
thing the library exists to provide: for each of many features (per-residue
quantities, per-pair correlations, ...) an effect size with a confidence
interval, a p-value, and -- across the whole family of features -- a
multiple-testing-corrected q-value. Effect sizes lead; p and q are secondary
(DECISIONS.md D-004).

Two modes, chosen by how much data each condition provides (DECISIONS.md D-005):

* **replicate mode** -- each condition has two or more replicate ensembles.
  The replicate is the unit of inference: each feature is reduced to one value
  per replicate, and conditions are compared with a permutation test and
  Hedges' g across replicate-level values. This is the statistically clean
  case.
* **bootstrap mode** -- each condition is a single run. Frames are
  autocorrelated, so the block bootstrap over frames supplies the uncertainty.
  A single run per condition cannot separate the condition effect from
  run-to-run variation, so every result carries that caveat.

The engine (:func:`compare_feature_matrices`, :func:`compare_frame_matrices`)
is independent of where the features come from; feature extraction and the
ensemble-level entry point live in :mod:`confdelta.features`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import stats
from .stats import ConfidenceInterval, CorrectionMethod, RandomState

__all__ = [
    "FeatureComparison",
    "ComparisonReport",
    "compare_feature_matrices",
    "compare_frame_matrices",
]

Mode = Literal["replicate", "bootstrap"]

_BOOTSTRAP_CAVEAT = (
    "Single run per condition: uncertainty is estimated by block bootstrap over "
    "frames, which captures within-run sampling error but cannot separate the "
    "condition effect from run-to-run variation. Supply replicate ensembles per "
    "condition for inference that can."
)


@dataclass(frozen=True)
class FeatureComparison:
    """Comparison of one feature between two conditions.

    ``effect_size`` is the primary quantity. It is Hedges' g (replicate mode) or
    Cohen's d (bootstrap mode); ``effect_size_name`` says which. ``effect_ci``
    is its confidence interval. ``pvalue`` and ``qvalue`` are secondary;
    ``qvalue`` is the ``pvalue`` after multiple-testing correction across all
    features in the report and is the one to compare against alpha.
    """

    feature: str
    mean_a: float
    mean_b: float
    difference: float
    effect_size: float
    effect_size_name: str
    effect_ci: ConfidenceInterval
    pvalue: float
    qvalue: float
    significant: bool


@dataclass(frozen=True)
class ComparisonReport:
    """The full result of comparing two conditions across many features.

    ``correction``, ``n_tests`` and ``alpha`` make the multiple-testing family
    explicit. ``min_pvalue`` (replicate mode) is the smallest p-value the design
    could produce; when it exceeds alpha the design is underpowered and no
    feature can be significant regardless of effect size, which
    :attr:`underpowered` reports. ``caveat`` carries the single-run warning in
    bootstrap mode.
    """

    features: list[FeatureComparison]
    mode: Mode
    correction: CorrectionMethod
    alpha: float
    n_tests: int
    n_significant: int
    n_a: int
    n_b: int
    min_pvalue: float | None
    caveat: str | None

    @property
    def underpowered(self) -> bool:
        """True when no feature could reach significance for this design.

        In replicate mode this is ``min_pvalue > alpha`` -- e.g. three
        replicates per condition, whose floor is 0.10, can never reach 0.05.
        The caller should then lead with effect sizes rather than report
        "nothing significant" as if the data were the limiting factor.
        """
        return self.min_pvalue is not None and self.min_pvalue > self.alpha

    def significant_features(self) -> list[FeatureComparison]:
        """Features whose corrected q-value is at or below alpha."""
        return [f for f in self.features if f.significant]

    def ranked_by_effect(self) -> list[FeatureComparison]:
        """Features ordered by descending absolute effect size."""
        return sorted(self.features, key=lambda f: abs(f.effect_size), reverse=True)


def _validate_matrix(m: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(m, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (rows x features); got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _check_feature_names(names: Sequence[str] | None, n_features: int) -> list[str]:
    if names is None:
        return [f"feature_{i}" for i in range(n_features)]
    names = list(names)
    if len(names) != n_features:
        raise ValueError(
            f"feature_names has {len(names)} entries but there are {n_features} features."
        )
    return names


def compare_feature_matrices(
    a: np.ndarray,
    b: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    correction: CorrectionMethod = "fdr_bh",
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_permutations: int = 10000,
    rng: RandomState = None,
) -> ComparisonReport:
    """Compare two conditions with replicate-level feature values (replicate mode).

    Parameters
    ----------
    a, b
        Arrays of shape ``(n_replicates, n_features)``: one row per replicate
        ensemble, one column per feature. The two may have different replicate
        counts but must share the feature columns.
    feature_names
        Names for the columns; defaults to ``feature_0`` ... .
    correction, alpha
        Multiple-testing correction across the feature family, and the
        significance threshold applied to the corrected q-values.
    confidence
        Confidence level for the effect-size intervals.
    n_permutations
        Maximum permutations for the per-feature permutation test; the test is
        exact when the number of distinct splits is smaller.

    Returns
    -------
    ComparisonReport
        In replicate mode, with per-feature Hedges' g, its analytic CI, a
        permutation p-value, and the FDR/Bonferroni-corrected q-value.
    """
    x = _validate_matrix(a, "a")
    y = _validate_matrix(b, "b")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"a and b must have the same number of features; got {x.shape[1]} and {y.shape[1]}."
        )
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError(
            "Replicate mode needs at least two replicates per condition; "
            f"got {x.shape[0]} and {y.shape[0]}. Use bootstrap mode for single runs."
        )
    names = _check_feature_names(feature_names, x.shape[1])
    generator = stats._as_generator(rng)

    effects: list[stats.ConfidenceInterval] = []
    pvalues = np.empty(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        col_a, col_b = x[:, j], y[:, j]
        effects.append(stats.hedges_g_interval(col_a, col_b, confidence=confidence))
        perm = stats.permutation_two_sample(col_a, col_b, n_resamples=n_permutations, rng=generator)
        pvalues[j] = perm.pvalue

    corrected = stats.correct_pvalues(pvalues, method=correction, alpha=alpha)
    min_pvalue = stats.min_attainable_pvalue(x.shape[0], y.shape[0])

    features = [
        FeatureComparison(
            feature=names[j],
            mean_a=float(x[:, j].mean()),
            mean_b=float(y[:, j].mean()),
            difference=float(x[:, j].mean() - y[:, j].mean()),
            effect_size=effects[j].estimate,
            effect_size_name="hedges_g",
            effect_ci=effects[j],
            pvalue=float(pvalues[j]),
            qvalue=float(corrected.qvalues[j]),
            significant=bool(corrected.rejected[j]),
        )
        for j in range(x.shape[1])
    ]

    return ComparisonReport(
        features=features,
        mode="replicate",
        correction=correction,
        alpha=alpha,
        n_tests=corrected.n_tests,
        n_significant=corrected.n_rejected,
        n_a=x.shape[0],
        n_b=y.shape[0],
        min_pvalue=min_pvalue,
        caveat=None,
    )


def _two_sample_block_bootstrap(
    col_a: np.ndarray,
    col_b: np.ndarray,
    statistic,
    *,
    n_resamples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Block-bootstrap distribution of a two-sample *statistic(col_a, col_b)*.

    Resamples each series with its own tau-derived block length, using the same
    circular moving-block scheme as :func:`stats.block_bootstrap`.
    """
    na, nb = col_a.size, col_b.size
    block_a = stats.auto_block_length(col_a)
    block_b = stats.auto_block_length(col_b)
    out = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        ra = col_a[stats.block_resample_indices(na, block_a, generator)]
        rb = col_b[stats.block_resample_indices(nb, block_b, generator)]
        out[i] = statistic(ra, rb)
    return out


def compare_frame_matrices(
    a_frames: np.ndarray,
    b_frames: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    correction: CorrectionMethod = "fdr_bh",
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    rng: RandomState = None,
) -> ComparisonReport:
    """Compare two single-run conditions frame-by-frame (bootstrap mode).

    Parameters
    ----------
    a_frames, b_frames
        Arrays of shape ``(n_frames, n_features)``: one row per trajectory
        frame. The two may have different frame counts.
    correction, alpha, confidence, n_resamples
        As for :func:`compare_feature_matrices`; ``n_resamples`` is the number
        of block-bootstrap resamples.

    Returns
    -------
    ComparisonReport
        In bootstrap mode, with per-feature Cohen's d (over frame-level values),
        a block-bootstrap CI for it, a block-bootstrap two-sided p-value for the
        difference of means, and the corrected q-value. ``caveat`` warns that a
        single run per condition cannot separate the effect from run-to-run
        variation.

    Notes
    -----
    Cohen's d here measures the separation of the two runs' frame-level
    distributions; it is a descriptive effect size, not evidence that the
    difference is a condition effect. The p-value is a block-bootstrap p, which
    respects autocorrelation, unlike a per-frame test that would treat
    correlated frames as independent.
    """
    x = _validate_matrix(a_frames, "a_frames")
    y = _validate_matrix(b_frames, "b_frames")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"a_frames and b_frames must have the same number of features; "
            f"got {x.shape[1]} and {y.shape[1]}."
        )
    names = _check_feature_names(feature_names, x.shape[1])
    generator = stats._as_generator(rng)

    def _mean_diff(u: np.ndarray, v: np.ndarray) -> float:
        return float(u.mean() - v.mean())

    def _cohens_d(u: np.ndarray, v: np.ndarray) -> float:
        return stats.cohens_d(u, v)

    pvalues = np.empty(x.shape[1], dtype=float)
    effect_cis: list[ConfidenceInterval] = []
    effects = np.empty(x.shape[1], dtype=float)

    for j in range(x.shape[1]):
        col_a, col_b = x[:, j], y[:, j]
        effects[j] = stats.cohens_d(col_a, col_b)

        diff_dist = _two_sample_block_bootstrap(
            col_a, col_b, _mean_diff, n_resamples=n_resamples, generator=generator
        )
        # Two-sided bootstrap p: how often the resampled difference sits on the
        # null side of zero, doubled, with a 1/n_resamples floor.
        frac_le = float(np.mean(diff_dist <= 0.0))
        frac_ge = float(np.mean(diff_dist >= 0.0))
        pvalues[j] = min(1.0, 2.0 * min(frac_le, frac_ge))
        pvalues[j] = max(pvalues[j], 1.0 / n_resamples)

        d_dist = _two_sample_block_bootstrap(
            col_a, col_b, _cohens_d, n_resamples=n_resamples, generator=generator
        )
        tail = (1.0 - confidence) / 2.0
        low, high = np.percentile(d_dist, [100.0 * tail, 100.0 * (1.0 - tail)])
        effect_cis.append(
            ConfidenceInterval(
                estimate=float(effects[j]),
                low=float(low),
                high=float(high),
                confidence=confidence,
                method="block-bootstrap-percentile",
            )
        )

    corrected = stats.correct_pvalues(pvalues, method=correction, alpha=alpha)

    features = [
        FeatureComparison(
            feature=names[j],
            mean_a=float(x[:, j].mean()),
            mean_b=float(y[:, j].mean()),
            difference=float(x[:, j].mean() - y[:, j].mean()),
            effect_size=float(effects[j]),
            effect_size_name="cohens_d",
            effect_ci=effect_cis[j],
            pvalue=float(pvalues[j]),
            qvalue=float(corrected.qvalues[j]),
            significant=bool(corrected.rejected[j]),
        )
        for j in range(x.shape[1])
    ]

    return ComparisonReport(
        features=features,
        mode="bootstrap",
        correction=correction,
        alpha=alpha,
        n_tests=corrected.n_tests,
        n_significant=corrected.n_rejected,
        n_a=x.shape[0],
        n_b=y.shape[0],
        min_pvalue=None,
        caveat=_BOOTSTRAP_CAVEAT,
    )
