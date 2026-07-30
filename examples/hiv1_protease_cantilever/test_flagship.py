"""Regression test for the HIV-1 protease cantilever flagship.

Runs the wild-type vs cantilever-disulfide (A71C/Q92C) comparison of per-residue
mobility (RMSF) on the committed ensemble files and asserts the paper's finding:
immobilising the cantilever rigidifies the cantilever and flaps, so the largest
per-residue mobility changes localise there rather than scattering across the
enzyme.

Localisation is read from the *largest* effects, not from the fraction of all
significant residues: with one 100 ns run per condition the block bootstrap
over-powers "significance" (most residues clear q<=alpha), so the fraction of
significant residues in any region just tracks that region's size. Where the
biggest, most reliable changes fall is the robust, faithful echo of the paper's
localised result. Replicate ensembles would sharpen this to a permutation test;
see prepare_ensembles.md.

This guards the *computation* on the committed single-run data — that the code
still produces the rep-1 result — not a replicate-confirmed scientific claim. A
two-replicate check shows the flap-tip effect is under-sampled at 100 ns and does
not yet reproduce in magnitude (README.md, DECISIONS.md D-039); the fix is more
and longer replicates, not a change to this assertion.

The test **skips** until the ensemble files are committed to this directory.
Once present it runs in CI, so a future change that breaks the reproduction
fails the build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from regions import FINDING_REGIONS, region_of, resid_of_label  # noqa: E402

from confdelta import Ensemble, EnsembleGroup, compare_ensemble_groups  # noqa: E402


def _ensembles(prefix: str) -> list[Path]:
    """Committed ensemble files for a condition: prefix.pdb or prefix_repN.pdb."""
    single = sorted(HERE.glob(f"{prefix}.pdb"))
    replicates = sorted(HERE.glob(f"{prefix}_rep*.pdb"))
    return replicates or single


def _group(prefix: str, label: str) -> EnsembleGroup | None:
    paths = _ensembles(prefix)
    if not paths:
        return None
    ensembles = [
        Ensemble.from_pdb_models(str(p), selection="name CA", name=f"{label}_{i}")
        for i, p in enumerate(paths)
    ]
    return EnsembleGroup(ensembles, label=label)


@pytest.fixture(scope="module")
def report():
    wt = _group("wt_ensemble", "wild_type") or _group("wt", "wild_type")
    ds = _group("ds_ensemble", "disulfide") or _group("ds", "disulfide")
    if wt is None or ds is None:
        pytest.skip("Flagship ensemble files not committed yet; see prepare_ensembles.md.")
    return compare_ensemble_groups(wt, ds, feature="rmsf", correction="fdr_bh", alpha=0.05, rng=0)


def test_some_residues_change_significantly(report):
    # Unless the design is underpowered, immobilising the cantilever must move
    # something; if it is underpowered, that is reported, not hidden.
    if report.underpowered:
        pytest.skip(f"design underpowered (p floor {report.min_pvalue:.3g}); see effect sizes")
    assert report.n_significant > 0


def test_largest_mobility_changes_localise_to_flap_and_cantilever(report):
    # The published finding, read from the largest effects (see module docstring
    # for why not "fraction of all significant"): a clear majority of the ten
    # biggest per-residue mobility changes fall in the flap/cantilever regions.
    top = report.ranked_by_effect()[:10]
    in_finding = [f for f in top if region_of(resid_of_label(f.feature)) in FINDING_REGIONS]
    assert len(in_finding) >= 7, (
        f"only {len(in_finding)}/10 of the largest mobility changes fall in the "
        "flap/cantilever regions; the reproduction expects them concentrated there."
    )


def test_largest_effect_is_cantilever_rigidification(report):
    # The single biggest, most reliable change is a residue of the flap/cantilever
    # becoming *less* mobile -- the cross-link immobilising what it was designed to.
    if report.underpowered:
        pytest.skip("design underpowered")
    top = report.ranked_by_effect()[0]
    assert region_of(resid_of_label(top.feature)) in FINDING_REGIONS
    # mean_a is wild-type MSF, mean_b the disulfide's; rigidification means a > b.
    assert top.mean_a > top.mean_b, "largest effect should be a loss of mobility (rigidification)"
