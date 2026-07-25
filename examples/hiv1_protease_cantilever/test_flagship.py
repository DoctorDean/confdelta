"""Regression test for the HIV-1 protease cantilever flagship.

Runs the wild-type vs cantilever-disulfide comparison on the committed ensemble
files and asserts the paper's finding: the significant per-residue changes
concentrate in the flap and cantilever regions rather than scattering across the
enzyme.

The test **skips** until the ensemble files are committed to this directory (see
prepare_ensembles.md). Once they are present it runs in CI, so a future change
that breaks the reproduction fails the build. The thresholds below encode the
qualitative published claim (a majority of the significant residues fall in the
flap/cantilever regions); tighten them against the real run when the data lands.
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
    return compare_ensemble_groups(wt, ds, correction="fdr_bh", alpha=0.05, rng=0)


def test_some_residues_change_significantly(report):
    # Unless the design is underpowered, immobilising the cantilever must move
    # something; if it is underpowered, that is reported, not hidden.
    if report.underpowered:
        pytest.skip(f"design underpowered (p floor {report.min_pvalue:.3g}); see effect sizes")
    assert report.n_significant > 0


def test_significant_changes_concentrate_in_flap_and_cantilever(report):
    if report.underpowered or report.n_significant == 0:
        pytest.skip("no significant residues to localise")
    significant = report.significant_features()
    in_finding = [f for f in significant if region_of(resid_of_label(f.feature)) in FINDING_REGIONS]
    fraction = len(in_finding) / len(significant)
    # The published finding: the change is localised to the flaps and cantilever.
    assert fraction >= 0.5, (
        f"only {fraction:.0%} of significant residues fall in the flap/cantilever "
        "regions; the reproduction expects a majority there."
    )


def test_largest_effect_is_in_a_finding_region(report):
    if report.underpowered:
        pytest.skip("design underpowered")
    top = report.ranked_by_effect()[0]
    assert region_of(resid_of_label(top.feature)) in FINDING_REGIONS
