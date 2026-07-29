"""Regression test for the cross-link contrast.

Asserts the region-level RMSF signature that tells the two engineered disulfides
apart: the cantilever cross-link (A71C/Q92C) quiets the flap tips, the
fulcrum-elbow cross-link (G16C/L38C) does not. A descriptive companion to
test_flagship.py. Skips until the three ensemble files are committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from contrast import CONSTRUCTS, WILD_TYPE, rmsf_by_region  # noqa: E402

A71C = "A71C/Q92C (cantilever)"
G16C = "G16C/L38C (fulcrum-elbow)"


@pytest.fixture(scope="module")
def region_rmsf():
    if not (WILD_TYPE.is_file() and all(p.is_file() for p in CONSTRUCTS.values())):
        pytest.skip("Contrast ensemble files not committed yet; see prepare_ensembles.md.")
    wt = rmsf_by_region(WILD_TYPE)
    per_construct = {name: rmsf_by_region(path) for name, path in CONSTRUCTS.items()}
    return wt, per_construct


def test_cantilever_crosslink_quiets_the_flap_tips(region_rmsf):
    wt, per_construct = region_rmsf
    # The flap tips curl in: a large drop in flap-tip RMSF vs wild type.
    assert wt["flap_tip"] - per_construct[A71C]["flap_tip"] > 0.8


def test_fulcrum_elbow_crosslink_leaves_the_flap_tips_mobile(region_rmsf):
    wt, per_construct = region_rmsf
    # Clamping the fulcrum-elbow hinge does not quiet the flap tips.
    assert wt["flap_tip"] - per_construct[G16C]["flap_tip"] < 0.3


def test_the_two_constructs_differ_most_at_the_flap_tips(region_rmsf):
    wt, per_construct = region_rmsf
    a71, g16 = per_construct[A71C], per_construct[G16C]
    gap = {r: abs((a71[r] - wt[r]) - (g16[r] - wt[r])) for r in wt}
    assert max(gap, key=lambda r: gap[r]) == "flap_tip"
