"""HIV-1 protease structural regions, by residue number.

Standard 99-residue-per-chain numbering (the enzyme is a homodimer, chains A and
B). These ranges are the widely-used definitions; adjust them if your construct
is numbered differently. Used to annotate each residue in the comparison so the
flap and cantilever signal can be read off directly.
"""

from __future__ import annotations

# (name, inclusive residue range). Order matters: the first match wins, so the
# catalytic triad and flap tips (subsets of larger regions) come first.
REGION_RANGES: list[tuple[str, range]] = [
    ("catalytic", range(25, 28)),  # Asp25-Thr26-Gly27
    ("flap_tip", range(48, 53)),  # Gly48..Gly52, the tips that curl in
    ("fulcrum", range(11, 23)),
    ("elbow", range(35, 43)),
    ("flap", range(43, 59)),
    ("cantilever", range(59, 76)),
]

# The regions the paper's finding is about: immobilising the cantilever makes the
# flap tips curl in and the protease favour a semi-open conformation. A faithful
# reproduction should concentrate its significant per-residue changes here.
FINDING_REGIONS = {"flap", "flap_tip", "cantilever"}


def region_of(resid: int) -> str:
    """Return the structural region a residue number falls in, or 'core'."""
    for name, span in REGION_RANGES:
        if resid in span:
            return name
    return "core"


def resid_of_label(label: str) -> int:
    """Extract the integer residue number from a ``CHAIN_RESID`` label."""
    _, _, resid = label.rpartition("_")
    return int(resid)
