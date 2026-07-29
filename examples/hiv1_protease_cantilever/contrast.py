#!/usr/bin/env python3
"""Contrast two cross-links: only the cantilever one curls the flaps in.

Same enzyme, same method, two different engineered disulfides:

* **A71C/Q92C** clamps the *cantilever* (residue 71 sits inside it) to the
  C-terminal strand -- the paper's construct.
* **G16C/L38C** clamps the *fulcrum* (16) to the *elbow* (38), the flap's hinge.

Comparing each construct's per-residue RMSF against the wild type shows they are
not the same perturbation. The cantilever cross-link quiets the flap tips -- they
curl in -- while the fulcrum-elbow cross-link leaves them fully mobile. That
specificity (same enzyme, different cross-link, different dynamics) is what
confdelta resolves, and it is the mechanistic point of the paper's cantilever
pocket: it is *the cantilever* that gates the flaps.

This is a descriptive, region-level companion to `reproduce.py` (which does the
rigorous per-residue statistical comparison for A71C/Q92C). Run: ``python
contrast.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from regions import region_of, resid_of_label

from confdelta import Ensemble
from confdelta.features import per_residue_rmsf

HERE = Path(__file__).parent
WILD_TYPE = HERE / "wt_ensemble.pdb"
CONSTRUCTS = {
    "A71C/Q92C (cantilever)": HERE / "ds_ensemble.pdb",
    "G16C/L38C (fulcrum-elbow)": HERE / "ds_g16c_l38c_ensemble.pdb",
}
REGION_ORDER = ["fulcrum", "elbow", "flap", "flap_tip", "cantilever", "catalytic", "core"]


def rmsf_by_region(ensemble_path) -> dict[str, float]:
    """Mean Cα RMSF (angstrom) per structural region for one ensemble."""
    ensemble = Ensemble.from_pdb_models(str(ensemble_path), selection="name CA", name="x")
    labels, values = per_residue_rmsf(ensemble)
    rmsf = np.sqrt(values.mean(axis=0))
    grouped: dict[str, list[float]] = {}
    for label, value in zip(labels, rmsf, strict=True):
        grouped.setdefault(region_of(resid_of_label(label)), []).append(float(value))
    return {region: float(np.mean(v)) for region, v in grouped.items()}


def _plot(wt, per_construct, path: Path) -> None:
    """Grouped bars of RMSF change vs wild type, per region, for both constructs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = REGION_ORDER
    x = np.arange(len(regions))
    colours = {"A71C/Q92C (cantilever)": "#c1121f", "G16C/L38C (fulcrum-elbow)": "#7209b7"}
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, region_rmsf) in enumerate(per_construct.items()):
        deltas = [region_rmsf[r] - wt[r] for r in regions]
        ax.bar(x + (i - 0.5) * width, deltas, width, label=name, color=colours.get(name))
    ax.axhline(0, color="#343a40", linewidth=0.8)
    # Point at the flap tips: the one region where the two constructs disagree.
    tip = regions.index("flap_tip")
    ax.axvspan(tip - 0.5, tip + 0.5, color="#ffd60a", alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta$ mean C$\alpha$ RMSF vs wild type ($\mathrm{\AA}$)")
    ax.set_title("Two cross-links, two dynamics: only A71C/Q92C quiets the flap tips")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    wt = rmsf_by_region(WILD_TYPE)
    per_construct = {name: rmsf_by_region(path) for name, path in CONSTRUCTS.items()}

    names = list(per_construct)
    print(f"Mean Cα RMSF (Å) by region — wild type vs each construct\n{'=' * 60}")
    header = f"{'region':<12}{'WT':>7}" + "".join(f"{n.split()[0]:>13}{'Δ':>7}" for n in names)
    print(header)
    for region in REGION_ORDER:
        row = f"{region:<12}{wt[region]:>7.2f}"
        for name in names:
            val = per_construct[name][region]
            row += f"{val:>13.2f}{val - wt[region]:>+7.2f}"
        print(row)

    tips = {n: per_construct[n]["flap_tip"] for n in names}
    print(
        f"\nFlap-tip mobility (RMSF, Å): wild type {wt['flap_tip']:.2f}; "
        + "; ".join(f"{n.split()[0]} {tips[n]:.2f}" for n in names)
    )
    print(
        "The cantilever cross-link curls the flap tips in; the fulcrum-elbow "
        "one leaves them mobile."
    )

    figure = HERE / "results" / "cross_link_contrast.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    _plot(wt, per_construct, figure)
    print(f"\nWrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
