#!/usr/bin/env python3
"""Reproduce the cantilever/flap finding of Sherry et al. with confdelta.

Paper: a cryptic cantilever pocket on HIV-1 protease; immobilising the
cantilever by disulfide cross-linking makes the flap tips curl in and the
protease favour a semi-open conformation (PubMed 39109919).

This script runs confdelta's per-residue statistical comparison between a
wild-type ensemble and a cantilever-disulfide ensemble and checks that the
significant per-residue changes concentrate in the flap and cantilever regions
-- the paper's qualitative finding, expressed as a quantitative, FDR-corrected,
per-residue result.

Each condition is one or more ensembles. Ensembles are read either as
multi-model PDBs (e.g. ColabFold output, or an MD trajectory saved as models)
or as a topology + trajectory pair. Supplying several per condition triggers
replicate-mode inference; a single one per condition uses the block bootstrap.

Usage
-----
    python reproduce.py \
        --wt        wt_ensemble.pdb \
        --disulfide ds_ensemble.pdb \
        --outdir    results/

    # replicate mode (several ensembles per condition)
    python reproduce.py \
        --wt        wt_rep1.pdb wt_rep2.pdb wt_rep3.pdb \
        --disulfide ds_rep1.pdb ds_rep2.pdb ds_rep3.pdb \
        --outdir results/

See prepare_ensembles.md for how to make the ensemble files from the workstation
trajectories.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from regions import FINDING_REGIONS, region_of, resid_of_label

from confdelta import Ensemble, EnsembleGroup, compare_ensemble_groups


def _load_condition(paths: list[str], name: str) -> EnsembleGroup:
    """Build an EnsembleGroup from one or more multi-model PDB files."""
    ensembles = []
    for i, path in enumerate(paths):
        resolved = Path(path)
        if not resolved.is_file():
            sys.exit(f"error: ensemble file not found: {resolved}")
        label = name if len(paths) == 1 else f"{name}_rep{i + 1}"
        ensembles.append(Ensemble.from_pdb_models(str(resolved), selection="name CA", name=label))
    return EnsembleGroup(ensembles, label=name)


def _write_annotated_csv(report, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["residue", "region", "effect_size", "ci_low", "ci_high", "qvalue", "significant"]
        )
        for f in report.ranked_by_effect():
            writer.writerow(
                [
                    f.feature,
                    region_of(resid_of_label(f.feature)),
                    f"{f.effect_size:.4g}",
                    f"{f.effect_ci.low:.4g}",
                    f"{f.effect_ci.high:.4g}",
                    f"{f.qvalue:.4g}",
                    int(f.significant),
                ]
            )


def _plot(report, path: Path) -> None:
    """Per-residue effect-size figure with the flap and cantilever bands shaded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = sorted(report.features, key=lambda f: resid_of_label(f.feature))
    resids = [resid_of_label(f.feature) for f in features]
    effects = [f.effect_size for f in features]
    colors = ["#c1121f" if f.significant else "#adb5bd" for f in features]

    fig, ax = plt.subplots(figsize=(11, 4))
    # Shade the regions the finding is about (flap 43-58, cantilever 62-78).
    for lo, hi, label in ((43, 58, "flap"), (62, 78, "cantilever")):
        ax.axvspan(lo, hi, color="#4361ee", alpha=0.08)
        ax.text(
            (lo + hi) / 2,
            ax.get_ylim()[1],
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#4361ee",
        )
    # Mark the engineered cross-link sites (G16C / L38C).
    for site in (16, 38):
        ax.axvline(site, color="#2a9d8f", linewidth=0.9, linestyle=":")
    ax.bar(resids, effects, color=colors, width=0.9)
    ax.axhline(0, color="#343a40", linewidth=0.8)
    ax.set_xlabel("residue number")
    ax.set_ylabel("effect size (Hedges' g)")
    ax.set_title("Per-residue change, wild-type vs cantilever-disulfide")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarise(report) -> dict[str, float]:
    """Return the headline numbers a regression test can assert on."""
    significant = report.significant_features()
    sig_in_finding = [
        f for f in significant if region_of(resid_of_label(f.feature)) in FINDING_REGIONS
    ]
    fraction = len(sig_in_finding) / len(significant) if significant else 0.0
    return {
        "n_tested": report.n_tests,
        "n_significant": len(significant),
        "n_significant_in_flap_or_cantilever": len(sig_in_finding),
        "fraction_of_significant_in_finding_regions": fraction,
        "underpowered": report.underpowered,
        "mode": report.mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wt", nargs="+", required=True, help="Wild-type ensemble PDB(s)")
    parser.add_argument(
        "--disulfide", nargs="+", required=True, help="Disulfide-construct ensemble PDB(s)"
    )
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--correction", default="fdr_bh", help="Multiple-testing correction")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for resampling")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wt = _load_condition(args.wt, "wild_type")
    disulfide = _load_condition(args.disulfide, "cantilever_disulfide")

    report = compare_ensemble_groups(
        wt, disulfide, correction=args.correction, alpha=args.alpha, rng=args.seed
    )

    _write_annotated_csv(report, outdir / "per_residue_statistics.csv")
    _plot(report, outdir / "per_residue_effect_sizes.png")

    stats = summarise(report)
    print(f"\nWild-type vs cantilever-disulfide  ({report.mode} mode)")
    print("=" * 60)
    if report.underpowered:
        print(f"UNDERPOWERED: p-value floor {report.min_pvalue:.3g} > alpha; read effect sizes.")
    if report.caveat:
        print(f"caveat: {report.caveat.splitlines()[0]}")
    print(f"residues tested:              {stats['n_tested']}")
    print(f"significant (q <= {args.alpha}):        {stats['n_significant']}")
    print(
        f"  of which in flap/cantilever: {stats['n_significant_in_flap_or_cantilever']} "
        f"({stats['fraction_of_significant_in_finding_regions']:.0%})"
    )
    print("\nLargest effects:")
    for f in report.ranked_by_effect()[:12]:
        region = region_of(resid_of_label(f.feature))
        mark = "*" if f.significant else " "
        print(f"  {mark} {f.feature:<8} {region:<11} g={f.effect_size:+.2f} q={f.qvalue:.3g}")
    print(
        f"\nWrote {outdir/'per_residue_statistics.csv'} and "
        f"{outdir/'per_residue_effect_sizes.png'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
