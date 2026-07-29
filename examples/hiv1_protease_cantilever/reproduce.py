#!/usr/bin/env python3
"""Reproduce the cantilever/flap finding of Sherry et al. with confdelta.

Paper: a cryptic cantilever pocket on HIV-1 protease; immobilising the
cantilever by disulfide cross-linking makes the flap tips curl in and the
protease favour a semi-open conformation (PubMed 39109919).

This script runs confdelta's per-residue statistical comparison between a
wild-type ensemble and a cantilever-disulfide ensemble. The default feature is
per-residue mobility (RMSF): the paper's finding is a flexibility result, so we
ask where the cross-link changes mobility. The largest, most reliable per-residue
changes concentrate in the cantilever and flaps -- the paper's qualitative
finding, expressed as effect sizes with confidence intervals and FDR-corrected
q-values, per residue.

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


def _chain_of(label: str) -> str:
    return label.rpartition("_")[0]


def _plot(report, path: Path) -> None:
    """Per-residue RMSF for both conditions, flap and cantilever bands shaded.

    RMSF is recovered as ``sqrt(mean)`` of the per-frame squared-fluctuation
    feature, per condition. Only meaningful for ``feature='rmsf'``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from regions import CROSSLINK_SITES

    feats = sorted(report.features, key=lambda f: (_chain_of(f.feature), resid_of_label(f.feature)))
    x = np.arange(len(feats))
    rmsf_wt = np.sqrt(np.clip([f.mean_a for f in feats], 0, None))
    rmsf_ds = np.sqrt(np.clip([f.mean_b for f in feats], 0, None))
    resids = [resid_of_label(f.feature) for f in feats]
    in_find = [region_of(r) in FINDING_REGIONS for r in resids]

    fig, ax = plt.subplots(figsize=(11, 4))
    # Shade contiguous flap/cantilever stretches (in each chain).
    start = None
    for i, b in enumerate(in_find + [False]):
        if b and start is None:
            start = i
        elif not b and start is not None:
            ax.axvspan(start - 0.5, i - 0.5, color="#4361ee", alpha=0.08)
            start = None
    # Divider between the two chains, and the engineered cross-link sites.
    n = len(feats)
    if any(_chain_of(f.feature) == "B" for f in feats):
        first_b = next(i for i, f in enumerate(feats) if _chain_of(f.feature) == "B")
        ax.axvline(first_b - 0.5, color="#adb5bd", linewidth=0.8)
    for i, r in enumerate(resids):
        if r in CROSSLINK_SITES:
            ax.axvline(i, color="#2a9d8f", linewidth=0.9, linestyle=":")
    ax.plot(x, rmsf_wt, color="#343a40", linewidth=1.3, label="wild-type")
    ax.plot(x, rmsf_ds, color="#c1121f", linewidth=1.3, label="A71C/Q92C")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("residue (chain A then chain B)")
    ax.set_ylabel(r"C$\alpha$ RMSF ($\mathrm{\AA}$)")
    ax.set_title("Per-residue mobility, wild-type vs cantilever disulfide (A71C/Q92C)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _in_finding(feature) -> bool:
    return region_of(resid_of_label(feature.feature)) in FINDING_REGIONS


def summarise(report, top_k: int = 10) -> dict[str, float]:
    """Return the headline numbers a regression test can assert on.

    The localisation of the finding is read from the *largest* effects
    (``top_k_fraction_in_finding_regions``), not from the fraction of all
    significant residues: with one run per condition the block bootstrap
    over-powers "significance" (most residues clear q<=alpha), so where the
    biggest, most reliable changes fall is the faithful, robust echo of the
    paper's localised result.
    """
    significant = report.significant_features()
    ranked = report.ranked_by_effect()
    top = ranked[:top_k]
    sig_in_finding = [f for f in significant if _in_finding(f)]
    return {
        "n_tested": report.n_tests,
        "n_significant": len(significant),
        "n_significant_in_flap_or_cantilever": len(sig_in_finding),
        "fraction_of_significant_in_finding_regions": (
            len(sig_in_finding) / len(significant) if significant else 0.0
        ),
        "top_k": top_k,
        "top_k_in_finding_regions": sum(_in_finding(f) for f in top),
        "top_k_fraction_in_finding_regions": (
            sum(_in_finding(f) for f in top) / len(top) if top else 0.0
        ),
        "largest_effect_region": (
            region_of(resid_of_label(ranked[0].feature)) if ranked else "none"
        ),
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
    parser.add_argument(
        "--feature",
        default="rmsf",
        help="Per-residue feature to compare: 'rmsf' (mobility, default) or 'contacts'",
    )
    parser.add_argument("--correction", default="fdr_bh", help="Multiple-testing correction")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for resampling")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wt = _load_condition(args.wt, "wild_type")
    disulfide = _load_condition(args.disulfide, "cantilever_disulfide")

    report = compare_ensemble_groups(
        wt,
        disulfide,
        feature=args.feature,
        correction=args.correction,
        alpha=args.alpha,
        rng=args.seed,
    )

    _write_annotated_csv(report, outdir / "per_residue_statistics.csv")
    figure = outdir / "per_residue_rmsf.png" if args.feature == "rmsf" else None
    if figure is not None:
        _plot(report, figure)

    stats = summarise(report)
    is_rmsf = args.feature == "rmsf"
    print(f"\nWild-type vs cantilever-disulfide (A71C/Q92C)  [{args.feature}, {report.mode} mode]")
    print("=" * 64)
    if report.underpowered:
        print(f"UNDERPOWERED: p-value floor {report.min_pvalue:.3g} > alpha; read effect sizes.")
    if report.caveat:
        print(f"caveat: {report.caveat.splitlines()[0]}")
    print(f"residues tested:                 {stats['n_tested']}")
    print(f"significant (q <= {args.alpha}):           {stats['n_significant']}")
    print(
        f"largest {stats['top_k']} effects in flap/cantilever: "
        f"{stats['top_k_in_finding_regions']}/{stats['top_k']} "
        f"({stats['top_k_fraction_in_finding_regions']:.0%}); "
        f"largest single effect is in the {stats['largest_effect_region']}"
    )
    print("\nLargest effects:")
    for f in report.ranked_by_effect()[:12]:
        region = region_of(resid_of_label(f.feature))
        mark = "*" if f.significant else " "
        extra = ""
        if is_rmsf:
            extra = f"  RMSF {max(f.mean_a, 0) ** 0.5:.2f}->{max(f.mean_b, 0) ** 0.5:.2f} A"
        print(
            f"  {mark} {f.feature:<8} {region:<11} "
            f"{f.effect_size_name}={f.effect_size:+.2f} q={f.qvalue:.3g}{extra}"
        )
    written = str(outdir / "per_residue_statistics.csv")
    if figure is not None:
        written += f" and {figure}"
    print(f"\nWrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
