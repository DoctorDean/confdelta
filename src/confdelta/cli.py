#!/usr/bin/env python3

"""confdelta command-line interface.

Two subcommands:

``compare``
    The primary one. Compares two conformational ensembles.

``single``
    Analyses one ensemble on its own. Secondary; it exists because the
    comparison is built on it.

The command line carries only what a typical run needs. Every other analysis
option lives in a JSON config file passed with ``--config``; run
``confdelta example-config`` to generate one populated with every option at its
default. This replaced a 5-subcommand, 192-flag interface in which two of the
subcommands overlapped and one advertised against itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config, write_example_config
from .core import AnalysisConfig, MDCompare
from .differential import DifferentialAnalyzer, DifferentialConfig
from .ensemble import Ensemble, EnsembleError, EnsembleGroup
from .utils import PerformanceMonitor, save_analysis_config


def _print_statistical_summary(report) -> None:
    """Print the per-residue statistical comparison, effect sizes first."""
    if report is None:
        print("\nPer-residue statistical comparison: not available.")
        return

    print("\nPer-residue statistical comparison")
    print("-" * 70)
    print(f"  mode:       {report.mode}")
    print(
        f"  correction: {report.correction}  (alpha {report.alpha}, "
        f"{report.n_tests} residues tested)"
    )

    if report.underpowered:
        print(
            f"  UNDERPOWERED: the smallest p-value this design can produce is "
            f"{report.min_pvalue:.3g}, above alpha, so no residue can reach\n"
            f"  significance regardless of effect size. Read the effect sizes below."
        )
    else:
        print(f"  significant residues (q <= {report.alpha}): {report.n_significant}")

    if report.caveat:
        # Wrap the single-run caveat under a clear marker.
        print("  caveat: " + report.caveat.replace("\n", "\n          "))

    ranked = report.ranked_by_effect()[:10]
    print("\n  Largest effects (residue: effect size [95% CI], q):")
    for feature in ranked:
        marker = "*" if feature.significant else " "
        print(
            f"   {marker} {feature.feature:<10} "
            f"{feature.effect_size:+.2f} "
            f"[{feature.effect_ci.low:+.2f}, {feature.effect_ci.high:+.2f}]  "
            f"q={feature.qvalue:.3g}"
        )
    print("\n  Full table: <output>/07_comprehensive_report/per_residue_statistics.csv")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class InputError(ValueError):
    """Raised for invalid user input, with a message meant for the user."""


def _validate_readable_file(path: str, what: str) -> Path:
    """Return *path* as a Path, or raise InputError explaining what is wrong."""
    resolved = Path(path)
    if not resolved.exists():
        raise InputError(f"{what} not found: {resolved}")
    if resolved.is_dir():
        raise InputError(f"{what} is a directory, not a file: {resolved}")
    try:
        with resolved.open("rb"):
            pass
    except OSError as exc:
        raise InputError(f"{what} cannot be read: {resolved} ({exc.strerror})") from exc
    return resolved


def _validate_trajectories(paths: list[str], label: str) -> str:
    """Validate trajectory paths for one condition and return the single one.

    Multiple trajectories per condition are accepted by the parser because that
    is the shape replicate-aware comparison needs, but pooling them here would
    concatenate independent runs into one pseudo-trajectory and destroy the
    replicate structure -- the exact error the statistical core is being built
    to avoid. So more than one is rejected rather than silently mishandled.
    """
    for path in paths:
        _validate_readable_file(path, f"Trajectory for condition {label}")
    if len(paths) > 1:
        raise InputError(
            f"Condition {label} was given {len(paths)} trajectories. Replicate-aware\n"
            f"comparison is not implemented yet, and concatenating replicates into one\n"
            f"trajectory would misrepresent them as a single continuous run.\n"
            f"Supply one trajectory per condition for now. If you genuinely want the\n"
            f"frames pooled, concatenate them yourself first so the choice is explicit."
        )
    return paths[0]


def _load_configs(config_path: str | None) -> tuple[AnalysisConfig, DifferentialConfig]:
    if config_path is None:
        return AnalysisConfig(), DifferentialConfig()
    return load_config(config_path)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def run_single(args: argparse.Namespace) -> int:
    """Analyse a single ensemble."""
    topology = _validate_readable_file(args.topology, "Topology")
    trajectory = _validate_readable_file(args.trajectory, "Trajectory")
    analysis_config, _ = _load_configs(args.config)

    ensemble = Ensemble.from_trajectory(str(topology), str(trajectory), name=args.name)

    workflow = MDCompare(analysis_config, args.output)
    monitor = PerformanceMonitor()

    monitor.start_step("Loading ensemble")
    workflow.add_prepared_simulation(ensemble.to_simulation())
    monitor.end_step()

    monitor.start_step("Analysis")
    results = workflow.run_analysis([args.name])
    monitor.end_step()

    if args.name in results:
        summary = results[args.name]
        print(f"\nResults for {args.name}:")
        print(f"  Contact maps:    {summary['contact_maps_computed']}")
        print(f"  Network nodes:   {summary['network_nodes']}")
        print(f"  Network edges:   {summary['network_edges']}")
        print(f"  Network density: {summary['network_density']:.4f}")

    save_analysis_config(analysis_config, Path(args.output) / "analysis_config.json")
    monitor.print_summary()
    print(f"\nResults saved to: {args.output}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    """Compare two conformational ensembles."""
    topology_a = _validate_readable_file(args.a_topology, "Topology for condition A")
    topology_b = _validate_readable_file(args.b_topology, "Topology for condition B")
    trajectory_a = _validate_trajectories(args.a_trajectory, "A")
    trajectory_b = _validate_trajectories(args.b_trajectory, "B")

    if args.a_name == args.b_name:
        raise InputError(
            f"Both conditions are named {args.a_name!r}. They must differ, because the\n"
            f"names label the output directories and every comparison table."
        )

    analysis_config, comparison_config = _load_configs(args.config)

    group_a = EnsembleGroup.single(
        Ensemble.from_trajectory(str(topology_a), trajectory_a, name=args.a_name)
    )
    group_b = EnsembleGroup.single(
        Ensemble.from_trajectory(str(topology_b), trajectory_b, name=args.b_name)
    )

    analyzer = DifferentialAnalyzer(comparison_config, args.output)
    monitor = PerformanceMonitor()

    monitor.start_step("Comparison")
    results = analyzer.run_ensemble_comparison(group_a, group_b, analysis_config)
    monitor.end_step()

    computed = [
        name
        for name, comparison in (
            ("network", results.network_comparison),
            ("dynamics", results.dynamics_comparison),
            ("energetics", results.energetics_comparison),
            ("kinetics", results.kinetics_comparison),
            ("allosteric", results.allosteric_comparison),
        )
        if comparison is not None
    ]

    print("\n" + "=" * 70)
    print(f"COMPARISON: {results.simulation_names[0]} vs {results.simulation_names[1]}")
    print("=" * 70)

    _print_statistical_summary(results.statistical_comparison)

    print(f"\nDescriptive comparisons computed: {', '.join(computed) if computed else 'none'}")
    print(
        "  (difference matrices, centrality and modularity deltas, pathway changes;\n"
        "   descriptive only -- see the statistical comparison above for inference)"
    )

    monitor.print_summary()
    print(f"\nResults saved to: {args.output}")
    return 0


def run_example_config(args: argparse.Namespace) -> int:
    path = write_example_config(args.output)
    print(f"Example configuration written to: {path}")
    print("Every option is set to its default. Delete the ones you do not need.")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="confdelta",
        description="confdelta: statistical comparison of protein conformational ensembles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Compare two ensembles (the primary use)
  confdelta compare -a wt.pdb wt.xtc -b mutant.pdb mutant.xtc -o results/

  # Name the conditions so the output tables read clearly
  confdelta compare -a wt.pdb wt.xtc --a-name wild_type \\
                    -b v82a.pdb v82a.xtc --b-name V82A \\
                    -o hiv_wt_vs_v82a/

  # Change any analysis option via a config file
  confdelta example-config -o study.json
  confdelta compare -a wt.pdb wt.xtc -b mut.pdb mut.xtc --config study.json

  # Analyse one ensemble on its own
  confdelta single -t system.pdb -x traj.xtc -n my_run -o results/

Every analysis option not shown above lives in the config file. Generate one
with `confdelta example-config`; it is written with all options at their
defaults and is documented in docs/CONFIGURATION.md.

https://github.com/DoctorDean/confdelta
        """,
    )
    subparsers = parser.add_subparsers(dest="mode", metavar="{compare,single,example-config}")

    # -- compare ------------------------------------------------------------
    compare = subparsers.add_parser(
        "compare",
        help="Compare two conformational ensembles (primary command)",
        description="Compare two conformational ensembles across networks, dynamics, "
        "energetics and kinetics.",
    )
    compare.add_argument(
        "-a",
        "--a",
        dest="a",
        nargs=2,
        metavar=("TOPOLOGY", "TRAJECTORY"),
        required=True,
        help="Condition A: topology file followed by trajectory file",
    )
    compare.add_argument(
        "-b",
        "--b",
        dest="b",
        nargs=2,
        metavar=("TOPOLOGY", "TRAJECTORY"),
        required=True,
        help="Condition B: topology file followed by trajectory file",
    )
    compare.add_argument("--a-name", default="A", help="Label for condition A (default: A)")
    compare.add_argument("--b-name", default="B", help="Label for condition B (default: B)")
    compare.add_argument("-o", "--output", default="confdelta_results", help="Output directory")
    compare.add_argument("-c", "--config", help="JSON config file with analysis options")

    # -- single -------------------------------------------------------------
    single = subparsers.add_parser(
        "single",
        help="Analyse a single ensemble",
        description="Analyse one conformational ensemble: networks, dynamics, "
        "energetics and kinetics.",
    )
    single.add_argument("-t", "--topology", required=True, help="Topology file")
    single.add_argument("-x", "--trajectory", required=True, help="Trajectory file")
    single.add_argument("-n", "--name", required=True, help="Label for this ensemble")
    single.add_argument("-o", "--output", default="confdelta_results", help="Output directory")
    single.add_argument("-c", "--config", help="JSON config file with analysis options")

    # -- example-config -----------------------------------------------------
    example = subparsers.add_parser(
        "example-config",
        help="Write a config file with every option at its default",
    )
    example.add_argument(
        "-o", "--output", default="confdelta.json", help="Where to write the config file"
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return 1

    # `-a TOPOLOGY TRAJECTORY` is friendlier on the command line than four
    # separate flags, but the rest of the code wants them apart. Trajectories
    # are carried as lists because that is the shape replicates will need.
    if args.mode == "compare":
        args.a_topology, trajectory_a = args.a
        args.b_topology, trajectory_b = args.b
        args.a_trajectory = [trajectory_a]
        args.b_trajectory = [trajectory_b]

    handlers = {
        "compare": run_compare,
        "single": run_single,
        "example-config": run_example_config,
    }

    try:
        return handlers[args.mode](args)
    except (InputError, ConfigError, EnsembleError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
