# Examples

Every command on this page is checked against the real command-line parser by
the test suite, so nothing here can drift out of existence.

> **Scope.** confdelta reports a per-residue **statistical** comparison —
> effect sizes with confidence intervals and multiple-testing-corrected
> q-values (written to `07_comprehensive_report/per_residue_statistics.csv`) —
> alongside the **descriptive** views below (difference matrices, centrality
> and modularity deltas, pathway changes), which carry no inference. Where this
> page says a descriptive quantity "changed", it means the number differs, not
> that the difference is significant.

## Compare two ensembles

The primary use. Wild-type against a mutant:

```bash
confdelta compare -a wt.pdb wt.xtc -b mutant.pdb mutant.xtc -o results/
```

Label the conditions so the output directories and tables read clearly:

```bash
confdelta compare \
  -a wt.pdb wt.xtc      --a-name wild_type \
  -b v82a.pdb v82a.xtc  --b-name V82A \
  -o hiv_wt_vs_v82a/
```

### What you get

```
hiv_wt_vs_v82a/
├── 01_individual_analyses/       full analysis of each ensemble separately
├── 02_network_comparisons/       edge weight and centrality changes
├── 03_dynamics_comparisons/      DCCM difference matrix and heatmap
├── 04_energetics_comparisons/    energy surface differences
├── 05_kinetics_comparisons/      MSM timescale changes
├── 06_allosteric_comparisons/    pathway disruption, hotspot rank changes
└── 07_comprehensive_report/      per-residue statistics + HTML summary
```

The tables written are:

| File | Contents |
|---|---|
| `07_comprehensive_report/per_residue_statistics.csv` | **The statistical comparison**: per-residue effect size, CI, and FDR-corrected q-value |
| `02_network_comparisons/edge_weight_changes.csv` | Per-contact weight change |
| `02_network_comparisons/<metric>_centrality_changes.csv` | Per-residue centrality change, one file per metric |
| `03_dynamics_comparisons/large_correlation_changes.csv` | Residue pairs whose \|Δρ\| exceeds `correlation_change_threshold` |
| `05_kinetics_comparisons/timescale_changes.csv` | Implied timescale ratios |
| `06_allosteric_comparisons/pathway_disruption_analysis.csv` | Per-pathway efficiency change |
| `06_allosteric_comparisons/hotspot_ranking_changes.csv` | Residues whose hotspot rank moved |

## Analyse a single ensemble

```bash
confdelta single -t system.pdb -x trajectory.xtc -n wild_type -o results/
```

Useful on its own, and it is the layer the comparison is built on.

## Change analysis options

The command line carries only what a typical run needs. Everything else lives
in a config file:

```bash
confdelta example-config -o study.json
```

That writes every option at its default. Delete the ones you do not need — an
omitted option keeps its default — then:

```bash
confdelta compare -a wt.pdb wt.xtc -b mut.pdb mut.xtc --config study.json
```

A config that narrows the analysis to networks and allostery, with named
allosteric endpoints:

```json
{
  "analysis": {
    "compute_msm": false,
    "compute_energy_landscape": false,
    "community_method": "louvain",
    "allosteric_source_nodes": ["A_50", "B_50"],
    "allosteric_target_nodes": ["A_25", "B_25"]
  },
  "comparison": {
    "correlation_change_threshold": 0.15
  }
}
```

Full option reference: [docs/CONFIGURATION.md](../CONFIGURATION.md).

## Markov state models

MSM analysis needs the optional deeptime backend:

```bash
pip install "confdelta[msm]"
```

Without it, MSM analysis is skipped and a warning is logged; nothing else is
affected. Configure it through the `analysis` section:

```json
{
  "analysis": {
    "compute_msm": true,
    "msm_lag_time": 20,
    "msm_n_clusters": 150,
    "msm_feature_type": "distances",
    "kinetic_timescales_count": 8,
    "metastable_state_count": 6
  }
}
```

If estimation fails, confdelta raises and records the error. It does not
substitute an approximate model — an earlier version silently returned invented
timescales when estimation failed.

## Python API

For just the statistical comparison, call `compare_ensemble_groups` directly.
An ensemble can be built from a trajectory, a multi-model PDB, a set of
predicted structures, or an in-memory array — all interchangeable:

```python
from confdelta import Ensemble, EnsembleGroup, compare_ensemble_groups

# Three replicate runs per condition -> the replicate is the unit of inference.
wt = EnsembleGroup(
    [Ensemble.from_trajectory("wt.pdb", f"wt_{i}.xtc") for i in range(3)], label="wild_type"
)
mut = EnsembleGroup(
    [Ensemble.from_trajectory("v82a.pdb", f"v82a_{i}.xtc") for i in range(3)], label="V82A"
)

report = compare_ensemble_groups(wt, mut, correction="fdr_bh", alpha=0.05)

for f in report.significant_features():
    print(f"{f.feature}: g={f.effect_size:+.2f} "
          f"[{f.effect_ci.low:+.2f}, {f.effect_ci.high:+.2f}]  q={f.qvalue:.3g}")

# Other sources, all interchangeable:
#   Ensemble.from_structures(["m0.pdb", "m1.pdb", ...])   # predicted structures
#   Ensemble.from_pdb_models("ensemble.pdb")              # NMR / AlphaFold models
#   Ensemble.from_coordinates(array, "topology.pdb")      # generative output
```

`report.underpowered` is True when the design cannot reach significance (three
replicates per condition have a p-value floor of 0.10); read the effect sizes in
that case. A single ensemble per condition switches to a block-bootstrap
`report.mode == "bootstrap"` with a caveat attached.

The full comparison, including the descriptive network/dynamics/allosteric
views and all output files, runs through `DifferentialAnalyzer`:

```python
from confdelta import AnalysisConfig, DifferentialAnalyzer, DifferentialConfig

analyzer = DifferentialAnalyzer(DifferentialConfig(), output_dir="results/")
results = analyzer.run_ensemble_comparison(wt, mut, AnalysisConfig(compute_msm=False))

report = results.statistical_comparison         # the ComparisonReport above
dccm_change = results.dynamics_comparison.dccm_difference_matrix
```

The statistical comparison returns typed objects (`ComparisonReport`,
`FeatureComparison`); the descriptive comparators still return the loosely
structured objects inherited from MD-Compare, so expect those to gain types
before 1.0.

---

## A note on what used to be here

This directory previously held five worked examples totalling roughly 2,500
lines. They were removed rather than updated: after the command line was
reduced from 192 flag instances to under ten per subcommand, **all 68 flags
they used had ceased to exist**, and not one of their commands could be run.

One of them also presented invented numbers as results — a pathway-disruption
table leading to "23% efficiency loss in primary allosteric pathways", a
correlation table, a centrality table, and a claim of "~40-60 significant
changes" — none of which any run had produced. Four of the CSV files it told
readers to open were never written by any code path.

The prose remains in git history. A validated worked example, reproducing a
published result with committed input data and a test asserting the numbers, is
a planned deliverable and will replace this page.
