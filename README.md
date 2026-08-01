# confdelta

**Statistically rigorous comparison of two protein conformational ensembles — networks, dynamics and kinetics — with proper multiple-testing correction.**

![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)
[![DOI](https://zenodo.org/badge/1205859988.svg)](https://doi.org/10.5281/zenodo.21678329)

Comparing two ensembles — wild-type against a mutant, apo against holo, one design against another — is something almost every MD study does and almost everyone hand-rolls in a notebook: subtract two averages, eyeball the difference, and rarely correct for having just run hundreds of per-residue comparisons at once. confdelta makes that comparison a first-class, tested operation: per residue it reports an **effect size with a confidence interval** and a **p-value corrected across all residues**, and it is honest about when a design is too small to conclude anything.

Single-trajectory analysis (residue interaction networks, DCCM, PCA, free-energy landscapes, Markov state models) is here too, but as the layer the comparison is built on — the comparison is the point.

## Install

```bash
pip install confdelta
```

Python 3.10–3.13, no conda required. Markov state models need one extra: `pip install "confdelta[msm]"`.

## Compare two ensembles

```bash
confdelta compare \
  -a wild_type.pdb wild_type.xtc  --a-name wild_type \
  -b mutant.pdb    mutant.xtc     --b-name mutant \
  -o results/
```

The run writes `results/07_comprehensive_report/per_residue_statistics.csv` — one row per residue, effect size and confidence interval first, corrected q-value after — and prints the largest effects. The output has this shape (the layout is real; the residues and numbers below are placeholders, not results from any particular system):

```
residue        g   ci_low  ci_high        q  sig
A_82       +2.41    +1.36    +3.46   0.0108    *
A_84       +1.62    +0.93    +2.31   0.0108    *
A_50       -1.54    -2.06    -1.02   0.0144    *
A_25       -0.97    -2.16    +0.23   0.2684
```

`g` is Hedges' g (effect size); `sig` marks residues significant after Benjamini–Hochberg
correction across every residue tested. The DCCM difference heatmap and the descriptive
network/allosteric tables are written alongside.

If you supply **replicate** ensembles per condition, the replicate is the unit of inference. If you supply a **single run** per condition, a block bootstrap over frames supplies the uncertainty and the output says plainly that one run cannot separate the condition effect from run-to-run variation. And if the design cannot reach significance at all — three replicates per condition have a p-value floor of 0.10 — confdelta tells you that and points you to the effect sizes, instead of reporting a misleading "nothing significant".

The `compare` command takes one run per condition, so it uses block-bootstrap (single-run)
inference. For **replicate-level** inference, build `EnsembleGroup`s from several runs and
use the Python API shown below.

## Common tasks

```bash
# Analyse a single ensemble on its own
confdelta single -t system.pdb -x trajectory.xtc -n my_run -o results/

# Generate a config file with every option, then run with it
confdelta example-config -o study.json
confdelta compare -a wt.pdb wt.xtc -b mut.pdb mut.xtc --config study.json
```

Every option not shown on the command line lives in the config file; the full reference is
in **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**, generated from the code and checked
against it by the test suite.

## Python API

The comparison is designed to be imported. An ensemble is built from any source — a
trajectory, a multi-model PDB, a set of predicted structures, or an in-memory coordinate
array — and the comparison code does not care which:

```python
from confdelta import Ensemble, EnsembleGroup, compare_ensemble_groups

wt  = EnsembleGroup([Ensemble.from_trajectory("wt.pdb",   f"wt_{i}.xtc")   for i in range(3)], label="wild_type")
mut = EnsembleGroup([Ensemble.from_trajectory("v82a.pdb", f"v82a_{i}.xtc") for i in range(3)], label="V82A")

report = compare_ensemble_groups(wt, mut, correction="fdr_bh", alpha=0.05)

for f in report.significant_features():
    print(f"{f.feature}: g={f.effect_size:+.2f} "
          f"[{f.effect_ci.low:+.2f}, {f.effect_ci.high:+.2f}]  q={f.qvalue:.3g}")
```

`report` is typed (`ComparisonReport` / `FeatureComparison`): `report.underpowered`,
`report.mode` (`"replicate"` or `"bootstrap"`), `report.ranked_by_effect()`, and the
per-residue effect sizes, confidence intervals and q-values are all attributes, not files.
Other sources: `Ensemble.from_structures([...])`, `.from_pdb_models(path)`,
`.from_coordinates(array, topology)`.

## What it does

- **Compare two ensembles, with statistics** — per-residue effect sizes, confidence
  intervals and FDR/Bonferroni-corrected q-values; replicate-level or block-bootstrap
  inference; honest reporting of underpowered designs. *This is the core.*
- **Descriptive difference views** — DCCM difference matrices, centrality and modularity
  deltas, energy-surface differences, allosteric pathway and hotspot changes. Reported
  without inference, alongside the statistics.
- **Single-ensemble analysis** — residue interaction networks and community detection,
  dynamic cross-correlation, PCA and free-energy landscapes, and (with the `msm` extra)
  Markov state models. The substrate the comparison is built on.

## Status

confdelta is `0.1.0` and pre-1.0: the statistical comparison returns typed objects, but the
descriptive comparators still return the loosely structured objects inherited from its
predecessor. Features wired so far are per-residue **contact number** and **RMSF**, plus
user-defined **geometric collective variables** (distances and angles). Expect the API to
firm up before 1.0. It continues **MD-Compare**; see [CHANGELOG.md](CHANGELOG.md) for the
lineage and [AUDIT.md](AUDIT.md) for an account of what was rebuilt and why.

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md) — installation and extras
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every configuration option
- [docs/geometric_cvs.md](docs/geometric_cvs.md) — measuring custom distances and angles
- [docs/examples/README.md](docs/examples/README.md) — worked examples

## Licence

MIT — see [LICENSE](LICENSE).
