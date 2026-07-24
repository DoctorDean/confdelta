# MD-Compare — Phase 0 Audit

**Date:** 2026-07-24
**Commit audited:** `13e7722` (`master`, clean tree)
**Auditor:** automated agent session, per the Phase 0 brief
**Method:** clean-room install into a fresh Python 3.12 virtualenv, full test-suite
execution, direct execution of the installed package's internals, and static reading of
all 11 003 lines of `src/`.

> **Ground rule applied throughout:** nothing in the README or docs was assumed true.
> Every claim below was executed or read in source. Where a finding is asserted, the
> evidence is reproduced.

---

## 1. Executive summary

The packaging is in **better** shape than expected. The science is in **worse** shape than
expected, and specifically the one capability the project's strategic repositioning depends
on — rigorous statistical comparison of two ensembles — **does not exist**. It is a set of
dataclasses, a directory-creating orchestrator, and hardcoded constants that are printed to
the user as scientific findings.

| Dimension | Verdict |
|---|---|
| Install from clean (`pip install .`, Python 3.12) | **Works.** No conda, no troubleshooting needed. |
| Test suite | **80 tests, 80 passing, 9.9 s.** ~25 make a real correctness assertion. |
| Coverage | 36% overall |
| Single-ensemble analysis (networks, DCCM, PCA, landscapes, MSM) | **Sound.** Verified analytically. |
| **Two-ensemble differential comparison** | **Placeholder.** See §5. |
| README accuracy | **61 of 90 documented CLI flags do not exist (68%).** |
| Public API stability | Low. File-path-locked input; results delivered as side effects. |

The repositioning is the right call. But it is a **build**, not a rewrite of the README.

---

## 2. Install and build

Fresh `python3.12 -m venv`, then `pip install /path/to/MD-Compare`:

```
Successfully built md-compare
Successfully installed GridDataFormats-1.2.0 MDAnalysis-2.10.0 contourpy-1.3.3
  cycler-0.12.1 filelock-3.32.0 fonttools-4.63.0 joblib-1.5.3 kiwisolver-1.5.0
  matplotlib-3.11.1 md-compare-1.5.0 mda-xdrlib-0.2.0 mmtf-python-1.1.3 mrcfile-1.5.4
  msgpack-1.2.1 narwhals-2.24.0 networkx-3.6.1 numpy-1.26.4 pandas-3.0.5 pillow-12.3.0
  pyparsing-3.3.2 python-dateutil-2.9.0.post0 scikit-learn-1.9.0 scipy-1.17.1
  seaborn-0.13.2 six-1.17.0 threadpoolctl-3.6.0 tqdm-4.69.0
```

- Wheel and sdist build cleanly.
- `import mdcompare`, all five submodules, and `md-compare --help` all work.
- **The README's "Troubleshooting → PyEMMA Installation Problems" section is already
  obsolete** for the pip path, because nothing in the core install needs PyEMMA.

**Blocker for modernisation:** the `numpy>=1.21.0,<2.0` ceiling in `pyproject.toml` held
numpy to 1.26.4 while resolving scipy 1.17.1 and pandas 3.0.5 alongside it. The ceiling is
almost certainly a leftover PyEMMA constraint and is the main thing standing between this
package and a modern dependency stack.

---

## 3. Test suite: how many tests assert science?

**80 tests, all passing.** Of those, roughly **25 make a real correctness assertion**; the
remaining ~55 are smoke tests that assert only `isinstance(result, dict)` or
`result is not None` — i.e. they verify that code ran without raising, not that it was right.

| File | Tests | Real correctness assertions |
|---|---|---|
| `tests/test_msm_backends.py` | 27 | **~14 — the best file in the repository.** Asserts stationary distribution sums to 1, leading eigenvalue ≈ 1, MFPT diagonal is zero, transition-matrix shape consistency, PCCA+ coarse-grained matrix shape and normalisation. This genuinely validates the deeptime adapter. |
| `tests/test_core.py` | 18 | ~10. Contact maps symmetric, bounded [0,1], correct shape; chain-ID fallback regression; pairwise-distance subsampling cap (50 atoms → 1225 pairs, not 400²/2). |
| `tests/test_resistance_classifier.py` | 11 | mixed |
| `tests/test_utils.py` | 13 | 3 (sequence-distance matrix shape / symmetry / zero diagonal). The other 10 assert only that a `dict` came back. |
| `tests/test_differential.py` | 11 | ~2 (identical inputs → zero difference matrix). **Nothing tests a p-value, an effect size, or a correction.** |

Coverage by module:

```
cli.py                                    406 stmts   4%
core.py                                  2726 stmts  29%
differential.py                           901 stmts  51%
msm_backends.py                           182 stmts  80%
experimental/resistance_classifier.py     148 stmts  86%
utils.py                                  327 stmts  49%
TOTAL                                    4743 stmts  36%
```

`tests/conftest.py` is good and should be kept and extended: it builds deterministic,
seeded, in-memory MDAnalysis universes, so the suite needs no external data and runs fast.

---

## 4. Subsystem classification

| Subsystem | Verdict | Evidence |
|---|---|---|
| Residue interaction networks / contact maps | **WORKS** | Verified: symmetric to 1e-6, all values in [0,1], shape (n_res, n_res). Tested. |
| Centrality | **WORKS** | Populated per-node dict via networkx. |
| Community detection (Leiden / Louvain / spectral / hierarchical) | **WORKS** | Verified: modularity 0.6228 on a synthetic 2-chain 24-residue system, 4 communities, within valid [-1,1]. Leiden path needs the `leiden` extra. |
| DCCM | **WORKS** | Verified analytically: symmetric, unit diagonal, bounded [-1,1] (observed range −0.198 … 1.000). |
| PCA | **WORKS** | Verified analytically: eigenvalues descending and non-negative; eigenvectors orthonormal — Gram matrix = I to 1e-6 over the first 3 components of a (72, 5) eigenvector matrix. |
| Free energy landscapes | **WORKS** | Verified: Boltzmann inversion with the minimum correctly shifted to 0 kT (observed min = 0.000000, max = 34.999 kT); gradients, Laplacian, minima and barriers all populated. |
| Markov state models | **WORKS (deeptime path)** | `msm_backends.py` is real and well-tested. **But** see the `SimplifiedMSM` defect in §6. |
| Allosteric pathway analysis | **RUNS-UNVERIFIED** | Produces populated output (pathways, communication-efficiency matrix, hotspots, redundancy, robustness, functional regions). No test asserts any value is correct. |
| Network robustness | **RUNS-UNVERIFIED** | `utils.py` comment: "Simplified robustness analysis". Smoke-tested only. |
| **Differential comparison framework** | **BROKEN** | §5. |

---

## 5. The differential framework — the critical finding

This is the v1.5.0 headline feature and the intended strategic core of the whole project.

### 5.1 There is no statistical machinery at all

- `multipletests` is **never called anywhere in the codebase**. `statsmodels` is not a
  dependency. The `--multiple-comparison-correction` flag and
  `DifferentialConfig.multiple_comparison_correction` are stored and never read.
- `bootstrap_iterations` is stored on the config and never read. There is no bootstrap
  anywhere. Two comments say "could implement bootstrap".
- There are exactly **four** real statistical calls in the entire package
  (`ks_2samp`, `mannwhitneyu`, `pearsonr`, `spearmanr`), all inside one function —
  `utils.py:488 compare_centrality_distributions` — which is **wired to nothing**.
- `differential.py:887` is a section header that reads, literally:
  `# PLACEHOLDER COMPARATOR CLASSES`.

### 5.2 Fabricated results are printed to the user as findings

Executed directly against the installed package:

```python
DifferentialAnalyzer._generate_executive_summary(None, {},                    ("WT","MUT"))
DifferentialAnalyzer._generate_executive_summary(None, {"a":1,"b":2,"c":3},   ("X","Y"))
```

Both return, byte for byte:

```
{'comparison': ..., 'total_analyses': ...,
 'significant_findings': 5,
 'overall_similarity': 0.78,
 'key_insights': ['Significant changes in allosteric communication',
                  'Altered energy landscape minima',
                  'Modified kinetic pathways']}
```

Only the two echoed fields vary. Everything a reader would interpret as a result is a
constant. Similarly:

| Function | Location | Behaviour |
|---|---|---|
| `_calculate_similarity_scores` | `differential.py:493` | Always returns `{'network': 0.85, 'dynamics': 0.72}` |
| `_identify_most_significant_changes` | `differential.py:512` | Always returns `[{'type': 'placeholder', 'significance': 0.001, 'description': 'Example change'}]` |
| `_perform_statistical_analysis` | `differential.py:453` | A `for` loop whose body is `pass` |
| `NetworkComparator._test_network_significance` | `differential.py:1105` | **Ignores both graph arguments.** Returns literal `{'topology_change_pvalue': 0.05, 'centrality_change_pvalue': 0.01, 'community_change_pvalue': 0.1}` |
| `_test_energy_significance`, `_test_population_changes`, `_test_pathway_significance`, `_test_hotspot_significance` | `differential.py:1605, 1611, 2252, 2260` | All placeholder stubs |

**These constants reach the user.** `cli.py:274-300`, inside
`run_comprehensive_differential_analysis` — the command `md-compare --help` labels
"(recommended)" — prints on every run:

```
Significant findings: 5
Overall similarity score: 0.780
Key insights:
  • Significant changes in allosteric communication
  • Altered energy landscape minima
  • Modified kinetic pathways
```

They are also written to disk: `overall_similarity_scores.csv` receives the hardcoded
0.85/0.72, and the pickled `ComprehensiveDifferentialResults` embeds the lot.

### 5.3 The one apparent statistical test is not a statistical test

`DynamicsComparator._perform_correlation_statistics`, `differential.py:1332`:

```python
p_value = max(0.001, 1.0 - abs(dccm_diff[i, j]))
```

A monotone transform of effect magnitude, labelled a p-value and thresholded at
`significance_threshold` (0.05). Measured behaviour:

| Δρ between the two ensembles | reported "significant pairs" | min reported "p" |
|---|---|---|
| 0.0 (identical ensembles) | 0 | 1.000 |
| 0.5 (large, biologically dramatic) | **0** | 0.500 |
| 1.9 (near the theoretical maximum) | 10 | 0.001 |

A residue pair can only be called significant when |Δρ| > 0.95. For DCCM differences that
essentially never occurs, so the test has **near-zero statistical power** — and would be
meaningless even if it fired, because the quantity is not a p-value.

The `effect_sizes` dict returned alongside contains no effect size. It is counts of matrix
cells above arbitrary thresholds (0.05 / 0.15 / 0.3), plus the mean absolute difference.
No Cohen's d, no Hedges' g, no Cliff's delta.

### 5.4 What the framework *does* have, and is worth keeping

The **descriptive** layer is largely real and should be preserved through the rebuild:

- Network: node and edge set differences, edge-weight changes, per-metric centrality deltas,
  centrality-ranking changes, community-structure comparison, modularity change, global
  property deltas.
- Dynamics: DCCM difference matrix, thresholded correlation changes, regional (quadrant)
  breakdown, PCA variance deltas, eigenvector overlap similarities, PC shift scores.
- Energetics: energy-surface difference, minima comparison, barrier-height changes,
  stability and entropy change estimates.
- Kinetics: timescale changes and ratios, transition-flux differences, metastable-state
  population changes, critical-pathway and bottleneck analysis.
- Allosteric: efficiency-matrix differences, pathway disruption scoring, hotspot rank
  changes, cross-chain communication analysis.

What is missing is **every inferential claim layered on top of it**.

---

## 6. Other correctness defects

**Fabricated MSM timescales.** `core.py:1267-1329`: when MSM construction fails, the code
prints "Creating simplified analysis with basic statistics…", builds a hand-rolled
`SimplifiedMSM` object, and returns:

```python
return simplified_model, np.array([1, 3, 5, 7, 9]), [np.array([10.0, 5.0, 2.0])]
```

The implied timescales `[10.0, 5.0, 2.0]` are invented. A silent fallback that manufactures
kinetic data is worse than an exception.

**Paired data analysed with an unpaired test.** `utils.py:488`
`compare_centrality_distributions` extracts centrality values for the *same* nodes from two
networks — matched pairs — then applies `mannwhitneyu`, an independent-samples test. The
correct test for node-matched values is Wilcoxon signed-rank. (`ks_2samp` on the marginals
is defensible as a distribution-shape comparison; Mann-Whitney is not.) The function is
currently unused, so nothing is presently wrong in output — but the intent must be
reimplemented correctly, not moved.

**Silently shrinking public API.** `__init__.py:31-73` wraps every public import in a bare
`try/except ImportError` that, on any failure, silently drops 10 names from `__all__`. A
downstream consumer gets an undiagnosable `ImportError` instead of "install the X extra".

---

## 7. Documentation accuracy

### 7.1 The README documents a CLI that largely does not exist

Extracted every `--flag` token from `README.md` and diffed against the live argparse tree:

- README documents **90** distinct long flags.
- The real CLI has **73** unique long flags across 5 subcommands (192 flag instances).
- **61 of the 90 documented flags do not exist** — 68%.
- Conversely, every flag the CLI *does* define is genuinely consumed. There are **no dead
  flags** in the implementation.

The entire "Statistical Analysis Parameters" section is invented:
`--significance-testing`, `--n-bootstrap`, `--confidence-level`,
`--multiple-testing-correction`, `--compare-method`, `--effect-size-threshold` — none exist.
So are `--verbose`, `--debug`, `--output-dir`, `--cutoff-distance`, `--dpi`, `--n-cores`,
`--resolution`, `--contact-selection`, `--centrality-measures`, all 12 `--plot-*` flags, and
the entire "Platform-Specific Options" and "Performance and Debugging" blocks.

### 7.2 The docs present invented numbers as expected results

`docs/examples/02_differential_analysis.md` shows numeric result tables labelled "Expected
findings" with scientific interpretation attached — for example a pathway-disruption table
leading to "**Flap-to-Active Site Communication Disrupted**: 23% efficiency loss in primary
allosteric pathways", and a correlation table leading to "Flaps (50-51) lose correlation with
active site (25-26)". No run produced these numbers. It further claims "Expected: ~40-60
significant changes across all analysis types".

It also documents four CSV files the pipeline never writes:
`statistical_significance_tests.csv`, `network_property_changes.csv`,
`pca_variance_changes.csv`, `cross_chain_communication_changes.csv`.

### 7.3 Minor

`md-compare --help` ships the placeholder URL `https://github.com/yourusername/md-compare`.
`docs/INSTALL.md` is real and useful but stale — it leads with conda and targets Python 3.8+.

---

## 8. CLI surface

192 flag instances, 73 unique, across five subcommands with unclear boundaries:

| Subcommand | Flags | Notes |
|---|---|---|
| `single` | 42 | Single-simulation analysis |
| `compare` | 39 | Multi-simulation, driven by a JSON config |
| `diff` | 46 | Self-described "legacy"; **prints a message telling the user to use `differential` instead** |
| `differential` | 68 | The "recommended" one — and the one that prints fabricated findings |
| `example-config` | 2 | Config scaffolding |

The package ships a deprecated command that advertises against itself, and two further
commands (`compare`, `differential`) whose distinction is not discoverable from `--help`.

---

## 9. Public API surface

`mdcompare/__init__.py` exports: `NetworkAnalyzer`, `AnalysisConfig`, `NetworkMetrics`,
`MDSimulation`, `SimulationConfig`, `MDComparator`, `OutputManager`, `MDCompare`,
`DifferentialAnalyzer`, `DifferentialConfig`.

Stability assessment for a downstream importer:

- **Ensemble input is hard-locked to files.** `MDSimulation` requires topology and trajectory
  *paths*; `run_differential_analysis()` takes six path strings. There is no supported way to
  pass an in-memory ensemble, a multi-model PDB, or a set of predicted structures. This
  blocks the source-agnostic requirement at its foundation — and, now that replicate-aware
  statistics are planned, it also blocks passing *n* runs per condition.
- **Results are side effects.** Value is delivered as CSVs, PNGs, and a pickled
  `ComprehensiveDifferentialResults`. There are no typed numeric return objects for
  downstream code to consume.
- **Comparators cannot be used standalone.** `DifferentialAnalyzer.__init__` creates seven
  output directories on construction (`differential.py:270`) before any analysis is requested.
- No meaningful type hints on the public surface; no `py.typed` marker.
- The import guard described in §6 makes partial installs undiagnosable.

---

## 10. Dependency risks

| Risk | Detail |
|---|---|
| `numpy>=1.21.0,<2.0` | **The main modernisation blocker.** Forces numpy 1.26 alongside scipy 1.17 / pandas 3.0. Almost certainly a leftover PyEMMA constraint. |
| PyEMMA | Effectively end-of-life, yet still the *preferred* backend in `msm_backends.select_backend("auto")` and the only MSM entry in the `all` extra. The deeptime path is present and better tested. |
| Python 3.8 | EOL October 2024. Still in `requires-python`, classifiers, ruff `target-version`, black `target-version`, and the CI matrix. |
| Python 3.13 | Not tested anywhere. |
| `scikit-learn` | Declared both as a core dependency and inside the `ml` extra. |
| `statsmodels` | Not a dependency — and, per §12, does not need to become one. |

---

## 11. Naming (availability checked; the brief's shortlist had not been)

| Candidate | PyPI | GitHub | Verdict |
|---|---|---|---|
| `mdelta` *(brief's recommendation)* | **TAKEN** — dormant single-cell lineage-tree aligner, last release Sep 2022 | — | **Ruled out** |
| `confdiff` | free | **`bytedance/ConfDiff` — 86★, ICML'24, protein conformation *generation*** | **Ruled out.** Same domain; it generates the ensembles this tool would consume. |
| `parallax` | **TAKEN** | — | Ruled out |
| `dynadiff` | free | `facebookresearch/dynadiff` — 47★, fMRI decoding | Weak |
| `trajdiff` | free | clean | Excluded by the brief (MD-locked) |
| `ensemblediff` | free | clean | Ensembl collision stands |
| **`confdelta`** | **free** | 1 repo, 0★, trivial | **Selected** |
| `ensdelta` | free | clean | Strong runner-up |

Cross-cutting finding: in 2026 a `*diff` suffix in a structural-biology name reads as
**diffusion model**, not *difference*. This argues against the entire `diff` family and
toward `delta`.

---

## 12. Findings that shape the rebuild

### 12.1 SciPy already provides most of the statistical machinery

Verified against scipy 1.17.1:

| Need | Available | Notes |
|---|---|---|
| Benjamini–Hochberg and Benjamini–Yekutieli FDR | `scipy.stats.false_discovery_control` | **Present.** Verified to produce monotone step-up q-values, with BY uniformly more conservative than BH. **Removes any need for a `statsmodels` runtime dependency.** |
| Exact and Monte-Carlo permutation tests | `scipy.stats.permutation_test` | Present; supports `permutation_type="independent"` and exact enumeration via `n_resamples=np.inf` |
| BCa / percentile / basic bootstrap CIs | `scipy.stats.bootstrap` | Present |
| **Block bootstrap over autocorrelated frames** | — | **Absent.** The one genuinely new numerical component that must be written. |

### 12.2 The replicate-count constraint

`scipy.stats.permutation_test` run exactly (`n_resamples=np.inf`) against a deliberately
absurd true effect (group means 0 vs 50):

| replicates per condition | exact two-sided p | minimum attainable p |
|---|---|---|
| 3 v 3 | **0.1000** | 0.1000 |
| 4 v 4 | 0.0286 | 0.0286 |
| 5 v 5 | 0.0079 | 0.0079 |

**Three replicates per condition can never yield a significant result at α = 0.05 from an
exact two-sided permutation test, regardless of effect size** — let alone survive FDR
correction across hundreds of residues. Most published MD studies run exactly three.

This is not a limitation to conceal. Detecting it at input-validation time and saying so
plainly is the single most valuable honest behaviour this library can offer, and it is why
the rebuilt API must **lead with effect sizes and confidence intervals and report p/q as
secondary**.

---

## 13. Conclusion

Keep: the single-ensemble analysis stack (verified correct), the deeptime MSM adapter and its
tests, the synthetic-universe test fixtures, and the descriptive layer of the comparators.

Delete: every fabricated constant and every mislabelled statistic, the `SimplifiedMSM`
fallback, the invented documentation, and roughly two thirds of the CLI.

Build: a real statistical core, and a source-agnostic, replicate-aware ensemble input model
to feed it.

The strategic repositioning is correct and the gap in the ecosystem is real. But the work is
to **make the claim true**, not to restate it.
