# Configuration reference

confdelta's command line carries only what a typical run needs. Every other
option lives in a JSON config file passed with `--config`.

Generate one with every option at its default:

```bash
confdelta example-config -o study.json
```

Then delete the options you do not need — anything omitted keeps its default —
and run:

```bash
confdelta compare -a wt.pdb wt.xtc -b mut.pdb mut.xtc --config study.json
```

## File format

Two sections, both optional:

```json
{
  "analysis": {
    "community_method": "louvain",
    "msm_lag_time": 20
  },
  "comparison": {
    "correlation_change_threshold": 0.15
  }
}
```

`analysis` configures how each ensemble is analysed; `comparison` configures how
the two are compared.

## Unknown options are errors

An unrecognised key fails the run and names the closest valid option:

```
Error: Unknown option(s) in the 'analysis' section of the config file:
  - 'msm_lag_tim'. Did you mean 'msm_lag_time'?
```

This is deliberate. Silently ignoring an unknown key means computing a result
with a default the user did not intend and never finding out.

## `analysis` options

| Option | Default | Notes |
|---|---|---|
| `cutoffs` | `None` |  |
| `interaction_types` | `None` |  |
| `threshold` | `0.2` |  |
| `timeout_seconds` | `300` |  |
| `segments` | `5` |  |
| `preprocess` | `True` |  |
| `align_selection` | `'name CA'` |  |
| `center_selection` | `'protein'` |  |
| `compute_dccm` | `True` |  |
| `compute_pca` | `True` |  |
| `pca_components` | `10` |  |
| `dccm_selection` | `'name CA'` |  |
| `compute_energy_landscape` | `True` |  |
| `landscape_temperature` | `310.0` | Kelvin |
| `landscape_bins` | `50` |  |
| `landscape_sigma` | `1.0` | Gaussian smoothing |
| `landscape_epsilon` | `1e-10` | Zero bin handling |
| `compute_communities` | `True` |  |
| `community_method` | `'leiden'` | leiden, louvain, spectral, hierarchical |
| `compute_paths` | `True` |  |
| `path_analysis_nodes` | `None` | Specific nodes for detailed path analysis |
| `allosteric_analysis` | `True` |  |
| `allosteric_source_nodes` | `None` | Source residues for allosteric analysis |
| `allosteric_target_nodes` | `None` | Target residues for allosteric analysis |
| `compute_msm` | `True` |  |
| `msm_backend` | `'auto'` | "auto" or "deeptime" (the only backend) |
| `msm_lag_time` | `10` | Lag time for MSM construction (frames) |
| `msm_n_clusters` | `100` | Number of clusters for discretization |
| `msm_stride` | `1` | Stride for coordinate extraction |
| `msm_feature_type` | `'distances'` | distances, angles, dihedrals, coordinates |
| `msm_max_distance_atoms` | `250` | cap atoms used for pairwise-distance features |
| `msm_clustering_method` | `'kmeans'` | kmeans, regular_space, minibatch_kmeans |
| `msm_validation_fraction` | `0.1` | Fraction of data for cross-validation |
| `msm_connectivity_threshold` | `0.05` | Minimum state probability for MSM |
| `compute_kinetics` | `True` |  |
| `kinetic_timescales_count` | `5` | Number of implied timescales to compute |
| `compute_metastable_states` | `True` |  |
| `metastable_state_count` | `5` | Number of metastable macrostates |

### Notes

- `cutoffs` and `interaction_types` default to `None` and are filled in at
  construction: `cutoffs` becomes `{"all_atom": 4.5, "ca_only": 8.0}` and
  `interaction_types` becomes `["distance"]`.
- `community_method` accepts `leiden`, `louvain`, `spectral` and
  `hierarchical`. `leiden` needs the `leiden` extra
  (`pip install "confdelta[leiden]"`); without it, detection falls back to
  another method.
- All `msm_*` options require the `msm` extra (`pip install "confdelta[msm]"`),
  which installs deeptime. With no MSM backend installed, MSM analysis is
  skipped and a warning is logged. `msm_backend` accepts `auto` or `deeptime`;
  PyEMMA was removed in 0.1.0.
- `allosteric_source_nodes` and `allosteric_target_nodes` take residue labels
  in `CHAIN_RESID` form, for example `["A_50", "B_50"]`. Left unset, sources
  and targets are chosen automatically from network centrality.

## `comparison` options

| Option | Default | Notes |
|---|---|---|
| `compare_networks` | `True` |  |
| `compare_dynamics` | `True` |  |
| `compare_energetics` | `True` |  |
| `compare_kinetics` | `True` |  |
| `compare_allosteric` | `True` |  |
| `statistical_feature` | `'contacts'` | Per-residue feature tested: `'contacts'` (packing) or `'rmsf'` (mobility); see confdelta.features.FEATURES |
| `multiple_comparison_correction` | `'fdr_bh'` | fdr_bh, fdr_by, bonferroni, none |
| `alpha` | `0.05` | Significance threshold on corrected q-values |
| `create_publication_figures` | `False` |  |
| `figure_dpi` | `300` |  |
| `heatmap_colormap` | `'RdBu_r'` |  |
| `create_html_report` | `True` |  |
| `correlation_change_threshold` | `0.2` | Minimum |Δρ| to report a residue pair |
| `efficiency_change_threshold` | `0.1` | Minimum communication efficiency change |
| `centrality_change_threshold` | `0.1` | Minimum centrality change |

### Notes

- `statistical_feature`, `multiple_comparison_correction` and `alpha` drive the
  **inferential** per-residue comparison: effect sizes, confidence intervals and
  corrected q-values. The three `*_change_threshold` options below drive the
  separate **descriptive** views and are magnitude filters only.

- The three `*_change_threshold` options are **magnitude filters**, not
  significance tests. They control which changes are large enough to be worth
  writing to a table. Nothing in confdelta currently tests whether a change is
  distinguishable from noise.

## What is deliberately absent

The config file's differential/network comparators report **descriptive
differences only** — node and edge deltas, correlation-matrix differences and the
like — with no significance testing or effect sizes, so there are no options to
configure them here. (Statistical comparison with effect sizes, confidence
intervals and corrected q-values is the separate `compare_ensemble_groups` API,
with the feature chosen by `statistical_feature`; see the README.)

An earlier version accepted `perform_statistical_tests`,
`significance_threshold`, `multiple_comparison_correction`,
`bootstrap_iterations` and `permutation_iterations`. None was read by any code
path — setting `multiple_comparison_correction` to `"bonferroni"` did nothing
at all. They were removed rather than left as decoration, and will return when
the statistical core is implemented. See `AUDIT.md` for the full account.
