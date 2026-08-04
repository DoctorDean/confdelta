# How-to: geometric collective variables

confdelta's built-in features (`contacts`, `rmsf`) are computed for every residue
automatically. When you have a *specific* geometric coordinate in mind — a
flap-opening distance, a hinge angle — use **`geometric_features`** to measure
exactly that and compare it between two conditions with the same effect
size / confidence interval / FDR-corrected q-value machinery.

## Define the CVs

```python
from confdelta import geometric_features

cvs = geometric_features(
    distances=[("A_25", "A_50")],                  # a distance, in angstroms
    angles=[("A_48", "A_49", "A_50")],             # an angle, in degrees
    dihedrals=[("A_47", "A_48", "A_49", "A_50")],  # a torsion, in (-180, 180] degrees
)
```

- A **distance** is a pair of points; an **angle** is a triple (measured at the
  *middle* point); a **dihedral** is a quadruple (the torsion about the central
  bond). Pass as many as you like.
- A **point** is a residue, or a **group** of residues measured at its centroid —
  pass a list to compare, say, one domain against another:

  ```python
  geometric_features(distances=[(["A_23", "A_24", "A_25"], ["A_84", "A_85"])])
  # label: dist:[A_23,A_24,A_25]-[A_84,A_85]
  ```

- Residues are named by **number** (`50`) or, where a number is ambiguous across
  chains, by **`CHAIN_RESID` label** (`"A_50"`). A plain number that matches more
  than one chain raises an error naming the options, so you never measure the
  wrong residue by accident.
- Points use each residue's **Cα by default** (`atom="CA"`); pass `atom="CB"`
  (etc.) to use another atom.

`geometric_features` returns a *feature extractor*, the same kind of object the
built-in features are, so it drops straight into the comparison.

## Compare two conditions

```python
from confdelta import Ensemble, EnsembleGroup, compare_ensemble_groups

wt  = EnsembleGroup([Ensemble.from_pdb_models("wt_r1.pdb"),  Ensemble.from_pdb_models("wt_r2.pdb")],  label="wt")
mut = EnsembleGroup([Ensemble.from_pdb_models("mut_r1.pdb"), Ensemble.from_pdb_models("mut_r2.pdb")], label="mut")

report = compare_ensemble_groups(wt, mut, feature=cvs)   # feature can be a name OR a callable

for f in report.ranked_by_effect():
    print(f"{f.feature:<22} {f.mean_a:6.1f} -> {f.mean_b:6.1f}  "
          f"g={f.effect_size:+.2f}  CI[{f.effect_ci.low:+.2f},{f.effect_ci.high:+.2f}]  q={f.qvalue:.2g}")
```

Each CV appears in the report labelled `dist:A_25-A_50` or `angle:A_48-A_49-A_50`,
with its point estimate per condition (`mean_a`, `mean_b`), effect size, CI and
corrected q-value — exactly like a per-residue feature.

The **mode is chosen from the data** (as always): two or more replicates per
condition → a replicate-level permutation test; a single run per condition → a
block bootstrap over frames, carrying the single-run caveat. Supplying `rng=0`
makes the resampling reproducible.

## Worked example: HIV-1 protease flap coordinates

The [cantilever flagship](../examples/hiv1_protease_cantilever/) tracks the
mechanism with three CVs straight from the paper:

```python
cvs = geometric_features(
    distances=[("A_25", "A_50"), ("B_25", "B_50")],   # flap opening D25–I50, per monomer
    angles=[("A_48", "A_49", "A_50"), ("B_48", "B_49", "B_50"),   # flap-tip angle (vertex 49)
            ("A_66", "A_67", "A_68"), ("B_66", "B_67", "B_68")],  # cantilever-tip angle (vertex 67)
)
```

The **flap-opening distance** D25–I50 has an established interpretation for HIV-1
protease: **closed < 17 Å, semi-open 17–21 Å, open > 21 Å**. You can read that off
the per-frame values directly:

```python
labels, values = cvs(Ensemble.from_pdb_models("wt_r1.pdb"))
opening = values[:, labels.index("dist:A_25-A_50")]
closed  = (opening < 17).mean()
semi    = ((opening >= 17) & (opening <= 21)).mean()
```

> **A note on sampling.** On these 100–250 ns trajectories the flap coordinates
> are **not converged** — flap opening is a slow, rare event, so the CVs vary as
> much between replicates of one condition as between conditions, and the
> cross-condition effects (flap tips curling in the cross-linked constructs) come
> out correctly-signed but **not significant** (q ≈ 0.6, underpowered at three
> replicates). See the flagship's
> [sampling caveat](../examples/hiv1_protease_cantilever/README.md) and
> `DECISIONS.md` (D-039). The feature is exact; the trajectories are the limit.
