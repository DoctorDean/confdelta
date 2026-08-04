# How-to: custom features

confdelta ships a handful of features (`contacts`, `rmsf`, geometric CVs), but the
comparison engine is feature-agnostic: **any per-frame quantity you can compute
from coordinates can be compared** with the same effect sizes, confidence
intervals and FDR-corrected q-values. Two routes, from easiest to most flexible.

## RMSD to a reference structure

`per_residue_rmsd` measures how far each residue sits from a structure *you*
choose — a closed state, a crystal structure, another condition's mean — rather
than from the ensemble's own mean (which is what `rmsf` does):

```python
from confdelta import Ensemble, EnsembleGroup, compare_ensemble_groups, per_residue_rmsd

closed = Ensemble.from_pdb_models("closed_state.pdb")   # the reference
rmsd_to_closed = per_residue_rmsd(closed)               # a feature extractor

report = compare_ensemble_groups(apo, holo, feature=rmsd_to_closed)
```

Each frame is Kabsch-superposed onto the reference (on all atoms by default, or
`align_selection="..."` for a rigid core), and each residue's squared displacement
from the reference is returned; `sqrt(mean_a)` recovers the per-residue RMSD
profile. The reference must describe the same residues as the ensembles.

## Bring your own per-frame observable

`feature_from_positions` wraps any function of a frame's coordinates into a
feature. Your function receives the selected atoms' positions `(n_atoms, 3)` and
returns one value per label:

```python
import numpy as np
from confdelta import compare_ensemble_groups, feature_from_positions

def radius_of_gyration(pos):
    centred = pos - pos.mean(axis=0)
    return np.array([np.sqrt((centred**2).sum(axis=1).mean())])

rg = feature_from_positions(radius_of_gyration, labels=["Rg"], selection="name CA")
report = compare_ensemble_groups(wild_type, mutant, feature=rg)
```

Return several values at once by giving several labels (the function must return
that many). `selection` restricts which atoms are passed (default: the whole
ensemble selection).

## The extractor protocol

Both of the above return a **`FeatureExtractor`**: a callable
`Ensemble -> (labels, values)` where `values` has shape `(n_frames, n_features)`.
`compare_ensemble_groups(feature=...)` accepts any such callable, so if you need
full control you can write one directly and still get the whole statistical
pipeline for free. The mode (replicate permutation test vs single-run block
bootstrap) is chosen from how many replicates each condition has, exactly as for
the built-in features.
