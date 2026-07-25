# Flagship: the HIV-1 protease cantilever

This worked example reproduces a published finding with confdelta and turns it
into a regression test: if a future change alters the answer, the test fails.

> **Status: harness complete, awaiting the ensemble data.** The reproduction
> script, region annotation and figure are done and verified on synthetic
> input. The real wild-type and cantilever-disulfide ensembles are generated
> from trajectories on a workstation (see [prepare_ensembles.md](prepare_ensembles.md));
> once they are committed here, the numbers below and the regression test are
> filled in from a real run. This file does **not** report invented numbers.

## The question

Sherry et al. identified a cryptic pocket on the *cantilever* of HIV-1 protease,
and showed by MD that **immobilising the cantilever with a disulfide cross-link
makes the flap tips curl in and the enzyme favour a semi-open conformation**
([PubMed 39109919](https://pubmed.ncbi.nlm.nih.gov/39109919/)).

That is a statement about *where* the dynamics change: at the flaps and the
cantilever, not uniformly across the enzyme. confdelta should be able to say the
same thing quantitatively — comparing a wild-type ensemble against the
disulfide construct, the residues that change significantly should concentrate
in the flap and cantilever regions.

## HIV-1 protease regions

Standard 99-residue-per-chain numbering (homodimer, chains A and B):

| Region | Residues |
|---|---|
| fulcrum | 11–22 |
| elbow | 35–42 |
| flap | 43–58 (tips 48–52) |
| cantilever | 59–75 |
| catalytic | 25–27 |

Defined in [regions.py](regions.py); the flap and cantilever are the regions the
finding predicts.

## Run it

```bash
python reproduce.py \
    --wt        wt_ensemble.pdb \
    --disulfide ds_ensemble.pdb \
    --outdir    results/
```

With replicate ensembles per condition (the statistically clean case), pass
several files per side:

```bash
python reproduce.py \
    --wt        wt_rep1.pdb wt_rep2.pdb wt_rep3.pdb \
    --disulfide ds_rep1.pdb ds_rep2.pdb ds_rep3.pdb \
    --outdir results/
```

It writes `results/per_residue_statistics.csv` (per-residue effect size,
confidence interval, FDR-corrected q-value, and the structural region each
residue belongs to) and `results/per_residue_effect_sizes.png` (effect size per
residue, significant residues highlighted, flap and cantilever bands shaded).

## What confirms the finding

The reproduction succeeds if the significant per-residue changes concentrate in
the flap and cantilever regions rather than scattering across the enzyme. The
script prints exactly that:

```
significant (q <= 0.05):        <n>
  of which in flap/cantilever:  <m>  (<m/n>%)
```

A high fraction in the flap/cantilever regions is the quantitative echo of the
paper's flap-curling result. The regression test
([test_flagship.py](test_flagship.py)) asserts it on the committed ensembles.

## Interpretation

confdelta reports *where* and *how much*, per residue, with the uncertainty and
multiple-testing correction that a by-hand difference of averages omits. The
flap tips (48–52) and the cantilever (59–75) carrying the largest, significant
effects is the same conclusion the paper reached from trajectory inspection —
here as an effect size with a confidence interval and a corrected q-value for
every residue, reproducible with one command.

## Provenance

The finding, the disulfide construct, and the region definitions are from the
maintainer's own published work. The ensembles are prepared from the
simulations described there; see [prepare_ensembles.md](prepare_ensembles.md).
