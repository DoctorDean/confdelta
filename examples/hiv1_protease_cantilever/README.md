# Flagship: the HIV-1 protease cantilever

This worked example reproduces a published finding with confdelta and turns it
into a regression test: if a future change alters the answer, the test fails. It
runs from **committed data** — two CA-only ensembles in this directory — so it
needs no download and reproduces with one command.

> ⚠️ **Sampling caveat — read before quoting these numbers.** The committed
> result is a **single 100 ns run per condition** (bootstrap mode), and a
> two-replicate check shows the headline effect is **not yet converged**. A
> second, independent 100 ns replicate of A71C/Q92C gives flap-tip RMSF ≈ 2.9 Å —
> as mobile as wild-type — versus 1.5 Å in the committed run; the cantilever
> ΔRMSF even changes sign between replicates. The rigid scaffold reproduces
> almost perfectly, but the flap/cantilever *mobility* differences do not at this
> sampling: the flap response is slow and allosteric, and 100 ns undersamples it.
> Treat the numbers below as a **demonstration of the workflow — and of
> confdelta's single-run caveat doing its job** — not as a replicate-confirmed
> result. A robust claim needs replicate mode (Path A) with ≥4–5 replicates per
> condition and longer trajectories; 250 ns replicates are being run for exactly
> this. See [DECISIONS.md](../../DECISIONS.md) (D-039).

## The question

Sherry et al. identified a cryptic pocket on the *cantilever* of HIV-1 protease,
and showed by MD that **immobilising the cantilever with a disulfide cross-link
makes the flap tips curl in and the enzyme favour a semi-open conformation**
([PubMed 39109919](https://pubmed.ncbi.nlm.nih.gov/39109919/)). The construct
here is the paper's **A71C/Q92C** cross-link: residue 71 sits inside the
cantilever (62–78) and 92 anchors it to the C-terminal strand, so the disulfide
clamps the cantilever. The *response* it provokes — where the protein's mobility
changes — is what we test for.

That is a statement about *where* the dynamics change: at the cantilever and
flaps, not uniformly across the enzyme. Because the finding is about *mobility*,
confdelta compares **per-residue RMSF** (Cα root-mean-square fluctuation): the
residues whose flexibility changes most should be the cantilever and the flaps.

## HIV-1 protease regions

Standard 99-residue-per-chain numbering (homodimer, chains A and B):

| Region | Residues |
|---|---|
| fulcrum | 11–22 |
| elbow | 35–42 |
| flap | 43–58 (tips 48–52) |
| cantilever | 62–78 |
| catalytic | 25–27 |
| cross-link sites | 71, 92 (A71C/Q92C) |

Defined in [regions.py](regions.py); the flap and cantilever are the regions the
finding predicts.

## The data

`wt_ensemble.pdb` and `ds_ensemble.pdb` are CA-only, 101-frame subsamples of
100 ns all-atom MD trajectories (OPLS-AA/L, SPC/E, 0.15 M NaCl) of the wild-type
and A71C/Q92C dimers. See [prepare_ensembles.md](prepare_ensembles.md) for how
they were made from the production trajectories, including the PBC handling.

## Run it

```bash
python reproduce.py \
    --wt        wt_ensemble.pdb \
    --disulfide ds_ensemble.pdb \
    --outdir    results/
```

It writes `results/per_residue_statistics.csv` (per-residue effect size,
confidence interval, FDR-corrected q-value, and the structural region each
residue belongs to) and `results/per_residue_rmsf.png` (RMSF per residue for
both conditions, flap and cantilever bands shaded, cross-link sites marked).

## What confirms the finding

With one run per condition the comparison is in **bootstrap mode**: uncertainty
comes from a block bootstrap over frames, which cannot separate the condition
effect from run-to-run variation (the result says so). It also over-powers
"significance" — most residues clear q ≤ 0.05 — so the localisation is read from
the *largest* effects, not the count of significant ones:

```
Wild-type vs cantilever-disulfide (A71C/Q92C)  [rmsf, bootstrap mode]
residues tested:                 198
significant (q <= 0.05):           99
largest 10 effects in flap/cantilever: 9/10 (90%); largest single effect is in the cantilever

Largest effects:
  * B_63     cantilever  cohens_d=+1.68 q=0.00162  RMSF 1.54->0.70 A
  * B_71     cantilever  cohens_d=+1.66 q=0.00162  RMSF 1.78->0.69 A
  * B_72     cantilever  cohens_d=+1.54 q=0.00162  RMSF 1.68->0.68 A
  * B_64     cantilever  cohens_d=+1.39 q=0.00162  RMSF 1.31->0.67 A
  * B_62     cantilever  cohens_d=+1.38 q=0.00162  RMSF 1.33->0.67 A
```

The cross-link site itself (residue 71) is immobilised from RMSF 1.78 Å to
0.69 Å, and the ten largest mobility changes are 90% in the flap/cantilever
regions — the quantitative echo of the paper's flap-curling result, now as an
effect size with a confidence interval and a corrected q-value per residue. The
regression test ([test_flagship.py](test_flagship.py)) asserts it.

## Specificity: a different cross-link, different dynamics

The finding is about *the cantilever* gating the flaps — not about disulfide
cross-linking in general. `contrast.py` makes that point with a second construct,
**G16C/L38C**, which clamps the fulcrum (16) to the elbow (38) — the flap's hinge
— instead of the cantilever:

```bash
python contrast.py   # writes results/cross_link_contrast.png
```

Mean Cα RMSF by region, each construct against wild type:

| Region | WT | A71C/Q92C Δ | G16C/L38C Δ |
|---|---|---|---|
| flap (43–58) | 1.57 | −0.63 | −0.25 |
| **flap tips (48–52)** | **2.93** | **−1.42** | **+0.01** |
| cantilever (62–78) | 1.19 | −0.29 | −0.30 |

Both cross-links rigidify the cantilever similarly, but only the **cantilever**
cross-link (A71C/Q92C) quiets the flap tips (2.93 → 1.50 Å); the fulcrum–elbow
cross-link leaves them fully mobile (2.93 → 2.94 Å). Same enzyme, same method,
two disulfides — two different dynamical signatures, which confdelta resolves.
`test_contrast.py` guards it.

This is a descriptive, region-level companion to the rigorous per-residue
comparison above. One honesty note: because both comparisons share the single
wild-type run, the *similar* cantilever reduction is the part least separable
from run-to-run variation; the **flap-tip contrast** — where the two constructs
plainly disagree — is the robust signal.

## Making it stronger

A single 100 ns run per condition is the honest weak point (hence the block
bootstrap caveat, and a chain-to-chain asymmetry in a homodimer that more
sampling would even out). Independent replicates per condition would move this
to **replicate mode** — a permutation test over replicate-level RMSF, the
statistically clean case. The harness already supports it: pass several files
per side (`--wt wt_rep1.pdb wt_rep2.pdb ...`), see
[prepare_ensembles.md](prepare_ensembles.md).

## Provenance

The finding, the disulfide construct (A71C/Q92C), and the region definitions are
from the maintainer's own published work. The ensembles are prepared from MD
simulations of that system; see [prepare_ensembles.md](prepare_ensembles.md).
