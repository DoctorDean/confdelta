# Preparing the ensemble files

`reproduce.py` reads each condition as one or more **CA-only multi-model PDB**
files. Those are small enough to commit to the repository (a few hundred KB per
ensemble), so the flagship reproduces from committed data with no download.

There are two routes. The **MD route** is the faithful one — it is the data the
paper's finding came from. The **ColabFold route** is a lighter-weight
alternative that also demonstrates confdelta's source-agnostic input, but it
predicts structures from sequence and does not model the disulfide's dynamical
effect, so treat it as a showcase rather than the validation.

## MD route (faithful reproduction)

On the workstation, for each condition (wild-type and the cantilever-disulfide
construct), subsample the production trajectory to CA atoms and ~50–100 frames
and save it as a multi-model PDB. With GROMACS:

```bash
# 1. Strip to CA atoms and take every Nth frame (tune -skip so ~50-100 frames
#    remain), writing a multi-model PDB. Answer "C-alpha" (or make an index
#    group) at the selection prompt.
gmx trjconv -s topol.tpr -f traj.xtc \
            -o wt_ensemble.pdb -pbc mol -ur compact \
            -skip 50 <<< "C-alpha"
```

or equivalently with MDAnalysis, which avoids the interactive prompt:

```python
import MDAnalysis as mda

u = mda.Universe("topol.tpr", "traj.xtc")
ca = u.select_atoms("name CA")
with mda.Writer("wt_ensemble.pdb", ca.n_atoms, multiframe=True) as w:
    for ts in u.trajectory[::50]:      # every 50th frame
        w.write(ca)
```

Repeat for the disulfide construct (`ds_ensemble.pdb`). If you have several
independent runs per condition, write one file per run
(`wt_rep1.pdb`, `wt_rep2.pdb`, ...) and pass them all to `reproduce.py`; that
enables replicate-mode inference, which is stronger than the single-run block
bootstrap.

Aim for a committed size under ~1 MB per file. CA-only at 50–100 frames for a
198-residue dimer is comfortably within that.

## ColabFold route (source-agnostic showcase)

On the workstation's ColabFold venv, generate a conformational ensemble by
subsampling the MSA (shallow MSAs make AlphaFold sample more conformational
diversity):

```bash
colabfold_batch --num-seeds 20 --max-msa 16:32 \
                wt_sequence.fasta   wt_colabfold/
colabfold_batch --num-seeds 20 --max-msa 16:32 \
                ds_sequence.fasta   ds_colabfold/
```

Then stack the per-seed models into one multi-model PDB per condition (or pass
the individual PDBs to `Ensemble.from_structures`). Note the disulfide
construct's sequence differs from wild-type only by the engineered cysteines,
which ColabFold will not cross-link, so this route illustrates the input
flexibility more than the dynamics.

## Committing

Place the final `*_ensemble.pdb` (or `*_repN.pdb`) files in this directory and
commit them. Then run `reproduce.py`, record the headline numbers in
[README.md](README.md), and fill in the expected values in
[test_flagship.py](test_flagship.py) so CI guards them.
