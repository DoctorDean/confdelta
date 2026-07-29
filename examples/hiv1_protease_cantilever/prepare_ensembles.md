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

This is how the committed `wt_ensemble.pdb` and `ds_ensemble.pdb` were made. For
each condition (wild-type and the A71C/Q92C construct), subsample the production
trajectory to CA atoms and ~100 frames and save it as a multi-model PDB. With
GROMACS:

```bash
# CA atoms, every 100th frame of a 100 ns / 10 ps-per-frame run -> 101 models.
# -pbc nojump keeps the dimer spatially continuous: the two chains are separate
# molecules, so -pbc mol can leave them on opposite sides of the box when the
# complex straddles a boundary, which corrupts inter-chain distances. nojump
# unwraps relative to the (whole) first frame and needs no image guessing.
printf "C-alpha\n" | gmx trjconv -s md.tpr -f md.xtc \
            -o wt_ensemble.pdb -pbc nojump -skip 100
```

Because the RMSF feature superposes each frame itself, absolute position and slow
box drift do not matter — only that each frame is a whole, continuous dimer,
which `-pbc nojump` guarantees. Verify by checking the minimum inter-chain CA
distance stays small across frames (a jump to tens of angstroms means a chain
wrapped and the fix did not take).

With MDAnalysis instead (avoids the interactive prompt), unwrap on the whole
trajectory before writing:

```python
import MDAnalysis as mda
from MDAnalysis import transformations as trans

u = mda.Universe("md.tpr", "md.xtc")
u.trajectory.add_transformations(trans.unwrap(u.atoms))
ca = u.select_atoms("name CA")
with mda.Writer("wt_ensemble.pdb", ca.n_atoms, multiframe=True) as w:
    for ts in u.trajectory[::100]:
        w.write(ca)
```

Repeat for the disulfide construct (`ds_ensemble.pdb`). If you have several
independent runs per condition, write one file per run
(`wt_rep1.pdb`, `wt_rep2.pdb`, ...) and pass them all to `reproduce.py`; that
enables replicate-mode inference (a permutation test over replicate-level RMSF),
which is stronger than the single-run block bootstrap.

CA-only at ~100 frames for a 198-residue dimer is ~1.5 MB per file. Fewer frames
shrink it further; the localisation is read from the largest effects and is
stable down to ~50 frames.

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
