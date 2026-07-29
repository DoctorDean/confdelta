# Design decisions

A running log of every non-obvious judgement call, with its reasoning. This is the design
rationale document: it exists so that the next contributor — or the next agent session — can
pick the project up cold and understand *why* the code looks the way it does.

Newest entries at the bottom of each phase. Each entry records the decision, the reasoning,
and what would have to change for the decision to be revisited.

---

## Phase 0 — Audit

### D-001 · The differential framework is rebuilt, not deleted

**Decision.** The v1.5.0 differential framework is classified BROKEN, but rather than
deleting it wholesale (the brief's default for BROKEN subsystems), its *descriptive* layer is
preserved and its *inferential* layer is deleted and rewritten.

**Reasoning.** The audit (`AUDIT.md` §5.4) found that the comparators genuinely compute
difference matrices, centrality deltas, eigenvector overlaps, timescale ratios, pathway
disruption and hotspot rank changes. That work is real and correct. What is fabricated is
everything layered on top: p-values, similarity scores, effect sizes and the executive
summary. Deleting the whole module would discard working code; keeping it intact would ship
fabrication. The split is along the descriptive/inferential line.

**Revisit if.** A comparator's descriptive output turns out to be wrong too, in which case
that individual comparator is deleted rather than rewired.

### D-002 · Package name is `confdelta`

**Decision.** Rename to `confdelta`. Maintainer decision at the Phase 0 checkpoint.

**Reasoning.** The brief's recommendation, `mdelta`, is **taken on PyPI** (a dormant
single-cell lineage-tree aligner). `confdiff`, the brief's second choice, collides with
`bytedance/ConfDiff` (86★, ICML'24) — a protein *conformation generation* model, i.e. the
same domain, and a producer of exactly the ensembles this library is meant to consume.
`parallax` is taken on PyPI. A cross-cutting finding drove the choice away from the whole
`*diff` family: in 2026 a `diff` suffix in a structural-biology name reads as **diffusion
model**, not *difference*. `confdelta` is free on PyPI, effectively clean on GitHub, names
both the object (conformational ensembles) and the operation (Δ), and carries no diffusion
ambiguity.

**Revisit if.** Never — a PyPI name is permanent once published. This decision is locked at
first publish, not before.

### D-003 · Version resets to 0.1.0

**Decision.** Reset from the claimed 1.5.0 to `0.1.0`; record the lineage in `CHANGELOG.md`.

**Reasoning.** 1.5.0 at 23 commits with an unimplemented headline feature signals version
inflation to any experienced reader. The rename legitimately justifies starting fresh. 1.0
should not be claimed until the public API is stable and defensible in public.

### D-004 · Statistics lead with effect sizes; p-values are secondary

**Decision.** Every comparator reports point estimate, confidence interval and effect size as
primary output; raw p and corrected q are reported alongside but are not the headline, and
are never the sole basis for a claim in the output.

**Reasoning.** This falls directly out of a measured constraint (`AUDIT.md` §12.2). An exact
two-sided permutation test on 3 replicates per condition has a **minimum attainable p-value of
0.10** — it cannot reach α = 0.05 no matter how large the true effect, let alone survive FDR
correction across hundreds of residues. Most published MD studies run exactly three
replicates. A library that leads with p-values would therefore report "nothing significant"
for the majority of legitimate real-world comparisons, which is both unhelpful and
misleading. Effect sizes with CIs remain informative at n = 3.

**Revisit if.** The field's replicate norms change substantially.

### D-005 · Unit of replication: trajectory when replicates exist, block bootstrap otherwise

**Decision.** Support both. When the user supplies *n* replicate trajectories per condition,
the replicate is the unit of inference. When only one run per condition is available, fall
back to a moving/circular block bootstrap over frames, with a mandatory caveat stamped into
the result object and printed. Maintainer decision at the Phase 0 checkpoint.

**Reasoning.** Requiring replicates is the most statistically defensible position, but would
reject the majority of real-world use including most published wild-type-vs-mutant studies.
Accepting single runs without qualification is pseudoreplication — frames are autocorrelated,
and naive per-frame testing is precisely the error the field makes in hand-rolled notebook
analyses. Supporting both, with the weaker path clearly labelled as such, covers the field as
it actually is without pretending the two paths are equivalent. Two single trajectories
cannot separate "the mutation changed it" from "these two runs differ", and the output must
say so.

**Consequence.** The input model must carry *n* ensembles per condition, so the
source-agnostic `Ensemble` / `EnsembleGroup` abstraction becomes a **prerequisite** of the
statistical core rather than a later refactor.

### D-006 · No `statsmodels` runtime dependency

**Decision.** Use `scipy.stats.false_discovery_control` for Benjamini–Hochberg and
Benjamini–Yekutieli correction, and `scipy.stats.permutation_test` / `scipy.stats.bootstrap`
for resampling. Implement Bonferroni directly. Cross-validate against
`statsmodels.multipletests` in the dev test suite only.

**Reasoning.** SciPy is already a core dependency and, as of 1.11, ships verified
implementations of everything needed except block resampling (`AUDIT.md` §12.1). Adding
statsmodels would be a heavy runtime dependency for functionality already present. Keeping it
as a *test-only* dependency preserves the independent cross-check without the install cost.

**Consequence.** `scipy>=1.11` becomes a hard floor, which in turn requires dropping Python
3.8/3.9 — so the dependency modernisation must happen in Phase 1, before the statistics.

### D-007 · Block bootstrap is the only new numerical primitive

**Decision.** Write a moving/circular block bootstrap with automatic block-length selection.
Everything else reuses SciPy.

**Reasoning.** SciPy's `bootstrap` has no block or time-series resampling mode (verified: no
`block`-related parameter in its signature), and IID resampling of autocorrelated frames
understates uncertainty. This is a small, well-specified, testable component. Its acceptance
test is **measured CI coverage on synthetic AR(1) data with known autocorrelation** — a block
bootstrap that fails to achieve nominal coverage is a failing test, not a passing one.

### D-008 · Phase order changed from the brief

**Decision.** Revised order: audit → rename + dependencies → amputate fabrication → ensemble
input model → statistical core → wire in + typed results → README → flagship example →
publish.

**Reasoning.** Two audit findings forced the reorder. First, the README rewrite cannot
honestly make the claim that defines the positioning until the statistics exist, so it moves
after the build rather than before. Second, D-005 means the input model must carry replicates,
so the source-agnostic `Ensemble` abstraction the brief scheduled for its Phase 6 becomes a
prerequisite of the statistical core. Dependency modernisation moves early because D-006
requires a modern SciPy floor.

### D-009 · `compare_centrality_distributions` is reimplemented, not moved

**Decision.** Delete `utils.py:488 compare_centrality_distributions`; reimplement its intent
in the new statistics module with a paired test.

**Reasoning.** It contains the only real statistical calls in the current package, but it
applies `mannwhitneyu` — an independent-samples test — to node-matched paired data. The
correct test for values matched by node is Wilcoxon signed-rank. The function is currently
wired to nothing, so no output is presently wrong, but the method is. Moving it would carry
the error forward.

### D-010 · Silent fallbacks that manufacture data are removed, not fixed

**Decision.** Delete the `SimplifiedMSM` fallback (`core.py:1267-1329`) and its invented
implied timescales `[10.0, 5.0, 2.0]`. Raise a diagnosable error instead.

**Reasoning.** A fallback that silently substitutes fabricated kinetic data for a failed MSM
construction is worse than a crash: the user gets numbers that look like results. The same
principle applies to the import guard in `__init__.py` that silently shrinks `__all__` — fail
loudly and name the missing extra.

---

## Phase 1 — Rename, identity, dependency modernisation

### D-011 · Class names keep their `MD` prefix for now

**Decision.** The rename changed the package, module, distribution and console
script names, but **not** the class names: `MDCompare`, `MDComparator` and
`MDSimulation` are unchanged.

**Reasoning.** Renaming classes is a separate API break from renaming the
package, and these classes are being restructured in Phase 5 anyway —
`MDSimulation` is the file-path-locked type that `Ensemble` replaces. Renaming
them now would mean breaking downstream imports twice, and would add churn to a
diff that is already large. The changelog records that they are provisional.

**Revisit at.** Phase 5, when the public API is defined.

### D-012 · The B905 sweep inserts `strict=False`, not `strict=True`

**Decision.** Raising the lint target to Python 3.10 flagged 14 `zip()` calls
without an explicit `strict=`. All 14 were fixed with `strict=False`.

**Reasoning.** `strict=False` is exactly the current behaviour — zip truncates
to the shortest input — so the sweep is provably behaviour-preserving.
`strict=True` would be more informative at several of these sites (for example
`zip(bars, values)` in plotting code, where the two are structurally the same
length), but it would raise where the code previously truncated silently. The
brief requires asking the maintainer before any change that could alter the
numerical output of an existing analysis, and Phase 1 is explicitly a rename
and dependency phase. Making these strict is a behaviour change and belongs
with the comparator rewrite.

**Revisit at.** Phase 5, per call site.

### D-013 · mypy is enforced immediately, with a shrinking debt list

**Decision.** mypy runs in CI as a blocking gate from 0.1.0. The 93
pre-existing type errors are quarantined behind an explicit
`[[tool.mypy.overrides]]` list naming four modules (`core`, `differential`,
`cli`, `experimental.resistance_classifier`). Every module *not* on that list
is checked and must stay clean.

**Reasoning.** The alternatives were to make mypy non-blocking (which means it
is ignored) or to fix 93 errors across 5 868 lines of `core.py` now (which is
the Phase 5 job, not Phase 1). An explicit, named allowlist makes the debt
visible and bounded, and — crucially — means every *new* module is type-checked
from birth. `ensemble.py` and `stats.py` will never be able to join the list.

**Constraint.** The list must only ever shrink. Adding a module to it is a
regression, not a fix.

### D-014 · No `python_version` pin in the mypy configuration

**Decision.** The mypy config deliberately omits `python_version`.

**Reasoning.** Pinning it to 3.10 makes mypy parse *third-party* stubs with
3.10 grammar. numpy 2.5's stubs use `type` statements (3.12+), so mypy failed
with a syntax error inside `numpy/__init__.pyi` before checking any confdelta
code at all. Without the pin, mypy uses the interpreter it runs under.
Version-specific behaviour across 3.10-3.13 is covered by the pytest matrix,
which is the more direct check anyway.

### D-015 · `docs/INSTALL.md` was rewritten rather than edited

**Decision.** Replaced the 369-line installation guide with a 90-line one.

**Reasoning.** The original was built end-to-end on premises that PyEMMA's
removal invalidated: conda as the recommended path, a `numpy<2.0` pin,
per-platform PyEMMA build workarounds, and a troubleshooting section for
PyEMMA install failures and NumPy 2.0 incompatibility. Editing around that
would have left a document whose structure still implied installation is hard.
It is not: `pip install confdelta` is the whole thing.

---

## Amputating the fabricated output

### D-016 · The five example documents were deleted, not updated

**Decision.** Removed all five worked examples in `docs/examples/`
(~2,500 lines) and replaced them with a single accurate page.

**Reasoning.** After the CLI collapse, **68 of the 68 flags** those documents
used had ceased to exist, and three of the five invoked subcommands (`diff`,
`differential`) that no longer exist. Not a single command in any of them could
be run. Updating them would have meant rewriting every command block in
documents whose structure was organised around option groups that no longer
exist — for example a "Statistical Options" section documenting flags that
never existed in the first place.

One of them, `02_differential_analysis.md`, additionally presented invented
numbers as results, with scientific interpretation attached, and told readers
to open four CSV files no code path writes.

This goes further than the approved scope, which called for stripping the
invented tables from that one file. The narrower change was not defensible once
it was clear every command in every file was broken: shipping documentation
that cannot be followed is the failure mode this project is removing.

**Mitigation.** The prose is in git history, the replacement page says so
explicitly, and a validated worked example is a planned deliverable that will
replace the page properly.

### D-017 · Dead configuration options were removed rather than left as stubs

**Decision.** Deleted nine `DifferentialConfig` options that no code path read,
including `multiple_comparison_correction`, `bootstrap_iterations` and
`permutation_iterations`.

**Reasoning.** A configuration surface is a promise. A user could set
`multiple_comparison_correction="bonferroni"`, see it accepted without
complaint, and reasonably conclude their results were corrected. Nothing read
it. Leaving these in place as forward-looking stubs would preserve exactly the
misleading signal the audit identified. They return when the statistical core
lands, and a parametrised test asserts each stays absent until then — passing
one now raises `TypeError`.

### D-018 · Multiple trajectories per condition are rejected, not concatenated

**Decision.** The `compare` command carries trajectories internally as lists,
which is the shape replicate-aware comparison needs, but supplying more than
one per condition raises an error explaining why.

**Reasoning.** MDAnalysis will happily concatenate a list of trajectories into
one continuous Universe. Doing that to independent replicates destroys the
replicate structure and presents *n* runs as a single long trajectory — the
precise pseudoreplication error the statistical core is being built to prevent
(see D-005). Accepting the flag shape now avoids a second breaking change
later; rejecting the plural case avoids silently doing the wrong thing in the
meantime. Users who genuinely want frames pooled are told to concatenate the
files themselves, so the choice is explicit and recorded.

### D-019 · Documentation accuracy is enforced by tests, not by discipline

**Decision.** Three test groups now fail when documentation drifts from code:
every option in `docs/CONFIGURATION.md` must exist and every existing option
must be documented; every `confdelta` command in `README.md` and `docs/` must
parse against the real argument parser.

**Reasoning.** The audit's single most embarrassing finding was that 61 of 90
documented CLI flags did not exist. That is not a mistake anyone makes
deliberately — it is what happens when documentation and code drift for 23
commits with nothing checking. Prose review does not catch it; a test does.

---

## The ensemble input model

### D-020 · The internal representation of every ensemble is an MDAnalysis Universe

**Decision.** All four `Ensemble` constructors normalise to a single internal
form: an MDAnalysis `Universe` carrying N frames, possibly in memory
(`MemoryReader`). Source-specific handling lives only in the constructors.

**Reasoning.** The verified-correct analysis pipeline (contact maps, DCCM, PCA,
landscapes, MSM) already consumes a `Universe`. Making `Universe` the common
representation means every source — trajectory, multi-model PDB, structure set,
coordinate array — reaches that pipeline unchanged, so the new input breadth
costs nothing in the analysis code and inherits its correctness. The
alternative, a bespoke ensemble representation with its own analysis paths,
would have doubled the surface to verify.

### D-021 · `from_coordinates` requires a topology

**Decision.** Building an ensemble from a raw coordinate array requires a
topology (a structure file or an existing Universe), not coordinates alone.

**Reasoning.** confdelta's analyses are residue-level: contact networks,
per-residue centrality, allosteric paths between named residues. None of that
is defined on an anonymous point cloud — you need to know which atom is which
residue of which chain. A topology supplies that. Crucially a topology is a
*structure*, not a trajectory, so requiring one does not reintroduce the
trajectory-file dependency the ensemble model exists to remove. Generative and
predicted ensembles always have an associated topology (the sequence they were
generated for), so this costs their use case nothing.

### D-022 · Bridge to MDSimulation rather than rewrite the pipeline now

**Decision.** `Ensemble.to_simulation()` produces an `MDSimulation` by
injecting the already-loaded universe, and the analysis pipeline consumes that.
`MDSimulation` is not yet removed.

**Reasoning.** `MDSimulation` is threaded through thousands of lines of
analysis and output code. Replacing it wholesale in the same change that
introduces `Ensemble` would make an already-large diff unreviewable and would
mix a new abstraction with a risky refactor. The bridge lets `Ensemble` become
the public input type immediately while `MDSimulation` is retired
incrementally behind it. The injection path is the same one the test fixtures
already use, so it is well exercised.

**Revisit at.** When the comparators are rewritten to return typed structured
results (the next phase), `MDSimulation` can be collapsed into `Ensemble`.

### D-023 · Replicates are accepted now, used later

**Decision.** `EnsembleGroup` holds N replicates today, but
`run_ensemble_comparison` analyses only the first and logs a warning when there
are more.

**Reasoning.** The replicate-aware *statistics* (permutation test over
replicate-level values, D-005) are the next body of work. But the input model
must carry replicates now, because changing the shape of the comparison API
later — from "one ensemble per condition" to "a group per condition" — would be
a second breaking change for every caller. Accepting the group shape now and
filling in the statistics behind it avoids that. Using only the first replicate
in the meantime is honest (it warns) and is no worse than the current
single-run behaviour.

---

## The statistical core

### D-024 · The block bootstrap's acceptance bar is near-nominal coverage, not exactly 0.95

**Decision.** The block-bootstrap acceptance test asserts that measured 95%
confidence-interval coverage on autocorrelated AR(1) data is *near-nominal*
(≥ 0.85 at N=1000, φ=0.8) and *dramatically better than the IID bootstrap*
(which undercovers to ~0.51), rather than asserting coverage ≥ 0.95.

**Reasoning.** Reaching exactly 0.95 coverage at moderate effective sample size
is not achievable by any method, not a shortcoming of this implementation. It
was measured directly during design: at N=1000, φ=0.8 (effective sample size
~100), even the effective-sample-size *t*-interval — the gold standard for the
mean of a dependent series — reaches only ~0.90, and the block bootstrap
percentile interval ~0.89. The undercoverage comes from estimating a long-run
variance from a short dependent series and shrinks as N_eff grows. Writing a
test that demanded 0.95 would either fail forever or force a dishonest tolerance
fudge. The honest, informative acceptance criteria are the two that actually
matter: the interval is close to nominal, and it fixes most of the gap the IID
bootstrap leaves open. Both are asserted; the docstring states the limitation
plainly so no caller over-trusts a narrow interval.

**Revisit if.** A calibrated interval (e.g. bootstrap-t with a variance-
stabilising transform, or a subsampling scheme) is added; then the bar can rise.

### D-025 · The block bootstrap is the general CI tool, despite mean-only alternatives being tighter

**Decision.** Confidence intervals come from the block bootstrap, not from an
effective-sample-size *t*-interval, even though the latter has marginally better
coverage for the mean specifically.

**Reasoning.** The ESS *t*-interval only works for the mean. The comparison
layer needs intervals for arbitrary statistics — correlation changes, centrality
deltas, modularity, effect sizes themselves — and the block bootstrap handles
all of them with one mechanism. A per-statistic patchwork of analytic intervals
would be more code to verify and would still leave gaps. The small coverage
edge for the mean does not justify special-casing it. `effective_sample_size`
is still exposed, because reporting N_eff to the user is independently valuable.

### D-026 · `min_attainable_pvalue` is a first-class function, not an internal check

**Decision.** The smallest p-value an exact permutation test can return for a
given pair of group sizes is a public function, and `PermutationResult` exposes
an `underpowered` property derived from it.

**Reasoning.** This is the single most important honest behaviour the library
can offer for the small designs the field actually runs (D-004). A tool that
silently reports "p = 0.10, not significant" for a three-versus-three comparison
with an enormous effect is misleading: the data are not the limiting factor, the
design is. Surfacing the floor lets the comparison layer say "this design cannot
reach significance regardless of effect size; read the effect size instead"
rather than implying the effect is absent. Making it public also lets downstream
tools check power before committing compute to a comparison.

---

## Wiring the statistics into the comparison

### D-027 · The first shipped feature is per-residue contact number

**Decision.** The statistical comparison operates, for now, on one per-residue
feature: contact number (how many other residues have their centroid within a
cutoff, per frame). The engine is feature-agnostic; more features are a registry
entry away.

**Reasoning.** The comparison engine needed at least one real, defensible
feature to be wired and demonstrated, and contact number is the best starting
choice: it is computed from internal distances only, so it needs no structural
alignment (alignment would be a correctness risk and a dependency in its own
right); it produces one scalar per residue, so a protein yields hundreds of
simultaneous tests and multiple-testing correction genuinely matters; and it is
a standard proxy for local packing that a mutation perturbs, so it exercises the
whole path on a scientifically meaningful quantity. RMSF and per-pair DCCM
changes are the obvious next features, but each needs alignment or a larger test
matrix, so they are deferred rather than rushed. `IDEAS.md` records them.

### D-028 · Statistics use all replicates; the descriptive views use the first

**Decision.** `run_ensemble_comparison` runs the statistical comparison over
every replicate of both conditions, but the legacy descriptive comparators
(network, DCCM, energetics, kinetics, allosteric) still run on the first
replicate of each condition only. The replicate warning states this split.

**Reasoning.** The statistical layer is the one that must be rigorous, and it is
new, so it was built to consume the full `EnsembleGroup` from the start. The
descriptive comparators are inherited MD-Compare code that operates on a single
`MDSimulation`; making them aggregate across replicates is a separate,
larger refactor with its own correctness questions (how to average a DCCM matrix
across runs, how to combine community structures). Rather than block the
statistical wiring on that, or silently use only one replicate for everything,
the split is made explicit in the warning. The descriptive views are clearly
secondary in the output and carry no inferential claim, so using one replicate
for them is honest as long as it is stated.

**Revisit at.** When the descriptive comparators are rewritten to return typed
results (a later phase), they can aggregate across replicates at the same time.

### D-029 · A failed statistical comparison degrades to None, not an abort

**Decision.** If `compare_ensemble_groups` raises inside
`run_ensemble_comparison` (an unsupported feature, mismatched systems), the
error is logged and `statistical_comparison` is set to None; the descriptive
analysis still completes.

**Reasoning.** The descriptive comparison is independently useful and may be all
some inputs support (e.g. two ensembles of genuinely different systems that a
user still wants a rough network diff for). Aborting the whole run because the
inferential layer could not be computed would throw away work the user can use.
The None is explicit and the CLI reports "not available", so the degradation is
visible, not silent.

---

## The flagship example

### D-030 · Flagship: HIV-1 protease cantilever disulfide, from PubMed 39109919

**Decision.** The flagship reproduces the finding of the maintainer's cryptic-
cantilever-pocket paper: immobilising the cantilever with a disulfide cross-link
makes the flap tips curl in and the protease favour a semi-open conformation.
The reproduction compares a wild-type ensemble against the disulfide construct
and checks that the significant per-residue changes concentrate in the flap
(43-58) and cantilever (59-75) regions.

**Reasoning.** It is the maintainer's own published work (a Phase 7 requirement),
and the finding is *localised* -- a statement about which residues change -- which
is exactly what confdelta's per-residue, FDR-corrected comparison is built to
express. A diffuse finding would not test the tool's value; a localised one does.
It also exercises the source-agnostic input model, since the ensembles can come
from MD trajectories or from ColabFold.

### D-031 · The regression test asserts the localisation, not a magic number

**Decision.** `test_flagship.py` asserts the *qualitative published claim* -- a
majority of significant residues fall in the flap/cantilever regions, and the
largest effect is in one of them -- rather than pinning an exact effect size or
count.

**Reasoning.** The scientific finding is "the change is localised to the flaps
and cantilever", so that is what the regression guards. Pinning an exact Hedges'
g would make the test brittle to trivial changes in the resampling seed or frame
subsampling without testing anything more meaningful. The localisation is both
the real finding and a stable target. The maintainer can tighten the thresholds
against the real run once the data is committed.

### D-032 · Flagship ships with the harness now, the data later

**Decision.** The reproduction script, region map, narrative and test are
committed now; the ensemble PDBs are added once generated on the workstation.
The test skips until the data is present, so CI stays green in the meantime, and
the README states plainly that it reports no numbers yet.

**Reasoning.** The trajectory data lives on a remote workstation, not the
development machine, so the reproduction cannot be completed in one sitting.
Committing the harness now lets the data step be a clean drop-in (generate two
small CA-only multi-model PDBs, commit, fill in the numbers) rather than a
from-scratch build later. Crucially, no fabricated numbers are committed in the
interim -- the whole project exists to remove those. The synthetic smoke test
proves the harness is correct without standing in for the result.

---

## Interface persistence (first downstream capability)

### D-033 · A different system for interface persistence: not HIV-1 protease

**Decision.** Interface persistence is validated on a nanobody-target complex,
not the HIV-1 protease flagship system.

**Reasoning.** HIV-1 protease is an obligate homodimer. Its dimer interface is
constitutive -- always present -- so contact persistence is trivially ~100% and
the metric measures nothing interesting. A nanobody-target complex has a real,
non-obligate binding interface where persistence and pre-organisation genuinely
vary between designs, which is the use case the capability exists for. The
feature is unit-tested now against synthetic two-chain complexes; the nanobody
simulation is the validation, to be run on the workstation.

### D-034 · Both contact definitions; comparison over a reference set

**Decision.** ``interface_persistence`` offers both a heavy-atom (4.5 A, default)
and a centroid (8 A) contact definition. ``compare_interface_persistence``
compares two designs over a caller-supplied *reference* contact set (the
designed interface), not the union of what each design happens to form.

**Reasoning.** Maintainer decisions. Heavy-atom distance is the standard,
chemically faithful interface definition and is the right default; the centroid
option is cheaper and alignment-free for quick or coarse work, and is consistent
with the existing contact-number feature. For the design-triage question -- "does
this binder hold the contacts it was designed to make?" -- the fixed reference
set is the right frame: a design that fails to form a designed contact must score
zero persistence on it, which a union-of-observed set would instead silently drop.

### D-035 · Interface RMSF uses a hand-rolled Kabsch superposition

**Decision.** ``interface_rmsf`` aligns frames with a small numpy Kabsch
implementation rather than MDAnalysis's ``AlignTraj`` / ``rms.RMSF``.

**Reasoning.** ``AlignTraj(in_memory=True)`` mutates the universe's coordinates
in place, which would corrupt the caller's ensemble, and ``rms.RMSF`` emits a
deprecation warning (and mis-worded guidance) in the installed MDAnalysis. A
dozen lines of Kabsch superposition are dependency-stable, do not mutate the
input, and are exactly testable -- a rigid-body-translated complex must show
~zero internal RMSF, which is asserted.

### D-036 · A per-residue mobility (RMSF) feature, alongside contact number

**Decision.** Add `per_residue_rmsf` to `confdelta.features` (registry name
`"rmsf"`): each frame is Kabsch-superposed onto the ensemble mean, and the
per-frame squared displacement of each residue from its mean position is
returned. Its per-frame mean is the residue's MSF, so `sqrt(mean_a)` /
`sqrt(mean_b)` in a report recover the two RMSF profiles.

**Reasoning.** The contact-number feature is a packing proxy; it cannot express
a *flexibility* change, which is what a great many ensemble comparisons — and
the flagship finding — are actually about. Framing mobility as a *per-frame*
quantity (squared displacement) lets it reuse the existing engine unchanged: the
same block bootstrap / permutation test that compares contact numbers compares
mobility. The Kabsch superposition is the one already written for
`interface_rmsf`; it was factored into a private `_align.superpose_to_mean` so
both callers share one implementation rather than duplicating it.

### D-037 · Flagship reproduces on A71C/Q92C with RMSF; localisation read from the largest effects

**Decision.** The flagship compares wild-type vs the **A71C/Q92C** cantilever
disulfide (the maintainer's paper construct, correcting the earlier G16C/L38C
placeholder in D-030) using the `rmsf` feature, on committed 101-frame CA
ensembles from 100 ns MD. Success is asserted on the **largest** effects — a
majority of the top-10 per-residue mobility changes fall in the flap/cantilever,
and the single largest is a cantilever rigidification — not on the fraction of
all significant residues.

**Reasoning.** Two things surfaced when the real trajectories arrived, and both
were verified before wiring anything (no fabricated pass). First, the shipped
contact-number feature sees the cross-link's effect as a *global* semi-open
rearrangement (cantilever, flaps, catalytic and elbow all shift), so the changes
do not localise to flap/cantilever by contact number; per-residue RMSF, being a
direct mobility measure, does — the ten largest RMSF changes are 90% in the
flap/cantilever and the cross-link site drops from 1.78 to 0.69 A. Second, a
single 100 ns run makes the block bootstrap over-power "significance" (99/198
residues clear q<=0.05), so "fraction of significant residues in a region" just
tracks region size and is the wrong operationalisation; the localisation of the
*largest, most reliable* effects is robust to that and is the faithful echo of
the paper's claim. Replicates would upgrade this to the replicate-mode
permutation test (D-005); the harness already accepts them.

### D-038 · A second cross-link as a contrast panel, not just a second data point

**Decision.** Ship the G16C/L38C construct in the flagship as a *contrast* to
A71C/Q92C (`contrast.py`, `test_contrast.py`, a committed
`ds_g16c_l38c_ensemble.pdb`), reporting region-level RMSF for both against the
wild type rather than a second full per-residue statistical run.

**Reasoning.** The published claim is specific — it is *the cantilever* that
gates the flaps — and a lone reproduction cannot show specificity. The two
cross-links make the point cleanly: A71C/Q92C (cantilever) drops flap-tip RMSF
from 2.93 to 1.50 A, while G16C/L38C (fulcrum-elbow) leaves it at 2.94; both
rigidify the cantilever similarly. The contrast lives at the region level
because that is where the effect is legible and the test is deterministic; the
rigorous per-residue effect-size/CI/q-value comparison remains `reproduce.py`.
Honesty note carried in the README and the test design: both comparisons share
the one wild-type run, so the *shared* cantilever reduction is the least certain
part and the *flap-tip divergence* is the robust, construct-specific signal.
