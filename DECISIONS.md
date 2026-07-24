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
