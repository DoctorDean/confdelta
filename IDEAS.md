# Ideas parking lot

Scope creep is the failure mode that killed the previous project in this lineage. Anything
that is not on the current phase plan goes here and stays here until the roadmap reaches it.

Nothing in this file is a commitment.

---

## Deferred to the roadmap (already scheduled)

- **Interface persistence** — fraction of frames retaining each designed or observed interface
  contact; interface RMSF; contact-persistence distribution; summary persistence score;
  binder pre-organisation (conformational spread of the smaller partner alone vs bound).
  Scheduled as the first downstream module *after* PyPI publication, as proof that the public
  API works. Deliberately narrow: one metric, clear definition.

## Raised during Phase 0, not scheduled

- **Ensembles from AlphaFold subsampling and generative emulators.** The source-agnostic
  `Ensemble` abstraction is being built to make this possible, but no loader for these sources
  is planned in the current roadmap. Add only when a concrete downstream need exists.

- **Secondary-structure-aware regional analysis.** `DynamicsComparator._analyze_regional_
  correlation_changes` currently splits the DCCM into matrix quadrants as a proxy for protein
  regions, with a comment admitting as much. Real regional analysis needs DSSP or equivalent.
  Worth doing; not now.

- **Graph-neural-network path in the resistance classifier.** `experimental/
  resistance_classifier.py` lazily imports `torch` + `torch_geometric` for a GNN path. It stays
  in `experimental/` and out of the public API. Do not promote without a validation study.

- **Multi-condition comparison (>2 ensembles).** The current design is deliberately two-group.
  Extending to *k* groups changes the statistical model (ANOVA-like, or all-pairs with a wider
  correction family). Real demand exists, but two-group must be correct first.

- **Paired / repeated-measures designs.** E.g. apo vs holo of the same construct across the
  same set of replicate seeds. A paired permutation test is more powerful than the unpaired
  one and is a natural extension once the unpaired path is validated.

- **Automatic block-length selection refinements.** The first implementation will use a
  τ_int-derived rule. Politis–White / Patton–Politis–White automatic selection is the more
  principled option if the simple rule underperforms in coverage testing.

- **Effective-sample-size reporting as a first-class output.** N_eff will be computed for the
  block-bootstrap path. Surfacing it prominently — "your 10 000 frames are worth 43 independent
  samples" — would be genuinely useful to users, and slightly brutal. Consider for the README's
  worked example.

- **Cytoscape / PyMOL session export.** `utils.py` already has
  `export_network_for_cytoscape`. A PyMOL companion that colours residues by effect size would
  be an obvious win for the flagship example. Not before publication.

- **Interactive dashboards.** The `viz` extra pulls in plotly and ipywidgets. Currently
  unexercised. Either build something real with them or drop the extra.
