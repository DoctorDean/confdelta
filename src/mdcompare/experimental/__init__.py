"""Experimental MD-Compare features.

Modules here are under active development; their APIs may change between
minor releases. They are intentionally kept out of the top-level
``mdcompare`` namespace so that importing the package never pulls in heavy
or optional machine-learning dependencies.

Currently provided
------------------
resistance_classifier
    Feature engineering and supervised classification of drug-resistance
    phenotypes from MD-derived descriptors (network, PCA, pocket and
    thermodynamic features).
"""

from __future__ import annotations

__all__ = ["resistance_classifier"]
