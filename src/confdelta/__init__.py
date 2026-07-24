#!/usr/bin/env python3
"""
confdelta: statistical comparison of protein conformational ensembles.

Compares two conformational ensembles across networks, dynamics, energetics
and kinetics. Single-ensemble analysis (residue interaction networks, DCCM,
PCA, free energy landscapes, Markov state models) is provided as the layer
the comparison is built on.

confdelta continues MD-Compare, which was previously versioned to 1.5.0. The
version was reset to 0.1.0 at the rename; see CHANGELOG.md.

Author: Dr Dean Sherry
License: MIT

Example
-------
>>> from confdelta import NetworkAnalyzer, AnalysisConfig
>>> config = AnalysisConfig(compute_msm=True, msm_lag_time=10)
>>> analyzer = NetworkAnalyzer(config)
"""

from __future__ import annotations

from ._version import __version__

__author__ = "Dr Dean Sherry"
__email__ = ""
__license__ = "MIT"
__url__ = "https://github.com/DoctorDean/confdelta"

# Version info tuple for programmatic access
__version_info__ = tuple(int(v) for v in __version__.split(".") if v.isdigit())

# Import main classes for convenient top-level access.
#
# These imports are deliberately NOT wrapped in a try/except. An earlier
# version swallowed ImportError here and silently dropped ten names from
# __all__, so a downstream `from confdelta import DifferentialAnalyzer` failed
# with a bare ImportError naming only the symbol, giving no clue which
# dependency was missing or which extra to install. A missing core dependency
# is a broken install, and it should say so at import time.
from .core import (
    AnalysisConfig,
    MDComparator,
    MDCompare,
    MDSimulation,
    NetworkAnalyzer,
    NetworkMetrics,
    OutputManager,
    SimulationConfig,
)
from .differential import (
    DifferentialAnalyzer,
    DifferentialConfig,
)

__all__ = [
    "NetworkAnalyzer",
    "AnalysisConfig",
    "NetworkMetrics",
    "MDSimulation",
    "SimulationConfig",
    "MDComparator",
    "OutputManager",
    "MDCompare",
    "DifferentialAnalyzer",
    "DifferentialConfig",
    "check_dependencies",
    "get_version_info",
    "FEATURES",
    "__version__",
    "__version_info__",
]


# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------


def _detect(module_name: str) -> bool:
    """Return True if *module_name* can be imported."""
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


FEATURES = {
    "deeptime": _detect("deeptime"),
    "igraph": _detect("igraph"),
    "leidenalg": _detect("leidenalg"),
    "pandas": _detect("pandas"),
}


def check_dependencies(verbose: bool = False) -> dict:
    """Check availability of optional dependencies.

    Parameters
    ----------
    verbose : bool
        If True, print a detailed dependency report.

    Returns
    -------
    dict
        Mapping of optional feature name to availability (bool).
    """
    if verbose:
        print(f"confdelta v{__version__}")
        print("=" * 40)
        print("Core dependencies:")
        core_deps = [
            "MDAnalysis",
            "networkx",
            "numpy",
            "scipy",
            "matplotlib",
            "seaborn",
            "sklearn",
        ]
        for dep in core_deps:
            mark = "OK " if _detect(dep) else "-- "
            print(f"  [{mark}] {dep}")

        print("\nOptional dependencies:")
        for feature, available in FEATURES.items():
            mark = "OK " if available else "-- "
            print(f"  [{mark}] {feature}")

        if FEATURES["deeptime"]:
            print("    -> Markov State Model analysis available")
        if FEATURES["igraph"] and FEATURES["leidenalg"]:
            print("    -> Leiden community detection available")
        if FEATURES["pandas"]:
            print("    -> Excel/CSV export available")

    return dict(FEATURES)


def get_version_info() -> dict:
    """Return version and key dependency information as a dict."""
    import sys

    info = {
        "confdelta_version": __version__,
        "python_version": sys.version,
        "platform": sys.platform,
        "features": dict(FEATURES),
    }

    for name, attr in (("MDAnalysis", "mdanalysis_version"), ("networkx", "networkx_version")):
        try:
            mod = __import__(name)
            info[attr] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[attr] = "not available"

    return info
