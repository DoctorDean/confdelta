#!/usr/bin/env python3
"""
MD-Compare: Comprehensive Protein Dynamics Analysis Platform

A toolkit for analyzing and comparing molecular dynamics simulations with
network analysis, conformational dynamics, energy landscapes, and kinetic
modeling capabilities.

Author: Dr Dean Sherry
License: MIT

Example
-------
>>> from mdcompare import NetworkAnalyzer, AnalysisConfig
>>> config = AnalysisConfig(compute_msm=True, msm_lag_time=10)
>>> analyzer = NetworkAnalyzer(config)
"""

from __future__ import annotations

__version__ = "1.5.0"
__author__ = "Dr Dean Sherry"
__email__ = ""
__license__ = "MIT"
__url__ = "https://github.com/DoctorDean/MD-Compare"

# Version info tuple for programmatic access
__version_info__ = tuple(int(v) for v in __version__.split(".") if v.isdigit())

# Import main classes for convenient top-level access.
try:
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
except ImportError:
    # Fallback for partial-install / development scenarios where heavy
    # scientific dependencies may not yet be present.
    __all__ = [
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
    "pyemma": _detect("pyemma"),
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
        print(f"MD-Compare v{__version__}")
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

        if FEATURES["pyemma"] or FEATURES["deeptime"]:
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
        "md_compare_version": __version__,
        "python_version": sys.version,
        "platform": sys.platform,
        "features": dict(FEATURES),
    }

    for name, attr in (("MDAnalysis", "mdanalysis_version"),
                        ("networkx", "networkx_version")):
        try:
            mod = __import__(name)
            info[attr] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[attr] = "not available"

    return info
