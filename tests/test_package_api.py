"""Tests for the confdelta package's import-time contract.

These pin behaviour that previously regressed silently: a bare
``except ImportError`` in ``confdelta/__init__.py`` dropped ten names from
``__all__`` when any core dependency failed to import, and ``core.py`` called
``sys.exit(1)`` at import time when MDAnalysis or networkx were missing.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import confdelta


class TestPublicSurface:
    def test_all_exported_names_actually_exist(self):
        """Every name in __all__ must be resolvable on the package."""
        missing = [name for name in confdelta.__all__ if not hasattr(confdelta, name)]
        assert missing == [], f"__all__ advertises names that do not exist: {missing}"

    def test_public_classes_are_exported(self):
        """The documented entry points must be importable from the top level."""
        for name in (
            "NetworkAnalyzer",
            "AnalysisConfig",
            "MDSimulation",
            "SimulationConfig",
            "MDCompare",
            "DifferentialAnalyzer",
            "DifferentialConfig",
        ):
            assert name in confdelta.__all__
            assert hasattr(confdelta, name)

    def test_all_is_not_silently_truncated(self):
        """__all__ must never shrink to the dependency-free fallback set.

        The removed fallback exported only five names, so a partially broken
        install presented a package that imported cleanly but was missing every
        analysis class -- with no error explaining why.
        """
        assert len(confdelta.__all__) > 5
        assert "DifferentialAnalyzer" in confdelta.__all__

    def test_version_is_single_sourced(self):
        from confdelta import _version

        assert confdelta.__version__ == _version.__version__
        assert confdelta.__version_info__[0] == int(confdelta.__version__.split(".")[0])


class TestImportSideEffects:
    def test_importing_the_package_does_not_exit_the_process(self):
        """`import confdelta` must never terminate the interpreter.

        core.py previously called sys.exit(1) at module scope when MDAnalysis
        or networkx could not be imported, which kills the host process of any
        tool that imports confdelta.
        """
        result = subprocess.run(
            [sys.executable, "-c", "import confdelta; print('ok')"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_no_module_calls_sys_exit_at_import_time(self):
        """Guard against reintroducing import-time process termination."""
        import pathlib

        pkg = pathlib.Path(confdelta.__file__).parent
        offenders = []
        for path in pkg.rglob("*.py"):
            source = path.read_text()
            for lineno, line in enumerate(source.splitlines(), 1):
                stripped = line.strip()
                # Module scope only: no leading indentation.
                if stripped.startswith("sys.exit(") and not line.startswith((" ", "\t")):
                    offenders.append(f"{path.name}:{lineno}")
        assert offenders == [], f"sys.exit() at module scope in: {offenders}"

    @pytest.mark.parametrize(
        "module",
        [
            "confdelta.core",
            "confdelta.differential",
            "confdelta.cli",
            "confdelta.utils",
            "confdelta.msm_backends",
        ],
    )
    def test_submodules_import(self, module):
        assert importlib.import_module(module) is not None
