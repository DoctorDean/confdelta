#!/usr/bin/env python3

"""
Regenerate ``requirements.txt`` from ``pyproject.toml``.

``requirements.txt`` is a convenience mirror of the canonical dependency
lists in ``pyproject.toml``. To stop the two from drifting, this script is
the single source of truth: it reads ``[project.dependencies]`` and the
``[project.optional-dependencies]`` extras and writes a fully generated,
clearly-marked ``requirements.txt``.

Usage
-----
    python scripts/generate_requirements.py            # write requirements.txt
    python scripts/generate_requirements.py --check     # CI: fail if stale
    python scripts/generate_requirements.py --stdout    # print, don't write

The ``--check`` mode regenerates in memory and compares against the file on
disk, exiting non-zero if they differ -- suitable for a CI guard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.8-3.10
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        print(
            "error: need 'tomllib' (Python 3.11+) or 'tomli'. " "Install with: pip install tomli",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

HEADER = """\
# =====================================================================
# AUTO-GENERATED FILE -- DO NOT EDIT BY HAND.
#
# Regenerate with:  python scripts/generate_requirements.py
# Source of truth:  pyproject.toml  ([project.dependencies] and extras)
#
# This file mirrors the packaging metadata for convenience
# (`pip install -r requirements.txt`). For a proper install prefer:
#     pip install .            (core)
#     pip install .[all]       (core + optional features)
#     pip install .[msm]       (core + Markov State Models)
# =====================================================================
"""


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def render_requirements(data: dict) -> str:
    """Build the requirements.txt contents from parsed pyproject data."""
    project = data.get("project", {})
    core = list(project.get("dependencies", []))
    extras = project.get("optional-dependencies", {})

    lines = [HEADER.rstrip(), ""]
    lines.append("# --- core runtime dependencies -----------------------------------")
    lines.extend(core)
    lines.append("")

    lines.append("# --- optional features -------------------------------------------")
    lines.append("# Install these via extras rather than uncommenting, e.g.:")
    # Show the extras and their install command; list contents as comments so
    # a plain `pip install -r requirements.txt` stays core-only.
    for name in sorted(extras):
        lines.append(f"#   pip install .[{name}]")
        for dep in extras[name]:
            lines.append(f"#       {dep}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if requirements.txt is out of date (no write)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print generated content to stdout instead of writing the file",
    )
    args = parser.parse_args(argv)

    if not PYPROJECT.exists():
        print(f"error: {PYPROJECT} not found", file=sys.stderr)
        return 2

    content = render_requirements(_load_pyproject())

    if args.stdout:
        sys.stdout.write(content)
        return 0

    if args.check:
        current = REQUIREMENTS.read_text() if REQUIREMENTS.exists() else ""
        if current != content:
            print(
                "requirements.txt is out of date with pyproject.toml.\n"
                "Run: python scripts/generate_requirements.py",
                file=sys.stderr,
            )
            return 1
        print("requirements.txt is up to date.")
        return 0

    REQUIREMENTS.write_text(content)
    print(f"Wrote {REQUIREMENTS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
