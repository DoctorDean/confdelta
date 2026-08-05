"""Single source of truth for the confdelta version.

``pyproject.toml`` reads this attribute via setuptools' dynamic-version
support, so the version is declared exactly once. Modules that need to stamp
the version into output import it from here rather than from the package
``__init__``, which would be circular.
"""

from __future__ import annotations

__version__ = "0.2.0"
