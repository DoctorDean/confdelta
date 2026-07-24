"""Configuration-file loading for the confdelta command line.

The CLI exposes only the handful of options a typical run needs. Everything
else lives in a JSON config file loaded by :func:`load_config`, which keeps the
command line short without hiding any capability.

The loader is deliberately strict. Research tooling usually accepts an unknown
key in silence and produces a result computed with defaults the user did not
intend; here an unrecognised key is an error, and the message names the closest
valid option.

Example
-------
>>> from confdelta.config import load_config
>>> analysis, comparison = load_config("study.json")  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import difflib
import json
from pathlib import Path
from typing import Any

from .core import AnalysisConfig
from .differential import DifferentialConfig

__all__ = [
    "ConfigError",
    "load_config",
    "example_config",
    "write_example_config",
]

# Top-level sections a config file may contain.
_ANALYSIS_SECTION = "analysis"
_COMPARISON_SECTION = "comparison"
_SECTIONS = (_ANALYSIS_SECTION, _COMPARISON_SECTION)


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or contains unknown keys.

    Carries a message intended to be shown directly to the user, so it should
    name the offending key and, where possible, suggest the intended one.
    """


def _field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _field_types(cls: type) -> dict[str, Any]:
    return {f.name: f.type for f in dataclasses.fields(cls)}


def _suggest(key: str, valid: set[str]) -> str:
    """Return a ' Did you mean ...?' fragment, or an empty string."""
    matches = difflib.get_close_matches(key, sorted(valid), n=1, cutoff=0.6)
    return f" Did you mean {matches[0]!r}?" if matches else ""


def _check_keys(section: str, provided: dict[str, Any], cls: type) -> None:
    valid = _field_names(cls)
    unknown = [key for key in provided if key not in valid]
    if not unknown:
        return
    lines = [f"Unknown option(s) in the {section!r} section of the config file:"]
    for key in sorted(unknown):
        lines.append(f"  - {key!r}.{_suggest(key, valid)}")
    lines.append(f"Valid {section!r} options are: {', '.join(sorted(valid))}")
    raise ConfigError("\n".join(lines))


def _coerce_section(section: str, provided: dict[str, Any], cls: type) -> dict[str, Any]:
    """Validate *provided* against the dataclass *cls* and return it unchanged.

    Type coercion is left to the dataclass; this checks only that every key is
    recognised, which is the failure mode that actually bites users.
    """
    if not isinstance(provided, dict):
        raise ConfigError(
            f"The {section!r} section of the config file must be a JSON object, "
            f"got {type(provided).__name__}."
        )
    _check_keys(section, provided, cls)
    return dict(provided)


def load_config(path: str | Path) -> tuple[AnalysisConfig, DifferentialConfig]:
    """Load analysis and comparison configuration from a JSON file.

    Parameters
    ----------
    path
        Path to a JSON config file. See :func:`example_config` for the shape,
        or generate one with ``confdelta example-config``.

    Returns
    -------
    (AnalysisConfig, DifferentialConfig)
        Both are fully populated; any option the file omits keeps its default.

    Raises
    ------
    ConfigError
        If the file is missing, is not valid JSON, is not a JSON object, has an
        unrecognised top-level section, or sets an unrecognised option.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Config file {config_path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config file {config_path} must contain a JSON object at the top level, "
            f"got {type(raw).__name__}."
        )

    unknown_sections = [key for key in raw if key not in _SECTIONS]
    if unknown_sections:
        lines = [f"Unknown section(s) in {config_path}:"]
        for key in sorted(unknown_sections):
            lines.append(f"  - {key!r}.{_suggest(key, set(_SECTIONS))}")
        lines.append(f"Valid sections are: {', '.join(_SECTIONS)}")
        raise ConfigError("\n".join(lines))

    analysis_kwargs = _coerce_section(
        _ANALYSIS_SECTION, raw.get(_ANALYSIS_SECTION, {}), AnalysisConfig
    )
    comparison_kwargs = _coerce_section(
        _COMPARISON_SECTION, raw.get(_COMPARISON_SECTION, {}), DifferentialConfig
    )

    return AnalysisConfig(**analysis_kwargs), DifferentialConfig(**comparison_kwargs)


def example_config() -> dict[str, Any]:
    """Return a config dict populated with every option at its default value.

    Generated from the dataclasses rather than hand-maintained, so it cannot
    drift out of sync with the code or advertise options that do not exist.
    """
    return {
        _ANALYSIS_SECTION: {
            name: value
            for name, value in dataclasses.asdict(AnalysisConfig()).items()
            if value is not None
        },
        _COMPARISON_SECTION: dataclasses.asdict(DifferentialConfig()),
    }


def write_example_config(path: str | Path) -> Path:
    """Write a fully populated example config to *path* and return the path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(example_config(), indent=2) + "\n")
    return output
