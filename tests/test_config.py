"""Tests for confdelta.config — CLI configuration-file loading."""

from __future__ import annotations

import json

import pytest

from confdelta.config import (
    ConfigError,
    example_config,
    load_config,
    write_example_config,
)


def _write(tmp_path, payload, name="cfg.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return path


class TestLoadConfig:
    def test_empty_config_gives_defaults(self, tmp_path):
        from confdelta.core import AnalysisConfig

        analysis, _ = load_config(_write(tmp_path, {}))
        assert analysis.threshold == AnalysisConfig().threshold
        assert analysis.community_method == AnalysisConfig().community_method

    def test_values_are_applied(self, tmp_path):
        analysis, comparison = load_config(
            _write(
                tmp_path,
                {
                    "analysis": {"threshold": 0.33, "pca_components": 7},
                    "comparison": {"figure_dpi": 150},
                },
            )
        )
        assert analysis.threshold == pytest.approx(0.33)
        assert analysis.pca_components == 7
        assert comparison.figure_dpi == 150

    def test_omitted_options_keep_defaults(self, tmp_path):
        from confdelta.core import AnalysisConfig

        analysis, _ = load_config(_write(tmp_path, {"analysis": {"threshold": 0.4}}))
        assert analysis.threshold == pytest.approx(0.4)
        assert analysis.msm_lag_time == AnalysisConfig().msm_lag_time


class TestStrictness:
    """Unknown keys must fail loudly rather than silently using a default."""

    def test_unknown_analysis_option_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="Unknown option"):
            load_config(_write(tmp_path, {"analysis": {"not_a_real_option": 1}}))

    def test_unknown_option_names_the_offender(self, tmp_path):
        with pytest.raises(ConfigError) as exc:
            load_config(_write(tmp_path, {"analysis": {"not_a_real_option": 1}}))
        assert "not_a_real_option" in str(exc.value)

    def test_near_miss_suggests_the_intended_option(self, tmp_path):
        """A typo should point at the option the user meant."""
        with pytest.raises(ConfigError) as exc:
            load_config(_write(tmp_path, {"analysis": {"msm_lag_tim": 5}}))
        assert "msm_lag_time" in str(exc.value)
        assert "Did you mean" in str(exc.value)

    def test_unknown_section_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="Unknown section"):
            load_config(_write(tmp_path, {"analsyis": {}}))

    def test_unknown_section_suggests_the_intended_one(self, tmp_path):
        with pytest.raises(ConfigError) as exc:
            load_config(_write(tmp_path, {"analsyis": {}}))
        assert "analysis" in str(exc.value)

    def test_comparison_options_are_checked_too(self, tmp_path):
        with pytest.raises(ConfigError, match="Unknown option"):
            load_config(_write(tmp_path, {"comparison": {"bogus": True}}))


class TestErrorMessages:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.json")

    def test_malformed_json_reports_position(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"analysis": {"threshold": }}')
        with pytest.raises(ConfigError) as exc:
            load_config(path)
        message = str(exc.value)
        assert "not valid JSON" in message
        assert "line" in message and "column" in message

    def test_top_level_must_be_an_object(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ConfigError, match="JSON object at the top level"):
            load_config(path)

    def test_section_must_be_an_object(self, tmp_path):
        with pytest.raises(ConfigError, match="must be a JSON object"):
            load_config(_write(tmp_path, {"analysis": [1, 2]}))


class TestExampleConfig:
    def test_example_has_both_sections(self):
        cfg = example_config()
        assert set(cfg) == {"analysis", "comparison"}

    def test_example_round_trips_through_the_loader(self, tmp_path):
        """The generated example must itself be a valid config.

        This is what stops the template drifting out of sync with the
        dataclasses and advertising options that do not exist.
        """
        path = write_example_config(tmp_path / "example.json")
        analysis, comparison = load_config(path)
        assert analysis is not None
        assert comparison is not None

    def test_example_matches_dataclass_defaults(self, tmp_path):
        from confdelta.core import AnalysisConfig

        path = write_example_config(tmp_path / "example.json")
        analysis, _ = load_config(path)
        assert analysis.threshold == AnalysisConfig().threshold
        assert analysis.msm_n_clusters == AnalysisConfig().msm_n_clusters

    def test_write_creates_parent_directories(self, tmp_path):
        path = write_example_config(tmp_path / "nested" / "dir" / "example.json")
        assert path.is_file()
