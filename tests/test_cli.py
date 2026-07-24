"""Tests for the confdelta command-line interface.

The CLI previously had 4% test coverage across 5 subcommands and 192 flag
instances. These tests pin the collapsed surface and, in particular, the
input validation: research tooling typically fails deep inside a stack trace,
and the point of the validation layer is to fail at the boundary instead.
"""

from __future__ import annotations

import json

import pytest

from confdelta import cli


@pytest.fixture
def files(tmp_path):
    """Two readable stand-in topology/trajectory files."""
    paths = {}
    for name in ("wt.pdb", "wt.xtc", "mut.pdb", "mut.xtc"):
        path = tmp_path / name
        path.write_bytes(b"placeholder")
        paths[name] = str(path)
    return paths


class TestParserSurface:
    def test_only_three_subcommands_exist(self):
        parser = cli.build_parser()
        subparsers = [
            action
            for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        ][0]
        assert set(subparsers.choices) == {"compare", "single", "example-config"}

    @pytest.mark.parametrize("removed", ["diff", "differential"])
    def test_removed_subcommands_are_gone(self, removed, capsys):
        """`diff` and `differential` were overlapping and are no longer accepted."""
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([removed, "-h"])

    def test_compare_has_few_flags(self):
        """The common case must need well under ten options."""
        parser = cli.build_parser()
        subparsers = [
            action
            for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        ][0]
        compare = subparsers.choices["compare"]
        flags = [a for a in compare._actions if a.dest != "help"]
        assert len(flags) < 10, f"compare exposes {len(flags)} options"

    def test_no_subcommand_prints_help_and_fails(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["confdelta"])
        assert cli.main() == 1
        assert "usage:" in capsys.readouterr().out


class TestFileValidation:
    def test_missing_topology_is_reported_by_name(self, monkeypatch, files, capsys, tmp_path):
        monkeypatch.setattr(
            "sys.argv",
            # fmt: off
            [
                "confdelta", "compare",
                "-a", str(tmp_path / "absent.pdb"), files["wt.xtc"],
                "-b", files["mut.pdb"], files["mut.xtc"],
            ],
            # fmt: on
        )
        assert cli.main() == 2
        err = capsys.readouterr().err
        assert "not found" in err
        assert "absent.pdb" in err

    def test_missing_trajectory_is_reported(self, monkeypatch, files, capsys, tmp_path):
        monkeypatch.setattr(
            "sys.argv",
            # fmt: off
            [
                "confdelta", "compare",
                "-a", files["wt.pdb"], str(tmp_path / "absent.xtc"),
                "-b", files["mut.pdb"], files["mut.xtc"],
            ],
            # fmt: on
        )
        assert cli.main() == 2
        assert "absent.xtc" in capsys.readouterr().err

    def test_directory_given_instead_of_file(self, monkeypatch, files, capsys, tmp_path):
        monkeypatch.setattr(
            "sys.argv",
            # fmt: off
            [
                "confdelta", "compare",
                "-a", str(tmp_path), files["wt.xtc"],
                "-b", files["mut.pdb"], files["mut.xtc"],
            ],
            # fmt: on
        )
        assert cli.main() == 2
        assert "is a directory" in capsys.readouterr().err

    def test_identical_condition_names_rejected(self, monkeypatch, files, capsys):
        monkeypatch.setattr(
            "sys.argv",
            # fmt: off
            [
                "confdelta", "compare",
                "-a", files["wt.pdb"], files["wt.xtc"], "--a-name", "same",
                "-b", files["mut.pdb"], files["mut.xtc"], "--b-name", "same",
            ],
            # fmt: on
        )
        assert cli.main() == 2
        assert "must differ" in capsys.readouterr().err

    def test_validation_runs_before_any_analysis(self, monkeypatch, files, capsys, tmp_path):
        """A bad path must fail before the trajectory is ever opened."""
        called = []
        monkeypatch.setattr(
            cli.DifferentialAnalyzer,
            "__init__",
            lambda self, *a, **k: called.append(1),
        )
        monkeypatch.setattr(
            "sys.argv",
            # fmt: off
            [
                "confdelta", "compare",
                "-a", str(tmp_path / "absent.pdb"), files["wt.xtc"],
                "-b", files["mut.pdb"], files["mut.xtc"],
            ],
            # fmt: on
        )
        assert cli.main() == 2
        assert called == [], "analysis was constructed despite invalid input"


class TestReplicateHandling:
    def test_multiple_trajectories_are_rejected_not_pooled(self, files, tmp_path):
        """Concatenating replicates would misrepresent them as one run."""
        extra = tmp_path / "wt2.xtc"
        extra.write_bytes(b"placeholder")
        with pytest.raises(cli.InputError) as exc:
            cli._validate_trajectories([files["wt.xtc"], str(extra)], "A")
        message = str(exc.value)
        assert "Replicate-aware" in message
        assert "concatenat" in message.lower()

    def test_single_trajectory_passes_through(self, files):
        assert cli._validate_trajectories([files["wt.xtc"]], "A") == files["wt.xtc"]


class TestConfigIntegration:
    def test_bad_config_key_surfaces_as_a_clean_error(self, monkeypatch, files, tmp_path, capsys):
        config = tmp_path / "bad.json"
        config.write_text(json.dumps({"analysis": {"msm_lag_tim": 5}}))
        monkeypatch.setattr(
            "sys.argv",
            # fmt: off
            [
                "confdelta", "compare",
                "-a", files["wt.pdb"], files["wt.xtc"],
                "-b", files["mut.pdb"], files["mut.xtc"],
                "--config", str(config),
            ],
            # fmt: on
        )
        assert cli.main() == 2
        err = capsys.readouterr().err
        assert "msm_lag_time" in err  # suggests the intended key
        assert "Traceback" not in err

    def test_example_config_writes_a_loadable_file(self, monkeypatch, tmp_path, capsys):
        out = tmp_path / "study.json"
        monkeypatch.setattr("sys.argv", ["confdelta", "example-config", "-o", str(out)])
        assert cli.main() == 0
        assert out.is_file()

        from confdelta.config import load_config

        analysis, comparison = load_config(out)
        assert analysis is not None and comparison is not None


class TestOutputHonesty:
    def test_descriptive_only_notice_mentions_the_missing_machinery(self):
        notice = cli._DESCRIPTIVE_ONLY_NOTICE
        assert "descriptive" in notice
        for missing in ("Significance", "effect sizes", "multiple-testing correction"):
            assert missing in notice

    def test_no_executive_summary_is_printed(self):
        """The fabricated executive summary must not return."""
        source = __import__("inspect").getsource(cli)
        assert "EXECUTIVE SUMMARY" not in source
        assert "executive_summary" not in source


class TestDocumentedCommandsAreReal:
    """Every confdelta command shown in the docs must parse.

    The five example documents this replaces used 68 distinct flags, every one
    of which had ceased to exist. These tests make that a test failure rather
    than something a reader discovers.
    """

    @staticmethod
    def _documented_commands():
        """Extract `confdelta ...` invocations from fenced bash blocks only.

        Restricted to fenced blocks so prose that happens to begin with the
        word "confdelta" is not mistaken for a command.
        """
        import pathlib
        import re
        import shlex

        root = pathlib.Path(__file__).resolve().parents[1]
        sources = [root / "README.md", *(root / "docs").rglob("*.md")]
        commands = []
        for path in sources:
            if not path.is_file():
                continue
            for block in re.findall(r"```(?:bash|sh|console)\n(.*?)```", path.read_text(), re.S):
                # Join shell line-continuations, then drop trailing comments.
                block = re.sub(r"\\\s*\n\s*", " ", block)
                for line in block.splitlines():
                    stripped = line.split("#", 1)[0].strip()
                    if not stripped.startswith("confdelta "):
                        continue
                    try:
                        commands.append((path.name, shlex.split(stripped)[1:]))
                    except ValueError:
                        continue
        return commands

    def test_some_commands_were_found(self):
        assert self._documented_commands(), "no documented commands found to check"

    def test_every_documented_command_parses(self, tmp_path, monkeypatch):
        """Parse each documented command; unknown flags or subcommands exit."""
        parser = cli.build_parser()
        monkeypatch.chdir(tmp_path)

        failures = []
        for source, argv in self._documented_commands():
            if not argv or argv[0].startswith("-"):
                continue
            try:
                parser.parse_args(argv)
            except SystemExit:
                failures.append(f"{source}: confdelta {' '.join(argv)}")
        assert failures == [], "documented commands that do not parse:\n" + "\n".join(failures)
