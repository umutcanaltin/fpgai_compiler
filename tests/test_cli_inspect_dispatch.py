from __future__ import annotations

import pytest

from fpgai import cli


class _Args:
    command = "inspect"
    config = "dummy.yml"
    json_output = "inspection.json"


def test_inspect_subcommand_does_not_route_to_auto_compile(monkeypatch):
    calls: list[tuple[str, str]] = []

    class _Parser:
        def parse_args(self):
            return _Args()

        def print_help(self):  # pragma: no cover
            raise AssertionError("help should not be printed")

    def fake_build_parser():
        return _Parser()

    def fake_run_from_config(config_path: str, *, action: str = "auto") -> int:
        calls.append((config_path, action))
        return 99

    def fake_inspect_from_config(config_path: str, *, json_output: str | None = None) -> int:
        assert config_path == "dummy.yml"
        assert json_output == "inspection.json"
        return 0

    monkeypatch.setattr(cli, "build_parser", fake_build_parser)
    monkeypatch.setattr(cli, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(cli, "inspect_from_config", fake_inspect_from_config)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert calls == []
