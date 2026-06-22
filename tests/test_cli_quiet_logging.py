from __future__ import annotations

from types import SimpleNamespace

from fpgai import cli


class _FakeCompiler:
    def __init__(self, cfg):
        self.cfg = cfg

    def compile(self):
        print("[vitis_hls][stdout] noisy tool output")
        return SimpleNamespace(
            out_dir="build/fake",
            hls_ran=True,
            hls_ok=True,
            hls_returncode=0,
            summary=lambda: "=============== FPGAI Compile Result ===============\nHLS ran              : True",
        )


def test_compile_default_captures_noisy_output(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(
        pipeline=SimpleNamespace(mode="inference"),
        raw={"benchmark": {"enabled": False}},
    )
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "Compiler", _FakeCompiler)

    rc = cli.run_from_config("config.yml", action="compile")

    captured = capsys.readouterr()
    assert rc == 0
    assert "[vitis_hls][stdout]" not in captured.out
    assert "Captured stdout log" in captured.out
    assert "[OK] Wrote artifacts" in captured.out


def test_compile_verbose_streams_noisy_output(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(
        pipeline=SimpleNamespace(mode="inference"),
        raw={"benchmark": {"enabled": False}},
    )
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "print_summary", lambda cfg: None)
    monkeypatch.setattr(cli, "Compiler", _FakeCompiler)

    rc = cli.run_from_config("config.yml", action="compile", verbose=True)

    captured = capsys.readouterr()
    assert rc == 0
    assert "[vitis_hls][stdout] noisy tool output" in captured.out
    assert "Captured stdout log" not in captured.out


def test_verbose_and_quiet_are_mutually_exclusive(capsys):
    rc = cli.run_from_config(
        "config.yml",
        action="compile",
        verbose=True,
        quiet=True,
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "cannot be used together" in captured.err
