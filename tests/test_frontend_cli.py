from pathlib import Path

from fpgai.cli import build_parser, main


def test_cli_has_frontend_not_model_workload_command():
    parser = build_parser()
    help_text = parser.format_help()
    assert "frontend" in help_text
    assert "workload" not in help_text


def test_frontend_list_command(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["fpgai", "frontend", "list"])
    main()
    out = capsys.readouterr().out
    assert "onnx" in out
    assert "stablehlo" in out
    assert "yolo" not in out.lower()


def test_frontend_routes_command(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["fpgai", "frontend", "routes"])
    main()
    out = capsys.readouterr().out
    assert "jax" in out
    assert "tensorflow" in out
    assert "pytorch" in out
    assert "onnx" in out
    assert "requires_upstream_legalization" in out
