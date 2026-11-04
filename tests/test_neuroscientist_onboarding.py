"""Source-only regression tests for one-command neuroscientist onboarding.

No real AMD programs or board hardware are invoked by these tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fpgai_research.toolchains import onboarding as app


def _args(tmp_path: Path, *, stage: str = "ooc", native: bool = False):
    import argparse
    return argparse.Namespace(
        onnx=str(tmp_path / "brain.onnx"), output=str(tmp_path / "new-case"),
        input=None, labels=None, architecture=None, board="kv260", part=None,
        clock_ns=10.0, tolerance=0.1, seed=197, stage=stage,
        vitis=None, vivado=None, hls_timeout=120, vivado_timeout=120,
        execute_native=native, plan=False,
    )


def _mock_prep(monkeypatch):
    from fpgai_research.toolchains import workflow

    def prepare(argv):
        assert argv[0] == "prepare"
        assert "--onnx" in argv and "--input" in argv
        assert "--part" in argv and "--case" in argv
        case = Path(argv[argv.index("--case") + 1])
        case.mkdir()
        (case / "manifest.json").write_text('{"cpp_sha256":"sample"}')
        (case / "model.cpp").write_text("void fpgai_model() {}")
        (case / "vivado").mkdir()
        return 0

    monkeypatch.setattr(workflow, "main", prepare)


def _evidence(*, level="E1-HLS-source", csim=None, timing=None, drc=None):
    return {
        "evidence": level, "csim_passed": csim,
        "metrics": {"latency_worst": 10 if level != "E1-HLS-source" else None},
        "vivado_metrics": (
            {"timing": {"timing_met": timing, "wns_ns": 0.5 if timing else -0.5},
             "drc": {"clean": drc}} if level == "E4-Vivado-implementation"
            else None
        ),
        "warnings": [],
    }


def test_vendor_path_supports_product_root_and_binary(tmp_path):
    product = tmp_path / "Vitis" / "2023.2"
    (product / "bin").mkdir(parents=True)
    binary = product / "bin" / "vitis_hls"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    setup = product / "settings64.sh"
    setup.write_text("export FPGAI_USER_TOOL_TEST=1\n")
    assert app.resolve_vendor_tool("vitis_hls", str(product)) == app.VendorTool(
        executable=str(binary.resolve()), settings_script=str(setup.resolve()))
    assert app.resolve_vendor_tool("vitis_hls", str(binary)).executable == str(binary)
    assert app.resolve_vendor_tool("vitis_hls", str(setup)).settings_script == str(setup)


def test_vendor_selection_never_silently_chooses_multiple_versions(tmp_path):
    root = tmp_path / "AMD"
    for version in ("2023.2", "2024.1"):
        folder = root / "Vitis" / version / "bin"
        folder.mkdir(parents=True)
        binary = folder / "vitis_hls"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
    with pytest.raises(ValueError, match="Several installations"):
        app.resolve_vendor_tool("vitis_hls", str(root))


def test_vendor_environment_sources_only_explicit_script_and_does_not_leak(tmp_path):
    if not __import__("shutil").which("bash"):
        pytest.skip("Requires bash")
    setup = tmp_path / "settings64.sh"
    setup.write_text("export FPGAI_PRIVATE_TEST=inside_child_only\n")
    original = os.environ.get("FPGAI_PRIVATE_TEST")
    env = app.vendor_environment([app.VendorTool("/path/to/vitis_hls", str(setup))])
    assert env is not None and env["FPGAI_PRIVATE_TEST"] == "inside_child_only"
    assert os.environ.get("FPGAI_PRIVATE_TEST") == original
    assert app.vendor_environment([]) is None


def test_read_only_plan_does_not_create_case_or_run_native(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(app, "_validate_model", lambda path: (1, 3, 8, 8))
    args = _args(tmp_path)
    assert app.run(args) == 0
    assert not Path(args.output).exists()
    report = json.loads(capsys.readouterr().out)
    assert report["action"] == "PLAN_ONLY_NO_FILES_CREATED"
    assert report["vendor_tools_executed"] is False
    assert report["board_bitstream_generated"] is False


def test_native_part_must_be_selected_not_inferred_from_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_validate_model", lambda path: (1, 4))
    args = _args(tmp_path, native=True)
    args.board = None
    args.part = None
    with pytest.raises(ValueError, match="Select --board"):
        app.run(args)


def test_source_stage_prepares_without_amd_toolchain(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_validate_model", lambda path: (1, 4))
    monkeypatch.setattr(app, "_create_synthetic_input",
                        lambda target, shape, seed: target.write_bytes(b"synthetic"))
    _mock_prep(monkeypatch)
    from fpgai_research.experiments import results
    monkeypatch.setattr(results, "summarize_case", lambda case: _evidence())
    args = _args(tmp_path, stage="source")
    assert app.run(args) == 0
    output = Path(args.output)
    summary = json.loads((output / "user_run_summary.json").read_text())
    assert summary["state"] == "SOURCE_READY"
    assert summary["board_bitstream_generated"] is False
    assert (output / "synthetic_sample.npy").read_bytes() == b"synthetic"


def test_existing_case_is_never_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_validate_model", lambda path: (1, 4))
    args = _args(tmp_path, stage="source")
    Path(args.output).mkdir()
    with pytest.raises(FileExistsError, match="Output already exists"):
        app.run(args)


def _native_mocks(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_validate_model", lambda path: (1, 4))
    monkeypatch.setattr(app, "_create_synthetic_input",
                        lambda target, shape, seed: target.write_bytes(b"synthetic"))
    monkeypatch.setattr(app, "resolve_vendor_tool", lambda name, root:
                        app.VendorTool("/mock/" + name, None))
    monkeypatch.setattr(app, "vendor_environment", lambda tools: {"FPGAI_TEST": "1"})
    _mock_prep(monkeypatch)


def test_failing_csim_blocks_vivado_even_if_hls_xml_exists(monkeypatch, tmp_path):
    _native_mocks(monkeypatch, tmp_path)
    from fpgai_research.experiments import results
    from fpgai_research.toolchains.vitis_hls import runner as vitis
    from fpgai_research.toolchains.vivado import runner as vivado
    monkeypatch.setattr(results, "summarize_case", lambda case:
                        _evidence(level="E3-HLS-synthesis-report", csim=False))
    called = []
    def fake_vitis(*args, **kwargs):
        called.append("vitis")
        assert kwargs["environment"]["FPGAI_TEST"] == "1"
        return {"completed_successfully": True}
    monkeypatch.setattr(vitis, "run_vitis", fake_vitis)
    monkeypatch.setattr(vivado, "run_vivado", lambda *a, **k:
                        pytest.fail("Vivado started after failed C simulation"))
    assert app.run(_args(tmp_path, native=True)) == 2
    assert called == ["vitis"]
    summary = json.loads((tmp_path / "new-case" / "user_run_summary.json").read_text())
    assert summary["state"] == "HLS_NOT_QUALIFIED"
    assert summary["csim_passed"] is False


def test_hls_stage_accepts_only_attested_e3_plus_passed_csim(monkeypatch, tmp_path):
    _native_mocks(monkeypatch, tmp_path)
    from fpgai_research.experiments import results
    from fpgai_research.toolchains.vitis_hls import runner as vitis
    monkeypatch.setattr(results, "summarize_case", lambda case:
                        _evidence(level="E3-HLS-synthesis-report", csim=True))
    monkeypatch.setattr(vitis, "run_vitis",
                        lambda *a, **k: {"completed_successfully": True})
    assert app.run(_args(tmp_path, stage="hls", native=True)) == 0
    summary = json.loads((tmp_path / "new-case" / "user_run_summary.json").read_text())
    assert summary["state"] == "HLS_QUALIFIED"


def test_ooc_stage_keeps_board_bitstream_false(monkeypatch, tmp_path):
    _native_mocks(monkeypatch, tmp_path)
    from fpgai_research.experiments import results
    from fpgai_research.toolchains.vitis_hls import runner as vitis
    from fpgai_research.toolchains.vivado import runner as vivado

    stages = iter([
        _evidence(level="E3-HLS-synthesis-report", csim=True),
        _evidence(level="E4-Vivado-implementation", csim=True, timing=True, drc=True),
        _evidence(level="E4-Vivado-implementation", csim=True, timing=True, drc=True),
    ])
    monkeypatch.setattr(results, "summarize_case", lambda case: next(stages))
    monkeypatch.setattr(vitis, "run_vitis",
                        lambda *a, **k: {"completed_successfully": True})
    monkeypatch.setattr(vivado, "run_vivado",
                        lambda *a, **k: {"completed_successfully": True})
    copied = []
    monkeypatch.setattr(app, "_stage_roms", lambda case: copied.append(case))
    assert app.run(_args(tmp_path, native=True)) == 0
    assert len(copied) == 1
    summary = json.loads((tmp_path / "new-case" / "user_run_summary.json").read_text())
    assert summary["state"] == "OOC_QUALIFIED"
    assert summary["routed_ooc"] is True
    assert summary["board_bitstream_generated"] is False
    assert summary["device_programmed"] is False


def test_actual_rom_staging_preserves_sources_and_refuses_overwrite(tmp_path):
    case = tmp_path / "case"
    rtl = case / "fpgai_hls_project" / "solution1" / "syn" / "verilog"
    rtl.mkdir(parents=True)
    (case / "vivado").mkdir()
    (rtl / "fpgai_model.v").write_text("module fpgai_model; endmodule\n")
    (rtl / "weights.dat").write_text("abcdef\n")
    app._stage_roms(case)
    assert (case / "vivado" / "weights.dat").read_bytes() == (rtl / "weights.dat").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        app._stage_roms(case)


def test_native_runner_passes_explicit_environment_only_to_subprocess(tmp_path):
    from fpgai_research.toolchains.vitis_hls.runner import run_vitis

    binary = tmp_path / "vitis_hls"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "if '-version' in sys.argv:\n"
        " print('Vitis HLS v2023.2')\n"
        "else:\n"
        " print('MARKER=' + os.environ.get('FPGAI_TEST_RUNNER', 'missing'))\n"
    )
    binary.chmod(0o755)
    (tmp_path / "run_hls.tcl").write_text("exit\n")
    (tmp_path / "model.cpp").write_text("source\n")
    (tmp_path / "tb.cpp").write_text("bench\n")
    receipt = run_vitis(tmp_path, executable=str(binary),
                        environment={"FPGAI_TEST_RUNNER": "injected"})
    assert receipt["completed_successfully"] is True
    assert "MARKER=injected" in (tmp_path / "logs" / "vitis.stdout.log").read_text()
    assert "FPGAI_TEST_RUNNER" not in json.dumps(receipt)
    assert os.environ.get("FPGAI_TEST_RUNNER") is None


def test_invalid_native_environment_rejected_before_tool_launch(tmp_path):
    from fpgai_research.toolchains.vitis_hls.runner import run_vitis
    binary = tmp_path / "vitis_hls"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    (tmp_path / "run_hls.tcl").write_text("exit\n")
    with pytest.raises(ValueError, match="Invalid explicit native tool environment"):
        run_vitis(tmp_path, executable=str(binary), environment={"BAD=KEY": "value"})


def test_ooc_timing_failure_is_not_called_qualified(monkeypatch, tmp_path):
    _native_mocks(monkeypatch, tmp_path)
    from fpgai_research.experiments import results
    from fpgai_research.toolchains.vitis_hls import runner as vitis
    from fpgai_research.toolchains.vivado import runner as vivado

    stages = iter([
        _evidence(level="E3-HLS-synthesis-report", csim=True),
        _evidence(level="E4-Vivado-implementation", csim=True, timing=False, drc=True),
        _evidence(level="E4-Vivado-implementation", csim=True, timing=False, drc=True),
    ])
    monkeypatch.setattr(results, "summarize_case", lambda case: next(stages))
    monkeypatch.setattr(vitis, "run_vitis",
                        lambda *a, **k: {"completed_successfully": True})
    monkeypatch.setattr(vivado, "run_vivado",
                        lambda *a, **k: {"completed_successfully": True})
    monkeypatch.setattr(app, "_stage_roms", lambda case: None)
    assert app.run(_args(tmp_path, native=True)) == 3
    summary = json.loads((tmp_path / "new-case" / "user_run_summary.json").read_text())
    assert summary["state"] == "OOC_NOT_QUALIFIED"
    assert summary["timing_met"] is False
    assert summary["board_bitstream_generated"] is False


def test_native_execution_requires_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_validate_model", lambda path: (1, 4))
    monkeypatch.setattr(app, "resolve_vendor_tool",
                        lambda *args: pytest.fail("Vendor resolution in dry run"))
    args = _args(tmp_path, stage="ooc", native=False)
    assert app.run(args) == 0
    assert not (tmp_path / "new-case").exists()


def test_vitis_hls_2023_2_explicit_installation_layout(tmp_path):
    """Recognize the common standalone Vitis_HLS/2023.2 Linux install."""
    product = tmp_path / "Xilinx" / "Vitis_HLS" / "2023.2"
    (product / "bin").mkdir(parents=True)
    binary = product / "bin" / "vitis_hls"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    setup = product / "settings64.sh"
    setup.write_text("export FPGAI_HLS_LOCAL_SETUP=1\n")
    resolved = app.resolve_vendor_tool("vitis_hls", str(product))
    assert resolved.executable == str(binary.resolve())
    assert resolved.settings_script == str(setup.resolve())
