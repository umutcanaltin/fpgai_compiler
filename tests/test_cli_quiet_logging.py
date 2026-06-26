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

def test_quiet_compile_result_includes_manifest_sections(tmp_path, capsys):
    from fpgai.cli import _print_compile_result
    from fpgai.engine.result import CompileResult
    from fpgai.ir.graph import Graph

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(
        """
{
  "pipeline_mode": "inference",
  "top_kernel_name": "deeplearn",
  "seconds": 0.1,
  "prediction_artifacts": {
    "model_profile_json": "reports/model_profile.json",
    "resource_prediction_json": "reports/resource_prediction.json",
    "timing_prediction_json": "reports/timing_prediction.json",
    "prediction_summary_md": "reports/prediction_summary.md"
  },
  "hls_artifacts": {
    "hls_ran": false,
    "hls_ok": null,
    "hls_returncode": null,
    "artifact_metadata": {
      "path": "hls_artifact_metadata.json",
      "file_count": 14
    }
  },
  "runtime_package": {
    "path": "runtime_package/package_manifest.json",
    "package_dir": "runtime_package",
    "status": "created",
    "deployable_overlay_present": false,
    "bitstream_present": false,
    "hwh_present": false,
    "xsa_present": false,
    "file_count": 2
  },
  "pipeline_stages": [
    {"name": "load_config", "status": "done"},
    {"name": "runtime_package", "status": "done"}
  ]
}
""",
        encoding="utf-8",
    )

    result = CompileResult(out_dir=out_dir, graph=Graph())
    _print_compile_result(result, quiet=True)

    captured = capsys.readouterr().out
    assert "Prediction artifacts:" in captured
    assert "HLS artifacts       : available" in captured
    assert "Vivado bridge       : not_requested" in captured
    assert "Runtime package     : created" in captured
    assert "Pipeline stages     :" in captured
    assert "runtime_package: done" in captured



def test_compile_result_summary_includes_pipeline_stages_when_manifest_exists(tmp_path):
    import json

    from types import SimpleNamespace

    from fpgai.engine.result import CompileResult

    manifest = {
        "pipeline_mode": "inference",
        "top_kernel_name": "deeplearn",
        "seconds": 0.123,
        "pipeline_stages": [
            {"name": "load_config", "status": "done"},
            {"name": "run_hls", "status": "skipped"},
            {"name": "vivado_bridge", "status": "not_requested"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = CompileResult(
        out_dir=tmp_path,
        graph=SimpleNamespace(
            ops=[],
            params={},
            inputs=[],
            outputs=[],
        ),
        hls_project_dir=None,
        host_project_dir=None,
    )

    summary = result.summary()

    assert "Manifest" in summary
    assert "Pipeline mode        : inference" in summary
    assert "Top kernel           : deeplearn" in summary
    assert "Compile seconds      : 0.123" in summary
    assert "Pipeline stages" in summary
    assert "load_config: done" in summary
    assert "run_hls: skipped" in summary
    assert "vivado_bridge: not_requested" in summary

def test_compile_result_summary_includes_prediction_artifacts_when_manifest_exists(tmp_path):
    import json
    from types import SimpleNamespace

    from fpgai.engine.result import CompileResult

    manifest = {
        "prediction_artifacts": {
            "model_profile_json": "reports/model_profile.json",
            "resource_prediction_json": "reports/resource_prediction.json",
            "timing_prediction_json": "reports/timing_prediction.json",
            "prediction_summary_md": "reports/prediction_summary.md",
        }
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    result = CompileResult(
        out_dir=tmp_path,
        graph=SimpleNamespace(
            ops=[],
            params={},
            inputs=[],
            outputs=[],
        ),
        hls_project_dir=None,
        host_project_dir=None,
    )

    summary = result.summary()

    assert "Prediction artifacts" in summary
    assert "reports/model_profile.json" in summary
    assert "reports/resource_prediction.json" in summary
    assert "reports/timing_prediction.json" in summary
    assert "reports/prediction_summary.md" in summary



def test_compile_result_summary_includes_board_fit_prediction_artifacts(tmp_path):
    import json

    from fpgai.engine.result import CompileResult

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest = {
        "prediction_artifacts": {
            "model_profile_json": str(out_dir / "reports" / "model_profile.json"),
            "resource_prediction_json": str(out_dir / "reports" / "resource_prediction.json"),
            "timing_prediction_json": str(out_dir / "reports" / "timing_prediction.json"),
            "prediction_summary_md": str(out_dir / "reports" / "prediction_summary.md"),
            "board_fit_json": str(out_dir / "reports" / "board_fit.json"),
            "board_fit_markdown": str(out_dir / "reports" / "board_fit.md"),
            "board_fit": {
                "status": "near_limit",
                "limiting_dimension": "target_clock_mhz",
                "vivado_allowed": True,
            },
        }
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = object.__new__(CompileResult)
    object.__setattr__(result, "out_dir", out_dir)

    summary = "\n".join(result._prediction_artifact_summary_lines())

    assert "Board fit JSON" in summary
    assert "Board fit summary" in summary
    assert "Board fit status: near_limit" in summary
    assert "Board fit limiting: target_clock_mhz" in summary
    assert "Vivado allowed by fit: True" in summary


def test_compile_result_summary_includes_hardware_knob_contract(tmp_path):
    import json

    from fpgai.engine.result import CompileResult

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest = {
        "hardware_knob_contract": {
            "json": str(out_dir / "reports" / "hardware_knob_contract.json"),
            "markdown": str(out_dir / "reports" / "hardware_knob_contract.md"),
            "knob_count": 18,
            "manual_yaml_count": 12,
            "changed_or_clamped_count": 1,
            "report_only_count": 1,
        }
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = object.__new__(CompileResult)
    object.__setattr__(result, "out_dir", out_dir)

    summary = "\n".join(result._hardware_knob_contract_summary_lines())

    assert "Hardware knob contract: available" in summary
    assert "JSON" in summary
    assert "Summary" in summary
    assert "Knob count: 18" in summary
    assert "Manual YAML knobs: 12" in summary
    assert "Changed/clamped: 1" in summary
    assert "Report-only: 1" in summary


def test_compile_result_summary_includes_fit_policy_gate(tmp_path):
    import json
    from types import SimpleNamespace

    from fpgai.engine.result import CompileResult

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest = {
        "fit_policy_gate": {
            "policy": "enforce",
            "board_fit_status": "over_limit",
            "board_fit_limiting_dimension": "dsp",
            "vivado_allowed_by_board_fit": False,
            "over_limit": True,
            "blocked": True,
            "warning": False,
            "severity": "error",
            "blocked_stages": ["vivado_impl", "bitstream", "deployable_runtime_overlay"],
            "reason": "Board fit status is over_limit under fit_policy=enforce.",
        }
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    graph = SimpleNamespace(
        ops=[],
        params={},
        inputs=["input"],
        outputs=["output"],
    )

    result = CompileResult(
        out_dir=out_dir,
        graph=graph,
        hls_project_dir=None,
        host_project_dir=None,
        hls_ran=False,
        hls_ok=None,
        hls_returncode=None,
        hls_stdout_log=None,
        hls_stderr_log=None,
        hls_csynth_report=None,
    )

    summary = result.summary()

    assert "Fit policy gate" in summary
    assert "policy: enforce" in summary
    assert "board_fit_status: over_limit" in summary
    assert "blocked: True" in summary
    assert "blocked_stages" in summary
