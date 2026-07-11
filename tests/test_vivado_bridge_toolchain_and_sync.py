from __future__ import annotations

import json
import os
from pathlib import Path

from fpgai.backends.vivado.run_bridge import (
    _classify_vivado_failure,
    _compile_output_dir_from_project_dir,
    _resolved_tool_command,
    _sync_bridge_result_to_compile_output,
)


def test_vivado_bridge_resolves_vivado_settings64_from_manifest(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    (build / "manifest.json").write_text(
        json.dumps(
            {
                "toolchain": {
                    "vivado": {
                        "settings64": "/opt/Xilinx/Vivado/2023.2/settings64.sh",
                        "executable": "vivado",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cmd, info = _resolved_tool_command(build, "vivado", ["-mode", "batch", "-source", "scripts/run_vivado.tcl"])

    assert cmd[:2] == ["bash", "-lc"]
    assert "settings64.sh" in cmd[2]
    assert "vivado" in cmd[2]
    assert info["uses_settings64"] is True
    assert info["settings64"] == "/opt/Xilinx/Vivado/2023.2/settings64.sh"


def test_compile_output_dir_resolves_manifest_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real_build"
    wrapper = tmp_path / "wrapper_build"
    real.mkdir()
    wrapper.mkdir()
    (real / "manifest.json").write_text("{}", encoding="utf-8")
    (wrapper / "manifest.json").symlink_to(real / "manifest.json")

    assert _compile_output_dir_from_project_dir(wrapper) == real.resolve()


def test_bridge_sync_overwrites_stale_reports_and_refreshes_package(tmp_path: Path) -> None:
    real = tmp_path / "real_build"
    wrapper_art = tmp_path / "exp" / "artifacts" / "d0"
    wrapper_build = wrapper_art / "build"
    bridge = wrapper_build / "vivado_bridge"

    real.mkdir(parents=True)
    wrapper_build.mkdir(parents=True)
    (real / "manifest.json").write_text("{}", encoding="utf-8")
    (wrapper_build / "manifest.json").symlink_to(real / "manifest.json")

    (real / "reports").mkdir()
    (real / "reports" / "vivado_implementation_report.json").write_text(
        json.dumps({"status": "tool_missing"}),
        encoding="utf-8",
    )
    (real / "reports" / "bitstream_report.json").write_text(
        json.dumps({"status": "tool_missing"}),
        encoding="utf-8",
    )

    (bridge / "bitstream").mkdir(parents=True)
    (bridge / "reports").mkdir(parents=True)
    (bridge / "logs").mkdir(parents=True)
    (bridge / "bitstream" / "fpgai_bd_wrapper.bit").write_text("bit", encoding="utf-8")
    (bridge / "bitstream" / "fpgai_bd.xsa").write_text("xsa", encoding="utf-8")
    (bridge / "reports" / "utilization_impl.rpt").write_text("util", encoding="utf-8")
    (bridge / "reports" / "timing_impl.rpt").write_text("timing", encoding="utf-8")
    (bridge / "reports" / "power_impl.rpt").write_text("power", encoding="utf-8")
    (bridge / "logs" / "vivado_build_stdout.log").write_text("", encoding="utf-8")
    (bridge / "logs" / "vivado_build_stderr.log").write_text("", encoding="utf-8")

    row = {
        "vivado_impl_requested": True,
        "vivado_ok": True,
        "vivado_ran": True,
        "vivado_returncode": 0,
        "vivado_stdout_log": str(bridge / "logs" / "vivado_build_stdout.log"),
        "vivado_stderr_log": str(bridge / "logs" / "vivado_build_stderr.log"),
        "vivado_tool": {"executable": "vivado", "uses_settings64": True},
    }

    _sync_bridge_result_to_compile_output(wrapper_art, bridge, row)

    impl = json.loads((real / "reports" / "vivado_implementation_report.json").read_text())
    bit = json.loads((real / "reports" / "bitstream_report.json").read_text())

    assert impl["status"] == "passed"
    assert impl["claimed_success"] is True
    assert bit["status"] == "passed"
    assert bit["bitstream_exists"] is True
    assert bit["xsa_exists"] is True
    assert (real / "vivado_bridge" / "bitstream" / "fpgai_bd_wrapper.bit").exists()
    assert (real / "runtime_package" / "package_manifest.json").exists()


def test_bridge_sync_impl_only_clears_unrequested_bitstream_and_refreshes_paper(tmp_path: Path) -> None:
    real = tmp_path / "real_build"
    wrapper_art = tmp_path / "exp" / "artifacts" / "d0"
    wrapper_build = wrapper_art / "build"
    bridge = wrapper_build / "vivado_bridge"

    real.mkdir(parents=True)
    wrapper_build.mkdir(parents=True)
    (real / "manifest.json").write_text("{}", encoding="utf-8")
    (wrapper_build / "manifest.json").symlink_to(real / "manifest.json")

    (real / "reports").mkdir()
    (real / "reports" / "paper_verification.json").write_text(
        json.dumps(
            {
                "pipeline_mode": "training_on_device",
                "paper_safe": False,
                "verification_flags": {
                    "source_generated": True,
                    "numeric_validated": False,
                    "hls_synthesized": True,
                    "vivado_implemented": False,
                    "bitstream_generated": False,
                    "fpga_executed": False,
                },
                "allowed_claims": {
                    "source_generation": True,
                    "numeric_correctness": False,
                    "hls_resource_timing": True,
                    "vivado_implementation": False,
                    "bitstream": False,
                    "real_fpga_runtime": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (real / "reports" / "paper_row.json").write_text(
        json.dumps({"vivado_implemented": False, "bitstream_generated": False}),
        encoding="utf-8",
    )
    (real / "runtime_package" / "hardware").mkdir(parents=True)
    (real / "runtime_package" / "hardware" / "stale.bit").write_text("stale", encoding="utf-8")
    (real / "vivado_bridge" / "bitstream").mkdir(parents=True)
    (real / "vivado_bridge" / "bitstream" / "stale.bit").write_text("stale", encoding="utf-8")

    (bridge / "bitstream").mkdir(parents=True)
    (bridge / "reports").mkdir(parents=True)
    (bridge / "logs").mkdir(parents=True)
    (bridge / "bitstream" / "unexpected.bit").write_text("bit", encoding="utf-8")
    (bridge / "bitstream" / "unexpected.xsa").write_text("xsa", encoding="utf-8")
    (bridge / "reports" / "utilization_impl.rpt").write_text("util", encoding="utf-8")
    (bridge / "reports" / "timing_impl.rpt").write_text("timing", encoding="utf-8")

    row = {
        "vivado_impl_requested": True,
        "bitstream_requested": False,
        "vivado_ok": True,
        "vivado_ran": True,
        "vivado_returncode": 0,
        "vivado_tool": {"executable": "vivado", "uses_settings64": True},
    }

    _sync_bridge_result_to_compile_output(wrapper_art, bridge, row)

    impl = json.loads((real / "reports" / "vivado_implementation_report.json").read_text())
    bit = json.loads((real / "reports" / "bitstream_report.json").read_text())
    paper = json.loads((real / "reports" / "paper_verification.json").read_text())
    package = json.loads((real / "runtime_package" / "package_manifest.json").read_text())

    assert impl["status"] == "passed"
    assert bit["requested"] is False
    assert bit["status"] == "not_requested"
    assert bit["bitstream_exists"] is False
    assert bit["xsa_exists"] is False
    assert paper["verification_flags"]["vivado_implemented"] is True
    assert paper["verification_flags"]["bitstream_generated"] is False
    assert package["hardware"]["bitstream"]["present"] is False
    assert not (real / "runtime_package" / "hardware" / "stale.bit").exists()
    assert not (real / "vivado_bridge" / "bitstream" / "stale.bit").exists()


def test_run_vivado_bridge_flow_is_shared_backend_api(tmp_path: Path, monkeypatch) -> None:
    from fpgai.backends.vivado import run_bridge as rb

    build = tmp_path / "build"
    (build / "hls").mkdir(parents=True)
    calls = {}

    def fake_generate(exp, board_name="pynq_z2", run_impl_default=False, run_bitstream_default=None):
        calls["generate"] = {
            "exp": Path(exp),
            "board_name": board_name,
            "run_impl_default": run_impl_default,
            "run_bitstream_default": run_bitstream_default,
        }
        return [{"design": "build", "vivado_bridge_generated": True}]

    def fake_run_for_artifact(artifact, export_hls_ip, run_vivado_synth, run_vivado_impl, run_bitstream, timeout_sec, force_hls_export=False):
        calls["run"] = {
            "artifact": Path(artifact),
            "export_hls_ip": export_hls_ip,
            "run_vivado_synth": run_vivado_synth,
            "run_vivado_impl": run_vivado_impl,
            "run_bitstream": run_bitstream,
            "timeout_sec": timeout_sec,
            "force_hls_export": force_hls_export,
        }
        return {
            "design": "build",
            "hls_ip_export_ok": True,
            "vivado_ok": True,
            "bitstream_exists": True,
            "xsa_exists": True,
        }

    monkeypatch.setattr(rb, "generate_vivado_bridge_for_experiment", fake_generate)
    monkeypatch.setattr(rb, "_run_for_artifact", fake_run_for_artifact)

    payload = rb.run_vivado_bridge_flow(
        build,
        board="kv260",
        export_hls_ip=True,
        run_vivado_impl=True,
        timeout_sec=123,
    )

    assert payload["ok"] is True
    assert payload["failed_rows"] == []
    assert calls["generate"]["board_name"] == "kv260"
    assert calls["generate"]["run_impl_default"] is True
    assert calls["generate"]["run_bitstream_default"] is True
    assert calls["run"]["artifact"] == build.resolve()
    assert calls["run"]["export_hls_ip"] is True
    assert calls["run"]["run_vivado_impl"] is True
    assert calls["run"]["run_bitstream"] is True
    assert calls["run"]["timeout_sec"] == 123
    assert (build / "vivado_bridge_run_artifacts.json").exists()


def test_run_vivado_bridge_flow_can_request_impl_without_bitstream(tmp_path: Path, monkeypatch) -> None:
    from fpgai.backends.vivado import run_bridge as rb

    build = tmp_path / "build"
    (build / "hls").mkdir(parents=True)
    calls = {}

    def fake_generate(exp, board_name="pynq_z2", run_impl_default=False, run_bitstream_default=None):
        calls["generate"] = {
            "run_impl_default": run_impl_default,
            "run_bitstream_default": run_bitstream_default,
        }
        return [{"design": "build", "vivado_bridge_generated": True}]

    def fake_run_for_artifact(artifact, export_hls_ip, run_vivado_synth, run_vivado_impl, run_bitstream, timeout_sec, force_hls_export=False):
        calls["run"] = {
            "run_vivado_impl": run_vivado_impl,
            "run_bitstream": run_bitstream,
        }
        return {
            "design": "build",
            "hls_ip_export_ok": True,
            "vivado_ok": True,
            "bitstream_exists": False,
            "xsa_exists": False,
        }

    monkeypatch.setattr(rb, "generate_vivado_bridge_for_experiment", fake_generate)
    monkeypatch.setattr(rb, "_run_for_artifact", fake_run_for_artifact)

    payload = rb.run_vivado_bridge_flow(
        build,
        export_hls_ip=True,
        run_vivado_impl=True,
        run_bitstream=False,
    )

    assert payload["ok"] is True
    assert payload["failed_rows"] == []
    assert calls["generate"]["run_impl_default"] is True
    assert calls["generate"]["run_bitstream_default"] is False
    assert calls["run"]["run_vivado_impl"] is True
    assert calls["run"]["run_bitstream"] is False


def test_compiler_orchestrates_yaml_requested_vivado_without_new_script() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")
    assert "run_vivado_bridge_flow" in source
    assert "_run_yaml_requested_vivado_bridge(out_dir, raw, build_stages)" in source
    assert "_update_manifest_after_vivado_bridge(out_dir, payload)" in source
    assert 'if build_stages.get("runtime_package"):' in source
    assert "emit_runtime_package(out_dir)" in source[source.find("def _run_yaml_requested_vivado_bridge"):source.find("@dataclass")]
    assert "subprocess" not in source[source.find("def _run_yaml_requested_vivado_bridge"):source.find("@dataclass")]


def test_bridge_sync_direct_build_dir_skips_self_copy(tmp_path: Path) -> None:
    build = tmp_path / "direct_build"
    bridge = build / "vivado_bridge"
    build.mkdir(parents=True)
    (build / "manifest.json").write_text("{}", encoding="utf-8")

    (bridge / "bitstream").mkdir(parents=True)
    (bridge / "reports").mkdir(parents=True)
    (bridge / "logs").mkdir(parents=True)
    (bridge / "bitstream" / "fpgai_bd_wrapper.bit").write_text("bit", encoding="utf-8")
    (bridge / "bitstream" / "fpgai_bd.xsa").write_text("xsa", encoding="utf-8")
    (bridge / "reports" / "utilization_impl.rpt").write_text("util", encoding="utf-8")
    (bridge / "reports" / "timing_impl.rpt").write_text("timing", encoding="utf-8")
    (bridge / "reports" / "power_impl.rpt").write_text("power", encoding="utf-8")
    (bridge / "vivado_bridge_manifest.json").write_text("{}", encoding="utf-8")

    row = {
        "vivado_impl_requested": True,
        "vivado_ok": True,
        "vivado_ran": True,
        "vivado_returncode": 0,
        "vivado_stdout_log": str(bridge / "logs" / "vivado_build_stdout.log"),
        "vivado_stderr_log": str(bridge / "logs" / "vivado_build_stderr.log"),
        "vivado_tool": {"executable": "vivado", "uses_settings64": True},
    }

    _sync_bridge_result_to_compile_output(build, bridge, row)

    impl = json.loads((build / "reports" / "vivado_implementation_report.json").read_text())
    bit = json.loads((build / "reports" / "bitstream_report.json").read_text())

    assert impl["status"] == "passed"
    assert bit["status"] == "passed"
    assert bit["bitstream_exists"] is True
    assert bit["xsa_exists"] is True
    assert (build / "runtime_package" / "package_manifest.json").exists()




def test_vivado_bridge_does_not_search_common_xilinx_roots_without_yaml_settings(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "Xilinx"
    settings = root / "Vivado" / "2023.2" / "settings64.sh"
    settings.parent.mkdir(parents=True)
    settings.write_text("export PATH=/fake/vivado/bin:$PATH\n", encoding="utf-8")

    build = tmp_path / "build"
    build.mkdir()
    (build / "manifest.json").write_text(json.dumps({"toolchain": {}}), encoding="utf-8")

    monkeypatch.setenv("FPGAI_XILINX_ROOTS", str(root))
    monkeypatch.setenv("PATH", "")

    cmd, info = _resolved_tool_command(build, "vivado", ["-mode", "batch", "-source", "scripts/run_vivado.tcl"])

    assert cmd[0] == "vivado"
    assert cmd[1:] == ["-mode", "batch", "-source", "scripts/run_vivado.tcl"]
    assert info["uses_settings64"] is False
    assert info["settings64"] is None
    assert info["source"] == "path_default"
    assert info["searched_roots"] == []
    assert info["searched_candidates"] == []


def test_vitis_hls_runner_sources_yaml_configured_settings64(tmp_path: Path, monkeypatch) -> None:
    from fpgai.backends.hls.runner import run_vitis_hls

    install = tmp_path / "Vitis_HLS" / "2023.2"
    bin_dir = install / "bin"
    bin_dir.mkdir(parents=True)
    settings = install / "settings64.sh"
    settings.write_text(f'export PATH="{bin_dir}:$PATH"\n', encoding="utf-8")

    fake = bin_dir / "vitis_hls"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "echo fake vitis_hls yaml source\n"
        "mkdir -p fpgai_hls_proj/sol1/syn/report\n"
        "printf 'fake csynth' > fpgai_hls_proj/sol1/syn/report/csynth.rpt\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    hls_dir = tmp_path / "hls"
    hls_dir.mkdir()
    (hls_dir / "run_hls.tcl").write_text("puts fake\n", encoding="utf-8")

    monkeypatch.setenv("FPGAI_XILINX_ROOTS", str(tmp_path / "ignored_root"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    result = run_vitis_hls(hls_dir=hls_dir, settings64=str(settings))

    assert result.ok is True
    assert result.returncode == 0
    assert "source" in result.command
    assert str(settings) in result.command
    assert result.csynth_report and result.csynth_report.endswith("csynth.rpt")


def test_vivado_failure_classifier_does_not_mislabel_impl_only_rows_as_bitstream(tmp_path: Path) -> None:
    bridge = tmp_path / "vivado_bridge"
    (bridge / "logs").mkdir(parents=True)
    (bridge / "logs" / "vivado_build_stdout.log").write_text(
        "launch_runs impl_1 -to_step write_bitstream\n"
        "ERROR: [Vivado 12-xxx] place_design failed\n",
        encoding="utf-8",
    )

    result = {"ok": False, "error": "Vivado returned non-zero status"}

    assert _classify_vivado_failure(bridge, result, run_bitstream=False) == "vivado_impl_failed_place_design"
    assert _classify_vivado_failure(bridge, result, run_bitstream=True) != "vivado_impl_failed_write_bitstream"


def test_vivado_failure_classifier_reports_write_bitstream_only_when_requested(tmp_path: Path) -> None:
    bridge = tmp_path / "vivado_bridge"
    (bridge / "logs").mkdir(parents=True)
    (bridge / "logs" / "vivado_build_stdout.log").write_text(
        "ERROR: [Vivado 12-xxx] write_bitstream failed after implementation\n",
        encoding="utf-8",
    )

    result = {"ok": False, "error": "Vivado returned non-zero status"}

    assert _classify_vivado_failure(bridge, result, run_bitstream=False) == "vivado_failed"
    assert _classify_vivado_failure(bridge, result, run_bitstream=True) == "vivado_impl_failed_write_bitstream"


def test_external_vivado_bridge_bd_is_interface_adaptive_for_m_axi_only_inference(tmp_path: Path) -> None:
    from fpgai.backends.vivado.vivado_bridge import generate_vivado_bridge_for_artifact

    build = tmp_path / "build"
    hls_src = build / "hls" / "src"
    hls_src.mkdir(parents=True)
    (build / "manifest.json").write_text(json.dumps({"top_name": "deeplearn"}), encoding="utf-8")
    (build / "hls" / "run_hls.tcl").write_text("puts hls\n", encoding="utf-8")
    (hls_src / "deeplearn.cpp").write_text(
        "void deeplearn(float* m_axi_gmem_input, float* m_axi_gmem_output) {\n"
        "#pragma HLS INTERFACE m_axi port=m_axi_gmem_input bundle=gmem_input\n"
        "#pragma HLS INTERFACE m_axi port=m_axi_gmem_output bundle=gmem_output\n"
        "#pragma HLS INTERFACE s_axilite port=return bundle=control\n"
        "}\n",
        encoding="utf-8",
    )

    result = generate_vivado_bridge_for_artifact(build, board_name="kv260", run_impl_default=True, run_bitstream_default=False)
    bd_source = Path(result["vivado_bridge_dir"]) / "scripts" / "create_bd.tcl"
    text = bd_source.read_text(encoding="utf-8")

    assert "fpgai_is_hls_m_axi_intf" in text
    assert "set use_dma [expr {$hls_axis_in ne \"\" && $hls_axis_out ne \"\"}]" in text
    assert "foreach hls_m_axi $hls_m_axi_ports" in text
    assert "*input*" not in text
    assert "*output*" not in text
    assert "$hls_in" not in text
    assert "m_axi_gmem_input" not in text  # no hard-coded mistaken stream wiring


def test_vivado_failure_classifier_reports_bd_interface_mode_mismatch(tmp_path: Path) -> None:
    bridge = tmp_path / "vivado_bridge"
    (bridge / "logs").mkdir(parents=True)
    (bridge / "logs" / "vivado_build_stderr.log").write_text(
        "ERROR: [BD 41-171] The modes of the interface pins 'M_AXIS_MM2S'(Master) "
        "and 'm_axi_gmem_input'(Master) are incompatible. They cannot be connected.\n"
        "ERROR: [BD 5-3] Error: running connect_bd_intf_net.\n",
        encoding="utf-8",
    )

    assert _classify_vivado_failure(bridge, {"ok": False}, run_bitstream=False) == "vivado_bd_failed_interface_mode_mismatch"


def test_yaml_vivado_bridge_existing_hls_ip_without_component_stays_report_only(tmp_path: Path, monkeypatch) -> None:
    from fpgai.engine import compiler as compiler_mod

    out_dir = tmp_path / "build"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "build_stages": {
                    "vivado_project": True,
                    "vivado_implementation": True,
                    "bitstream": False,
                    "runtime_package": False,
                },
                "pipeline_stages": [{"name": "vivado_project", "status": "pending"}],
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_run_vivado_bridge_flow(*args, **kwargs):
        captured.update(kwargs)
        return {
            "generated": [{"design": "build", "vivado_bridge_generated": True}],
            "tool_runs": [],
            "failed_rows": [],
            "ok": True,
        }

    monkeypatch.setattr(compiler_mod, "run_vivado_bridge_flow", fake_run_vivado_bridge_flow)
    monkeypatch.setattr(compiler_mod, "emit_experiment_artifact_reports", lambda *_args, **_kwargs: None)

    compiler_mod._run_yaml_requested_vivado_bridge(
        out_dir,
        {"build": {"existing_hls_ip": True}, "targets": {"platform": {"board": "kv260"}}},
        {"vivado_project": True, "vivado_implementation": True, "bitstream": False, "runtime_package": False},
    )

    assert captured["export_hls_ip"] is False
    assert captured["run_vivado_impl"] is False
    assert captured["run_bitstream"] is False


def test_yaml_vivado_bridge_owned_hls_synthesis_exports_ip_and_runs_impl(tmp_path: Path, monkeypatch) -> None:
    from fpgai.engine import compiler as compiler_mod

    out_dir = tmp_path / "build"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "build_stages": {
                    "vivado_project": True,
                    "vivado_implementation": True,
                    "bitstream": False,
                    "runtime_package": False,
                },
                "pipeline_stages": [{"name": "vivado_project", "status": "pending"}],
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_run_vivado_bridge_flow(*args, **kwargs):
        captured.update(kwargs)
        return {
            "generated": [{"design": "build", "vivado_bridge_generated": True}],
            "tool_runs": [{"design": "build", "vivado_ok": True, "hls_ip_export_ok": True}],
            "failed_rows": [],
            "ok": True,
        }

    monkeypatch.setattr(compiler_mod, "run_vivado_bridge_flow", fake_run_vivado_bridge_flow)
    monkeypatch.setattr(compiler_mod, "emit_experiment_artifact_reports", lambda *_args, **_kwargs: None)

    compiler_mod._run_yaml_requested_vivado_bridge(
        out_dir,
        {"build": {"existing_hls_ip": False}, "targets": {"platform": {"board": "kv260"}}},
        {"vivado_project": True, "vivado_implementation": True, "hls_synthesis": True, "bitstream": False, "runtime_package": False},
    )

    assert captured["export_hls_ip"] is True
    assert captured["run_vivado_impl"] is True
    assert captured["run_bitstream"] is False
