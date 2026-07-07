from __future__ import annotations

import json
from pathlib import Path

from fpgai.backends.vivado.run_bridge import (
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
