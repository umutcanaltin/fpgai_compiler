"""Public AMD setup, deployment, and research-only licensing guardrails.

These checks intentionally require no FPGA tools, board, license server,
network access or vendor binaries.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import check_amd_environment as doctor
from scripts import prepare_public_preview as preview


ROOT = Path(__file__).resolve().parents[1]


def test_environment_inspection_never_invokes_vendor_binaries(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Vendor binary invocation is forbidden")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(shutil, "which", lambda _: None)

    report = doctor.inspect_environment({
        "FPGAI_AMD_ROOT": "/path/to/installation",
        "FPGAI_AMD_VERSION": "2023.2",
        "PLATFORM_REPO_PATHS": "/missing/board1:/missing/board2",
        "FPGAI_XRT_SETUP": "/missing/xrt/setup.sh",
        "LM_LICENSE_FILE": "DO_NOT_PRINT_THIS_SECRET",
    })
    assert report["vendor_commands_executed"] is False
    assert report["hardware_programmed"] is False
    assert not report["tools"]["vitis_hls"]["available"]
    assert not report["tools"]["vivado"]["available"]
    assert report["settings"]["FPGAI_AMD_VERSION"] == "2023.2"
    assert "DO_NOT_PRINT_THIS_SECRET" not in str(report)
    assert doctor.required_missing(
        report, hls=True, vivado=True, platform=True, xrt=True
    ) == ("vitis_hls", "vivado", "PLATFORM_REPO_PATHS", "XRT_SETUP")


def test_environment_inspection_discovers_explicit_exec_and_platform(tmp_path):
    vitis = tmp_path / "vitis_hls"
    vivado = tmp_path / "vivado"
    for executable in (vitis, vivado):
        executable.write_text("#!/usr/bin/env sh\nexit 0\n")
        executable.chmod(0o700)
    platforms = tmp_path / "platforms"
    platforms.mkdir()
    xrt = tmp_path / "xrt" / "setup.sh"
    xrt.parent.mkdir()
    xrt.write_text("# stub\n")
    report = doctor.inspect_environment({
        "FPGAI_VITIS_HLS_EXECUTABLE": str(vitis),
        "FPGAI_VIVADO_EXECUTABLE": str(vivado),
        "PLATFORM_REPO_PATHS": str(platforms),
        "FPGAI_XRT_SETUP": str(xrt),
    })
    assert report["tools"]["vitis_hls"]["available"]
    assert report["tools"]["vivado"]["available"]
    assert doctor.required_missing(
        report, hls=True, vivado=True, platform=True, xrt=True
    ) == ()
    assert not report["vendor_commands_executed"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="Requires Bash")
def test_sourceable_vendor_settings_require_explicit_install_root(tmp_path):
    settings = ROOT / "scripts/setup_amd_environment.sh"
    empty = subprocess.run(
        ["bash", "-c", f'source "{settings}"'],
        env={**os.environ, "FPGAI_AMD_ROOT": ""},
        text=True, capture_output=True, check=False,
    )
    assert empty.returncode != 0
    assert "Set FPGAI_AMD_ROOT" in empty.stderr
    assert not (tmp_path/"unexpected").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="Requires Bash")
def test_sourceable_vendor_settings_accept_fake_tools_without_running_them(
    tmp_path,
):
    settings = ROOT / "scripts/setup_amd_environment.sh"
    vitis = tmp_path / "Vitis/2023.2"
    vivado = tmp_path / "Vivado/2023.2"
    vitis.mkdir(parents=True)
    vivado.mkdir(parents=True)
    (vitis/"settings64.sh").write_text('export XILINX_VITIS="stub-vitis"\n')
    (vivado/"settings64.sh").write_text('export XILINX_VIVADO="stub-vivado"\n')
    result = subprocess.run(
        ["bash", "-c", f'source "{settings}" && '
         'test "$XILINX_VITIS" = stub-vitis && '
         'test "$XILINX_VIVADO" = stub-vivado'],
        env={**os.environ, "FPGAI_AMD_ROOT": str(tmp_path),
             "FPGAI_AMD_VERSION": "2023.2", "FPGAI_SOURCE_XRT": "0"},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "No compilation or FPGA execution" in result.stdout


def test_synthetic_input_generator_is_deterministic_and_create_only(tmp_path):
    pytest.importorskip("onnx")
    import numpy as np
    from examples import make_neural_network_onnx
    from examples import make_sample_input
    model = make_neural_network_onnx.save_model(
        "cnn_strided_dense", tmp_path/"tiny.onnx")
    first = make_sample_input.save_sample(model, tmp_path/"first.npy")
    second = make_sample_input.save_sample(model, tmp_path/"second.npy")
    arr = np.load(first, allow_pickle=False)
    assert arr.shape == (2, 1, 7, 8)
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(
        arr, np.load(second, allow_pickle=False))
    with pytest.raises(FileExistsError, match="overwrite"):
        make_sample_input.save_sample(model, first)
    with pytest.raises(ValueError, match="Seed"):
        make_sample_input.save_sample(model, tmp_path/"invalid.npy", seed=-2)


def test_distribution_contains_deployment_guides_and_helpers_not_a_live_license():
    expected = (
        "docs/ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md",
        "docs/WHO_CAN_USE_FPGAI.md",
        "docs/AMD_TOOLCHAIN_SETUP.md",
        "docs/FPGA_DEPLOYMENT.md",
        "scripts/setup_amd_environment.sh",
        "scripts/check_amd_environment.py",
        "examples/make_sample_input.py",
    )
    assert all(preview.allowed(path, profile="compiler") for path in expected)
    manifest = (ROOT/"MANIFEST.in").read_text()
    for p in expected:
        assert p in manifest or (
            p == "examples/make_sample_input.py"
            and "recursive-include examples *.py" in manifest)
    proposal = (ROOT/"docs/ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md").read_text()
    assert "NOT EFFECTIVE" in proposal
    assert "not an operative" in proposal
    assert "OSI" in (ROOT/"docs/WHO_CAN_USE_FPGAI.md").read_text()
    # No final license can be silently issued by a documentation-only sprint.
    assert not (ROOT/"LICENSE").is_file()
    assert not (ROOT/"LICENSE.md").is_file()


def test_deployment_docs_do_not_confuse_ooc_with_bitstream():
    guide = (ROOT/"docs/FPGA_DEPLOYMENT.md").read_text()
    assert "out-of-context" in guide
    assert "does **not** invoke" in guide and "write_bitstream" in guide
    assert "AXI" in guide and "bitstream" in guide
    assert "PYNQ" in guide and "Alveo" in guide and "KV260" in guide
    assert "fpgai-research-tools prepare" in guide
    assert "fpgai-research-tools run-vitis" in guide
    assert "fpgai-research-tools run-vivado" not in guide or "out-of-context" in guide


def test_embedded_linux_paths_are_optional_but_fail_closed_when_required(tmp_path):
    rootfs=tmp_path/"rootfs.ext4"
    rootfs.write_bytes(b"stub")
    sysroot=tmp_path/"sysroot"
    sysroot.mkdir()
    report=doctor.inspect_environment({
        "ROOTFS":str(rootfs),
        "SYSROOT":str(sysroot),
        "SDKTARGETSYSROOT":str(sysroot),
    })
    assert doctor.required_missing(
        report,hls=False,vivado=False,platform=False,xrt=False,
        rootfs=True,sysroot=True,
    )==()
    report_missing=doctor.inspect_environment({
        "ROOTFS":str(tmp_path/"missing.ext4"),
        "SYSROOT":str(tmp_path/"missing-sysroot"),
    })
    assert doctor.required_missing(
        report_missing,hls=False,vivado=False,platform=False,xrt=False,
        rootfs=True,sysroot=True,
    )==("ROOTFS","SYSROOT")
