"""Zero-native tests for the experimental PYNQ-Z2 AXI DMA board workflow."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json

import numpy as np
import pytest

from fpgai_research.toolchains.board.bridge import (
    bridge_cpp, bridge_testbench_cpp, bridge_hls_tcl, validate_tensor_contract,
)
from fpgai_research.toolchains.board.project import pynq_z2_board_tcl
from fpgai_research.toolchains.board import cli
from fpgai_research.toolchains.board.pynq_runtime import (
    encode_ap_fixed, decode_ap_fixed,
)


def _manifest():
    return {
        "input_shape": [1, 1, 6, 6],
        "output_shape": [1, 2],
        "precision_bits": 16,
        "integer_bits": 6,
    }


def _args(tmp_path, *, native=False, board="pynq-z2"):
    return argparse.Namespace(
        case=str(tmp_path / "original"), onnx=str(tmp_path / "original.onnx"),
        output=str(tmp_path / "new_board"), board=board,
        board_part="vendor.test:pynq-z2:part0:1.0", clock_ns=10.0,
        stage="bitstream", vitis="/mock/vitis", vivado="/mock/vivado",
        hls_timeout=7200, vivado_timeout=14400,
        execute_native=native, plan=False,
        accept_board_advisories=False,
    )


def test_fixed_shapes_and_stream_contract():
    assert validate_tensor_contract(_manifest()) == (36, 2, 16, 6)
    src = bridge_cpp(_manifest())
    assert '#include "model.cpp"' in src
    assert "extern \"C\" void fpgai_dma(" in src
    assert "#pragma HLS INTERFACE axis port=axis_in" in src
    assert "#pragma HLS INTERFACE axis port=axis_out" in src
    assert "#pragma HLS INTERFACE s_axilite port=return bundle=control" in src
    assert "x[i].range(15,0)=word.data.range(15,0)" in src
    assert "fpgai_model(x,y)" in src
    assert "word.last=(i==1)" in src
    assert "word.keep=-1" in src and "word.strb=-1" in src
    assert "FPGAI_BOARD_DMA_CSIM_PASS" in bridge_testbench_cpp(_manifest())
    assert "csim_design" in bridge_hls_tcl(part="xc7z020clg400-1", clock_ns=10)
    assert "csynth_design" in bridge_hls_tcl(part="xc7z020clg400-1", clock_ns=10)
    assert "export_design -format ip_catalog" in bridge_hls_tcl(
        part="xc7z020clg400-1", clock_ns=10)


@pytest.mark.parametrize("update", [
    {"input_shape": [0, 4]}, {"output_shape": []},
    {"input_shape": [262145]}, {"precision_bits": 33},
    {"integer_bits": 0}, {"integer_bits": 17}, {"input_shape": [2.0, 3]},
])
def test_bad_shapes_and_formats_rejected(update):
    manifest = {**_manifest(), **update}
    with pytest.raises(ValueError):
        validate_tensor_contract(manifest)


def test_board_recipe_contains_real_ps_dma_path_and_programming():
    tcl = pynq_z2_board_tcl(
        board_part="vendor.test:pynq-z2:part0:1.0",
        part="xc7z020clg400-1", clock_ns=10.0)
    for token in (
        "processing_system7", "axi_dma:7.1", "M_AXI_GP0", "S_AXI_HP0",
        "M_AXIS_MM2S", "S_AXIS_S2MM", "fpgai_dma_0/axis_in",
        "fpgai_dma_0/axis_out", "s_axi_control", "proc_sys_reset",
        "FCLK_CLK0", "FCLK_RESET0_N", "assign_bd_address",
        "validate_bd_design", "generate_target all", "make_wrapper",
        "launch_runs impl_1 -to_step write_bitstream",
        "get_timing_paths -setup", "report_drc", "system.hwh",
    ):
        assert token in tcl, token
    assert "set_property SEVERITY" not in tcl
    assert "set_msg_config" not in tcl
    assert "write_bitstream -force" not in tcl


@pytest.mark.parametrize("kwargs", [
    {"board_part": "bad; puts PWNED"},
    {"part": "xck26-sfvc784-2LV-c"},
    {"clock_ns": 15.0},
])
def test_wrong_device_and_untrusted_tcl_values_rejected(kwargs):
    values = {
        "board_part": "vendor.test:pynq-z2:part0:1.0",
        "part": "xc7z020clg400-1", "clock_ns": 10.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        pynq_z2_board_tcl(**values)


def test_board_plan_never_runs_vendor_tools_or_writes_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_verify_original_case",
                        lambda case, onnx: _manifest())
    monkeypatch.setattr(cli, "resolve_vendor_tool",
                        lambda *args, **kwargs: pytest.fail("Vendor resolved in plan"))
    original = tmp_path / "original"
    original.mkdir()
    # Even in a mocked read-only plan, the original provenance manifest
    # must exist. It is checked before allocating a new board build folder.
    manifest_path = original / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    manifest_before = manifest_path.read_bytes()
    (tmp_path / "original.onnx").write_bytes(b"fake-only-plan")
    args = _args(tmp_path)
    assert cli.run(args) == 0
    assert not (tmp_path / "new_board").exists()
    assert manifest_path.read_bytes() == manifest_before
    report = json.loads(capsys.readouterr().out)
    assert report["action"] == "PLAN_ONLY_NO_FILES_CREATED"
    assert report["board_inference_verified"] is False
    assert report["transport_bits_per_element"] == 32


def test_kv260_cannot_reuse_pynq_z2_ps_preset(tmp_path):
    with pytest.raises(ValueError, match="distinct Kria"):
        cli.run(_args(tmp_path, board="kv260"))


def test_numerical_pack_roundtrip_ap_fixed_16_6():
    values = np.array([0.0, 0.125, -0.125, 0.5, -1.5, 1.5], dtype=np.float32)
    bits = encode_ap_fixed(values, width=16, integer_bits=6)
    assert bits.dtype == np.uint32
    restored = decode_ap_fixed(bits, width=16, integer_bits=6)
    assert np.array_equal(restored, values)
    assert int(bits[2]) == 0xFF80
    assert int(bits[0]) == 0


def test_quantization_saturation_and_sign():
    values = np.array([99.0, -99.0, -0.00048828125, +0.00048828125],
                      dtype=np.float32)
    bits = encode_ap_fixed(values, width=16, integer_bits=6)
    actual = decode_ap_fixed(bits, width=16, integer_bits=6)
    assert actual[0] == np.float32(32767 / 1024)
    assert actual[1] == np.float32(-32.0)
    assert actual[2] == np.float32(0.0)
    assert actual[3] == np.float32(1.0 / 1024.0)


def test_wrong_input_format_fails_before_hardware_import():
    with pytest.raises(ValueError, match="float32"):
        encode_ap_fixed(np.ones(2, dtype=np.float64), width=16, integer_bits=6)


def test_strict_overlay_hash_and_drc_gate(tmp_path):
    from fpgai_research.toolchains.board.pynq_runtime import _verify_bundle
    bit = tmp_path / "fpgai_dma.bit"
    hwh = tmp_path / "fpgai_dma.hwh"
    bit.write_bytes(b"board-image-synthetic")
    hwh.write_bytes(b"<hw/>")
    doc = {
        "schema": "fpgai-research/board-dma-build/v1",
        "board": "pynq-z2",
        "state": "BOARD_BITSTREAM_DRC_CLEAN_UNTESTED_ON_DEVICE",
        "drc_clean": True, "strict_board_drc_clean_qualified": True,
        "input_elements": 6, "output_elements": 2,
        "input_shape": [1, 6], "output_shape": [1, 2],
        "payload_bits": 16, "integer_bits": 6,
        "bitstream_sha256": sha256(bit.read_bytes()).hexdigest(),
        "hwh_sha256": sha256(hwh.read_bytes()).hexdigest(),
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(doc))
    assert _verify_bundle(path, bit)["output_elements"] == 2
    bit.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Bitstream/HWH"):
        _verify_bundle(path, bit)
    bit.write_bytes(b"board-image-synthetic")
    doc["drc_clean"] = False
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="unqualified"):
        _verify_bundle(path, bit)



def test_warning_only_board_image_requires_second_approval(tmp_path):
    from fpgai_research.toolchains.board.pynq_runtime import _verify_bundle
    bit = tmp_path / "fpgai_dma.bit"
    hwh = tmp_path / "fpgai_dma.hwh"
    bit.write_bytes(b"bit")
    hwh.write_bytes(b"metadata")
    doc = {
        "schema": "fpgai-research/board-dma-build/v1",
        "board": "pynq-z2",
        "state": "BOARD_BITSTREAM_WITH_ADVISORIES_NOT_DEVICE_TESTED",
        "drc_clean": False,
        "drc_critical_warnings": 0,
        "drc_unconstrained_io": False,
        "warning_only_eligible": True,
        "user_accepted_board_advisories": True,
        "input_elements": 6, "output_elements": 2,
        "input_shape": [1, 6], "output_shape": [1, 2],
        "payload_bits": 16, "integer_bits": 6,
        "bitstream_sha256": sha256(bit.read_bytes()).hexdigest(),
        "hwh_sha256": sha256(hwh.read_bytes()).hexdigest(),
    }
    summary = tmp_path / "build.json"
    summary.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="unapproved DRC"):
        _verify_bundle(summary, bit)
    assert _verify_bundle(summary, bit, allow_advisories=True)["drc_clean"] is False
    doc["drc_critical_warnings"] = 1
    summary.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="unqualified"):
        _verify_bundle(summary, bit, allow_advisories=True)


def test_shape_mismatch_rejected_even_if_element_count_matches(tmp_path):
    from fpgai_research.toolchains.board.pynq_runtime import _verify_bundle
    bit = tmp_path / "fpgai_dma.bit"
    hwh = tmp_path / "fpgai_dma.hwh"
    bit.write_bytes(b"bit")
    hwh.write_bytes(b"hwh")
    doc = {
        "schema": "fpgai-research/board-dma-build/v1", "board": "pynq-z2",
        "state": "BOARD_BITSTREAM_DRC_CLEAN_UNTESTED_ON_DEVICE",
        "drc_clean": True, "strict_board_drc_clean_qualified": True,
        "input_elements": 2, "output_elements": 2,
        "input_shape": [1, 3], "output_shape": [1, 2],
        "payload_bits": 16, "integer_bits": 6,
        "bitstream_sha256": sha256(bit.read_bytes()).hexdigest(),
        "hwh_sha256": sha256(hwh.read_bytes()).hexdigest(),
    }
    summary = tmp_path / "build.json"
    summary.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Shape and element count"):
        _verify_bundle(summary, bit)


def test_board_reference_matches_original_model_ir(tmp_path):
    pytest.importorskip("onnx")
    from examples.make_neural_network_onnx import save_model
    from fpgai_research.toolchains.board.reference import generate_reference
    onnx_path = save_model("cnn_pool_dense", tmp_path / "model.onnx")
    x = np.zeros((1, 1, 6, 6), dtype=np.float32)
    np.save(tmp_path / "input.npy", x, allow_pickle=False)
    output_path = tmp_path / "expected.npy"
    result = generate_reference(str(onnx_path), str(tmp_path / "input.npy"),
                                str(output_path))
    assert result["output_shape"] == [1, 2]
    expected = np.load(output_path, allow_pickle=False)
    assert expected.shape == (1, 2)
    assert expected.dtype == np.float32
    assert result["on_device_comparison_performed"] is False
    with pytest.raises(FileExistsError):
        generate_reference(str(onnx_path), str(tmp_path / "input.npy"),
                           str(output_path))



def test_hls_plan_is_allowed_without_installed_board_part(
        monkeypatch, tmp_path, capsys):
    """Exporting AXI IP for a known silicon part does not need a PS board preset."""
    monkeypatch.setattr(cli, "_verify_original_case",
                        lambda case, onnx: _manifest())
    monkeypatch.setattr(cli, "resolve_vendor_tool",
                        lambda *args, **kwargs: pytest.fail("No vendor in read-only plan"))
    original = tmp_path / "original"
    original.mkdir()
    (original / "manifest.json").write_text(json.dumps(_manifest()))
    (tmp_path / "original.onnx").write_bytes(b"fake-onnx-for-source-only-plan")
    args = _args(tmp_path)
    args.board_part = None
    args.stage = "hls"
    assert cli.run(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["action"] == "PLAN_ONLY_NO_FILES_CREATED"
    assert plan["board_part"] is None
    assert plan["board_part_required_for_full_bitstream"] is True
    assert plan["part"] == "xc7z020clg400-1"
    assert plan["board_preset_used_for_hls"] is False
    assert not (tmp_path / "new_board").exists()


def test_bitstream_plan_still_requires_explicit_installed_board_preset(tmp_path):
    args = _args(tmp_path)
    args.board_part = None
    with pytest.raises(ValueError, match="Full board bitstream requires --board-part"):
        cli.run(args)
    assert not (tmp_path / "new_board").exists()


def test_vivado_board_tcl_accepts_explicit_repo_without_mutating_vendor_install():
    from pathlib import Path
    tcl = pynq_z2_board_tcl(
        board_part="vendor.test:pynq-z2:part0:1.0",
        part="xc7z020clg400-1", clock_ns=10.0,
    )
    discovery = (
        Path(__file__).resolve().parents[1] /
        "scripts/list_pynq_z2_board_parts.tcl"
    ).read_text()
    for src in (tcl, discovery):
        assert "FPGAI_BOARD_REPO" in src
        assert "set_param board.repoPaths [list $board_repo]" in src
        assert "if {![file isdirectory $board_repo]}" in src
