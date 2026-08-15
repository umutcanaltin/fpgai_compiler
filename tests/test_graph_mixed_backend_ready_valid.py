from pathlib import Path

from fpgai.ir.graph import Graph
from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend import (
    GraphMixedBackendPhysicalRequest,
    HLSPhysicalBinding,
    VHDLPhysicalBinding,
    emit_graph_mixed_backend_physical_project,
)
from fpgai.implementations.mixed_backend.physical import _ready_valid_hls_ports
from fpgai.implementations.vhdl_integration import parse_vhdl_scalar_stream_abi


def _fake_axis_hls(root: Path, top: str, expression: str) -> Path:
    rtl = root / top
    rtl.mkdir()
    (rtl / f"{top}.v").write_text(
        f'''module {top}(ap_clk, ap_rst_n, input_TDATA, input_TVALID, input_TREADY, output_TDATA, output_TVALID, output_TREADY);\ninput ap_clk;\ninput ap_rst_n;\ninput [15:0] input_TDATA;\ninput input_TVALID;\noutput input_TREADY;\noutput [15:0] output_TDATA;\noutput output_TVALID;\ninput output_TREADY;\nassign input_TREADY = output_TREADY;\nassign output_TDATA = {expression};\nassign output_TVALID = input_TVALID;\nendmodule\n''',
        encoding="utf-8",
    )
    return rtl


def _graph() -> Graph:
    graph = Graph("ready_valid_chain")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "a", "b", "output"):
        graph.add_tensor(name, (1,), "int16")
    graph.add_op("Scale2", ["input"], ["a"], name="hls_pre")
    graph.add_op("VHDLIdentity", ["a"], ["b"], name="vhdl_mid")
    graph.add_op("Add1", ["b"], ["output"], name="hls_post")
    return graph


def test_ready_valid_hls_axis_ports_are_mapped():
    ports = {
        "ap_clk": ("input", 1),
        "ap_rst_n": ("input", 1),
        "input_TDATA": ("input", 16),
        "input_TVALID": ("input", 1),
        "input_TREADY": ("output", 1),
        "output_TDATA": ("output", 16),
        "output_TVALID": ("output", 1),
        "output_TREADY": ("input", 1),
    }
    mapped = _ready_valid_hls_ports(ports)
    assert mapped["input_ready"] == "input_TREADY"
    assert mapped["output_ready"] == "output_TREADY"


def test_ready_valid_vhdl_contract_parses():
    contract = implementation_contract_from_manifest(Path("examples/packages/identity_ready_valid_vhdl"))
    abi = parse_vhdl_scalar_stream_abi(contract)
    assert abi.abi == "scalar_ready_valid_v1"
    assert abi.input_ready == "input_ready"
    assert abi.output_ready == "output_ready"


def test_graph_ready_valid_emits_backpressure_chain(tmp_path):
    pre = _fake_axis_hls(tmp_path, "scale2_axis", "input_TDATA << 1")
    post = _fake_axis_hls(tmp_path, "add1_axis", "input_TDATA + 1")
    vhdl = implementation_contract_from_manifest(Path("examples/packages/identity_ready_valid_vhdl"))
    result = emit_graph_mixed_backend_physical_project(
        GraphMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=_graph(),
            bindings={
                "hls_pre": HLSPhysicalBinding("hls_pre", pre, "scale2_axis"),
                "vhdl_mid": VHDLPhysicalBinding("vhdl_mid", vhdl),
                "hls_post": HLSPhysicalBinding("hls_post", post, "add1_axis"),
            },
            physical_profile="linear_scalar_ready_valid_v1",
        )
    )
    assert result.ok, result.issues
    wrapper = result.wrapper.read_text(encoding="utf-8")
    report = result.report_path.read_text(encoding="utf-8")
    assert "input_ready" in wrapper
    assert "output_ready" in wrapper
    assert ".input_TREADY(input_ready)" in wrapper
    assert '"backpressure": true' in report
    assert '"interface": "scalar_ready_valid_v1"' in report


def test_ready_valid_profile_rejects_valid_only_vhdl(tmp_path):
    pre = _fake_axis_hls(tmp_path, "scale2_axis", "input_TDATA << 1")
    post = _fake_axis_hls(tmp_path, "add1_axis", "input_TDATA + 1")
    old_vhdl = implementation_contract_from_manifest(Path("examples/packages/scale_bias_vhdl"))
    result = emit_graph_mixed_backend_physical_project(
        GraphMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=_graph(),
            bindings={
                "hls_pre": HLSPhysicalBinding("hls_pre", pre, "scale2_axis"),
                "vhdl_mid": VHDLPhysicalBinding("vhdl_mid", old_vhdl),
                "hls_post": HLSPhysicalBinding("hls_post", post, "add1_axis"),
            },
            physical_profile="linear_scalar_ready_valid_v1",
        )
    )
    assert not result.ok
    assert result.issues[0].code == "MIXGRAPH015"

def test_ready_valid_hls_axis_ports_accept_vitis_renamed_prefixes():
    ports = {
        "ap_clk": ("input", 1),
        "ap_rst_n": ("input", 1),
        "input_r_TDATA": ("input", 16),
        "input_r_TVALID": ("input", 1),
        "input_r_TREADY": ("output", 1),
        "output_r_TDATA": ("output", 16),
        "output_r_TVALID": ("output", 1),
        "output_r_TREADY": ("input", 1),
    }
    mapped = _ready_valid_hls_ports(ports)
    assert mapped["input_data"] == "input_r_TDATA"
    assert mapped["input_valid"] == "input_r_TVALID"
    assert mapped["input_ready"] == "input_r_TREADY"
    assert mapped["output_data"] == "output_r_TDATA"
    assert mapped["output_valid"] == "output_r_TVALID"
    assert mapped["output_ready"] == "output_r_TREADY"


def test_ready_valid_hls_axis_ports_reject_ambiguous_multiple_input_streams():
    import pytest

    ports = {
        "a_TDATA": ("input", 16), "a_TVALID": ("input", 1), "a_TREADY": ("output", 1),
        "b_TDATA": ("input", 16), "b_TVALID": ("input", 1), "b_TREADY": ("output", 1),
        "y_TDATA": ("output", 16), "y_TVALID": ("output", 1), "y_TREADY": ("input", 1),
    }
    with pytest.raises(ValueError, match="MIXRTL009"):
        _ready_valid_hls_ports(ports)
