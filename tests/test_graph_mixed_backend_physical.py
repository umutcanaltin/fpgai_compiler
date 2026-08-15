from pathlib import Path

from fpgai.ir.graph import Graph
from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend import (
    GraphMixedBackendPhysicalRequest,
    HLSPhysicalBinding,
    VHDLPhysicalBinding,
    emit_graph_mixed_backend_physical_project,
)


def _fake_hls(root: Path, top: str, expression: str) -> Path:
    rtl = root / top
    rtl.mkdir()
    (rtl / f"{top}.v").write_text(
        f'''module {top}(input_data, input_valid, output_data, output_valid);\ninput [15:0] input_data;\ninput input_valid;\noutput [15:0] output_data;\noutput output_valid;\nassign output_data = {expression};\nassign output_valid = input_valid;\nendmodule\n''',
        encoding="utf-8",
    )
    return rtl


def _graph() -> Graph:
    graph = Graph("physical_chain")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "a", "b", "output"):
        graph.add_tensor(name, (1,), "int16")
    graph.add_op("Scale2", ["input"], ["a"], name="hls_pre")
    graph.add_op("VHDLIdentity", ["a"], ["b"], name="vhdl_mid")
    graph.add_op("Add1", ["b"], ["output"], name="hls_post")
    return graph


def test_graph_physical_emits_both_backend_directions(tmp_path):
    pre = _fake_hls(tmp_path, "scale2_hls", "input_data << 1")
    post = _fake_hls(tmp_path, "add1_hls", "input_data + 1")
    vhdl = implementation_contract_from_manifest(Path("examples/packages/scale_bias_vhdl"))
    result = emit_graph_mixed_backend_physical_project(
        GraphMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=_graph(),
            bindings={
                "hls_pre": HLSPhysicalBinding("hls_pre", pre, "scale2_hls"),
                "vhdl_mid": VHDLPhysicalBinding("vhdl_mid", vhdl),
                "hls_post": HLSPhysicalBinding("hls_post", post, "add1_hls"),
            },
        )
    )
    assert result.ok, result.issues
    wrapper = result.wrapper.read_text(encoding="utf-8")
    report = result.report_path.read_text(encoding="utf-8")
    assert wrapper.index("scale2_hls") < wrapper.index("scale_bias_vhdl") < wrapper.index("add1_hls")
    assert '"from_backend": "vitis_hls"' in report
    assert '"to_backend": "vhdl"' in report
    assert '"from_backend": "vhdl"' in report
    assert '"to_backend": "vitis_hls"' in report
    assert '"tensor": "a"' in report
    assert '"tensor": "b"' in report


def test_graph_physical_rejects_width_mismatch(tmp_path):
    pre = _fake_hls(tmp_path, "scale2_hls", "input_data << 1")
    post = _fake_hls(tmp_path, "add1_hls", "input_data + 1")
    graph = _graph()
    graph.tensors["a"].dtype = "int32"
    vhdl = implementation_contract_from_manifest(Path("examples/packages/scale_bias_vhdl"))
    result = emit_graph_mixed_backend_physical_project(
        GraphMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=graph,
            bindings={
                "hls_pre": HLSPhysicalBinding("hls_pre", pre, "scale2_hls"),
                "vhdl_mid": VHDLPhysicalBinding("vhdl_mid", vhdl),
                "hls_post": HLSPhysicalBinding("hls_post", post, "add1_hls"),
            },
        )
    )
    assert not result.ok
    assert result.issues[0].code == "MIXGRAPH005"
