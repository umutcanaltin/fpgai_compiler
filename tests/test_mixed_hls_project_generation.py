from pathlib import Path

from fpgai.backends.hls.codegen import emit_hls_stub
from fpgai.implementations import build_hls_composition_plan, implementation_contract_from_manifest
from fpgai.ir.graph import Graph


def test_normal_hls_project_contains_external_source_and_composition_metadata(tmp_path):
    graph = Graph("mixed")
    graph.inputs = ["x"]
    graph.outputs = ["z"]
    for name in ("x", "y", "z"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["x"], ["y"], "relu_0")
    graph.add_op("ScaleBias", ["y"], ["z"], "scale_bias_0", {
        "_fpgai_external_operator": {"operator_id": "community.operator.scale_bias", "package_id": "community.scale_bias_operator"},
    })
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    plan = build_hls_composition_plan(graph, selected_contracts={"scale_bias_0": contract})
    project = emit_hls_stub(
        graph=graph, out_dir=tmp_path, top_name="deeplearn",
        hls_options={"pipeline_mode": "inference", "weights_mode": "embedded", "run_csim": False, "run_csynth": False},
        external_composition_plan=plan,
    )
    assert list((project.hls_dir / "src" / "external").rglob("*.cpp"))
    assert "scale_bias_hls" in project.top_cpp.read_text()
    assert "src/external" in project.run_tcl.read_text()
    assert '"external_composition_present": true' in (project.hls_dir / "codegen_meta.json").read_text()
