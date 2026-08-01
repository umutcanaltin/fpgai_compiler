from pathlib import Path

from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
from fpgai.implementations import build_hls_composition_plan, implementation_contract_from_manifest
from fpgai.ir.graph import Graph


def test_normal_top_emitter_calls_selected_external_package():
    graph = Graph("mixed")
    graph.inputs = ["x"]
    graph.outputs = ["z"]
    for name in ("x", "y", "z"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["x"], ["y"], "relu_0")
    graph.add_op("ScaleBias", ["y"], ["z"], "scale_bias_0", {
        "scale": 2.0, "bias": 1.0,
        "_fpgai_external_operator": {"operator_id": "community.operator.scale_bias", "package_id": "community.scale_bias_operator"},
    })
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    plan = build_hls_composition_plan(graph, selected_contracts={"scale_bias_0": contract})
    source = emit_top_cpp(graph, top_name="deeplearn", weights_mode="embedded", external_composition_plan=plan)
    assert "void scale_bias_hls(" in source
    assert "scale_bias_hls(fpgai_external_scale_bias_0_input, fpgai_external_scale_bias_0_output, 4, 2.0f, 1.0f);" in source
    assert "layer_1_out[i] = (op1_act_t)fpgai_external_scale_bias_0_output[i];" in source
    assert "Unsupported mixed-precision HLS op: ScaleBias" not in source
