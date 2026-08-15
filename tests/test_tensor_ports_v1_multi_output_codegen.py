from pathlib import Path

from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.hls_composition import build_hls_composition_plan
from fpgai.ir.graph import Graph


def test_tensor_ports_v1_multi_output_external_node_is_emitted_inside_dag():
    graph = Graph("external_multi_output_dag")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    for name in ("input", "identity", "scaled", "summed", "out"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("SplitScale", ["input"], ["identity", "scaled"], "split_scale_0", {
        "scale": 2.0,
        "_fpgai_external_operator": {
            "operator_id": "community.operator.split_scale",
            "package_id": "community.split_scale_operator",
            "package_version": "1.0.0",
        },
    })
    graph.add_op("Add", ["identity", "scaled"], ["summed"], "add_0")
    graph.add_op("Relu", ["summed"], ["out"], "relu_0")
    contract = implementation_contract_from_manifest(Path("examples/packages/split_scale_tensor_ports_hls"))
    plan = build_hls_composition_plan(graph, selected_contracts={"split_scale_0": contract})
    raw_cfg = {"numerics": {"kind": "fixed", "defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}}
    allocation = build_hls_buffer_allocation(graph, raw_cfg=raw_cfg)
    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg=raw_cfg,
        external_composition_plan=plan,
        buffer_allocation=allocation,
    )
    assert "split_scale_tensor_ports_hls(" in source
    assert "identity" in source
    assert "scaled" in source
    assert "add_vec_typed" in source
