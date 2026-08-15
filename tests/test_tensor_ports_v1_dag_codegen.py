from pathlib import Path
from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.hls_composition import build_hls_composition_plan
from fpgai.ir.graph import Graph


def test_tensor_ports_v1_external_add_is_emitted_inside_dag():
    graph = Graph("external_add_dag")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    for name in ("input", "relu", "identity", "out"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["relu"], "relu_0")
    graph.add_op("Identity", ["input"], ["identity"], "identity_0")
    graph.add_op("Add", ["relu", "identity"], ["out"], "external_add", {
        "_fpgai_external_operator": {
            "operator_id": "fpgai.operator.add",
            "package_id": "community.add_operator",
            "package_version": "1.0.0",
        }
    })
    contract = implementation_contract_from_manifest(Path("examples/packages/add_tensor_ports_hls"))
    plan = build_hls_composition_plan(graph, selected_contracts={"external_add": contract})
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
    assert "External tensor_ports_v1 implementation" in source
    assert "add_tensor_ports_hls(" in source
    assert "relu" in source
    assert "identity" in source
