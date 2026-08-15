from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.ir.graph import Graph


def test_branch_aware_top_uses_existing_add_kernel_and_liveness_buffers():
    graph = Graph("residual")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    for name in ("input", "left", "right", "out"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["left"], name="relu0")
    graph.add_op("Sigmoid", ["input"], ["right"], name="sigmoid0")
    graph.add_op("Add", ["left", "right"], ["out"], name="add0")

    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg={"numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}},
    )

    assert "add_vec_typed<4" in source
    assert "FPGAI_BUFFER_PROVENANCE" in source
    assert "fpgai_buffer_" in source
    assert "General graph Add requires" not in source
