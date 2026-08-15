from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.ir.graph import Graph
from fpgai.ir.liveness import analyze_tensor_liveness


def _graph():
    graph = Graph("residual")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    for name in ("input", "left", "right", "sum", "out"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["left"], name="relu0")
    graph.add_op("Sigmoid", ["input"], ["right"], name="sigmoid0")
    graph.add_op("Add", ["left", "right"], ["sum"], name="add0")
    graph.add_op("Sigmoid", ["sum"], ["out"], name="sigmoid1")
    return graph


def test_liveness_buffer_allocator_reuses_only_dead_compatible_ranges():
    graph = _graph()
    liveness = analyze_tensor_liveness(graph)
    allocation = build_hls_buffer_allocation(
        graph,
        raw_cfg={"numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}},
        tensor_liveness=liveness,
    )

    assert liveness["has_branching"] is True
    assert allocation["mode"] == "liveness"
    assert allocation["slot_count"] >= 3
    assert allocation["tensor_to_buffer"]["left"] != allocation["tensor_to_buffer"]["right"]
    assert allocation["tensor_to_buffer"]["input"] != allocation["tensor_to_buffer"]["left"]
    assert set(allocation["tensor_to_buffer"]) == {"input", "left", "right", "sum", "out"}
