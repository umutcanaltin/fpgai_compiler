from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.ir import Graph, Op


def _graph():
    g = Graph("relu_graph")
    g.add_tensor("x", (1, 4))
    g.add_tensor("y", (1, 4))
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_op("Relu", ["x"], ["y"], name="relu0")
    return g


def test_network_dataflow_emits_real_hls_pragma():
    cpp = emit_dag_top_cpp(
        _graph(),
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg={"architecture": {"network": {"execution": {"mode": "dataflow"}}}},
    )
    assert "#pragma HLS DATAFLOW" in cpp
    assert "FPGAI_NETWORK_EXECUTION mode=dataflow physical=pragma" in cpp


def test_network_sequential_does_not_emit_dataflow_pragma():
    cpp = emit_dag_top_cpp(
        _graph(),
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg={"architecture": {"network": {"execution": {"mode": "sequential"}}}},
    )
    assert "#pragma HLS DATAFLOW" not in cpp
    assert "FPGAI_NETWORK_EXECUTION mode=sequential physical=implemented" in cpp
