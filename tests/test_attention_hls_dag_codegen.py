import numpy as np

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.ir.graph import Graph
from fpgai.ir.passes.attention_lowering import plan_attention_lowering


def _cfg():
    return {
        "numerics": {
            "defaults": {
                "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12},
            }
        },
        "targets": {"hls": {"control_protocol": "s_axilite"}},
    }


def _attention_graph() -> Graph:
    g = Graph("attention_core")
    g.inputs = ["q", "k", "v"]
    g.outputs = ["context"]
    for name, shape in {
        "q": (1, 4, 8),
        "k": (1, 4, 8),
        "v": (1, 4, 8),
        "kt": (1, 8, 4),
        "scores": (1, 4, 4),
        "scaled": (1, 4, 4),
        "probs": (1, 4, 4),
        "context": (1, 4, 8),
    }.items():
        g.add_tensor(name, shape, "float32")
    g.add_tensor("scale", (), "float32")
    g.constants["scale"] = np.asarray([1.0 / np.sqrt(8.0)], dtype=np.float32)
    g.add_op("Transpose", ["k"], ["kt"], name="transpose_k", attrs={"perm": [0, 2, 1]})
    g.add_op("MatMul", ["q", "kt"], ["scores"], name="score_matmul")
    g.add_op("Mul", ["scores", "scale"], ["scaled"], name="scale_scores")
    g.add_op("Softmax", ["scaled"], ["probs"], name="attention_softmax", attrs={"axis": -1})
    g.add_op("MatMul", ["probs", "v"], ["context"], name="value_matmul")
    plan_attention_lowering(g, tile_m=2, tile_n=2, tile_k=4)
    return g


def test_attention_core_uses_branch_aware_hls_kernels_and_multi_input_axis_segments():
    source = emit_dag_top_cpp(
        _attention_graph(),
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg=_cfg(),
    )
    assert "#include \"layers/attention.h\"" in source
    assert "FPGAI_INPUT_SEGMENT tensor=q words=32" in source
    assert "FPGAI_INPUT_SEGMENT tensor=k words=32" in source
    assert "FPGAI_INPUT_SEGMENT tensor=v words=32" in source
    assert "transpose_2d<4, 8" in source
    assert "matmul_tiled<4, 8, 4" in source
    assert "matmul_tiled<4, 4, 8" in source
    assert "softmax_rows<4, 4" in source
    assert "scale_vector<16" in source
    assert ", 2, 2, 4>(" in source
