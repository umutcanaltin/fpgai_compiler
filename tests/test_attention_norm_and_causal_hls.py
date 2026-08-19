from __future__ import annotations

import numpy as np

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.benchmark.graph_reference import execute_graph_reference
from fpgai.ir import Graph
from fpgai.ir.passes.attention_lowering import plan_attention_lowering


def _cfg():
    return {"numerics": {"defaults": {
        "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "bias": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12},
    }}, "targets": {"hls": {"control_protocol": "s_axilite"}}}


def test_rmsnorm_and_layernorm_emit_real_hls_kernels():
    g = Graph("norms")
    for name, shape in {"x": (1, 2, 4), "scale": (4,), "bias": (4,), "rms": (1, 2, 4), "y": (1, 2, 4)}.items():
        g.add_tensor(name, shape, "float32")
    g.inputs = ["x", "scale", "bias"]
    g.outputs = ["y"]
    g.add_op("RMSNorm", ["x", "scale"], ["rms"], name="rms", attrs={"axis": -1, "epsilon": 1e-5})
    g.add_op("LayerNormalization", ["rms", "scale", "bias"], ["y"], name="ln", attrs={"axis": -1, "epsilon": 1e-5})
    cpp = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "rms_norm_rows<" in cpp
    assert "layer_norm_rows<" in cpp


def test_causal_mask_is_part_of_attention_plan_and_reference():
    g = Graph("causal_attention")
    for name, shape in {"q": (1, 4, 8), "kt": (1, 8, 4), "v": (1, 4, 8), "scores": (1, 4, 4), "masked": (1, 4, 4), "probs": (1, 4, 4), "out": (1, 4, 8)}.items():
        g.add_tensor(name, shape, "float32")
    g.inputs = ["q", "kt", "v"]; g.outputs = ["out"]
    g.add_op("MatMul", ["q", "kt"], ["scores"], name="scores")
    g.add_op("CausalMask", ["scores"], ["masked"], name="causal", attrs={"diagonal": 0, "masked_value": -32.0})
    g.add_op("Softmax", ["masked"], ["probs"], name="softmax", attrs={"axis": -1})
    g.add_op("MatMul", ["probs", "v"], ["out"], name="values")
    plans = plan_attention_lowering(g, tile_m=2, tile_n=2, tile_k=4)
    assert len(plans) == 1 and plans[0].causal is True and plans[0].mask_op == "causal"
    cpp = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "causal_mask_rows<" in cpp
    inputs = {"q": np.ones((1,4,8), np.float32), "kt": np.ones((1,8,4), np.float32), "v": np.arange(32, dtype=np.float32).reshape(1,4,8)}
    out = execute_graph_reference(g, inputs)
    assert out.shape == (1,4,8)
    assert np.isfinite(out).all()
