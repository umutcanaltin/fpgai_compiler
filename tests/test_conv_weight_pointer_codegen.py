from types import SimpleNamespace
import numpy as np

from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.ir.graph import Graph, Op, TensorSpec


def _cfg():
    return {"numerics": {"kind": "fixed", "defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}}


def test_dag_conv_passes_embedded_weights_as_flat_pointer():
    g = Graph(name="conv")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.tensors["x"] = TensorSpec(name="x", shape=(1, 1, 4, 4), dtype="float32")
    g.tensors["y"] = TensorSpec(name="y", shape=(1, 1, 4, 4), dtype="float32")
    g.tensors["W"] = TensorSpec(name="W", shape=(1, 1, 3, 3), dtype="float32")
    g.tensors["B"] = TensorSpec(name="B", shape=(1,), dtype="float32")
    g.constants["W"] = np.ones((1, 1, 3, 3), dtype=np.float32)
    g.constants["B"] = np.zeros((1,), dtype=np.float32)
    g.ops = [Op("Conv", "conv", ["x", "W", "B"], ["y"], {"strides": [1, 1], "pads": [1, 1, 1, 1]})]
    alloc = build_hls_buffer_allocation(g, raw_cfg=_cfg())
    src = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg(), buffer_allocation=alloc)
    assert "reinterpret_cast<const op0_wgt_t*>(W0), B0" in src
