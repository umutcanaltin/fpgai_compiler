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


def test_dag_conv_runtime_import_export_uses_shared_weight_command_abi():
    g = Graph(name="conv_runtime_weights")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.tensors["x"] = TensorSpec(name="x", shape=(1, 1, 4, 4), dtype="float32")
    g.tensors["y"] = TensorSpec(name="y", shape=(1, 1, 4, 4), dtype="float32")
    g.tensors["W"] = TensorSpec(name="W", shape=(1, 1, 3, 3), dtype="float32")
    g.tensors["B"] = TensorSpec(name="B", shape=(1,), dtype="float32")
    g.constants["W"] = np.ones((1, 1, 3, 3), dtype=np.float32)
    g.constants["B"] = np.zeros((1,), dtype=np.float32)
    g.ops = [Op("Conv", "conv", ["x", "W", "B"], ["y"], {"strides": [1, 1], "pads": [1, 1, 1, 1]})]
    cfg = _cfg()
    cfg["weights"] = {"mode": "import_export"}
    cfg["memory"] = {"storage": {"weights": "bram"}}

    src = emit_dag_top_cpp(
        g,
        top_name="deeplearn",
        weights_mode="ddr",
        raw_cfg=cfg,
        buffer_allocation=build_hls_buffer_allocation(g, raw_cfg=cfg),
    )

    assert "ap_uint<32>* weights_mem" in src
    assert "int mode" in src
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in src
    assert "FPGAI_MODE_IMPORT_WEIGHTS = 1" in src
    assert "FPGAI_MODE_EXPORT_WEIGHTS = 2" in src
    assert "static op0_wgt_t W0[9]" in src
    assert "impl=bram" in src
    assert "fpgai_load_ddr_vector<op0_wgt_t, 9>" in src
    assert "fpgai_store_ddr_vector<op0_wgt_t, 9>" in src
    assert "reinterpret_cast<const op0_wgt_t*>(W0), B0" in src


def test_dag_runtime_weights_and_m_axi_io_share_one_top_signature():
    g = Graph(name="conv_runtime_weights_m_axi_io")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.tensors["x"] = TensorSpec(name="x", shape=(1, 1, 4, 4), dtype="float32")
    g.tensors["y"] = TensorSpec(name="y", shape=(1, 1, 4, 4), dtype="float32")
    g.tensors["W"] = TensorSpec(name="W", shape=(1, 1, 3, 3), dtype="float32")
    g.tensors["B"] = TensorSpec(name="B", shape=(1,), dtype="float32")
    g.constants["W"] = np.ones((1, 1, 3, 3), dtype=np.float32)
    g.constants["B"] = np.zeros((1,), dtype=np.float32)
    g.ops = [Op("Conv", "conv", ["x", "W", "B"], ["y"], {"strides": [1, 1], "pads": [1, 1, 1, 1]})]
    cfg = _cfg()
    cfg["weights"] = {"mode": "import_export"}
    cfg["memory"] = {"storage": {"weights": "bram"}}
    cfg["data_movement"] = {
        "inputs": {"interface": "m_axi", "transport": "ps_runtime"},
        "outputs": {"interface": "m_axi", "transport": "ps_runtime"},
    }

    src = emit_dag_top_cpp(
        g,
        top_name="deeplearn",
        weights_mode="ddr",
        raw_cfg=cfg,
        buffer_allocation=build_hls_buffer_allocation(g, raw_cfg=cfg),
    )

    assert "const ap_uint<32>* input_mem" in src
    assert "ap_uint<32>* output_mem" in src
    assert "ap_uint<32>* weights_mem" in src
    assert "int mode" in src
    assert "#pragma HLS INTERFACE m_axi port=input_mem" in src
    assert "#pragma HLS INTERFACE m_axi port=output_mem" in src
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in src
    assert "FPGAI_MODE_IMPORT_WEIGHTS = 1" in src
    assert "FPGAI_MODE_EXPORT_WEIGHTS = 2" in src
