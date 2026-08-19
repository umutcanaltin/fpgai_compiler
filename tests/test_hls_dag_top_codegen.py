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


def test_branch_aware_top_honors_m_axi_input_output_contract():
    graph = Graph("residual_m_axi")
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
        raw_cfg={
            "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}},
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False}},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False}},
            },
        },
    )

    assert "const ap_uint<32>* input_mem" in source
    assert "ap_uint<32>* output_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=input_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=output_mem" in source
    assert "m_axi full input import" in source
    assert "m_axi full output export" in source
    assert "axis port=in_stream" not in source
    assert "axis port=out_stream" not in source


def test_branch_aware_top_applies_readability_contract_and_tiled_axis_marker():
    graph = Graph("readability_tiled")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    for name in ("input", "left", "right", "out"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["left"], name="relu0")
    graph.add_op("Sigmoid", ["input"], ["right"], name="sigmoid0")
    graph.add_op("Add", ["left", "right"], ["out"], name="add0")

    base = {
        "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}},
        "data_movement": {
            "inputs": {"import": {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}},
            "outputs": {"export": {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}},
        },
    }
    high_cfg = {**base, "codegen": {"readability": "high"}}
    compact_cfg = {**base, "codegen": {"readability": "compact"}}

    high = emit_dag_top_cpp(graph, top_name="deeplearn", weights_mode="embedded", raw_cfg=high_cfg)
    compact = emit_dag_top_cpp(graph, top_name="deeplearn", weights_mode="embedded", raw_cfg=compact_cfg)

    assert "FPGAI generated HLS top" in high[:512]
    assert "FPGAI generated HLS top" not in compact[:512]
    assert sum(line.strip().startswith("//") for line in high.splitlines()) > sum(
        line.strip().startswith("//") for line in compact.splitlines()
    )
    assert "FPGAI AXI-stream tiled input/output movement" in high
    assert "FPGAI_AXIS_INPUT_TILE_SIZE" in high
    assert "FPGAI_AXIS_OUTPUT_TILE_SIZE" in high
    assert "packet.last = ((tile_base + lane + 1)" in high


def test_branch_aware_top_reuses_generic_pool_kernel_for_maxpool():
    graph = Graph("maxpool_dag")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    graph.add_tensor("input", (1, 1, 4, 4), "float32")
    graph.add_tensor("out", (1, 1, 2, 2), "float32")
    graph.add_op(
        "MaxPool",
        ["input"],
        ["out"],
        name="pool0",
        attrs={"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]},
    )

    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg={"numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}},
    )

    assert '#include "layers/pool.h"' in source
    assert "maxpool2d_typed<4, 4, 1, 2, 2, 2, 2" in source


def test_branch_aware_spatial_flatten_restores_onnx_channel_major_order():
    graph = Graph("cnn_flatten_layout")
    graph.inputs = ["input"]
    graph.outputs = ["flat"]
    graph.add_tensor("input", (1, 2, 4, 4), "float32")
    graph.add_tensor("pooled", (1, 2, 2, 2), "float32")
    graph.add_tensor("flat", (1, 8), "float32")
    graph.add_op(
        "MaxPool",
        ["input"],
        ["pooled"],
        name="pool0",
        attrs={"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]},
    )
    graph.add_op("Flatten", ["pooled"], ["flat"], name="flatten0", attrs={"axis": 1})

    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg={"numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}},
    )

    assert "FPGAI layout bridge: internal HWC-flat -> ONNX NCHW flatten for Flatten" in source
    assert "const int source = (row * 2 + column) * 2 + channel;" in source
    assert "const int destination = (channel * 2 + row) * 2 + column;" in source


def test_branch_aware_non_spatial_flatten_remains_plain_copy():
    graph = Graph("sequence_flatten")
    graph.inputs = ["input"]
    graph.outputs = ["flat"]
    graph.add_tensor("input", (1, 4, 8), "float32")
    graph.add_tensor("flat", (1, 32), "float32")
    graph.add_op("Identity", ["input"], ["identity"], name="id0")
    graph.add_tensor("identity", (1, 4, 8), "float32")
    # Rebuild op ordering after adding the tensor for Graph implementations that
    # require output tensors to exist before emission.
    graph.ops = []
    graph.add_op("Identity", ["input"], ["identity"], name="id0")
    graph.add_op("Flatten", ["identity"], ["flat"], name="flatten0", attrs={"axis": 1})

    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg={"numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}},
    )

    assert "FPGAI layout bridge" not in source
    assert "reshape_copy_typed<32" in source


def test_branch_aware_runtime_preload_emits_scalar_bit_helpers_before_runtime_loaders():
    import numpy as np

    graph = Graph("runtime_preload_dense")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    graph.add_tensor("input", (1, 4), "float32")
    graph.add_tensor("W", (3, 4), "float32")
    graph.add_tensor("B", (3,), "float32")
    graph.add_tensor("out", (1, 3), "float32")
    graph.constants["W"] = np.arange(12, dtype=np.float32).reshape(3, 4)
    graph.constants["B"] = np.zeros((3,), dtype=np.float32)
    graph.add_op("Dense", ["input", "W", "B"], ["out"], name="dense0")

    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="ddr",
        raw_cfg={
            "weights": {"mode": "import"},
            "memory": {"storage": {"weights": "bram"}},
            "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}},
        },
    )

    assert "static inline T bits_to_value(unsigned int bits)" in source
    assert "static inline unsigned int value_to_bits(T value)" in source
    assert "fpgai_load_ddr_vector" in source
    assert source.index("static inline T bits_to_value") < source.index("fpgai_load_ddr_vector")
    assert "FPGAI_MODE_IMPORT_WEIGHTS" in source


def test_branch_aware_ddr_tiled_reuses_shared_weight_helpers_with_hwc_conv_layout():
    import numpy as np

    graph = Graph("ddr_tiled_conv")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    graph.add_tensor("input", (1, 1, 4, 4), "float32")
    graph.add_tensor("W", (2, 1, 3, 3), "float32")
    graph.add_tensor("B", (2,), "float32")
    graph.add_tensor("out", (1, 2, 2, 2), "float32")
    graph.constants["W"] = np.ones((2, 1, 3, 3), dtype=np.float32)
    graph.constants["B"] = np.zeros((2,), dtype=np.float32)
    graph.add_op(
        "Conv",
        ["input", "W", "B"],
        ["out"],
        name="conv0",
        attrs={"strides": [1, 1], "pads": [0, 0, 0, 0], "group": 1},
    )

    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="ddr_tiled",
        raw_cfg={"numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}},
    )

    assert "const ap_uint<32>* weights_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in source
    assert "conv2d_ddr_tiled<" in source
    assert "input[(ih * IN_W + iw) * IN_C + ic]" in source
    assert "output[(oh * OUT_W + ow) * OUT_C + oc]" in source
    assert "reinterpret_cast<const op0_wgt_t*>(W0)" not in source


def _embedded_dense_graph_for_static_storage():
    import numpy as np

    graph = Graph("embedded_static_dense")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    graph.add_tensor("input", (1, 4), "float32")
    graph.add_tensor("W", (3, 4), "float32")
    graph.add_tensor("B", (3,), "float32")
    graph.add_tensor("out", (1, 3), "float32")
    graph.constants["W"] = np.arange(12, dtype=np.float32).reshape(3, 4)
    graph.constants["B"] = np.zeros((3,), dtype=np.float32)
    graph.add_op("Dense", ["input", "W", "B"], ["out"], name="dense0")
    return graph


def test_branch_aware_embedded_bram_materializes_function_scope_weight_storage():
    graph = _embedded_dense_graph_for_static_storage()
    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg={
            "memory": {"storage": {"weights": "bram"}},
            "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}},
        },
    )

    assert "FPGAI bram_static weight storage" in source
    assert "static op0_wgt_t W0[12];" in source
    assert "#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=bram" in source
    assert "W0[i] = fpgai::W0[i];" in source
    assert "B0[i] = fpgai::B0[i];" in source
    assert "static bool fpgai_static_weights_initialized = false;" in source
    assert "static op0_wgt_t W0[12] = {" not in source


def test_branch_aware_embedded_uram_materializes_function_scope_weight_storage():
    graph = _embedded_dense_graph_for_static_storage()
    source = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg={
            "memory": {"storage": {"weights": "uram"}},
            "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}},
        },
    )

    assert "FPGAI uram_static weight storage" in source
    assert "static op0_wgt_t W0[12];" in source
    assert "#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=uram" in source
    assert "W0[i] = fpgai::W0[i];" in source
    assert "B0[i] = fpgai::B0[i];" in source
    assert "static bool fpgai_static_weights_initialized = false;" in source
    assert "static op0_wgt_t W0[12] = {" not in source


def test_branch_aware_add_compile_plan_materially_changes_hls_template_arguments():
    graph = Graph("add_arch")
    graph.inputs = ["left", "right"]
    graph.outputs = ["out"]
    for name in ("left", "right", "out"):
        graph.add_tensor(name, (1, 8), "float32")
    graph.add_op("Add", ["left", "right"], ["out"], name="add0")
    compile_plan = {"layer_plans": [{
        "node_name": "add0",
        "architecture": {
            "pipeline": {"ii": 2},
            "parallelism": {"pe": 4, "simd": 1, "unroll": {"element": 4}},
            "partitioning": {"factor": 1, "targets": {"input": 2, "output": 4}},
            "tiling": {"sizes": {}},
        },
    }]}
    source = emit_dag_top_cpp(
        graph, top_name="deeplearn", weights_mode="embedded", compile_plan=compile_plan,
        raw_cfg={"numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}},
    )
    assert "FPGAI_ARCH_EFFECT op=Add name=add0 pipeline_ii=2 element_unroll=4" in source
    assert ", 2, 4, 2, 4>(" in source


def test_branch_aware_rejects_explicit_architecture_controls_when_kernel_cannot_realize_them():
    graph = Graph("relu_arch_unsupported")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    graph.add_tensor("input", (1, 4), "float32")
    graph.add_tensor("out", (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["out"], name="relu0")
    compile_plan = {"layer_plans": [{
        "node_name": "relu0",
        "architecture": {
            "pipeline": {"ii": 2},
            "parallelism": {"pe": 2, "simd": 1, "unroll": {"element": 2}},
            "partitioning": {"factor": 1, "targets": {}},
            "tiling": {"sizes": {}},
        },
        "notes": {
            "architecture_control_sources": {
                "effective_request": {"pipeline": {"ii": 2}, "parallelism": {"pe": 2}}
            }
        },
    }]}
    import pytest
    with pytest.raises(RuntimeError, match="HLSDAG105"):
        emit_dag_top_cpp(
            graph, top_name="deeplearn", weights_mode="embedded", compile_plan=compile_plan,
            raw_cfg={"numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}}}},
        )
