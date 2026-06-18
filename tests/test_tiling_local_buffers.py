from __future__ import annotations

from fpgai.backends.hls.emit.conv_tiling_codegen import emit_conv_tiled_helper_cpp
from fpgai.backends.hls.emit.dense_tiling_codegen import emit_dense_tiled_helper_cpp


def test_dense_tiled_helper_uses_local_tile_buffers() -> None:
    source = emit_dense_tiled_helper_cpp()

    assert "IN_T input_tile[TILE_IN];" in source
    assert "W_T weight_tile[TILE_OUT][TILE_IN];" in source
    assert "ACC_T acc_tile[TILE_OUT];" in source
    assert "#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1" in source
    assert "#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1" in source
    assert "#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2" in source
    assert "dense_load_input_tile:" in source
    assert "dense_load_weight_tile_out:" in source
    assert "dense_compute_tile_out:" in source
    assert "dense_store_output_tile:" in source


def test_conv_tiled_helper_uses_local_tile_buffers() -> None:
    source = emit_conv_tiled_helper_cpp()

    assert "ACC_T acc_tile[TILE_OC][TILE_OH][TILE_OW];" in source
    assert "IN_T input_tile[TILE_IC][TILE_OH * STRIDE + K][TILE_OW * STRIDE + K];" in source
    assert "W_T weight_tile[TILE_OC][TILE_IC][K][K];" in source
    assert "#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1" in source
    assert "#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1" in source
    assert "#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1" in source
    assert "#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2" in source
    assert "conv_load_input_ic:" in source
    assert "conv_load_weight_oc:" in source
    assert "conv_compute_oc:" in source
    assert "conv_store_output_oc:" in source
