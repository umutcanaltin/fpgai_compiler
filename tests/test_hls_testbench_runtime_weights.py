from pathlib import Path

from fpgai.backends.hls.testbench import emit_tb_cpp
from fpgai.backends.hls.testbench_train import emit_tb_train_cpp


def test_inference_ddr_tiled_testbench_passes_runtime_weight_buffer(tmp_path: Path) -> None:
    emit_tb_cpp(
        tmp_path,
        top_name="deeplearn",
        in_words=4,
        out_words=2,
        weights_mode="ddr_tiled",
        weight_words=7,
        raw_cfg={"numerics": {"defaults": {"activation": {"total_bits": 16, "int_bits": 6}}}},
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert '#include <ap_int.h>' in tb
    assert '#include "fpgai_params.h"' in tb
    assert "const ap_uint<32>* weights_mem" in tb
    assert "fpgai::fpgai_runtime_weight_word_count()" in tb
    assert "fpgai::fpgai_fill_runtime_weight_words(weights_mem.data(), actual_weight_words)" in tb
    assert "deeplearn(in_stream, out_stream, weights_mem.data())" in tb
    assert "deeplearn(in_stream, out_stream);" not in tb


def test_training_ddr_tiled_testbench_passes_runtime_weight_buffer(tmp_path: Path) -> None:
    emit_tb_train_cpp(
        tmp_path,
        graph=None,  # emit_tb_train_cpp currently does not inspect graph.
        top_name="deeplearn_train",
        in_words=4,
        out_words=2,
        weights_mode="ddr_tiled",
        weight_words=3,
        preload_weights=[0.1, 0.2, 0.3],
        training_cfg={"execution": {"train_steps": 1, "batch_size": 1}},
        output_dir=str(tmp_path),
        raw_cfg={"data_movement": {"ps_pl": {"weights": {"mode": "ddr_tiled"}}}},
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* weights_mem" in tb
    assert "std::vector<ap_uint<32> > weights_mem(3);" in tb
    assert "weights_mem[i] = u.i;" in tb
    assert "deeplearn_train(in_stream, out_stream, aux_stream, weights_mem.data(), 0)" not in tb
    assert "weights_mem.data(), 1)" in tb
    assert "weights_mem.data(), 2)" in tb
    assert "deeplearn_train(in_stream, out_stream, aux_stream, weights_mem.data(), 1)" in tb
    assert "push_f32(aux_stream, preload[i]" not in tb
