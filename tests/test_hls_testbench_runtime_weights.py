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



def test_training_gradient_export_testbench_declares_and_passes_gradient_buffer(tmp_path: Path) -> None:
    emit_tb_train_cpp(
        tmp_path,
        graph=None,
        top_name="deeplearn_train",
        in_words=4,
        out_words=2,
        weights_mode="ddr_tiled_mutable",
        weight_words=5,
        preload_weights=[0.1, 0.2, 0.3, 0.4, 0.5],
        training_cfg={"execution": {"train_steps": 1, "batch_size": 1}},
        output_dir=str(tmp_path),
        raw_cfg={
            "data_movement": {
                "weights": {"import": {"interface": "m_axi", "policy": "tiled"}},
                "gradients": {"export": {"interface": "m_axi", "policy": "tiled"}},
            }
        },
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* weights_mem" in tb
    assert "ap_uint<32>* gradients_mem" in tb
    assert "std::vector<ap_uint<32> > weights_mem(5);" in tb
    assert "std::vector<ap_uint<32> > gradients_mem(5);" in tb
    assert "weights_mem.data(), gradients_mem.data(), 1)" in tb
    assert "weights_mem.data(), gradients_mem.data(), 2)" in tb
    assert "weights_mem.data(), gradients_mem.data(), 8)" in tb
    assert "gradients_after.bin" in tb


def test_training_m_axi_tensor_edge_testbench_signature_matches_generated_top_order(tmp_path: Path) -> None:
    emit_tb_train_cpp(
        tmp_path,
        graph=None,
        top_name="deeplearn_train",
        in_words=8,
        out_words=2,
        weights_mode="ddr_tiled_mutable",
        weight_words=5,
        preload_weights=[0.1, 0.2, 0.3, 0.4, 0.5],
        training_cfg={"execution": {"train_steps": 1, "batch_size": 1}},
        output_dir=str(tmp_path),
        raw_cfg={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 64}},
                "labels": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 64}},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 64}},
                "weights": {
                    "import": {"interface": "m_axi", "policy": "tiled"},
                    "export": {"interface": "m_axi", "policy": "tiled"},
                },
                "gradients": {"export": {"interface": "m_axi", "policy": "tiled"}},
            }
        },
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    signature = "".join(tb[tb.index('extern "C" void deeplearn_train('):tb.index(');', tb.index('extern "C" void deeplearn_train('))].split())
    assert "ap_uint<32>*weights_mem,ap_uint<32>*input_mem,ap_uint<32>*label_mem,ap_uint<32>*output_mem,ap_uint<32>*gradients_mem,intmode" in signature
    assert "std::vector<ap_uint<32> > input_mem(8);" in tb
    assert "std::vector<ap_uint<32> > label_mem(2);" in tb
    assert "std::vector<ap_uint<32> > output_mem(2);" in tb
    assert "deeplearn_train(in_stream, out_stream, aux_stream, weights_mem.data(), input_mem.data(), label_mem.data(), output_mem.data(), gradients_mem.data(), 1)" in tb
    assert "deeplearn_train(in_stream, out_stream, aux_stream, weights_mem.data(), input_mem.data(), label_mem.data(), output_mem.data(), gradients_mem.data(), 2)" in tb
    assert "deeplearn_train(in_stream, out_stream, aux_stream, weights_mem.data(), input_mem.data(), label_mem.data(), output_mem.data(), gradients_mem.data(), 8)" in tb


def test_inference_m_axi_input_output_testbench_matches_generated_top_abi(tmp_path: Path) -> None:
    emit_tb_cpp(
        tmp_path,
        top_name="deeplearn",
        in_words=8,
        out_words=2,
        weights_mode="embedded",
        raw_cfg={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False}},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False}},
            },
            "numerics": {"defaults": {"activation": {"total_bits": 16, "int_bits": 6}}},
        },
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    signature = "".join(tb[tb.index('extern "C" void deeplearn('):tb.index(');', tb.index('extern "C" void deeplearn('))].split())
    assert "constap_uint<32>*input_mem,ap_uint<32>*output_mem" in signature
    assert "std::vector<ap_uint<32> > input_mem(8);" in tb
    assert "std::vector<ap_uint<32> > output_mem(2);" in tb
    assert "input_mem[index] = fpgai_float_to_bits(value);" in tb
    assert "output_data.push_back(fpgai_bits_to_float(output_mem[index].to_uint()));" in tb
    assert "deeplearn(input_mem.data(), output_mem.data());" in tb
    assert "extern \"C\" void deeplearn(hls::stream<axis_t>& in, hls::stream<axis_t>& out);" not in tb


def test_inference_m_axi_testbench_escapes_c_string_newlines(tmp_path: Path) -> None:
    emit_tb_cpp(
        tmp_path,
        top_name="deeplearn",
        in_words=8,
        out_words=2,
        weights_mode="embedded",
        raw_cfg={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False}},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False}},
            },
            "numerics": {"defaults": {"activation": {"total_bits": 16, "int_bits": 6}}},
        },
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert 'Could not open input file: %s\\n", in_path);' in tb
    assert 'Loaded %d inputs from %s\\n", n_floats, in_path);' in tb
    assert 'Running inference...\\n");' in tb
    assert 'Received %zu outputs.\\n", output_data.size());' in tb
    assert 'Could not open input file: %s\n", in_path);' not in tb
    assert 'printf("[TB] Running inference...\n");' not in tb
