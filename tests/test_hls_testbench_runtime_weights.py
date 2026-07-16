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
    assert "ap_uint<32>* weights_mem" in tb
    assert "int mode" in tb
    assert "fpgai::fpgai_runtime_weight_word_count()" in tb
    assert "fpgai::fpgai_fill_runtime_weight_words(weights_mem.data(), actual_weight_words)" in tb
    assert "deeplearn(in_stream, out_stream, weights_mem.data(), 1);" in tb
    assert "deeplearn(in_stream, out_stream, weights_mem.data(), 0);" in tb
    assert tb.index("weights_mem.data(), 1)") < tb.index("weights_mem.data(), 0)")
    assert "deeplearn(in_stream, out_stream, weights_mem.data());" not in tb


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
    assert "std::vector<ap_uint<32> > input_mem((size_t)(input_words_per_record > 0 ? input_words_per_record : 1));" in tb
    assert "std::vector<ap_uint<32> > label_mem((size_t)(target_words_per_record > 0 ? target_words_per_record : 1));" in tb
    assert "std::vector<ap_uint<32> > output_mem((size_t)(target_words_per_record > 0 ? target_words_per_record : 1));" in tb
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


def test_inference_runtime_weights_with_m_axi_io_uses_full_resolved_abi(tmp_path: Path) -> None:
    emit_tb_cpp(
        tmp_path,
        top_name="deeplearn",
        in_words=784,
        out_words=10,
        weights_mode="ddr",
        weight_words=101770,
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
    assert "constap_uint<32>*input_mem,ap_uint<32>*output_mem,ap_uint<32>*weights_mem,intmode" in signature
    assert "std::vector<ap_uint<32> > input_mem(784);" in tb
    assert "std::vector<ap_uint<32> > output_mem(10);" in tb
    assert "deeplearn(input_mem.data(), output_mem.data(), weights_mem.data(), 1);" in tb
    assert "deeplearn(input_mem.data(), output_mem.data(), weights_mem.data(), 0);" in tb
    assert tb.index("weights_mem.data(), 1)") < tb.index("weights_mem.data(), 0)")
    assert "hls::stream<axis_t> in_stream;" not in tb
    assert "hls::stream<axis_t> out_stream;" not in tb
    assert "output_data.reserve(requested_samples * FPGAI_OUTPUT_VALUES);" in tb
    assert "expected %d outputs, received %zu" in tb


def test_inference_runtime_weight_m_axi_testbench_escapes_c_string_newlines(tmp_path: Path) -> None:
    emit_tb_cpp(
        tmp_path,
        top_name="deeplearn",
        in_words=784,
        out_words=10,
        weights_mode="ddr",
        weight_words=101770,
        raw_cfg={
            "data_movement": {
                "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False}},
                "outputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": False}},
            },
            "numerics": {"defaults": {"activation": {"total_bits": 16, "int_bits": 6}}},
        },
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert 'weights_mode=runtime_import\\n", 1, 1);' in tb
    assert 'DDR buffer...\\n", actual_weight_words);' in tb
    assert 'Importing runtime weights...\\n");' in tb
    assert 'Running inference sample %d/%d...\\n"' in tb
    assert 'Received %zu outputs across %d samples.\\n"' in tb
    assert 'Could not open output file for writing: %s\\n", out_path);' in tb
    assert 'printf("[TB] Importing runtime weights...\n");' not in tb


def test_inference_runtime_weight_dataset_batch_imports_once_and_runs_each_sample(tmp_path: Path) -> None:
    emit_tb_cpp(
        tmp_path,
        top_name="deeplearn",
        in_words=4,
        out_words=2,
        weights_mode="ddr",
        weight_words=7,
        sample_count=3,
        raw_cfg={
            "runtime": {"sequence": ["import_weights", "run_inference", "export_weights"]},
            "data_movement": {
                "inputs": {"interface": "m_axi", "policy": "full"},
                "outputs": {"interface": "m_axi", "policy": "full"},
            },
        },
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert "const int requested_samples = 3;" in tb
    assert "for (int sample_index = 0; sample_index < requested_samples; ++sample_index)" in tb
    assert tb.count("weights_mem.data(), 1);") == 1
    assert "weights_mem.data(), 0);" in tb
    assert tb.count("weights_mem.data(), 2);") == 1
    assert "output_data.reserve(requested_samples * FPGAI_OUTPUT_VALUES);" in tb
    assert 'sample_count_requested' in tb
    assert 'weight_import_count' in tb
    assert 'weight_export_count' in tb
    assert 'inference_invocation_count' in tb


def test_inference_runtime_weight_dataset_batch_rejects_misaligned_input_count(tmp_path: Path) -> None:
    emit_tb_cpp(
        tmp_path,
        top_name="deeplearn",
        in_words=4,
        out_words=2,
        weights_mode="ddr",
        weight_words=7,
        sample_count=3,
        raw_cfg={
            "data_movement": {
                "inputs": {"interface": "m_axi", "policy": "full"},
                "outputs": {"interface": "m_axi", "policy": "full"},
            }
        },
    )
    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert "n_floats != requested_samples * sample_words" in tb
    assert "return 5;" in tb


def test_training_multi_epoch_accuracy_codegen_expands_all_template_tokens(tmp_path: Path) -> None:
    emit_tb_train_cpp(
        tmp_path,
        graph=None,
        top_name="deeplearn_train",
        in_words=4,
        out_words=2,
        weights_mode="embedded",
        weight_words=3,
        preload_weights=[0.1, 0.2, 0.3],
        training_cfg={
            "batch": {"size": 1, "epochs": 2, "mode": "accumulated"},
            "validation": {"convergence_smoke": True, "loss_eval_records": 2},
        },
        output_dir=str(tmp_path),
        raw_cfg={},
    )

    tb = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert "auto evaluate_accuracy" in tb
    assert "deeplearn_train(in_stream, out_stream, aux_stream, 0);" in tb
    assert 'drain_exact(out_stream, 2, "accuracy_eval")' in tb
    for unresolved in [
        "{eval_record_mem_pack}",
        "{top_name}",
        "{movement_call_args}",
        "{int(out_words)}",
    ]:
        assert unresolved not in tb


def test_training_testbench_emits_pre_post_held_out_evaluation(tmp_path: Path) -> None:
    emit_tb_train_cpp(
        tmp_path,
        graph=None,
        top_name="deeplearn",
        in_words=4,
        out_words=3,
        weights_mode="embedded",
        weight_words=6,
        preload_weights=[0.0] * 6,
        training_cfg={"batch": {"size": 1, "epochs": 1}},
        raw_cfg={"training": {"batch": {"size": 1, "epochs": 1}}},
        dataset_sample_count=2,
        held_out_sample_count=3,
    )
    text = (tmp_path / "tb.cpp").read_text(encoding="utf-8")
    assert "held_out_input_path = argc >= 6" in text
    assert "HeldOutMetrics held_out_before = evaluate_held_out(\"before\")" in text
    assert "HeldOutMetrics held_out_after = evaluate_held_out(\"after\")" in text
    assert "held_out_validation_summary.json" in text
    assert "held_out_predictions_before.csv" in text
    assert "held_out_predictions_after.csv" in text
    assert "held_out_curve.csv" in text
    assert "deeplearn(in_stream, out_stream, aux_stream, 0);" in text
