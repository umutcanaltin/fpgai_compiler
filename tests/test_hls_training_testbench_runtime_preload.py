from pathlib import Path

from fpgai.backends.hls.testbench_train import emit_tb_train_cpp


def _emit(tmp_path: Path, mode: str) -> str:
    emit_tb_train_cpp(
        tmp_path,
        graph=None,  # emit_tb_train_cpp does not inspect graph today.
        top_name="deeplearn",
        in_words=3072,
        out_words=10,
        weights_mode=mode,
        weight_words=16,
        preload_weights=[0.0] * 16,
        training_cfg={"execution": {"train_steps": 1, "batch_size": 1}},
        output_dir=str(tmp_path),
        raw_cfg={},
    )
    return (tmp_path / "tb.cpp").read_text(encoding="utf-8")


def test_training_ddr_tiled_preloads_m_axi_weights_without_mode0_aux_call(tmp_path: Path) -> None:
    tb = _emit(tmp_path, "ddr_tiled")

    assert "ap_uint<32>* weights_mem" in tb
    assert "std::vector<ap_uint<32> > weights_mem(16);" in tb
    assert "weights_mem[i] = u.i;" in tb
    assert "deeplearn(in_stream, out_stream, aux_stream, weights_mem.data(), 0);" not in tb
    assert "deeplearn(in_stream, out_stream, aux_stream, weights_mem.data(), 1);" in tb
    assert "deeplearn(in_stream, out_stream, aux_stream, weights_mem.data(), 2);" in tb


def test_training_stream_preload_still_uses_mode0_aux_stream(tmp_path: Path) -> None:
    tb = _emit(tmp_path, "stream")

    assert "ap_uint<32>* weights_mem" not in tb
    assert "push_f32(aux_stream, preload[i], i + 1 == preload.size());" in tb
    assert "deeplearn(in_stream, out_stream, aux_stream, 0);" in tb
