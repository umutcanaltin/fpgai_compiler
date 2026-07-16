from __future__ import annotations

import pytest

from fpgai.engine.compiler import _resolve_training_optimizer_loss_contract


def _raw(optimizer_type: str) -> dict:
    return {
        "training": {
            "optimizer": {
                "type": optimizer_type,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1.0e-8,
            },
            "storage": {
                "optimizer_state": "none" if optimizer_type == "sgd" else "bram",
            },
            "loss": {"type": "mse"},
        }
    }


def test_sgd_support_contract_is_end_to_end_multi_epoch() -> None:
    contract = _resolve_training_optimizer_loss_contract(_raw("sgd"))
    status = contract["optimizer"]["support_status"]
    assert contract["schema_version"] == 2
    assert status["generated_hls_update"] == "implemented"
    assert status["single_step_numeric_validation"] == "implemented"
    assert status["dataset_multi_epoch_reference"] == "implemented"
    assert status["end_to_end_multi_epoch_validation"] == "implemented"
    assert status["board_runtime_validation"] == "not_validated"


def test_momentum_support_contract_is_end_to_end_multi_epoch_hls_validated() -> None:
    contract = _resolve_training_optimizer_loss_contract(_raw("momentum"))
    status = contract["optimizer"]["support_status"]
    assert status["generated_hls_update"] == "implemented"
    assert status["single_step_reference"] == "implemented"
    assert status["single_step_numeric_validation"] == "implemented"
    assert status["dataset_multi_epoch_reference"] == "implemented"
    assert status["dataset_multi_epoch_hls"] == "implemented"
    assert status["end_to_end_multi_epoch_validation"] == "implemented"


def test_adam_support_contract_remains_single_step_only() -> None:
    contract = _resolve_training_optimizer_loss_contract(_raw("adam"))
    status = contract["optimizer"]["support_status"]
    assert status["generated_hls_update"] == "implemented"
    assert status["single_step_reference"] == "implemented"
    assert status["single_step_numeric_validation"] == "implemented"
    assert status["dataset_multi_epoch_reference"] == "not_implemented"
    assert status["dataset_multi_epoch_hls"] == "partial"
    assert status["end_to_end_multi_epoch_validation"] == "partial"


def test_stateful_optimizer_rejects_none_storage() -> None:
    raw = _raw("momentum")
    raw["training"]["storage"]["optimizer_state"] = "none"
    with pytest.raises(ValueError, match="requires persistent optimizer state"):
        _resolve_training_optimizer_loss_contract(raw)


@pytest.mark.parametrize(
    ("optimizer_type", "field", "value", "message"),
    [
        ("momentum", "momentum", 1.0, "0 <= momentum < 1"),
        ("adam", "beta1", 1.0, "0 <= beta1 < 1"),
        ("adam", "beta2", -0.1, "0 <= beta2 < 1"),
        ("adam", "epsilon", -1.0e-8, "epsilon must be positive"),
    ],
)
def test_optimizer_parameter_ranges(
    optimizer_type: str,
    field: str,
    value: float,
    message: str,
) -> None:
    raw = _raw(optimizer_type)
    raw["training"]["optimizer"][field] = value
    with pytest.raises(ValueError, match=message):
        _resolve_training_optimizer_loss_contract(raw)


def test_live_momentum_repair_replaces_tiled_updates_in_accumulated_and_direct_paths() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import (
        _fpgai_p3d_f4c_materialize_live_momentum_updates,
    )

    source = '''
using namespace fpgai;
static wgt_t W_dense0[2][2] = {{0}};
static bias_t B_dense0[2] = {0};
static grad_wgt_t dW_dense0[4];
static grad_bias_t dB_dense0[2];
static opt_t FPGAI_MOMENTUM_W_dense0[4];
static opt_t FPGAI_MOMENTUM_B_dense0[2];
extern "C" void deeplearn(int mode) {
  if (mode == FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS || mode == 4) {
    fpgai::sgd_update_wgt_tiled<4, wgt_t, grad_wgt_t, upd_t, acc_t, 1, 4>(W_dense0, dW_dense0, (upd_t)0.005f);
    fpgai::sgd_update_bias_tiled<2, bias_t, grad_bias_t, upd_t, acc_t, 1, 2>(B_dense0, dB_dense0, (upd_t)0.005f);
    return;
  }
  fpgai::sgd_update_wgt_tiled<4, wgt_t, grad_wgt_t, upd_t, acc_t, 2, 4>(W_dense0, dW_dense0, (upd_t)0.005f);
}
'''
    repaired = _fpgai_p3d_f4c_materialize_live_momentum_updates(
        source,
        raw_cfg={
            "training": {
                "optimizer": {
                    "type": "momentum",
                    "learning_rate": 0.005,
                    "momentum": 0.9,
                }
            }
        },
    )

    assert "fpgai::sgd_update_wgt_tiled<" not in repaired
    assert "fpgai::sgd_update_bias_tiled<" not in repaired
    assert repaired.count("FPGAI_MOMENTUM_W_dense0[i] =") == 2
    assert repaired.count("FPGAI_MOMENTUM_B_dense0[i] =") == 1
    assert "FPGAI live Momentum update-path repair" in repaired


def test_momentum_optimizer_state_export_uses_canonical_parameter_order() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import (
        _fpgai_insert_optimizer_state_export_capture,
    )

    source = '''
using namespace fpgai;
static wgt_t W_dense0[2][2] = {{0}};
static bias_t B_dense0[2] = {0};
static wgt_t W_dense1[1][3] = {{0}};
static bias_t B_dense1[1] = {0};
static opt_t FPGAI_MOMENTUM_B_dense0[2];
static opt_t FPGAI_MOMENTUM_B_dense1[1];
static opt_t FPGAI_MOMENTUM_W_dense0[4];
static opt_t FPGAI_MOMENTUM_W_dense1[3];
extern "C" void deeplearn(ap_uint<32>* optimizer_state_mem, int mode) {
  if (mode == FPGAI_MODE_RESET_ACCUMULATORS || mode == 5) { return; }
}
'''
    generated = _fpgai_insert_optimizer_state_export_capture(
        source,
        raw_cfg={
            "training": {
                "optimizer": {"type": "momentum"},
            },
            "data_movement": {
                "optimizer_state": {
                    "export": {"interface": "m_axi", "policy": "full"}
                }
            },
        },
    )

    comments = [
        "optimizer_state tensor FPGAI_MOMENTUM_W_dense0: offset_words=0, count_words=4",
        "optimizer_state tensor FPGAI_MOMENTUM_B_dense0: offset_words=4, count_words=2",
        "optimizer_state tensor FPGAI_MOMENTUM_W_dense1: offset_words=6, count_words=3",
        "optimizer_state tensor FPGAI_MOMENTUM_B_dense1: offset_words=9, count_words=1",
    ]
    positions = [generated.index(comment) for comment in comments]
    assert positions == sorted(positions)
    assert "Export order follows canonical parameter order" in generated


def test_momentum_optimizer_state_export_rewrites_existing_bias_first_block() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import (
        _fpgai_insert_optimizer_state_export_capture,
    )

    source = r'''
using namespace fpgai;
static wgt_t W_dense0[2][2] = {{0}};
static bias_t B_dense0[2] = {0};
static wgt_t W_dense1[1][3] = {{0}};
static bias_t B_dense1[1] = {0};
static opt_t FPGAI_MOMENTUM_B_dense0[2];
static opt_t FPGAI_MOMENTUM_B_dense1[1];
static opt_t FPGAI_MOMENTUM_W_dense0[4];
static opt_t FPGAI_MOMENTUM_W_dense1[3];
// FPGAI optimizer-state export/capture mode.
static const int FPGAI_MODE_EXPORT_OPTIMIZER_STATE = 9;
static ap_uint<32> fpgai_pack_optimizer_state_float32(float value) { return 0; }
extern "C" void deeplearn(ap_uint<32>* optimizer_state_mem, int mode) {
  if (mode == FPGAI_MODE_EXPORT_OPTIMIZER_STATE || mode == 9) {
    // optimizer_state tensor FPGAI_MOMENTUM_B_dense0: offset_words=0, count_words=2
    for (int i = 0; i < 2; ++i) { optimizer_state_mem[0 + i] = fpgai_pack_optimizer_state_float32((float)FPGAI_MOMENTUM_B_dense0[i]); }
    // optimizer_state tensor FPGAI_MOMENTUM_B_dense1: offset_words=2, count_words=1
    for (int i = 0; i < 1; ++i) { optimizer_state_mem[2 + i] = fpgai_pack_optimizer_state_float32((float)FPGAI_MOMENTUM_B_dense1[i]); }
    // optimizer_state tensor FPGAI_MOMENTUM_W_dense0: offset_words=3, count_words=4
    for (int i = 0; i < 4; ++i) { optimizer_state_mem[3 + i] = fpgai_pack_optimizer_state_float32((float)FPGAI_MOMENTUM_W_dense0[i]); }
    // optimizer_state tensor FPGAI_MOMENTUM_W_dense1: offset_words=7, count_words=3
    for (int i = 0; i < 3; ++i) { optimizer_state_mem[7 + i] = fpgai_pack_optimizer_state_float32((float)FPGAI_MOMENTUM_W_dense1[i]); }
    return;
  }
  if (mode == FPGAI_MODE_RESET_ACCUMULATORS || mode == 5) { return; }
}
'''
    generated = _fpgai_insert_optimizer_state_export_capture(
        source,
        raw_cfg={
            "training": {"optimizer": {"type": "momentum"}},
            "data_movement": {
                "optimizer_state": {
                    "export": {"interface": "m_axi", "policy": "full"}
                }
            },
        },
    )

    expected = [
        "optimizer_state tensor FPGAI_MOMENTUM_W_dense0: offset_words=0, count_words=4",
        "optimizer_state tensor FPGAI_MOMENTUM_B_dense0: offset_words=4, count_words=2",
        "optimizer_state tensor FPGAI_MOMENTUM_W_dense1: offset_words=6, count_words=3",
        "optimizer_state tensor FPGAI_MOMENTUM_B_dense1: offset_words=9, count_words=1",
    ]
    positions = [generated.index(item) for item in expected]
    assert positions == sorted(positions)
    assert "FPGAI_MOMENTUM_B_dense0: offset_words=0" not in generated
    assert generated.count("if (mode == FPGAI_MODE_EXPORT_OPTIMIZER_STATE || mode == 9)") == 1


def test_optimizer_state_comparison_accepts_every_word_within_one_lsb(tmp_path) -> None:
    import struct
    from fpgai.validation.numeric import _optimizer_state_validation_payload

    ref = tmp_path / "optimizer_state_after_ref.bin"
    got = tmp_path / "optimizer_state_after.bin"
    lsb = 1.0 / 65536.0
    ref.write_bytes(struct.pack("<4f", 0.0, 0.25, -0.5, 1.0))
    got.write_bytes(struct.pack("<4f", lsb, 0.25 - lsb, -0.5, 1.0 + lsb))

    payload = _optimizer_state_validation_payload(
        {
            "requested": True,
            "optimizer": "momentum",
            "layout": "canonical_parameter_order",
            "comparisons": {"packed_optimizer_state_after": {"ref": ref, "got": got}},
        },
        raw_config={
            "numerics": {
                "training": {
                    "optimizer_state": {"total_bits": 24, "int_bits": 8}
                }
            }
        },
    )

    comparison = payload["comparisons"]["packed_optimizer_state_after"]
    assert payload["status"] == "implemented"
    assert payload["implementation_status"] == "implemented"
    assert payload["passed"] is True
    assert comparison["all_words_within_one_lsb"] is True
    assert comparison["within_one_lsb"] == 4
    assert comparison["reference_words"] == comparison["hls_words"] == 4


def test_numeric_report_writes_dedicated_optimizer_state_artifacts(tmp_path) -> None:
    import json
    import struct
    from fpgai.validation.numeric import emit_numeric_validation_report

    ref = tmp_path / "optimizer_state_after_ref.bin"
    got = tmp_path / "optimizer_state_after.bin"
    lsb = 1.0 / 65536.0
    ref.write_bytes(struct.pack("<3f", 0.0, 0.5, -0.25))
    got.write_bytes(struct.pack("<3f", lsb, 0.5, -0.25 - lsb))

    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="training_on_device",
        source_generated=True,
        hls_ran=True,
        hls_ok=True,
        optimizer_state_artifacts={
            "requested": True,
            "optimizer": "momentum",
            "layout": "canonical_parameter_order",
            "layout_version": 1,
            "reference_domain": "hardware_domain_fixed_point",
            "comparisons": {
                "packed_optimizer_state_after": {"ref": ref, "got": got}
            },
        },
        raw_config={
            "numerics": {
                "training": {
                    "optimizer_state": {"total_bits": 24, "int_bits": 8}
                }
            }
        },
    )

    payload = json.loads(
        artifacts["optimizer_state_validation_json"].read_text(encoding="utf-8")
    )
    assert payload["status"] == "implemented"
    assert payload["passed"] is True
    assert artifacts["optimizer_state_validation_md"].exists()
    assert "All words within one LSB: `true`" in artifacts[
        "optimizer_state_validation_md"
    ].read_text(encoding="utf-8")
