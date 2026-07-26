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


def test_adam_support_contract_is_end_to_end_multi_epoch_hls_validated() -> None:
    contract = _resolve_training_optimizer_loss_contract(_raw("adam"))
    status = contract["optimizer"]["support_status"]
    assert status["generated_hls_update"] == "implemented"
    assert status["single_step_reference"] == "implemented"
    assert status["single_step_numeric_validation"] == "implemented"
    assert status["dataset_multi_epoch_reference"] == "implemented"
    assert status["dataset_multi_epoch_hls"] == "implemented"
    assert status["end_to_end_multi_epoch_validation"] == "implemented"


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
static acc_t ACC_dW_dense0[4];
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



def test_adam_optimizer_state_export_uses_m_v_then_step_canonical_order() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import (
        _fpgai_insert_optimizer_state_export_capture,
    )

    source = r'''
using namespace fpgai;
static wgt_t W_dense0[4] = {0};
static bias_t B_dense0[2] = {0};
static wgt_t W_dense1[3] = {0};
static bias_t B_dense1[1] = {0};
static opt_t FPGAI_ADAM_V_B_dense1[1];
static opt_t FPGAI_ADAM_M_B_dense0[2];
static opt_t FPGAI_ADAM_V_W_dense0[4];
static opt_t FPGAI_ADAM_M_W_dense1[3];
static opt_t FPGAI_ADAM_M_W_dense0[4];
static opt_t FPGAI_ADAM_V_B_dense0[2];
static opt_t FPGAI_ADAM_M_B_dense1[1];
static opt_t FPGAI_ADAM_V_W_dense1[3];
static unsigned long long FPGAI_ADAM_STEP = 0ULL;
extern "C" void deeplearn(ap_uint<32>* optimizer_state_mem, int mode) {
  if (mode == FPGAI_MODE_RESET_ACCUMULATORS || mode == 5) { return; }
}
'''
    generated = _fpgai_insert_optimizer_state_export_capture(
        source,
        raw_cfg={
            "training": {"optimizer": {"type": "adam"}},
            "data_movement": {
                "optimizer_state": {
                    "export": {"interface": "m_axi", "policy": "full"}
                }
            },
        },
    )

    expected = [
        "optimizer_state tensor FPGAI_ADAM_M_W_dense0: offset_words=0, count_words=4",
        "optimizer_state tensor FPGAI_ADAM_M_B_dense0: offset_words=4, count_words=2",
        "optimizer_state tensor FPGAI_ADAM_M_W_dense1: offset_words=6, count_words=3",
        "optimizer_state tensor FPGAI_ADAM_M_B_dense1: offset_words=9, count_words=1",
        "optimizer_state tensor FPGAI_ADAM_V_W_dense0: offset_words=10, count_words=4",
        "optimizer_state tensor FPGAI_ADAM_V_B_dense0: offset_words=14, count_words=2",
        "optimizer_state tensor FPGAI_ADAM_V_W_dense1: offset_words=16, count_words=3",
        "optimizer_state tensor FPGAI_ADAM_V_B_dense1: offset_words=19, count_words=1",
        "optimizer_state scalar FPGAI_ADAM_STEP: offset_words=20, count_words=1",
    ]
    positions = [generated.index(item) for item in expected]
    assert positions == sorted(positions)
    assert "FPGAI_OPTIMIZER_STATE_EXPORT_WORDS = 21" in generated
    assert "optimizer_state_mem[20]" in generated
    assert "(float)FPGAI_ADAM_STEP" in generated

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


def _synthetic_adam_update_source() -> str:
    return r'''
#include "fpgai_training.hpp"
using namespace fpgai;
static wgt_t W_dense0[4];
static bias_t B_dense0[2];
static grad_wgt_t dW_dense0[4];
static acc_t ACC_dW_dense0[4];
static grad_bias_t dB_dense0[2];
extern "C" void deeplearn(int mode) {
  if (mode == FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS || mode == 4) {
    fpgai::sgd_update_wgt_typed<4, wgt_t, grad_wgt_t, upd_t, acc_t, 1, 4>(W_dense0, dW_dense0, (upd_t)0.001f);
    fpgai::sgd_update_bias_typed<2, bias_t, grad_bias_t, upd_t, acc_t, 1, 2>(B_dense0, dB_dense0, (upd_t)0.001f);
    return;
  }
  fpgai::sgd_update_wgt_typed<4, wgt_t, grad_wgt_t, upd_t, acc_t, 1, 4>(W_dense0, dW_dense0, (upd_t)0.001f);
  fpgai::sgd_update_bias_typed<2, bias_t, grad_bias_t, upd_t, acc_t, 1, 2>(B_dense0, dB_dense0, (upd_t)0.001f);
}
'''


def test_generated_adam_step_advances_once_per_update_path_without_bias_correction() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_ensure_adam_final_contract

    generated = _fpgai_ensure_adam_final_contract(
        _synthetic_adam_update_source(),
        raw_cfg={
            "training": {
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1.0e-8,
                    "bias_correction": False,
                }
            }
        },
    )

    assert "static unsigned long long FPGAI_ADAM_STEP = 0ULL;" in generated
    assert "static float FPGAI_ADAM_BETA1_POWER = 1.0f;" in generated
    assert "static float FPGAI_ADAM_BETA2_POWER = 1.0f;" in generated
    assert generated.count("FPGAI_ADAM_STEP += 1ULL;") == 2
    assert generated.count("optimizer-step advance (accumulated_gradients)") == 1
    assert generated.count("optimizer-step advance (direct_training)") == 1
    assert "FPGAI_ADAM_INV_BIAS1" not in generated
    assert "sqrtf(adam_v_used) +" in generated
    assert "sqrtf((float)FPGAI_ADAM_V_" not in generated


def test_generated_adam_bias_correction_uses_updated_persistent_step_state() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_ensure_adam_final_contract

    generated = _fpgai_ensure_adam_final_contract(
        _synthetic_adam_update_source(),
        raw_cfg={
            "training": {
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1.0e-8,
                    "bias_correction": True,
                }
            }
        },
    )

    assert "// bias_correction = true" in generated
    assert generated.count("FPGAI_ADAM_STEP += 1ULL;") == 2
    assert generated.count("FPGAI_ADAM_BETA1_POWER *=") == 2
    assert generated.count("FPGAI_ADAM_BETA2_POWER *=") == 2
    assert generated.count("const float FPGAI_ADAM_INV_BIAS1") == 2
    assert generated.count("const float FPGAI_ADAM_INV_BIAS2") == 2
    assert "adam_m_used *= FPGAI_ADAM_INV_BIAS1;" in generated
    assert "adam_v_used *= FPGAI_ADAM_INV_BIAS2;" in generated
    first_step = generated.index("FPGAI_ADAM_STEP += 1ULL;")
    first_correction = generated.index("const float FPGAI_ADAM_INV_BIAS1")
    first_update = generated.index("// FPGAI Adam optimizer update for")
    assert first_step < first_correction < first_update




def test_generated_adam_replaces_native_tiled_mode4_sgd_helpers() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_ensure_adam_final_contract

    source = r'''
#include "fpgai_training.hpp"
using namespace fpgai;
static wgt_t W_dense0[4];
static bias_t B_dense0[2];
static grad_wgt_t dW_dense0[4];
static acc_t ACC_dW_dense0[4];
static grad_bias_t dB_dense0[2];
extern "C" void deeplearn(int mode) {
  if (mode == FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS || mode == 4) {
    fpgai::sgd_update_wgt_tiled<4, wgt_t, grad_wgt_t, upd_t, acc_t, 1, 4>(W_dense0, dW_dense0, (upd_t)0.00100000f);
    fpgai::sgd_update_bias_tiled<2, bias_t, grad_bias_t, upd_t, acc_t, 1, 2>(B_dense0, dB_dense0, (upd_t)0.00100000f);
    return;
  }
}
'''
    generated = _fpgai_ensure_adam_final_contract(
        source,
        raw_cfg={
            "training": {
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1.0e-8,
                    "bias_correction": True,
                }
            },
            "numerics": {
                "training": {
                    "update_accum": {"total_bits": 24, "int_bits": 10}
                }
            },
        },
    )

    mode4_start = generated.index(
        "if (mode == FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS || mode == 4)"
    )
    mode4_end = generated.index("return;", mode4_start)
    mode4 = generated[mode4_start:mode4_end]

    assert "sgd_update_wgt_tiled" not in mode4
    assert "sgd_update_bias_tiled" not in mode4
    assert mode4.count("FPGAI_ADAM_STEP += 1ULL;") == 1
    assert "FPGAI Adam optimizer update for W_dense0" in mode4
    assert "FPGAI Adam optimizer update for B_dense0" in mode4
    assert "FPGAI_ADAM_M_W_dense0[i] =" in mode4
    assert "FPGAI_ADAM_V_W_dense0[i] =" in mode4
    assert "FPGAI_ADAM_M_B_dense0[i] =" in mode4
    assert "FPGAI_ADAM_V_B_dense0[i] =" in mode4


def test_generated_adam_epsilon_is_clamped_to_update_accum_lsb() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_ensure_adam_final_contract

    generated = _fpgai_ensure_adam_final_contract(
        _synthetic_adam_update_source(),
        raw_cfg={
            "training": {
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1.0e-8,
                    "bias_correction": True,
                }
            },
            "numerics": {
                "training": {
                    "update_accum": {"total_bits": 24, "int_bits": 10}
                }
            },
        },
    )

    assert "epsilon_requested = 1.00000000e-08" in generated
    assert "epsilon_effective = 6.10351562e-05" in generated
    assert "epsilon_policy = clamp_to_one_update_accum_lsb" in generated
    assert "sqrtf(adam_v_used) + (float)6.10351562e-05f" in generated


def test_adam_epsilon_resolution_uses_canonical_update_accum_contract() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_ensure_adam_final_contract
    from fpgai.backends.hls.emit.types_h import resolve_training_numeric_specs
    from fpgai.benchmark.training_dataset_reference import _training_numeric_specs

    raw = {
        "training": {
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1.0e-8,
                "bias_correction": True,
            }
        },
        "numerics": {
            "defaults": {
                "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10}
            },
            "training": {
                "update_accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 4}
            },
        },
    }

    canonical = resolve_training_numeric_specs(raw)["update_accum"]
    reference = _training_numeric_specs(raw)["update_accum"]
    assert canonical == reference
    assert canonical["total_bits"] == 24
    assert canonical["int_bits"] == 4

    generated = _fpgai_ensure_adam_final_contract(
        _synthetic_adam_update_source(), raw_cfg=raw
    )
    assert "epsilon_effective = 9.53674316e-07" in generated
    assert "epsilon_fractional_bits = 20" in generated
    assert "epsilon_resolution_source = canonical_training_numeric_specs.update_accum" in generated


def test_training_numeric_spec_partial_override_inherits_fallback_fields() -> None:
    from fpgai.backends.hls.emit.types_h import resolve_training_numeric_specs

    specs = resolve_training_numeric_specs({
        "numerics": {
            "defaults": {
                "accum": {"type": "ap_fixed", "total_bits": 30, "int_bits": 9}
            },
            "training": {
                "update_accum": {"int_bits": 5}
            },
        }
    })

    assert specs["update_accum"] == {
        "type": "ap_fixed",
        "total_bits": 30,
        "int_bits": 5,
    }


def test_hardware_domain_adam_writeback_matches_generated_float32_expression_order() -> None:
    import inspect
    from fpgai.benchmark import training_dataset_reference

    source = inspect.getsource(training_dataset_reference._hardware_batch_update)
    assert "inverse_bias1 = np.float32" in source
    assert "m_used = (m_used * inverse_bias1).astype(np.float32)" in source
    assert "v_used = (v_used * inverse_bias2).astype(np.float32)" in source
    assert "numerator = (np.float32(learning_rate) * m_used).astype(np.float32)" in source
    assert "adam_delta = (numerator / denominator).astype(np.float32)" in source
    assert "ratio = m_used / denominator" not in source


def test_hardware_domain_adam_uses_raw_float32_beta_literals() -> None:
    import inspect
    from fpgai.benchmark import training_dataset_reference

    source = inspect.getsource(training_dataset_reference._hardware_batch_update)
    assert "beta1_effective = float(np.float32(beta1))" in source
    assert "beta2_effective = float(np.float32(beta2))" in source
    assert "quantize_ap_fixed_array(np.asarray([beta1]" not in source
    assert "quantize_ap_fixed_array(np.asarray([beta2]" not in source
    assert '"beta_resolution_source": "generated_hls_raw_float32_literal"' in source


def test_optimizer_state_multi_update_accepts_sparse_two_lsb_propagation_and_reports_segments(tmp_path) -> None:
    import json
    import struct
    from fpgai.validation.numeric import _optimizer_state_validation_payload

    root = tmp_path / "hardware_domain"
    trace = root / "per_sample_trace"
    trace.mkdir(parents=True)
    ref = root / "optimizer_state_after_ref.bin"
    got = tmp_path / "optimizer_state_after.bin"
    lsb = 1.0 / 65536.0

    # Two parameter words -> m[2], v[2], step[1]. One m word propagates to 2 LSB.
    ref.write_bytes(struct.pack("<5f", 0.25, -0.5, 0.125, -0.25, 2.0))
    got.write_bytes(struct.pack("<5f", 0.25 + 2 * lsb, -0.5, 0.125 + lsb, -0.25, 2.0))
    (root / "training_hardware_domain_reference.json").write_text(
        json.dumps({"optimizer_updates": 2}), encoding="utf-8"
    )
    (trace / "parameter_layer_map.json").write_text(
        json.dumps({
            "schema_version": 1,
            "entries": [
                {"layer": "dense0", "role": "weight", "offset": 0, "count": 1, "shape": [1]},
                {"layer": "dense0", "role": "bias", "offset": 1, "count": 1, "shape": [1]},
            ],
        }),
        encoding="utf-8",
    )

    payload = _optimizer_state_validation_payload(
        {
            "requested": True,
            "optimizer": "adam",
            "layout": "m_then_v_then_step_canonical_parameter_order",
            "comparisons": {"packed_optimizer_state_after": {"ref": ref, "got": got}},
        },
        raw_config={"numerics": {"training": {"optimizer_state": {"total_bits": 24, "int_bits": 8}}}},
    )

    comparison = payload["comparisons"]["packed_optimizer_state_after"]
    assert payload["status"] == "implemented"
    assert payload["passed"] is True
    assert comparison["classification"] == "propagated_quantization_aligned"
    assert comparison["within_one_lsb"] == 4
    assert comparison["within_two_lsb"] == 5
    assert comparison["above_two_lsb"] == 0
    assert comparison["maximum_lsb_distance"] == 2
    assert [segment["name"] for segment in comparison["segments"]] == [
        "m_W_dense0", "m_B_dense0", "v_W_dense0", "v_B_dense0", "step"
    ]
    assert [segment["maximum_lsb_distance"] for segment in comparison["segments"]] == [2, 0, 1, 0, 0]
    assert all("max_lsb_distance" not in segment for segment in comparison["segments"])


def test_optimizer_state_single_update_remains_strictly_one_lsb(tmp_path) -> None:
    import json
    import struct
    from fpgai.validation.numeric import _optimizer_state_validation_payload

    root = tmp_path / "hardware_domain"
    root.mkdir()
    ref = root / "optimizer_state_after_ref.bin"
    got = tmp_path / "optimizer_state_after.bin"
    lsb = 1.0 / 65536.0
    ref.write_bytes(struct.pack("<2f", 0.0, 1.0))
    got.write_bytes(struct.pack("<2f", 2 * lsb, 1.0))
    (root / "training_hardware_domain_reference.json").write_text(
        json.dumps({"optimizer_updates": 1}), encoding="utf-8"
    )

    payload = _optimizer_state_validation_payload(
        {
            "requested": True,
            "optimizer": "momentum",
            "comparisons": {"packed_optimizer_state_after": {"ref": ref, "got": got}},
        },
        raw_config={"numerics": {"training": {"optimizer_state": {"total_bits": 24, "int_bits": 8}}}},
    )
    comparison = payload["comparisons"]["packed_optimizer_state_after"]
    assert payload["passed"] is False
    assert comparison["classification"] == "failed"
    assert comparison["numeric_tolerance"]["allowed_lsb"] == 1


def test_parameter_update_validation_reports_role_specific_lsb_segments(tmp_path) -> None:
    import json
    import struct
    from fpgai.validation.numeric import _parameter_update_validation_payload

    root = tmp_path / "hardware_domain"
    trace = root / "per_sample_trace"
    trace.mkdir(parents=True)
    ref = root / "weights_after_ref.bin"
    got = tmp_path / "weights_after.bin"
    weight_lsb = 1.0 / 4096.0
    bias_lsb = 1.0 / 65536.0
    ref.write_bytes(struct.pack("<2f", 0.25, -0.5))
    got.write_bytes(struct.pack("<2f", 0.25 + weight_lsb, -0.5 + bias_lsb))
    (root / "training_hardware_domain_reference.json").write_text(json.dumps({"optimizer_updates": 1}), encoding="utf-8")
    (trace / "parameter_layer_map.json").write_text(json.dumps({"entries": [
        {"layer": "dense0", "role": "weight", "offset": 0, "count": 1},
        {"layer": "dense0", "role": "bias", "offset": 1, "count": 1},
    ]}), encoding="utf-8")
    payload = _parameter_update_validation_payload(
        {"requested": True, "ref": ref, "got": got},
        raw_config={"numerics": {"defaults": {
            "weight": {"total_bits": 20, "int_bits": 8},
            "bias": {"total_bits": 32, "int_bits": 16},
        }}},
    )
    assert payload["status"] == "implemented"
    assert payload["classification"] == "quantization_aligned"
    assert [segment["name"] for segment in payload["segments"]] == ["W_dense0", "B_dense0"]
    assert [segment["maximum_lsb_distance"] for segment in payload["segments"]] == [1, 1]


def test_adam_shared_arithmetic_emits_single_reusable_correction_owner() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_ensure_adam_final_contract
    raw = _raw("adam")
    raw["training"]["optimizer"]["implementation"] = {"arithmetic": "shared", "update_parallelism": 1}
    source = """
#include <hls_math.h>
using namespace fpgai;
static grad_wgt_t dW_dense0[4];
static wgt_t W_dense0[4];
extern "C" void deeplearn() {
  fpgai::sgd_update_wgt_typed<4, wgt_t, grad_wgt_t, upd_t>(W_dense0, dW_dense0, (upd_t)0.01f);
}
"""
    generated = _fpgai_ensure_adam_final_contract(source, raw_cfg=raw)
    assert "fpgai_adam_delta_shared" in generated
    assert "#pragma HLS INLINE off" in generated
    assert "arithmetic_strategy = shared" in generated
    assert "float adam_delta = fpgai_adam_delta_shared" in generated


def test_training_testbench_emits_per_update_trace_contract() -> None:
    from pathlib import Path
    text = Path("fpgai/backends/hls/testbench_train.py").read_text(encoding="utf-8")
    assert "per_update_trace" in text
    assert "capture_update_trace" in text
    assert "update_%04d_weights.bin" in text
    assert "update_%04d_optimizer_state.bin" in text


def test_optimizer_resource_strategy_reports_shared_synthesis_scope(tmp_path) -> None:
    from fpgai.validation.numeric import _optimizer_resource_strategy_payload
    report = tmp_path / "csynth.xml"
    report.write_text("""<Report><AreaEstimates><Resources><BRAM_18K>12</BRAM_18K><DSP>7</DSP><FF>345</FF><LUT>678</LUT><URAM>3</URAM></Resources></AreaEstimates><PerformanceEstimates><SummaryOfOverallLatency><Best-caseLatency>100</Best-caseLatency><Worst-caseLatency>120</Worst-caseLatency><Interval-min>4</Interval-min><Interval-max>5</Interval-max></SummaryOfOverallLatency><SummaryOfTimingAnalysis><EstimatedClockPeriod>4.5</EstimatedClockPeriod></SummaryOfTimingAnalysis></PerformanceEstimates></Report>""", encoding="utf-8")
    payload = _optimizer_resource_strategy_payload(
        {"training": {"optimizer": {"implementation": {"arithmetic": "shared", "update_parallelism": 1}}}, "memory": {"optimizer_state_storage": "uram"}},
        hls_ran=True,
        hls_ok=True,
        hls_csynth_report=report,
    )
    assert payload["mechanism"] == "single_non_inlined_adam_correction_owner"
    assert payload["hls_synthesis_status"] == "available"
    assert payload["optimizer_state_storage"] == "uram"
    assert payload["csynth_report_present"] is True
    assert payload["hls_metrics"]["actual_lut"] == 678
    assert payload["hls_metrics"]["actual_uram"] == 3
    assert payload["hls_metrics"]["interval_min_cycles"] == 4
    assert payload["hls_metrics"]["estimated_clock_period_ns"] == 4.5


def test_update_behavior_trace_reports_final_boundary_path() -> None:
    from fpgai.validation.numeric import _training_update_behavior_payload
    parameter = {
        "passed": True,
        "optimizer_updates": 2,
        "segments": [{"name": "W_dense0", "count": 2, "exact_words": 1}],
    }
    optimizer = {
        "passed": True,
        "optimizer": "adam",
        "comparisons": {"packed_optimizer_state_after": {
            "classification": "propagated_quantization_aligned",
            "optimizer_updates": 2,
            "segments": [{"name": "m_W_dense1", "count": 2, "exact_words": 1}],
        }},
    }
    payload = _training_update_behavior_payload(parameter, optimizer, raw_config={})
    assert payload["status"] == "implemented"
    assert payload["first_divergent_update"] == 1
    assert payload["first_divergent_layer"] == "dense0"
    assert payload["first_divergent_tensor"] == "W_dense0"
    assert payload["propagation_path"] == ["W_dense0", "m_W_dense1"]
    assert payload["final_classification"] == "propagated_quantization_aligned"


def test_adam_persistent_state_arrays_materialize_requested_uram_bindings() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_ensure_adam_final_contract
    raw = _raw("adam")
    raw.setdefault("training", {}).setdefault("storage", {})["optimizer_state"] = "uram"
    source = """
#include <hls_math.h>
using namespace fpgai;
static grad_wgt_t dW_dense0[4];
static wgt_t W_dense0[4];
extern "C" void deeplearn() {
  fpgai::sgd_update_wgt_typed<4, wgt_t, grad_wgt_t, upd_t>(W_dense0, dW_dense0, (upd_t)0.01f);
}
"""
    generated = _fpgai_ensure_adam_final_contract(source, raw_cfg=raw)
    assert "static opt_t FPGAI_ADAM_M_W_dense0[4];" in generated
    m_binding = "#pragma HLS BIND_STORAGE variable=FPGAI_ADAM_M_W_dense0 type=ram_2p impl=uram"
    v_binding = "#pragma HLS BIND_STORAGE variable=FPGAI_ADAM_V_W_dense0 type=ram_2p impl=uram"
    assert m_binding in generated
    assert v_binding in generated
    top_start = generated.index('extern "C" void deeplearn() {')
    assert generated.index(m_binding) > top_start
    assert generated.index(v_binding) > top_start
    assert m_binding not in generated[:top_start]
    assert v_binding not in generated[:top_start]


def test_optimizer_resource_strategy_reports_state_array_source_bindings(tmp_path) -> None:
    from fpgai.validation.numeric import _optimizer_resource_strategy_payload
    hls_root = tmp_path / "hls"
    report = hls_root / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt"
    report.parent.mkdir(parents=True)
    report.write_text("<Report><AreaEstimates><Resources><URAM>4</URAM><BRAM_18K>20</BRAM_18K><LUT>100</LUT><FF>200</FF><DSP>3</DSP></Resources></AreaEstimates></Report>", encoding="utf-8")
    source = hls_root / "src" / "deeplearn.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("""
static opt_t FPGAI_ADAM_M_W_dense0[4];
#pragma HLS BIND_STORAGE variable=FPGAI_ADAM_M_W_dense0 type=ram_2p impl=uram
static opt_t FPGAI_ADAM_V_W_dense0[4];
#pragma HLS BIND_STORAGE variable=FPGAI_ADAM_V_W_dense0 type=ram_2p impl=uram
""", encoding="utf-8")
    payload = _optimizer_resource_strategy_payload(
        {"training": {"storage": {"optimizer_state": "uram"}}},
        hls_ran=True,
        hls_ok=True,
        hls_csynth_report=report,
    )
    assert payload["requested_state_impl"] == "uram"
    assert payload["synthesized_storage_status"] == "observed"
    assert [item["binding_status"] for item in payload["state_array_bindings"]] == ["materialized", "materialized"]
    assert all(item["source_binding"] == "uram" for item in payload["state_array_bindings"])


def test_hls_board_fit_promotes_csynth_resources_and_preserves_prediction(tmp_path) -> None:
    import json
    from types import SimpleNamespace
    from fpgai.engine.compiler import Compiler
    from fpgai.reporting.hardware_feasibility import emit_board_fit_report

    out_dir = tmp_path / "compile"
    reports = out_dir / "reports"
    raw = {
        "targets": {"platform": {"board": "kv260", "part": "xck26-sfvc784-2LV-c"}},
        "memory": {"weight_storage": "bram", "optimizer_state_storage": "uram"},
    }
    prediction = emit_board_fit_report(
        reports,
        resource_data={"lut": 352567, "ff": 458097, "dsp": 94, "bram_18k": 778, "uram": 1},
        board="kv260",
        part="xck26-sfvc784-2LV-c",
        target_clock_mhz=200.0,
        source="prediction",
        raw_config=raw,
        build_stages={},
    )
    prediction_artifacts = {"board_fit": prediction}

    report = tmp_path / "deeplearn_csynth.xml"
    report.write_text(
        "<Report><AreaEstimates><Resources>"
        "<BRAM_18K>668</BRAM_18K><DSP>177</DSP><FF>100469</FF>"
        "<LUT>155028</LUT><URAM>56</URAM>"
        "</Resources></AreaEstimates><PerformanceEstimates>"
        "<SummaryOfOverallLatency><Worst-caseLatency>1252785</Worst-caseLatency>"
        "</SummaryOfOverallLatency></PerformanceEstimates></Report>",
        encoding="utf-8",
    )

    compiler = Compiler.__new__(Compiler)
    compiler.cfg = SimpleNamespace(raw=raw)
    refreshed = compiler._refresh_board_fit_from_hls(
        out_dir=out_dir,
        compile_plan=SimpleNamespace(clock_mhz=200.0),
        hls_run=SimpleNamespace(ok=True, csynth_report=report),
        prediction_artifacts=prediction_artifacts,
        build_stages={},
    )

    payload = json.loads((reports / "board_fit.json").read_text(encoding="utf-8"))
    assert payload["format"] == "fpgai.board_fit.v2"
    assert payload["source"] == "hls_synthesis"
    assert payload["active_fit_source"] == "hls_synthesis"
    assert payload["normalized_resources"]["lut"] == 155028
    assert payload["normalized_resources"]["ff"] == 100469
    assert payload["normalized_resources"]["bram_18k"] == 668
    assert payload["normalized_resources"]["uram"] == 56
    assert payload["fit"]["limiting_dimension"] == "bram_18k"
    assert payload["prediction_fit"]["source"] == "prediction"
    assert payload["hls_synthesis_fit"]["source"] == "hls_synthesis"
    assert refreshed["board_fit"]["source"] == "hls_synthesis"
    assert (reports / "board_fit_prediction.json").exists()
    assert (reports / "board_fit_hls_synthesis.json").exists()


def test_training_resource_ownership_maps_generated_arrays_to_yaml_knobs(tmp_path) -> None:
    from fpgai.validation.numeric import _optimizer_resource_strategy_payload

    hls_root = tmp_path / "hls"
    report = hls_root / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt"
    report.parent.mkdir(parents=True)
    report.write_text("<Report><AreaEstimates><Resources><URAM>4</URAM><BRAM_18K>20</BRAM_18K><LUT>100</LUT><FF>200</FF><DSP>3</DSP></Resources></AreaEstimates></Report>", encoding="utf-8")
    source = hls_root / "src" / "deeplearn.cpp"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
static wgt_t W_dense0[1024];
#pragma HLS BIND_STORAGE variable=W_dense0 type=ram_2p impl=bram
static grad_wgt_t dW_dense0[1024];
#pragma HLS BIND_STORAGE variable=dW_dense0 type=ram_2p impl=bram
static opt_t FPGAI_ADAM_M_W_dense0[1024];
#pragma HLS BIND_STORAGE variable=FPGAI_ADAM_M_W_dense0 type=ram_2p impl=uram
""",
        encoding="utf-8",
    )
    raw = {
        "optimization": {"parallel": {"pe": 1, "simd": 1, "partition_factor": 1}},
        "memory": {"weight_storage": "bram", "optimizer_state_storage": "uram"},
        "training": {
            "storage": {"gradient": "bram", "optimizer_state": "uram"},
            "optimizer": {"implementation": {"arithmetic": "shared", "update_parallelism": 1}},
        },
    }
    payload = _optimizer_resource_strategy_payload(
        raw,
        hls_ran=True,
        hls_ok=True,
        hls_csynth_report=report,
    )
    ownership = payload["training_resource_ownership"]
    assert ownership["status"] == "implemented"
    by_name = {row["name"]: row for row in ownership["owners"]}
    assert by_name["W_dense0"]["owning_yaml_knob"] == "memory.weight_storage"
    assert by_name["dW_dense0"]["owning_yaml_knob"] == "training.storage.parameter_gradient"
    assert by_name["FPGAI_ADAM_M_W_dense0"]["owning_yaml_knob"] == "training.storage.optimizer_state"
    assert by_name["FPGAI_ADAM_M_W_dense0"]["source_binding"] == "uram"
    assert any(row["path"] == "optimization.parallel.partition_factor" for row in ownership["hardware_knob_trace"])
    assert any(row["knob"] == "optimization.parallel.pe" for row in ownership["recommended_knob_actions"])


def test_gradient_materialization_modes_are_selectable_and_change_generated_source() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_p3d_f4lm_materialize_gradient_export

    source = '''
static grad_wgt_t dW_dense0[8];
static grad_bias_t dB_dense0[2];
static float OUT_grad_dense0[10];
extern "C" void deeplearn(
  hls::stream<axis_t>& out,
  int mode
) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
  for (int i = 0; i < 8; ++i) OUT_grad_dense0[i] = (float)W_dense0[i];
  for (int i = 0; i < 2; ++i) OUT_grad_dense0[8 + i] = (float)B_dense0[i];
  emit_stream_block<10>(out, OUT_grad_dense0, true);
  for (int i = 0; i < 8; ++i) OUT_grad_dense0[i] = (float)dW_dense0[i];
  for (int i = 0; i < 2; ++i) OUT_grad_dense0[8 + i] = (float)dB_dense0[i];
  emit_stream_block<10>(out, OUT_grad_dense0, true);
  for (int i = 0; i < 8; ++i) OUT_grad_dense0[i] = (float)ACC_dW_dense0[i];
  for (int i = 0; i < 2; ++i) OUT_grad_dense0[8 + i] = (float)ACC_dB_dense0[i];
  emit_stream_block<10>(out, OUT_grad_dense0, true);
}
'''
    full = _fpgai_p3d_f4lm_materialize_gradient_export(
        source,
        raw_cfg={"training": {"gradients": {"materialization": "full"}}},
    )
    assert "OUT_grad_dense0[10]" in full

    tiled = _fpgai_p3d_f4lm_materialize_gradient_export(
        source,
        raw_cfg={"training": {"gradients": {"materialization": "tiled", "tile_size": 4}, "memory_lifetime": {"policy": "phase_shared"}}},
    )
    assert "OUT_grad_dense0[10]" not in tiled
    assert "FPGAI_SHARED_GRADIENT_EXPORT_TILE[FPGAI_GRADIENT_MATERIALIZATION_TILE_SIZE]" in tiled
    assert "#define FPGAI_GRADIENT_MATERIALIZATION_TILE_SIZE 4" in tiled
    assert "lifetime_policy=phase_shared" in tiled
    assert "OUT_grad_dense0" not in tiled
    assert "tiled weights materialization" in tiled
    assert "tiled gradients materialization" in tiled
    assert "tiled accumulated_gradients materialization" in tiled

    streamed = _fpgai_p3d_f4lm_materialize_gradient_export(
        source,
        raw_cfg={"training": {"gradients": {"materialization": "streamed"}, "memory_lifetime": {"policy": "separate"}}},
    )
    assert "OUT_grad_dense0[10]" not in streamed
    assert "no OUT_grad scratch array" in streamed
    assert "FPGAI_SHARED_GRADIENT_EXPORT_TILE" not in streamed
    assert "OUT_grad_dense0" not in streamed
    assert "streamed weights materialization" in streamed
    assert "streamed gradients materialization" in streamed
    assert "streamed accumulated_gradients materialization" in streamed


def test_phase_shared_rejects_full_gradient_materialization() -> None:
    import pytest
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_p3d_f4lm_gradient_policy

    with pytest.raises(ValueError, match="phase_shared requires"):
        _fpgai_p3d_f4lm_gradient_policy({
            "training": {
                "gradients": {"materialization": "full"},
                "memory_lifetime": {"policy": "phase_shared"},
            }
        })


def test_board_fit_limiting_dimension_prioritizes_over_limit_over_near_limit_clock() -> None:
    from fpgai.reporting.hardware_feasibility import classify_board_fit

    result = classify_board_fit(
        board="kv260",
        part="xck26-sfvc784-2LV-c",
        resources={
            "lut": 156465,
            "ff": 100726,
            "dsp": 177,
            "bram_18k": 481,
            "uram": 56,
            "target_clock_mhz": 200.0,
        },
    )
    assert result["status"] == "over_limit"
    assert result["resources"]["target_clock_mhz"]["status"] == "near_limit"
    assert result["resources"]["bram_18k"]["status"] == "over_limit"
    assert result["limiting_dimension"] == "bram_18k"


def test_training_command_latency_separates_aggregate_hls_range_from_source_bounds(tmp_path) -> None:
    from fpgai.validation.numeric import _optimizer_resource_strategy_payload

    hls_root = tmp_path / "hls"
    report = hls_root / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt"
    report.parent.mkdir(parents=True)
    report.write_text(
        "<Report><PerformanceEstimates><SummaryOfOverallLatency>"
        "<Best-caseLatency>100</Best-caseLatency><Worst-caseLatency>900</Worst-caseLatency>"
        "<Interval-min>101</Interval-min><Interval-max>901</Interval-max>"
        "</SummaryOfOverallLatency></PerformanceEstimates>"
        "<AreaEstimates><Resources><BRAM_18K>8</BRAM_18K><LUT>20</LUT><FF>30</FF><DSP>2</DSP><URAM>1</URAM></Resources></AreaEstimates></Report>",
        encoding="utf-8",
    )
    source = hls_root / "src" / "deeplearn.cpp"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
static wgt_t W_dense0[8];
static bias_t B_dense0[2];
static const int FPGAI_OPTIMIZER_STATE_EXPORT_WORDS = 21;
""",
        encoding="utf-8",
    )
    payload = _optimizer_resource_strategy_payload(
        {
            "training": {
                "gradients": {"materialization": "tiled", "tile_size": 4},
                "memory_lifetime": {"policy": "phase_shared"},
            }
        },
        hls_ran=True,
        hls_ok=True,
        hls_csynth_report=report,
    )
    latency = payload["training_command_latency"]
    assert latency["aggregate_hls_top"]["latency_max_cycles"] == 900.0
    assert latency["commands"]["run_training"]["status"] == "aggregate_hls_range_only"
    assert latency["commands"]["export_weights"]["output_words"] == 10
    assert latency["commands"]["export_gradients"]["materialization"] == "tiled"
    assert latency["commands"]["export_gradients"]["tile_size"] == 4
    assert latency["commands"]["export_optimizer_state"]["output_words"] == 21


def test_parameter_gradient_policy_defaults_and_real_storage_options() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f4op_parameter_gradient_policy

    assert _fpgai_f4op_parameter_gradient_policy({}) == ("full_buffer", "bram")
    assert _fpgai_f4op_parameter_gradient_policy({
        "training": {
            "gradients": {"computation": "full_buffer"},
            "storage": {"parameter_gradient": "uram"},
        }
    }) == ("full_buffer", "uram")


def test_parameter_gradient_policy_rejects_unlowered_storage_modes_explicitly() -> None:
    import pytest
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f4op_parameter_gradient_policy

    with pytest.raises(ValueError, match="no real external-memory or recomputation lowering"):
        _fpgai_f4op_parameter_gradient_policy({
            "training": {"storage": {"parameter_gradient": "ddr"}}
        })


def test_parameter_gradient_contract_changes_generated_source_banner() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f4op_materialize_parameter_gradient_contract

    source = 'extern "C" void deeplearn() {}\n'
    out = _fpgai_f4op_materialize_parameter_gradient_contract(
        source,
        raw_cfg={
            "training": {
                "gradients": {"computation": "full_buffer"},
                "storage": {"parameter_gradient": "uram"},
            }
        },
    )
    assert "parameter-gradient computation=full_buffer" in out
    assert "storage=uram" in out
    assert "owner=training.storage.parameter_gradient" in out


def test_tiled_accumulate_and_fused_update_policies_are_independently_selectable() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f4op_parameter_gradient_policy

    assert _fpgai_f4op_parameter_gradient_policy({
        "training": {
            "gradients": {"computation": "tiled_accumulate"},
            "storage": {"parameter_gradient": "bram"},
        }
    }) == ("tiled_accumulate", "bram")
    assert _fpgai_f4op_parameter_gradient_policy({
        "training": {"gradients": {"computation": "fused_update"}}
    }) == ("fused_update", "recompute")


def test_real_dense_adam_tiled_accumulate_removes_complete_dw_and_recomputes_export() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f4o_materialize_dense_tiled_accumulate

    source = r'''
static act_t BUF_input[2];
static grad_act_t GRAD_output[2];
static wgt_t W_dense0[4];
static bias_t B_dense0[2];
static grad_wgt_t dW_dense0[4];
static acc_t ACC_dW_dense0[4];
static grad_bias_t dB_dense0[2];
static opt_t FPGAI_ADAM_M_W_dense0[4];
static opt_t FPGAI_ADAM_V_W_dense0[4];
extern "C" void deeplearn(int mode) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS BIND_STORAGE variable=dW_dense0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ACC_dW_dense0 type=ram_2p impl=bram
  fpgai::dense_weight_grad_typed<2, 2, act_t, grad_act_t, grad_wgt_t, acc_t, 1, 1, 1, 1, 1, 1>(BUF_input, GRAD_output, dW_dense0);
  for (int i = 0; i < 4; ++i) ACC_dW_dense0[i] = (acc_t)0;
  for (int i = 0; i < 4; ++i) ACC_dW_dense0[i] += (acc_t)dW_dense0[i];
  // FPGAI Adam optimizer update for W_dense0.
  for (int i = 0; i < 4; ++i) {
    float grad_value = (float)dW_dense0[i];
    FPGAI_ADAM_M_W_dense0[i] = (opt_t)((0.9f * (float)FPGAI_ADAM_M_W_dense0[i]) + (0.1f * grad_value));
    FPGAI_ADAM_V_W_dense0[i] = (opt_t)((0.999f * (float)FPGAI_ADAM_V_W_dense0[i]) + (0.001f * grad_value * grad_value));
    W_dense0[i] = (wgt_t)((float)W_dense0[i] - grad_value);
  }
  for (int i = 0; i < 4; ++i) write_f32(out, (float)dW_dense0[i], false);
  for (int i = 0; i < 4; ++i) write_f32(out, (float)ACC_dW_dense0[i], false);
}
'''
    lowered = _fpgai_f4o_materialize_dense_tiled_accumulate(
        source,
        raw_cfg={
            "training": {
                "optimizer": {"type": "adam"},
                "batch": {"mode": "direct", "size": 1},
                "gradient_accumulation": {"steps": 1},
                "storage": {"parameter_gradient": "bram"},
                "gradients": {
                    "computation": "tiled_accumulate",
                    "materialization": "streamed",
                    "tile_size": 2,
                },
            }
        },
    )
    assert "static grad_wgt_t dW_dense0[4]" not in lowered
    assert "static grad_wgt_t FPGAI_DW_TILE_dense0[2]" in lowered
    assert "complete dW_dense0 array of 4 elements is not materialized" in lowered
    assert "gradient_index / 2" in lowered
    assert "gradient_index % 2" in lowered
    assert "FPGAI_ADAM_M_W_dense0[gradient_index]" in lowered
    assert "dW_dense0[" not in lowered
    assert "ACC_dW_dense0[" not in lowered
    assert "ACC_(grad_wgt_t)" not in lowered
    assert "BUF_input[i % 2]" in lowered
    assert "GRAD_output[i / 2]" in lowered


def test_tiled_accumulate_rejects_accumulated_batch_until_persistent_tiled_schedule_exists() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f4o_materialize_dense_tiled_accumulate

    with pytest.raises(ValueError, match="direct single-record updates"):
        _fpgai_f4o_materialize_dense_tiled_accumulate(
            "static grad_wgt_t dW_dense0[4];",
            raw_cfg={
                "training": {
                    "optimizer": {"type": "adam"},
                    "batch": {"mode": "accumulated", "size": 10},
                    "gradients": {
                        "computation": "tiled_accumulate",
                        "materialization": "tiled",
                        "tile_size": 2,
                    },
                }
            },
        )


def test_tiled_accumulate_example_selects_direct_dense_adam_profile() -> None:
    from pathlib import Path
    import yaml

    raw = yaml.safe_load(
        Path("configs/examples/training_adam_kv260_tiled_accumulate_direct.yml").read_text(encoding="utf-8")
    )
    assert raw["training"]["gradients"]["computation"] == "tiled_accumulate"
    assert raw["training"]["gradients"]["materialization"] == "tiled"
    assert raw["training"]["batch"]["mode"] == "direct"
    assert raw["training"]["batch"]["size"] == 1
    assert raw["training"]["gradient_accumulation"]["steps"] == 1
    assert raw["training"]["optimizer"]["type"] == "adam"
    assert raw["training"]["batch"]["epochs"] == 1
    assert "dataset" not in raw.get("validation", {})


def test_fused_update_policy_defaults_to_recompute_and_rejects_materialized_storage() -> None:
    import pytest
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f4op_parameter_gradient_policy

    assert _fpgai_f4op_parameter_gradient_policy({
        "training": {"gradients": {"computation": "fused_update"}}
    }) == ("fused_update", "recompute")
    assert _fpgai_f4op_parameter_gradient_policy({
        "training": {
            "gradients": {"computation": "fused_update"},
            "storage": {"parameter_gradient": "recompute"},
        }
    }) == ("fused_update", "recompute")
    with pytest.raises(ValueError, match="does not materialize a parameter-gradient buffer"):
        _fpgai_f4op_parameter_gradient_policy({
            "training": {
                "gradients": {"computation": "fused_update"},
                "storage": {"parameter_gradient": "bram"},
            }
        })


def test_real_dense_adam_fused_update_removes_full_and_tiled_gradient_storage() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f5a_materialize_dense_fused_update

    source = r'''
static act_t BUF_input[2];
static grad_act_t GRAD_output[2];
static wgt_t W_dense0[4];
static bias_t B_dense0[2];
static grad_wgt_t dW_dense0[4];
static acc_t ACC_dW_dense0[4];
static grad_bias_t dB_dense0[2];
static opt_t FPGAI_ADAM_M_W_dense0[4];
static opt_t FPGAI_ADAM_V_W_dense0[4];
extern "C" void deeplearn(int mode) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS BIND_STORAGE variable=dW_dense0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ACC_dW_dense0 type=ram_2p impl=bram
  fpgai::dense_weight_grad_typed<2, 2, act_t, grad_act_t, grad_wgt_t, acc_t, 1, 1, 1, 1, 1, 1>(BUF_input, GRAD_output, dW_dense0);
  for (int i = 0; i < 4; ++i) ACC_dW_dense0[i] = (acc_t)0;
  for (int i = 0; i < 4; ++i) ACC_dW_dense0[i] += (acc_t)dW_dense0[i];
  // FPGAI Adam optimizer update for W_dense0.
  for (int i = 0; i < 4; ++i) {
    float grad_value = (float)dW_dense0[i];
    FPGAI_ADAM_M_W_dense0[i] = (opt_t)((0.9f * (float)FPGAI_ADAM_M_W_dense0[i]) + (0.1f * grad_value));
    FPGAI_ADAM_V_W_dense0[i] = (opt_t)((0.999f * (float)FPGAI_ADAM_V_W_dense0[i]) + (0.001f * grad_value * grad_value));
    W_dense0[i] = (wgt_t)((float)W_dense0[i] - grad_value);
  }
  for (int i = 0; i < 4; ++i) write_f32(out, (float)dW_dense0[i], false);
  for (int i = 0; i < 4; ++i) write_f32(out, (float)ACC_dW_dense0[i], false);
}
'''
    lowered = _fpgai_f5a_materialize_dense_fused_update(
        source,
        raw_cfg={
            "training": {
                "optimizer": {"type": "adam"},
                "batch": {"mode": "direct", "size": 1},
                "gradient_accumulation": {"steps": 1},
                "storage": {"parameter_gradient": "recompute"},
                "gradients": {
                    "computation": "fused_update",
                    "export_policy": "recompute",
                },
            }
        },
    )
    assert "static grad_wgt_t dW_dense0[4]" not in lowered
    assert "ACC_dW_dense0[" not in lowered
    assert "FPGAI_DW_TILE_dense0" not in lowered
    assert "no complete or tiled dW_dense0 buffer is materialized" in lowered
    assert "const float fused_grad_value" in lowered
    assert "FPGAI_ADAM_M_W_dense0[gradient_index]" in lowered
    assert "FPGAI_ADAM_V_W_dense0[gradient_index]" in lowered
    assert "BUF_input[i % 2]" in lowered
    assert "GRAD_output[i / 2]" in lowered


def test_fused_update_rejects_non_direct_or_non_adam_profiles() -> None:
    import pytest
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f5a_materialize_dense_fused_update

    with pytest.raises(ValueError, match="optimizer.type=adam"):
        _fpgai_f5a_materialize_dense_fused_update(
            "static grad_wgt_t dW_dense0[4];",
            raw_cfg={"training": {"optimizer": {"type": "sgd"}, "gradients": {"computation": "fused_update"}}},
        )
    with pytest.raises(ValueError, match="direct single-record updates"):
        _fpgai_f5a_materialize_dense_fused_update(
            "static grad_wgt_t dW_dense0[4];",
            raw_cfg={
                "training": {
                    "optimizer": {"type": "adam"},
                    "batch": {"mode": "accumulated", "size": 4},
                    "gradients": {"computation": "fused_update"},
                }
            },
        )


def test_compiler_memory_semantics_enable_fused_update_with_recompute_storage() -> None:
    from types import SimpleNamespace
    from fpgai.engine.compiler import Compiler

    compiler = Compiler.__new__(Compiler)
    compiler._resolve_weight_movement_semantics = lambda raw: {}
    compiler._resolve_activation_storage_semantics = lambda raw: {}
    compile_plan = SimpleNamespace(notes={})
    memory_plan = SimpleNamespace(notes={})

    semantics = compiler._annotate_memory_movement_semantics(
        compile_plan,
        memory_plan,
        {
            "training": {
                "optimizer": {"type": "adam"},
                "batch": {"mode": "direct", "size": 1},
                "gradient_accumulation": {"steps": 1},
                "gradients": {
                    "computation": "fused_update",
                    "materialization": "streamed",
                },
            }
        },
    )

    assert semantics["parameter_gradient_computation"] == "fused_update"
    assert semantics["resolved_gradient_storage"] == "recompute"
    assert semantics["gradient_storage_semantics"] == "parameter_gradient_recompute"
    assert compile_plan.notes["parameter_gradient_computation"] == "fused_update"
    assert memory_plan.notes["resolved_gradient_storage"] == "recompute"


def test_fused_update_defaults_gradient_materialization_to_streamed() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_p3d_f4lm_gradient_policy

    assert _fpgai_p3d_f4lm_gradient_policy({
        "training": {
            "gradients": {"computation": "fused_update"},
            "memory_lifetime": {"policy": "phase_shared"},
        }
    }) == ("streamed", 256, "phase_shared")

    with pytest.raises(ValueError, match="requires training.gradients.materialization=streamed"):
        _fpgai_p3d_f4lm_gradient_policy({
            "training": {
                "gradients": {
                    "computation": "fused_update",
                    "materialization": "full",
                }
            }
        })


def test_fused_update_removes_multiple_dense_gradient_calls_by_discovered_spans() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f5a_materialize_dense_fused_update

    source = r'''
static act_t BUF_input[2];
static act_t BUF_hidden[2];
static grad_act_t GRAD_hidden[2];
static grad_act_t GRAD_output[1];
static wgt_t W_dense0[4];
static wgt_t W_dense1[2];
static grad_wgt_t dW_dense0[4];
static grad_wgt_t dW_dense1[2];
static opt_t FPGAI_ADAM_M_W_dense0[4];
static opt_t FPGAI_ADAM_V_W_dense0[4];
static opt_t FPGAI_ADAM_M_W_dense1[2];
static opt_t FPGAI_ADAM_V_W_dense1[2];
extern "C" void deeplearn(int mode) {
  fpgai::dense_weight_grad_typed<2, 2, act_t, grad_act_t, grad_wgt_t, acc_t, 1, 1, 1, 1, 1, 1>(BUF_input, GRAD_hidden, dW_dense0);
  fpgai::dense_weight_grad_typed<
      2, 1, act_t, grad_act_t, grad_wgt_t, acc_t, 1, 1, 1, 1, 1, 1
  >(BUF_hidden, GRAD_output, dW_dense1);
  // FPGAI Adam optimizer update for W_dense0.
  for (int i = 0; i < 4; ++i) {
    float grad_value = (float)dW_dense0[i];
    FPGAI_ADAM_M_W_dense0[i] = grad_value;
    FPGAI_ADAM_V_W_dense0[i] = grad_value * grad_value;
    W_dense0[i] = W_dense0[i] - grad_value;
  }
  // FPGAI Adam optimizer update for W_dense1.
  for (int i = 0; i < 2; ++i) {
    float grad_value = (float)dW_dense1[i];
    FPGAI_ADAM_M_W_dense1[i] = grad_value;
    FPGAI_ADAM_V_W_dense1[i] = grad_value * grad_value;
    W_dense1[i] = W_dense1[i] - grad_value;
  }
}
'''
    lowered = _fpgai_f5a_materialize_dense_fused_update(
        source,
        raw_cfg={
            "training": {
                "optimizer": {"type": "adam"},
                "batch": {"mode": "direct", "size": 1},
                "gradient_accumulation": {"steps": 1},
                "storage": {"parameter_gradient": "recompute"},
                "gradients": {"computation": "fused_update", "export_policy": "recompute"},
            }
        },
    )
    assert "dense_weight_grad_typed" not in lowered
    assert "static grad_wgt_t dW_dense0" not in lowered
    assert "static grad_wgt_t dW_dense1" not in lowered
    assert lowered.count("const float fused_grad_value") == 2
    assert "FPGAI_ADAM_M_W_dense0[gradient_index]" in lowered
    assert "FPGAI_ADAM_M_W_dense1[gradient_index]" in lowered


def test_fused_update_removes_dead_dw_assignment_loops_before_recompute_substitution() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_f5a_materialize_dense_fused_update

    source = r'''
static act_t BUF_input[2];
static grad_act_t GRAD_output[2];
static wgt_t W_dense0[4];
static grad_wgt_t dW_dense0[4];
static acc_t ACC_dW_dense0[4];
static opt_t FPGAI_ADAM_M_W_dense0[4];
static opt_t FPGAI_ADAM_V_W_dense0[4];
extern "C" void deeplearn(int mode) {
  fpgai::dense_weight_grad_typed<2, 2, act_t, grad_act_t, grad_wgt_t, acc_t, 1, 1, 1, 1, 1, 1>(BUF_input, GRAD_output, dW_dense0);
  for (int i = 0; i < 4; ++i) dW_dense0[i] = (grad_wgt_t)0;
  for (int i = 0; i < 4; ++i) dW_dense0[i] = (grad_wgt_t)(((float)ACC_dW_dense0[i]) / 1.0f);
  // FPGAI Adam optimizer update for W_dense0.
  for (int i = 0; i < 4; ++i) {
    float grad_value = (float)dW_dense0[i];
    FPGAI_ADAM_M_W_dense0[i] = grad_value;
    FPGAI_ADAM_V_W_dense0[i] = grad_value * grad_value;
    W_dense0[i] = W_dense0[i] - grad_value;
  }
  for (int i = 0; i < 4; ++i) write_f32(out, (float)dW_dense0[i], false);
}
'''
    lowered = _fpgai_f5a_materialize_dense_fused_update(
        source,
        raw_cfg={
            "training": {
                "optimizer": {"type": "adam"},
                "batch": {"mode": "direct", "size": 1},
                "gradient_accumulation": {"steps": 1},
                "storage": {"parameter_gradient": "recompute"},
                "gradients": {"computation": "fused_update", "export_policy": "recompute"},
            }
        },
    )
    assert "removed obsolete materialized-gradient loop for dW_dense0" in lowered
    assert ") = (grad_wgt_t)" not in lowered
    assert "const float fused_grad_value" in lowered
    assert "write_f32(out, (float)(grad_wgt_t)" in lowered


def test_training_plan_records_execution_schedule_and_gradient_mechanism(tmp_path) -> None:
    from types import SimpleNamespace
    from fpgai.engine.training import build_training_plan, emit_training_artifacts

    graph = SimpleNamespace(ops=[])
    raw = {
        "training": {
            "optimizer": {"type": "adam", "learning_rate": 0.001},
            "loss": {"type": "mse"},
            "batch": {"mode": "direct", "size": 1, "epochs": 1},
            "gradient_accumulation": {"steps": 1},
            "storage": {"parameter_gradient": "recompute"},
            "gradients": {
                "computation": "fused_update",
                "materialization": "streamed",
                "export_policy": "recompute",
            },
        }
    }
    plan = build_training_plan(graph, raw)
    assert plan.execution_schedule["epochs"] == 1
    assert plan.execution_schedule["batch_size"] == 1
    assert plan.gradient_mechanism["computation"] == "fused_update"
    assert plan.gradient_mechanism["direct_optimizer_consumption"] is True
    assert plan.gradient_mechanism["complete_parameter_gradient_buffer"] is False
    path = emit_training_artifacts(tmp_path, plan)
    text = path.read_text()
    assert '"gradient_mechanism"' in text
    assert '"execution_schedule"' in text


def test_fused_training_plan_reports_physical_gradient_storage_and_equivalence_contract(tmp_path) -> None:
    import json
    from types import SimpleNamespace
    from fpgai.engine.training import build_training_plan, emit_training_artifacts

    graph = SimpleNamespace(ops=[])
    raw = {
        "memory": {"gradient_storage": "uram", "optimizer_state_storage": "uram"},
        "training": {
            "optimizer": {"type": "adam", "learning_rate": 0.001},
            "loss": {"type": "cross_entropy"},
            "batch": {"mode": "direct", "size": 1, "epochs": 1, "seed": 42},
            "gradients": {
                "computation": "fused_update",
                "materialization": "streamed",
                "export_policy": "recompute",
            },
        },
    }
    plan = build_training_plan(graph, raw)
    mechanism = plan.gradient_mechanism
    assert mechanism["parameter_gradient_storage"] == "none"
    assert mechanism["configured_gradient_region"] == "uram"
    assert mechanism["persistent_optimizer_state_storage"] == "uram"
    assert plan.execution_schedule["workload_resolution"] == "kernel_invocation"
    assert plan.execution_schedule["kernel_calls_per_optimizer_step"] == 1
    assert plan.execution_schedule["forward_backward_calls_per_kernel_call"] == 1
    assert plan.execution_schedule["optimizer_updates_per_kernel_call"] == 1

    emit_training_artifacts(tmp_path, plan)
    payload = json.loads(
        (tmp_path / "training" / "gradient_mechanism_equivalence.json").read_text()
    )
    assert payload["current_mechanism"] == "fused_update"
    assert payload["contract_status"] == "resolved"
    assert payload["numeric_equivalence_status"] == "capture_pending"
    assert len(payload["workload_fingerprint_sha256"]) == 64
    assert payload["required_comparisons"]["weights_max_abs_error"] is None
    assert "identical workload_fingerprint_sha256" in payload["comparison_rule"]


def test_training_artifacts_record_extensible_implementation_stack(tmp_path) -> None:
    import json
    from types import SimpleNamespace
    from fpgai.engine.training import build_training_plan, emit_training_artifacts

    raw = {
        "targets": {"platform": {"board": "kv260"}},
        "implementations": {
            "model_family": "community.models.compact_mlp",
            "model": "community.models.compact_mlp_vhdl",
            "operators": {"Dense": "community.dense.streaming_vhdl"},
            "memory_policy": "community.memory.lifetime_aliasing",
            "streaming_policy": "community.streaming.layer_pipeline",
            "transport_policy": "community.transport.double_buffered_dma",
            "numerical_policy": "community.numeric.block_fp8",
            "backend": "community.backend.mixed_rtl",
            "toolchain": "vivado",
        },
        "training": {
            "optimizer": {"type": "adam", "learning_rate": 0.001},
            "loss": {"type": "mse"},
            "batch": {"mode": "direct", "size": 1, "epochs": 1},
            "gradients": {"computation": "fused_update"},
        },
    }
    plan = build_training_plan(SimpleNamespace(ops=[]), raw)
    stack = plan.implementation_stack
    assert stack["model_family"] == "community.models.compact_mlp"
    assert stack["operator_implementations"]["Dense"] == "community.dense.streaming_vhdl"
    assert stack["memory_policy"] == "community.memory.lifetime_aliasing"
    assert stack["streaming_policy"] == "community.streaming.layer_pipeline"
    assert stack["transport_policy"] == "community.transport.double_buffered_dma"
    assert stack["training_mechanism"] == "fused_update"
    assert stack["backend"] == "community.backend.mixed_rtl"
    assert stack["board"] == "kv260"

    emit_training_artifacts(tmp_path, plan)
    payload = json.loads((tmp_path / "training" / "gradient_mechanism_equivalence.json").read_text())
    assert payload["implementation_stack"] == stack
    assert len(payload["implementation_stack_fingerprint_sha256"]) == 64
    # Workload and implementation identities are intentionally separate.
    assert payload["implementation_stack_fingerprint_sha256"] != payload["workload_fingerprint_sha256"]
