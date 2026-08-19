from __future__ import annotations

from fpgai.benchmark.reference_model_profiles import load_reference_model_profile


def test_yolo11n_reference_profile_is_benchmark_only() -> None:
    profile = load_reference_model_profile("examples/reference/model_profiles/yolo11n.yml")
    assert profile["name"] == "yolo11n"
    assert profile["compiler_special_case"] is False
    assert profile["policy"]["custom_shapes_and_layers_allowed"] is True
    assert "Concat" in profile["expected_generic_operator_families_after_export"]
    assert "Resize" in profile["expected_generic_operator_families_after_export"]


def test_smollm2_reference_profile_records_real_architecture_without_compiler_path() -> None:
    profile = load_reference_model_profile("examples/reference/model_profiles/smollm2_135m.yml")
    facts = profile["reference_facts"]
    assert profile["compiler_special_case"] is False
    assert facts["hidden_size"] == 576
    assert facts["intermediate_size"] == 1536
    assert facts["num_hidden_layers"] == 30
    assert facts["num_attention_heads"] == 9
    assert facts["num_key_value_heads"] == 3
    assert facts["max_position_embeddings"] == 8192
    assert profile["policy"]["weight_and_state_storage_are_user_choices"] is True
