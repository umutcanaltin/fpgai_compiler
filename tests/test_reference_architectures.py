from __future__ import annotations

import pytest

from fpgai.benchmark.reference_architectures import load_reference_architecture


@pytest.mark.parametrize(
    "path,family,label",
    [
        ("examples/reference/architectures/llm_like_gqa_decoder.yml", "llm_like", "LLM-like GQA Decoder"),
        ("examples/reference/architectures/llm_like_mha_decoder.yml", "llm_like", "LLM-like MHA Decoder"),
        ("examples/reference/architectures/yolo_like_multiscale_detector.yml", "yolo_like", "YOLO-like Multi-scale Detector"),
        ("examples/reference/architectures/single_stage_detector.yml", "single_stage_detection", "Generic Single-stage Detector"),
    ],
)
def test_reference_architecture_profiles_are_generic_and_benchmark_ready(path: str, family: str, label: str) -> None:
    profile = load_reference_architecture(path)
    assert profile.family == family
    assert profile.benchmark_label == label
    assert profile.compiler_special_case is False
    assert profile.graph_source == "maintained_generic_graph"
    assert profile.features


def test_reference_architecture_loader_rejects_model_special_case(tmp_path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text(
        "schema: fpgai.reference-architecture/v1\nname: bad\nfamily: llm_like\ncompiler_special_case: true\n"
        "graph_source: maintained_generic_graph\nbenchmark_label: Bad\nfeatures: [attention]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="REFARCH004"):
        load_reference_architecture(path)


def test_reference_architecture_suite_records_non_sweep_benchmark_policy() -> None:
    import yaml
    data = yaml.safe_load(open("examples/reference/architectures/suite.yml", "r", encoding="utf-8"))
    assert data["schema"] == "fpgai.reference-architecture-suite/v1"
    assert len(data["architectures"]) == 4
    assert data["policy"]["compiler_special_cases"] is False
    assert data["policy"]["architecture_sweeps_are_compiler_feature"] is False
    assert data["policy"]["user_hardware_knobs_remain_selectable"] is True
    assert data["policy"]["named_model_claim_requires_exact_graph_validation"] is True
