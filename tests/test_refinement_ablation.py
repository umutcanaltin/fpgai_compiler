"""Scientific source-level IR comparison requires constant functional semantics."""
import shutil
import pytest
from fpgai_research.experiments.refinement_ablation import run_refinement_ablation
from fpgai_research.model_ir.artifacts import read_ir_artifact


def test_reproducible_operator_local_stage_artifacts(tmp_path):
    first = run_refinement_ablation(tmp_path / "a")
    second = run_refinement_ablation(tmp_path / "b")
    assert first == second
    diff = first["comparison"]
    assert diff["functional_preserved"]
    assert diff["changed_effective_mapping_operators"] == ["layer0"]
    assert diff["changed_lowered_operators"] == ["layer0"]
    assert diff["hls_source_changed"]
    assert first["host_cpp"] is None
    assert first["vendor_tools_executed"] is False
    assert read_ir_artifact(tmp_path / "a" / "baseline_ir.json")
    with pytest.raises(FileExistsError, match="Refusing"):
        run_refinement_ablation(tmp_path / "a")


@pytest.mark.skipif(shutil.which("g++") is None, reason="GCC not installed")
def test_ablation_both_generated_cpp_versions_correct(tmp_path):
    receipt = run_refinement_ablation(tmp_path / "study", host_cpp=True)
    assert receipt["host_cpp"]["baseline"]["accepted"]
    assert receipt["host_cpp"]["operator_local"]["accepted"]
    assert receipt["host_cpp"]["baseline"]["cpp_sha256"] != receipt["host_cpp"]["operator_local"]["cpp_sha256"]
    assert receipt["evidence"] == "E0/E1+GCC-host-float"
