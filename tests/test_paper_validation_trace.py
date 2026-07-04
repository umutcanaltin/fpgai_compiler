from __future__ import annotations

import json
from pathlib import Path

from fpgai.reporting.paper_validation import build_paper_validation_trace, write_paper_validation_trace


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _minimal_example(tmp_path: Path, name: str, *, pipeline_mode: str = "inference", vivado_project: bool = False) -> Path:
    out = tmp_path / name
    (out / "hls" / "src").mkdir(parents=True)
    (out / "hostcpp").mkdir(parents=True)
    (out / "runtime_package").mkdir(parents=True)
    (out / "reports").mkdir(parents=True)
    (out / "hls" / "src" / "deeplearn.cpp").write_text("void deeplearn() {}\n", encoding="utf-8")
    _write_json(out / "runtime_package" / "package_manifest.json", {"runtime_io": {}})
    _write_json(out / "manifest.json", {
        "pipeline_mode": pipeline_mode,
        "top_kernel_name": "deeplearn",
        "configuration": {"effective": {"build_stages": {
            "cpp": True,
            "host_cpp": True,
            "hls_project": vivado_project,
            "hls_synthesis": False,
            "vivado_project": vivado_project,
            "vivado_implementation": False,
            "bitstream": False,
            "runtime_package": True,
            "reports": True,
            "testbench": True,
        }}},
    })
    for name_, data in {
        "movement_contract_validation": {"status": "passed", "passed": True, "blocking_failure": False},
        "data_movement_plan": {"status": "validated"},
        "ps_pl_transfer_plan": {"status": "validated"},
        "generated_cpp_validation": {"status": "passed"},
        "generated_cpp_readability": {"status": "passed"},
        "generated_hls_explanation": {"status": "present"},
        "hardware_knob_contract": {"knobs": [{"path": "optimization.pipeline.ii", "source": "manual_yaml", "status": "applied"}]},
        "resource_prediction": {"status": "estimated"},
        "timing_prediction": {"status": "estimated"},
        "board_fit": {"status": "near_limit"},
        "hls_synthesis_report": {"status": "not_requested"},
        "vivado_implementation_report": {"status": "not_requested"},
        "vivado_validation_report": {"status": "not_requested"},
        "bitstream_report": {"status": "not_requested"},
    }.items():
        _write_json(out / "reports" / f"{name_}.json", data)
    if pipeline_mode == "training_on_device":
        (out / "training").mkdir()
        _write_json(out / "training" / "training_plan.json", {"optimizer": "sgd"})
        _write_json(out / "runtime_package" / "buffer_plan.json", {"buffers": [{"role": "labels"}, {"role": "weights"}, {"name": "gradients_mem"}]})
    if vivado_project:
        (out / "vivado").mkdir()
        (out / "vivado" / "project.tcl").write_text("# project\n", encoding="utf-8")
        (out / "vivado" / "bd.tcl").write_text("# bd\n", encoding="utf-8")
        (out / "vivado" / "run_vivado.tcl").write_text("# run\n", encoding="utf-8")
        _write_json(out / "reports" / "vivado_bd_validation.json", {"status": "passed", "validation_boundary": "structural only"})
    else:
        _write_json(out / "reports" / "vivado_bd_validation.json", {"status": "not_requested"})
    return out


def test_paper_validation_trace_keeps_generated_vivado_project_separate_from_vivado_implementation_passed(tmp_path: Path) -> None:
    out = _minimal_example(tmp_path, "vivado_project", vivado_project=True)
    chain = build_paper_validation_trace([out])
    example = chain["examples"][0]
    assert chain["passed"] is True
    assert example["validation_status"]["vivado_implementation_passed"] is False
    assert example["support_level"] == "compiler_only"
    vivado_claim = next(c for c in example["claims"] if c["claim_id"] == "vivado_project_handoff_generated")
    assert vivado_claim["validation_level"] == "vivado_project_generated"
    assert "Vivado implementation met timing" in vivado_claim["pending_validation"]


def test_paper_validation_trace_training_claim_is_static_not_runtime_convergence(tmp_path: Path) -> None:
    out = _minimal_example(tmp_path, "training", pipeline_mode="training_on_device")
    chain = build_paper_validation_trace([out])
    example = chain["examples"][0]
    training_claim = next(c for c in example["claims"] if c["claim_id"] == "training_pipeline_materialized")
    assert training_claim["paper_ready"] is True
    assert training_claim["validation_level"] == "static_artifact_validation"
    assert "FPGA training converged" in training_claim["pending_validation"]
    assert example["validation_status"]["fpga_runtime_validation"] is False


def test_paper_validation_trace_marks_failed_movement_as_not_paper_ready(tmp_path: Path) -> None:
    out = _minimal_example(tmp_path, "bad_movement")
    _write_json(out / "reports" / "movement_contract_validation.json", {"status": "failed", "passed": False, "blocking_failure": True})
    chain = build_paper_validation_trace([out])
    assert chain["passed"] is False
    claim = next(c for c in chain["examples"][0]["claims"] if c["claim_id"] == "data_movement_contract_materialized")
    assert claim["paper_ready"] is False
    assert "reports/movement_contract_validation.json:passed" in claim["required_validation"]



def test_paper_validation_trace_does_not_fail_when_vivado_not_requested_placeholder_exists(tmp_path: Path) -> None:
    out = _minimal_example(tmp_path, "cpp_only", vivado_project=False)
    chain = build_paper_validation_trace([out])
    example = chain["examples"][0]
    assert chain["passed"] is True
    assert all(c["claim_id"] != "vivado_project_handoff_generated" for c in example["claims"])
    assert example["artifact_status"]["vivado_bd_validation"] == "not_requested"

def test_paper_validation_trace_writes_json_and_markdown(tmp_path: Path) -> None:
    out = _minimal_example(tmp_path, "example")
    json_path = tmp_path / "paper_validation_trace.json"
    md_path = tmp_path / "paper_validation_trace.md"
    chain = write_paper_validation_trace([out], json_path, md_path)
    assert chain["passed"] is True
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["artifact_kind"] == "paper_validation_trace"
    md = md_path.read_text(encoding="utf-8")
    assert "FPGAI Paper Validation Trace" in md
    assert "compiler_estimates_available_with_validation_boundary" in md
