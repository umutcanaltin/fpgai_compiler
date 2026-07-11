from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from fpgai.config.loader import load_config
from fpgai.paper.experiment_setup import generate_experiment_setup_artifacts


MATRIX = Path("paper_experiments/paper_experiment_matrix.yml")


def _load_matrix() -> dict:
    return yaml.safe_load(MATRIX.read_text(encoding="utf-8"))


def test_paper_experiment_matrix_has_controlled_inference_and_training_rows() -> None:
    matrix = _load_matrix()
    rows = matrix["experiments"]
    ids = [row["id"] for row in rows]

    assert len(ids) == len(set(ids))
    assert sum(1 for row in rows if row["section"] == "inference") >= 9
    assert sum(1 for row in rows if row["section"] == "training") >= 8
    assert any(row["expected_claim_level"] == "level_3_bitstream_package" for row in rows)
    assert any(row["expected_claim_level"] == "level_4_board_execution" for row in rows)
    assert any(row["knob_axis"] == "precision_fx8" for row in rows)
    assert any(row["knob_axis"] == "optimizer_adam" for row in rows)
    assert any(row["knob_axis"] == "real_training_curve" for row in rows)


def test_paper_experiment_configs_are_valid_and_match_sections() -> None:
    matrix = _load_matrix()
    for row in matrix["experiments"]:
        cfg = load_config(row["config"])
        expected_mode = "inference" if row["section"] == "inference" else "training_on_device"
        assert cfg.pipeline.mode == expected_mode
        metadata = cfg.raw.get("metadata", {}).get("paper_experiment", {})
        assert metadata.get("id") == row["id"]
        assert metadata.get("expected_claim_level") == row["expected_claim_level"]
        stages = cfg.raw.get("build", {}).get("stages", {})
        assert stages.get("hls_synthesis") is True
        assert stages.get("vivado_implementation") is True
        if row["expected_claim_level"] in {"level_3_bitstream_package", "level_4_board_execution"}:
            assert stages.get("bitstream") is True


def test_generate_paper_experiment_setup_artifacts(tmp_path: Path) -> None:
    manifest = generate_experiment_setup_artifacts(MATRIX, output_dir=tmp_path / "setup")

    assert manifest["status"] == "ready"
    assert manifest["error_count"] == 0
    assert manifest["experiment_count"] == 19
    assert manifest["by_section"]["inference"] == 10
    assert manifest["by_section"]["training"] == 9
    assert manifest["by_expected_claim_level"]["level_4_board_execution"] == 2
    assert (tmp_path / "setup/experiment_setup_manifest.md").exists()
    assert (tmp_path / "setup/compile_command_plan.md").exists()
    assert (tmp_path / "setup/compile_command_plan.sh.txt").exists()

    with (tmp_path / "setup/paper_experiment_setup_rows.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 19
    assert rows[0]["id"] == "I0_baseline_fx16_embedded"
    assert rows[0]["config_valid"] == "True"
    assert any(row["knob_axis"] == "real_inference_runtime" for row in rows)
    assert any(row["knob_axis"] == "real_training_curve" for row in rows)

    payload = json.loads((tmp_path / "setup/experiment_setup_manifest.json").read_text(encoding="utf-8"))
    assert payload["tables"]["compile_command_plan_csv"].endswith("compile_command_plan.csv")


def test_compile_command_plan_preserves_compile_config_token_boundary(tmp_path: Path) -> None:
    generate_experiment_setup_artifacts(MATRIX, output_dir=tmp_path / "setup")

    plan_md = (tmp_path / "setup/compile_command_plan.md").read_text(encoding="utf-8")
    plan_sh = (tmp_path / "setup/compile_command_plan.sh.txt").read_text(encoding="utf-8")
    selected_sh = (tmp_path / "setup/compile_selected_smoke.sh.txt").read_text(encoding="utf-8")
    board_sh = (tmp_path / "setup/compile_board_runtime_candidates.sh.txt").read_text(encoding="utf-8")

    for text in (plan_md, plan_sh, selected_sh, board_sh):
        assert "compile--config" not in text
        assert "python -m fpgai.cli compile --config" in text

    assert "set -e" not in selected_sh
    assert "run_fpgai_compile" in selected_sh
    assert "compile_status.tsv" in selected_sh
    assert "continuing to remaining paper rows" in selected_sh

    assert "T8_real_fpga_training_curve_candidate" in plan_md
    assert "python -m fpgai.cli compile --config paper_experiments/training/T8_real_fpga_training_curve_candidate.yml" in plan_md
    assert "I0_baseline_fx16_embedded" in selected_sh
    assert "T8_real_fpga_training_curve_candidate" in board_sh

    with (tmp_path / "setup/compile_command_plan.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(row["command_valid"] == "True" for row in rows)

    manifest = json.loads((tmp_path / "setup/experiment_setup_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tables"]["compile_selected_smoke_sh_txt"].endswith("compile_selected_smoke.sh.txt")
    assert manifest["tables"]["compile_board_runtime_candidates_sh_txt"].endswith("compile_board_runtime_candidates.sh.txt")


def test_compile_plan_validates_artifact_status_after_successful_process(tmp_path: Path) -> None:
    generate_experiment_setup_artifacts(MATRIX, output_dir=tmp_path / "setup")
    selected_sh = (tmp_path / "setup/compile_selected_smoke.sh.txt").read_text(encoding="utf-8")

    assert "--validate-artifact" in selected_sh
    assert "--configured-stage" in selected_sh
    assert "--expected-claim-level" in selected_sh
    assert "project_out_dir" in selected_sh
    assert "artifact_status" in selected_sh


def test_validate_compile_artifacts_rejects_failed_hls_for_vivado_stage(tmp_path: Path) -> None:
    from fpgai.paper.experiment_setup import validate_compile_artifacts

    out = tmp_path / "failed_inference"
    (out / "hls/src").mkdir(parents=True)
    (out / "hls/src/deeplearn.cpp").write_text("void deeplearn() {}", encoding="utf-8")
    (out / "reports").mkdir(parents=True)
    (out / "reports/paper_experiment_row.json").write_text(
        json.dumps({
            "source_generated": True,
            "hls_ok": False,
            "vivado_implemented": True,
            "bitstream_generated": True,
            "runtime_package_created": True,
            "runtime_package_validated": True,
            "claim_level": "level_0_compiler_artifact",
        }),
        encoding="utf-8",
    )

    payload = validate_compile_artifacts(out, configured_stage="vivado_implementation", expected_claim_level="level_2_vivado_implementation")

    assert payload["status"] == "failed"
    assert any(check["name"] == "hls_ok" for check in payload["failed_checks"])


def test_validate_compile_artifacts_accepts_bitstream_package_with_validated_runtime(tmp_path: Path) -> None:
    from fpgai.paper.experiment_setup import validate_compile_artifacts

    out = tmp_path / "valid_bitstream"
    (out / "reports").mkdir(parents=True)
    (out / "reports/paper_experiment_row.json").write_text(
        json.dumps({
            "source_generated": True,
            "hls_ok": True,
            "vivado_implemented": True,
            "bitstream_generated": True,
            "runtime_package_created": True,
            "runtime_package_validated": True,
            "claim_level": "level_3_bitstream_package",
        }),
        encoding="utf-8",
    )

    payload = validate_compile_artifacts(out, configured_stage="bitstream_package", expected_claim_level="level_3_bitstream_package")

    assert payload["status"] == "passed"
    assert payload["failed_checks"] == []
