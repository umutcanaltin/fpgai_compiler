from pathlib import Path
import csv
import json

from fpgai.experiments.design_matrix import expand_design_matrix
from fpgai.experiments.sweep_runner import SweepRunner


def test_sweep_runner_dry_run_creates_outputs(tmp_path):
    cfg = {
        "name": "dry",
        "command_template": "echo {x}",
        "defaults": {"config_path": "configs/examples/default_compile.yml", "model_path": "model.onnx", "board": "kv260"},
        "parameters": {"x": [1, 2, 3]},
    }
    points = expand_design_matrix(cfg)
    runner = SweepRunner(tmp_path / "experiments" / "dry", repo_root=tmp_path, dry_run=True)
    payload = runner.run_points(points)
    assert len(payload["results"]) == 3
    assert all(r["status"] == "passed" for r in payload["results"])
    assert (tmp_path / "experiments" / "dry" / "summary.md").exists()
    assert (tmp_path / "experiments" / "dry" / "logs").exists()


def test_sweep_runner_failure_does_not_stop(tmp_path):
    cfg = {
        "name": "fail_continue",
        "defaults": {"board": "kv260"},
        "design_points": [
            {"name": "bad", "command": "python -c 'import sys; sys.exit(3)'"},
            {"name": "good", "command": "python -c 'print(42)'"},
        ],
    }
    points = expand_design_matrix(cfg)
    runner = SweepRunner(tmp_path / "exp", repo_root=tmp_path, dry_run=False)
    payload = runner.run_points(points)
    statuses = {r["design_name"]: r["status"] for r in payload["results"]}
    assert statuses["bad"] == "failed"
    assert statuses["good"] == "passed"


def test_sweep_runner_resume_retries_failed_without_duplicate_success_records(tmp_path):
    cfg = {
        "name": "resume_retry",
        "defaults": {"board": "kv260"},
        "design_points": [
            {"name": "point", "command": "python -c 'import sys; sys.exit(3)'"},
        ],
    }
    points = expand_design_matrix(cfg)
    experiment_dir = tmp_path / "exp_retry"

    first = SweepRunner(experiment_dir, repo_root=tmp_path, dry_run=False).run_points(points)
    assert first["failed_count"] == 1
    assert first["attempt_count"] == 1

    fixed_cfg = {
        "name": "resume_retry",
        "defaults": {"board": "kv260"},
        "design_points": [
            {"name": "point", "command": "python -c 'print(42)'"},
        ],
    }
    fixed_points = expand_design_matrix(fixed_cfg)
    second = SweepRunner(experiment_dir, repo_root=tmp_path, dry_run=False).run_points(fixed_points)
    assert second["attempt_count"] == 2
    assert second["result_count"] == 1
    assert second["failed_count"] == 0
    assert second["passed_count"] == 1
    assert second["results"][0]["status"] == "passed"

    # A third resume sees the latest successful attempt and does not append a
    # synthetic skipped record.
    third = SweepRunner(experiment_dir, repo_root=tmp_path, dry_run=False).run_points(fixed_points)
    assert third["attempt_count"] == 2
    assert third["result_count"] == 1


def test_sweep_runner_requires_declared_artifacts(tmp_path):
    cfg = {
        "name": "artifact_contract",
        "defaults": {"board": "kv260"},
        "design_points": [
            {
                "name": "point",
                "command": "python -c \"from pathlib import Path; p=Path('build/point'); p.mkdir(parents=True, exist_ok=True); print('[OK] Wrote artifacts to: build/point')\"",
            },
        ],
    }
    points = expand_design_matrix(cfg)
    runner = SweepRunner(
        tmp_path / "exp",
        repo_root=tmp_path,
        dry_run=False,
        required_artifacts=["manifest.json", "reports/training_dataset_execution.json"],
    )
    payload = runner.run_points(points)
    record = payload["results"][0]
    assert record["status"] == "failed"
    assert record["returncode"] == 0
    assert record["out_dir"] == str((tmp_path / "build/point").resolve())
    assert record["artifact_validation"]["missing"] == [
        "manifest.json",
        "reports/training_dataset_execution.json",
    ]
    assert "required artifact validation failed" in record["error"]


def test_sweep_runner_resolves_stdout_out_dir_and_passes_artifact_contract(tmp_path):
    command = (
        "python -c \"from pathlib import Path; "
        "p=Path('actual/build/reports'); p.mkdir(parents=True, exist_ok=True); "
        "Path('actual/build/manifest.json').write_text('{}'); "
        "Path('actual/build/reports/training_dataset_execution.json').write_text('{}'); "
        "print('[OK] Wrote artifacts to: actual/build')\""
    )
    cfg = {
        "name": "artifact_contract_pass",
        "defaults": {"board": "kv260"},
        "design_points": [{"name": "point", "command": command}],
    }
    points = expand_design_matrix(cfg)
    runner = SweepRunner(
        tmp_path / "exp",
        repo_root=tmp_path,
        dry_run=False,
        required_artifacts=["manifest.json", "reports/training_dataset_execution.json"],
    )
    payload = runner.run_points(points)
    record = payload["results"][0]
    assert record["status"] == "passed"
    assert record["out_dir"] == str((tmp_path / "actual/build").resolve())
    assert record["artifact_validation"]["status"] == "passed"
    assert record["artifact_validation"]["missing"] == []


def test_materialized_sweep_project_out_dir_is_absolute(tmp_path):
    base = tmp_path / "base.yml"
    base.write_text("project:\n  out_dir: build/old\n", encoding="utf-8")
    cfg = {
        "name": "absolute_artifacts",
        "command_template": "python -c 'print(42)'",
        "materialize_configs": {"enabled": True, "preserve_artifacts": True},
        "defaults": {"config_path": str(base), "board": "kv260"},
        "design_points": [{"name": "point"}],
    }
    points = expand_design_matrix(cfg)
    runner = SweepRunner(
        tmp_path / "experiment",
        repo_root=tmp_path,
        dry_run=True,
        materialize_configs=cfg["materialize_configs"],
        command_template=cfg["command_template"],
    )
    payload = runner.run_points(points)
    record = payload["results"][0]
    out_dir = Path(record["out_dir"])
    assert out_dir.is_absolute()
    generated = Path(record["config_path"])
    if not generated.is_absolute():
        generated = tmp_path / generated
    data = __import__("yaml").safe_load(generated.read_text(encoding="utf-8"))
    assert Path(data["project"]["out_dir"]).is_absolute()


def test_training_learning_ablation_reports_separate_exposure_and_updates(tmp_path):
    from fpgai.experiments.report_writer import write_training_learning_ablation_reports

    out_dir = tmp_path / "build"
    reports = out_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "training_dataset_execution.json").write_text(json.dumps({
        "dataset_sample_count": 10,
        "record_visits_executed": 30,
        "optimizer_updates": 6,
        "epochs_completed": 3,
    }), encoding="utf-8")
    (reports / "training_learning_behavior.json").write_text(json.dumps({
        "headline_domain": "hardware_fixed_point",
        "numeric_validation_status": "passed",
        "domains": {
            "hardware_fixed_point": {
                "initial_dataset_loss": 2.3,
                "final_dataset_loss": 2.2,
                "initial_accuracy": 0.1,
                "final_accuracy": 0.4,
            }
        },
    }), encoding="utf-8")
    (reports / "training_dataset_comparison.json").write_text(json.dumps({
        "status": "passed", "passed": True,
    }), encoding="utf-8")
    (reports / "training_dataset_model_contract.json").write_text(json.dumps({
        "status": "compatible", "claim_scope": "small_sample_learning_smoke_only",
    }), encoding="utf-8")
    (reports / "board_fit.json").write_text(json.dumps({"status": "over_limit"}), encoding="utf-8")

    payload = {"results": [{
        "status": "passed",
        "design_name": "b5",
        "out_dir": str(out_dir),
        "parameters": {"batch_size": 5, "epochs": 3, "max_updates": 6},
    }]}
    paths = write_training_learning_ablation_reports(
        tmp_path / "experiment",
        payload,
        {"comparison_basis": "equal_update_budget"},
    )
    assert paths["csv"].exists()
    text = paths["csv"].read_text(encoding="utf-8")
    assert "record_visits_executed,optimizer_updates" in text
    assert "equal_update_budget" in text
    eligibility = json.loads(paths["eligibility"].read_text(encoding="utf-8"))
    assert eligibility["eligible_count"] == 0
    assert eligibility["rows"][0]["confounders"] == ["small_sample_learning_smoke_only"]


def test_equal_update_budget_sweep_uses_canonical_max_updates_mapping():
    import yaml

    cfg = yaml.safe_load(Path("configs/sweeps/training_batch_equal_update_budget.yml").read_text(encoding="utf-8"))
    mapping = cfg["materialize_configs"]["parameter_mappings"]["max_updates"]
    assert mapping["path"] == "training.batch.max_updates"
    assert cfg["analysis"]["comparison_basis"] == "equal_update_budget"
    assert {point["max_updates"] for point in cfg["design_points"]} == {6}
    assert {point["batch_size"] for point in cfg["design_points"]} == {2, 5, 10}


def test_training_learning_ablation_reports_classify_controls_and_normalize(tmp_path):
    from fpgai.experiments.report_writer import write_training_learning_ablation_reports

    def make_build(name: str, visits: int, updates: int, final_loss: float) -> Path:
        out_dir = tmp_path / name
        reports = out_dir / "reports"
        reports.mkdir(parents=True)
        (reports / "training_dataset_execution.json").write_text(json.dumps({
            "dataset_sample_count": 10,
            "record_visits_executed": visits,
            "optimizer_updates": updates,
            "epochs_completed": visits // 10,
        }), encoding="utf-8")
        (reports / "training_learning_behavior.json").write_text(json.dumps({
            "headline_domain": "hardware_fixed_point",
            "numeric_validation_status": "passed",
            "domains": {"hardware_fixed_point": {
                "initial_dataset_loss": 2.3,
                "final_dataset_loss": final_loss,
                "initial_accuracy": 0.0,
                "final_accuracy": 0.2,
            }},
        }), encoding="utf-8")
        (reports / "training_dataset_comparison.json").write_text(
            json.dumps({"status": "passed", "passed": True}), encoding="utf-8"
        )
        (reports / "training_dataset_model_contract.json").write_text(json.dumps({
            "status": "compatible", "claim_scope": "small_sample_learning_smoke_only",
        }), encoding="utf-8")
        return out_dir

    control = make_build("control", 20, 10, 2.26)
    primary = make_build("primary", 30, 15, 2.25)
    payload = {"results": [
        {"status": "passed", "design_name": "control", "out_dir": str(control),
         "parameters": {"batch_size": 2, "epochs": 2, "seed": 42, "comparison_role": "duration_control"}},
        {"status": "passed", "design_name": "primary", "out_dir": str(primary),
         "parameters": {"batch_size": 2, "epochs": 3, "seed": 42, "comparison_role": "primary_comparison"}},
    ]}
    paths = write_training_learning_ablation_reports(
        tmp_path / "experiment", payload,
        {"comparison_basis": "equal_sample_exposure", "primary_epochs": 3},
    )
    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8", newline="")))
    assert rows[0]["comparison_role"] == "duration_control"
    assert rows[1]["comparison_role"] == "primary_comparison"
    assert float(rows[1]["record_visits_per_update"]) == 2.0
    assert float(rows[1]["loss_reduction_per_optimizer_update"]) > 0.0
    eligibility = json.loads(paths["eligibility"].read_text(encoding="utf-8"))
    assert eligibility["schema_version"] == 2
    assert eligibility["seed_count"] == 1
    assert eligibility["statistical_claim_eligible"] is False
    assert eligibility["capabilities"]["mechanism_validated"] is True


def test_paired_training_batch_ablation_report_keeps_bases_separate(tmp_path):
    from fpgai.experiments.report_writer import write_paired_training_batch_ablation_reports

    fields = [
        "design_name", "comparison_basis", "comparison_role", "batch_size",
        "record_visits_executed", "optimizer_updates", "record_visits_per_update",
        "initial_loss", "final_loss", "loss_reduction",
        "loss_reduction_per_record_visit", "loss_reduction_per_optimizer_update",
        "initial_accuracy", "final_accuracy", "accuracy_delta", "seed_count",
        "replicate_count", "variance_available", "statistical_claim_eligible",
        "claim_scope", "paper_claim_eligible", "confounders",
    ]
    dirs = {}
    for basis in ("equal_sample_exposure", "equal_update_budget"):
        d = tmp_path / basis
        d.mkdir()
        with (d / "training_learning_ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "design_name": basis, "comparison_basis": basis,
                "comparison_role": "primary_comparison", "batch_size": 2,
                "record_visits_executed": 30, "optimizer_updates": 6,
                "record_visits_per_update": 5, "initial_loss": 2.3,
                "final_loss": 2.2, "loss_reduction": 0.1,
                "loss_reduction_per_record_visit": 0.0033,
                "loss_reduction_per_optimizer_update": 0.0167,
                "initial_accuracy": 0, "final_accuracy": 0.2,
                "accuracy_delta": 0.2, "seed_count": 1, "replicate_count": 1,
                "variance_available": False, "statistical_claim_eligible": False,
                "claim_scope": "small_sample_learning_smoke_only",
                "paper_claim_eligible": False, "confounders": "small_sample_learning_smoke_only",
            })
        dirs[basis] = d
    paths = write_paired_training_batch_ablation_reports(tmp_path / "paired", dirs)
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "equal_sample_exposure" in text
    assert "equal_update_budget" in text
    interpretation = json.loads(paths["interpretation"].read_text(encoding="utf-8"))
    assert interpretation["available_comparison_bases"] == ["equal_sample_exposure", "equal_update_budget"]
    assert interpretation["statistical_comparison_supported"] is False



def test_paired_training_batch_ablation_report_accepts_legacy_summary_without_role(tmp_path):
    from fpgai.experiments.report_writer import write_paired_training_batch_ablation_reports

    legacy_dir = tmp_path / "legacy_equal_updates"
    legacy_dir.mkdir()
    legacy_fields = [
        "design_name", "comparison_basis", "batch_size",
        "record_visits_executed", "optimizer_updates",
        "initial_loss", "final_loss", "initial_accuracy", "final_accuracy",
        "claim_scope", "paper_claim_eligible",
    ]
    with (legacy_dir / "training_learning_ablation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow({
            "design_name": "legacy_b2",
            "comparison_basis": "equal_update_budget",
            "batch_size": 2,
            "record_visits_executed": 12,
            "optimizer_updates": 6,
            "initial_loss": 2.3,
            "final_loss": 2.2,
            "initial_accuracy": 0.0,
            "final_accuracy": 0.1,
            "claim_scope": "small_sample_learning_smoke_only",
            "paper_claim_eligible": False,
        })

    paths = write_paired_training_batch_ablation_reports(
        tmp_path / "paired", {"equal_update_budget": legacy_dir}
    )
    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8", newline="")))
    assert rows[0]["comparison_role"] == "primary_comparison"
    assert rows[0]["seed_count"] == "1"
    assert "primary_comparison" in paths["markdown"].read_text(encoding="utf-8")

def test_strict_equal_exposure_sweep_has_three_primary_batch_sizes():
    import yaml

    cfg = yaml.safe_load(Path("configs/sweeps/training_batch_equal_exposure_strict3.yml").read_text(encoding="utf-8"))
    assert cfg["analysis"]["comparison_basis"] == "equal_sample_exposure"
    assert cfg["analysis"]["primary_epochs"] == 3
    assert {point["batch_size"] for point in cfg["design_points"]} == {2, 5, 10}
    assert {point["epochs"] for point in cfg["design_points"]} == {3}
    assert {point["comparison_role"] for point in cfg["design_points"]} == {"primary_comparison"}
