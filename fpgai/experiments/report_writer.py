"""Markdown reporting for FPGAI experiment sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("\n", " ")


def write_summary_markdown(experiment_dir: str | Path, results_payload: Mapping[str, Any] | None = None) -> Path:
    experiment_dir = Path(experiment_dir)
    results_path = experiment_dir / "results.json"
    if results_payload is None:
        results_payload = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"results": []}
    records = list(results_payload.get("results", []))
    total = len(records)
    passed = sum(1 for r in records if r.get("status") == "passed")
    failed = sum(1 for r in records if r.get("status") == "failed")
    skipped = sum(1 for r in records if r.get("status") == "skipped")
    lines = [
        "# FPGAI Experiment Summary",
        "",
        f"Experiment directory: `{experiment_dir}`",
        "",
        "## Status",
        "",
        f"- Total design points: {total}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Skipped/resumed: {skipped}",
        "",
        "## Results",
        "",
        "| # | Design | Status | Board | Config | Duration (s) |",
        "|---:|---|---|---|---|---:|",
    ]
    for r in records:
        lines.append(
            "| {idx} | `{name}` | {status} | {board} | `{cfg}` | {dur} |".format(
                idx=_fmt(r.get("design_index")),
                name=_fmt(r.get("design_name")),
                status=_fmt(r.get("status")),
                board=_fmt(r.get("board")),
                cfg=_fmt(r.get("config_path")),
                dur=_fmt(r.get("duration_sec")),
            )
        )
    lines.extend([
        "",
        "## Reproducibility metadata",
        "",
        "Every row in `results.json` and `results.csv` includes commit hash, config path, model path, tool version, and board target when available.",
    ])
    out = experiment_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out



def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _domain_metrics(learning: Mapping[str, Any], domain: str) -> Mapping[str, Any]:
    domains = learning.get("domains")
    if isinstance(domains, Mapping):
        value = domains.get(domain)
        if isinstance(value, Mapping):
            return value
    return {}


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _classify_comparison_role(
    comparison_basis: str,
    parameters: Mapping[str, Any],
    analysis_config: Mapping[str, Any],
) -> str:
    explicit = parameters.get("comparison_role")
    if explicit:
        return str(explicit)
    primary_epochs = analysis_config.get("primary_epochs")
    if comparison_basis == "equal_sample_exposure" and primary_epochs is not None:
        return "primary_comparison" if parameters.get("epochs") == primary_epochs else "duration_control"
    return "primary_comparison"


def write_training_learning_ablation_reports(
    experiment_dir: str | Path,
    results_payload: Mapping[str, Any],
    analysis_config: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write canonical paper-facing summaries for training ablation sweeps.

    The report keeps record exposure and optimizer-update budgets independent,
    classifies control rows, and exposes normalized metrics without pretending
    that one deterministic run provides statistical support.
    """

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(analysis_config or {})
    comparison_basis = str(cfg.get("comparison_basis") or "unspecified")
    rows: list[dict[str, Any]] = []

    for record in results_payload.get("results", []):
        if not isinstance(record, Mapping) or record.get("status") != "passed":
            continue
        out_value = record.get("out_dir") or record.get("project_out_dir")
        if not out_value:
            continue
        out_dir = Path(str(out_value))
        reports = out_dir / "reports"
        execution = _load_json(reports / "training_dataset_execution.json")
        learning = _load_json(reports / "training_learning_behavior.json")
        comparison = _load_json(reports / "training_dataset_comparison.json")
        contract = _load_json(reports / "training_dataset_model_contract.json")
        board_fit = _load_json(reports / "board_fit.json")

        headline_domain = str(learning.get("headline_domain") or learning.get("decision_reference_domain") or "hardware_fixed_point")
        headline = _domain_metrics(learning, headline_domain)
        parameters = record.get("parameters") if isinstance(record.get("parameters"), Mapping) else {}

        initial_loss = headline.get("initial_dataset_loss", learning.get("initial_dataset_loss"))
        final_loss = headline.get("final_dataset_loss", learning.get("final_dataset_loss"))
        initial_accuracy = headline.get("initial_accuracy", learning.get("initial_accuracy"))
        final_accuracy = headline.get("final_accuracy", learning.get("final_accuracy"))
        loss_delta = None
        loss_reduction = None
        if isinstance(initial_loss, (int, float)) and isinstance(final_loss, (int, float)):
            loss_delta = float(final_loss) - float(initial_loss)
            loss_reduction = float(initial_loss) - float(final_loss)
        accuracy_delta = None
        if isinstance(initial_accuracy, (int, float)) and isinstance(final_accuracy, (int, float)):
            accuracy_delta = float(final_accuracy) - float(initial_accuracy)

        sample_count = execution.get("dataset_sample_count")
        record_visits = execution.get("record_visits_executed", execution.get("dataset_records_consumed"))
        updates = execution.get("optimizer_updates")
        epochs = execution.get("epochs_completed")
        batch_size = parameters.get("batch_size")
        max_updates = parameters.get("max_updates")
        comparison_role = _classify_comparison_role(comparison_basis, parameters, cfg)

        eligible = bool(comparison.get("passed")) and contract.get("status") == "compatible"
        confounders: list[str] = []
        if comparison_basis == "equal_sample_exposure" and record_visits is None:
            confounders.append("missing_record_visits")
        if comparison_basis == "equal_update_budget" and updates is None:
            confounders.append("missing_optimizer_updates")
        if comparison_role != "primary_comparison":
            confounders.append(comparison_role)
        if contract.get("claim_scope") not in {None, "", "convergence_and_generalization"}:
            confounders.append(str(contract.get("claim_scope")))

        rows.append({
            "design_name": record.get("design_name"),
            "comparison_basis": comparison_basis,
            "comparison_role": comparison_role,
            "batch_size": batch_size,
            "epochs_requested": parameters.get("epochs"),
            "max_updates_requested": max_updates,
            "seed": parameters.get("seed"),
            "dataset_sample_count": sample_count,
            "record_visits_executed": record_visits,
            "optimizer_updates": updates,
            "epochs_completed": epochs,
            "record_visits_per_update": _safe_div(record_visits, updates),
            "samples_per_update": _safe_div(record_visits, updates),
            "headline_domain": headline_domain,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_delta": loss_delta,
            "loss_reduction": loss_reduction,
            "loss_reduction_per_record_visit": _safe_div(loss_reduction, record_visits),
            "loss_reduction_per_optimizer_update": _safe_div(loss_reduction, updates),
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "accuracy_delta": accuracy_delta,
            "numeric_validation_status": learning.get("numeric_validation_status", comparison.get("status")),
            "comparison_passed": comparison.get("passed"),
            "claim_scope": contract.get("claim_scope"),
            "board_fit_status": board_fit.get("status"),
            "paper_claim_eligible": eligible and not confounders,
            "confounders": ";".join(confounders),
            "out_dir": str(out_dir),
        })

    seed_values = {row["seed"] for row in rows if row.get("seed") is not None}
    seed_count = len(seed_values)
    groups: dict[tuple[Any, Any], int] = {}
    for row in rows:
        key = (row.get("comparison_role"), row.get("batch_size"))
        groups[key] = groups.get(key, 0) + 1
    replicate_count = max(groups.values(), default=0)
    variance_available = seed_count >= 2 or replicate_count >= 2
    statistical_claim_eligible = bool(rows) and variance_available and all(
        row["numeric_validation_status"] == "passed" for row in rows
    )

    for row in rows:
        row["seed_count"] = seed_count
        row["replicate_count"] = groups.get((row.get("comparison_role"), row.get("batch_size")), 0)
        row["variance_available"] = variance_available
        row["statistical_claim_eligible"] = statistical_claim_eligible

    csv_path = experiment_dir / "training_learning_ablation_summary.csv"
    fields = [
        "design_name", "comparison_basis", "comparison_role", "batch_size", "epochs_requested",
        "max_updates_requested", "seed", "dataset_sample_count", "record_visits_executed",
        "optimizer_updates", "epochs_completed", "record_visits_per_update", "samples_per_update",
        "headline_domain", "initial_loss", "final_loss", "loss_delta", "loss_reduction",
        "loss_reduction_per_record_visit", "loss_reduction_per_optimizer_update",
        "initial_accuracy", "final_accuracy", "accuracy_delta", "numeric_validation_status",
        "comparison_passed", "claim_scope", "board_fit_status", "paper_claim_eligible",
        "seed_count", "replicate_count", "variance_available", "statistical_claim_eligible",
        "confounders", "out_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = experiment_dir / "training_learning_ablation_summary.md"
    lines = [
        "# FPGAI Training Learning Ablation Summary",
        "",
        f"Comparison basis: `{comparison_basis}`",
        "",
        "| Design | Role | Batch | Record visits | Updates | Visits/update | Loss reduction | Loss/visit | Loss/update | Final accuracy | Validation | Eligible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{design_name}` | {comparison_role} | {batch_size} | {record_visits_executed} | {optimizer_updates} | {record_visits_per_update} | {loss_reduction} | {loss_reduction_per_record_visit} | {loss_reduction_per_optimizer_update} | {final_accuracy} | {numeric_validation_status} | {paper_claim_eligible} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    lines.extend([
        "",
        "## Statistical readiness",
        "",
        f"- Unique seeds: {seed_count}",
        f"- Maximum replicates per comparison cell: {replicate_count}",
        f"- Variance available: {variance_available}",
        f"- Statistical claim eligible: {statistical_claim_eligible}",
        "",
        "## Interpretation contract",
        "",
        "- `equal_sample_exposure` holds record visits constant while optimizer updates may differ.",
        "- `equal_update_budget` holds optimizer updates constant while record exposure may differ.",
        "- `duration_control` rows are retained but excluded from the primary comparison.",
        "- Normalized metrics describe this execution only; they are not substitutes for replicated statistics.",
        "- Small-sample smoke workloads are never promoted to convergence/generalization claims.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    capability = {
        "mechanism_validated": bool(rows) and all(row["numeric_validation_status"] == "passed" for row in rows),
        "learning_observed": any(isinstance(row.get("loss_reduction"), (int, float)) and row["loss_reduction"] > 0 for row in rows),
        "comparative_trend_observed": len([row for row in rows if row["comparison_role"] == "primary_comparison"]) >= 2,
        "statistical_comparison_supported": statistical_claim_eligible,
        "convergence_supported": any(row.get("claim_scope") == "convergence_and_generalization" for row in rows),
        "generalization_supported": any(row.get("claim_scope") == "convergence_and_generalization" for row in rows),
    }

    eligibility_path = experiment_dir / "training_claim_eligibility.json"
    eligibility_path.write_text(json.dumps({
        "schema_version": 2,
        "comparison_basis": comparison_basis,
        "row_count": len(rows),
        "eligible_count": sum(1 for row in rows if row["paper_claim_eligible"]),
        "seed_count": seed_count,
        "replicate_count": replicate_count,
        "variance_available": variance_available,
        "statistical_claim_eligible": statistical_claim_eligible,
        "capabilities": capability,
        "rows": [
            {
                "design_name": row["design_name"],
                "comparison_role": row["comparison_role"],
                "paper_claim_eligible": row["paper_claim_eligible"],
                "claim_scope": row["claim_scope"],
                "confounders": row["confounders"].split(";") if row["confounders"] else [],
            }
            for row in rows
        ],
    }, indent=2, sort_keys=True), encoding="utf-8")

    return {"csv": csv_path, "markdown": md_path, "eligibility": eligibility_path}


def write_paired_training_batch_ablation_reports(
    output_dir: str | Path,
    experiment_dirs: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Combine already-generated ablation summaries without merging claims."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    available: dict[str, str] = {}
    missing: dict[str, str] = {}
    for basis, raw_dir in experiment_dirs.items():
        experiment_dir = Path(raw_dir)
        summary = experiment_dir / "training_learning_ablation_summary.csv"
        if not summary.exists():
            missing[str(basis)] = str(summary)
            continue
        available[str(basis)] = str(summary)
        with summary.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                copied = dict(row)
                copied.setdefault("comparison_basis", str(basis))
                copied.setdefault("comparison_role", "primary_comparison")
                copied.setdefault("record_visits_per_update", "")
                copied.setdefault("loss_reduction", "")
                copied.setdefault("loss_reduction_per_record_visit", "")
                copied.setdefault("loss_reduction_per_optimizer_update", "")
                copied.setdefault("seed_count", "1")
                copied.setdefault("replicate_count", "1")
                copied.setdefault("variance_available", "False")
                copied.setdefault("statistical_claim_eligible", "False")
                copied.setdefault("confounders", "")
                copied["source_experiment_dir"] = str(experiment_dir)
                rows.append(copied)

    fields = [
        "comparison_basis", "comparison_role", "design_name", "batch_size",
        "record_visits_executed", "optimizer_updates", "record_visits_per_update",
        "initial_loss", "final_loss", "loss_reduction",
        "loss_reduction_per_record_visit", "loss_reduction_per_optimizer_update",
        "initial_accuracy", "final_accuracy", "accuracy_delta", "seed_count",
        "replicate_count", "variance_available", "statistical_claim_eligible",
        "claim_scope", "paper_claim_eligible", "confounders", "source_experiment_dir",
    ]
    csv_path = output_dir / "training_batch_ablation_paired_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "training_batch_ablation_paired_summary.md"
    lines = [
        "# FPGAI Paired Batch-Ablation Summary",
        "",
        "The comparison bases remain separate; this table does not treat unequal budgets as equivalent.",
        "",
        "| Basis | Role | Batch | Visits | Updates | Loss reduction | Loss/visit | Loss/update | Final accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {comparison_basis} | {comparison_role} | {batch_size} | {record_visits_executed} | {optimizer_updates} | {loss_reduction} | {loss_reduction_per_record_visit} | {loss_reduction_per_optimizer_update} | {final_accuracy} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    interpretation_path = output_dir / "training_batch_ablation_interpretation.json"
    interpretation_path.write_text(json.dumps({
        "schema_version": 1,
        "available_comparison_bases": sorted(available),
        "missing_comparison_bases": missing,
        "row_count": len(rows),
        "interpretation_contract": {
            "equal_sample_exposure": "Compares update frequency while holding record exposure constant.",
            "equal_update_budget": "Compares record exposure while holding optimizer updates constant.",
            "cross_basis_claim": "Descriptive only until larger held-out datasets and replicated seeds are available.",
        },
        "statistical_comparison_supported": bool(rows) and all(
            str(row.get("statistical_claim_eligible", "")).lower() == "true" for row in rows
        ),
    }, indent=2, sort_keys=True), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "interpretation": interpretation_path}

