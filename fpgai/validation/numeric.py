"""Public numeric validation report entry point.

The implementation is split by responsibility while this module preserves the
existing import path for contributors and downstream users.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .numeric_common import (
    _compare_file_pair,
    _exists,
    _path_or_none,
    _precision_aware_inference_limits,
)
from .numeric_quality import _task_quality_payload
from .numeric_optimizer import (
    _optimizer_resource_strategy_payload,
    _optimizer_state_validation_payload,
    _parameter_update_validation_payload,
    _training_update_behavior_payload,
)
from .numeric_training import (
    _batch_accumulation_validation_payload,
    _gradient_export_validation_payload,
    _loss_validation_payload,
    _training_compare_payload,
    _training_reference_payload,
    _training_tiled_io_validation_payload,
)

def emit_numeric_validation_report(
    out_dir: str | Path,
    *,
    pipeline_mode: str,
    source_generated: bool,
    hls_ran: bool = False,
    hls_ok: bool | None = None,
    hls_csynth_report: str | Path | None = None,
    training_reference_result: Any = None,
    training_compare_result: Any = None,
    inference_reference_artifacts: dict[str, Any] | None = None,
    gradient_export_artifacts: dict[str, Any] | None = None,
    optimizer_state_artifacts: dict[str, Any] | None = None,
    parameter_update_artifacts: dict[str, Any] | None = None,
    raw_config: dict[str, Any] | None = None,
    runtime_sequence: Any = None,
) -> dict[str, Path]:
    """Write numeric validation reports and return artifact paths.

    The report is intentionally conservative:
    - inference is marked ``not_run`` unless explicit inference validation
      artifacts are provided;
    - training is marked ``passed`` only when the training comparison result
      exists and exposes a summary/results file;
    - missing HLS/testbench artifacts are recorded as missing artifacts, not as
      success.
    """

    out = Path(out_dir)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    pipeline_mode = str(pipeline_mode or "inference")
    training_reference = _training_reference_payload(training_reference_result)
    training_compare = _training_compare_payload(training_compare_result)

    gradient_export_validation = _gradient_export_validation_payload(gradient_export_artifacts)
    optimizer_state_validation = _optimizer_state_validation_payload(optimizer_state_artifacts, raw_config=raw_config)
    parameter_update_validation = _parameter_update_validation_payload(parameter_update_artifacts, raw_config=raw_config)
    update_behavior_trace = _training_update_behavior_payload(parameter_update_validation, optimizer_state_validation, raw_config=raw_config)
    optimizer_resource_strategy = _optimizer_resource_strategy_payload(raw_config, hls_ran=hls_ran, hls_ok=hls_ok, hls_csynth_report=hls_csynth_report)
    batch_accumulation_validation = _batch_accumulation_validation_payload(
        out,
        pipeline_mode=pipeline_mode,
        raw_config=raw_config,
        runtime_sequence=runtime_sequence,
        training_reference_result=training_reference_result,
        training_compare_result=training_compare_result,
    )
    loss_validation = _loss_validation_payload(
        out,
        pipeline_mode=pipeline_mode,
        raw_config=raw_config,
        training_reference_result=training_reference_result,
        training_compare_result=training_compare_result,
    )
    training_tiled_io_validation = _training_tiled_io_validation_payload(
        out,
        pipeline_mode=pipeline_mode,
        raw_config=raw_config,
        training_reference_result=training_reference_result,
        training_compare_result=training_compare_result,
    )

    task_quality: dict[str, Any] = {"status": "not_applicable", "task": "not_applicable", "decision_status": "not_applicable"}
    if pipeline_mode == "training_on_device":
        if training_compare is not None:
            status = "passed"
            reason = "training reference and generated/testbench comparison artifacts are available"
        elif training_reference is not None:
            status = "reference_only"
            reason = "Python training reference exists, but generated/testbench comparison artifacts are missing"
        else:
            status = "not_run"
            reason = "no training numeric reference or generated/testbench comparison artifacts were found"
    else:
        inference_reference_artifacts = inference_reference_artifacts or {}
        output_compare = None
        if inference_reference_artifacts.get("outputs_hw") and inference_reference_artifacts.get("outputs_ref"):
            limits = _precision_aware_inference_limits(raw_config)
            output_compare = _compare_file_pair(
                inference_reference_artifacts.get("outputs_ref"),
                inference_reference_artifacts.get("outputs_hw"),
                max_abs_error_limit=float(limits["max_abs_error_limit"]),
                mean_abs_error_limit=float(limits["mean_abs_error_limit"]),
                rmse_limit=float(limits["rmse_limit"]),
                min_cosine_similarity=float(limits["min_cosine_similarity"]),
            )
        task_quality = _task_quality_payload(
            out,
            raw_config=raw_config,
            output_compare=output_compare,
            inference_reference_artifacts=inference_reference_artifacts,
        )
        if output_compare is None:
            status = "not_run"
            reason = "inference numeric comparison artifacts are not available for this compile path"
        elif output_compare.get("passed") is True:
            status = "passed"
            reason = "inference output comparison artifacts were compared successfully"
        else:
            compare_status = str(output_compare.get("status") or "failed_numeric_validation")
            if compare_status == "compared":
                status = "failed_tolerance"
                failed_checks = [
                    str(check.get("name"))
                    for check in output_compare.get("checks", [])
                    if isinstance(check, dict) and check.get("passed") is False
                ]
                if failed_checks:
                    reason = "inference output comparison completed but failed tolerance check(s): " + ", ".join(failed_checks)
                else:
                    reason = "inference output comparison completed but failed configured precision-aware tolerance checks"
            elif compare_status in {"shape_mismatch", "missing_or_unreadable", "empty_generated_output"}:
                status = "execution_artifact_invalid" if compare_status == "empty_generated_output" else compare_status
                reason = (
                    "generated inference output artifact is empty"
                    if compare_status == "empty_generated_output"
                    else f"inference output comparison could not be accepted: {compare_status}"
                )
            else:
                status = "failed_numeric_validation"
                reason = f"inference output comparison failed with status={compare_status}"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "numeric_validation",
        "pipeline_mode": pipeline_mode,
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "source_generated": bool(source_generated),
        "hls_ran": bool(hls_ran),
        "hls_ok": hls_ok,
        "inference": {
            "status": status if pipeline_mode != "training_on_device" else "not_applicable",
            "inputs_bin": _path_or_none(inference_reference_artifacts.get("inputs_bin")) if inference_reference_artifacts else None,
            "outputs_hw": _path_or_none(inference_reference_artifacts.get("outputs_hw")) if inference_reference_artifacts else None,
            "outputs_ref": _path_or_none(inference_reference_artifacts.get("outputs_ref")) if inference_reference_artifacts else None,
            "outputs_hw_exists": _exists(inference_reference_artifacts.get("outputs_hw")) if inference_reference_artifacts else False,
            "outputs_ref_exists": _exists(inference_reference_artifacts.get("outputs_ref")) if inference_reference_artifacts else False,
            "output_compare": output_compare if pipeline_mode != "training_on_device" else None,
            "task_quality": task_quality if pipeline_mode != "training_on_device" else {"status": "not_applicable", "task": "not_applicable", "decision_status": "not_applicable"},
        },
        "training": {
            "status": status if pipeline_mode == "training_on_device" else "not_applicable",
            "reference": training_reference,
            "comparison": training_compare,
            "checks": [] if training_compare is None else [
                {"name": "gradients", "metric": "cosine_similarity", "value": training_compare.get("grad_cosine"), "passed": training_compare.get("grad_cosine") is None or training_compare.get("grad_cosine") >= 0.99},
                {"name": "weights_after", "metric": "cosine_similarity", "value": training_compare.get("weight_after_cosine"), "passed": training_compare.get("weight_after_cosine") is None or training_compare.get("weight_after_cosine") >= 0.99},
                {"name": "weight_delta", "metric": "cosine_similarity", "value": training_compare.get("weight_delta_cosine"), "passed": training_compare.get("weight_delta_cosine") is None or training_compare.get("weight_delta_cosine") >= 0.99},
            ],
        },
        "gradient_export": gradient_export_validation,
        "optimizer_state_validation": optimizer_state_validation,
        "parameter_update_validation": parameter_update_validation,
        "training_update_behavior_trace": update_behavior_trace,
        "optimizer_resource_strategy": optimizer_resource_strategy,
        "batch_accumulation": batch_accumulation_validation,
        "loss_validation": loss_validation,
        "training_tiled_io": training_tiled_io_validation,
        "validation_claim_allowed": {
            "numeric_correctness": status == "passed",
        },
    }

    json_path = reports / "numeric_validation.json"
    md_path = reports / "numeric_validation.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Numeric validation",
        "",
        f"- Pipeline mode: `{pipeline_mode}`",
        f"- Status: `{status}`",
        f"- Passed: `{str(status == 'passed').lower()}`",
        f"- Reason: {reason}",
        f"- Source generated: `{str(bool(source_generated)).lower()}`",
        f"- HLS ran: `{str(bool(hls_ran)).lower()}`",
    ]
    if pipeline_mode == "training_on_device":
        lines += [
            "",
            "## Training artifacts",
            f"- Python reference: `{ 'yes' if training_reference is not None else 'no' }`",
            f"- Generated/testbench comparison: `{ 'yes' if training_compare is not None else 'no' }`",
            f"- Gradient export validation: `{gradient_export_validation.get('status', 'not_requested')}`",
            f"- Optimizer-state validation: `{optimizer_state_validation.get('status', 'not_requested')}`",
            f"- Parameter-update validation: `{parameter_update_validation.get('status', 'not_requested')}`",
            f"- Update behavior trace: `{update_behavior_trace.get('status', 'partial')}`",
            f"- Batch accumulation validation: `{batch_accumulation_validation.get('status', 'not_requested')}`",
            f"- Loss validation: `{loss_validation.get('status', 'not_requested')}`",
            f"- Training tiled-I/O validation: `{training_tiled_io_validation.get('status', 'not_requested')}`",
        ]
    else:
        lines += [
            "",
            "## Inference artifacts",
            "- Final output comparison: `not available`" if output_compare is None else f"- Final output comparison: `{output_compare.get('status', 'available')}`",
            f"- Task quality: `{task_quality.get('decision_status', 'not_applicable')}`",
            f"- Task quality reason: {task_quality.get('decision_reason', 'not applicable')}",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    optimizer_json_path = reports / "optimizer_state_validation.json"
    optimizer_md_path = reports / "optimizer_state_validation.md"
    optimizer_json_path.write_text(json.dumps(optimizer_state_validation, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    optimizer_lines = [
        "# Optimizer-state validation",
        "",
        f"- Optimizer: `{optimizer_state_validation.get('optimizer', 'not_applicable')}`",
        f"- Status: `{optimizer_state_validation.get('status', 'not_validated')}`",
        f"- Implementation status: `{optimizer_state_validation.get('implementation_status', optimizer_state_validation.get('status', 'not_validated'))}`",
        f"- Passed: `{str(bool(optimizer_state_validation.get('passed', False))).lower()}`",
        f"- Layout: `{optimizer_state_validation.get('layout', 'canonical_parameter_order')}`",
    ]
    for name, comparison in (optimizer_state_validation.get("comparisons", {}) or {}).items():
        if not isinstance(comparison, dict):
            continue
        optimizer_lines += [
            "",
            f"## {name}",
            f"- Reference words: `{comparison.get('reference_words')}`",
            f"- HLS words: `{comparison.get('hls_words')}`",
            f"- MAE: `{comparison.get('mae')}`",
            f"- Maximum absolute error: `{comparison.get('max_abs_error')}`",
            f"- Relative L2: `{comparison.get('relative_l2')}`",
            f"- Cosine similarity: `{comparison.get('cosine_similarity')}`",
            f"- Exact words: `{comparison.get('exact_words')}`",
            f"- Within one LSB: `{comparison.get('within_one_lsb')}`",
            f"- Within two LSBs: `{comparison.get('within_two_lsb')}`",
            f"- Above two LSBs: `{comparison.get('above_two_lsb')}`",
            f"- Maximum LSB distance: `{comparison.get('maximum_lsb_distance')}`",
            f"- Classification: `{comparison.get('classification')}`",
            f"- All words within one LSB: `{str(bool(comparison.get('all_words_within_one_lsb', False))).lower()}`",
            f"- All words within two LSBs: `{str(bool(comparison.get('all_words_within_two_lsb', False))).lower()}`",
        ]
    optimizer_md_path.write_text("\n".join(optimizer_lines) + "\n", encoding="utf-8")

    parameter_json_path = reports / "parameter_update_validation.json"
    parameter_md_path = reports / "parameter_update_validation.md"
    parameter_json_path.write_text(json.dumps(parameter_update_validation, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    parameter_lines = ["# Parameter-update validation", "", f"- Status: `{parameter_update_validation.get('status', 'not_validated')}`", f"- Classification: `{parameter_update_validation.get('classification')}`", f"- Passed: `{str(bool(parameter_update_validation.get('passed', False))).lower()}`", f"- Maximum LSB distance: `{parameter_update_validation.get('maximum_lsb_distance')}`"]
    for segment in parameter_update_validation.get("segments", []) or []:
        parameter_lines += ["", f"## {segment.get('name')}", f"- Exact words: `{segment.get('exact_words')}/{segment.get('count')}`", f"- Within one LSB: `{segment.get('within_one_lsb')}`", f"- Within two LSBs: `{segment.get('within_two_lsb')}`", f"- Above two LSBs: `{segment.get('above_two_lsb')}`", f"- Maximum LSB distance: `{segment.get('maximum_lsb_distance')}`"]
    parameter_md_path.write_text("\n".join(parameter_lines) + "\n", encoding="utf-8")
    behavior_json_path = reports / "training_update_behavior_trace.json"
    behavior_md_path = reports / "training_update_behavior_trace.md"
    behavior_json_path.write_text(json.dumps(update_behavior_trace, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    behavior_md_path.write_text("# Training update behavior trace\n\n" + f"- Status: `{update_behavior_trace.get('status')}`\n- Optimizer updates: `{update_behavior_trace.get('optimizer_updates')}`\n- Final classification: `{update_behavior_trace.get('final_classification')}`\n- First divergent update: `{update_behavior_trace.get('first_divergent_update')}`\n- First divergent layer: `{update_behavior_trace.get('first_divergent_layer')}`\n- First divergent tensor: `{update_behavior_trace.get('first_divergent_tensor')}`\n- Propagation path: `{' -> '.join(update_behavior_trace.get('propagation_path', [])) or 'none'}`\n", encoding="utf-8")

    behavior_csv_path = reports / "training_update_behavior_trace.csv"
    behavior_csv_lines = ["update,tensor_kind,name,classification,passed,maximum_lsb_distance"]
    for row in update_behavior_trace.get("per_update", []) or []:
        for tensor_kind in ("weights", "optimizer_state"):
            comparison = row.get(tensor_kind, {}) or {}
            segments = comparison.get("segments", []) or []
            if segments:
                for segment in segments:
                    behavior_csv_lines.append(f"{row.get('update')},{tensor_kind},{segment.get('name')},{comparison.get('classification')},{str(bool(comparison.get('passed'))).lower()},{segment.get('maximum_lsb_distance')}")
            else:
                behavior_csv_lines.append(f"{row.get('update')},{tensor_kind},global,{comparison.get('classification')},{str(bool(comparison.get('passed'))).lower()},{comparison.get('maximum_lsb_distance')}")
    behavior_csv_path.write_text("\n".join(behavior_csv_lines) + "\n", encoding="utf-8")
    resource_json_path = reports / "optimizer_resource_strategy.json"
    resource_md_path = reports / "optimizer_resource_strategy.md"
    ownership_json_path = reports / "training_resource_ownership.json"
    ownership_md_path = reports / "training_resource_ownership.md"
    command_latency_json_path = reports / "training_command_latency.json"
    command_latency_md_path = reports / "training_command_latency.md"
    resource_json_path.write_text(json.dumps(optimizer_resource_strategy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    resource_md_path.write_text("# Optimizer resource strategy\n\n" + f"- Arithmetic: `{optimizer_resource_strategy.get('arithmetic')}`\n- Update parallelism: `{optimizer_resource_strategy.get('update_parallelism')}`\n- State storage: `{optimizer_resource_strategy.get('optimizer_state_storage')}`\n- Mechanism: `{optimizer_resource_strategy.get('mechanism')}`\n- HLS synthesis status: `{optimizer_resource_strategy.get('hls_synthesis_status')}`\n- C synthesis report present: `{str(bool(optimizer_resource_strategy.get('csynth_report_present'))).lower()}`\n- C synthesis report: `{optimizer_resource_strategy.get('csynth_report')}`\n- HLS metrics: `{optimizer_resource_strategy.get('hls_metrics')}`\n- Baseline comparison: `{optimizer_resource_strategy.get('baseline_comparison_status')}`\n", encoding="utf-8")
    ownership = optimizer_resource_strategy.get("training_resource_ownership", {})
    ownership_json_path.write_text(json.dumps(ownership, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    owner_lines = ["# Training resource ownership", "", f"- Status: `{ownership.get('status')}`", f"- Owner count: `{ownership.get('owner_count')}`", f"- Claim scope: `{ownership.get('claim_scope')}`", "", "## Largest generated owners", ""]
    for row in ownership.get("top_owners", [])[:12]:
        owner_lines.append(f"- `{row.get('name')}`: role=`{row.get('role')}`, bits=`{row.get('total_bits')}`, binding=`{row.get('source_binding')}`, knob=`{row.get('owning_yaml_knob')}`")
    owner_lines.extend(["", "## Knob actions", ""])
    for row in ownership.get("recommended_knob_actions", []):
        owner_lines.append(f"- Priority {row.get('priority')}: `{row.get('knob')}` — {row.get('action')} ({row.get('reason')})")
    ownership_md_path.write_text("\n".join(owner_lines) + "\n", encoding="utf-8")
    command_latency = optimizer_resource_strategy.get("training_command_latency", {})
    command_latency_json_path.write_text(json.dumps(command_latency, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    latency_lines = [
        "# Training command latency",
        "",
        f"- Status: `{command_latency.get('status')}`",
        f"- Claim scope: `{command_latency.get('claim_scope')}`",
        f"- Aggregate HLS top range: `{command_latency.get('aggregate_hls_top')}`",
        "",
        "## Commands",
        "",
    ]
    for command, row in command_latency.get("commands", {}).items():
        latency_lines.append(f"- `{command}`: status=`{row.get('status')}`, details=`{row}`")
    command_latency_md_path.write_text("\n".join(latency_lines) + "\n", encoding="utf-8")

    return {
        "numeric_validation_json": json_path,
        "numeric_validation_md": md_path,
        "optimizer_state_validation_json": optimizer_json_path,
        "optimizer_state_validation_md": optimizer_md_path,
        "parameter_update_validation_json": parameter_json_path,
        "parameter_update_validation_md": parameter_md_path,
        "training_update_behavior_trace_json": behavior_json_path,
        "training_update_behavior_trace_md": behavior_md_path,
        "training_update_behavior_trace_csv": behavior_csv_path,
        "optimizer_resource_strategy_json": resource_json_path,
        "optimizer_resource_strategy_md": resource_md_path,
        "training_resource_ownership_json": ownership_json_path,
        "training_resource_ownership_md": ownership_md_path,
        "training_command_latency_json": command_latency_json_path,
        "training_command_latency_md": command_latency_md_path,
    }
