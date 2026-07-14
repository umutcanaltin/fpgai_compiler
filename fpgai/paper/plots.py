"""Paper plot data/figure generation from generated FPGAI reports.

This module intentionally consumes existing compile/runtime reports. It does
not execute HLS, Vivado, or FPGA hardware, and it never fabricates board-runtime
latency, energy, or training-curve measurements. Missing board measurements are
recorded as pending inputs in the plot manifest.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fpgai.reporting.paper_results import build_master_result_row

SCHEMA_VERSION = 1
ARTIFACT_KIND = "fpgai_paper_plot_artifacts"

PLOT_ROW_FIELDS = [
    "section",
    "design_id",
    "mode",
    "board",
    "precision",
    "memory_strategy",
    "data_movement",
    "parallel_factor",
    "pipeline_policy",
    "tiling",
    "hls_status",
    "hls_lut",
    "hls_dsp",
    "hls_bram",
    "hls_latency_max",
    "vivado_implementation_status",
    "vivado_lut",
    "vivado_dsp",
    "vivado_bram",
    "vivado_wns",
    "vivado_power_w",
    "runtime_status",
    "runtime_latency_ms_mean",
    "runtime_throughput",
    "runtime_accuracy",
    "runtime_training_step_ms",
    "runtime_energy_mj_estimated",
    "support_status",
]

KNOB_ROW_FIELDS = [
    "section",
    "design_id",
    "yaml_path",
    "source",
    "requested",
    "effective",
    "status",
    "applied_to",
]

TRAINING_CURVE_FIELDS = [
    "design_id",
    "source",
    "step",
    "epoch",
    "batch",
    "mode",
    "loss",
    "accuracy",
    "runtime_seconds",
    "cumulative_runtime_seconds",
    "gradient_norm",
    "weight_delta_norm",
    "gradient_cosine_vs_reference",
    "weight_after_cosine_vs_reference",
    "weight_delta_cosine_vs_reference",
    "status",
]

NUMERIC_ROW_FIELDS = [
    "section",
    "design_id",
    "precision",
    "numeric_validation_status",
    "numeric_passed",
    "numeric_quality",
    "numeric_reason",
    "paper_numeric_claim_allowed",
    "inference_output_status",
    "numeric_sample_count",
    "mse",
    "rmse",
    "mae",
    "max_abs_error",
    "cosine_similarity",
    "training_status",
    "gradient_cosine",
    "weight_after_cosine",
    "weight_delta_cosine",
    "grad_mae",
    "grad_max_abs",
    "weight_after_mae",
    "weight_after_max_abs",
    "initial_loss",
    "final_loss",
    "loss_delta",
    "gradient_export_status",
    "optimizer_state_status",
    "batch_accumulation_status",
    "loss_validation_status",
    "training_tiled_io_status",
    "task",
    "task_quality_status",
    "decision_status",
    "decision_reason",
    "dataset_source",
    "dataset_sample_count",
    "class_count",
    "labels_status",
    "targets_status",
    "reference_top1_first",
    "generated_top1_first",
    "prediction_agreement_vs_reference",
    "class_change_count",
    "reference_top1_accuracy",
    "generated_top1_accuracy",
    "top1_accuracy_drop_pct",
    "reference_top5_accuracy",
    "generated_top5_accuracy",
    "top5_accuracy_drop_pct",
    "confidence_delta_mean",
    "confidence_delta_max",
    "reference_output_mae",
    "reference_output_rmse",
    "target_mae_reference",
    "target_mae_generated",
    "target_rmse_reference",
    "target_rmse_generated",
    "target_r2_reference",
    "target_r2_generated",
    "mae_increase",
    "rmse_increase",
    "hls_latency_max",
    "vivado_lut",
    "vivado_dsp",
    "vivado_bram",
    "vivado_power_w",
    "vivado_wns",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _format_table_value(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    text = text.replace("\n", " ").strip()
    return text if text else "—"


def _md_escape(value: Any) -> str:
    return _format_table_value(value).replace("|", "\\|")


def _short_design_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "—"
    if text.startswith("I") and "_" in text:
        return text
    if text.startswith("T") and "_" in text:
        return text
    return text


def _precision_family(value: Any) -> str:
    """Return a compact paper-table precision label.

    The raw reports often store the full precision mapping as a dictionary.
    That is useful for JSON/CSV traceability, but it is unreadable in paper
    tables.  We keep the table label compact while preserving the full report in
    the source artifact files.
    """
    if value in (None, ""):
        return "—"
    data = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "—"
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except Exception:
                return text
        else:
            return text
    if not isinstance(data, Mapping):
        return str(data)

    defaults = data.get("defaults") if isinstance(data.get("defaults"), Mapping) else {}
    training = data.get("training") if isinstance(data.get("training"), Mapping) else {}

    def fmt(node: Any) -> str:
        if not isinstance(node, Mapping):
            return ""
        total = node.get("total_bits")
        integer = node.get("int_bits")
        typ = str(node.get("type") or "").replace("ap_", "")
        if total not in (None, "") and integer not in (None, ""):
            return f"fx{total}_{integer}"
        if total not in (None, ""):
            return f"fx{total}"
        return typ

    act = fmt(defaults.get("activation"))
    wt = fmt(defaults.get("weight"))
    grad = fmt(training.get("grad"))
    state = fmt(training.get("optimizer_state"))
    parts: list[str] = []
    if act and wt and act == wt:
        parts.append(act)
    else:
        if act:
            parts.append(f"act {act}")
        if wt:
            parts.append(f"wt {wt}")
    if grad:
        parts.append(f"grad {grad}")
    if state and state != grad:
        parts.append(f"opt {state}")
    return ", ".join(parts) if parts else json.dumps(data, sort_keys=True)


def _paper_clean_row(row: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    cleaned["design_id"] = _short_design_label(cleaned.get("design_id"))
    cleaned["precision"] = _precision_family(cleaned.get("precision"))
    return cleaned


def _paper_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    section = 0 if row.get("section") == "inference" else 1
    design = str(row.get("design_id") or "")
    order = 999
    if len(design) > 1 and design[0] in {"I", "T"}:
        digits = ""
        for ch in design[1:]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            try:
                order = int(digits)
            except Exception:
                order = 999
    return (section, order, design)


def _write_table_md(path: Path, *, title: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], labels: Mapping[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = labels or {}
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows available from existing artifacts.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    headers = [labels.get(field, field) for field in fields]
    lines.append("| " + " | ".join(_md_escape(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_md_escape(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table_pair(table_dir: Path, *, stem: str, title: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], labels: Mapping[str, str] | None = None) -> dict[str, str]:
    csv_path = table_dir / f"{stem}.csv"
    md_path = table_dir / f"{stem}.md"
    _write_csv(csv_path, rows, fields)
    _write_table_md(md_path, title=title, rows=rows, fields=fields, labels=labels)
    return {"csv": csv_path.as_posix(), "md": md_path.as_posix(), "row_count": str(len(rows))}


def _paper_table_rows(plot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in plot_rows:
        row = _paper_clean_row(raw)
        rows.append({
            "section": row.get("section"),
            "design_id": row.get("design_id"),
            "mode": row.get("mode"),
            "board": row.get("board"),
            "precision": row.get("precision"),
            "memory_strategy": row.get("memory_strategy"),
            "pipeline_policy": row.get("pipeline_policy"),
            "hls_status": row.get("hls_status"),
            "vivado_status": row.get("vivado_implementation_status"),
            "runtime_status": row.get("runtime_status"),
            "support_status": row.get("support_status"),
        })
    return sorted(rows, key=_paper_sort_key)


def _hls_table_rows(plot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in plot_rows:
        row = _paper_clean_row(raw)
        if not any(_float_or_none(row.get(key)) is not None for key in ("hls_lut", "hls_dsp", "hls_bram", "hls_latency_max")):
            continue
        rows.append({
            "section": row.get("section"),
            "design_id": row.get("design_id"),
            "hls_status": row.get("hls_status"),
            "hls_latency_max": row.get("hls_latency_max"),
            "hls_lut": row.get("hls_lut"),
            "hls_dsp": row.get("hls_dsp"),
            "hls_bram": row.get("hls_bram"),
            "hls_ii": row.get("hls_ii"),
            "hls_clock_period_ns": row.get("hls_clock_period_ns"),
        })
    return sorted(rows, key=_paper_sort_key)


def _vivado_table_rows(plot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in plot_rows:
        row = _paper_clean_row(raw)
        if not any(_float_or_none(row.get(key)) is not None for key in ("vivado_lut", "vivado_dsp", "vivado_bram", "vivado_wns", "vivado_power_w")):
            continue
        rows.append({
            "section": row.get("section"),
            "design_id": row.get("design_id"),
            "vivado_status": row.get("vivado_implementation_status"),
            "vivado_lut": row.get("vivado_lut"),
            "vivado_dsp": row.get("vivado_dsp"),
            "vivado_bram": row.get("vivado_bram"),
            "vivado_wns": row.get("vivado_wns"),
            "vivado_power_w": row.get("vivado_power_w"),
        })
    return sorted(rows, key=_paper_sort_key)


def _runtime_status_rows(plot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in plot_rows:
        row = _paper_clean_row(raw)
        rows.append({
            "section": row.get("section"),
            "design_id": row.get("design_id"),
            "runtime_status": row.get("runtime_status") or "pending_board_runtime",
            "runtime_latency_ms_mean": row.get("runtime_latency_ms_mean"),
            "runtime_throughput": row.get("runtime_throughput"),
            "runtime_accuracy": row.get("runtime_accuracy"),
            "runtime_training_step_ms": row.get("runtime_training_step_ms"),
            "runtime_energy_mj_estimated": row.get("runtime_energy_mj_estimated"),
        })
    return sorted(rows, key=_paper_sort_key)


def _pending_measurement_rows(plot_rows: Sequence[Mapping[str, Any]], curve_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in plot_rows:
        design = row.get("design_id")
        section = row.get("section")
        if section == "inference":
            required = ["runtime_latency_ms_mean", "runtime_throughput", "runtime_accuracy", "runtime_energy_mj_estimated"]
        else:
            required = ["runtime_training_step_ms", "runtime_loss_before", "runtime_loss_after"]
        for key in required:
            if row.get(key) in (None, ""):
                rows.append({"section": section, "design_id": design, "measurement": key, "status": "pending_board_runtime"})
        if section == "training" and not any(str(c.get("design_id")) == str(design) for c in curve_rows):
            rows.append({"section": section, "design_id": design, "measurement": "board_training_curve.csv", "status": "pending_board_runtime"})
    return rows


def _knob_effect_rows(knob_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], int] = {}
    for row in knob_rows:
        raw_effective = row.get("effective") if row.get("effective") not in (None, "") else row.get("requested")
        key = (
            str(row.get("yaml_path") or "unknown"),
            _format_table_value(raw_effective),
            str(row.get("status") or "unknown"),
        )
        groups[key] = groups.get(key, 0) + 1
    return [
        {"knob": knob, "effective": effective, "status": status, "design_count": count}
        for (knob, effective, status), count in sorted(groups.items())
    ]


def _write_paper_tables(table_dir: Path, *, plot_rows: Sequence[Mapping[str, Any]], knob_rows: Sequence[Mapping[str, Any]], curve_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "design_id": "Design",
        "hls_latency_max": "HLS latency cycles",
        "hls_lut": "HLS LUT",
        "hls_dsp": "HLS DSP",
        "hls_bram": "HLS BRAM",
        "vivado_lut": "Vivado LUT",
        "vivado_dsp": "Vivado DSP",
        "vivado_bram": "Vivado BRAM",
        "vivado_wns": "WNS ns",
        "vivado_power_w": "Power W",
    }
    tables: dict[str, Any] = {}
    tables["table_01_experiment_overview"] = _write_table_pair(
        table_dir, stem="table_01_experiment_overview", title="Experiment overview", rows=_paper_table_rows(plot_rows),
        fields=["section", "design_id", "mode", "board", "precision", "memory_strategy", "pipeline_policy", "hls_status", "vivado_status", "runtime_status", "support_status"], labels=labels,
    )
    tables["table_02_design_knobs"] = _write_table_pair(
        table_dir, stem="table_02_design_knobs", title="Design knob settings", rows=list(knob_rows),
        fields=KNOB_ROW_FIELDS, labels={"yaml_path": "YAML knob", "applied_to": "Applied to"},
    )
    tables["table_03_hls_results"] = _write_table_pair(
        table_dir, stem="table_03_hls_results", title="HLS results", rows=_hls_table_rows(plot_rows),
        fields=["section", "design_id", "hls_status", "hls_latency_max", "hls_lut", "hls_dsp", "hls_bram", "hls_ii", "hls_clock_period_ns"], labels=labels,
    )
    tables["table_04_vivado_results"] = _write_table_pair(
        table_dir, stem="table_04_vivado_results", title="Vivado implementation results", rows=_vivado_table_rows(plot_rows),
        fields=["section", "design_id", "vivado_status", "vivado_lut", "vivado_dsp", "vivado_bram", "vivado_wns", "vivado_power_w"], labels=labels,
    )
    tables["table_05_runtime_status"] = _write_table_pair(
        table_dir, stem="table_05_runtime_status", title="Runtime measurement status", rows=_runtime_status_rows(plot_rows),
        fields=["section", "design_id", "runtime_status", "runtime_latency_ms_mean", "runtime_throughput", "runtime_accuracy", "runtime_training_step_ms", "runtime_energy_mj_estimated"], labels=labels,
    )
    tables["table_06_pending_measurements"] = _write_table_pair(
        table_dir, stem="table_06_pending_measurements", title="Pending real board measurements", rows=_pending_measurement_rows(plot_rows, curve_rows),
        fields=["section", "design_id", "measurement", "status"], labels=labels,
    )
    tables["table_07_knob_effect_summary"] = _write_table_pair(
        table_dir, stem="table_07_knob_effect_summary", title="Knob effect coverage summary", rows=_knob_effect_rows(knob_rows),
        fields=["knob", "effective", "status", "design_count"], labels={"design_count": "Design count"},
    )
    return tables


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _section_for_mode(mode: Any) -> str:
    text = str(mode or "").lower()
    return "training" if "train" in text else "inference"


def _has_status(value: Any) -> bool:
    return str(value or "").strip().lower() not in {"", "not_run", "missing", "not_requested", "required_validation"}





def _first_nonempty_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return ""

def _precision_from_contract(out_dir: Path) -> Any:
    payload = _read_json(out_dir / "reports" / "hardware_knob_contract.json")
    knobs = payload.get("knobs")
    if not isinstance(knobs, list):
        return ""
    for item in knobs:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or item.get("name") or item.get("config_path") or "").lower()
        if "precision" not in path:
            continue
        for key in ("effective", "resolved_value", "effective_value", "value", "requested", "requested_value"):
            value = item.get(key)
            if value not in (None, "", {}, []):
                return value
    return ""

def _runtime_energy_mj(row: Mapping[str, Any]) -> float | None:
    latency_ms = _float_or_none(row.get("runtime_latency_ms_mean"))
    training_ms = _float_or_none(row.get("runtime_training_step_ms"))
    power_w = _float_or_none(row.get("vivado_power_w"))
    if power_w is None:
        return None
    ms = latency_ms if latency_ms is not None else training_ms
    if ms is None:
        return None
    # W * ms = mJ.
    return power_w * ms


def _plot_row(out_dir: Path) -> dict[str, Any]:
    row = build_master_result_row(out_dir)
    section = _section_for_mode(row.get("mode"))
    energy = _runtime_energy_mj(row)
    return {
        "section": section,
        "design_id": row.get("design_id", out_dir.name),
        "mode": row.get("mode", ""),
        "board": row.get("board", ""),
        "precision": _first_nonempty_value(row.get("precision", ""), _precision_from_contract(out_dir)),
        "memory_strategy": row.get("memory_strategy", ""),
        "data_movement": row.get("data_movement", ""),
        "parallel_factor": row.get("parallel_factor", ""),
        "pipeline_policy": row.get("pipeline_policy", ""),
        "tiling": row.get("tiling", ""),
        "hls_status": row.get("hls_status", ""),
        "hls_lut": row.get("hls_lut", ""),
        "hls_dsp": row.get("hls_dsp", ""),
        "hls_bram": row.get("hls_bram", ""),
        "hls_latency_max": row.get("hls_latency_max", ""),
        "vivado_implementation_status": row.get("vivado_implementation_status", ""),
        "vivado_lut": row.get("vivado_lut", ""),
        "vivado_dsp": row.get("vivado_dsp", ""),
        "vivado_bram": row.get("vivado_bram", ""),
        "vivado_wns": row.get("vivado_wns", ""),
        "vivado_power_w": row.get("vivado_power_w", ""),
        "runtime_status": row.get("runtime_status", ""),
        "runtime_latency_ms_mean": row.get("runtime_latency_ms_mean", ""),
        "runtime_throughput": row.get("runtime_throughput", ""),
        "runtime_accuracy": row.get("runtime_accuracy", ""),
        "runtime_training_step_ms": row.get("runtime_training_step_ms", ""),
        "runtime_energy_mj_estimated": energy if energy is not None else "",
        "support_status": row.get("support_status", ""),
    }


def _knob_rows(out_dir: Path, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _read_json(out_dir / "reports" / "hardware_knob_contract.json")
    knobs = payload.get("knobs")
    if not isinstance(knobs, list):
        return []
    section = str(row.get("section") or _section_for_mode(row.get("mode")))
    design = str(row.get("design_id") or out_dir.name)
    rows: list[dict[str, Any]] = []
    for item in knobs:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "section": section,
                "design_id": design,
                "yaml_path": str(item.get("path") or item.get("name") or item.get("config_path") or ""),
                "source": str(item.get("source") or ""),
                "requested": item.get("requested"),
                "effective": item.get("effective", item.get("resolved_value")),
                "status": str(item.get("status") or ""),
                "applied_to": "; ".join(str(x) for x in item.get("applied_to", []) if x is not None)
                if isinstance(item.get("applied_to"), list)
                else str(item.get("applied_to") or ""),
            }
        )
    return rows



def _norm_numeric_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _find_numeric_value(data: Any, *candidate_keys: str) -> Any:
    wanted = {_norm_numeric_key(key) for key in candidate_keys}
    seen: set[int] = set()

    def walk(obj: Any) -> Any:
        oid = id(obj)
        if oid in seen:
            return ""
        seen.add(oid)
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                if _norm_numeric_key(key) in wanted:
                    number = _float_or_none(value)
                    if number is not None:
                        return number
                    if value not in (None, "", [], {}):
                        return value
            for value in obj.values():
                found = walk(value)
                if found not in (None, "", [], {}):
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = walk(value)
                if found not in (None, "", [], {}):
                    return found
        return ""

    return walk(data)


def _nested_mapping(data: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, Mapping):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, Mapping) else {}


def _numeric_quality(status: Any, passed: Any, section: str) -> str:
    text = str(status or "").strip().lower()
    if text == "passed" or passed is True:
        return "passed"
    if text in {"compared", "ok"}:
        return "passed"
    if text in {"reference_only"}:
        return "reference_only"
    if text in {"not_applicable"}:
        return "not_applicable"
    if text in {"failed", "failed_tolerance", "shape_mismatch", "missing_or_unreadable", "failed_numeric_validation"}:
        return "failed_numeric_validation"
    if section == "training" and text in {"not_run", "", "missing"}:
        return "pending_numeric_validation"
    if section == "inference" and text in {"not_run", "", "missing"}:
        return "pending_numeric_validation"
    return text or "pending_numeric_validation"


def _rmse_from_mse(value: Any) -> Any:
    mse = _float_or_none(value)
    if mse is None or mse < 0:
        return ""
    return math.sqrt(mse)


def _numeric_validation_row(out_dir: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(out_dir / "reports" / "numeric_validation.json")
    section = str(row.get("section") or _section_for_mode(row.get("mode")))
    status = str(payload.get("status") or ("missing" if not payload else "present"))
    passed = payload.get("passed")
    inference = _nested_mapping(payload, "inference")
    output_compare = inference.get("output_compare") if isinstance(inference.get("output_compare"), Mapping) else {}
    if section == "inference" and output_compare:
        compare_status = str(output_compare.get("status") or "")
        if output_compare.get("passed") is True:
            status = "passed"
            passed = True
        elif compare_status == "compared":
            status = "failed_tolerance"
            passed = False
        elif compare_status in {"shape_mismatch", "missing_or_unreadable"}:
            status = compare_status
            passed = False
    task_quality = inference.get("task_quality") if isinstance(inference.get("task_quality"), Mapping) else {}
    training = _nested_mapping(payload, "training")
    training_compare = training.get("comparison") if isinstance(training.get("comparison"), Mapping) else {}
    loss_validation = _nested_mapping(payload, "loss_validation")

    mse = output_compare.get("mse") if section == "inference" else ""
    paper_claim = _nested_mapping(payload, "paper_claim_allowed").get("numeric_correctness")
    gradient_export = _nested_mapping(payload, "gradient_export")
    optimizer_state = _nested_mapping(payload, "optimizer_state_validation")
    batch_accumulation = _nested_mapping(payload, "batch_accumulation")
    tiled_io = _nested_mapping(payload, "training_tiled_io")

    initial_loss = _find_numeric_value(loss_validation, "initial_loss", "loss_before", "before_loss", "reference_initial_loss")
    final_loss = _find_numeric_value(loss_validation, "final_loss", "loss_after", "after_loss", "reference_final_loss")
    if initial_loss in (None, ""):
        initial_loss = _find_numeric_value(training, "initial_loss", "loss_before")
    if final_loss in (None, ""):
        final_loss = _find_numeric_value(training, "final_loss", "loss_after")
    loss_delta = ""
    init_f = _float_or_none(initial_loss)
    final_f = _float_or_none(final_loss)
    if init_f is not None and final_f is not None:
        loss_delta = final_f - init_f

    return {
        "section": section,
        "design_id": row.get("design_id"),
        "precision": _precision_family(row.get("precision")),
        "numeric_validation_status": status,
        "numeric_passed": passed if passed not in (None, "") else "",
        "numeric_quality": _numeric_quality(status, passed, section),
        "numeric_reason": payload.get("reason") or "numeric_validation.json missing",
        "paper_numeric_claim_allowed": paper_claim if paper_claim not in (None, "") else False,
        "inference_output_status": output_compare.get("status", "not_applicable" if section != "inference" else "missing"),
        "numeric_sample_count": output_compare.get("num_compared", ""),
        "mse": mse,
        "rmse": _rmse_from_mse(mse),
        "mae": output_compare.get("mae", "") if section == "inference" else "",
        "max_abs_error": output_compare.get("max_abs_error", "") if section == "inference" else "",
        "cosine_similarity": output_compare.get("cosine_similarity", "") if section == "inference" else "",
        "training_status": training.get("status", "not_applicable" if section != "training" else status),
        "gradient_cosine": training_compare.get("grad_cosine", ""),
        "weight_after_cosine": training_compare.get("weight_after_cosine", ""),
        "weight_delta_cosine": training_compare.get("weight_delta_cosine", ""),
        "grad_mae": training_compare.get("grad_mae", ""),
        "grad_max_abs": training_compare.get("grad_max_abs", ""),
        "weight_after_mae": training_compare.get("weight_after_mae", ""),
        "weight_after_max_abs": training_compare.get("weight_after_max_abs", ""),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_delta": loss_delta,
        "gradient_export_status": gradient_export.get("status", "not_applicable"),
        "optimizer_state_status": optimizer_state.get("status", "not_applicable"),
        "batch_accumulation_status": batch_accumulation.get("status", "not_applicable"),
        "loss_validation_status": loss_validation.get("status", "not_applicable"),
        "training_tiled_io_status": tiled_io.get("status", "not_applicable"),
        "task": task_quality.get("task", "not_applicable" if section != "inference" else "auto"),
        "task_quality_status": task_quality.get("status", "not_applicable" if section != "inference" else "missing"),
        "decision_status": task_quality.get("decision_status", "not_applicable" if section != "inference" else "pending_numeric_artifacts"),
        "decision_reason": task_quality.get("decision_reason", ""),
        "dataset_source": task_quality.get("dataset_source", ""),
        "dataset_sample_count": task_quality.get("sample_count", ""),
        "class_count": task_quality.get("class_count", ""),
        "labels_status": task_quality.get("labels_status", "not_applicable"),
        "targets_status": task_quality.get("targets_status", "not_applicable"),
        "reference_top1_first": task_quality.get("reference_top1_first", ""),
        "generated_top1_first": task_quality.get("generated_top1_first", ""),
        "prediction_agreement_vs_reference": task_quality.get("prediction_agreement_vs_reference", ""),
        "class_change_count": task_quality.get("class_change_count", ""),
        "reference_top1_accuracy": task_quality.get("reference_top1_accuracy", ""),
        "generated_top1_accuracy": task_quality.get("generated_top1_accuracy", ""),
        "top1_accuracy_drop_pct": task_quality.get("top1_accuracy_drop_pct", ""),
        "reference_top5_accuracy": task_quality.get("reference_top5_accuracy", ""),
        "generated_top5_accuracy": task_quality.get("generated_top5_accuracy", ""),
        "top5_accuracy_drop_pct": task_quality.get("top5_accuracy_drop_pct", ""),
        "confidence_delta_mean": task_quality.get("confidence_delta_mean", ""),
        "confidence_delta_max": task_quality.get("confidence_delta_max", ""),
        "reference_output_mae": task_quality.get("reference_output_mae", ""),
        "reference_output_rmse": task_quality.get("reference_output_rmse", ""),
        "target_mae_reference": task_quality.get("target_mae_reference", ""),
        "target_mae_generated": task_quality.get("target_mae_generated", ""),
        "target_rmse_reference": task_quality.get("target_rmse_reference", ""),
        "target_rmse_generated": task_quality.get("target_rmse_generated", ""),
        "target_r2_reference": task_quality.get("target_r2_reference", ""),
        "target_r2_generated": task_quality.get("target_r2_generated", ""),
        "mae_increase": task_quality.get("mae_increase", ""),
        "rmse_increase": task_quality.get("rmse_increase", ""),
        "hls_latency_max": row.get("hls_latency_max", ""),
        "vivado_lut": row.get("vivado_lut", ""),
        "vivado_dsp": row.get("vivado_dsp", ""),
        "vivado_bram": row.get("vivado_bram", ""),
        "vivado_power_w": row.get("vivado_power_w", ""),
        "vivado_wns": row.get("vivado_wns", ""),
    }


def _numeric_summary_rows(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in sorted(numeric_rows, key=_paper_sort_key)]


def _inference_precision_numeric_rows(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in numeric_rows:
        design = str(row.get("design_id") or "")
        if row.get("section") == "inference" and (design.startswith(("I0_", "I1_", "I2_")) or "precision" in design):
            rows.append(dict(row))
    return sorted(rows, key=_paper_sort_key)


def _training_numeric_rows(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted([dict(row) for row in numeric_rows if row.get("section") == "training"], key=_paper_sort_key)


def _training_precision_numeric_rows(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # This table is intentionally generated even before dedicated T9/T10/T11
    # precision rows exist.  It provides a stable destination for current
    # training-step numeric behavior and future training precision variants.
    return _training_numeric_rows(numeric_rows)




def _inference_task_quality_rows(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted([dict(row) for row in numeric_rows if row.get("section") == "inference"], key=_paper_sort_key)


def _precision_decision_matrix_rows(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _inference_precision_numeric_rows(numeric_rows):
        rows.append({
            **dict(row),
            "quality_metric": (
                row.get("generated_top1_accuracy")
                if row.get("generated_top1_accuracy") not in (None, "")
                else row.get("prediction_agreement_vs_reference")
                if row.get("prediction_agreement_vs_reference") not in (None, "")
                else row.get("cosine_similarity")
            ),
            "quality_metric_name": (
                "top1_accuracy"
                if row.get("generated_top1_accuracy") not in (None, "")
                else "prediction_agreement"
                if row.get("prediction_agreement_vs_reference") not in (None, "")
                else "cosine_similarity"
            ),
            "resource_saving_lut_vs_fx16": "",
            "resource_saving_bram_vs_fx16": "",
        })
    baseline = next((r for r in rows if str(r.get("design_id") or "").startswith("I0_")), None)
    if baseline:
        base_lut = _float_or_none(baseline.get("vivado_lut"))
        base_bram = _float_or_none(baseline.get("vivado_bram"))
        for row in rows:
            lut = _float_or_none(row.get("vivado_lut"))
            bram = _float_or_none(row.get("vivado_bram"))
            if base_lut not in (None, 0) and lut is not None:
                row["resource_saving_lut_vs_fx16"] = (base_lut - lut) / base_lut * 100.0
            if base_bram not in (None, 0) and bram is not None:
                row["resource_saving_bram_vs_fx16"] = (base_bram - bram) / base_bram * 100.0
    return rows


def _training_task_quality_rows(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _training_numeric_rows(numeric_rows):
        decision = "recommended_quality"
        reason = "training reference comparison is available"
        wd = _float_or_none(row.get("weight_delta_cosine"))
        wa = _float_or_none(row.get("weight_after_cosine"))
        gc = _float_or_none(row.get("gradient_cosine"))
        if row.get("numeric_quality") != "passed":
            decision = "pending_numeric_validation"
            reason = "training numeric validation did not pass"
        elif wd is not None and wd < 0.75:
            decision = "aggressive_training_update"
            reason = "weight-delta cosine is materially lower than the reference-update direction"
        elif (wd is not None and wd < 0.9) or (wa is not None and wa < 0.99):
            decision = "acceptable_training_tradeoff"
            reason = "training update differs from reference but preserves usable gradient/weight evidence"
        elif gc is not None and gc >= 0.99:
            decision = "reference_aligned_training_step"
            reason = "gradient direction is aligned with the Python reference"
        rows.append({**dict(row), "training_decision_status": decision, "training_decision_reason": reason})
    return rows

def _has_numeric_metric(rows: Sequence[Mapping[str, Any]], *keys: str) -> bool:
    return any(any(_float_or_none(row.get(key)) is not None for key in keys) for row in rows)

def _training_curve_rows(out_dir: Path, design_id: str) -> list[dict[str, Any]]:
    candidates = [
        out_dir / "reports" / "paper_training_curve.csv",
        out_dir / "training" / "training_curve.csv",
        out_dir / "reports" / "board_training_curve.csv",
        out_dir / "runtime_package" / "board_training_curve.csv",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open(newline="", encoding="utf-8") as f:
            rows: list[dict[str, Any]] = []
            for raw in csv.DictReader(f):
                row = {field: raw.get(field, "") for field in TRAINING_CURVE_FIELDS}
                row["design_id"] = raw.get("design_id") or raw.get("experiment") or design_id
                rows.append(row)
            if rows:
                return rows
    return []


def _xml(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _svg_axes(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="32" y="34" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111">{_xml(title)}</text>',
    ]
    if subtitle:
        lines.append(f'<text x="32" y="58" font-family="Arial, sans-serif" font-size="12" fill="#555">{_xml(subtitle)}</text>')
    return lines


def _finish_svg(lines: list[str], path: Path) -> Path:
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _bar_svg(rows: Sequence[Mapping[str, Any]], *, value_key: str, label_key: str, title: str, y_label: str, out_path: Path) -> Path | None:
    values: list[tuple[str, float]] = []
    for row in rows:
        value = _float_or_none(row.get(value_key))
        if value is None:
            continue
        values.append((str(row.get(label_key) or row.get("design_id") or "design"), value))
    if not values:
        return None
    width = 900
    height = max(360, 120 + len(values) * 42)
    left = 240
    right = 40
    top = 85
    row_h = 34
    max_v = max(v for _, v in values) or 1.0
    lines = _svg_axes(width, height, title, y_label)
    lines.append(f'<line x1="{left}" y1="{top-8}" x2="{left}" y2="{top + row_h*len(values)}" stroke="#333" stroke-width="1"/>')
    for idx, (label, value) in enumerate(values):
        y = top + idx * row_h
        bar_w = (width - left - right) * (value / max_v)
        lines.append(f'<text x="32" y="{y+18}" font-family="Arial, sans-serif" font-size="12" fill="#222">{_xml(label[:34])}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="22" rx="3" fill="#4a6fa5"/>')
        lines.append(f'<text x="{left + bar_w + 6:.2f}" y="{y+16}" font-family="Arial, sans-serif" font-size="12" fill="#222">{value:.4g}</text>')
    return _finish_svg(lines, out_path)


def _line_svg(rows: Sequence[Mapping[str, Any]], *, x_key: str, y_key: str, title: str, subtitle: str, out_path: Path) -> Path | None:
    points: list[tuple[float, float, Mapping[str, Any]]] = []
    for idx, row in enumerate(rows):
        x = _float_or_none(row.get(x_key))
        y = _float_or_none(row.get(y_key))
        if x is None:
            x = float(idx)
        if y is None:
            continue
        points.append((x, y, row))
    if len(points) < 2:
        return None
    width, height = 900, 420
    left, right, top, bottom = 70, 40, 80, 60
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        max_x += 1.0
    if math.isclose(min_y, max_y):
        max_y += 1.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    lines = _svg_axes(width, height, title, subtitle)
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    lines.append(f'<text x="{left}" y="{height-20}" font-family="Arial, sans-serif" font-size="12" fill="#444">{_xml(x_key)}</text>')
    lines.append(f'<text x="16" y="{top+18}" font-family="Arial, sans-serif" font-size="12" fill="#444" transform="rotate(-90 16,{top+18})">{_xml(y_key)}</text>')
    path_d = " ".join(("M" if i == 0 else "L") + f" {sx(x):.2f} {sy(y):.2f}" for i, (x, y, _) in enumerate(points))
    lines.append(f'<path d="{path_d}" fill="none" stroke="#4a6fa5" stroke-width="3"/>')
    for x, y, row in points:
        lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="#4a6fa5"/>')
    lines.append(f'<text x="{left}" y="{top+plot_h+24}" font-family="Arial, sans-serif" font-size="12" fill="#444">x: {min_x:.4g} → {max_x:.4g}</text>')
    lines.append(f'<text x="{left+220}" y="{top+plot_h+24}" font-family="Arial, sans-serif" font-size="12" fill="#444">y: {min_y:.4g} → {max_y:.4g}</text>')
    return _finish_svg(lines, out_path)


def _status_svg(statuses: Mapping[str, str], out_path: Path) -> Path:
    width, height = 980, max(360, 110 + len(statuses) * 44)
    lines = _svg_axes(width, height, "Paper plot manifest", "Created plots and pending real-measurement inputs")
    y = 88
    for name, status in statuses.items():
        fill = "#2e7d32" if status == "created" else ("#ef6c00" if status.startswith("pending") else "#777")
        lines.append(f'<circle cx="42" cy="{y-5}" r="7" fill="{fill}"/>')
        lines.append(f'<text x="60" y="{y}" font-family="Arial, sans-serif" font-size="13" fill="#111">{_xml(name)}: {_xml(status)}</text>')
        y += 34
    return _finish_svg(lines, out_path)




def _is_runtime_package_dir(path: Path) -> bool:
    return path.name == "runtime_package" or "runtime_package" in path.parts


def _is_paper_experiment_dir(path: Path) -> bool:
    return "paper_experiments" in path.parts


def _is_probably_compile_output(path: Path) -> bool:
    if _is_runtime_package_dir(path):
        return False
    if not ((path / "manifest.json").exists() and (path / "reports").exists()):
        return False
    # Exclude top-level aggregate/report directories that are not real compile outputs.
    if path.name in {"reports", "plots", "experiment_setup", "paper_results"}:
        return False
    return True

def _discover_out_dirs(paths: Iterable[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if _is_probably_compile_output(p):
            discovered.append(p)
            continue
        if p.exists() and p.is_dir():
            for candidate in sorted(p.rglob("manifest.json")):
                out = candidate.parent
                if _is_probably_compile_output(out):
                    discovered.append(out)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in discovered:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    # When the user points the plotter at a broad build directory that contains
    # the frozen paper experiment matrix, default to that matrix instead of
    # mixing in stale scratch/example outputs.  This keeps paper tables focused
    # while preserving backward compatibility for directories without
    # paper_experiments outputs.
    paper_rows = [p for p in out if _is_paper_experiment_dir(p)]
    if paper_rows:
        return sorted(paper_rows, key=lambda p: p.as_posix())
    return sorted(out, key=lambda p: p.as_posix())


def generate_paper_plot_artifacts(out_dirs: Iterable[str | Path], *, output_dir: str | Path = "paper_results/plots") -> dict[str, Any]:
    """Build paper plot data, SVG figures, and a manifest from compile outputs."""
    out = Path(output_dir)
    data_dir = out / "data"
    fig_dir = out / "figures"
    table_dir = out / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    compile_dirs = _discover_out_dirs(out_dirs)
    plot_rows = sorted((_plot_row(path) for path in compile_dirs), key=_paper_sort_key)
    knob_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    for path, row in zip(compile_dirs, plot_rows):
        knob_rows.extend(_knob_rows(path, row))
        numeric_rows.append(_numeric_validation_row(path, row))
        if row.get("section") == "training":
            curve_rows.extend(_training_curve_rows(path, str(row.get("design_id") or path.name)))

    inference_rows = [r for r in plot_rows if r.get("section") == "inference"]
    training_rows = [r for r in plot_rows if r.get("section") == "training"]
    runtime_inference = [r for r in inference_rows if _has_status(r.get("runtime_status"))]
    runtime_training = [r for r in training_rows if _has_status(r.get("runtime_status"))]

    _write_csv(data_dir / "paper_plot_rows.csv", plot_rows, PLOT_ROW_FIELDS)
    _write_csv(data_dir / "inference_design_rows.csv", inference_rows, PLOT_ROW_FIELDS)
    _write_csv(data_dir / "training_design_rows.csv", training_rows, PLOT_ROW_FIELDS)
    _write_csv(data_dir / "hardware_knob_settings.csv", knob_rows, KNOB_ROW_FIELDS)
    _write_csv(data_dir / "board_training_curve_rows.csv", curve_rows, TRAINING_CURVE_FIELDS)
    _write_csv(data_dir / "numeric_validation_rows.csv", numeric_rows, NUMERIC_ROW_FIELDS)
    table_files = _write_paper_tables(table_dir, plot_rows=plot_rows, knob_rows=knob_rows, curve_rows=curve_rows)
    comparison_rows = _comparison_rows(plot_rows)
    table_files["table_08_result_comparisons"] = _write_table_pair(
        table_dir,
        stem="table_08_result_comparisons",
        title="Computed result comparisons",
        rows=comparison_rows,
        fields=[
            "comparison",
            "group",
            "result_classification",
            "baseline_design",
            "variant_design",
            "metric_label",
            "baseline_value",
            "variant_value",
            "delta",
            "percent_change",
            "artifact_status",
            "interpretation",
        ],
        labels={
            "percent_change": "Change %",
            "baseline_value": "Baseline",
            "variant_value": "Variant",
            "result_classification": "Result classification",
            "artifact_status": "Artifact status",
        },
    )
    table_files["table_09_result_classification_summary"] = _write_table_pair(
        table_dir,
        stem="table_09_result_classification_summary",
        title="Result classification summary",
        rows=_classification_summary_rows(comparison_rows),
        fields=["result_classification", "comparison_count", "metric_row_count", "comparisons"],
        labels={"result_classification": "Result classification", "comparison_count": "Comparison count", "metric_row_count": "Metric row count"},
    )
    table_files["table_10_numeric_validation_summary"] = _write_table_pair(
        table_dir,
        stem="table_10_numeric_validation_summary",
        title="Numeric validation summary",
        rows=_numeric_summary_rows(numeric_rows),
        fields=["section", "design_id", "precision", "numeric_validation_status", "numeric_quality", "paper_numeric_claim_allowed", "max_abs_error", "mae", "rmse", "cosine_similarity", "gradient_cosine", "weight_after_cosine", "weight_delta_cosine", "loss_validation_status"],
        labels={"numeric_validation_status": "numeric status", "paper_numeric_claim_allowed": "claim allowed", "max_abs_error": "max abs error", "cosine_similarity": "cosine", "gradient_cosine": "gradient cosine", "weight_after_cosine": "weight-after cosine", "weight_delta_cosine": "weight-delta cosine"},
    )
    table_files["table_11_inference_precision_numeric_tradeoff"] = _write_table_pair(
        table_dir,
        stem="table_11_inference_precision_numeric_tradeoff",
        title="Inference precision numeric tradeoff",
        rows=_inference_precision_numeric_rows(numeric_rows),
        fields=["design_id", "precision", "hls_latency_max", "vivado_lut", "vivado_dsp", "vivado_bram", "vivado_power_w", "numeric_validation_status", "numeric_quality", "max_abs_error", "mae", "rmse", "cosine_similarity"],
        labels={"design_id": "Design", "hls_latency_max": "HLS latency cycles", "vivado_lut": "Vivado LUT", "vivado_dsp": "Vivado DSP", "vivado_bram": "Vivado BRAM", "vivado_power_w": "Power W", "max_abs_error": "max abs error", "cosine_similarity": "cosine"},
    )
    table_files["table_12_training_numeric_validation"] = _write_table_pair(
        table_dir,
        stem="table_12_training_numeric_validation",
        title="Training numeric validation",
        rows=_training_numeric_rows(numeric_rows),
        fields=["design_id", "precision", "numeric_validation_status", "numeric_quality", "gradient_cosine", "weight_after_cosine", "weight_delta_cosine", "grad_mae", "grad_max_abs", "weight_after_mae", "weight_after_max_abs", "initial_loss", "final_loss", "loss_delta", "gradient_export_status", "optimizer_state_status", "batch_accumulation_status", "training_tiled_io_status"],
        labels={"design_id": "Design", "gradient_cosine": "gradient cosine", "weight_after_cosine": "weight-after cosine", "weight_delta_cosine": "weight-delta cosine", "grad_mae": "gradient MAE", "grad_max_abs": "gradient max abs", "weight_after_mae": "weight-after MAE", "weight_after_max_abs": "weight-after max abs"},
    )
    table_files["table_13_training_precision_numeric_tradeoff"] = _write_table_pair(
        table_dir,
        stem="table_13_training_precision_numeric_tradeoff",
        title="Training precision numeric tradeoff",
        rows=_training_precision_numeric_rows(numeric_rows),
        fields=["design_id", "precision", "hls_latency_max", "vivado_lut", "vivado_dsp", "vivado_bram", "vivado_power_w", "numeric_validation_status", "numeric_quality", "gradient_cosine", "weight_after_cosine", "weight_delta_cosine", "grad_mae", "grad_max_abs"],
        labels={"design_id": "Design", "hls_latency_max": "HLS latency cycles", "vivado_lut": "Vivado LUT", "vivado_dsp": "Vivado DSP", "vivado_bram": "Vivado BRAM", "vivado_power_w": "Power W", "gradient_cosine": "gradient cosine", "weight_after_cosine": "weight-after cosine", "weight_delta_cosine": "weight-delta cosine"},
    )
    table_files["table_14_inference_task_quality_tradeoff"] = _write_table_pair(
        table_dir,
        stem="table_14_inference_task_quality_tradeoff",
        title="Inference task-quality tradeoff",
        rows=_inference_task_quality_rows(numeric_rows),
        fields=["design_id", "precision", "task", "dataset_source", "dataset_sample_count", "decision_status", "prediction_agreement_vs_reference", "class_change_count", "reference_top1_first", "generated_top1_first", "reference_top1_accuracy", "generated_top1_accuracy", "top1_accuracy_drop_pct", "target_mae_reference", "target_mae_generated", "mae_increase", "hls_latency_max", "vivado_lut", "vivado_bram"],
        labels={"design_id": "Design", "dataset_sample_count": "samples", "decision_status": "decision", "prediction_agreement_vs_reference": "prediction agreement", "reference_top1_first": "ref top-1", "generated_top1_first": "generated top-1", "reference_top1_accuracy": "ref top-1 acc", "generated_top1_accuracy": "generated top-1 acc", "top1_accuracy_drop_pct": "top-1 delta pct", "target_mae_reference": "ref target MAE", "target_mae_generated": "generated target MAE", "mae_increase": "MAE increase", "hls_latency_max": "HLS latency cycles", "vivado_lut": "Vivado LUT", "vivado_bram": "Vivado BRAM"},
    )
    table_files["table_15_precision_decision_matrix"] = _write_table_pair(
        table_dir,
        stem="table_15_precision_decision_matrix",
        title="Precision decision matrix",
        rows=_precision_decision_matrix_rows(numeric_rows),
        fields=["design_id", "precision", "decision_status", "decision_reason", "quality_metric_name", "quality_metric", "max_abs_error", "mae", "rmse", "cosine_similarity", "resource_saving_lut_vs_fx16", "resource_saving_bram_vs_fx16", "hls_latency_max", "vivado_lut", "vivado_dsp", "vivado_bram", "vivado_power_w"],
        labels={"design_id": "Design", "decision_status": "decision", "decision_reason": "decision reason", "quality_metric_name": "quality metric", "resource_saving_lut_vs_fx16": "LUT saving vs fx16 %", "resource_saving_bram_vs_fx16": "BRAM saving vs fx16 %", "hls_latency_max": "HLS latency cycles", "vivado_lut": "Vivado LUT", "vivado_dsp": "Vivado DSP", "vivado_bram": "Vivado BRAM", "vivado_power_w": "Power W"},
    )
    table_files["table_16_training_task_quality_tradeoff"] = _write_table_pair(
        table_dir,
        stem="table_16_training_task_quality_tradeoff",
        title="Training task-quality tradeoff",
        rows=_training_task_quality_rows(numeric_rows),
        fields=["design_id", "precision", "training_decision_status", "training_decision_reason", "initial_loss", "final_loss", "loss_delta", "gradient_cosine", "weight_after_cosine", "weight_delta_cosine", "grad_mae", "weight_after_mae", "hls_latency_max", "vivado_lut", "vivado_dsp", "vivado_bram"],
        labels={"design_id": "Design", "training_decision_status": "decision", "training_decision_reason": "decision reason", "gradient_cosine": "gradient cosine", "weight_after_cosine": "weight-after cosine", "weight_delta_cosine": "weight-delta cosine", "grad_mae": "gradient MAE", "weight_after_mae": "weight-after MAE", "hls_latency_max": "HLS latency cycles", "vivado_lut": "Vivado LUT", "vivado_dsp": "Vivado DSP", "vivado_bram": "Vivado BRAM"},
    )
    table_files["table_17_training_precision_decision_matrix"] = _write_table_pair(
        table_dir,
        stem="table_17_training_precision_decision_matrix",
        title="Training precision decision matrix",
        rows=_training_task_quality_rows(numeric_rows),
        fields=["design_id", "precision", "training_decision_status", "gradient_cosine", "weight_delta_cosine", "loss_delta", "hls_latency_max", "vivado_lut", "vivado_dsp", "vivado_bram", "vivado_power_w"],
        labels={"design_id": "Design", "training_decision_status": "decision", "gradient_cosine": "gradient cosine", "weight_delta_cosine": "weight-delta cosine", "hls_latency_max": "HLS latency cycles", "vivado_lut": "Vivado LUT", "vivado_dsp": "Vivado DSP", "vivado_bram": "Vivado BRAM", "vivado_power_w": "Power W"},
    )

    created: dict[str, str] = {}
    pending: dict[str, str] = {}

    def record(name: str, path: Path | None, pending_reason: str) -> None:
        if path is not None and path.exists():
            created[name] = path.as_posix()
        else:
            pending[name] = pending_reason

    record("figure_00_plot_status", _status_svg({}, fig_dir / "figure_00_plot_status.svg"), "internal")
    record(
        "figure_01_inference_hls_latency",
        _bar_svg(inference_rows, value_key="hls_latency_max", label_key="design_id", title="Inference HLS latency by design", y_label="HLS latency cycles", out_path=fig_dir / "figure_01_inference_hls_latency.svg"),
        "no inference HLS latency rows available",
    )
    record(
        "figure_02_inference_vivado_lut",
        _bar_svg(inference_rows, value_key="vivado_lut", label_key="design_id", title="Inference Vivado LUT by design", y_label="Implemented LUTs", out_path=fig_dir / "figure_02_inference_vivado_lut.svg"),
        "no inference Vivado implementation rows available",
    )
    record(
        "figure_03_inference_real_latency_ms",
        _bar_svg(runtime_inference, value_key="runtime_latency_ms_mean", label_key="design_id", title="Real inference latency", y_label="Mean latency (ms)", out_path=fig_dir / "figure_03_inference_real_latency_ms.svg"),
        "pending real inference board-runtime measurements",
    )
    record(
        "figure_04_inference_energy_mj",
        _bar_svg(runtime_inference, value_key="runtime_energy_mj_estimated", label_key="design_id", title="Real inference energy", y_label="Energy per inference (mJ, power × latency)", out_path=fig_dir / "figure_04_inference_energy_mj.svg"),
        "pending real inference latency/power measurements",
    )
    record(
        "figure_05_training_hls_latency",
        _bar_svg(training_rows, value_key="hls_latency_max", label_key="design_id", title="Training HLS latency by design", y_label="HLS latency cycles", out_path=fig_dir / "figure_05_training_hls_latency.svg"),
        "no training HLS latency rows available",
    )
    record(
        "figure_06_training_vivado_lut",
        _bar_svg(training_rows, value_key="vivado_lut", label_key="design_id", title="Training Vivado LUT by design", y_label="Implemented LUTs", out_path=fig_dir / "figure_06_training_vivado_lut.svg"),
        "no training Vivado implementation rows available",
    )
    record(
        "figure_07_training_step_ms",
        _bar_svg(runtime_training, value_key="runtime_training_step_ms", label_key="design_id", title="Real training step latency", y_label="Training step latency (ms)", out_path=fig_dir / "figure_07_training_step_ms.svg"),
        "pending real training board-runtime measurements",
    )
    record(
        "figure_08_training_curve_loss",
        _line_svg(curve_rows, x_key="step", y_key="loss", title="Real FPGA training loss curve", subtitle="Generated only from board-runtime curve rows", out_path=fig_dir / "figure_08_training_curve_loss.svg"),
        "pending real FPGA training-curve rows",
    )
    record(
        "figure_09_knob_status_counts",
        _bar_svg(_status_count_rows(knob_rows), value_key="count", label_key="status", title="Hardware knob status coverage", y_label="Knob count", out_path=fig_dir / "figure_09_knob_status_counts.svg"),
        "no hardware knob contract rows available",
    )
    record(
        "figure_10_training_vivado_power",
        _bar_svg(training_rows, value_key="vivado_power_w", label_key="design_id", title="Training Vivado power by design", y_label="Implemented design power (W)", out_path=fig_dir / "figure_10_training_vivado_power.svg"),
        "no training Vivado power rows available",
    )
    record(
        "figure_11_training_vivado_dsp",
        _bar_svg(training_rows, value_key="vivado_dsp", label_key="design_id", title="Training Vivado DSP by design", y_label="Implemented DSP count", out_path=fig_dir / "figure_11_training_vivado_dsp.svg"),
        "no training Vivado DSP rows available",
    )
    record(
        "figure_12_all_hls_lut",
        _bar_svg(plot_rows, value_key="hls_lut", label_key="design_id", title="HLS LUT by design", y_label="HLS LUT estimate", out_path=fig_dir / "figure_12_all_hls_lut.svg"),
        "no HLS LUT rows available",
    )
    record(
        "figure_13_all_vivado_power",
        _bar_svg(plot_rows, value_key="vivado_power_w", label_key="design_id", title="Vivado power by design", y_label="Implemented design power (W)", out_path=fig_dir / "figure_13_all_vivado_power.svg"),
        "no Vivado power rows available",
    )
    record(
        "figure_14_all_vivado_wns",
        _bar_svg(plot_rows, value_key="vivado_wns", label_key="design_id", title="Vivado timing slack by design", y_label="Worst negative slack / WNS (ns)", out_path=fig_dir / "figure_14_all_vivado_wns.svg"),
        "no Vivado WNS rows available",
    )
    inference_numeric = _inference_precision_numeric_rows(numeric_rows)
    training_numeric = _training_numeric_rows(numeric_rows)
    record(
        "figure_15_precision_numeric_error",
        _bar_svg(inference_numeric, value_key="max_abs_error", label_key="design_id", title="Inference precision numeric error", y_label="max abs error versus reference", out_path=fig_dir / "figure_15_precision_numeric_error.svg"),
        "pending inference numeric output comparisons",
    )
    record(
        "figure_16_precision_resource_numeric_tradeoff",
        _line_svg(inference_numeric, x_key="max_abs_error", y_key="vivado_lut", title="Precision numeric-resource tradeoff", subtitle="x=max abs error, y=Vivado LUT", out_path=fig_dir / "figure_16_precision_resource_numeric_tradeoff.svg"),
        "pending precision numeric/resource tradeoff points",
    )
    record(
        "figure_17_training_gradient_fidelity",
        _bar_svg(training_numeric, value_key="gradient_cosine", label_key="design_id", title="Training gradient fidelity", y_label="gradient cosine vs reference", out_path=fig_dir / "figure_17_training_gradient_fidelity.svg"),
        "pending training gradient comparison artifacts",
    )
    task_quality_rows = _inference_task_quality_rows(numeric_rows)
    precision_decision_rows = _precision_decision_matrix_rows(numeric_rows)
    training_decision_rows = _training_task_quality_rows(numeric_rows)
    record(
        "figure_18_inference_quality_vs_lut",
        _line_svg(task_quality_rows, x_key="prediction_agreement_vs_reference", y_key="vivado_lut", title="Inference quality-resource decision", subtitle="x=prediction agreement, y=Vivado LUT", out_path=fig_dir / "figure_18_inference_quality_vs_lut.svg"),
        "pending inference task-quality agreement metrics",
    )
    record(
        "figure_19_precision_decision_lut_saving",
        _bar_svg(precision_decision_rows, value_key="resource_saving_lut_vs_fx16", label_key="design_id", title="Precision LUT saving versus fx16", y_label="LUT saving vs fx16 (%)", out_path=fig_dir / "figure_19_precision_decision_lut_saving.svg"),
        "pending precision decision matrix rows",
    )
    record(
        "figure_20_error_vs_bram",
        _line_svg(precision_decision_rows, x_key="max_abs_error", y_key="vivado_bram", title="Precision error-BRAM tradeoff", subtitle="x=max abs error, y=Vivado BRAM", out_path=fig_dir / "figure_20_error_vs_bram.svg"),
        "pending precision error/resource metrics",
    )
    record(
        "figure_21_training_loss_delta",
        _bar_svg(training_decision_rows, value_key="loss_delta", label_key="design_id", title="Training loss delta by design", y_label="final loss - initial loss", out_path=fig_dir / "figure_21_training_loss_delta.svg"),
        "pending training loss-delta metrics",
    )
    record(
        "figure_22_training_update_vs_lut",
        _line_svg(training_decision_rows, x_key="weight_delta_cosine", y_key="vivado_lut", title="Training update-resource decision", subtitle="x=weight-delta cosine, y=Vivado LUT", out_path=fig_dir / "figure_22_training_update_vs_lut.svg"),
        "pending training update/resource metrics",
    )

    status_for_svg = {**{k: "created" for k in created}, **{k: "pending" for k in pending}}
    _status_svg(status_for_svg, fig_dir / "figure_00_plot_status.svg")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "created",
        "compile_output_count": len(compile_dirs),
        "sections": {
            "inference": {
                "design_count": len(inference_rows),
                "real_runtime_rows": len(runtime_inference),
                "numeric_validation_rows": sum(1 for row in numeric_rows if row.get("section") == "inference"),
                "numeric_passed_rows": sum(1 for row in numeric_rows if row.get("section") == "inference" and row.get("numeric_quality") == "passed"),
                "required_real_measurements": ["runtime_latency_ms_mean", "runtime_throughput", "runtime_accuracy", "vivado_power_w", "runtime_energy_mj_estimated"],
            },
            "training": {
                "design_count": len(training_rows),
                "real_runtime_rows": len(runtime_training),
                "real_training_curve_rows": len(curve_rows),
                "numeric_validation_rows": sum(1 for row in numeric_rows if row.get("section") == "training"),
                "numeric_passed_rows": sum(1 for row in numeric_rows if row.get("section") == "training" and row.get("numeric_quality") == "passed"),
                "required_real_measurements": ["runtime_training_step_ms", "runtime_loss_before", "runtime_loss_after", "board_training_curve.csv"],
            },
        },
        "created_figures": created,
        "pending_figures": pending,
        "data_files": {
            "paper_plot_rows_csv": (data_dir / "paper_plot_rows.csv").as_posix(),
            "inference_design_rows_csv": (data_dir / "inference_design_rows.csv").as_posix(),
            "training_design_rows_csv": (data_dir / "training_design_rows.csv").as_posix(),
            "hardware_knob_settings_csv": (data_dir / "hardware_knob_settings.csv").as_posix(),
            "board_training_curve_rows_csv": (data_dir / "board_training_curve_rows.csv").as_posix(),
            "numeric_validation_rows_csv": (data_dir / "numeric_validation_rows.csv").as_posix(),
        },
        "table_files": table_files,
        "claim_boundary": "Plots are generated only from existing report/runtime artifacts. Missing board-runtime latency, energy, and training-curve measurements are marked pending, not fabricated.",
    }
    narrative_files = _write_captions_and_summary(
        out,
        manifest=manifest,
        plot_rows=plot_rows,
        table_files=table_files,
        comparison_rows=comparison_rows,
        numeric_rows=numeric_rows,
    )
    manifest["narrative_files"] = narrative_files
    _write_json(out / "paper_plot_manifest.json", manifest)
    _write_md(out / "paper_plot_manifest.md", manifest)
    return manifest


def _status_count_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return [{"status": status, "count": count} for status, count in sorted(counts.items())]




def _pct_change(new: Any, base: Any) -> str:
    new_f = _float_or_none(new)
    base_f = _float_or_none(base)
    if new_f is None or base_f is None or math.isclose(base_f, 0.0):
        return ""
    return f"{((new_f - base_f) / base_f) * 100.0:.2f}"


def _metric_delta(new: Any, base: Any) -> str:
    new_f = _float_or_none(new)
    base_f = _float_or_none(base)
    if new_f is None or base_f is None:
        return ""
    return f"{new_f - base_f:.4g}"


def _row_by_design(plot_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("design_id") or ""): row for row in plot_rows if row.get("design_id")}



_COMPARISON_METRICS: list[tuple[str, str]] = [
    ("hls_latency_max", "HLS latency cycles"),
    ("hls_lut", "HLS LUT"),
    ("hls_dsp", "HLS DSP"),
    ("hls_bram", "HLS BRAM"),
    ("vivado_lut", "Vivado LUT"),
    ("vivado_dsp", "Vivado DSP"),
    ("vivado_bram", "Vivado BRAM"),
    ("vivado_power_w", "Power W"),
    ("vivado_wns", "WNS ns"),
]

_COMPARISON_DEFINITIONS: list[dict[str, str]] = [
    {
        "comparison": "precision_fx8_vs_fx16",
        "baseline": "I0_baseline_fx16_embedded",
        "variant": "I1_precision_fx8_embedded",
        "group": "Precision",
        "description": "fx8_3 versus fx16_6 baseline",
        "expected_effect": "lower precision should reduce resources with limited latency change",
        "classification_hint": "strong_result",
    },
    {
        "comparison": "precision_fx24_vs_fx16",
        "baseline": "I0_baseline_fx16_embedded",
        "variant": "I2_precision_fx24_embedded",
        "group": "Precision",
        "description": "fx24_10 versus fx16_6 baseline",
        "expected_effect": "higher precision should increase arithmetic/storage cost",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "parallel_pe2_vs_pe1",
        "baseline": "I0_baseline_fx16_embedded",
        "variant": "I3_parallel_pe2",
        "group": "Parallelism",
        "description": "PE=2/SIMD=2 versus baseline PE=1",
        "expected_effect": "parallelism should reduce HLS latency while increasing resources",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "parallel_pe4_vs_pe1",
        "baseline": "I0_baseline_fx16_embedded",
        "variant": "I4_parallel_pe4",
        "group": "Parallelism",
        "description": "PE=4/SIMD=4 versus baseline PE=1",
        "expected_effect": "stronger parallelism should further reduce HLS latency with higher resource cost",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "parallel_pe4_vs_pe2",
        "baseline": "I3_parallel_pe2",
        "variant": "I4_parallel_pe4",
        "group": "Parallelism",
        "description": "PE=4/SIMD=4 versus PE=2/SIMD=2",
        "expected_effect": "stronger parallelism should reduce latency at additional DSP/resource cost",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "weight_import_m_axi_vs_embedded",
        "baseline": "I0_baseline_fx16_embedded",
        "variant": "I7_weight_import_m_axi",
        "group": "Inference memory movement",
        "description": "m_axi weight import versus embedded weights",
        "expected_effect": "runtime weight import changes memory interfaces and implementation resources",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "pipeline_latency_first_vs_baseline",
        "baseline": "I0_baseline_fx16_embedded",
        "variant": "I5_pipeline_latency_first",
        "group": "Pipeline policy",
        "description": "latency-first pipeline policy versus baseline",
        "expected_effect": "latency-first policy should change generated HLS/Vivado schedules/resources",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "pipeline_resource_first_vs_baseline",
        "baseline": "I0_baseline_fx16_embedded",
        "variant": "I6_pipeline_resource_first",
        "group": "Pipeline policy",
        "description": "resource-first pipeline policy versus baseline",
        "expected_effect": "resource-first policy may match baseline when no additional optimization is requested",
        "classification_hint": "no_observable_effect",
    },
    {
        "comparison": "training_momentum_vs_sgd",
        "baseline": "T0_sgd_tiled_m_axi",
        "variant": "T1_momentum_tiled_m_axi",
        "group": "Training optimizer",
        "description": "Momentum optimizer versus SGD baseline",
        "expected_effect": "momentum may reuse the same datapath for this compact model unless state movement changes are materialized",
        "classification_hint": "no_observable_effect",
    },
    {
        "comparison": "training_adam_vs_sgd",
        "baseline": "T0_sgd_tiled_m_axi",
        "variant": "T2_adam_tiled_m_axi",
        "group": "Training optimizer",
        "description": "Adam optimizer-state/update path versus SGD baseline",
        "expected_effect": "Adam should add optimizer-state storage/movement cost",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "training_cross_entropy_vs_mse",
        "baseline": "T0_sgd_tiled_m_axi",
        "variant": "T3_cross_entropy_tiled_m_axi",
        "group": "Training loss",
        "description": "Cross-entropy training loss versus MSE/SGD baseline",
        "expected_effect": "cross-entropy should add arithmetic/resource cost",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "training_tile32_vs_tile64",
        "baseline": "T0_sgd_tiled_m_axi",
        "variant": "T4_tile32_m_axi",
        "group": "Training memory",
        "description": "tile32 m_axi training movement versus tile64 baseline",
        "expected_effect": "smaller training tile can change HLS latency and implementation timing/resource balance",
        "classification_hint": "strong_result",
    },
    {
        "comparison": "training_tile128_vs_tile64",
        "baseline": "T0_sgd_tiled_m_axi",
        "variant": "T5_tile128_m_axi",
        "group": "Training memory",
        "description": "tile128 m_axi training movement versus tile64 baseline",
        "expected_effect": "larger training tile can increase local buffering/schedule cost",
        "classification_hint": "expected_tradeoff",
    },
    {
        "comparison": "training_accum_batch2_vs_sgd",
        "baseline": "T0_sgd_tiled_m_axi",
        "variant": "T6_accum_batch2_m_axi",
        "group": "Training accumulation",
        "description": "accumulated batch-2 training versus one-step SGD baseline",
        "expected_effect": "accumulation support may reuse the same synthesized datapath in the compact test design",
        "classification_hint": "no_observable_effect",
    },
    {
        "comparison": "training_bitstream_vs_sgd",
        "baseline": "T0_sgd_tiled_m_axi",
        "variant": "T7_deployable_training_bitstream",
        "group": "Training deployability",
        "description": "training bitstream/package candidate versus SGD tiled baseline",
        "expected_effect": "deployability row should preserve design metrics while validating packaging/implementation flow",
        "classification_hint": "deployability_result",
    },
]


def _artifact_status_for_comparison(base: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    hls_values = {str(base.get("hls_status") or "").lower(), str(variant.get("hls_status") or "").lower()}
    vivado_values = {
        str(base.get("vivado_implementation_status") or "").lower(),
        str(variant.get("vivado_implementation_status") or "").lower(),
    }
    hls_ok = all(value in {"parsed", "passed", "ok", "available"} for value in hls_values if value)
    vivado_ok = all(value in {"passed", "implemented", "ok", "available"} for value in vivado_values if value)
    if hls_ok and vivado_ok:
        return "hls_vivado_passed"
    if hls_ok:
        return "hls_only"
    return "artifact_status_incomplete"


def _changed_metric_count(rows: Sequence[Mapping[str, Any]]) -> int:
    changed = 0
    for row in rows:
        delta = _float_or_none(row.get("delta"))
        if delta is not None and not math.isclose(delta, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            changed += 1
    return changed


def _comparison_latency_improved(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if row.get("metric") != "hls_latency_max":
            continue
        delta = _float_or_none(row.get("delta"))
        return delta is not None and delta < 0
    return False


def _comparison_resources_increased(rows: Sequence[Mapping[str, Any]]) -> bool:
    resource_metrics = {"hls_lut", "hls_dsp", "hls_bram", "vivado_lut", "vivado_dsp", "vivado_bram", "vivado_power_w"}
    for row in rows:
        if row.get("metric") not in resource_metrics:
            continue
        delta = _float_or_none(row.get("delta"))
        if delta is not None and delta > 0:
            return True
    return False


def _classify_comparison(definition: Mapping[str, str], rows: Sequence[Mapping[str, Any]], base: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    group = str(definition.get("group") or "")
    hint = str(definition.get("classification_hint") or "")
    runtime_statuses = {str(base.get("runtime_status") or "").lower(), str(variant.get("runtime_status") or "").lower()}
    if any(value and value not in {"not_run", "missing", "not_requested", "required_validation"} for value in runtime_statuses):
        return "pending_runtime" if not rows else hint or "strong_result"
    if "deployability" in group.lower() or hint == "deployability_result":
        return "deployability_result"
    if not rows:
        return "pending_runtime"
    if _artifact_status_for_comparison(base, variant) == "artifact_status_incomplete":
        return "pending_runtime"
    if _changed_metric_count(rows) == 0:
        return "no_observable_effect"
    if hint == "no_observable_effect":
        return "no_observable_effect" if _changed_metric_count(rows) == 0 else "expected_tradeoff"
    if _comparison_latency_improved(rows) and _comparison_resources_increased(rows):
        return "expected_tradeoff"
    return hint or "strong_result"


def _comparison_interpretation(definition: Mapping[str, str], classification: str, rows: Sequence[Mapping[str, Any]]) -> str:
    changed = _changed_metric_count(rows)
    expected = str(definition.get("expected_effect") or "")
    if classification == "no_observable_effect":
        return "No observable HLS/Vivado metric change in the current frozen artifacts; use as coverage/deployability evidence, not as a performance-resource claim."
    if classification == "deployability_result":
        return "Use as an implementation/package validation point; identical metrics are acceptable because the comparison isolates deployability rather than a hardware optimization knob."
    if classification == "pending_runtime":
        return "Runtime-dependent claim remains pending until board-runtime measurements are imported."
    if classification == "expected_tradeoff":
        return f"Expected tradeoff: {expected} Metrics changed in {changed} tracked HLS/Vivado field(s)."
    return f"Strong artifact-backed result: {expected} Metrics changed in {changed} tracked HLS/Vivado field(s)."


def _comparison_rows(plot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return paper-ready pairwise comparisons for the frozen matrix.

    The comparison set is deliberately explicit rather than inferred from file
    names.  This keeps the paper narrative stable and makes every claim tied to
    a named baseline/variant pair.
    """
    by_design = _row_by_design(plot_rows)
    rows: list[dict[str, Any]] = []
    for definition in _COMPARISON_DEFINITIONS:
        comparison_id = definition["comparison"]
        base_id = definition["baseline"]
        variant_id = definition["variant"]
        base = by_design.get(base_id)
        variant = by_design.get(variant_id)
        if base is None or variant is None:
            continue
        per_metric: list[dict[str, Any]] = []
        for metric, label in _COMPARISON_METRICS:
            base_value = base.get(metric)
            variant_value = variant.get(metric)
            if _float_or_none(base_value) is None or _float_or_none(variant_value) is None:
                continue
            per_metric.append(
                {
                    "comparison": comparison_id,
                    "group": definition["group"],
                    "description": definition["description"],
                    "expected_effect": definition["expected_effect"],
                    "baseline_design": base_id,
                    "variant_design": variant_id,
                    "metric": metric,
                    "metric_label": label,
                    "baseline_value": base_value,
                    "variant_value": variant_value,
                    "delta": _metric_delta(variant_value, base_value),
                    "percent_change": _pct_change(variant_value, base_value),
                    "artifact_status": _artifact_status_for_comparison(base, variant),
                }
            )
        classification = _classify_comparison(definition, per_metric, base, variant)
        interpretation = _comparison_interpretation(definition, classification, per_metric)
        for row in per_metric:
            row["result_classification"] = classification
            row["interpretation"] = interpretation
            rows.append(row)
    return rows



def _classification_summary_rows(comparison_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    comparison_names: dict[str, set[str]] = {}
    for row in comparison_rows:
        classification = str(row.get("result_classification") or "unknown")
        comparison = str(row.get("comparison") or "")
        bucket = buckets.setdefault(
            classification,
            {
                "result_classification": classification,
                "comparison_count": 0,
                "metric_row_count": 0,
                "comparisons": "",
            },
        )
        bucket["metric_row_count"] += 1
        comparison_names.setdefault(classification, set()).add(comparison)
    order = {
        "strong_result": 0,
        "expected_tradeoff": 1,
        "deployability_result": 2,
        "no_observable_effect": 3,
        "pending_runtime": 4,
    }
    rows: list[dict[str, Any]] = []
    for classification, bucket in buckets.items():
        names = sorted(x for x in comparison_names.get(classification, set()) if x)
        bucket["comparison_count"] = len(names)
        bucket["comparisons"] = ", ".join(names)
        rows.append(bucket)
    return sorted(rows, key=lambda row: (order.get(str(row.get("result_classification")), 99), str(row.get("result_classification"))))


def _representative_comparison_rows(comparison_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Choose one readable row per comparison for summaries/claim prose.

    Prefer latency, then LUT, then power so summaries do not become dominated by
    every resource column from the first comparison.
    """
    preference = ["hls_latency_max", "vivado_lut", "hls_lut", "vivado_power_w", "vivado_wns"]
    by_comparison: dict[str, list[Mapping[str, Any]]] = {}
    for row in comparison_rows:
        by_comparison.setdefault(str(row.get("comparison") or ""), []).append(row)
    representatives: list[Mapping[str, Any]] = []
    for comparison, rows in by_comparison.items():
        if not comparison:
            continue
        chosen = rows[0]
        for metric in preference:
            match = next((row for row in rows if row.get("metric") == metric), None)
            if match is not None:
                chosen = match
                break
        representatives.append(chosen)
    return representatives


def _comparison_sentence(row: Mapping[str, Any]) -> str:
    metric = str(row.get("metric_label") or row.get("metric") or "metric")
    variant = str(row.get("variant_design") or "variant")
    base = str(row.get("baseline_design") or "baseline")
    pct = row.get("percent_change")
    delta = row.get("delta")
    direction = "changed"
    try:
        pct_f = float(str(pct))
        if pct_f < 0:
            direction = "decreased"
        elif pct_f > 0:
            direction = "increased"
    except Exception:
        pct_f = None
    if pct_f is None:
        return f"{metric}: {variant} differs from {base} by {delta}."
    return f"{metric}: {variant} {direction} by {abs(pct_f):.2f}% versus {base} (delta {delta})."


def _figure_caption_rows(created: Mapping[str, str], pending: Mapping[str, str]) -> list[dict[str, Any]]:
    captions = {
        "figure_00_plot_status": "Artifact-status overview showing which paper figures are generated and which require real board-runtime measurements.",
        "figure_01_inference_hls_latency": "Inference HLS latency across the frozen inference subset, generated from Vitis HLS synthesis reports.",
        "figure_02_inference_vivado_lut": "Implemented inference LUT usage across the frozen inference subset, generated from Vivado implementation reports.",
        "figure_03_inference_real_latency_ms": "Real KV260 inference latency; generated only after board-runtime measurements are imported.",
        "figure_04_inference_energy_mj": "Estimated real inference energy per inference using board-runtime latency and Vivado power.",
        "figure_05_training_hls_latency": "Training HLS latency across the frozen training subset, generated from Vitis HLS synthesis reports.",
        "figure_06_training_vivado_lut": "Implemented training LUT usage across the frozen training subset, generated from Vivado implementation reports.",
        "figure_07_training_step_ms": "Real KV260 training-step latency; generated only after board-runtime training measurements are imported.",
        "figure_08_training_curve_loss": "Real FPGA training loss curve; generated only from board-runtime training curve rows.",
        "figure_09_knob_status_counts": "Coverage of YAML/hardware knob application status across generated hardware knob contracts.",
        "figure_10_training_vivado_power": "Implemented training design power reported by Vivado for the frozen training subset.",
        "figure_11_training_vivado_dsp": "Implemented training DSP usage reported by Vivado for the frozen training subset.",
        "figure_12_all_hls_lut": "HLS LUT comparison across all frozen paper designs with HLS reports.",
        "figure_13_all_vivado_power": "Vivado power comparison across all frozen paper designs with implementation reports.",
        "figure_14_all_vivado_wns": "Vivado timing slack comparison across all frozen paper designs with implementation reports.",
        "figure_15_precision_numeric_error": "Inference precision numeric error against the Python/ONNX reference for precision-focused rows, generated only when reference-output comparison artifacts are available.",
        "figure_16_precision_resource_numeric_tradeoff": "Precision tradeoff between reference-output error and implemented LUT cost, generated from numeric-validation and Vivado artifacts.",
        "figure_17_training_gradient_fidelity": "Training gradient fidelity against the Python reference across on-device-training rows, generated from numeric-validation artifacts.",
        "figure_18_inference_quality_vs_lut": "Inference task-quality/resource decision plot. When labels are available, quality can be task accuracy; otherwise it uses generated-vs-reference prediction agreement.",
        "figure_19_precision_decision_lut_saving": "Precision decision plot showing implemented LUT saving versus the fx16 baseline for precision-focused rows.",
        "figure_20_error_vs_bram": "Precision decision plot showing output error versus implemented BRAM usage.",
        "figure_21_training_loss_delta": "Training decision plot showing loss change over the available reference/testbench training step artifacts.",
        "figure_22_training_update_vs_lut": "Training decision plot showing update-direction fidelity against implemented LUT cost.",
    }
    rows: list[dict[str, Any]] = []
    for name in sorted(set(captions) | set(created) | set(pending)):
        rows.append({
            "figure": name,
            "status": "created" if name in created else "pending",
            "path": created.get(name, ""),
            "caption": captions.get(name, pending.get(name, "")),
            "pending_reason": pending.get(name, ""),
        })
    return rows


def _table_caption_rows(table_files: Mapping[str, Any]) -> list[dict[str, Any]]:
    captions = {
        "table_01_experiment_overview": "Overview of frozen paper experiment rows, pipeline mode, compact precision label, and artifact support status.",
        "table_02_design_knobs": "Resolved YAML/hardware knob settings and where each knob was applied in the compiler/hardware flow.",
        "table_03_hls_results": "Vitis HLS synthesis metrics used for paper HLS latency/resource comparisons.",
        "table_04_vivado_results": "Vivado implementation metrics used for paper resource, timing, and power comparisons.",
        "table_05_runtime_status": "Board-runtime measurement availability and pending runtime metrics.",
        "table_06_pending_measurements": "Explicit list of real board measurements still required before claiming runtime or training-curve results.",
        "table_07_knob_effect_summary": "Aggregated coverage of knob statuses across generated hardware knob contracts.",
        "table_08_result_comparisons": "Pairwise paper comparisons computed from the frozen experiment subset, including automatic result classification and interpretation.",
        "table_09_result_classification_summary": "Summary count of comparison classifications used to separate strong results, expected tradeoffs, deployability rows, no-observable-effect rows, and pending runtime claims.",
        "table_10_numeric_validation_summary": "Numeric validation status for each frozen design, including inference output-error metrics and training gradient/weight-update metrics when available.",
        "table_11_inference_precision_numeric_tradeoff": "Inference precision rows combining HLS/Vivado resource metrics with numeric output fidelity metrics.",
        "table_12_training_numeric_validation": "Training numeric validation metrics for gradients, weight updates, losses, optimizer-state export, accumulation, and tiled I/O where available.",
        "table_13_training_precision_numeric_tradeoff": "Training precision/numeric behavior table prepared for current training rows and future training precision variants.",
        "table_14_inference_task_quality_tradeoff": "Task-aware inference decision table combining generated-vs-reference prediction agreement, optional classification/regression dataset metrics, and HLS/Vivado costs.",
        "table_15_precision_decision_matrix": "Precision decision matrix that lets users choose fx8/fx16/fx24 using quality metrics, output error, and resource savings.",
        "table_16_training_task_quality_tradeoff": "Task-aware training decision table combining loss/update fidelity and HLS/Vivado costs.",
        "table_17_training_precision_decision_matrix": "Training decision matrix prepared for current training rows and future training precision/dataset variants.",
    }
    rows: list[dict[str, Any]] = []
    for name in sorted(set(captions) | set(table_files)):
        value = table_files.get(name, {})
        rows.append({
            "table": name,
            "path": value.get("md", "") if isinstance(value, Mapping) else str(value),
            "row_count": value.get("row_count", "") if isinstance(value, Mapping) else "",
            "caption": captions.get(name, ""),
        })
    return rows



def _paper_results_subsection_text(
    plot_rows: Sequence[Mapping[str, Any]],
    representative_rows: Sequence[Mapping[str, Any]],
    classification_rows: Sequence[Mapping[str, Any]],
    numeric_rows: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    inference_count = sum(1 for row in plot_rows if row.get("section") == "inference")
    training_count = sum(1 for row in plot_rows if row.get("section") == "training")
    hls_count = sum(1 for row in plot_rows if _float_or_none(row.get("hls_latency_max")) is not None)
    vivado_count = sum(1 for row in plot_rows if _float_or_none(row.get("vivado_lut")) is not None)
    runtime_count = sum(1 for row in plot_rows if _has_status(row.get("runtime_status")))
    numeric_rows = list(numeric_rows or [])
    numeric_passed = sum(1 for row in numeric_rows if row.get("numeric_quality") == "passed")
    numeric_total = len(numeric_rows)

    lines = [
        "# Paper-ready results subsection draft",
        "",
        "## Results",
        "",
        (
            f"The frozen FPGAI paper matrix contains {inference_count} inference design rows and "
            f"{training_count} on-device-training design rows. All reported HLS and Vivado values "
            "in this section are generated from existing synthesis and implementation artifacts; "
            "board-runtime latency, energy, and training-curve measurements are intentionally "
            "excluded until physical KV260 runtime artifacts are imported."
        ),
        "",
        (
            f"HLS metrics are available for {hls_count} design rows and Vivado implementation "
            f"metrics are available for {vivado_count} design rows. Numeric-validation rows are "
            f"available for {numeric_total} design rows, with {numeric_passed} row(s) currently marked "
            "as passed. The current matrix has "
            f"{runtime_count} row(s) with imported runtime measurements, so runtime performance "
            "claims remain pending."
        ),
        "",
        "### Numeric behavior validation",
        "",
    ]
    if numeric_rows:
        inference_numeric = [row for row in numeric_rows if row.get("section") == "inference"]
        training_numeric = [row for row in numeric_rows if row.get("section") == "training"]
        inf_passed = sum(1 for row in inference_numeric if row.get("numeric_quality") == "passed")
        train_passed = sum(1 for row in training_numeric if row.get("numeric_quality") == "passed")
        lines.append(
            f"- Inference numeric validation: {inf_passed}/{len(inference_numeric)} row(s) passed or contain reference-output comparison evidence."
        )
        lines.append(
            f"- Training numeric validation: {train_passed}/{len(training_numeric)} row(s) passed or contain gradient/weight-update comparison evidence."
        )
        for row in training_numeric[:5]:
            if row.get("gradient_cosine") not in (None, "") or row.get("weight_delta_cosine") not in (None, ""):
                lines.append(
                    f"- **{row.get('design_id')}**: gradient cosine={_format_table_value(row.get('gradient_cosine'))}, "
                    f"weight-delta cosine={_format_table_value(row.get('weight_delta_cosine'))}, status=`{row.get('numeric_quality')}`."
                )
        decision_rows = _precision_decision_matrix_rows(numeric_rows)
        if decision_rows:
            lines.extend(["", "### Task-aware decision reporting", ""])
            for row in decision_rows[:5]:
                lines.append(
                    f"- **{row.get('design_id')}**: decision=`{row.get('decision_status')}`, "
                    f"quality metric `{row.get('quality_metric_name')}`={_format_table_value(row.get('quality_metric'))}, "
                    f"LUT saving vs fx16={_format_table_value(row.get('resource_saving_lut_vs_fx16'))}%, "
                    f"BRAM saving vs fx16={_format_table_value(row.get('resource_saving_bram_vs_fx16'))}%."
                )
            lines.append(
                "These decision labels are not a hard pass/fail gate: they summarize the behavioral cost of a YAML precision choice together with HLS/Vivado latency/resource effects. "
                "When labels or regression targets are provided, FPGAI reports task accuracy or target-error deltas; otherwise it reports generated-vs-reference prediction/output agreement."
            )
    else:
        lines.append("- Numeric-validation rows were not available in the current artifact set.")
    lines.extend([
        "",
        "### Inference design effects",
        "",
    ])
    inference_reps = [row for row in representative_rows if str(row.get("baseline_design") or "").startswith("I")]
    if inference_reps:
        for row in inference_reps:
            lines.append(f"- **{row.get('comparison')}** (`{row.get('result_classification')}`): {_comparison_sentence(row)}")
    else:
        lines.append("- No inference comparison rows were available in the current artifact set.")
    lines.extend(["", "### Training design effects", ""])
    training_reps = [row for row in representative_rows if str(row.get("baseline_design") or "").startswith("T")]
    if training_reps:
        for row in training_reps:
            lines.append(f"- **{row.get('comparison')}** (`{row.get('result_classification')}`): {_comparison_sentence(row)}")
    else:
        lines.append("- No training comparison rows were available in the current artifact set.")
    lines.extend(["", "### Claim boundary", ""])
    lines.append(
        "The generated comparisons support HLS/Vivado artifact claims, deployability/package claims, "
        "and explicit no-observable-effect classifications. They do not yet support measured FPGA "
        "latency, measured energy, or measured training convergence claims. Those claims require "
        "board-runtime CSV/report imports and should remain in the pending-measurement tables."
    )
    lines.extend(["", "### Classification summary", ""])
    if classification_rows:
        for row in classification_rows:
            lines.append(
                f"- `{row.get('result_classification')}`: {row.get('comparison_count')} comparison(s) "
                f"covering {row.get('metric_row_count')} metric row(s)."
            )
    else:
        lines.append("- No classified comparison rows were available.")
    return "\n".join(lines) + "\n"


def _write_captions_and_summary(
    out_dir: Path,
    *,
    manifest: Mapping[str, Any],
    plot_rows: Sequence[Mapping[str, Any]],
    table_files: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    numeric_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    created = manifest.get("created_figures") if isinstance(manifest.get("created_figures"), Mapping) else {}
    pending = manifest.get("pending_figures") if isinstance(manifest.get("pending_figures"), Mapping) else {}
    figures = _figure_caption_rows(created, pending)
    tables = _table_caption_rows(table_files)

    figure_md = out_dir / "figure_captions.md"
    table_md = out_dir / "table_captions.md"
    claims_md = out_dir / "paper_claims_from_artifacts.md"
    summary_md = out_dir / "paper_results_summary.md"
    gallery_md = out_dir / "plot_gallery.md"
    gallery_html = out_dir / "plot_gallery.html"
    subsection_md = out_dir / "paper_results_subsection.md"

    _write_table_md(
        figure_md,
        title="Figure captions",
        rows=figures,
        fields=["figure", "status", "path", "caption", "pending_reason"],
    )
    _write_table_md(
        table_md,
        title="Table captions",
        rows=tables,
        fields=["table", "path", "row_count", "caption"],
    )

    claim_lines = [
        "# Paper claims from artifacts",
        "",
        "These claims are generated only from existing FPGAI reports and runtime artifacts.",
        "",
        "## Supported current claims",
    ]
    hls_count = sum(1 for row in plot_rows if _float_or_none(row.get("hls_lut")) is not None or _float_or_none(row.get("hls_latency_max")) is not None)
    vivado_count = sum(1 for row in plot_rows if _float_or_none(row.get("vivado_lut")) is not None or _float_or_none(row.get("vivado_power_w")) is not None)
    bitstream_count = sum(1 for row in plot_rows if str(row.get("support_status") or "").lower().find("bitstream") >= 0 or str(row.get("design_id") or "").find("bitstream") >= 0)
    classification_rows = _classification_summary_rows(comparison_rows)
    representative_rows = _representative_comparison_rows(comparison_rows)
    numeric_rows = list(numeric_rows or [])
    numeric_passed = sum(1 for row in numeric_rows if row.get("numeric_quality") == "passed")
    claim_lines.append(f"- HLS synthesis metrics are available for `{hls_count}` frozen paper row(s).")
    claim_lines.append(f"- Vivado implementation metrics are available for `{vivado_count}` frozen paper row(s).")
    claim_lines.append(f"- Numeric-validation rows are available for `{len(numeric_rows)}` frozen paper row(s), with `{numeric_passed}` row(s) marked passed.")
    claim_lines.append(f"- Deployable bitstream candidate rows are present for at least `{bitstream_count}` design(s), subject to runtime-package validation tables.")
    claim_lines.append("- Real board latency, energy, and FPGA training curves remain pending until board-runtime CSV/report artifacts are imported.")
    claim_lines.extend(["", "## Result classification boundary"])
    if classification_rows:
        for row in classification_rows:
            claim_lines.append(
                f"- `{row.get('result_classification')}`: `{row.get('comparison_count')}` comparison(s), `{row.get('metric_row_count')}` metric row(s)."
            )
    else:
        claim_lines.append("- No classified comparison rows were available from the current frozen subset.")
    claim_lines.extend(["", "## Representative computed result statements"])
    if representative_rows:
        for row in representative_rows:
            classification = str(row.get("result_classification") or "")
            claim_lines.append(f"- `{classification}` — {_comparison_sentence(row)}")
    else:
        claim_lines.append("- No pairwise comparisons were available from the current frozen subset.")
    claim_lines.extend(["", "## Full metric-level comparisons"])
    if comparison_rows:
        for row in comparison_rows:
            claim_lines.append(
                f"- `{row.get('result_classification')}` `{row.get('comparison')}` — {_comparison_sentence(row)}"
            )
    else:
        claim_lines.append("- No pairwise comparisons were available from the current frozen subset.")
    claims_md.write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    lines = [
        "# FPGAI paper results summary",
        "",
        f"- compile_output_count: `{manifest.get('compile_output_count')}`",
        f"- created_figures: `{len(created)}`",
        f"- pending_figures: `{len(pending)}`",
        "",
        "## Current paper story",
        "",
        "The current frozen subset supports an inference-first and training-second results section. HLS/Vivado figures are generated from existing synthesis/implementation artifacts. Real latency, energy, and FPGA training-curve figures are intentionally pending until board-runtime measurements are imported.",
        "",
        "## Numeric validation summary",
        "",
        f"- numeric_validation_rows: `{len(numeric_rows)}`",
        f"- numeric_validation_passed_rows: `{numeric_passed}`",
        "- Use `tables/table_10_numeric_validation_summary.md`, `table_11_inference_precision_numeric_tradeoff.md`, and `table_12_training_numeric_validation.md` for numeric behavior claims.",
        "- Use `tables/table_14_inference_task_quality_tradeoff.md`, `table_15_precision_decision_matrix.md`, `table_16_training_task_quality_tradeoff.md`, and `table_17_training_precision_decision_matrix.md` for user-facing decision reports.",
        "",
        "## Key computed comparisons",
    ]
    if representative_rows:
        for row in representative_rows:
            classification = str(row.get("result_classification") or "")
            lines.append(f"- `{classification}` — {_comparison_sentence(row)}")
    else:
        lines.append("- No pairwise comparison rows available yet.")
    lines.extend(["", "## Result classification summary"])
    if classification_rows:
        for row in classification_rows:
            lines.append(
                f"- `{row.get('result_classification')}`: `{row.get('comparison_count')}` comparison(s), `{row.get('metric_row_count')}` metric row(s)."
            )
    else:
        lines.append("- No classified comparison rows available yet.")
    lines.extend([
        "",
        "## Use in paper",
        "",
        "- Use `figure_captions.md` and `table_captions.md` for caption drafting.",
        "- Use `paper_claims_from_artifacts.md` to avoid over-claiming runtime behavior.",
        "- Open `plot_gallery.html` locally to inspect all generated SVG plots in one page.",
    ])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    subsection_md.write_text(_paper_results_subsection_text(plot_rows, representative_rows, classification_rows, numeric_rows), encoding="utf-8")

    gal_lines = ["# FPGAI plot gallery", ""]
    for row in figures:
        if row.get("status") != "created":
            continue
        path = str(row.get("path") or "")
        if not path:
            continue
        rel = Path(path).relative_to(out_dir) if Path(path).is_absolute() is False and path.startswith(out_dir.as_posix()) else Path(path)
        if str(rel).startswith(out_dir.as_posix()):
            rel = Path(str(rel)[len(out_dir.as_posix()):].lstrip("/"))
        gal_lines.extend([f"## {row.get('figure')}", "", str(row.get("caption") or ""), "", f"![{row.get('figure')}]({rel.as_posix()})", ""])
    gallery_md.write_text("\n".join(gal_lines), encoding="utf-8")

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>FPGAI plot gallery</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;background:#fafafa;color:#111} .fig{background:white;border:1px solid #ddd;border-radius:8px;padding:18px;margin:0 0 28px 0} img{max-width:100%;height:auto} h1{margin-bottom:4px} p{color:#444}</style>",
        "</head><body><h1>FPGAI plot gallery</h1><p>Generated from existing compile/runtime artifacts only.</p>",
    ]
    for row in figures:
        if row.get("status") != "created":
            continue
        path = str(row.get("path") or "")
        if not path:
            continue
        rel = Path(path)
        if path.startswith(out_dir.as_posix()):
            rel = Path(path[len(out_dir.as_posix()):].lstrip("/"))
        html_lines.append(f"<div class='fig'><h2>{_xml(row.get('figure'))}</h2><p>{_xml(row.get('caption'))}</p><img src='{_xml(rel.as_posix())}'></div>")
    html_lines.append("</body></html>")
    gallery_html.write_text("\n".join(html_lines) + "\n", encoding="utf-8")

    return {
        "paper_results_summary_md": summary_md.as_posix(),
        "figure_captions_md": figure_md.as_posix(),
        "table_captions_md": table_md.as_posix(),
        "paper_claims_from_artifacts_md": claims_md.as_posix(),
        "plot_gallery_md": gallery_md.as_posix(),
        "plot_gallery_html": gallery_html.as_posix(),
        "paper_results_subsection_md": subsection_md.as_posix(),
    }

def _write_md(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# FPGAI paper plot manifest",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- compile_output_count: `{manifest.get('compile_output_count')}`",
        "",
        "## Sections",
    ]
    sections = manifest.get("sections") if isinstance(manifest.get("sections"), Mapping) else {}
    for name, section in sections.items():
        lines.append(f"### {name}")
        if isinstance(section, Mapping):
            for key, value in section.items():
                lines.append(f"- {key}: `{_csv_value(value)}`")
        lines.append("")
    lines.extend(["## Created figures"])
    created = manifest.get("created_figures") if isinstance(manifest.get("created_figures"), Mapping) else {}
    if created:
        for name, value in created.items():
            lines.append(f"- `{name}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Pending figures"])
    pending = manifest.get("pending_figures") if isinstance(manifest.get("pending_figures"), Mapping) else {}
    if pending:
        for name, value in pending.items():
            lines.append(f"- `{name}`: {value}")
    else:
        lines.append("- none")
    lines.extend(["", "## Paper tables"])
    table_files = manifest.get("table_files") if isinstance(manifest.get("table_files"), Mapping) else {}
    if table_files:
        for name, value in table_files.items():
            if isinstance(value, Mapping):
                lines.append(f"- `{name}`: `{value.get('md')}` rows=`{value.get('row_count')}`")
            else:
                lines.append(f"- `{name}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Narrative files"])
    narrative_files = manifest.get("narrative_files") if isinstance(manifest.get("narrative_files"), Mapping) else {}
    if narrative_files:
        for name, value in narrative_files.items():
            lines.append(f"- `{name}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", str(manifest.get("claim_boundary") or "")])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate FPGAI paper plot data and SVG figures from compile outputs.")
    parser.add_argument("out_dirs", nargs="*", help="Compile output directories or parent directory to scan.")
    parser.add_argument("--output-dir", default="paper_results/plots")
    args = parser.parse_args(argv)
    if not args.out_dirs:
        parser.error("provide at least one compile output directory or parent directory")
    manifest = generate_paper_plot_artifacts(args.out_dirs, output_dir=args.output_dir)
    print(json.dumps({"status": manifest["status"], "created_figures": len(manifest["created_figures"]), "pending_figures": len(manifest["pending_figures"])}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
