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
    for path, row in zip(compile_dirs, plot_rows):
        knob_rows.extend(_knob_rows(path, row))
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
    table_files = _write_paper_tables(table_dir, plot_rows=plot_rows, knob_rows=knob_rows, curve_rows=curve_rows)
    comparison_rows = _comparison_rows(plot_rows)
    table_files["table_08_result_comparisons"] = _write_table_pair(
        table_dir,
        stem="table_08_result_comparisons",
        title="Computed result comparisons",
        rows=comparison_rows,
        fields=["comparison", "group", "baseline_design", "variant_design", "metric_label", "baseline_value", "variant_value", "delta", "percent_change"],
        labels={"percent_change": "Change %", "baseline_value": "Baseline", "variant_value": "Variant"},
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
                "required_real_measurements": ["runtime_latency_ms_mean", "runtime_throughput", "runtime_accuracy", "vivado_power_w", "runtime_energy_mj_estimated"],
            },
            "training": {
                "design_count": len(training_rows),
                "real_runtime_rows": len(runtime_training),
                "real_training_curve_rows": len(curve_rows),
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


def _comparison_rows(plot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return paper-ready pairwise comparisons for the frozen first subset."""
    by_design = _row_by_design(plot_rows)
    comparisons = [
        ("precision_fx8_vs_fx16", "I0_baseline_fx16_embedded", "I1_precision_fx8_embedded", "Precision", "fx8_3 versus fx16_6 baseline"),
        ("parallel_pe2_vs_pe1", "I0_baseline_fx16_embedded", "I3_parallel_pe2", "Parallelism", "PE=2 versus baseline PE=1"),
        ("deployable_inference_vs_baseline", "I0_baseline_fx16_embedded", "I8_deployable_bitstream_candidate", "Deployability", "Inference bitstream candidate versus baseline"),
        ("training_tile32_vs_sgd", "T0_sgd_tiled_m_axi", "T4_tile32_m_axi", "Training memory", "Tile-size design versus SGD tiled baseline"),
        ("training_bitstream_vs_sgd", "T0_sgd_tiled_m_axi", "T7_deployable_training_bitstream", "Training deployability", "Training bitstream candidate versus SGD tiled baseline"),
    ]
    metrics = [
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
    rows: list[dict[str, Any]] = []
    for comparison_id, base_id, variant_id, group, description in comparisons:
        base = by_design.get(base_id)
        variant = by_design.get(variant_id)
        if base is None or variant is None:
            continue
        for metric, label in metrics:
            base_value = base.get(metric)
            variant_value = variant.get(metric)
            if _float_or_none(base_value) is None or _float_or_none(variant_value) is None:
                continue
            rows.append({
                "comparison": comparison_id,
                "group": group,
                "description": description,
                "baseline_design": base_id,
                "variant_design": variant_id,
                "metric": metric,
                "metric_label": label,
                "baseline_value": base_value,
                "variant_value": variant_value,
                "delta": _metric_delta(variant_value, base_value),
                "percent_change": _pct_change(variant_value, base_value),
            })
    return rows


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
        "table_08_result_comparisons": "Pairwise paper comparisons computed from the frozen experiment subset.",
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


def _write_captions_and_summary(
    out_dir: Path,
    *,
    manifest: Mapping[str, Any],
    plot_rows: Sequence[Mapping[str, Any]],
    table_files: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
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
    claim_lines.append(f"- HLS synthesis metrics are available for `{hls_count}` frozen paper row(s).")
    claim_lines.append(f"- Vivado implementation metrics are available for `{vivado_count}` frozen paper row(s).")
    claim_lines.append(f"- Deployable bitstream candidate rows are present for at least `{bitstream_count}` design(s), subject to runtime-package validation tables.")
    claim_lines.append("- Real board latency, energy, and FPGA training curves remain pending until board-runtime CSV/report artifacts are imported.")
    claim_lines.extend(["", "## Computed pairwise result statements"])
    if comparison_rows:
        for row in comparison_rows:
            claim_lines.append(f"- {_comparison_sentence(row)}")
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
        "## Key computed comparisons",
    ]
    if comparison_rows:
        for row in comparison_rows[:16]:
            lines.append(f"- {_comparison_sentence(row)}")
    else:
        lines.append("- No pairwise comparison rows available yet.")
    lines.extend([
        "",
        "## Use in paper",
        "",
        "- Use `figure_captions.md` and `table_captions.md` for caption drafting.",
        "- Use `paper_claims_from_artifacts.md` to avoid over-claiming runtime behavior.",
        "- Open `plot_gallery.html` locally to inspect all generated SVG plots in one page.",
    ])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
