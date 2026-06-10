from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


RESOURCE_FIELDS = {
    "lut": ("predicted_lut", "actual_lut"),
    "ff": ("predicted_ff", "actual_ff"),
    "dsp": ("predicted_dsp", "actual_dsp"),
    "bram18": ("predicted_bram18", "actual_bram18"),
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def _find_best_report_path(report_path: str | Path) -> Path:
    requested = Path(report_path)
    parent = requested if requested.is_dir() else requested.parent

    preferred_names = [
        "deeplearn_csynth.xml",
        "csynth.xml",
        "deeplearn_csynth.rpt",
        "csynth.rpt",
    ]

    for name in preferred_names:
        candidate = parent / name

        if candidate.is_file():
            return candidate

    xml_candidates = sorted(parent.glob("*_csynth.xml"))

    if xml_candidates:
        return xml_candidates[0]

    report_candidates = sorted(parent.glob("*_csynth.rpt"))

    if report_candidates:
        return report_candidates[0]

    return requested


def _empty_actual() -> Dict[str, float]:
    return {
        "actual_lut": 0,
        "actual_ff": 0,
        "actual_dsp": 0,
        "actual_bram18": 0,
        "actual_latency_cycles": 0.0,
    }


def _extract_xml_tag(
    text: str,
    tag: str,
    default: float | int = 0,
) -> float | int:
    match = re.search(
        rf"<{re.escape(tag)}>\s*([^<]+?)\s*</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return default

    if isinstance(default, float):
        return _safe_float(match.group(1), default)

    return _safe_int(match.group(1), int(default))


def _extract_latency_from_xml(text: str) -> float:
    tags = [
        "Average-caseLatency",
        "Worst-caseLatency",
        "Best-caseLatency",
        "Latency",
    ]

    for tag in tags:
        value = float(_extract_xml_tag(text, tag, 0.0))

        if value > 0.0:
            return value

    return 0.0


def _extract_from_xml(xml_path: Path) -> Dict[str, float]:
    text = xml_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return {
        "actual_lut": _safe_int(
            _extract_xml_tag(text, "LUT", 0)
        ),
        "actual_ff": _safe_int(
            _extract_xml_tag(text, "FF", 0)
        ),
        "actual_dsp": _safe_int(
            _extract_xml_tag(text, "DSP", 0)
        ),
        "actual_bram18": _safe_int(
            _extract_xml_tag(text, "BRAM_18K", 0)
        ),
        "actual_latency_cycles": _extract_latency_from_xml(
            text
        ),
    }


def _parse_total_resource_row(text: str) -> Dict[str, float]:
    result = _empty_actual()

    patterns = [
        (
            r"\|\s*Total\s*\|"
            r"\s*([0-9,]+)\s*\|"
            r"\s*([0-9,]+)\s*\|"
            r"\s*([0-9,]+)\s*\|"
            r"\s*([0-9,]+)\s*\|"
        ),
        (
            r"\|\s*Utilization Estimates\s*\|"
            r".*?"
            r"\|\s*Total\s*\|"
            r"\s*([0-9,]+)\s*\|"
            r"\s*([0-9,]+)\s*\|"
            r"\s*([0-9,]+)\s*\|"
            r"\s*([0-9,]+)\s*\|"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match is None:
            continue

        result["actual_bram18"] = _safe_int(match.group(1))
        result["actual_dsp"] = _safe_int(match.group(2))
        result["actual_ff"] = _safe_int(match.group(3))
        result["actual_lut"] = _safe_int(match.group(4))
        break

    return result


def _parse_latency_from_report(text: str) -> float:
    patterns = [
        (
            r"\|\s*Latency\s*\([^)]*\)"
            r".*?\|\s*([0-9,]+)\s*\|"
        ),
        r"Average-caseLatency[^0-9]*([0-9,]+)",
        r"Latency.*?min\s*=\s*([0-9,]+)",
        r"\|\s*Latency\s*\|\s*([0-9,]+)\s*\|",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match is not None:
            value = _safe_float(match.group(1))

            if value > 0.0:
                return value

    return 0.0


def _count_dsp_bindings(text: str) -> int:
    names = set()

    for match in re.finditer(
        (
            r"^\|\s*([A-Za-z0-9_.$]+)\s*\|"
            r".*?\|\s*dsp_slice\s*\|"
        ),
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        names.add(match.group(1).strip())

    return len(names)


def _extract_from_report(report_path: Path) -> Dict[str, float]:
    text = report_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    result = _parse_total_resource_row(text)
    result["actual_latency_cycles"] = (
        _parse_latency_from_report(text)
    )

    if result["actual_dsp"] == 0:
        result["actual_dsp"] = _count_dsp_bindings(text)

    return result


def parse_hls_csynth_report(
    report_path: str | Path,
) -> Dict[str, float]:
    selected = _find_best_report_path(report_path)

    if not selected.is_file():
        return _empty_actual()

    if selected.suffix.lower() == ".xml":
        return _extract_from_xml(selected)

    return _extract_from_report(selected)


def _relative_error(
    predicted: float,
    actual: float,
) -> float | None:
    if actual <= 0.0:
        return None

    return (predicted - actual) / actual


def _absolute_percentage_error(
    predicted: float,
    actual: float,
) -> float | None:
    relative_error = _relative_error(predicted, actual)

    if relative_error is None:
        return None

    return abs(relative_error) * 100.0


def _actual_to_predicted_ratio(
    predicted: float,
    actual: float,
) -> float | None:
    if predicted <= 0.0 or actual <= 0.0:
        return None

    return actual / predicted


def _quality_label(
    percentage_error: float | None,
) -> str:
    if percentage_error is None:
        return "unavailable"

    if percentage_error <= 10.0:
        return "excellent"

    if percentage_error <= 25.0:
        return "good"

    if percentage_error <= 50.0:
        return "rough"

    return "poor"


def _error_direction(
    predicted: float,
    actual: float,
) -> str:
    if actual <= 0.0:
        return "unavailable"

    if math.isclose(predicted, actual):
        return "matched"

    return "overestimated" if predicted > actual else "underestimated"


def _comparison_row(
    *,
    predicted: float,
    actual: float,
) -> Dict[str, Any]:
    percentage_error = _absolute_percentage_error(
        predicted,
        actual,
    )

    return {
        "predicted": predicted,
        "actual": actual,
        "signed_error": predicted - actual,
        "signed_relative_error": _relative_error(
            predicted,
            actual,
        ),
        "absolute_percentage_error": percentage_error,
        "actual_to_predicted_ratio": (
            _actual_to_predicted_ratio(
                predicted,
                actual,
            )
        ),
        "direction": _error_direction(
            predicted,
            actual,
        ),
        "quality": _quality_label(percentage_error),
    }


def _diagnostic_ratio_payload(
    design_space_summary: Mapping[str, Any],
    actual: Mapping[str, float],
    predicted_cycles: float,
    actual_cycles: float,
) -> Dict[str, Any]:
    resources: Dict[str, float | None] = {}

    for resource_name, (
        predicted_key,
        actual_key,
    ) in RESOURCE_FIELDS.items():
        resources[f"{resource_name}_scale"] = (
            _actual_to_predicted_ratio(
                float(
                    design_space_summary.get(
                        predicted_key,
                        0,
                    )
                    or 0
                ),
                float(actual.get(actual_key, 0) or 0),
            )
        )

    performance = {
        "cycle_scale": _actual_to_predicted_ratio(
            predicted_cycles,
            actual_cycles,
        )
    }

    return {
        "resources": resources,
        "performance": performance,
        "yaml": {
            "analysis": {
                "design_space": {
                    "calibration": {
                        "resources": resources,
                        "performance": performance,
                    }
                }
            }
        },
        "deprecated": True,
        "note": (
            "These ratios are retained for backward compatibility. "
            "They are diagnostic residual ratios, not a replacement "
            "for correcting operator, memory, interface, or schedule models."
        ),
    }


def _model_diagnostics(
    comparison: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    poor_fields = []
    underestimated_fields = []
    overestimated_fields = []

    for name, row in comparison.items():
        if row.get("quality") == "poor":
            poor_fields.append(name)

        if row.get("direction") == "underestimated":
            underestimated_fields.append(name)
        elif row.get("direction") == "overestimated":
            overestimated_fields.append(name)

    likely_causes = []

    if "dsp" in underestimated_fields:
        likely_causes.append(
            "DSP underestimation: inspect generated multiplier, "
            "division, exponential, and type-conversion operators."
        )

    if "bram18" in underestimated_fields:
        likely_causes.append(
            "BRAM underestimation: inspect array partitioning, banking, "
            "line buffers, parameter storage, and duplicated memories."
        )

    if "lut" in underestimated_fields or "ff" in underestimated_fields:
        likely_causes.append(
            "Logic underestimation: inspect top-level interfaces, "
            "conversion loops, control FSMs, and operator implementation costs."
        )

    if "latency_cycles" in underestimated_fields:
        likely_causes.append(
            "Latency underestimation: inspect loop initiation intervals, "
            "pipeline stalls, function boundaries, and memory dependencies."
        )

    if "latency_cycles" in overestimated_fields:
        likely_causes.append(
            "Latency overestimation: inspect overlapping loops, dataflow, "
            "pipeline concurrency, and actual achieved initiation intervals."
        )

    return {
        "poor_fields": poor_fields,
        "underestimated_fields": underestimated_fields,
        "overestimated_fields": overestimated_fields,
        "likely_causes": likely_causes,
        "requires_model_revision": bool(poor_fields),
    }


def _format_number(
    value: Any,
    *,
    digits: int = 2,
) -> str:
    if value is None:
        return "n/a"

    return f"{_safe_float(value):.{digits}f}"


def _write_unavailable_result(
    *,
    comparison_dir: Path,
    results_json: Path,
    summary_txt: Path,
    reason: str,
    terminal_message: str,
    design_space_summary: Mapping[str, Any],
    requested_path: str | Path | None = None,
    selected_path: Path | None = None,
) -> "EstimateVsHlsResult":
    terminal_summary = "\n".join(
        [
            "=============== FPGAI Estimate vs HLS ===============",
            terminal_message,
            "=====================================================",
        ]
    )

    payload: Dict[str, Any] = {
        "available": False,
        "reason": reason,
        "estimated": dict(design_space_summary),
        "actual": None,
        "comparison": None,
        "model_diagnostics": None,
        "calibration_recommendation": None,
    }

    if requested_path is not None:
        payload["csynth_report_path_requested"] = str(
            requested_path
        )

    if selected_path is not None:
        payload["csynth_report_path_used"] = str(
            selected_path
        )

    results_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_txt.write_text(
        terminal_summary + "\n",
        encoding="utf-8",
    )

    return EstimateVsHlsResult(
        out_dir=comparison_dir,
        results_json=results_json,
        summary_txt=summary_txt,
        terminal_summary=terminal_summary,
    )


@dataclass(frozen=True)
class EstimateVsHlsResult:
    out_dir: Path
    results_json: Path
    summary_txt: Path
    terminal_summary: str


def run_estimate_vs_hls_compare(
    *,
    out_dir: str | Path,
    design_space_summary: Dict[str, float],
    csynth_report_path: str | Path | None,
    clock_mhz: float,
) -> EstimateVsHlsResult:
    output_root = Path(out_dir).resolve()
    comparison_dir = output_root / "estimate_vs_hls"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    results_json = comparison_dir / "results.json"
    summary_txt = comparison_dir / "summary.txt"

    if not csynth_report_path:
        return _write_unavailable_result(
            comparison_dir=comparison_dir,
            results_json=results_json,
            summary_txt=summary_txt,
            reason="No HLS synthesis report path was provided",
            terminal_message="HLS synthesis report unavailable.",
            design_space_summary=design_space_summary,
        )

    selected_report = _find_best_report_path(
        csynth_report_path
    )

    if not selected_report.is_file():
        return _write_unavailable_result(
            comparison_dir=comparison_dir,
            results_json=results_json,
            summary_txt=summary_txt,
            reason="HLS synthesis report file does not exist",
            terminal_message=(
                "HLS synthesis report was not found: "
                f"{selected_report}"
            ),
            design_space_summary=design_space_summary,
            requested_path=csynth_report_path,
            selected_path=selected_report,
        )

    actual = parse_hls_csynth_report(selected_report)
    effective_clock_mhz = max(
        1e-9,
        float(clock_mhz),
    )

    actual_latency_cycles = float(
        actual["actual_latency_cycles"]
    )
    actual_latency_ms = (
        actual_latency_cycles
        / (effective_clock_mhz * 1_000.0)
        if actual_latency_cycles > 0.0
        else 0.0
    )

    predicted_latency_ms = float(
        design_space_summary.get(
            "predicted_latency_ms",
            0.0,
        )
        or 0.0
    )
    predicted_cycles = float(
        design_space_summary.get(
            "predicted_cycles",
            (
                predicted_latency_ms
                * effective_clock_mhz
                * 1_000.0
            ),
        )
        or 0.0
    )

    comparison = {
        "lut": _comparison_row(
            predicted=float(
                design_space_summary.get(
                    "predicted_lut",
                    0,
                )
                or 0
            ),
            actual=float(actual["actual_lut"]),
        ),
        "ff": _comparison_row(
            predicted=float(
                design_space_summary.get(
                    "predicted_ff",
                    0,
                )
                or 0
            ),
            actual=float(actual["actual_ff"]),
        ),
        "dsp": _comparison_row(
            predicted=float(
                design_space_summary.get(
                    "predicted_dsp",
                    0,
                )
                or 0
            ),
            actual=float(actual["actual_dsp"]),
        ),
        "bram18": _comparison_row(
            predicted=float(
                design_space_summary.get(
                    "predicted_bram18",
                    0,
                )
                or 0
            ),
            actual=float(actual["actual_bram18"]),
        ),
        "latency_cycles": _comparison_row(
            predicted=predicted_cycles,
            actual=actual_latency_cycles,
        ),
        "latency_ms": _comparison_row(
            predicted=predicted_latency_ms,
            actual=actual_latency_ms,
        ),
    }

    diagnostics = _model_diagnostics(comparison)

    # Compatibility field for existing readers and tests. These ratios
    # are no longer presented as the primary estimator solution.
    legacy_ratios = _diagnostic_ratio_payload(
        design_space_summary,
        actual,
        predicted_cycles,
        actual_latency_cycles,
    )

    payload = {
        "available": True,
        "comparison_model": "analytical_validation_v2",
        "csynth_report_path_requested": str(
            csynth_report_path
        ),
        "csynth_report_path_used": str(
            selected_report
        ),
        "clock_mhz": effective_clock_mhz,
        "estimated": design_space_summary,
        "actual": {
            **actual,
            "actual_latency_ms": actual_latency_ms,
        },
        "comparison": comparison,
        "relative_error": {
            name: row["signed_relative_error"]
            for name, row in comparison.items()
        },
        "absolute_percentage_error": {
            name: row["absolute_percentage_error"]
            for name, row in comparison.items()
        },
        "model_diagnostics": diagnostics,
        "observed_residual_ratios": {
            "resources": legacy_ratios["resources"],
            "performance": legacy_ratios["performance"],
        },
        "calibration_recommendation": legacy_ratios,
    }

    terminal_lines = [
        "=============== FPGAI Estimate vs HLS ===============",
        f"Report   used={selected_report.name}",
    ]

    for label, key in [
        ("LUT", "lut"),
        ("FF", "ff"),
        ("DSP", "dsp"),
        ("BRAM18", "bram18"),
    ]:
        row = comparison[key]

        terminal_lines.append(
            f"{label:<8} "
            f"pred={int(row['predicted'])}  "
            f"actual={int(row['actual'])}  "
            f"error={_format_number(row['absolute_percentage_error'])}%  "
            f"direction={row['direction']}  "
            f"quality={row['quality']}"
        )

    latency_row = comparison["latency_ms"]

    terminal_lines.append(
        "Latency  "
        f"pred={predicted_latency_ms:.4f} ms  "
        f"actual={actual_latency_ms:.4f} ms  "
        f"error={_format_number(latency_row['absolute_percentage_error'])}%  "
        f"direction={latency_row['direction']}  "
        f"quality={latency_row['quality']}"
    )

    terminal_lines.append(
        "-----------------------------------------------------"
    )

    if diagnostics["requires_model_revision"]:
        terminal_lines.append(
            "Analytical model revision required:"
        )

        for cause in diagnostics["likely_causes"]:
            terminal_lines.append(f" - {cause}")
    else:
        terminal_lines.append(
            "Analytical estimates are within the current quality limits."
        )

    resources = legacy_ratios["resources"]
    performance = legacy_ratios["performance"]

    terminal_lines.extend(
        [
            "-----------------------------------------------------",
            (
                "Suggested calibration scales "
                "(legacy diagnostic ratios only):"
            ),
            (
                " LUT    : "
                f"{_format_number(resources['lut_scale'], digits=4)}"
            ),
            (
                " FF     : "
                f"{_format_number(resources['ff_scale'], digits=4)}"
            ),
            (
                " DSP    : "
                f"{_format_number(resources['dsp_scale'], digits=4)}"
            ),
            (
                " BRAM18 : "
                f"{_format_number(resources['bram18_scale'], digits=4)}"
            ),
            (
                " Cycles : "
                f"{_format_number(performance['cycle_scale'], digits=4)}"
            ),
            (
                "Ratios are diagnostic. Correct the responsible "
                "operator or schedule model instead of applying them globally."
            ),
            "=====================================================",
        ]
    )

    terminal_summary = "\n".join(terminal_lines)

    results_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_txt.write_text(
        terminal_summary + "\n",
        encoding="utf-8",
    )

    return EstimateVsHlsResult(
        out_dir=comparison_dir,
        results_json=results_json,
        summary_txt=summary_txt,
        terminal_summary=terminal_summary,
    )