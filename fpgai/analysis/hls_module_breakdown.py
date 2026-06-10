from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


RESOURCE_TAGS = {
    "lut": "LUT",
    "ff": "FF",
    "dsp": "DSP",
    "bram18": "BRAM_18K",
}


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        text = str(value).replace(",", "").strip()
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        text = str(value).replace(",", "").strip()
        result = float(text)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def _extract_tag(
    text: str,
    tag: str,
) -> str | None:
    match = re.search(
        rf"<{re.escape(tag)}>\s*([^<]+?)\s*</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(1).strip()


def _extract_first_tag(
    text: str,
    tags: List[str],
) -> str | None:
    for tag in tags:
        value = _extract_tag(text, tag)

        if value:
            return value

    return None


def _module_name(
    text: str,
    report_path: Path,
) -> str:
    value = _extract_first_tag(
        text,
        [
            "TopModelName",
            "TopFunctionName",
            "FunctionName",
            "ModuleName",
        ],
    )

    if value:
        return value

    name = report_path.stem

    if name.endswith("_csynth"):
        name = name[:-7]

    return name


def _latency_cycles(text: str) -> float:
    for tag in [
        "Average-caseLatency",
        "Worst-caseLatency",
        "Best-caseLatency",
        "Latency",
    ]:
        value = _safe_float(
            _extract_tag(text, tag),
            0.0,
        )

        if value > 0.0:
            return value

    return 0.0


def _initiation_interval(text: str) -> float:
    for tag in [
        "Interval-min",
        "IntervalMax",
        "Interval-max",
        "PipelineII",
        "InitiationInterval",
    ]:
        value = _safe_float(
            _extract_tag(text, tag),
            0.0,
        )

        if value > 0.0:
            return value

    return 0.0


def _parse_xml_report(
    report_path: Path,
) -> Dict[str, Any]:
    text = report_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    resources = {
        name: _safe_int(
            _extract_tag(text, tag),
            0,
        )
        for name, tag in RESOURCE_TAGS.items()
    }

    module = _module_name(
        text,
        report_path,
    )

    return {
        "module": module,
        "report_path": str(report_path),
        "report_name": report_path.name,
        "lut": resources["lut"],
        "ff": resources["ff"],
        "dsp": resources["dsp"],
        "bram18": resources["bram18"],
        "latency_cycles": _latency_cycles(text),
        "initiation_interval": _initiation_interval(text),
    }


def _classify_module(
    module_name: str,
) -> str:
    name = module_name.lower()

    patterns = [
        ("Softmax", ["softmax"]),
        ("Sigmoid", ["sigmoid"]),
        ("LeakyRelu", ["leaky_relu", "leakyrelu"]),
        ("Relu", ["relu"]),
        ("Conv", ["conv2d", "conv"]),
        ("Dense", ["dense", "gemm", "matmul"]),
        ("MaxPool", ["maxpool", "max_pool"]),
        ("AvgPool", ["avgpool", "averagepool"]),
        (
            "BatchNormalization",
            ["batchnorm", "batch_normalization"],
        ),
        ("Add", ["add_vec", "elementwise_add"]),
        ("Reshape", ["reshape", "flatten"]),
        (
            "Interface",
            [
                "axis",
                "stream",
                "value_to_bits",
                "bits_to_value",
            ],
        ),
    ]

    for op_type, fragments in patterns:
        if any(fragment in name for fragment in fragments):
            return op_type

    return "Unknown"


def _is_top_module(
    module_name: str,
    requested_top_name: str | None,
) -> bool:
    name = module_name.strip().lower()

    if requested_top_name:
        return name == requested_top_name.strip().lower()

    return name in {
        "deeplearn",
        "fpgai",
        "top",
    }


def _deduplicate_reports(
    reports: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}

    for report in reports:
        module = str(report["module"])
        current = selected.get(module)

        if current is None:
            selected[module] = report
            continue

        current_score = (
            int(current["lut"])
            + int(current["ff"])
            + 10 * int(current["dsp"])
            + 10 * int(current["bram18"])
        )
        new_score = (
            int(report["lut"])
            + int(report["ff"])
            + 10 * int(report["dsp"])
            + 10 * int(report["bram18"])
        )

        if new_score > current_score:
            selected[module] = report

    return list(selected.values())


def collect_hls_module_reports(
    report_path: str | Path,
    *,
    top_name: str | None = None,
) -> Dict[str, Any]:
    requested = Path(report_path)

    if requested.is_dir():
        search_root = requested
    else:
        search_root = requested.parent

    xml_paths = sorted(
        search_root.rglob("*_csynth.xml")
    )

    if (
        requested.is_file()
        and requested.suffix.lower() == ".xml"
        and requested not in xml_paths
    ):
        xml_paths.append(requested)

    parsed = [
        _parse_xml_report(path)
        for path in xml_paths
        if path.is_file()
    ]
    reports = _deduplicate_reports(parsed)

    for report in reports:
        report["op_type"] = _classify_module(
            str(report["module"])
        )
        report["is_top"] = _is_top_module(
            str(report["module"]),
            top_name,
        )

    reports.sort(
        key=lambda row: (
            bool(row["is_top"]),
            int(row["dsp"]),
            int(row["lut"]),
            int(row["ff"]),
        ),
        reverse=True,
    )

    top_reports = [
        report
        for report in reports
        if report["is_top"]
    ]
    child_reports = [
        report
        for report in reports
        if not report["is_top"]
    ]

    top_report = (
        top_reports[0]
        if top_reports
        else (
            reports[0]
            if len(reports) == 1
            else None
        )
    )

    child_sum = {
        "lut": sum(
            int(report["lut"])
            for report in child_reports
        ),
        "ff": sum(
            int(report["ff"])
            for report in child_reports
        ),
        "dsp": sum(
            int(report["dsp"])
            for report in child_reports
        ),
        "bram18": sum(
            int(report["bram18"])
            for report in child_reports
        ),
    }

    by_operator: Dict[str, Dict[str, int]] = {}

    for report in child_reports:
        op_type = str(report["op_type"])

        if op_type not in by_operator:
            by_operator[op_type] = {
                "module_count": 0,
                "lut": 0,
                "ff": 0,
                "dsp": 0,
                "bram18": 0,
            }

        aggregate = by_operator[op_type]
        aggregate["module_count"] += 1
        aggregate["lut"] += int(report["lut"])
        aggregate["ff"] += int(report["ff"])
        aggregate["dsp"] += int(report["dsp"])
        aggregate["bram18"] += int(report["bram18"])

    return {
        "available": bool(reports),
        "search_root": str(search_root),
        "top_name": top_name,
        "top_report": top_report,
        "module_count": len(reports),
        "child_module_count": len(child_reports),
        "child_sum": child_sum,
        "by_operator": by_operator,
        "modules": reports,
    }


def _terminal_summary(
    payload: Dict[str, Any],
) -> str:
    lines = [
        "=============== FPGAI HLS Module Breakdown ==============="
    ]

    if not payload["available"]:
        lines.extend(
            [
                "No per-function *_csynth.xml reports were found.",
                "===========================================================",
            ]
        )
        return "\n".join(lines)

    top_report = payload.get("top_report")

    if top_report:
        lines.append(
            "Top      "
            f"{top_report['module']}  "
            f"LUT={top_report['lut']}  "
            f"FF={top_report['ff']}  "
            f"DSP={top_report['dsp']}  "
            f"BRAM18={top_report['bram18']}  "
            f"Cycles={top_report['latency_cycles']:.0f}"
        )

    lines.append(
        "-----------------------------------------------------------"
    )
    lines.append(
        "Module                                   "
        "Type          LUT      FF       DSP   BRAM  Cycles"
    )

    child_modules = [
        module
        for module in payload["modules"]
        if not module["is_top"]
    ]

    for module in child_modules[:25]:
        lines.append(
            f"{str(module['module'])[:40]:<40} "
            f"{str(module['op_type'])[:12]:<12} "
            f"{int(module['lut']):<8} "
            f"{int(module['ff']):<8} "
            f"{int(module['dsp']):<5} "
            f"{int(module['bram18']):<5} "
            f"{float(module['latency_cycles']):.0f}"
        )

    if len(child_modules) > 25:
        lines.append(
            f"... {len(child_modules) - 25} more modules in results.json"
        )

    lines.append(
        "-----------------------------------------------------------"
    )

    if payload["by_operator"]:
        lines.append("Aggregated child reports by operator:")

        ordered = sorted(
            payload["by_operator"].items(),
            key=lambda item: (
                item[1]["dsp"],
                item[1]["lut"],
            ),
            reverse=True,
        )

        for op_type, values in ordered:
            lines.append(
                f" {op_type:<20} "
                f"modules={values['module_count']:<3} "
                f"LUT={values['lut']:<8} "
                f"FF={values['ff']:<8} "
                f"DSP={values['dsp']:<5} "
                f"BRAM18={values['bram18']}"
            )

    lines.append(
        "==========================================================="
    )

    return "\n".join(lines)


@dataclass(frozen=True)
class HlsModuleBreakdownResult:
    out_dir: Path
    results_json: Path
    results_csv: Path
    summary_txt: Path
    terminal_summary: str
    available: bool


def run_hls_module_breakdown(
    *,
    out_dir: str | Path,
    report_path: str | Path,
    top_name: str | None = None,
) -> HlsModuleBreakdownResult:
    output_root = Path(out_dir).resolve()
    breakdown_dir = (
        output_root
        / "estimate_vs_hls"
        / "modules"
    )
    breakdown_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = collect_hls_module_reports(
        report_path,
        top_name=top_name,
    )
    terminal_summary = _terminal_summary(payload)

    results_json = breakdown_dir / "results.json"
    results_csv = breakdown_dir / "modules.csv"
    summary_txt = breakdown_dir / "summary.txt"

    results_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = [
        "module",
        "op_type",
        "is_top",
        "lut",
        "ff",
        "dsp",
        "bram18",
        "latency_cycles",
        "initiation_interval",
        "report_name",
        "report_path",
    ]

    with results_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(payload["modules"])

    summary_txt.write_text(
        terminal_summary + "\n",
        encoding="utf-8",
    )

    return HlsModuleBreakdownResult(
        out_dir=breakdown_dir,
        results_json=results_json,
        results_csv=results_csv,
        summary_txt=summary_txt,
        terminal_summary=terminal_summary,
        available=bool(payload["available"]),
    )