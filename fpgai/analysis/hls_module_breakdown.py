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

OPERATOR_PATTERNS = [
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

HELPER_FRAGMENTS = (
    "_pipeline_",
    "_loop_",
    "pipeline_vitis_loop",
    "_proc",
    "_entry",
    "_exit",
    "_read_",
    "_write_",
    "_load_",
    "_store_",
)


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(
            float(
                str(value)
                .replace(",", "")
                .strip()
            )
        )
    except (TypeError, ValueError):
        return default


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(
            str(value)
            .replace(",", "")
            .strip()
        )
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
        value = _extract_tag(
            text,
            tag,
        )

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


def _latency_cycles(
    text: str,
) -> float:
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


def _initiation_interval(
    text: str,
) -> float:
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


def _classify_module(
    module_name: str,
) -> str:
    name = module_name.lower()

    for op_type, fragments in OPERATOR_PATTERNS:
        if any(
            fragment in name
            for fragment in fragments
        ):
            return op_type

    return "Unknown"


def _is_top_module(
    module_name: str,
    requested_top_name: str | None,
) -> bool:
    normalized = module_name.strip().lower()

    if requested_top_name:
        return (
            normalized
            == requested_top_name.strip().lower()
        )

    return normalized in {
        "deeplearn",
        "fpgai",
        "top",
    }


def _is_generated_helper(
    module_name: str,
) -> bool:
    normalized = module_name.lower()

    return any(
        fragment in normalized
        for fragment in HELPER_FRAGMENTS
    )


def _parent_hint(
    module_name: str,
) -> str | None:
    normalized = module_name

    separators = [
        "_Pipeline_",
        "_pipeline_",
        "_Loop_",
        "_loop_",
        "_proc",
    ]

    for separator in separators:
        if separator in normalized:
            parent = normalized.split(
                separator,
                1,
            )[0]

            return parent or None

    return None


def _parse_xml_report(
    report_path: Path,
) -> Dict[str, Any]:
    text = report_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )
    module = _module_name(
        text,
        report_path,
    )

    return {
        "module": module,
        "report_path": str(report_path),
        "report_name": report_path.name,
        "lut": _safe_int(
            _extract_tag(
                text,
                RESOURCE_TAGS["lut"],
            )
        ),
        "ff": _safe_int(
            _extract_tag(
                text,
                RESOURCE_TAGS["ff"],
            )
        ),
        "dsp": _safe_int(
            _extract_tag(
                text,
                RESOURCE_TAGS["dsp"],
            )
        ),
        "bram18": _safe_int(
            _extract_tag(
                text,
                RESOURCE_TAGS["bram18"],
            )
        ),
        "latency_cycles": _latency_cycles(
            text
        ),
        "initiation_interval": (
            _initiation_interval(text)
        ),
    }


def _report_score(
    report: Dict[str, Any],
) -> int:
    return (
        int(report["lut"])
        + int(report["ff"])
        + 10 * int(report["dsp"])
        + 10 * int(report["bram18"])
    )


def _deduplicate_reports(
    reports: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}

    for report in reports:
        module = str(report["module"])
        current = selected.get(module)

        if (
            current is None
            or _report_score(report)
            > _report_score(current)
        ):
            selected[module] = report

    return list(selected.values())


def _resource_sum(
    reports: List[Dict[str, Any]],
) -> Dict[str, int]:
    return {
        "lut": sum(
            int(report["lut"])
            for report in reports
        ),
        "ff": sum(
            int(report["ff"])
            for report in reports
        ),
        "dsp": sum(
            int(report["dsp"])
            for report in reports
        ),
        "bram18": sum(
            int(report["bram18"])
            for report in reports
        ),
    }


def _aggregate_by_operator(
    reports: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}

    for report in reports:
        op_type = str(report["op_type"])

        aggregate = result.setdefault(
            op_type,
            {
                "module_count": 0,
                "lut": 0,
                "ff": 0,
                "dsp": 0,
                "bram18": 0,
            },
        )

        aggregate["module_count"] += 1
        aggregate["lut"] += int(
            report["lut"]
        )
        aggregate["ff"] += int(
            report["ff"]
        )
        aggregate["dsp"] += int(
            report["dsp"]
        )
        aggregate["bram18"] += int(
            report["bram18"]
        )

    return result


def _select_primary_modules(
    reports: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates = [
        report
        for report in reports
        if (
            not report["is_top"]
            and not report["is_generated_helper"]
        )
    ]

    grouped: Dict[
        tuple[str, str],
        List[Dict[str, Any]],
    ] = {}

    for report in candidates:
        op_type = str(report["op_type"])
        module = str(report["module"])

        # Multiple reports can exist for one templated function because
        # Vitis creates separate internal stages. Keep the complete
        # function report, which is normally the largest report.
        normalized_family = re.sub(
            r"_(?:stage|part|block|region)_?\d*$",
            "",
            module,
            flags=re.IGNORECASE,
        )

        key = (
            op_type,
            normalized_family,
        )
        grouped.setdefault(
            key,
            [],
        ).append(report)

    primary: List[Dict[str, Any]] = []

    for group in grouped.values():
        selected = max(
            group,
            key=lambda report: (
                _report_score(report),
                float(
                    report[
                        "latency_cycles"
                    ]
                ),
            ),
        )
        primary.append(selected)

    primary.sort(
        key=lambda report: (
            int(report["dsp"]),
            int(report["lut"]),
            int(report["ff"]),
        ),
        reverse=True,
    )

    return primary


def collect_hls_module_reports(
    report_path: str | Path,
    *,
    top_name: str | None = None,
) -> Dict[str, Any]:
    requested = Path(report_path)

    search_root = (
        requested
        if requested.is_dir()
        else requested.parent
    )

    xml_paths = sorted(
        search_root.rglob(
            "*_csynth.xml"
        )
    )

    if (
        requested.is_file()
        and requested.suffix.lower()
        == ".xml"
        and requested not in xml_paths
    ):
        xml_paths.append(requested)

    reports = _deduplicate_reports(
        [
            _parse_xml_report(path)
            for path in xml_paths
            if path.is_file()
        ]
    )

    for report in reports:
        module = str(report["module"])

        report["op_type"] = (
            _classify_module(module)
        )
        report["is_top"] = (
            _is_top_module(
                module,
                top_name,
            )
        )
        report["is_generated_helper"] = (
            _is_generated_helper(module)
        )
        report["parent_hint"] = (
            _parent_hint(module)
        )

        if report["is_top"]:
            report["hierarchy_role"] = "top"
        elif report["is_generated_helper"]:
            report["hierarchy_role"] = "helper"
        else:
            report["hierarchy_role"] = "primary"

    reports.sort(
        key=lambda report: (
            bool(report["is_top"]),
            (
                report[
                    "hierarchy_role"
                ]
                == "primary"
            ),
            int(report["dsp"]),
            int(report["lut"]),
        ),
        reverse=True,
    )

    top_reports = [
        report
        for report in reports
        if report["is_top"]
    ]
    helper_reports = [
        report
        for report in reports
        if (
            not report["is_top"]
            and report[
                "is_generated_helper"
            ]
        )
    ]
    primary_reports = (
        _select_primary_modules(
            reports
        )
    )

    primary_names = {
        str(report["module"])
        for report in primary_reports
    }

    for report in reports:
        if (
            report["hierarchy_role"]
            == "primary"
            and str(report["module"])
            not in primary_names
        ):
            report["hierarchy_role"] = (
                "secondary"
            )

    top_report = (
        max(
            top_reports,
            key=_report_score,
        )
        if top_reports
        else (
            reports[0]
            if len(reports) == 1
            else None
        )
    )

    primary_sum = _resource_sum(
        primary_reports
    )
    helper_sum = _resource_sum(
        helper_reports
    )

    top_resources = (
        {
            "lut": int(top_report["lut"]),
            "ff": int(top_report["ff"]),
            "dsp": int(top_report["dsp"]),
            "bram18": int(
                top_report["bram18"]
            ),
        }
        if top_report is not None
        else {
            "lut": 0,
            "ff": 0,
            "dsp": 0,
            "bram18": 0,
        }
    )

    unassigned_top_resources = {
        resource: max(
            0,
            top_resources[resource]
            - primary_sum[resource],
        )
        for resource in [
            "lut",
            "ff",
            "dsp",
            "bram18",
        ]
    }

    return {
        "format": (
            "fpgai.hls_module_breakdown.v2"
        ),
        "available": bool(reports),
        "search_root": str(search_root),
        "top_name": top_name,
        "top_report": top_report,
        "module_count": len(reports),
        "primary_module_count": len(
            primary_reports
        ),
        "helper_module_count": len(
            helper_reports
        ),
        "child_module_count": (
            len(reports)
            - len(top_reports)
        ),
        # Compatibility key. It now correctly represents only primary
        # function reports and excludes nested pipeline helpers.
        "child_sum": primary_sum,
        "primary_sum": primary_sum,
        "helper_sum": helper_sum,
        "top_resources": top_resources,
        "unassigned_top_resources": (
            unassigned_top_resources
        ),
        "by_operator": (
            _aggregate_by_operator(
                primary_reports
            )
        ),
        "helper_by_operator": (
            _aggregate_by_operator(
                helper_reports
            )
        ),
        "primary_modules": primary_reports,
        "helper_modules": helper_reports,
        "modules": reports,
        "aggregation_note": (
            "Primary operator totals exclude generated pipeline and loop "
            "helper reports. Helper resources are hierarchical subsets and "
            "must not be added to their parent function resources."
        ),
    }


def _terminal_summary(
    payload: Dict[str, Any],
) -> str:
    lines = [
        (
            "=============== FPGAI HLS "
            "Module Breakdown ==============="
        )
    ]

    if not payload["available"]:
        lines.extend(
            [
                (
                    "No per-function "
                    "*_csynth.xml reports were found."
                ),
                (
                    "================================"
                    "==========================="
                ),
            ]
        )
        return "\n".join(lines)

    top = payload.get(
        "top_report"
    )

    if top is not None:
        lines.append(
            f"Top      {top['module']}  "
            f"LUT={top['lut']}  "
            f"FF={top['ff']}  "
            f"DSP={top['dsp']}  "
            f"BRAM18={top['bram18']}  "
            f"Cycles="
            f"{float(top['latency_cycles']):.0f}"
        )

    lines.extend(
        [
            (
                "--------------------------------"
                "---------------------------"
            ),
            (
                "Primary module                           "
                "Type          LUT      FF       "
                "DSP   BRAM  Cycles"
            ),
        ]
    )

    for module in payload[
        "primary_modules"
    ]:
        lines.append(
            f"{str(module['module'])[:40]:<40} "
            f"{str(module['op_type'])[:12]:<12} "
            f"{int(module['lut']):<8} "
            f"{int(module['ff']):<8} "
            f"{int(module['dsp']):<5} "
            f"{int(module['bram18']):<5} "
            f"{float(module['latency_cycles']):.0f}"
        )

    lines.append(
        "--------------------------------"
        "---------------------------"
    )

    if payload["by_operator"]:
        lines.append(
            "Primary resources by operator:"
        )

        ordered = sorted(
            payload[
                "by_operator"
            ].items(),
            key=lambda item: (
                item[1]["dsp"],
                item[1]["lut"],
            ),
            reverse=True,
        )

        for op_type, values in ordered:
            lines.append(
                f" {op_type:<20} "
                f"modules="
                f"{values['module_count']:<3} "
                f"LUT={values['lut']:<8} "
                f"FF={values['ff']:<8} "
                f"DSP={values['dsp']:<5} "
                f"BRAM18="
                f"{values['bram18']}"
            )

    unassigned = payload[
        "unassigned_top_resources"
    ]

    lines.extend(
        [
            (
                "--------------------------------"
                "---------------------------"
            ),
            (
                "Top-level/unassigned resources: "
                f"LUT={unassigned['lut']}  "
                f"FF={unassigned['ff']}  "
                f"DSP={unassigned['dsp']}  "
                f"BRAM18={unassigned['bram18']}"
            ),
            (
                f"Excluded generated helpers: "
                f"{payload['helper_module_count']}"
            ),
            (
                "Generated helper reports are "
                "hierarchical subsets and are not summed."
            ),
            (
                "================================"
                "==========================="
            ),
        ]
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
    output_root = Path(
        out_dir
    ).resolve()
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
    terminal_summary = (
        _terminal_summary(payload)
    )

    results_json = (
        breakdown_dir / "results.json"
    )
    results_csv = (
        breakdown_dir / "modules.csv"
    )
    summary_txt = (
        breakdown_dir / "summary.txt"
    )

    results_json.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fields = [
        "module",
        "op_type",
        "hierarchy_role",
        "is_top",
        "is_generated_helper",
        "parent_hint",
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
        writer.writerows(
            payload["modules"]
        )

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
        available=bool(
            payload["available"]
        ),
    )