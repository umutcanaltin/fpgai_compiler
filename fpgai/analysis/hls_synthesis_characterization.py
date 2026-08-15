from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from fpgai.analysis.hls_estimate_compare import parse_hls_csynth_report


@dataclass(frozen=True)
class HLSSynthesisCharacterization:
    status: str
    report_path: str | None
    scope: str
    top_name: str
    target_clock_mhz: float
    target_clock_period_ns: float
    estimated_clock_period_ns: float | None
    estimated_fmax_mhz: float | None
    timing_margin_ns: float | None
    target_met: bool | None
    latency_min_cycles: int | None
    latency_max_cycles: int | None
    initiation_interval_min: int | None
    initiation_interval_max: int | None
    resources: Mapping[str, int]
    participating_external_packages: tuple[str, ...] = ()
    declared_implementation_metrics: Mapping[str, Mapping[str, Any]] | None = None
    measurement_comparability: str = "whole_design_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.hls-synthesis-characterization/v1",
            "status": self.status,
            "scope": self.scope,
            "top_name": self.top_name,
            "report_path": self.report_path,
            "clock": {
                "target_mhz": self.target_clock_mhz,
                "target_period_ns": self.target_clock_period_ns,
                "estimated_period_ns": self.estimated_clock_period_ns,
                "estimated_fmax_mhz": self.estimated_fmax_mhz,
                "timing_margin_ns": self.timing_margin_ns,
                "target_met": self.target_met,
            },
            "latency": {
                "min_cycles": self.latency_min_cycles,
                "max_cycles": self.latency_max_cycles,
                "initiation_interval_min": self.initiation_interval_min,
                "initiation_interval_max": self.initiation_interval_max,
            },
            "resources": dict(self.resources),
            "participating_external_packages": list(self.participating_external_packages),
            "declared_implementation_metrics": dict(self.declared_implementation_metrics or {}),
            "measurement_comparability": self.measurement_comparability,
            "validation_level": "hls_synthesized" if self.status == "passed" else "unavailable",
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_float(value)
    return None if number is None else int(number)


def _xml_text(root: ET.Element, names: tuple[str, ...]) -> str | None:
    wanted = {name.lower() for name in names}
    for node in root.iter():
        if node.tag.split("}")[-1].lower() in wanted and node.text:
            text = node.text.strip()
            if text:
                return text
    return None


def _parse_xml_details(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    return {
        "estimated_clock_period_ns": _finite_float(
            _xml_text(root, ("EstimatedClockPeriod", "EstimatedClock", "ClockPeriod"))
        ),
        "latency_min_cycles": _integer(
            _xml_text(root, ("Best-caseLatency", "LatencyMin", "MinLatency"))
        ),
        "latency_max_cycles": _integer(
            _xml_text(root, ("Worst-caseLatency", "LatencyMax", "MaxLatency", "Average-caseLatency"))
        ),
        "initiation_interval_min": _integer(
            _xml_text(root, ("Interval-min", "IntervalMin", "MinInterval"))
        ),
        "initiation_interval_max": _integer(
            _xml_text(root, ("Interval-max", "IntervalMax", "MaxInterval", "Interval"))
        ),
    }


def _search_number(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            value = _finite_float(match.group(1).replace(",", ""))
            if value is not None:
                return value
    return None


def _parse_text_details(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    latency_min = _search_number(text, (
        r"Latency\s*\(cycles\).*?min\s*=\s*([0-9,.]+)",
        r"Best-case\s+Latency\s*[:=]\s*([0-9,.]+)",
    ))
    latency_max = _search_number(text, (
        r"Latency\s*\(cycles\).*?max\s*=\s*([0-9,.]+)",
        r"Worst-case\s+Latency\s*[:=]\s*([0-9,.]+)",
        r"Latency\s*\(cycles\)\s*[:=]\s*([0-9,.]+)",
    ))
    ii_min = _search_number(text, (
        r"Interval.*?min\s*=\s*([0-9,.]+)",
        r"Initiation\s+Interval.*?min\s*=\s*([0-9,.]+)",
    ))
    ii_max = _search_number(text, (
        r"Interval.*?max\s*=\s*([0-9,.]+)",
        r"Initiation\s+Interval.*?max\s*=\s*([0-9,.]+)",
        r"Initiation\s+Interval\s*[:=]\s*([0-9,.]+)",
    ))
    return {
        "estimated_clock_period_ns": _search_number(text, (
            r"Estimated\s+Clock\s+Period\s*[:=]\s*([0-9,.]+)",
            r"EstimatedClockPeriod\s*[:=]\s*([0-9,.]+)",
        )),
        "latency_min_cycles": _integer(latency_min),
        "latency_max_cycles": _integer(latency_max if latency_max is not None else latency_min),
        "initiation_interval_min": _integer(ii_min),
        "initiation_interval_max": _integer(ii_max if ii_max is not None else ii_min),
    }


def characterize_hls_synthesis(
    *,
    csynth_report_path: str | Path | None,
    target_clock_mhz: float,
    top_name: str,
    participating_external_packages: tuple[str, ...] = (),
    declared_implementation_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    scope: str = "mixed_graph_top",
) -> HLSSynthesisCharacterization:
    target_mhz = float(target_clock_mhz)
    if not math.isfinite(target_mhz) or target_mhz <= 0:
        raise ValueError("target_clock_mhz must be a positive finite value")
    target_period = 1000.0 / target_mhz

    if csynth_report_path is None:
        return HLSSynthesisCharacterization(
            status="not_run",
            report_path=None,
            scope=scope,
            top_name=top_name,
            target_clock_mhz=target_mhz,
            target_clock_period_ns=target_period,
            estimated_clock_period_ns=None,
            estimated_fmax_mhz=None,
            timing_margin_ns=None,
            target_met=None,
            latency_min_cycles=None,
            latency_max_cycles=None,
            initiation_interval_min=None,
            initiation_interval_max=None,
            resources={"lut": 0, "ff": 0, "dsp": 0, "bram18": 0, "uram": 0},
            participating_external_packages=participating_external_packages,
            declared_implementation_metrics=declared_implementation_metrics,
        )

    path = Path(csynth_report_path).resolve()
    if not path.is_file():
        return HLSSynthesisCharacterization(
            status="report_missing",
            report_path=str(path),
            scope=scope,
            top_name=top_name,
            target_clock_mhz=target_mhz,
            target_clock_period_ns=target_period,
            estimated_clock_period_ns=None,
            estimated_fmax_mhz=None,
            timing_margin_ns=None,
            target_met=None,
            latency_min_cycles=None,
            latency_max_cycles=None,
            initiation_interval_min=None,
            initiation_interval_max=None,
            resources={"lut": 0, "ff": 0, "dsp": 0, "bram18": 0, "uram": 0},
            participating_external_packages=participating_external_packages,
            declared_implementation_metrics=declared_implementation_metrics,
        )

    selected = path
    sibling_xml = path.parent / f"{top_name}_csynth.xml"
    if sibling_xml.is_file():
        selected = sibling_xml
    elif (path.parent / "csynth.xml").is_file():
        selected = path.parent / "csynth.xml"

    details = _parse_xml_details(selected) if selected.suffix.lower() == ".xml" else _parse_text_details(selected)
    actual = parse_hls_csynth_report(selected)
    estimated_period = details["estimated_clock_period_ns"]
    fmax = None if not estimated_period or estimated_period <= 0 else 1000.0 / estimated_period
    margin = None if estimated_period is None else target_period - estimated_period
    target_met = None if estimated_period is None else estimated_period <= target_period

    latency_max = details["latency_max_cycles"]
    if latency_max is None:
        parsed_latency = _integer(actual.get("actual_latency_cycles"))
        latency_max = parsed_latency if parsed_latency and parsed_latency > 0 else None

    return HLSSynthesisCharacterization(
        status="passed",
        report_path=str(selected),
        scope=scope,
        top_name=top_name,
        target_clock_mhz=target_mhz,
        target_clock_period_ns=target_period,
        estimated_clock_period_ns=estimated_period,
        estimated_fmax_mhz=fmax,
        timing_margin_ns=margin,
        target_met=target_met,
        latency_min_cycles=details["latency_min_cycles"],
        latency_max_cycles=latency_max,
        initiation_interval_min=details["initiation_interval_min"],
        initiation_interval_max=details["initiation_interval_max"],
        resources={
            "lut": int(actual.get("actual_lut", 0) or 0),
            "ff": int(actual.get("actual_ff", 0) or 0),
            "dsp": int(actual.get("actual_dsp", 0) or 0),
            "bram18": int(actual.get("actual_bram18", 0) or 0),
            "uram": int(actual.get("actual_uram", 0) or 0),
        },
        participating_external_packages=participating_external_packages,
        declared_implementation_metrics=declared_implementation_metrics,
    )


def write_hls_synthesis_characterization(
    characterization: HLSSynthesisCharacterization,
    reports_dir: str | Path,
) -> tuple[Path, Path]:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "hls_synthesis_characterization.json"
    md_path = root / "hls_synthesis_characterization.md"
    payload = characterization.to_dict()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    clock = payload["clock"]
    latency = payload["latency"]
    resources = payload["resources"]
    lines = [
        "# HLS synthesis characterization",
        "",
        f"- Status: `{payload['status']}`",
        f"- Scope: `{payload['scope']}`",
        f"- Top: `{payload['top_name']}`",
        f"- Validation level: `{payload['validation_level']}`",
        f"- Target clock: {clock['target_mhz']} MHz ({clock['target_period_ns']:.6g} ns)",
        f"- Estimated clock period: {clock['estimated_period_ns']}",
        f"- Estimated Fmax: {clock['estimated_fmax_mhz']}",
        f"- Target met: {clock['target_met']}",
        f"- Latency cycles: {latency['min_cycles']} .. {latency['max_cycles']}",
        f"- Initiation interval: {latency['initiation_interval_min']} .. {latency['initiation_interval_max']}",
        f"- Resources: LUT={resources['lut']}, FF={resources['ff']}, DSP={resources['dsp']}, BRAM18={resources['bram18']}, URAM={resources['uram']}",
        "",
        "## Measurement scope",
        "",
        "These are whole-design HLS synthesis measurements for the generated top. They are not attributed to an individual external operator in a mixed graph.",
    ]
    if payload["participating_external_packages"]:
        lines.extend(["", "## Participating external packages", ""])
        lines.extend(f"- `{item}`" for item in payload["participating_external_packages"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
