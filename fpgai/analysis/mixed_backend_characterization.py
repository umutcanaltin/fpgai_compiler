from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fpgai.reporting.vivado_bridge_artifacts import _parse_power, _parse_timing, _parse_utilization


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _parse_uram(text: str) -> int | None:
    for pattern in (r"\|\s*URAM\s*\|\s*([0-9,]+)", r"\|\s*URAM288\s*\|\s*([0-9,]+)"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def characterize_mixed_backend_implementation(
    reports_dir: str | Path,
    *,
    target_clock_mhz: float,
    scope: str = "quantized_residual_cnn_hls_plus_vhdl_transport",
) -> dict[str, Any]:
    reports = Path(reports_dir).resolve()
    util_path = reports / "dag_mixed_backend_utilization_impl.rpt"
    timing_path = reports / "dag_mixed_backend_timing_impl.rpt"
    power_path = reports / "dag_mixed_backend_power_impl.rpt"

    util_text = _read(util_path)
    timing = _parse_timing(_read(timing_path))
    resources = _parse_utilization(util_text)
    power = _parse_power(_read(power_path))
    wns = timing.get("wns_ns")
    target_mhz = float(target_clock_mhz)
    target_period_ns = 1000.0 / target_mhz
    timing_met = None if wns is None else float(wns) >= 0.0

    return {
        "schema": "fpgai.mixed-backend-implementation-characterization/v1",
        "status": (
            "passed"
            if util_path.is_file() and timing_path.is_file() and timing_met is not False
            else ("timing_failed" if util_path.is_file() and timing_path.is_file() else "not_run")
        ),
        "validation_level": "vivado_implemented" if util_path.is_file() and timing_path.is_file() else "vivado_synthesized",
        "scope": str(scope),
        "clock": {
            "target_mhz": target_mhz,
            "target_period_ns": target_period_ns,
            "wns_ns": wns,
            "tns_ns": timing.get("tns_ns"),
            "whs_ns": timing.get("whs_ns"),
            "ths_ns": timing.get("ths_ns"),
            "timing_met": timing_met,
        },
        "resources": {
            "lut": resources.get("lut"),
            "ff": resources.get("ff"),
            "bram": resources.get("bram"),
            "dsp": resources.get("dsp"),
            "uram": _parse_uram(util_text),
        },
        "power": power,
        "artifacts": {
            "utilization_report": str(util_path) if util_path.is_file() else None,
            "timing_report": str(timing_path) if timing_path.is_file() else None,
            "power_report": str(power_path) if power_path.is_file() else None,
        },
        "measurement_comparability": "whole_design_only",
        "fmax_semantics": "WNS validates the requested clock only; this report does not claim measured Fmax.",
        "usage": {"platform_scope": "research", "production_path": "morfics"},
    }


def write_mixed_backend_characterization(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
