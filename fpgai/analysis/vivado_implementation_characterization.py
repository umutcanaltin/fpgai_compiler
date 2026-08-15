from __future__ import annotations

"""Post-Vivado implementation characterization for compiler manifests.

Timing interpretation intentionally separates *constraint compliance* from any
attempt to infer a maximum clock frequency. Vivado WNS is a slack measurement;
it is not, in general, an achieved-Fmax measurement. A simple
``target_period - WNS`` conversion is only meaningful for a restricted class of
single-clock paths and can become physically nonsensical (for example, when
positive slack is larger than the requested period).
"""

import json
import re
from pathlib import Path
from typing import Any, Mapping

from fpgai.reporting.vivado_bridge_artifacts import _parse_power, _parse_timing, _parse_utilization


def _read(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _first(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(p for p in root.glob(pattern) if p.is_file())
        if matches:
            return matches[0]
    return None


def _parse_uram(text: str) -> int | None:
    for pat in (r"\|\s*URAM\s*\|\s*([0-9,]+)", r"\|\s*URAM288\s*\|\s*([0-9,]+)"):
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _slack_based_frequency_estimate(
    *, target_period_ns: float, wns_ns: float | None
) -> tuple[float | None, float | None, str]:
    """Return a conservative slack-based estimate only when mathematically sane.

    This estimate is *not* called achieved Fmax. Even a mathematically valid
    WNS-derived number can be invalidated by generated clocks, multicycle paths,
    asynchronous groups, or other timing exceptions. A real Fmax measurement
    requires an explicit clock sweep / implementation experiment.
    """

    if wns_ns is None:
        return None, None, "unavailable_no_wns"

    candidate_period = float(target_period_ns) - float(wns_ns)
    if candidate_period <= 0.0:
        return None, None, "not_computable_wns_exceeds_or_equals_target_period"

    return candidate_period, 1000.0 / candidate_period, "slack_based_estimate_only"


def characterize_vivado_implementation(
    build_dir: str | Path,
    *,
    target_clock_mhz: float,
    external_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(build_dir).resolve()
    bridge = root / "vivado_bridge"
    reports = bridge / "reports"
    util_path = _first(reports, ("utilization_impl.rpt", "*utilization*impl*.rpt", "*util*.rpt"))
    timing_path = _first(reports, ("timing_impl.rpt", "*timing*impl*.rpt", "*timing*.rpt"))
    power_path = _first(reports, ("power_impl.rpt", "*power*impl*.rpt", "*power*.rpt"))
    bit = _first(bridge, ("bitstream/*.bit", "project/**/*.bit"))
    xsa = _first(bridge, ("bitstream/*.xsa", "project/**/*.xsa"))

    util_text = _read(util_path)
    timing = _parse_timing(_read(timing_path))
    resources = _parse_utilization(util_text)
    power = _parse_power(_read(power_path))
    uram = _parse_uram(util_text)

    target_mhz = float(target_clock_mhz)
    period_ns = 1000.0 / target_mhz
    wns = timing.get("wns_ns")
    timing_met = None if wns is None else float(wns) >= 0.0
    slack_period, slack_fmax, fmax_status = _slack_based_frequency_estimate(
        target_period_ns=period_ns,
        wns_ns=None if wns is None else float(wns),
    )

    implementation_present = bool(util_path or timing_path)
    status = "passed" if implementation_present and timing_met is not False else ("timing_failed" if implementation_present and timing_met is False else "not_run")
    validation_level = "vivado_implemented" if implementation_present else "hls_synthesized"
    if bit and xsa:
        validation_level = "bitstream_generated"

    return {
        "schema": "fpgai.vivado-implementation-characterization/v1",
        "status": status,
        "validation_level": validation_level,
        "scope": "mixed_graph_top",
        "measurement_comparability": "whole_design_only",
        "clock": {
            "target_mhz": target_mhz,
            "target_period_ns": period_ns,
            "wns_ns": wns,
            "tns_ns": timing.get("tns_ns"),
            "whs_ns": timing.get("whs_ns"),
            "ths_ns": timing.get("ths_ns"),
            "timing_met": timing_met,
            # Kept for schema/backward compatibility, but no longer populated
            # when WNS cannot support a meaningful period derivation.
            "derived_achieved_period_ns": slack_period,
            "derived_fmax_mhz": slack_fmax,
            "fmax_derivation_status": fmax_status,
            "fmax_semantics": (
                "WNS validates the requested timing constraint. Any non-null derived_fmax_mhz "
                "is a slack-based estimate only, not a measured achieved Fmax. A clock sweep is "
                "required for implementation-level Fmax characterization."
            ),
        },
        "resources": {
            "lut": resources.get("lut"),
            "ff": resources.get("ff"),
            "bram": resources.get("bram"),
            "dsp": resources.get("dsp"),
            "uram": uram,
        },
        "power": power,
        "artifacts": {
            "utilization_report": str(util_path) if util_path else None,
            "timing_report": str(timing_path) if timing_path else None,
            "power_report": str(power_path) if power_path else None,
            "bitstream": str(bit) if bit else None,
            "xsa": str(xsa) if xsa else None,
        },
        "external_provenance": dict(external_provenance or {}),
        "usage": {"platform_scope": "research", "production_path": "morfics"},
    }


def write_vivado_implementation_characterization(payload: Mapping[str, Any], reports_dir: str | Path) -> tuple[Path, Path]:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "vivado_implementation_characterization.json"
    md_path = root / "vivado_implementation_characterization.md"
    json_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clock = payload.get("clock", {})
    resources = payload.get("resources", {})
    power = payload.get("power", {})
    lines = [
        "# Vivado implementation characterization", "",
        f"- Status: `{payload.get('status')}`",
        f"- Validation level: `{payload.get('validation_level')}`",
        f"- Scope: `{payload.get('scope')}`",
        f"- WNS / TNS: {clock.get('wns_ns')} / {clock.get('tns_ns')} ns",
        f"- Timing met at requested clock: `{clock.get('timing_met')}`",
        f"- Requested clock: {clock.get('target_mhz')} MHz ({clock.get('target_period_ns')} ns)",
        f"- Slack-based Fmax estimate: {clock.get('derived_fmax_mhz')} MHz",
        f"- Fmax derivation status: `{clock.get('fmax_derivation_status')}`",
        f"- Resources: LUT={resources.get('lut')}, FF={resources.get('ff')}, DSP={resources.get('dsp')}, BRAM={resources.get('bram')}, URAM={resources.get('uram')}",
        f"- Total on-chip power: {power.get('total_on_chip_power_w')} W",
        "",
        "WNS is used to determine whether the requested timing constraint is met. It is not treated as an achieved-Fmax measurement. A clock sweep is required for implementation-level Fmax characterization.",
        "",
        "Measurements are for the complete implemented top and are not attributed to one external node.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
