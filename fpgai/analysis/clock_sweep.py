from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class ClockSweepPoint:
    target_mhz: float
    status: str
    timing_met: bool | None
    wns_ns: float | None
    out_dir: str | None = None
    requested_period_ns: float | None = None
    hls_period_ns: float | None = None
    vivado_constraint_period_ns: float | None = None
    vivado_clock_mhz: float | None = None
    constraint_verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_mhz": self.target_mhz,
            "requested_period_ns": self.requested_period_ns,
            "hls_period_ns": self.hls_period_ns,
            "vivado_constraint_period_ns": self.vivado_constraint_period_ns,
            "vivado_clock_mhz": self.vivado_clock_mhz,
            "constraint_verified": self.constraint_verified,
            "status": self.status,
            "timing_met": self.timing_met,
            "wns_ns": self.wns_ns,
            "out_dir": self.out_dir,
        }


def parse_vivado_clock_period(timing_report: str | Path, *, clock_name: str = "clk_pl_0") -> float | None:
    """Read the implemented PL clock period from a Vivado timing summary.

    The detailed clock block is preferred over HLS/OOC constraints because it
    describes the clock that actually constrained the implemented block design.
    """
    path = Path(timing_report)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(
        rf"Clock Name:\s*{re.escape(clock_name)}\s*.*?Period\(ns\):\s*([-+0-9.eE]+)",
        re.DOTALL,
    )
    match = pat.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def _period_matches(requested: float, actual: float) -> bool:
    # Allow formatting/clock-generator rounding while rejecting a materially
    # different system clock (for example 3.333 ns requested vs 10 ns actual).
    tolerance = max(0.05, abs(requested) * 0.01)
    return abs(requested - actual) <= tolerance


def summarize_clock_sweep(points: Iterable[ClockSweepPoint]) -> dict[str, Any]:
    ordered = sorted(points, key=lambda p: p.target_mhz)
    verified = [p for p in ordered if p.constraint_verified]
    passing = [p for p in verified if p.timing_met is True and p.status == "passed"]
    failing = [p for p in verified if p.timing_met is False or p.status == "timing_failed"]
    highest_pass = max((p.target_mhz for p in passing), default=None)
    first_fail = min((p.target_mhz for p in failing if highest_pass is None or p.target_mhz > highest_pass), default=None)
    unverified = [p for p in ordered if not p.constraint_verified]
    status = "passed" if passing else ("constraint_unverified" if unverified else "no_passing_point")
    return {
        "schema": "fpgai.vivado-clock-sweep/v2",
        "status": status,
        "points": [p.to_dict() for p in ordered],
        "highest_passing_mhz": highest_pass,
        "first_failing_mhz_above_pass": first_fail,
        "fmax_bracket_mhz": {
            "lower_bound": highest_pass,
            "upper_bound": first_fail,
        },
        "unverified_point_count": len(unverified),
        "measurement_semantics": (
            "Only points whose implemented Vivado clock constraint matches the requested clock contribute to the Fmax bracket. "
            "highest_passing_mhz is an implementation-tested lower bound; when first_failing_mhz_above_pass is available, "
            "the implementation-level Fmax lies within the reported bracket."
        ),
    }


def point_from_characterization(
    target_mhz: float,
    payload: Mapping[str, Any],
    *,
    out_dir: str | None = None,
    timing_report: str | Path | None = None,
    hls_payload: Mapping[str, Any] | None = None,
) -> ClockSweepPoint:
    clock = payload.get("clock", {}) if isinstance(payload, Mapping) else {}
    requested_period = 1000.0 / float(target_mhz)
    hls_clock = hls_payload.get("clock", {}) if isinstance(hls_payload, Mapping) else {}
    hls_period = hls_clock.get("target_period_ns")
    actual_period = parse_vivado_clock_period(timing_report) if timing_report else None
    constraint_verified = actual_period is not None and _period_matches(requested_period, actual_period)
    actual_mhz = (1000.0 / actual_period) if actual_period else None

    raw_status = str(payload.get("status", "unknown"))
    if actual_period is None:
        status = "constraint_unverified"
    elif not constraint_verified:
        status = "constraint_mismatch"
    elif clock.get("timing_met") is False:
        status = "timing_failed"
    else:
        status = raw_status

    return ClockSweepPoint(
        target_mhz=float(target_mhz),
        status=status,
        timing_met=clock.get("timing_met"),
        wns_ns=clock.get("wns_ns"),
        out_dir=out_dir,
        requested_period_ns=requested_period,
        hls_period_ns=float(hls_period) if hls_period is not None else None,
        vivado_constraint_period_ns=actual_period,
        vivado_clock_mhz=actual_mhz,
        constraint_verified=constraint_verified,
    )
