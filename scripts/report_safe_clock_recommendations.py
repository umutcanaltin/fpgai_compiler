#!/usr/bin/env python3
"""Generate safe-clock recommendations from FPGAI Vivado evidence.

Reads <experiment>/vivado_bridge_evidence/evidence.json and writes:
  <experiment>/safe_clock_report/safe_clock_report.csv
  <experiment>/safe_clock_report/safe_clock_report.md
  <experiment>/safe_clock_report/safe_clock_report.json

The script does not rerun Vivado. It estimates the minimum passing period from
reported WNS relative to the current target period:

    estimated_min_period_ns = target_period_ns - wns_ns
    estimated_fmax_mhz      = 1000 / estimated_min_period_ns

A safety margin can be added to recommend a conservative clock.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_TARGET_PERIOD_NS = 5.0
DEFAULT_MARGIN_PCT = 10.0


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y"}
    return bool(value)


def classify(row: Dict[str, Any], lut_limit: int, dsp_limit: int) -> str:
    bit = as_bool(row.get("bitstream_exists") or row.get("bit") or row.get("bitstream"))
    xsa = as_bool(row.get("xsa_exists") or row.get("xsa"))
    wns = as_float(row.get("wns_ns") or row.get("WNS_ns"))
    lut = as_float(row.get("lut") or row.get("LUT"))
    dsp = as_float(row.get("dsp") or row.get("DSP"))
    if lut is not None and lut > lut_limit:
        return "resource_fail"
    if dsp is not None and dsp > dsp_limit:
        return "resource_fail"
    if not bit or not xsa:
        return "implementation_fail"
    if wns is not None and wns < 0:
        return "timing_fail"
    return "pass"


def recommendation(row: Dict[str, Any], target_period_ns: float, margin_pct: float, lut_limit: int, dsp_limit: int) -> Dict[str, Any]:
    design = row.get("design") or row.get("design_name") or "unknown"
    wns = as_float(row.get("wns_ns") or row.get("WNS_ns"))
    power = as_float(row.get("total_power_w") or row.get("power_w"))
    lut = as_float(row.get("lut") or row.get("LUT"))
    ff = as_float(row.get("ff") or row.get("FF"))
    bram = as_float(row.get("bram") or row.get("BRAM"))
    dsp = as_float(row.get("dsp") or row.get("DSP"))
    energy = as_float(row.get("estimated_energy_j") or row.get("energy_j"))
    bit = as_bool(row.get("bitstream_exists") or row.get("bit") or row.get("bitstream"))
    xsa = as_bool(row.get("xsa_exists") or row.get("xsa"))
    status = classify(row, lut_limit, dsp_limit)

    min_period = None
    fmax = None
    recommended_period = None
    recommended_clock = None

    if wns is not None:
        # Positive WNS means current period could theoretically shrink.
        # Negative WNS means period must increase by abs(WNS).
        min_period = max(0.001, target_period_ns - wns)
        fmax = 1000.0 / min_period
        recommended_period = min_period * (1.0 + margin_pct / 100.0)
        recommended_clock = 1000.0 / recommended_period

    if status == "resource_fail":
        note = "Does not fit this FPGA resource envelope; select lower parallelism or larger FPGA."
    elif status == "implementation_fail":
        note = "Implementation did not produce bitstream/XSA; inspect Vivado logs."
    elif status == "timing_fail":
        note = "Bitstream/XSA generated but timing is negative; use lower clock or safer policy."
    else:
        note = "Timing closes at target clock; recommended clock includes safety margin."

    return {
        "design": design,
        "status": status,
        "bitstream_exists": bit,
        "xsa_exists": xsa,
        "target_period_ns": target_period_ns,
        "target_clock_mhz": 1000.0 / target_period_ns,
        "wns_ns": wns,
        "estimated_min_period_ns": min_period,
        "estimated_fmax_mhz": fmax,
        "recommended_safe_period_ns": recommended_period,
        "recommended_safe_clock_mhz": recommended_clock,
        "power_w": power,
        "lut": lut,
        "lut_util_pct": None if lut is None else 100.0 * lut / lut_limit,
        "ff": ff,
        "bram": bram,
        "dsp": dsp,
        "dsp_util_pct": None if dsp is None else 100.0 * dsp / dsp_limit,
        "energy_j": energy,
        "note": note,
    }


def fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "design", "status", "target_clock_mhz", "wns_ns", "estimated_fmax_mhz",
        "recommended_safe_clock_mhz", "lut", "lut_util_pct", "dsp", "dsp_util_pct",
        "bitstream_exists", "xsa_exists", "note",
    ]
    lines = ["# FPGAI safe clock recommendation report", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(h)) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=Path)
    p.add_argument("--target-period-ns", type=float, default=DEFAULT_TARGET_PERIOD_NS)
    p.add_argument("--margin-pct", type=float, default=DEFAULT_MARGIN_PCT)
    p.add_argument("--lut-limit", type=int, default=53200)
    p.add_argument("--dsp-limit", type=int, default=220)
    args = p.parse_args(argv)

    evidence_path = args.experiment / "vivado_bridge_evidence" / "evidence.json"
    if not evidence_path.exists():
        raise SystemExit(f"Missing evidence file: {evidence_path}")

    raw = json.loads(evidence_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        records = raw.get("records") or raw.get("designs") or raw.get("results") or []
    else:
        records = raw
    if not isinstance(records, list):
        raise SystemExit(f"Could not parse records from {evidence_path}")

    rows = [recommendation(r, args.target_period_ns, args.margin_pct, args.lut_limit, args.dsp_limit) for r in records]
    rows.sort(key=lambda r: str(r.get("design", "")))

    out = args.experiment / "safe_clock_report"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "safe_clock_report.json"
    csv_path = out / "safe_clock_report.csv"
    md_path = out / "safe_clock_report.md"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    write_markdown(md_path, rows)
    print(md_path.read_text(encoding="utf-8"))
    print(f"[OK] Wrote {json_path}")
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
