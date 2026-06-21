#!/usr/bin/env python3
"""Extract Vivado/Vitis HLS evidence from FPGAI experiment artifacts.

This extractor is intentionally defensive because different sprints may produce
slightly different directory layouts:

  artifacts/<design>/build/vivado_bridge/...
  artifacts/<design>/vivado_bridge/...

It extracts:
  - HLS IP export/component.xml status
  - Vivado report/bitstream/XSA status
  - power from report_power output
  - utilization: LUT/FF/BRAM/DSP
  - timing: WNS/TNS/WHS/THS
  - HLS latency/cycles and target clock period
  - estimated energy = power_w * latency_cycles * clock_period_ns * 1e-9

Outputs:
  <experiment>/vivado_bridge_evidence/evidence.json
  <experiment>/vivado_bridge_evidence/evidence.csv
  <experiment>/vivado_bridge_evidence/summary.md
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip().replace(",", "")
        if not s or s.lower() in {"none", "nan", "n/a", "-"}:
            return None
        return float(s)
    except Exception:
        return None


def _int(x: Any) -> Optional[int]:
    f = _float(x)
    if f is None:
        return None
    return int(round(f))


def _parse_power(text: str) -> Dict[str, Optional[float]]:
    # Supports common Vivado report_power table rows:
    # | Total On-Chip Power (W) | 1.234 |
    # Total On-Chip Power (W)  1.234
    out: Dict[str, Optional[float]] = {
        "total_power_w": None,
        "total_on_chip_power_w": None,
        "dynamic_power_w": None,
        "static_power_w": None,
    }
    patterns = [
        ("total_on_chip_power_w", r"Total\s+On-Chip\s+Power\s*\(W\)\s*\|?\s*([0-9]+(?:\.[0-9]+)?)"),
        ("total_power_w", r"Total\s+Power\s*\(W\)\s*\|?\s*([0-9]+(?:\.[0-9]+)?)"),
        ("dynamic_power_w", r"Dynamic\s*\(W\)\s*\|?\s*([0-9]+(?:\.[0-9]+)?)"),
        ("static_power_w", r"(?:Device\s+)?Static\s*\(W\)\s*\|?\s*([0-9]+(?:\.[0-9]+)?)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out[key] = _float(m.group(1))
    if out["total_power_w"] is None:
        out["total_power_w"] = out["total_on_chip_power_w"]
    if out["total_on_chip_power_w"] is None:
        out["total_on_chip_power_w"] = out["total_power_w"]
    return out


def _parse_timing(text: str) -> Dict[str, Optional[float]]:
    out = {"wns_ns": None, "tns_ns": None, "whs_ns": None, "ths_ns": None}
    # Common summary lines:
    # WNS(ns)      TNS(ns) ...
    # 0.703        0.000
    # or routed log: WNS=0.703 | TNS=0.000 | WHS=0.019 | THS=0.000
    for key, label in [("wns_ns", "WNS"), ("tns_ns", "TNS"), ("whs_ns", "WHS"), ("ths_ns", "THS")]:
        m = re.search(label + r"\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if m:
            out[key] = _float(m.group(1))
    # Table fallback: find header with WNS and first numeric row after it.
    if out["wns_ns"] is None:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "WNS" in line and "TNS" in line:
                for j in range(i + 1, min(i + 8, len(lines))):
                    nums = re.findall(r"-?\d+\.\d+|-?\d+", lines[j])
                    if len(nums) >= 2:
                        out["wns_ns"] = _float(nums[0])
                        out["tns_ns"] = _float(nums[1])
                        if len(nums) >= 5:
                            out["whs_ns"] = _float(nums[4])
                        break
                break
    return out


def _parse_utilization(text: str) -> Dict[str, Optional[int]]:
    out = {"lut": None, "ff": None, "bram": None, "dsp": None}

    # Vivado util tables frequently use rows like:
    # | CLB LUTs | 34024 | ...
    # | Slice LUTs | 34024 | ...
    row_patterns = [
        ("lut", [r"\|\s*(?:CLB\s+LUTs|Slice\s+LUTs|LUT\s+as\s+Logic)\s*\|\s*([0-9,]+)\s*\|",
                 r"(?:CLB\s+LUTs|Slice\s+LUTs)\s*\|?\s*([0-9,]+)"]),
        ("ff", [r"\|\s*(?:CLB\s+Registers|Slice\s+Registers|Register\s+as\s+Flip\s+Flop)\s*\|\s*([0-9,]+)\s*\|",
                r"(?:CLB\s+Registers|Slice\s+Registers)\s*\|?\s*([0-9,]+)"]),
        ("bram", [r"\|\s*(?:Block\s+RAM\s+Tile|RAMB36/FIFO|RAMB18)\s*\|\s*([0-9,]+(?:\.[0-9]+)?)\s*\|",
                  r"Block\s+RAM\s+Tile\s*\|?\s*([0-9,]+(?:\.[0-9]+)?)"]),
        ("dsp", [r"\|\s*DSPs?\s*\|\s*([0-9,]+)\s*\|",
                 r"\|\s*DSP48E[12]?\s*\|\s*([0-9,]+)\s*\|",
                 r"DSP48E[12]?\s*\|?\s*([0-9,]+)"]),
    ]
    for key, pats in row_patterns:
        for pat in pats:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                out[key] = _int(m.group(1))
                break

    # Cell-usage fallback, useful for OOC reports that lack full board summary.
    if out["dsp"] is None:
        m = re.search(r"\|\s*DSP48E[12]?\s*\|\s*([0-9,]+)\s*\|", text, re.IGNORECASE)
        if m:
            out["dsp"] = _int(m.group(1))
    if out["ff"] is None:
        # Sum common FF cells if no summary row exists.
        total = 0
        found = False
        for cell in ("FDRE", "FDSE", "FDE", "FDCE", "FDPE"):
            m = re.search(r"\|\s*" + cell + r"\s*\|\s*([0-9,]+)\s*\|", text, re.IGNORECASE)
            if m:
                total += int(str(m.group(1)).replace(",", ""))
                found = True
        if found:
            out["ff"] = total
    if out["bram"] is None:
        # Count RAMB36 as 1 tile, RAMB18 as 0.5 tile if summary is unavailable.
        bram = 0.0
        found = False
        for cell, weight in (("RAMB36E1", 1.0), ("RAMB36E2", 1.0), ("RAMB18E1", 0.5), ("RAMB18E2", 0.5)):
            m = re.search(r"\|\s*" + cell + r"\s*\|\s*([0-9,]+)\s*\|", text, re.IGNORECASE)
            if m:
                bram += weight * int(str(m.group(1)).replace(",", ""))
                found = True
        if found:
            out["bram"] = int(round(bram))
    if out["lut"] is None:
        # Approximate LUT count from cell usage if summary not present.
        total = 0
        found = False
        for cell in ("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"):
            m = re.search(r"\|\s*" + cell + r"\s*\|\s*([0-9,]+)\s*\|", text, re.IGNORECASE)
            if m:
                total += int(str(m.group(1)).replace(",", ""))
                found = True
        if found:
            out["lut"] = total
    return out


def _parse_hls_latency_and_clock(hls_root: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "hls_latency_min_cycles": None,
        "hls_latency_max_cycles": None,
        "hls_interval_min_cycles": None,
        "hls_interval_max_cycles": None,
        "hls_target_clock_ns": None,
        "hls_estimated_clock_ns": None,
    }
    candidates = list(hls_root.rglob("*csynth.rpt")) + list(hls_root.rglob("csynth.xml")) + list(hls_root.rglob("*.xml"))
    for p in candidates:
        text = _read_text(p)
        if not text:
            continue
        # XML-style tags.
        tag_map = {
            "hls_latency_min_cycles": ["LatencyMin", "Best-caseLatency", "min_latency"],
            "hls_latency_max_cycles": ["LatencyMax", "Worst-caseLatency", "max_latency"],
            "hls_interval_min_cycles": ["IntervalMin", "min_interval"],
            "hls_interval_max_cycles": ["IntervalMax", "max_interval"],
            "hls_target_clock_ns": ["TargetClockPeriod"],
            "hls_estimated_clock_ns": ["EstimatedClockPeriod"],
        }
        for key, tags in tag_map.items():
            if out[key] is not None:
                continue
            for tag in tags:
                m = re.search(r"<" + re.escape(tag) + r">\s*([^<]+?)\s*</" + re.escape(tag) + r">", text)
                if m:
                    out[key] = _float(m.group(1))
                    break
        # RPT-style fallback.
        if out["hls_target_clock_ns"] is None:
            m = re.search(r"Target\s+clock\s+period\s*:?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            if m:
                out["hls_target_clock_ns"] = _float(m.group(1))
        if out["hls_estimated_clock_ns"] is None:
            m = re.search(r"Estimated\s+clock\s+period\s*:?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            if m:
                out["hls_estimated_clock_ns"] = _float(m.group(1))
        # Common latency table: Latency (cycles) min max ...
        if out["hls_latency_max_cycles"] is None and "Latency" in text:
            m = re.search(r"Latency\s*\(cycles\).*?\n.*?([0-9]+)\s*\|?\s*([0-9]+)", text, re.IGNORECASE | re.DOTALL)
            if m:
                out["hls_latency_min_cycles"] = _float(m.group(1))
                out["hls_latency_max_cycles"] = _float(m.group(2))
    return out


def _find_bridge_dir(design_dir: Path) -> Path:
    for rel in ("build/vivado_bridge", "vivado_bridge"):
        p = design_dir / rel
        if p.exists():
            return p
    # Return canonical expected path even if missing, so generated status is clear.
    return design_dir / "build" / "vivado_bridge"


def _design_dirs(exp: Path) -> List[Path]:
    art = exp / "artifacts"
    if not art.exists():
        return []
    return sorted([p for p in art.iterdir() if p.is_dir()])


def _choose_report(bridge: Path, kind: str) -> Optional[Path]:
    reports = bridge / "reports"
    names = {
        "power": ["power_impl.rpt", "power_synth.rpt", "*_power_routed.rpt", "*power*.rpt"],
        "util": ["utilization_impl.rpt", "utilization_synth.rpt", "*utilization*.rpt", "*util*.rpt"],
        "timing": ["timing_impl.rpt", "timing_synth.rpt", "*timing_summary*.rpt", "*timing*.rpt"],
    }[kind]
    for name in names:
        hits = sorted(reports.glob(name)) if reports.exists() else []
        if hits:
            return hits[0]
    # Search project run reports if copied report directory is incomplete.
    for name in names:
        hits = sorted((bridge / "project").rglob(name)) if (bridge / "project").exists() else []
        if hits:
            return hits[0]
    return None


def _bit_xsa_status(bridge: Path) -> Tuple[bool, bool]:
    bit_hits = list((bridge / "bitstream").glob("*.bit")) + list((bridge / "project").rglob("*.bit"))
    xsa_hits = list((bridge / "bitstream").glob("*.xsa")) + list((bridge / "project").rglob("*.xsa"))
    return bool(bit_hits), bool(xsa_hits)


def extract(exp: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for design_dir in _design_dirs(exp):
        design = design_dir.name
        bridge = _find_bridge_dir(design_dir)
        hls_root = design_dir / "build" / "hls"
        manifest = bridge / "vivado_bridge_manifest.json"
        manifest_data: Dict[str, Any] = {}
        if manifest.exists():
            try:
                manifest_data = json.loads(manifest.read_text())
            except Exception:
                manifest_data = {}

        component_xmls = list(bridge.rglob("component.xml")) if bridge.exists() else []
        bit_exists, xsa_exists = _bit_xsa_status(bridge)
        power_path = _choose_report(bridge, "power")
        util_path = _choose_report(bridge, "util")
        timing_path = _choose_report(bridge, "timing")

        power = _parse_power(_read_text(power_path)) if power_path else _parse_power("")
        util = _parse_utilization(_read_text(util_path)) if util_path else _parse_utilization("")
        timing = _parse_timing(_read_text(timing_path)) if timing_path else _parse_timing("")
        hls = _parse_hls_latency_and_clock(hls_root)

        latency_cycles = hls.get("hls_latency_max_cycles") or hls.get("hls_latency_min_cycles")
        clock_ns = hls.get("hls_target_clock_ns") or hls.get("hls_estimated_clock_ns")
        latency_seconds = None
        energy_j = None
        if latency_cycles is not None and clock_ns is not None:
            latency_seconds = float(latency_cycles) * float(clock_ns) * 1e-9
        if latency_seconds is not None and power.get("total_power_w") is not None:
            energy_j = float(power["total_power_w"]) * latency_seconds

        record = {
            "design": design,
            "vivado_bridge_generated": bridge.exists(),
            "hls_export_tcl_generated": (bridge / "scripts" / "export_hls_ip.tcl").exists(),
            "vivado_tcl_generated": (bridge / "scripts" / "run_vivado.tcl").exists(),
            "block_design_tcl_generated": (bridge / "scripts" / "create_bd.tcl").exists(),
            "board": manifest_data.get("board"),
            "part": manifest_data.get("part"),
            "hls_ip_exported": bool(component_xmls),
            "component_xml_exists": bool(component_xmls),
            "component_xml_count": len(component_xmls),
            "vivado_reports_present": bool(power_path or util_path or timing_path),
            "power_report": str(power_path) if power_path else None,
            "utilization_report": str(util_path) if util_path else None,
            "timing_report": str(timing_path) if timing_path else None,
            "bitstream_exists": bit_exists,
            "xsa_exists": xsa_exists,
            **power,
            **timing,
            **util,
            **hls,
            "hls_latency_seconds": latency_seconds,
            "estimated_energy_j": energy_j,
        }
        records.append(record)
    return records


def _write_outputs(exp: Path, records: List[Dict[str, Any]]) -> None:
    out_dir = exp / "vivado_bridge_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evidence.json"
    csv_path = out_dir / "evidence.csv"
    md_path = out_dir / "summary.md"

    payload = {"schema_version": 2, "experiment_dir": str(exp), "records": records}
    json_path.write_text(json.dumps(payload, indent=2))

    fields = [
        "design", "vivado_bridge_generated", "hls_ip_exported", "vivado_reports_present",
        "bitstream_exists", "xsa_exists", "total_power_w", "dynamic_power_w", "static_power_w",
        "wns_ns", "tns_ns", "whs_ns", "ths_ns", "lut", "ff", "bram", "dsp",
        "hls_latency_min_cycles", "hls_latency_max_cycles", "hls_target_clock_ns",
        "hls_estimated_clock_ns", "hls_latency_seconds", "estimated_energy_j",
        "power_report", "utilization_report", "timing_report",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    lines = ["# Sprint Vivado bridge evidence", ""]
    lines.append("| design | reports | bit | xsa | power_w | WNS_ns | LUT | FF | BRAM | DSP | HLS cycles | energy_j |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        lines.append(
            "| {design} | {reports} | {bit} | {xsa} | {power} | {wns} | {lut} | {ff} | {bram} | {dsp} | {cycles} | {energy} |".format(
                design=r.get("design"),
                reports=r.get("vivado_reports_present"),
                bit=r.get("bitstream_exists"),
                xsa=r.get("xsa_exists"),
                power=r.get("total_power_w"),
                wns=r.get("wns_ns"),
                lut=r.get("lut"),
                ff=r.get("ff"),
                bram=r.get("bram"),
                dsp=r.get("dsp"),
                cycles=r.get("hls_latency_max_cycles"),
                energy=r.get("estimated_energy_j"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path.read_text())
    print(f"[OK] Wrote {json_path}")
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {md_path}")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python scripts/extract_vivado_bridge_evidence.py <experiment_dir>", file=sys.stderr)
        return 2
    exp = Path(argv[1])
    records = extract(exp)
    _write_outputs(exp, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
