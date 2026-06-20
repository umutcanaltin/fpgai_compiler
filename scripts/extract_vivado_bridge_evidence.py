#!/usr/bin/env python3
"""Extract Vivado bridge, IP, timing, utilization, power, and energy evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _first_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_timing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"timing_report_exists": False}
    txt = path.read_text(errors="ignore")
    return {
        "timing_report_exists": True,
        "wns_ns": _first_float(r"WNS\(ns\)\s+([-+]?\d+(?:\.\d+)?)", txt),
        "tns_ns": _first_float(r"TNS\(ns\)\s+([-+]?\d+(?:\.\d+)?)", txt),
        "whs_ns": _first_float(r"WHS\(ns\)\s+([-+]?\d+(?:\.\d+)?)", txt),
    }


def _parse_power(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"power_report_exists": False}
    txt = path.read_text(errors="ignore")
    total = _first_float(r"Total On-Chip Power \(W\)\s*\|?\s*([-+]?\d+(?:\.\d+)?)", txt)
    if total is None:
        total = _first_float(r"Total On-Chip Power.*?([-+]?\d+(?:\.\d+)?)\s*W", txt)
    return {
        "power_report_exists": True,
        "total_on_chip_power_w": total,
    }


def _parse_util(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"utilization_report_exists": False}
    txt = path.read_text(errors="ignore")
    # Vivado report formatting varies. Keep this deliberately conservative.
    def util(name: str) -> Optional[int]:
        m = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*(\d+)", txt, re.IGNORECASE)
        return int(m.group(1)) if m else None
    return {
        "utilization_report_exists": True,
        "lut": util("CLB LUTs") or util("Slice LUTs"),
        "ff": util("CLB Registers") or util("Slice Registers"),
        "bram": util("Block RAM Tile") or util("RAMB36/FIFO"),
        "dsp": util("DSPs") or util("DSP48E1"),
    }


def _iter_artifacts(exp: Path) -> Iterable[Path]:
    arts = exp / "artifacts"
    if arts.exists():
        for p in sorted(arts.iterdir()):
            if p.is_dir():
                yield p
    else:
        yield exp


def _component_xml_paths(bridge: Path) -> List[str]:
    paths = []
    for root in [bridge / "hls_ip", bridge.parent / "hls" / "fpgai_hls_proj" / "sol1" / "impl" / "ip"]:
        if root.exists():
            paths.extend([p.as_posix() for p in root.glob("**/component.xml")])
    return sorted(set(paths))


def _has_component_xml(bridge: Path) -> bool:
    return bool(_component_xml_paths(bridge))


def _evidence_for_artifact(artifact: Path) -> Dict[str, Any]:
    build = artifact / "build" if (artifact / "build").exists() else artifact
    bridge = build / "vivado_bridge"
    man = _load_json(bridge / "vivado_bridge_manifest.json") or {}
    reports = bridge / "reports"
    bit_dir = bridge / "bitstream"
    ip_dir = bridge / "hls_ip"

    timing = _parse_timing(reports / "timing_impl.rpt")
    if not timing.get("timing_report_exists"):
        timing = _parse_timing(reports / "timing_synth.rpt")
    util = _parse_util(reports / "utilization_impl.rpt")
    if not util.get("utilization_report_exists"):
        util = _parse_util(reports / "utilization_synth.rpt")
    power = _parse_power(reports / "power_impl.rpt")
    if not power.get("power_report_exists"):
        power = _parse_power(reports / "power_synth.rpt")

    total_power = power.get("total_on_chip_power_w")
    # Latency parsing will be strengthened after HLS synth report collection is standardized.
    hls_latency_s = None
    energy_j = total_power * hls_latency_s if total_power is not None and hls_latency_s is not None else None

    row = {
        "design": artifact.name,
        "vivado_bridge_generated": bool(man.get("vivado_bridge_generated") or bridge.exists()),
        "vivado_tcl_generated": (bridge / "scripts" / "run_vivado.tcl").exists(),
        "hls_export_tcl_generated": (bridge / "scripts" / "export_hls_ip.tcl").exists(),
        "block_design_tcl_generated": (bridge / "scripts" / "create_bd.tcl").exists(),
        "board": man.get("board"),
        "part": man.get("part"),
        "top_name": man.get("top_name"),
        "hls_ip_exported": _has_component_xml(bridge),
        "component_xml_exists": _has_component_xml(bridge),
        "component_xml_count": len(_component_xml_paths(bridge)),
        "component_xml_paths": _component_xml_paths(bridge),
        "hls_ip_export_run": man.get("hls_ip_export_run"),
        "hls_ip_export_error": man.get("hls_ip_export_error"),
        "vivado_reports_present": any(reports.glob("*.rpt")) if reports.exists() else False,
        "bitstream_exists": bool(list(bit_dir.glob("*.bit"))) if bit_dir.exists() else False,
        "xsa_exists": bool(list(bit_dir.glob("*.xsa"))) if bit_dir.exists() else False,
        "estimated_energy_j": energy_j,
        **timing,
        **util,
        **power,
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    args = ap.parse_args()
    exp = Path(args.experiment)
    rows = [_evidence_for_artifact(a) for a in _iter_artifacts(exp)]
    out_dir = exp / "vivado_bridge_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    print("# Sprint 14A/B Vivado bridge evidence")
    print()
    print("| design | vivado_bridge_generated | hls_export_tcl_generated | vivado_tcl_generated | block_design_tcl_generated | board | part | hls_ip_exported | component_xml_exists | component_xml_count | vivado_reports_present | bitstream_exists | xsa_exists | total_power_w | wns_ns | lut | ff | bram | dsp | estimated_energy_j |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            "| {design} | {vivado_bridge_generated} | {hls_export_tcl_generated} | {vivado_tcl_generated} | {block_design_tcl_generated} | {board} | {part} | {hls_ip_exported} | {component_xml_exists} | {component_xml_count} | {vivado_reports_present} | {bitstream_exists} | {xsa_exists} | {total_power_w} | {wns_ns} | {lut} | {ff} | {bram} | {dsp} | {energy} |".format(
                design=r.get("design", ""),
                vivado_bridge_generated=r.get("vivado_bridge_generated"),
                hls_export_tcl_generated=r.get("hls_export_tcl_generated"),
                vivado_tcl_generated=r.get("vivado_tcl_generated"),
                block_design_tcl_generated=r.get("block_design_tcl_generated"),
                board=r.get("board") or "",
                part=r.get("part") or "",
                hls_ip_exported=r.get("hls_ip_exported"),
                component_xml_exists=r.get("component_xml_exists"),
                component_xml_count=r.get("component_xml_count"),
                vivado_reports_present=r.get("vivado_reports_present"),
                bitstream_exists=r.get("bitstream_exists"),
                xsa_exists=r.get("xsa_exists"),
                total_power_w=r.get("total_on_chip_power_w"),
                wns_ns=r.get("wns_ns"),
                lut=r.get("lut"),
                ff=r.get("ff"),
                bram=r.get("bram"),
                dsp=r.get("dsp"),
                energy=r.get("estimated_energy_j"),
            )
        )
    print()
    print(f"[OK] Wrote {out_dir / 'evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
