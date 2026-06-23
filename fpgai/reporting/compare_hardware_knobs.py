#!/usr/bin/env python3
"""Compare FPGAI hardware knob evidence.

Reads vivado_bridge_evidence/evidence.json and optional results.json from an
experiment directory, classifies design names into precision/parallel/pipeline
families, and writes a CSV + Markdown comparison table.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_records(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("records", "results", "designs"):
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def classify_design(name: str) -> Dict[str, str]:
    lower = name.lower()
    precision = ""
    parallel = ""
    pipeline = ""
    axis = "other"

    if "fx8" in lower:
        precision = "fx8_like"
        axis = "precision"
    elif "fx12" in lower:
        precision = "fx12_like"
        axis = "precision"
    elif "fx16" in lower:
        precision = "fx16_like"
        axis = "precision"

    m = re.search(r"parallel[_-]?(\d+)", lower)
    if m:
        parallel = m.group(1)
        axis = "parallel"

    if "conservative" in lower:
        pipeline = "conservative"
        axis = "pipeline"
    elif "balanced" in lower and "pipeline" in lower:
        pipeline = "balanced"
        axis = "pipeline"
    elif "aggressive" in lower:
        pipeline = "aggressive"
        axis = "pipeline"

    return {"axis": axis, "precision": precision, "parallel": parallel, "pipeline": pipeline}


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v) >= 1:
            return f"{v:.6g}"
        return f"{v:.6g}"
    return str(v)


def write_markdown(path: Path, rows: List[Dict[str, Any]], headers: List[str]) -> None:
    lines = []
    lines.append("# FPGAI hardware knob comparison")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(h, "")) for h in headers) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python scripts/compare_hardware_knobs.py <experiment_dir>")
        return 2

    exp = Path(argv[1])
    evidence_path = exp / "vivado_bridge_evidence" / "evidence.json"
    if not evidence_path.exists():
        raise SystemExit(f"Missing evidence file: {evidence_path}")

    evidence = as_records(load_json(evidence_path))

    rows: List[Dict[str, Any]] = []
    for r in evidence:
        name = str(r.get("design") or r.get("design_name") or "")
        cls = classify_design(name)
        row: Dict[str, Any] = {
            "design": name,
            "axis": cls["axis"],
            "precision": cls["precision"],
            "parallel": cls["parallel"],
            "pipeline": cls["pipeline"],
            "reports": r.get("vivado_reports_present") or r.get("reports"),
            "bit": r.get("bitstream_exists") or r.get("bit"),
            "xsa": r.get("xsa_exists") or r.get("xsa"),
            "power_w": r.get("total_power_w") if r.get("total_power_w") is not None else r.get("power_w"),
            "wns_ns": r.get("wns_ns") if r.get("wns_ns") is not None else r.get("WNS_ns"),
            "lut": r.get("lut") if r.get("lut") is not None else r.get("LUT"),
            "ff": r.get("ff") if r.get("ff") is not None else r.get("FF"),
            "bram": r.get("bram") if r.get("bram") is not None else r.get("BRAM"),
            "dsp": r.get("dsp") if r.get("dsp") is not None else r.get("DSP"),
            "hls_cycles": r.get("hls_latency_cycles") if r.get("hls_latency_cycles") is not None else r.get("HLS cycles"),
            "energy_j": r.get("estimated_energy_j") if r.get("estimated_energy_j") is not None else r.get("energy_j"),
        }
        rows.append(row)

    # Stable grouping: precision, parallel, pipeline, other; then design name.
    axis_order = {"precision": 0, "parallel": 1, "pipeline": 2, "other": 3}
    rows.sort(key=lambda x: (axis_order.get(str(x.get("axis")), 99), str(x.get("design", ""))))

    out_dir = exp / "hardware_knob_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        "design",
        "axis",
        "precision",
        "parallel",
        "pipeline",
        "reports",
        "bit",
        "xsa",
        "power_w",
        "wns_ns",
        "lut",
        "ff",
        "bram",
        "dsp",
        "hls_cycles",
        "energy_j",
    ]

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    md_path = out_dir / "comparison.md"
    write_markdown(md_path, rows, headers)

    print(md_path.read_text(encoding="utf-8"))
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
