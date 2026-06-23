#!/usr/bin/env python3
"""Classify FPGAI Vivado bridge artifacts as feasible / resource_fail / timing_fail / tool_fail.

Usage:
  python -m fpgai.reporting.hardware_feasibility experiments/<run_dir>

Inputs:
  <run_dir>/vivado_bridge_evidence/evidence.json
  <run_dir>/vivado_bridge_run_evidence.json, if present
Outputs:
  <run_dir>/hardware_feasibility/feasibility.csv
  <run_dir>/hardware_feasibility/feasibility.md
  <run_dir>/hardware_feasibility/feasibility.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BOARD_LIMITS = {
    "pynq_z2": {"lut": 53200, "ff": 106400, "bram": 140, "dsp": 220},
    "xc7z020clg400-1": {"lut": 53200, "ff": 106400, "bram": 140, "dsp": 220},
}

FIELDS = [
    "design", "status", "reason", "board", "part", "bitstream_exists", "xsa_exists",
    "vivado_ok", "vivado_returncode", "power_w", "wns_ns", "lut", "lut_limit",
    "lut_util_pct", "ff", "ff_limit", "ff_util_pct", "bram", "bram_limit",
    "bram_util_pct", "dsp", "dsp_limit", "dsp_util_pct", "energy_j", "error_hint",
]


def _to_float(x: Any) -> Optional[float]:
    if x in (None, "", "None"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    f = _to_float(x)
    if f is None:
        return None
    return int(round(f))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _records_from_evidence(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "designs", "evidence"):
            if isinstance(data.get(key), list):
                return data[key]
        # maybe dict keyed by design
        if all(isinstance(v, dict) for v in data.values()):
            out = []
            for k, v in data.items():
                vv = dict(v)
                vv.setdefault("design", k)
                out.append(vv)
            return out
    raise SystemExit("Could not find artifact records in evidence.json")


def _run_map(run_path: Path) -> Dict[str, Dict[str, Any]]:
    p = run_path / "vivado_bridge_run_evidence.json"
    if not p.exists():
        return {}
    data = _load_json(p)
    records = data.get("records", data if isinstance(data, list) else [])
    out = {}
    for r in records:
        name = r.get("design") or r.get("design_name")
        if name:
            out[name] = r
    return out


def _read_tail(path: Optional[str], n: int = 40000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    txt = p.read_text(errors="ignore")
    return txt[-n:]


def _error_hint(run_rec: Dict[str, Any]) -> str:
    txt = "\n".join([
        str(run_rec.get("error", "")),
        _read_tail(run_rec.get("stdout_log")),
        _read_tail(run_rec.get("stderr_log")),
    ])
    if re.search(r"UTLZ-1|over-utilized|overutilized|requires more .* than are available", txt, re.I):
        return "Vivado DRC resource over-utilization"
    if re.search(r"place_design.*failed|Placer not run", txt, re.I):
        return "Vivado place_design failed"
    if re.search(r"route_design.*failed", txt, re.I):
        return "Vivado route_design failed"
    if re.search(r"timing", txt, re.I) and re.search(r"failed|violation", txt, re.I):
        return "Timing-related implementation failure"
    return ""


def _pct(value: Optional[int], limit: Optional[int]) -> Optional[float]:
    if value is None or not limit:
        return None
    return round(100.0 * value / limit, 2)


def classify_record(rec: Dict[str, Any], run_rec: Dict[str, Any], default_board: str = "pynq_z2") -> Dict[str, Any]:
    design = rec.get("design") or rec.get("design_name") or rec.get("name") or "unknown"
    board = rec.get("board") or run_rec.get("board") or default_board
    part = rec.get("part") or run_rec.get("part") or ""
    limits = BOARD_LIMITS.get(str(board), BOARD_LIMITS.get(str(part), BOARD_LIMITS["pynq_z2"]))

    lut = _to_int(rec.get("lut") or rec.get("LUT"))
    ff = _to_int(rec.get("ff") or rec.get("FF"))
    bram = _to_int(rec.get("bram") or rec.get("BRAM"))
    dsp = _to_int(rec.get("dsp") or rec.get("DSP"))
    wns = _to_float(rec.get("wns_ns") or rec.get("WNS_ns") or rec.get("WNS"))
    power = _to_float(rec.get("total_power_w") or rec.get("power_w"))
    energy = _to_float(rec.get("estimated_energy_j") or rec.get("energy_j"))
    bit = bool(rec.get("bitstream_exists") or rec.get("bit") or run_rec.get("bitstream"))
    xsa = bool(rec.get("xsa_exists") or rec.get("xsa") or run_rec.get("xsa"))
    vivado_ok = run_rec.get("vivado_ok")
    vivado_returncode = run_rec.get("vivado_returncode")

    over = []
    for key, val in (("lut", lut), ("ff", ff), ("bram", bram), ("dsp", dsp)):
        lim = limits.get(key)
        if val is not None and lim is not None and val > lim:
            over.append(f"{key.upper()} {val}>{lim}")

    hint = _error_hint(run_rec)
    status = "unknown"
    reason = ""
    if bit and xsa and (wns is None or wns >= 0):
        status = "pass"
        reason = "bitstream/xsa generated and timing closed or no timing violation parsed"
    elif over or "resource" in hint.lower() or "over-util" in hint.lower():
        status = "resource_fail"
        reason = "; ".join(over) if over else hint
    elif wns is not None and wns < 0:
        status = "timing_fail"
        reason = f"negative WNS {wns} ns"
    elif not bit or not xsa or vivado_ok is False or vivado_returncode not in (None, 0, "0"):
        status = "tool_fail"
        reason = hint or "Vivado did not produce bitstream/xsa"

    return {
        "design": design,
        "status": status,
        "reason": reason,
        "board": board,
        "part": part,
        "bitstream_exists": bit,
        "xsa_exists": xsa,
        "vivado_ok": vivado_ok,
        "vivado_returncode": vivado_returncode,
        "power_w": power,
        "wns_ns": wns,
        "lut": lut,
        "lut_limit": limits.get("lut"),
        "lut_util_pct": _pct(lut, limits.get("lut")),
        "ff": ff,
        "ff_limit": limits.get("ff"),
        "ff_util_pct": _pct(ff, limits.get("ff")),
        "bram": bram,
        "bram_limit": limits.get("bram"),
        "bram_util_pct": _pct(bram, limits.get("bram")),
        "dsp": dsp,
        "dsp_limit": limits.get("dsp"),
        "dsp_util_pct": _pct(dsp, limits.get("dsp")),
        "energy_j": energy,
        "error_hint": hint,
    }


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["design", "status", "reason", "bitstream_exists", "xsa_exists", "power_w", "wns_ns", "lut", "lut_util_pct", "ff", "bram", "dsp", "dsp_util_pct", "energy_j"]
    lines = ["# FPGAI hardware feasibility classification", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    run_path = Path(argv[1])
    ev_path = run_path / "vivado_bridge_evidence" / "evidence.json"
    if not ev_path.exists():
        raise SystemExit(f"Missing {ev_path}")
    records = _records_from_evidence(_load_json(ev_path))
    runs = _run_map(run_path)
    rows = []
    for rec in records:
        design = rec.get("design") or rec.get("design_name") or rec.get("name") or "unknown"
        rows.append(classify_record(rec, runs.get(design, {})))
    rows.sort(key=lambda r: r["design"])

    out = run_path / "hardware_feasibility"
    out.mkdir(parents=True, exist_ok=True)
    (out / "feasibility.json").write_text(json.dumps(rows, indent=2))
    with (out / "feasibility.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    write_md(out / "feasibility.md", rows)
    print((out / "feasibility.md").read_text())
    print(f"[OK] Wrote {out / 'feasibility.json'}")
    print(f"[OK] Wrote {out / 'feasibility.csv'}")
    print(f"[OK] Wrote {out / 'feasibility.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
