#!/usr/bin/env python3
"""
Collect Vivado implementation reports from FPGAI experiment artifacts.

Vivado implementation report goals:
  - Do not rerun Vivado.
  - Summarize only existing artifacts.
  - Prefer Vivado report files over broad JSON crawling.
  - Avoid parsing HLS internal JSON/register files as design metrics.
  - Leave missing values blank/UNKNOWN rather than inventing results.

Outputs:
  <out>/vivado_impl_summary.csv
  <out>/vivado_impl_summary.json
  <out>/vivado_impl_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ImplRow:
    experiment: str
    design: str
    status: str = "UNKNOWN"
    failure_type: str = ""
    bitstream: bool = False
    xsa: bool = False
    wns_ns: float | None = None
    fmax_mhz: float | None = None
    safe_clock_mhz: float | None = None
    power_w: float | None = None
    lut: int | None = None
    ff: int | None = None
    bram: int | None = None
    dsp: int | None = None
    artifact_dir: str = ""
    notes: str = ""


NUM = r"[-+]?\d+(?:\.\d+)?"


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return None


def clean_int_token(s: str) -> int | None:
    s = s.strip().replace(",", "")
    if not re.fullmatch(r"[-+]?\d+", s):
        return None
    try:
        return int(s)
    except Exception:
        return None


def clean_float_token(s: str) -> float | None:
    s = s.strip().replace(",", "")
    if not re.fullmatch(NUM, s):
        return None
    try:
        return float(s)
    except Exception:
        return None


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def find_design_artifact_dirs(exp: Path) -> list[Path]:
    art = exp / "artifacts"
    if not art.exists():
        return []
    return sorted([p for p in art.iterdir() if p.is_dir()])


def has_bitstream(design_dir: Path) -> bool:
    # Search fairly broadly, but ignore hidden cache dirs.
    for p in design_dir.rglob("*.bit"):
        if any(part.startswith(".") for part in p.parts):
            continue
        return True
    return False


def has_xsa(design_dir: Path) -> bool:
    for p in design_dir.rglob("*.xsa"):
        if any(part.startswith(".") for part in p.parts):
            continue
        return True
    return False


def report_candidates(design_dir: Path, kind: str) -> list[Path]:
    files: list[Path] = []
    pats: list[str]
    if kind == "util":
        pats = ["*util*.rpt", "*utilization*.rpt", "*resource*.rpt"]
    elif kind == "timing":
        pats = ["*timing*.rpt", "*timing_summary*.rpt"]
    elif kind == "power":
        pats = ["*power*.rpt"]
    else:
        pats = ["*.rpt"]

    for pat in pats:
        files.extend(design_dir.rglob(pat))

    def score(p: Path) -> tuple[int, int, str]:
        s = str(p).lower()
        # Prefer Vivado bridge/impl reports, avoid HLS/autopilot internals.
        penalty = 0
        if "/hls/" in s or ".autopilot" in s or "csim" in s:
            penalty += 100
        if "vivado_bridge" in s or "/vivado" in s or "impl" in s or "synth" in s:
            penalty -= 20
        return (penalty, len(s), s)

    uniq = sorted(set(files), key=score)
    return uniq


def parse_vivado_utilization_rpt(path: Path) -> dict[str, int | None]:
    txt = read_text(path)
    out: dict[str, int | None] = {"lut": None, "ff": None, "bram": None, "dsp": None}

    # Vivado utilization tables are pipe-delimited. Used value is usually the 2nd column after the resource name.
    # Examples:
    # | CLB LUTs*          | 21738 | ... |
    # | CLB Registers      | 29374 | ... |
    # | Block RAM Tile     |    35 | ... |
    # | DSPs               |    64 | ... |
    patterns = {
        "lut": [r"\|\s*CLB LUTs\*?\s*\|\s*([\d,]+)\s*\|", r"\|\s*Slice LUTs\s*\|\s*([\d,]+)\s*\|"],
        "ff": [r"\|\s*CLB Registers\s*\|\s*([\d,]+)\s*\|", r"\|\s*Slice Registers\s*\|\s*([\d,]+)\s*\|"],
        "bram": [r"\|\s*Block RAM Tile\s*\|\s*([\d,]+)\s*\|", r"\|\s*RAMB(?:18|36)\s*\|\s*([\d,]+)\s*\|"],
        "dsp": [r"\|\s*DSPs\s*\|\s*([\d,]+)\s*\|", r"\|\s*DSP48E\w*\s*\|\s*([\d,]+)\s*\|"],
    }

    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, txt, re.IGNORECASE)
            if m:
                val = clean_int_token(m.group(1))
                if val is not None:
                    out[key] = val
                    break

    # Fallback for non-pipe report formats.
    fallback = {
        "lut": [r"CLB LUTs\*?\s+([\d,]+)", r"Slice LUTs\s+([\d,]+)"],
        "ff": [r"CLB Registers\s+([\d,]+)", r"Slice Registers\s+([\d,]+)"],
        "bram": [r"Block RAM Tile\s+([\d,]+)"],
        "dsp": [r"DSPs\s+([\d,]+)", r"DSP48E\w*\s+([\d,]+)"],
    }
    for key, pats in fallback.items():
        if out[key] is not None:
            continue
        for pat in pats:
            m = re.search(pat, txt, re.IGNORECASE)
            if m:
                val = clean_int_token(m.group(1))
                if val is not None:
                    out[key] = val
                    break

    return out


def parse_timing_rpt(path: Path) -> float | None:
    txt = read_text(path)
    # Common Vivado header/table forms.
    pats = [
        r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*([-+]?\d+(?:\.\d+)?)\s+",
        r"WNS\s*\(ns\)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        r"Worst Negative Slack[^\n:=]*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
        r"\bWNS\b[^\n]*?([-+]?\d+(?:\.\d+)?)",
    ]
    for pat in pats:
        m = re.search(pat, txt, re.IGNORECASE | re.DOTALL)
        if m:
            return clean_float_token(m.group(1))
    return None


def parse_power_rpt(path: Path) -> float | None:
    txt = read_text(path)
    pats = [
        r"Total On-Chip Power\s*\(W\)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
        r"\|\s*Total On-Chip Power\s*\(W\)\s*\|\s*([-+]?\d+(?:\.\d+)?)\s*\|",
        r"Total On-Chip Power[^\n]*?([-+]?\d+(?:\.\d+)?)\s*W",
    ]
    for pat in pats:
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            return clean_float_token(m.group(1))
    return None


def find_json_values(obj: Any, keys: set[str]) -> list[Any]:
    vals: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = str(k).lower()
            if kk in keys:
                vals.append(v)
            vals.extend(find_json_values(v, keys))
    elif isinstance(obj, list):
        for x in obj:
            vals.extend(find_json_values(x, keys))
    return vals


def json_number_from_files(design_dir: Path, key_sets: list[set[str]], prefer_names: tuple[str, ...] = ()) -> float | int | None:
    files = list(design_dir.rglob("*.json"))

    def score(p: Path) -> tuple[int, int, str]:
        s = str(p).lower()
        penalty = 0
        if ".autopilot" in s or "/hls/" in s or "layerwise_precision" in s or "sol1_data" in s:
            penalty += 100
        if any(name in s for name in prefer_names):
            penalty -= 50
        if any(name in s for name in ("vivado", "impl", "summary", "evidence", "result", "metrics")):
            penalty -= 10
        return (penalty, len(s), s)

    for p in sorted(files, key=score):
        if score(p)[0] >= 100:
            continue
        obj = read_json(p)
        if obj is None:
            continue
        for ks in key_sets:
            vals = find_json_values(obj, ks)
            for v in vals:
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    fv = clean_float_token(v)
                    if fv is not None:
                        return fv
    return None


def derive_fmax_and_safe_clock(wns_ns: float | None, target_clock_mhz: float = 200.0) -> tuple[float | None, float | None]:
    if wns_ns is None:
        return None, None
    period_ns = 1000.0 / target_clock_mhz
    critical_path_ns = period_ns - wns_ns
    if critical_path_ns <= 0:
        return None, None
    fmax = 1000.0 / critical_path_ns
    safe = fmax * 0.909  # roughly 10% guard-band
    return round(fmax, 3), round(safe, 3)


def collect_row(exp: Path, design_dir: Path) -> ImplRow:
    row = ImplRow(experiment=str(exp), design=design_dir.name, artifact_dir=str(design_dir))
    row.bitstream = has_bitstream(design_dir)
    row.xsa = has_xsa(design_dir)

    # Reports first.
    for p in report_candidates(design_dir, "util"):
        util = parse_vivado_utilization_rpt(p)
        if any(v is not None for v in util.values()):
            row.lut = util["lut"]
            row.ff = util["ff"]
            row.bram = util["bram"]
            row.dsp = util["dsp"]
            break

    for p in report_candidates(design_dir, "timing"):
        row.wns_ns = parse_timing_rpt(p)
        if row.wns_ns is not None:
            break

    for p in report_candidates(design_dir, "power"):
        row.power_w = parse_power_rpt(p)
        if row.power_w is not None:
            break

    # JSON fallback, but only from likely summary/evidence files, not broad HLS internals.
    if row.lut is None:
        v = json_number_from_files(design_dir, [{"lut", "luts", "vivado_lut", "lut_used", "used_lut"}], ("vivado", "summary", "evidence"))
        if isinstance(v, (int, float)) and v > 10:
            row.lut = int(v)
    if row.ff is None:
        v = json_number_from_files(design_dir, [{"ff", "ffs", "vivado_ff", "ff_used", "used_ff"}], ("vivado", "summary", "evidence"))
        if isinstance(v, (int, float)) and v > 10:
            row.ff = int(v)
    if row.bram is None:
        v = json_number_from_files(design_dir, [{"bram", "brams", "vivado_bram", "bram_used", "used_bram"}], ("vivado", "summary", "evidence"))
        if isinstance(v, (int, float)):
            row.bram = int(round(float(v)))
    if row.dsp is None:
        v = json_number_from_files(design_dir, [{"dsp", "dsps", "vivado_dsp", "dsp_used", "used_dsp"}], ("vivado", "summary", "evidence"))
        if isinstance(v, (int, float)):
            row.dsp = int(round(float(v)))
    if row.wns_ns is None:
        v = json_number_from_files(design_dir, [{"wns", "wns_ns", "worst_negative_slack", "timing_wns_ns"}], ("vivado", "summary", "evidence"))
        if isinstance(v, (int, float)):
            row.wns_ns = float(v)
    if row.power_w is None:
        v = json_number_from_files(design_dir, [{"power_w", "total_power_w", "total_on_chip_power_w", "vivado_power_w"}], ("vivado", "summary", "evidence"))
        if isinstance(v, (int, float)) and v > 0:
            row.power_w = float(v)

    row.fmax_mhz, row.safe_clock_mhz = derive_fmax_and_safe_clock(row.wns_ns)

    if row.wns_ns is not None and row.wns_ns < 0:
        row.status = "timing_fail"
        row.failure_type = "timing_fail"
    elif row.bitstream and row.xsa:
        row.status = "pass"
    elif row.bitstream:
        row.status = "partial_bitstream_only"
    else:
        # Heuristic for expected resource-fail design names.
        lname = row.design.lower()
        if "resource_fail" in lname or "parallel_3" in lname or "parallel_4" in lname:
            row.status = "resource_fail_or_no_bitstream"
            row.failure_type = "resource_or_impl_fail"
        else:
            row.status = "no_bitstream"
            row.failure_type = "missing_bitstream"

    missing = []
    for name in ("wns_ns", "lut", "ff", "bram", "dsp"):
        if getattr(row, name) is None:
            missing.append(name)
    if missing:
        row.notes = "missing: " + ",".join(missing)
    return row


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def write_outputs(rows: list[ImplRow], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "vivado_impl_summary.csv"
    json_path = out / "vivado_impl_summary.json"
    md_path = out / "vivado_impl_summary.md"

    fields = list(asdict(rows[0]).keys()) if rows else list(ImplRow("", "").__dict__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))

    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    bitstream_rows = sum(1 for r in rows if r.bitstream)
    xsa_rows = sum(1 for r in rows if r.xsa)
    neg_wns_rows = sum(1 for r in rows if r.wns_ns is not None and r.wns_ns < 0)
    resource_rows = sum(1 for r in rows if r.lut is not None and r.ff is not None and r.bram is not None and r.dsp is not None)
    timing_rows = sum(1 for r in rows if r.wns_ns is not None)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Vivado Implementation Report Summary\n\n")
        f.write("This table is generated from existing experiment artifacts. Missing values are left blank or UNKNOWN.\n\n")
        f.write(f"Total design rows: {len(rows)}\n\n")
        f.write(f"Bitstream rows: {bitstream_rows}\n\n")
        f.write(f"XSA rows: {xsa_rows}\n\n")
        f.write(f"Timing rows with WNS: {timing_rows}\n\n")
        f.write(f"Rows with complete resources: {resource_rows}\n\n")
        f.write(f"Negative-WNS rows: {neg_wns_rows}\n\n")
        f.write("| experiment | design | status | failure_type | bitstream | xsa | wns_ns | fmax_mhz | safe_clock_mhz | power_w | LUT | FF | BRAM | DSP | notes | artifact_dir |\n")
        f.write("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in rows:
            f.write(
                f"| {r.experiment} | {r.design} | {r.status} | {r.failure_type} | "
                f"{r.bitstream} | {r.xsa} | {fmt(r.wns_ns)} | {fmt(r.fmax_mhz)} | {fmt(r.safe_clock_mhz)} | "
                f"{fmt(r.power_w)} | {fmt(r.lut)} | {fmt(r.ff)} | {fmt(r.bram)} | {fmt(r.dsp)} | "
                f"{r.notes} | {r.artifact_dir} |\n"
            )
        f.write("\n## Safe claim\n\n")
        f.write("FPGAI extracts implementation-level timing, resource, power, bitstream, and XSA reports for evaluated designs when those artifacts are present.\n\n")
        f.write("## Limitation\n\n")
        f.write("This collector summarizes existing artifacts only. It does not rerun Vivado and it does not infer missing values as passing results.\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"rows={len(rows)} bitstream_rows={bitstream_rows} xsa_rows={xsa_rows} "
        f"timing_rows={timing_rows} resource_rows={resource_rows} negative_wns_rows={neg_wns_rows}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", help="Experiment directories to scan")
    ap.add_argument("--out", default="reports/vivado_impl_summary")
    args = ap.parse_args()

    rows: list[ImplRow] = []
    for exp_s in args.experiments:
        exp = Path(exp_s)
        if not exp.exists():
            print(f"WARNING: missing experiment directory: {exp}")
            continue
        for design_dir in find_design_artifact_dirs(exp):
            rows.append(collect_row(exp, design_dir))

    rows.sort(key=lambda r: (r.experiment, r.design))
    write_outputs(rows, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
