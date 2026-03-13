from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import re
import xml.etree.ElementTree as ET


def _safe_int(x, default=0):
    try:
        s = str(x).replace(",", "").strip()
        return int(float(s))
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        s = str(x).replace(",", "").strip()
        return float(s)
    except Exception:
        return default


def _extract_from_xml(xml_path: Path) -> Dict[str, float]:
    root = ET.parse(xml_path).getroot()
    text = ET.tostring(root, encoding="unicode")

    out = {
        "actual_lut": 0,
        "actual_ff": 0,
        "actual_dsp": 0,
        "actual_bram18": 0,
        "actual_latency_cycles": 0.0,
    }

    patterns = {
        "actual_lut": r"<LUT>([^<]+)</LUT>",
        "actual_ff": r"<FF>([^<]+)</FF>",
        "actual_dsp": r"<DSP>([^<]+)</DSP>",
        "actual_bram18": r"<BRAM_18K>([^<]+)</BRAM_18K>",
        "actual_latency_cycles": r"<Average-caseLatency>([^<]+)</Average-caseLatency>",
    }

    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            if "latency" in k:
                out[k] = _safe_float(m.group(1), 0.0)
            else:
                out[k] = _safe_int(m.group(1), 0)

    return out


def _extract_from_rpt(rpt_path: Path) -> Dict[str, float]:
    txt = rpt_path.read_text(encoding="utf-8", errors="ignore")

    out = {
        "actual_lut": 0,
        "actual_ff": 0,
        "actual_dsp": 0,
        "actual_bram18": 0,
        "actual_latency_cycles": 0.0,
    }

    patterns = [
        ("actual_lut", r"\|\s*LUT\s*\|\s*([0-9,]+)\s*\|"),
        ("actual_ff", r"\|\s*FF\s*\|\s*([0-9,]+)\s*\|"),
        ("actual_dsp", r"\|\s*DSP\s*\|\s*([0-9,]+)\s*\|"),
        ("actual_bram18", r"\|\s*BRAM_18K\s*\|\s*([0-9,]+)\s*\|"),
        ("actual_latency_cycles", r"Latency.*?min\s*=\s*([0-9,]+)"),
    ]

    for k, pat in patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE | re.DOTALL)
        if m:
            if "latency" in k:
                out[k] = _safe_float(m.group(1), 0.0)
            else:
                out[k] = _safe_int(m.group(1), 0)

    return out


def parse_hls_csynth_report(report_path: str | Path) -> Dict[str, float]:
    p = Path(report_path)
    if not p.exists():
        return {
            "actual_lut": 0,
            "actual_ff": 0,
            "actual_dsp": 0,
            "actual_bram18": 0,
            "actual_latency_cycles": 0.0,
        }

    if p.suffix.lower() == ".xml":
        return _extract_from_xml(p)
    return _extract_from_rpt(p)


def _rel_err(pred, act):
    pred = float(pred)
    act = float(act)
    denom = max(1.0, abs(act))
    return (pred - act) / denom


@dataclass(frozen=True)
class EstimateVsHlsResult:
    out_dir: Path
    results_json: Path
    summary_txt: Path
    terminal_summary: str


def run_estimate_vs_hls_compare(
    *,
    out_dir: str | Path,
    design_space_summary: Dict[str, float],
    csynth_report_path: str | Path | None,
    clock_mhz: float,
) -> EstimateVsHlsResult:
    out_dir = Path(out_dir).resolve()
    cdir = out_dir / "estimate_vs_hls"
    cdir.mkdir(parents=True, exist_ok=True)

    if not csynth_report_path:
        terminal = (
            "=============== FPGAI Estimate vs HLS ===============\n"
            "HLS csynth report unavailable. Current run appears to be csim-only,\n"
            "so only analytical estimates are available.\n"
            "======================================================"
        )
        summary_txt = cdir / "summary.txt"
        results_json = cdir / "results.json"
        payload = {
            "available": False,
            "reason": "No csynth report path available",
            "estimated": design_space_summary,
            "actual": None,
        }
        results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary_txt.write_text(terminal + "\n", encoding="utf-8")
        return EstimateVsHlsResult(
            out_dir=cdir,
            results_json=results_json,
            summary_txt=summary_txt,
            terminal_summary=terminal,
        )

    actual = parse_hls_csynth_report(csynth_report_path)
    actual_latency_ms = (actual["actual_latency_cycles"] / (clock_mhz * 1e3)) if actual["actual_latency_cycles"] > 0 else 0.0

    payload = {
        "available": True,
        "estimated": design_space_summary,
        "actual": {
            **actual,
            "actual_latency_ms": actual_latency_ms,
        },
        "relative_error": {
            "lut": _rel_err(design_space_summary.get("predicted_lut", 0), actual["actual_lut"]),
            "ff": _rel_err(design_space_summary.get("predicted_ff", 0), actual["actual_ff"]),
            "dsp": _rel_err(design_space_summary.get("predicted_dsp", 0), actual["actual_dsp"]),
            "bram18": _rel_err(design_space_summary.get("predicted_bram18", 0), actual["actual_bram18"]),
            "latency_ms": _rel_err(design_space_summary.get("predicted_latency_ms", 0.0), actual_latency_ms),
        },
    }

    terminal = "\n".join([
        "=============== FPGAI Estimate vs HLS ===============",
        f"LUT      pred={design_space_summary.get('predicted_lut', 0)}  actual={actual['actual_lut']}",
        f"FF       pred={design_space_summary.get('predicted_ff', 0)}  actual={actual['actual_ff']}",
        f"DSP      pred={design_space_summary.get('predicted_dsp', 0)}  actual={actual['actual_dsp']}",
        f"BRAM18   pred={design_space_summary.get('predicted_bram18', 0)}  actual={actual['actual_bram18']}",
        f"Latency  pred={design_space_summary.get('predicted_latency_ms', 0.0):.4f} ms  actual={actual_latency_ms:.4f} ms",
        "=====================================================",
    ])

    results_json = cdir / "results.json"
    summary_txt = cdir / "summary.txt"
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_txt.write_text(terminal + "\n", encoding="utf-8")

    return EstimateVsHlsResult(
        out_dir=cdir,
        results_json=results_json,
        summary_txt=summary_txt,
        terminal_summary=terminal,
    )