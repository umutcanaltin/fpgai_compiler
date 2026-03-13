from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
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


def _find_best_report_path(report_path: str | Path) -> Path:
    p = Path(report_path)
    parent = p.parent

    preferred = [
        parent / "deeplearn_csynth.xml",
        parent / "csynth.xml",
        parent / "deeplearn_csynth.rpt",
        parent / "csynth.rpt",
    ]

    for cand in preferred:
        if cand.exists():
            return cand

    if p.exists():
        return p

    return p


def _extract_xml_tag(text: str, tag: str, default=0):
    m = re.search(rf"<{tag}>([^<]+)</{tag}>", text)
    if not m:
        return default
    val = m.group(1)
    if isinstance(default, float):
        return _safe_float(val, default)
    return _safe_int(val, default)


def _extract_from_xml(xml_path: Path) -> Dict[str, float]:
    txt = xml_path.read_text(encoding="utf-8", errors="ignore")

    out = {
        "actual_lut": 0,
        "actual_ff": 0,
        "actual_dsp": 0,
        "actual_bram18": 0,
        "actual_latency_cycles": 0.0,
    }

    # Common HLS XML tags
    out["actual_lut"] = _extract_xml_tag(txt, "LUT", 0)
    out["actual_ff"] = _extract_xml_tag(txt, "FF", 0)
    out["actual_dsp"] = _extract_xml_tag(txt, "DSP", 0)
    out["actual_bram18"] = _extract_xml_tag(txt, "BRAM_18K", 0)

    # Latency tags vary slightly between reports
    lat_patterns = [
        r"<Average-caseLatency>([^<]+)</Average-caseLatency>",
        r"<Best-caseLatency>([^<]+)</Best-caseLatency>",
        r"<Worst-caseLatency>([^<]+)</Worst-caseLatency>",
        r"<Latency>([^<]+)</Latency>",
    ]
    for pat in lat_patterns:
        m = re.search(pat, txt)
        if m:
            out["actual_latency_cycles"] = _safe_float(m.group(1), 0.0)
            break

    return out


def _parse_total_row(txt: str) -> Dict[str, float]:
    out = {
        "actual_lut": 0,
        "actual_ff": 0,
        "actual_dsp": 0,
        "actual_bram18": 0,
        "actual_latency_cycles": 0.0,
    }

    m = re.search(
        r"\|Total\s*\|\s*([0-9,]+)\s*\|\s*([0-9,]+)\s*\|\s*([0-9,]+)\s*\|\s*([0-9,]+)\s*\|",
        txt,
        flags=re.IGNORECASE,
    )
    if m:
        out["actual_bram18"] = _safe_int(m.group(1), 0)
        out["actual_dsp"] = _safe_int(m.group(2), 0)
        out["actual_ff"] = _safe_int(m.group(3), 0)
        out["actual_lut"] = _safe_int(m.group(4), 0)

    return out


def _parse_latency(txt: str) -> float:
    patterns = [
        r"Latency.*?min\s*=\s*([0-9,]+)",
        r"\|\s*Latency\s*\|\s*([0-9,]+)\s*\|",
        r"Average-caseLatency.*?([0-9,]+)",
    ]
    for pat in patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return _safe_float(m.group(1), 0.0)
    return 0.0


def _count_dsp_bindings(txt: str) -> int:
    names = set()
    for m in re.finditer(r"^\|\s*([A-Za-z0-9_]+)\s*\|.*?\|\s*dsp_slice\s*\|", txt, flags=re.MULTILINE):
        names.add(m.group(1).strip())
    return len(names)


def _extract_from_rpt(rpt_path: Path) -> Dict[str, float]:
    txt = rpt_path.read_text(encoding="utf-8", errors="ignore")

    out = {
        "actual_lut": 0,
        "actual_ff": 0,
        "actual_dsp": 0,
        "actual_bram18": 0,
        "actual_latency_cycles": 0.0,
    }

    out.update(_parse_total_row(txt))
    out["actual_latency_cycles"] = _parse_latency(txt)

    if out["actual_dsp"] == 0:
        out["actual_dsp"] = _count_dsp_bindings(txt)

    return out


def parse_hls_csynth_report(report_path: str | Path) -> Dict[str, float]:
    p = _find_best_report_path(report_path)

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
            "HLS csynth report unavailable.\n"
            "====================================================="
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

    best_path = _find_best_report_path(csynth_report_path)
    actual = parse_hls_csynth_report(best_path)
    actual_latency_ms = (actual["actual_latency_cycles"] / (clock_mhz * 1e3)) if actual["actual_latency_cycles"] > 0 else 0.0

    payload = {
        "available": True,
        "csynth_report_path_requested": str(csynth_report_path),
        "csynth_report_path_used": str(best_path),
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
        f"Report   used={best_path.name}",
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