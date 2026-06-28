"""Build numeric paper-result CSVs from FPGAI prediction, HLS, and Vivado artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PAPER_DESIGN_ORDER = [
    "pynq_z2_baseline_safe_fx16",
    "kv260_baseline_safe_fx16",
    "kr260_baseline_safe_fx16",
    "kv260_precision_fx16_6",
    "kv260_precision_fx12_4",
    "kv260_precision_fx8_3",
    "kv260_parallel_x1",
    "kv260_parallel_x2",
    "kv260_parallel_x4",
    "kv260_parallel_x8",
    "kv260_pipeline_balanced_ii2",
    "kv260_pipeline_aggressive_ii1",
    "kv260_tiling_small",
    "kv260_tiling_medium",
    "kv260_tiling_large",
    "kv260_memory_bram",
    "kv260_memory_uram",
    "kv260_combined_aggressive_fx8",
    "training_kv260_safe_fx16_6",
    "training_kv260_aggressive_fx8_3",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).replace(",", "").strip()
        if text in {"", "NA", "---"}:
            return None
        if text.startswith("<"):
            text = text[1:]
        return float(text)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    f = _as_float(value)
    if f is None:
        return None
    return int(round(f))


def _infer_board(design: str) -> str:
    if design.startswith("pynq_z2"):
        return "pynq_z2"
    if design.startswith("kv260") or design.startswith("training_kv260"):
        return "kv260"
    if design.startswith("kr260"):
        return "kr260"
    return ""


def _infer_mode(design: str) -> str:
    return "training_on_device" if design.startswith("training_") else "inference"


def _infer_group(design: str) -> str:
    if design.startswith("training_"):
        return "training"
    if "baseline" in design:
        return "baseline"
    if "precision" in design:
        return "precision"
    if "parallel" in design:
        return "parallelism"
    if "pipeline" in design:
        return "pipeline"
    if "tiling" in design:
        return "tiling"
    if "memory" in design:
        return "memory"
    if "combined" in design:
        return "combined"
    return "other"


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        out.append(value)
        for child in value.values():
            out.extend(_walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(_walk_dicts(child))
    return out


def _resources_from_prediction(data: dict[str, Any]) -> dict[str, Any]:
    # Preferred current schema:
    # totals.predicted_lut / predicted_ff / predicted_dsp / predicted_bram18.
    totals = data.get("totals")
    if isinstance(totals, dict):
        return {
            "prediction_lut": _as_int(totals.get("predicted_lut")),
            "prediction_ff": _as_int(totals.get("predicted_ff")),
            "prediction_dsp": _as_int(totals.get("predicted_dsp")),
            "prediction_bram18": _as_float(totals.get("predicted_bram18")),
            "prediction_uram": _as_float(totals.get("predicted_uram")),
            "prediction_model": data.get("analytical_model") or data.get("model") or "",
        }

    # Fallback for older schemas.
    candidates = _walk_dicts(data)

    def score(src: dict[str, Any]) -> int:
        keys = {str(k).lower() for k in src}
        aliases = [
            {"predicted_lut", "lut", "luts"},
            {"predicted_ff", "ff", "ffs"},
            {"predicted_dsp", "dsp", "dsps"},
            {"predicted_bram18", "bram18", "bram_18k", "bram"},
            {"predicted_uram", "uram"},
        ]
        return sum(1 for group in aliases if keys & group)

    candidates = sorted(candidates, key=score, reverse=True)

    def pick(*names: str) -> Any:
        lowered = {n.lower() for n in names}
        for src in candidates:
            for key, value in src.items():
                if str(key).lower() in lowered:
                    return value
        return None

    return {
        "prediction_lut": _as_int(pick("predicted_lut", "lut", "LUT", "luts", "LUTs")),
        "prediction_ff": _as_int(pick("predicted_ff", "ff", "FF", "ffs", "FFs")),
        "prediction_dsp": _as_int(pick("predicted_dsp", "dsp", "DSP", "dsps", "DSPs")),
        "prediction_bram18": _as_float(pick("predicted_bram18", "bram18", "BRAM18", "bram_18k", "BRAM_18K", "bram")),
        "prediction_uram": _as_float(pick("predicted_uram", "uram", "URAM")),
        "prediction_model": data.get("analytical_model") or data.get("model") or "",
    }


def _timing_from_prediction(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction_cycles": _as_float(data.get("predicted_cycles")),
        "prediction_latency_ms": _as_float(data.get("predicted_latency_ms")),
        "prediction_throughput_fps": _as_float(data.get("predicted_throughput_fps")),
        "prediction_clock_mhz": _as_float(data.get("clock_mhz")),
    }


def _find_top_hls_xml(run_dir: Path) -> Path | None:
    report_dir = run_dir / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"
    for name in ("deeplearn_csynth.xml", "train_top_csynth.xml", "csynth.xml"):
        p = report_dir / name
        if p.exists():
            return p
    xmls = sorted(report_dir.glob("*_csynth.xml"))
    return xmls[0] if xmls else None


def _xml_text(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _parse_hls_xml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "hls_available": False,
            "hls_xml": "",
        }

    root = ET.parse(path).getroot()

    res = "AreaEstimates/Resources"
    lat = "PerformanceEstimates/SummaryOfOverallLatency"
    timing = "PerformanceEstimates/SummaryOfTimingAnalysis"

    return {
        "hls_available": True,
        "hls_xml": str(path),
        "hls_lut": _as_int(_xml_text(root, f"{res}/LUT")),
        "hls_ff": _as_int(_xml_text(root, f"{res}/FF")),
        "hls_dsp": _as_int(_xml_text(root, f"{res}/DSP")),
        "hls_bram18": _as_float(_xml_text(root, f"{res}/BRAM_18K")),
        "hls_uram": _as_float(_xml_text(root, f"{res}/URAM")),
        "hls_available_lut": _as_int(_xml_text(root, "AreaEstimates/AvailableResources/LUT")),
        "hls_available_ff": _as_int(_xml_text(root, "AreaEstimates/AvailableResources/FF")),
        "hls_available_dsp": _as_int(_xml_text(root, "AreaEstimates/AvailableResources/DSP")),
        "hls_available_bram18": _as_float(_xml_text(root, "AreaEstimates/AvailableResources/BRAM_18K")),
        "hls_available_uram": _as_float(_xml_text(root, "AreaEstimates/AvailableResources/URAM")),
        "hls_estimated_clock_period_ns": _as_float(_xml_text(root, f"{timing}/EstimatedClockPeriod")),
        "hls_latency_best_cycles": _as_float(_xml_text(root, f"{lat}/Best-caseLatency")),
        "hls_latency_avg_cycles": _as_float(_xml_text(root, f"{lat}/Average-caseLatency")),
        "hls_latency_worst_cycles": _as_float(_xml_text(root, f"{lat}/Worst-caseLatency")),
        "hls_interval_min": _as_float(_xml_text(root, f"{lat}/Interval-min")),
        "hls_interval_max": _as_float(_xml_text(root, f"{lat}/Interval-max")),
    }


def _row_value_from_vivado_table(text: str, label: str) -> dict[str, Any]:
    # Matches lines like:
    # | CLB LUTs | 4691 | 0 | 0 | 117120 | 4.01 |
    pattern = re.compile(
        r"\|\s*" + re.escape(label) + r"\s*\|\s*([<\d.,]+)\s*\|\s*[^|]*\|\s*[^|]*\|\s*([<\d.,]+)?\s*\|\s*([<\d.,]+)?\s*\|"
    )
    m = pattern.search(text)
    if not m:
        return {"used": None, "available": None, "util_pct": None}
    return {
        "used": _as_float(m.group(1)),
        "available": _as_float(m.group(2)),
        "util_pct": _as_float(m.group(3)),
    }


def _first_row_value(text: str, *labels: str) -> dict[str, Any]:
    for label in labels:
        row = _row_value_from_vivado_table(text, label)
        if row.get("used") is not None:
            return row
    return {"used": None, "available": None, "util_pct": None}


def _parse_vivado_utilization(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    clb_luts = _first_row_value(text, "CLB LUTs", "Slice LUTs")
    lut_logic = _first_row_value(text, "LUT as Logic")
    regs = _first_row_value(text, "CLB Registers", "Slice Registers")
    ramb18 = _first_row_value(text, "RAMB18")
    uram = _first_row_value(text, "URAM")
    dsps = _first_row_value(text, "DSPs", "DSP48E1", "DSP48E2")
    return {
        "vivado_lut": clb_luts["used"],
        "vivado_lut_available": clb_luts["available"],
        "vivado_lut_util_pct": clb_luts["util_pct"],
        "vivado_lut_logic": lut_logic["used"],
        "vivado_ff": regs["used"],
        "vivado_ff_available": regs["available"],
        "vivado_ff_util_pct": regs["util_pct"],
        "vivado_bram18": ramb18["used"],
        "vivado_bram18_available": ramb18["available"],
        "vivado_bram18_util_pct": ramb18["util_pct"],
        "vivado_uram": uram["used"],
        "vivado_uram_available": uram["available"],
        "vivado_uram_util_pct": uram["util_pct"],
        "vivado_dsp": dsps["used"],
        "vivado_dsp_available": dsps["available"],
        "vivado_dsp_util_pct": dsps["util_pct"],
    }


def _parse_vivado_timing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    # First numeric line after Design Timing Summary header is enough.
    m = re.search(
        r"\n\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+\d+\s+\d+\s+"
        r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
        text,
    )
    if not m:
        return {}
    return {
        "vivado_wns_ns": _as_float(m.group(1)),
        "vivado_tns_ns": _as_float(m.group(2)),
        "vivado_whs_ns": _as_float(m.group(3)),
        "vivado_ths_ns": _as_float(m.group(4)),
    }


def _parse_vivado_power(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")

    def power_value(label: str) -> float | None:
        m = re.search(r"\|\s*" + re.escape(label) + r"\s*\|\s*([<\d.]+)\s*\|", text)
        return _as_float(m.group(1)) if m else None

    return {
        "vivado_total_on_chip_power_w": power_value("Total On-Chip Power (W)"),
        "vivado_dynamic_power_w": power_value("Dynamic (W)"),
        "vivado_static_power_w": power_value("Device Static (W)"),
    }


def _parse_capacity_failure(run_dir: Path) -> dict[str, Any]:
    bridge = run_dir / "vivado_bridge"
    texts = []
    for root in [bridge / "logs", bridge / "project" / "fpgai_vivado.runs" / "impl_1", bridge]:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".log", ".rpt", ".txt", ".jou"}:
                    try:
                        texts.append(p.read_text(errors="ignore"))
                    except Exception:
                        pass
    text = "\n".join(texts)

    out: dict[str, Any] = {}
    m_logic = re.search(
        r"requires\s+(\d+)\s+of such cell types but only\s+(\d+)\s+compatible sites.*?LUT as Logic",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m_logic:
        m_logic = re.search(
            r"LUT as Logic over-utilized.*?requires\s+(\d+).*?only\s+(\d+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    if m_logic:
        req = int(m_logic.group(1))
        avail = int(m_logic.group(2))
        out["failure_lut_logic_required"] = req
        out["failure_lut_logic_available"] = avail
        out["failure_lut_logic_util_pct"] = round(100.0 * req / avail, 2) if avail else None

    m_slice = re.search(
        r"Slice LUTs over-utilized.*?requires\s+(\d+).*?only\s+(\d+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_slice:
        req = int(m_slice.group(1))
        avail = int(m_slice.group(2))
        out["failure_slice_lut_required"] = req
        out["failure_slice_lut_available"] = avail
        out["failure_slice_lut_util_pct"] = round(100.0 * req / avail, 2) if avail else None

    return out


def _read_artifact_index(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _common_row(design: str) -> dict[str, Any]:
    return {
        "design": design,
        "board": _infer_board(design),
        "mode": _infer_mode(design),
        "group": _infer_group(design),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
    ap.add_argument("--index", default="paper_results/index/paper_artifact_index.csv")
    ap.add_argument("--out", default="paper_results/parsed")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    index_rows = _read_artifact_index(Path(args.index))

    pred_rows: list[dict[str, Any]] = []
    hls_rows: list[dict[str, Any]] = []
    vivado_rows: list[dict[str, Any]] = []
    joined_rows: list[dict[str, Any]] = []

    by_design = {r["design"]: r for r in index_rows}
    ordered = [d for d in PAPER_DESIGN_ORDER if d in by_design] + [
        d for d in sorted(by_design) if d not in PAPER_DESIGN_ORDER
    ]

    for design in ordered:
        idx = by_design[design]
        run_dir = base / "runs" / design

        pred = _common_row(design)
        pred.update(_resources_from_prediction(_read_json(run_dir / "reports" / "resource_prediction.json")))
        pred.update(_timing_from_prediction(_read_json(run_dir / "reports" / "timing_prediction.json")))
        pred_rows.append(pred)

        hls = _common_row(design)
        hls.update(_parse_hls_xml(_find_top_hls_xml(run_dir)))
        hls_rows.append(hls)

        viv = _common_row(design)
        viv.update(
            {
                "vivado_ok": idx.get("vivado_ok"),
                "vivado_returncode": idx.get("vivado_returncode"),
                "vivado_failure_class": idx.get("vivado_failure_class"),
                "vivado_error": idx.get("vivado_error"),
                "bitstream_exists": idx.get("bitstream_exists"),
                "xsa_exists": idx.get("xsa_exists"),
            }
        )
        reports = run_dir / "vivado_bridge" / "reports"
        viv.update(_parse_vivado_utilization(reports / "utilization_impl.rpt"))
        viv.update(_parse_vivado_timing(reports / "timing_impl.rpt"))
        viv.update(_parse_vivado_power(reports / "power_impl.rpt"))
        viv.update(_parse_capacity_failure(run_dir))
        vivado_rows.append(viv)

        joined = {}
        joined.update(idx)
        joined.update(pred)
        joined.update(hls)
        joined.update(viv)
        joined_rows.append(joined)

    pred_fields = [
        "design", "board", "mode", "group",
        "prediction_model", "prediction_lut", "prediction_ff", "prediction_dsp", "prediction_bram18", "prediction_uram",
        "prediction_cycles", "prediction_latency_ms", "prediction_throughput_fps", "prediction_clock_mhz",
    ]
    hls_fields = [
        "design", "board", "mode", "group", "hls_available", "hls_xml",
        "hls_lut", "hls_ff", "hls_dsp", "hls_bram18", "hls_uram",
        "hls_available_lut", "hls_available_ff", "hls_available_dsp", "hls_available_bram18", "hls_available_uram",
        "hls_estimated_clock_period_ns", "hls_latency_best_cycles", "hls_latency_avg_cycles", "hls_latency_worst_cycles",
        "hls_interval_min", "hls_interval_max",
    ]
    vivado_fields = [
        "design", "board", "mode", "group",
        "vivado_ok", "vivado_returncode", "vivado_failure_class", "vivado_error", "bitstream_exists", "xsa_exists",
        "vivado_lut", "vivado_lut_logic", "vivado_lut_available", "vivado_lut_util_pct",
        "vivado_ff", "vivado_ff_available", "vivado_ff_util_pct",
        "vivado_dsp", "vivado_dsp_available", "vivado_dsp_util_pct",
        "vivado_bram18", "vivado_bram18_available", "vivado_bram18_util_pct",
        "vivado_uram", "vivado_uram_available", "vivado_uram_util_pct",
        "vivado_wns_ns", "vivado_tns_ns", "vivado_whs_ns", "vivado_ths_ns",
        "vivado_total_on_chip_power_w", "vivado_dynamic_power_w", "vivado_static_power_w",
        "failure_lut_logic_required", "failure_lut_logic_available", "failure_lut_logic_util_pct",
        "failure_slice_lut_required", "failure_slice_lut_available", "failure_slice_lut_util_pct",
    ]

    joined_fields = list(dict.fromkeys(list(index_rows[0].keys()) + pred_fields + hls_fields + vivado_fields))

    _write_csv(out / "prediction_numeric_results.csv", pred_rows, pred_fields)
    _write_csv(out / "hls_numeric_results.csv", hls_rows, hls_fields)
    _write_csv(out / "vivado_numeric_results.csv", vivado_rows, vivado_fields)
    _write_csv(out / "paper_numeric_joined.csv", joined_rows, joined_fields)

    print(f"[OK] wrote {out / 'prediction_numeric_results.csv'}")
    print(f"[OK] wrote {out / 'hls_numeric_results.csv'}")
    print(f"[OK] wrote {out / 'vivado_numeric_results.csv'}")
    print(f"[OK] wrote {out / 'paper_numeric_joined.csv'}")

    print("[SUMMARY]")
    print(f"designs={len(joined_rows)}")
    print(f"hls_numeric_rows={sum(1 for r in hls_rows if r.get('hls_available'))}")
    print(f"vivado_numeric_rows={sum(1 for r in vivado_rows if r.get('vivado_lut') not in (None, ''))}")
    print(f"vivado_power_rows={sum(1 for r in vivado_rows if r.get('vivado_total_on_chip_power_w') not in (None, ''))}")
    print(f"capacity_failure_rows={sum(1 for r in vivado_rows if r.get('failure_slice_lut_required') not in (None, ''))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
