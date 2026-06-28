from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
INPUT_CSV_CANDIDATES = [
    Path("sprint26_paper_prediction_codegen_results.csv"),
    ROOT / "prediction_codegen_results" / "results_recollected.csv",
]
OUT_DIR = ROOT / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "stage2_prediction_vs_hls_csynth.csv"
OUT_MD = OUT_DIR / "stage2_prediction_vs_hls_csynth.md"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", ""))
    except Exception:
        return None


def _pct_error(pred: Any, real: Any) -> str:
    p = _float(pred)
    r = _float(real)
    if p is None or r is None or r == 0:
        return ""
    return f"{((p - r) / r) * 100.0:.2f}%"


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return ""
    x = _float(v)
    if x is None:
        return str(v)
    if abs(x) >= 1000:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.6g}"


def _text_num(root: ET.Element, candidates: list[str]) -> float | None:
    # Match by suffix and exact tag because Vitis report XML varies by version.
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in candidates or any(tag.lower() == c.lower() for c in candidates):
            txt = (elem.text or "").strip()
            val = _float(txt)
            if val is not None:
                return val
    return None


def _parse_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()

    # Common Vitis HLS XML names across versions.
    latency_min = _text_num(root, ["MinLatency", "LatencyMin", "min_latency"])
    latency_max = _text_num(root, ["MaxLatency", "LatencyMax", "max_latency"])
    interval_min = _text_num(root, ["MinInterval", "IntervalMin", "min_interval"])
    interval_max = _text_num(root, ["MaxInterval", "IntervalMax", "max_interval"])

    lut = _text_num(root, ["LUT", "LUTs", "CLB LUTs"])
    ff = _text_num(root, ["FF", "FFs", "Register", "Registers"])
    dsp = _text_num(root, ["DSP", "DSP48E", "DSP48E2"])
    bram = _text_num(root, ["BRAM_18K", "BRAM18K", "BRAM", "BlockRAM"])
    uram = _text_num(root, ["URAM", "URAM288"])

    return {
        "hls_latency_min_cycles": latency_min,
        "hls_latency_max_cycles": latency_max,
        "hls_interval_min": interval_min,
        "hls_interval_max": interval_max,
        "hls_lut": lut,
        "hls_ff": ff,
        "hls_dsp": dsp,
        "hls_bram18": bram,
        "hls_uram": uram,
    }


def _extract_text_table_value(text: str, names: list[str]) -> float | None:
    # Conservative fallback for .rpt files.
    for name in names:
        m = re.search(rf"{re.escape(name)}\s*[:|]\s*([0-9,]+(?:\.[0-9]+)?)", text, re.I)
        if m:
            return _float(m.group(1))
    return None


def _parse_rpt(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "hls_latency_min_cycles": _extract_text_table_value(text, ["Latency min", "Min Latency"]),
        "hls_latency_max_cycles": _extract_text_table_value(text, ["Latency max", "Max Latency"]),
        "hls_lut": _extract_text_table_value(text, ["LUT", "CLB LUTs"]),
        "hls_ff": _extract_text_table_value(text, ["FF", "Registers"]),
        "hls_dsp": _extract_text_table_value(text, ["DSP", "DSP48E"]),
        "hls_bram18": _extract_text_table_value(text, ["BRAM_18K", "BRAM"]),
        "hls_uram": _extract_text_table_value(text, ["URAM"]),
    }


def _find_report(run_dir: Path) -> tuple[str, Path | None]:
    report_dir = run_dir / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"

    preferred = [
        report_dir / "deeplearn_csynth.xml",
        report_dir / "csynth.xml",
        report_dir / "deeplearn_csynth.rpt",
        report_dir / "csynth.rpt",
    ]
    for p in preferred:
        if p.exists():
            return "full_csynth", p

    design_size = [
        report_dir / "csynth_design_size.xml",
        report_dir / "csynth_design_size.rpt",
    ]
    for p in design_size:
        if p.exists():
            return "design_size_only", p

    if (run_dir / "hls_artifact_metadata.json").exists():
        return "hls_metadata_only", run_dir / "hls_artifact_metadata.json"

    return "missing", None


def _parse_report(status: str, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if status == "full_csynth" and path.suffix == ".xml":
        try:
            return _parse_xml(path)
        except Exception as e:
            return {"parse_error": str(e)}
    if status == "full_csynth" and path.suffix == ".rpt":
        return _parse_rpt(path)
    if status == "design_size_only":
        return {
            "hls_latency_min_cycles": "",
            "hls_latency_max_cycles": "",
            "hls_lut": "",
            "hls_ff": "",
            "hls_dsp": "",
            "hls_bram18": "",
            "hls_uram": "",
        }
    return {}


def main() -> int:
    input_csv = next((p for p in INPUT_CSV_CANDIDATES if p.exists()), None)
    if input_csv is None:
        raise FileNotFoundError(
            "No prediction/codegen CSV found. Checked: "
            + ", ".join(str(p) for p in INPUT_CSV_CANDIDATES)
        )
    print(f"[INFO] using input CSV: {input_csv}")
    rows = _read_rows(input_csv)
    out_rows: list[dict[str, Any]] = []

    for r in rows:
        run_dir = Path(r["run_dir"])
        hls_status, report_path = _find_report(run_dir)
        parsed = _parse_report(hls_status, report_path)

        hls_latency = parsed.get("hls_latency_max_cycles") or parsed.get("hls_latency_min_cycles")

        row = {
            "name": r["name"],
            "mode": r["mode"],
            "board": r["board"],
            "precision": r["precision"],
            "pe": r["pe"],
            "simd": r["simd"],
            "unroll": r["unroll"],
            "partition": r["partition"],
            "pipeline_style": r["pipeline_style"],
            "ii": r["ii"],
            "weight_storage": r["weight_storage"],
            "hls_status": hls_status,
            "hls_report": str(report_path) if report_path else "",
            "pred_lut": r.get("pred_lut", ""),
            "hls_lut": parsed.get("hls_lut", ""),
            "lut_error": _pct_error(r.get("pred_lut", ""), parsed.get("hls_lut", "")),
            "pred_ff": r.get("pred_ff", ""),
            "hls_ff": parsed.get("hls_ff", ""),
            "ff_error": _pct_error(r.get("pred_ff", ""), parsed.get("hls_ff", "")),
            "pred_dsp": r.get("pred_dsp", ""),
            "hls_dsp": parsed.get("hls_dsp", ""),
            "dsp_error": _pct_error(r.get("pred_dsp", ""), parsed.get("hls_dsp", "")),
            "pred_bram18": r.get("pred_bram18", ""),
            "hls_bram18": parsed.get("hls_bram18", ""),
            "bram18_error": _pct_error(r.get("pred_bram18", ""), parsed.get("hls_bram18", "")),
            "pred_cycles": r.get("pred_cycles", ""),
            "hls_latency_cycles": hls_latency if hls_latency is not None else "",
            "latency_error": _pct_error(r.get("pred_cycles", ""), hls_latency),
            "hls_interval_max": parsed.get("hls_interval_max", ""),
            "parse_error": parsed.get("parse_error", ""),
        }
        out_rows.append(row)

    fields = [
        "name",
        "mode",
        "board",
        "precision",
        "pe",
        "simd",
        "unroll",
        "partition",
        "pipeline_style",
        "ii",
        "weight_storage",
        "hls_status",
        "pred_lut",
        "hls_lut",
        "lut_error",
        "pred_ff",
        "hls_ff",
        "ff_error",
        "pred_dsp",
        "hls_dsp",
        "dsp_error",
        "pred_bram18",
        "hls_bram18",
        "bram18_error",
        "pred_cycles",
        "hls_latency_cycles",
        "latency_error",
        "hls_interval_max",
        "hls_report",
        "parse_error",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    md = [
        "# Stage 2 prediction vs HLS csynth table",
        "",
        "Source labels:",
        "- `full_csynth`: Vitis HLS csynth XML/RPT available.",
        "- `design_size_only`: Vitis generated design-size report only; no top-level csynth numbers parsed.",
        "- `hls_metadata_only`: FPGAI generated HLS metadata but no csynth report.",
        "",
        "| design | HLS status | precision | knobs | memory | pred LUT | HLS LUT | LUT err | pred DSP | HLS DSP | DSP err | pred BRAM18 | HLS BRAM18 | BRAM err | pred cycles | HLS cycles | lat err |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in out_rows:
        knobs = f"PE={r['pe']}, SIMD={r['simd']}, unroll={r['unroll']}, part={r['partition']}, II={r['ii']}"
        md.append(
            f"| `{r['name']}` | {r['hls_status']} | {r['precision']} | {knobs} | {r['weight_storage']} | "
            f"{_fmt(r['pred_lut'])} | {_fmt(r['hls_lut'])} | {r['lut_error']} | "
            f"{_fmt(r['pred_dsp'])} | {_fmt(r['hls_dsp'])} | {r['dsp_error']} | "
            f"{_fmt(r['pred_bram18'])} | {_fmt(r['hls_bram18'])} | {r['bram18_error']} | "
            f"{_fmt(r['pred_cycles'])} | {_fmt(r['hls_latency_cycles'])} | {r['latency_error']} |"
        )

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] wrote {OUT_CSV}")
    print(f"[OK] wrote {OUT_MD}")

    counts: dict[str, int] = {}
    for r in out_rows:
        counts[r["hls_status"]] = counts.get(r["hls_status"], 0) + 1
    print("[OK] hls_status_counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
