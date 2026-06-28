"""Build paper-ready result tables from parsed FPGAI numeric artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]], fields: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in fields) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def _tex_escape(value: Any) -> str:
    s = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _write_tex(path: Path, rows: list[dict[str, Any]], fields: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colspec = "l" * len(fields)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join(_tex_escape(f) for f in fields) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(" & ".join(_tex_escape(row.get(f, "")) for f in fields) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + _tex_escape(caption) + r"}")
    lines.append(r"\label{" + _tex_escape(label) + r"}")
    lines.append(r"\end{table}")
    lines.append("")
    path.write_text("\n".join(lines))


def _num(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", "None", "NA"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _ape(pred: float | None, actual: float | None) -> str:
    if pred is None or actual is None or actual == 0:
        return ""
    return f"{abs(pred - actual) / actual * 100.0:.2f}"


def _ratio(num: float | None, den: float | None) -> str:
    if num is None or den is None or den == 0:
        return ""
    return f"{num / den:.3f}"


def _artifact_coverage(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["paper_status"]] = statuses.get(r["paper_status"], 0) + 1

    return [
        {"artifact": "prediction numeric rows", "count": sum(bool(r.get("prediction_lut")) for r in rows), "total": len(rows)},
        {"artifact": "HLS csynth numeric rows", "count": sum(bool(r.get("hls_lut")) for r in rows), "total": len(rows)},
        {"artifact": "HLS latency rows", "count": sum(bool(r.get("hls_latency_worst_cycles")) for r in rows), "total": len(rows)},
        {"artifact": "Vivado implementation numeric rows", "count": sum(bool(r.get("vivado_lut")) for r in rows), "total": 5},
        {"artifact": "Vivado estimated power rows", "count": sum(bool(r.get("vivado_total_on_chip_power_w")) for r in rows), "total": 5},
        {"artifact": "Vivado bitstream/XSA ready", "count": statuses.get("vivado_impl_bitstream_ready", 0), "total": 6},
        {"artifact": "Vivado board-capacity rejected", "count": statuses.get("vivado_board_capacity_rejected", 0), "total": 6},
        {"artifact": "HLS-only designs", "count": statuses.get("hls_only", 0), "total": len(rows)},
    ]


def _prediction_vs_hls(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        pred_lut = _num(r, "prediction_lut")
        pred_dsp = _num(r, "prediction_dsp")
        pred_bram = _num(r, "prediction_bram18")
        hls_lut = _num(r, "hls_lut")
        hls_dsp = _num(r, "hls_dsp")
        hls_bram = _num(r, "hls_bram18")
        out.append({
            "design": r["design"],
            "group": r["group"],
            "mode": r["mode"],
            "pred_lut": r.get("prediction_lut", ""),
            "hls_lut": r.get("hls_lut", ""),
            "lut_ape_pct": _ape(pred_lut, hls_lut),
            "pred_dsp": r.get("prediction_dsp", ""),
            "hls_dsp": r.get("hls_dsp", ""),
            "dsp_ape_pct": _ape(pred_dsp, hls_dsp),
            "pred_bram18": r.get("prediction_bram18", ""),
            "hls_bram18": r.get("hls_bram18", ""),
            "bram18_ape_pct": _ape(pred_bram, hls_bram),
            "hls_latency_cycles": r.get("hls_latency_worst_cycles", ""),
        })
    return out


def _hls_vs_vivado(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not r.get("vivado_lut") and not r.get("vivado_failure_class"):
            continue
        hls_lut = _num(r, "hls_lut")
        viv_lut = _num(r, "vivado_lut")
        hls_dsp = _num(r, "hls_dsp")
        viv_dsp = _num(r, "vivado_dsp")
        hls_bram = _num(r, "hls_bram18")
        viv_bram = _num(r, "vivado_bram18")
        out.append({
            "design": r["design"],
            "paper_status": r["paper_status"],
            "hls_lut": r.get("hls_lut", ""),
            "vivado_lut": r.get("vivado_lut", ""),
            "vivado_over_hls_lut": _ratio(viv_lut, hls_lut),
            "hls_dsp": r.get("hls_dsp", ""),
            "vivado_dsp": r.get("vivado_dsp", ""),
            "vivado_over_hls_dsp": _ratio(viv_dsp, hls_dsp),
            "hls_bram18": r.get("hls_bram18", ""),
            "vivado_bram18": r.get("vivado_bram18", ""),
            "vivado_over_hls_bram18": _ratio(viv_bram, hls_bram),
            "wns_ns": r.get("vivado_wns_ns", ""),
            "vivado_power_w_est": r.get("vivado_total_on_chip_power_w", ""),
            "failure_class": r.get("vivado_failure_class", ""),
        })
    return out


def _knob_effects(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    baseline = next((r for r in rows if r["design"] == "kv260_baseline_safe_fx16"), None)
    if baseline is None:
        return []

    b_lut = _num(baseline, "hls_lut")
    b_dsp = _num(baseline, "hls_dsp")
    b_lat = _num(baseline, "hls_latency_worst_cycles")

    out = []
    for r in rows:
        if r["board"] != "kv260" or r["mode"] != "inference":
            continue
        hls_lut = _num(r, "hls_lut")
        hls_dsp = _num(r, "hls_dsp")
        hls_lat = _num(r, "hls_latency_worst_cycles")
        out.append({
            "design": r["design"],
            "group": r["group"],
            "hls_lut": r.get("hls_lut", ""),
            "lut_vs_baseline": _ratio(hls_lut, b_lut),
            "hls_dsp": r.get("hls_dsp", ""),
            "dsp_vs_baseline": _ratio(hls_dsp, b_dsp),
            "hls_latency_cycles": r.get("hls_latency_worst_cycles", ""),
            "latency_vs_baseline": _ratio(hls_lat, b_lat),
            "vivado_status": r.get("paper_status", ""),
        })
    return out


def _training_capacity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not r["design"].startswith("training_"):
            continue
        out.append({
            "design": r["design"],
            "precision_or_config": r["design"].replace("training_kv260_", ""),
            "prediction_lut": r.get("prediction_lut", ""),
            "hls_lut": r.get("hls_lut", ""),
            "hls_dsp": r.get("hls_dsp", ""),
            "hls_bram18": r.get("hls_bram18", ""),
            "hls_latency_cycles": r.get("hls_latency_worst_cycles", ""),
            "vivado_status": r.get("paper_status", ""),
            "failure_class": r.get("vivado_failure_class", ""),
            "required_slice_luts": r.get("failure_slice_lut_required", ""),
            "available_slice_luts": r.get("failure_slice_lut_available", ""),
            "slice_lut_util_pct": r.get("failure_slice_lut_util_pct", ""),
        })
    return out



def _design_short(design: str) -> str:
    replacements = [
        ("pynq_z2_baseline_safe_fx16", "PYNQ-Z2 base"),
        ("kv260_baseline_safe_fx16", "KV260 base"),
        ("kr260_baseline_safe_fx16", "KR260 base"),
        ("kv260_precision_fx16_6", "fx16"),
        ("kv260_precision_fx12_4", "fx12"),
        ("kv260_precision_fx8_3", "fx8"),
        ("kv260_parallel_x1", "x1"),
        ("kv260_parallel_x2", "x2"),
        ("kv260_parallel_x4", "x4"),
        ("kv260_parallel_x8", "x8"),
        ("kv260_pipeline_balanced_ii2", "pipe ii2"),
        ("kv260_pipeline_aggressive_ii1", "pipe ii1"),
        ("kv260_tiling_small", "tile S"),
        ("kv260_tiling_medium", "tile M"),
        ("kv260_tiling_large", "tile L"),
        ("kv260_memory_bram", "BRAM"),
        ("kv260_memory_uram", "URAM"),
        ("kv260_combined_aggressive_fx8", "combined fx8"),
        ("training_kv260_safe_fx16_6", "train safe"),
        ("training_kv260_aggressive_fx8_3", "train fx8"),
    ]
    for old, new in replacements:
        if design == old:
            return new
    return design.replace("kv260_", "").replace("_", " ")


def _fmt_int(value: str) -> str:
    n = _num({"x": value}, "x")
    if n is None:
        return ""
    return str(int(round(n)))


def _fmt_float(value: str, digits: int = 2) -> str:
    n = _num({"x": value}, "x")
    if n is None:
        return ""
    return f"{n:.{digits}f}"


def _status_short(status: str) -> str:
    return {
        "vivado_impl_bitstream_ready": "bit",
        "vivado_board_capacity_rejected": "cap reject",
        "hls_only": "HLS only",
    }.get(status, status)


def _arxiv_artifact_coverage(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], str]:
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["paper_status"]] = statuses.get(r["paper_status"], 0) + 1

    table = [
        {"Artifact": "Prediction", "Count": f"{sum(bool(r.get('prediction_lut')) for r in rows)}/20"},
        {"Artifact": "Vitis HLS csynth", "Count": f"{sum(bool(r.get('hls_lut')) for r in rows)}/20"},
        {"Artifact": "HLS latency", "Count": f"{sum(bool(r.get('hls_latency_worst_cycles')) for r in rows)}/20"},
        {"Artifact": "Vivado bitstream/XSA", "Count": f"{statuses.get('vivado_impl_bitstream_ready', 0)}/6"},
        {"Artifact": "Capacity rejection", "Count": f"{statuses.get('vivado_board_capacity_rejected', 0)}/6"},
        {"Artifact": "HLS-only designs", "Count": f"{statuses.get('hls_only', 0)}/20"},
    ]
    return table, ["Artifact", "Count"], "Artifact coverage for the FPGAI paper experiment matrix."


def _arxiv_prediction_vs_hls_summary(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], str]:
    selected = [
        "kv260_baseline_safe_fx16",
        "kv260_precision_fx16_6",
        "kv260_precision_fx12_4",
        "kv260_precision_fx8_3",
        "kv260_parallel_x1",
        "kv260_parallel_x8",
        "kv260_combined_aggressive_fx8",
        "training_kv260_safe_fx16_6",
        "training_kv260_aggressive_fx8_3",
    ]
    by_design = {r["design"]: r for r in rows}
    table = []
    for d in selected:
        r = by_design[d]
        table.append({
            "Design": _design_short(d),
            "Mode": "train" if r["mode"].startswith("training") else "infer",
            "Pred LUT": _fmt_int(r.get("prediction_lut", "")),
            "HLS LUT": _fmt_int(r.get("hls_lut", "")),
            "LUT err.": f"{_fmt_float(_ape(_num(r, 'prediction_lut'), _num(r, 'hls_lut')), 1)}%",
            "Pred DSP": _fmt_int(r.get("prediction_dsp", "")),
            "HLS DSP": _fmt_int(r.get("hls_dsp", "")),
            "Lat. cyc.": _fmt_int(r.get("hls_latency_worst_cycles", "")),
        })
    return table, ["Design", "Mode", "Pred LUT", "HLS LUT", "LUT err.", "Pred DSP", "HLS DSP", "Lat. cyc."], "Compact prediction-versus-HLS comparison for representative inference and training designs."


def _arxiv_vivado_subset(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], str]:
    table = []
    for r in rows:
        if not r.get("vivado_lut") and not r.get("vivado_failure_class"):
            continue
        table.append({
            "Design": _design_short(r["design"]),
            "Status": _status_short(r["paper_status"]),
            "HLS LUT": _fmt_int(r.get("hls_lut", "")),
            "Impl LUT": _fmt_int(r.get("vivado_lut", "")),
            "Impl DSP": _fmt_int(r.get("vivado_dsp", "")),
            "WNS": _fmt_float(r.get("vivado_wns_ns", ""), 2),
            "Pwr est.": _fmt_float(r.get("vivado_total_on_chip_power_w", ""), 2),
        })
    return table, ["Design", "Status", "HLS LUT", "Impl LUT", "Impl DSP", "WNS", "Pwr est."], "Vivado implementation subset. Power is Vivado-estimated, not measured board power."


def _arxiv_knob_summary(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], str]:
    selected = [
        "kv260_baseline_safe_fx16",
        "kv260_precision_fx16_6",
        "kv260_precision_fx12_4",
        "kv260_precision_fx8_3",
        "kv260_parallel_x1",
        "kv260_parallel_x2",
        "kv260_parallel_x4",
        "kv260_parallel_x8",
        "kv260_pipeline_aggressive_ii1",
        "kv260_memory_uram",
        "kv260_combined_aggressive_fx8",
    ]
    by_design = {r["design"]: r for r in rows}
    baseline = by_design["kv260_baseline_safe_fx16"]
    b_lut = _num(baseline, "hls_lut")
    b_dsp = _num(baseline, "hls_dsp")
    b_lat = _num(baseline, "hls_latency_worst_cycles")

    table = []
    for d in selected:
        r = by_design[d]
        table.append({
            "Design": _design_short(d),
            "Group": r["group"],
            "LUT x": _ratio(_num(r, "hls_lut"), b_lut),
            "DSP x": _ratio(_num(r, "hls_dsp"), b_dsp),
            "Lat. x": _ratio(_num(r, "hls_latency_worst_cycles"), b_lat),
            "Status": _status_short(r["paper_status"]),
        })
    return table, ["Design", "Group", "LUT x", "DSP x", "Lat. x", "Status"], "Effect of compiler knobs relative to the KV260 baseline using Vitis HLS results."


def _arxiv_training_capacity(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], str]:
    table = []
    for r in rows:
        if not r["design"].startswith("training_"):
            continue
        table.append({
            "Design": _design_short(r["design"]),
            "Pred LUT": _fmt_int(r.get("prediction_lut", "")),
            "HLS LUT": _fmt_int(r.get("hls_lut", "")),
            "HLS DSP": _fmt_int(r.get("hls_dsp", "")),
            "Status": _status_short(r.get("paper_status", "")),
            "Slice LUT util.": (r.get("failure_slice_lut_util_pct", "") + "%") if r.get("failure_slice_lut_util_pct") else "",
        })
    return table, ["Design", "Pred LUT", "HLS LUT", "HLS DSP", "Status", "Slice LUT util."], "Training designs. The safe training configuration is explicitly rejected by Vivado due to KV260 LUT capacity."


def _write_arxiv_tables(rows: list[dict[str, str]], out: Path) -> None:
    arxiv_specs = [
        ("table_arxiv_artifact_coverage", _arxiv_artifact_coverage(rows)),
        ("table_arxiv_prediction_vs_hls", _arxiv_prediction_vs_hls_summary(rows)),
        ("table_arxiv_vivado_subset", _arxiv_vivado_subset(rows)),
        ("table_arxiv_knob_summary", _arxiv_knob_summary(rows)),
        ("table_arxiv_training_capacity", _arxiv_training_capacity(rows)),
    ]
    for stem, (table_rows, fields, caption) in arxiv_specs:
        _write_csv(out / f"{stem}.csv", table_rows, fields)
        _write_md(out / f"{stem}.md", table_rows, fields, caption)
        _write_tex(out / f"{stem}.tex", table_rows, fields, caption, f"tab:{stem}")
        print(f"[OK] wrote {out / f'{stem}.csv'}")
        print(f"[OK] wrote {out / f'{stem}.md'}")
        print(f"[OK] wrote {out / f'{stem}.tex'}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="paper_results/parsed/paper_numeric_joined.csv")
    ap.add_argument("--out", default="paper_results/tables")
    args = ap.parse_args()

    rows = _read_rows(Path(args.input))
    out = Path(args.out)

    tables = [
        ("table_1_artifact_coverage", "Artifact coverage", _artifact_coverage(rows),
         ["artifact", "count", "total"]),
        ("table_2_prediction_vs_hls", "Prediction vs HLS", _prediction_vs_hls(rows),
         ["design", "group", "mode", "pred_lut", "hls_lut", "lut_ape_pct", "pred_dsp", "hls_dsp", "dsp_ape_pct", "pred_bram18", "hls_bram18", "bram18_ape_pct", "hls_latency_cycles"]),
        ("table_3_hls_vs_vivado", "HLS vs Vivado implementation", _hls_vs_vivado(rows),
         ["design", "paper_status", "hls_lut", "vivado_lut", "vivado_over_hls_lut", "hls_dsp", "vivado_dsp", "vivado_over_hls_dsp", "hls_bram18", "vivado_bram18", "vivado_over_hls_bram18", "wns_ns", "vivado_power_w_est", "failure_class"]),
        ("table_4_knob_effects", "Design knob effects", _knob_effects(rows),
         ["design", "group", "hls_lut", "lut_vs_baseline", "hls_dsp", "dsp_vs_baseline", "hls_latency_cycles", "latency_vs_baseline", "vivado_status"]),
        ("table_5_training_capacity", "Training capacity", _training_capacity(rows),
         ["design", "precision_or_config", "prediction_lut", "hls_lut", "hls_dsp", "hls_bram18", "hls_latency_cycles", "vivado_status", "failure_class", "required_slice_luts", "available_slice_luts", "slice_lut_util_pct"]),
    ]

    for stem, title, table_rows, fields in tables:
        _write_csv(out / f"{stem}.csv", table_rows, fields)
        _write_md(out / f"{stem}.md", table_rows, fields, title)
        _write_tex(out / f"{stem}.tex", table_rows, fields, title, f"tab:{stem}")
        print(f"[OK] wrote {out / f'{stem}.csv'}")
        print(f"[OK] wrote {out / f'{stem}.md'}")
        print(f"[OK] wrote {out / f'{stem}.tex'}")

    _write_arxiv_tables(rows, out)

    print("[SUMMARY]")
    print(f"input_rows={len(rows)}")
    for stem, _, table_rows, _ in tables:
        print(f"{stem}_rows={len(table_rows)}")
    print("arxiv_table_count=5")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
