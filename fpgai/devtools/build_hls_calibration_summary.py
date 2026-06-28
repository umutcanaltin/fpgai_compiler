from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median

ROOT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
IN_CSV = ROOT / "paper_tables" / "stage2_prediction_vs_hls_csynth.csv"
OUT_CSV = ROOT / "paper_tables" / "stage2_hls_calibration_errors.csv"
OUT_MD = ROOT / "paper_tables" / "stage2_hls_calibration_summary.md"


def _num(x):
    if x is None:
        return None
    s = str(x).strip().replace("%", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _first(row, names, default=""):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _ape(pred, actual):
    pred = _num(pred)
    actual = _num(actual)
    if pred is None or actual is None or actual == 0:
        return None
    return abs(pred - actual) / abs(actual) * 100.0


def _signed(pred, actual):
    pred = _num(pred)
    actual = _num(actual)
    if pred is None or actual is None or actual == 0:
        return None
    return (pred - actual) / actual * 100.0


def _fmt(x):
    if x is None:
        return ""
    return f"{x:.2f}"


def _group(design):
    if design.startswith("training_"):
        return "training"
    if "precision" in design:
        return "precision"
    if "parallel" in design:
        return "parallelism"
    if "pipeline" in design:
        return "pipeline"
    if "tiling" in design:
        return "tiling"
    if "memory" in design or "combined" in design:
        return "memory_or_combined"
    return "baseline"


def _normalize_row(row):
    design = _first(row, ["design", "design_name", "name"])
    hls_status = _first(row, ["hls_status", "HLS status", "status"])
    precision = _first(row, ["precision", "precision_mode"])
    return {
        "design": design,
        "group": _group(design),
        "precision": precision,
        "hls_status": hls_status,
        "pred_lut": _first(row, ["pred_lut", "pred LUT", "predicted_lut"]),
        "hls_lut": _first(row, ["hls_lut", "HLS LUT", "hls LUT"]),
        "pred_dsp": _first(row, ["pred_dsp", "pred DSP", "predicted_dsp"]),
        "hls_dsp": _first(row, ["hls_dsp", "HLS DSP", "hls DSP"]),
        "pred_bram18": _first(row, ["pred_bram18", "pred BRAM18", "predicted_bram18"]),
        "hls_bram18": _first(row, ["hls_bram18", "HLS BRAM18", "hls BRAM18"]),
    }


if not IN_CSV.exists():
    raise SystemExit(f"Missing input CSV: {IN_CSV}")

rows = []
with IN_CSV.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print("[INFO] input columns:", reader.fieldnames)
    for raw in reader:
        row = _normalize_row(raw)
        if row["hls_status"] != "full_csynth":
            continue
        for res in ["lut", "dsp", "bram18"]:
            row[f"{res}_signed_error_pct"] = _fmt(_signed(row.get(f"pred_{res}"), row.get(f"hls_{res}")))
            row[f"{res}_abs_error_pct"] = _fmt(_ape(row.get(f"pred_{res}"), row.get(f"hls_{res}")))
        rows.append(row)

fields = [
    "design", "group", "precision", "hls_status",
    "pred_lut", "hls_lut", "lut_signed_error_pct", "lut_abs_error_pct",
    "pred_dsp", "hls_dsp", "dsp_signed_error_pct", "dsp_abs_error_pct",
    "pred_bram18", "hls_bram18", "bram18_signed_error_pct", "bram18_abs_error_pct",
]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def metric_summary(metric, selected):
    vals = [_num(r.get(f"{metric}_abs_error_pct")) for r in selected]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "max": max(vals),
    }


groups = [
    "all",
    "inference",
    "training",
    "baseline",
    "precision",
    "parallelism",
    "pipeline",
    "tiling",
    "memory_or_combined",
]

lines = [
    "# Stage 2 HLS calibration summary",
    "",
    "This summary compares raw pre-HLS analytical predictions with top-level Vitis HLS csynth reports.",
    "",
    "Important boundary: these are HLS csynth comparisons, not Vivado implementation or board-runtime measurements.",
    "",
    f"Full-csynth designs: {len(rows)}",
    "",
    "| group | n | LUT mean APE | LUT median APE | DSP mean APE | DSP median APE | BRAM18 mean APE | BRAM18 median APE |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]

for group in groups:
    if group == "all":
        selected = rows
    elif group == "inference":
        selected = [r for r in rows if not r["design"].startswith("training_")]
    else:
        selected = [r for r in rows if r["group"] == group]

    if not selected:
        continue

    lut = metric_summary("lut", selected)
    dsp = metric_summary("dsp", selected)
    bram = metric_summary("bram18", selected)

    lines.append(
        f"| {group} | {len(selected)} | "
        f"{_fmt(lut['mean']) if lut else ''} | {_fmt(lut['median']) if lut else ''} | "
        f"{_fmt(dsp['mean']) if dsp else ''} | {_fmt(dsp['median']) if dsp else ''} | "
        f"{_fmt(bram['mean']) if bram else ''} | {_fmt(bram['median']) if bram else ''} |"
    )

lines += [
    "",
    "## Largest absolute errors",
    "",
    "| design | metric | pred | HLS | signed error | absolute error |",
    "|---|---|---:|---:|---:|---:|",
]

err_rows = []
for r in rows:
    for metric, pred_key, hls_key, err_prefix in [
        ("LUT", "pred_lut", "hls_lut", "lut"),
        ("DSP", "pred_dsp", "hls_dsp", "dsp"),
        ("BRAM18", "pred_bram18", "hls_bram18", "bram18"),
    ]:
        abs_err = _num(r.get(f"{err_prefix}_abs_error_pct"))
        signed_err = _num(r.get(f"{err_prefix}_signed_error_pct"))
        if abs_err is not None:
            err_rows.append((abs_err, r, metric, pred_key, hls_key, signed_err))

for abs_err, r, metric, pred_key, hls_key, signed_err in sorted(err_rows, key=lambda x: x[0], reverse=True)[:12]:
    lines.append(
        f"| `{r['design']}` | {metric} | {r.get(pred_key, '')} | {r.get(hls_key, '')} | "
        f"{_fmt(signed_err)}% | {_fmt(abs_err)}% |"
    )

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"[OK] wrote {OUT_CSV}")
print(f"[OK] wrote {OUT_MD}")
print(f"[OK] rows={len(rows)}")
