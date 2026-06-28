from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
IN_CSV = ROOT / "paper_tables" / "stage2_prediction_vs_hls_csynth.csv"
OUT_CSV = ROOT / "paper_tables" / "stage2_full_csynth_subset.csv"
OUT_MD = ROOT / "paper_tables" / "stage2_full_csynth_subset.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v: str) -> float | None:
    try:
        if v == "":
            return None
        return float(v)
    except Exception:
        return None


def pct(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.2f}%"


def pct_change(value: str, baseline: str) -> str:
    v = fnum(value)
    b = fnum(baseline)
    if v is None or b is None or b == 0:
        return ""
    return f"{((v - b) / b) * 100.0:.2f}%"


def fmt(v: str) -> str:
    x = fnum(v)
    if x is None:
        return v
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.2f}"


def main() -> int:
    rows = [r for r in read_rows(IN_CSV) if r["hls_status"] == "full_csynth"]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_name = {r["name"]: r for r in rows}

    lines = [
        "# Stage 2 full-csynth HLS subset",
        "",
        "This table includes only designs with top-level Vitis HLS csynth reports.",
        "Prediction columns are pre-HLS estimates; HLS columns are parsed from csynth reports.",
        "",
        "| design | precision | PE/SIMD | II | pred LUT | HLS LUT | pred DSP | HLS DSP | pred BRAM18 | HLS BRAM18 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['precision']} | {r['pe']}/{r['simd']} | {r['ii']} | "
            f"{fmt(r['pred_lut'])} | {fmt(r['hls_lut'])} | "
            f"{fmt(r['pred_dsp'])} | {fmt(r['hls_dsp'])} | "
            f"{fmt(r['pred_bram18'])} | {fmt(r['hls_bram18'])} |"
        )

    # compact interpretation block
    fx16 = by_name.get("kv260_precision_fx16_6")
    fx12 = by_name.get("kv260_precision_fx12_4")
    fx8 = by_name.get("kv260_precision_fx8_3")
    x1 = by_name.get("kv260_parallel_x1")
    x8 = by_name.get("kv260_parallel_x8")
    ii2 = by_name.get("kv260_pipeline_balanced_ii2")
    ii1 = by_name.get("kv260_pipeline_aggressive_ii1")

    lines += ["", "## HLS-observed effects", ""]

    if fx16 and fx12 and fx8:
        lines.append(
            "- Precision: compared with `fx16_6`, HLS LUT changes are "
            f"`fx12_4`: {pct_change(fx12['hls_lut'], fx16['hls_lut'])}, "
            f"`fx8_3`: {pct_change(fx8['hls_lut'], fx16['hls_lut'])}; "
            f"HLS DSP changes are `fx12_4`: {pct_change(fx12['hls_dsp'], fx16['hls_dsp'])}, "
            f"`fx8_3`: {pct_change(fx8['hls_dsp'], fx16['hls_dsp'])}."
        )

    if x1 and x8:
        lines.append(
            "- Parallelism: compared with `x1`, `x8` changes HLS DSP by "
            f"{pct_change(x8['hls_dsp'], x1['hls_dsp'])}, HLS LUT by "
            f"{pct_change(x8['hls_lut'], x1['hls_lut'])}, and HLS BRAM18 by "
            f"{pct_change(x8['hls_bram18'], x1['hls_bram18'])}."
        )

    if ii2 and ii1:
        lines.append(
            "- Pipeline: compared with balanced `II=2`, aggressive `II=1` changes HLS LUT by "
            f"{pct_change(ii1['hls_lut'], ii2['hls_lut'])} and HLS DSP by "
            f"{pct_change(ii1['hls_dsp'], ii2['hls_dsp'])}."
        )

    lines += [
        "",
        "## Truth boundary",
        "",
        "- These are Vitis HLS csynth results, not Vivado implementation or board runtime results.",
        "- Prediction error remains visible in the full Stage 2 table and should be described as uncalibrated/analytical until calibrated against more HLS samples.",
        "- Training rows are excluded from this subset because they did not produce top-level csynth reports in this run.",
        "- URAM inference rows are included after the embedded-parameter BIND_STORAGE fix. Current embedded URAM mode is HLS-valid, but real URAM-resident parameter storage still requires runtime-loaded or uninitialized URAM buffers.",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {OUT_CSV}")
    print(f"[OK] wrote {OUT_MD}")
    print(f"[OK] full_csynth_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
