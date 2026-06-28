from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
INPUT_CSV = Path("sprint26_paper_prediction_codegen_results.csv")
OUT_DIR = ROOT / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_CSV = OUT_DIR / "stage1_prediction_codegen_table.csv"
TABLE_MD = OUT_DIR / "stage1_prediction_codegen_table.md"
EFFECTS_CSV = OUT_DIR / "stage1_decision_effects.csv"
EFFECTS_MD = OUT_DIR / "stage1_decision_effects.md"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _paper_group(name: str) -> str:
    if "baseline_safe" in name:
        return "board_baseline"
    if "_precision_" in name:
        return "precision"
    if "_parallel_x" in name:
        return "parallelism"
    if "_pipeline_" in name:
        return "pipeline"
    if "_tiling_" in name:
        return "tiling"
    if "_memory_" in name:
        return "memory"
    if "combined_aggressive" in name:
        return "combined"
    if name.startswith("training_"):
        return "training"
    return "other"


def _float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


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


def _pct_change(value: Any, baseline: Any) -> str:
    v = _float(value)
    b = _float(baseline)
    if v is None or b is None or b == 0:
        return ""
    return f"{((v - b) / b) * 100.0:.2f}%"


def _find(rows: list[dict[str, str]], name: str) -> dict[str, str] | None:
    for r in rows:
        if r.get("name") == name:
            return r
    return None


def _has_csynth(row: dict[str, str]) -> str:
    run_dir = Path(row["run_dir"])
    return str(any(run_dir.glob("hls/**/csynth.rpt")))


def main() -> int:
    raw_rows = _read_rows(INPUT_CSV)

    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        row = {
            "name": r["name"],
            "group": _paper_group(r["name"]),
            "source": "prediction+codegen+hls_artifact_presence",
            "mode": r["mode"],
            "board": r["board"],
            "precision": r["precision"],
            "pe": r["pe"],
            "simd": r["simd"],
            "unroll": r["unroll"],
            "partition": r["partition"],
            "pipeline_style": r["pipeline_style"],
            "ii": r["ii"],
            "dense_tile": r["dense_tile"],
            "conv_tile": r["conv_tile"],
            "weight_storage": r["weight_storage"],
            "double_buffer": r["double_buffer"],
            "board_fit": r["board_fit_status"],
            "contract_knobs": r["contract_knob_count"],
            "pred_lut": r.get("pred_lut", ""),
            "pred_ff": r.get("pred_ff", ""),
            "pred_dsp": r.get("pred_dsp", ""),
            "pred_bram18": r.get("pred_bram18", ""),
            "pred_cycles": r.get("pred_cycles", ""),
            "pred_latency_ms": r.get("pred_latency_ms", ""),
            "pred_throughput_fps": r.get("pred_throughput_fps", ""),
            "clock_mhz": r.get("pred_clock_mhz", ""),
            "hls_csynth_present": _has_csynth(r),
            "run_dir": r["run_dir"],
        }
        rows.append(row)

    fields = [
        "name",
        "group",
        "source",
        "mode",
        "board",
        "precision",
        "pe",
        "simd",
        "unroll",
        "partition",
        "pipeline_style",
        "ii",
        "dense_tile",
        "conv_tile",
        "weight_storage",
        "double_buffer",
        "board_fit",
        "contract_knobs",
        "pred_lut",
        "pred_ff",
        "pred_dsp",
        "pred_bram18",
        "pred_cycles",
        "pred_latency_ms",
        "pred_throughput_fps",
        "clock_mhz",
        "hls_csynth_present",
        "run_dir",
    ]

    with TABLE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Stage 1 prediction/codegen/HLS-presence table",
        "",
        "Source label: `prediction+codegen+hls_artifact_presence`.",
        "This table reports prediction artifacts and verifies generated HLS/csynth artifact presence. Exact parsed HLS numbers are handled in the next stage.",
        "",
        "| design | group | board | precision | knobs | memory | fit | LUT | FF | DSP | BRAM18 | cycles | latency ms | throughput fps | HLS report |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in rows:
        knobs = f"PE={r['pe']}, SIMD={r['simd']}, unroll={r['unroll']}, part={r['partition']}, II={r['ii']}"
        md.append(
            f"| `{r['name']}` | {r['group']} | {r['board']} | {r['precision']} | {knobs} | "
            f"{r['weight_storage']} | {r['board_fit']} | {_fmt(r['pred_lut'])} | {_fmt(r['pred_ff'])} | "
            f"{_fmt(r['pred_dsp'])} | {_fmt(r['pred_bram18'])} | {_fmt(r['pred_cycles'])} | "
            f"{_fmt(r['pred_latency_ms'])} | {_fmt(r['pred_throughput_fps'])} | {r['hls_csynth_present']} |"
        )

    TABLE_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Effects: compare within each isolated group where possible.
    baselines: dict[str, str] = {
        "board_baseline": "kv260_baseline_safe_fx16",
        "precision": "kv260_precision_fx16_6",
        "parallelism": "kv260_parallel_x1",
        "pipeline": "kv260_pipeline_balanced_ii2",
        "tiling": "kv260_tiling_small",
        "memory": "kv260_memory_bram",
        "combined": "kv260_precision_fx16_6",
        "training": "training_kv260_safe_fx16_6",
    }

    by_name = {r["name"]: r for r in rows}
    effects: list[dict[str, Any]] = []

    for r in rows:
        group = r["group"]
        base_name = baselines.get(group, "kv260_precision_fx16_6")
        base = by_name.get(base_name)
        if base is None:
            continue

        effects.append(
            {
                "name": r["name"],
                "group": group,
                "compared_to": base_name,
                "decision": (
                    f"precision={r['precision']}, pe/simd/unroll/partition="
                    f"{r['pe']}/{r['simd']}/{r['unroll']}/{r['partition']}, "
                    f"pipeline={r['pipeline_style']} II={r['ii']}, "
                    f"dense_tile={r['dense_tile']}, memory={r['weight_storage']}"
                ),
                "lut_change": _pct_change(r["pred_lut"], base["pred_lut"]),
                "ff_change": _pct_change(r["pred_ff"], base["pred_ff"]),
                "dsp_change": _pct_change(r["pred_dsp"], base["pred_dsp"]),
                "bram18_change": _pct_change(r["pred_bram18"], base["pred_bram18"]),
                "cycles_change": _pct_change(r["pred_cycles"], base["pred_cycles"]),
                "latency_change": _pct_change(r["pred_latency_ms"], base["pred_latency_ms"]),
                "throughput_change": _pct_change(r["pred_throughput_fps"], base["pred_throughput_fps"]),
                "source": "prediction",
            }
        )

    effect_fields = [
        "name",
        "group",
        "compared_to",
        "decision",
        "lut_change",
        "ff_change",
        "dsp_change",
        "bram18_change",
        "cycles_change",
        "latency_change",
        "throughput_change",
        "source",
    ]

    with EFFECTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=effect_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(effects)

    emd = [
        "# Stage 1 design-decision effect table",
        "",
        "Source label: `prediction`.",
        "Each row is compared against the baseline for its own group.",
        "",
        "| design | group | compared to | LUT | FF | DSP | BRAM18 | cycles | latency | throughput |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for e in effects:
        emd.append(
            f"| `{e['name']}` | {e['group']} | `{e['compared_to']}` | "
            f"{e['lut_change']} | {e['ff_change']} | {e['dsp_change']} | {e['bram18_change']} | "
            f"{e['cycles_change']} | {e['latency_change']} | {e['throughput_change']} |"
        )

    EFFECTS_MD.write_text("\n".join(emd) + "\n", encoding="utf-8")

    print(f"[OK] wrote {TABLE_CSV}")
    print(f"[OK] wrote {TABLE_MD}")
    print(f"[OK] wrote {EFFECTS_CSV}")
    print(f"[OK] wrote {EFFECTS_MD}")
    print(f"[OK] rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
