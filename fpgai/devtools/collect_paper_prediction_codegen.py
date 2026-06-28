from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
MANIFEST = ROOT / "generated_config_manifest.json"
OUT = ROOT / "prediction_codegen_results"
RESULTS_JSON = OUT / "results_recollected.json"
RESULTS_CSV = OUT / "results_recollected.csv"
SUMMARY_MD = OUT / "summary_recollected.md"


EXPECTED_REPORTS = [
    "ir/compile_plan.json",
    "ir/memory_plan.json",
    "ir/comm_plan.json",
    "reports/model_profile.json",
    "reports/resource_prediction.json",
    "reports/timing_prediction.json",
    "reports/prediction_summary.md",
    "reports/board_fit.json",
    "reports/board_fit.md",
    "reports/hardware_knob_contract.json",
    "reports/hardware_knob_contract.md",
]


EXPECTED_HLS = [
    "hls/src/deeplearn.cpp",
    "hls/include/fpgai_types.h",
    "hls/include/fpgai_params.h",
    "hls/src/fpgai_params.cpp",
    "hls/run_hls.tcl",
]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_nested(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _resource_summary(run_dir: Path) -> dict[str, Any]:
    data = _read_json(run_dir / "reports/resource_prediction.json")
    if not isinstance(data, dict):
        return {}

    out: dict[str, Any] = {}

    for source in [data, data.get("totals"), data.get("summary"), data.get("prediction")]:
        if isinstance(source, dict):
            for key in [
                "lut",
                "ff",
                "bram",
                "bram18",
                "uram",
                "dsp",
                "predicted_lut",
                "predicted_ff",
                "predicted_bram18",
                "predicted_uram",
                "predicted_dsp",
            ]:
                if key in source:
                    out[key.replace("predicted_", "")] = source[key]

    return out


def _timing_summary(run_dir: Path) -> dict[str, Any]:
    data = _read_json(run_dir / "reports/timing_prediction.json")
    if not isinstance(data, dict):
        return {}

    out: dict[str, Any] = {}

    for source in [data, data.get("summary"), data.get("prediction")]:
        if isinstance(source, dict):
            for key in [
                "clock_mhz",
                "latency_ms",
                "estimated_latency_ms",
                "predicted_latency_ms",
                "throughput_fps",
                "predicted_throughput_fps",
                "cycles",
                "predicted_cycles",
            ]:
                if key in source:
                    clean = key.replace("estimated_", "").replace("predicted_", "")
                    out[clean] = source[key]

    return out


def _board_fit_status(run_dir: Path) -> str | None:
    data = _read_json(run_dir / "reports/board_fit.json")
    if not isinstance(data, dict):
        return None
    return (
        data.get("status")
        or data.get("fit_status")
        or _get_nested(data, "summary", "status")
        or _get_nested(data, "fit", "status")
    )


def _contract_counts(run_dir: Path) -> dict[str, Any]:
    data = _read_json(run_dir / "reports/hardware_knob_contract.json")
    if not isinstance(data, dict):
        return {"contract_knob_count": None, "contract_status_counts": {}}

    knobs = data.get("knobs", [])
    counts: dict[str, int] = {}
    if isinstance(knobs, list):
        for knob in knobs:
            if isinstance(knob, dict):
                status = str(knob.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
        return {"contract_knob_count": len(knobs), "contract_status_counts": counts}

    return {"contract_knob_count": None, "contract_status_counts": counts}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    for row in manifest:
        name = row["name"]
        run_dir = ROOT / "runs" / name

        reports = {rel: (run_dir / rel).exists() for rel in EXPECTED_REPORTS}
        hls = {rel: (run_dir / rel).exists() for rel in EXPECTED_HLS}

        missing_reports = [rel for rel, ok in reports.items() if not ok]
        missing_hls = [rel for rel, ok in hls.items() if not ok]

        result = {
            **row,
            "run_dir": str(run_dir),
            "passed": not missing_reports and not missing_hls,
            "missing_reports": missing_reports,
            "missing_hls": missing_hls,
            "board_fit_status": _board_fit_status(run_dir),
            "resource_summary": _resource_summary(run_dir),
            "timing_summary": _timing_summary(run_dir),
            **_contract_counts(run_dir),
        }
        results.append(result)

    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    flat_rows = []
    for r in results:
        flat = {
            "name": r["name"],
            "passed": r["passed"],
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
            "board_fit_status": r["board_fit_status"],
            "contract_knob_count": r["contract_knob_count"],
            "missing_reports_count": len(r["missing_reports"]),
            "missing_hls_count": len(r["missing_hls"]),
            "run_dir": r["run_dir"],
        }

        for k, v in r["resource_summary"].items():
            flat[f"pred_{k}"] = v
        for k, v in r["timing_summary"].items():
            flat[f"pred_{k}"] = v

        flat_rows.append(flat)

    fieldnames = sorted({k for row in flat_rows for k in row.keys()})
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    lines = [
        "# Sprint 26 prediction/codegen recollected results",
        "",
        f"Total designs: {len(results)}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
        "| name | status | mode | board | precision | knobs | board fit | missing reports | missing HLS |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        knobs = (
            f"pe={r['pe']}, simd={r['simd']}, unroll={r['unroll']}, "
            f"part={r['partition']}, {r['pipeline_style']} II={r['ii']}, "
            f"mem={r['weight_storage']}"
        )
        lines.append(
            f"| `{r['name']}` | {status} | {r['mode']} | {r['board']} | {r['precision']} | "
            f"{knobs} | {r['board_fit_status']} | "
            f"{len(r['missing_reports'])} | {len(r['missing_hls'])} |"
        )

    if failed:
        lines += ["", "## Failed designs", ""]
        for r in results:
            if not r["passed"]:
                lines += [
                    f"### {r['name']}",
                    "",
                    f"- missing reports: `{r['missing_reports']}`",
                    f"- missing HLS: `{r['missing_hls']}`",
                    "",
                ]

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {RESULTS_JSON}")
    print(f"[OK] wrote {RESULTS_CSV}")
    print(f"[OK] wrote {SUMMARY_MD}")
    print(f"[OK] passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
