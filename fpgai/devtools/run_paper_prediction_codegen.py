from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
MANIFEST = ROOT / "generated_config_manifest.json"
OUT = ROOT / "prediction_codegen_results"
LOG_DIR = OUT / "logs"
RESULTS_JSON = OUT / "results.json"
RESULTS_CSV = OUT / "results.csv"
SUMMARY_MD = OUT / "summary.md"


EXPECTED_REPORTS = [
    "ir/compile_plan.json",
    "reports/model_profile.json",
    "reports/resource_prediction.json",
    "reports/timing_prediction.json",
    "reports/prediction_summary.md",
    "reports/board_fit.json",
    "reports/board_fit.md",
    "reports/hardware_knob_contract.json",
    "reports/hardware_knob_contract.md",
    "ir/memory_plan.json",
    "ir/comm_plan.json",
]


EXPECTED_HLS = [
    "hls/src/deeplearn.cpp",
    "hls/include/fpgai_types.h",
    "hls/include/fpgai_params.h",
    "hls/src/fpgai_params.cpp",
    "hls/run_hls.tcl",
]


def _load_manifest() -> list[dict[str, Any]]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("generated_config_manifest.json must contain a list")
    return data


def _run_one(row: dict[str, Any]) -> dict[str, Any]:
    name = row["name"]
    config_path = Path(row["config_path"])
    run_dir = ROOT / "runs" / name

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_log = LOG_DIR / f"{name}.stdout.log"
    stderr_log = LOG_DIR / f"{name}.stderr.log"

    if run_dir.exists():
        shutil.rmtree(run_dir)

    cmd = [
        sys.executable,
        "-B",
        "main.py",
        "compile",
        "--config",
        str(config_path),
    ]

    start = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.time() - start

    stdout_log.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    stderr_log.write_text(proc.stderr, encoding="utf-8", errors="ignore")

    reports = {rel: (run_dir / rel).exists() for rel in EXPECTED_REPORTS}
    hls = {rel: (run_dir / rel).exists() for rel in EXPECTED_HLS}

    contract_path = run_dir / "reports" / "hardware_knob_contract.json"
    contract_knob_count = None
    contract_status_counts: dict[str, int] = {}
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            knobs = contract.get("knobs", [])
            if isinstance(knobs, list):
                contract_knob_count = len(knobs)
                for knob in knobs:
                    if isinstance(knob, dict):
                        status = str(knob.get("status", "unknown"))
                        contract_status_counts[status] = contract_status_counts.get(status, 0) + 1
        except Exception:
            pass

    board_fit_path = run_dir / "reports" / "board_fit.json"
    board_fit_status = None
    if board_fit_path.exists():
        try:
            board_fit = json.loads(board_fit_path.read_text(encoding="utf-8"))
            board_fit_status = (
                board_fit.get("status")
                or board_fit.get("fit_status")
                or board_fit.get("summary", {}).get("status")
            )
        except Exception:
            pass

    resource_path = run_dir / "reports" / "resource_prediction.json"
    resource_summary: dict[str, Any] = {}
    if resource_path.exists():
        try:
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            for key in ["lut", "ff", "bram", "bram18", "uram", "dsp"]:
                if key in resource:
                    resource_summary[key] = resource[key]
            totals = resource.get("totals")
            if isinstance(totals, dict):
                for key in ["lut", "ff", "bram", "bram18", "uram", "dsp"]:
                    if key in totals:
                        resource_summary[key] = totals[key]
        except Exception:
            pass

    timing_path = run_dir / "reports" / "timing_prediction.json"
    timing_summary: dict[str, Any] = {}
    if timing_path.exists():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            for key in ["latency_ms", "estimated_latency_ms", "clock_mhz", "throughput_fps"]:
                if key in timing:
                    timing_summary[key] = timing[key]
            summary = timing.get("summary")
            if isinstance(summary, dict):
                for key in ["latency_ms", "estimated_latency_ms", "clock_mhz", "throughput_fps"]:
                    if key in summary:
                        timing_summary[key] = summary[key]
        except Exception:
            pass

    missing_reports = [rel for rel, ok in reports.items() if not ok]
    missing_hls = [rel for rel, ok in hls.items() if not ok]

    passed = proc.returncode == 0 and not missing_reports and not missing_hls

    return {
        **row,
        "returncode": proc.returncode,
        "passed": passed,
        "elapsed_sec": round(elapsed, 3),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "missing_reports": missing_reports,
        "missing_hls": missing_hls,
        "contract_knob_count": contract_knob_count,
        "contract_status_counts": contract_status_counts,
        "board_fit_status": board_fit_status,
        "resource_summary": resource_summary,
        "timing_summary": timing_summary,
    }


def _write_outputs(results: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    flat_rows = []
    for r in results:
        flat = {
            "name": r["name"],
            "passed": r["passed"],
            "returncode": r["returncode"],
            "elapsed_sec": r["elapsed_sec"],
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
            "board_fit_status": r.get("board_fit_status"),
            "contract_knob_count": r.get("contract_knob_count"),
            "missing_reports_count": len(r["missing_reports"]),
            "missing_hls_count": len(r["missing_hls"]),
            "run_dir": r["run_dir"],
        }
        for k, v in r.get("resource_summary", {}).items():
            flat[f"pred_{k}"] = v
        for k, v in r.get("timing_summary", {}).items():
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
        "# Sprint 26 prediction/codegen results",
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
            f"{knobs} | {r.get('board_fit_status')} | "
            f"{len(r['missing_reports'])} | {len(r['missing_hls'])} |"
        )

    lines.append("")
    lines.append("## Failed designs")
    lines.append("")
    failed_rows = [r for r in results if not r["passed"]]
    if not failed_rows:
        lines.append("None.")
    else:
        for r in failed_rows:
            lines.append(f"### {r['name']}")
            lines.append("")
            lines.append(f"- returncode: `{r['returncode']}`")
            lines.append(f"- stdout: `{r['stdout_log']}`")
            lines.append(f"- stderr: `{r['stderr_log']}`")
            lines.append(f"- missing reports: `{r['missing_reports']}`")
            lines.append(f"- missing HLS: `{r['missing_hls']}`")
            lines.append("")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _load_manifest()
    results: list[dict[str, Any]] = []

    print(f"[INFO] designs: {len(rows)}")
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row['name']}")
        result = _run_one(row)
        results.append(result)
        print(
            f"  returncode={result['returncode']} passed={result['passed']} "
            f"elapsed={result['elapsed_sec']}s"
        )

    _write_outputs(results)

    print(f"[OK] wrote {RESULTS_JSON}")
    print(f"[OK] wrote {RESULTS_CSV}")
    print(f"[OK] wrote {SUMMARY_MD}")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"[FAIL] failed designs: {len(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
