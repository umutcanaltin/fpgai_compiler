"""Build a paper artifact index from FPGAI experiment outputs.

This tool intentionally reports artifact availability/status only.
Resource/timing numeric parsing is handled by later paper-result builders.
"""

from __future__ import annotations

import argparse
import csv
import json
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


FIELDNAMES = [
    "design",
    "board",
    "mode",
    "group",
    "run_dir",
    "manifest_exists",
    "prediction_available",
    "model_profile_exists",
    "resource_prediction_exists",
    "timing_prediction_exists",
    "prediction_summary_exists",
    "hls_available",
    "hls_top_csynth_xml",
    "hls_top_csynth_rpt",
    "hls_status",
    "vivado_bridge_available",
    "vivado_manifest_exists",
    "vivado_run_artifacts_exists",
    "vivado_ok",
    "vivado_returncode",
    "vivado_failure_class",
    "vivado_error",
    "vivado_reports_present",
    "bitstream_exists",
    "xsa_exists",
    "paper_status",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _bool_cell(value: Any) -> bool:
    return bool(value)


def _infer_board(design: str, manifest: dict[str, Any]) -> str:
    for key in ("board", "board_name", "target_board"):
        val = manifest.get(key)
        if isinstance(val, str) and val:
            return val
    if design.startswith("pynq_z2"):
        return "pynq_z2"
    if design.startswith("kv260") or design.startswith("training_kv260"):
        return "kv260"
    if design.startswith("kr260"):
        return "kr260"
    return ""


def _infer_mode(design: str, manifest: dict[str, Any]) -> str:
    for key in ("mode", "pipeline_mode", "compile_mode"):
        val = manifest.get(key)
        if isinstance(val, str) and val:
            return val
    if design.startswith("training_"):
        return "training"
    return "inference"


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


def _find_top_csynth(run_dir: Path) -> tuple[str, str]:
    report_dir = run_dir / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"

    # Prefer top-level reports.
    candidates = [
        ("deeplearn_csynth.xml", "deeplearn_csynth.rpt"),
        ("train_top_csynth.xml", "train_top_csynth.rpt"),
        ("csynth.xml", "csynth.rpt"),
    ]
    for xml_name, rpt_name in candidates:
        xml = report_dir / xml_name
        rpt = report_dir / rpt_name
        if xml.exists() or rpt.exists():
            return (str(xml) if xml.exists() else "", str(rpt) if rpt.exists() else "")

    xmls = sorted(report_dir.glob("*_csynth.xml"))
    rpts = sorted(report_dir.glob("*_csynth.rpt"))
    return (str(xmls[0]) if xmls else "", str(rpts[0]) if rpts else "")


def _paper_status(row: dict[str, Any]) -> str:
    if not row["manifest_exists"]:
        return "missing_run_manifest"
    if not row["prediction_available"]:
        return "missing_prediction"
    if not row["hls_available"]:
        return "missing_hls_csynth"

    failure_class = str(row.get("vivado_failure_class") or "")
    if failure_class.startswith("vivado_impl_failed_board_capacity"):
        return "vivado_board_capacity_rejected"

    if row["vivado_bridge_available"]:
        if row["vivado_ok"] is True and row["bitstream_exists"] is True and row["xsa_exists"] is True:
            return "vivado_impl_bitstream_ready"
        if row["vivado_ok"] is False:
            return "vivado_failed"
        return "vivado_bridge_present"

    return "hls_only"


def _row_for_design(base: Path, design: str) -> dict[str, Any]:
    run_dir = base / "runs" / design
    manifest = _read_json(run_dir / "manifest.json")

    model_profile = run_dir / "reports" / "model_profile.json"
    resource_prediction = run_dir / "reports" / "resource_prediction.json"
    timing_prediction = run_dir / "reports" / "timing_prediction.json"
    prediction_summary = run_dir / "reports" / "prediction_summary.md"

    hls_xml, hls_rpt = _find_top_csynth(run_dir)

    vivado_manifest_path = run_dir / "vivado_bridge" / "vivado_bridge_manifest.json"
    vivado_manifest = _read_json(vivado_manifest_path)
    vivado_run_artifacts = run_dir / "vivado_bridge_run_artifacts.json"

    vivado_bridge_available = bool((run_dir / "vivado_bridge").exists())
    vivado_ok = vivado_manifest.get("vivado_ok")
    if vivado_ok is not None:
        vivado_ok = bool(vivado_ok)

    row: dict[str, Any] = {
        "design": design,
        "board": _infer_board(design, manifest),
        "mode": _infer_mode(design, manifest),
        "group": _infer_group(design),
        "run_dir": str(run_dir),
        "manifest_exists": (run_dir / "manifest.json").exists(),
        "prediction_available": resource_prediction.exists() and timing_prediction.exists(),
        "model_profile_exists": model_profile.exists(),
        "resource_prediction_exists": resource_prediction.exists(),
        "timing_prediction_exists": timing_prediction.exists(),
        "prediction_summary_exists": prediction_summary.exists(),
        "hls_available": bool(hls_xml or hls_rpt),
        "hls_top_csynth_xml": hls_xml,
        "hls_top_csynth_rpt": hls_rpt,
        "hls_status": "full_csynth" if bool(hls_xml or hls_rpt) else "missing",
        "vivado_bridge_available": vivado_bridge_available,
        "vivado_manifest_exists": vivado_manifest_path.exists(),
        "vivado_run_artifacts_exists": vivado_run_artifacts.exists(),
        "vivado_ok": vivado_ok,
        "vivado_returncode": vivado_manifest.get("vivado_returncode"),
        "vivado_failure_class": vivado_manifest.get("vivado_failure_class") or "",
        "vivado_error": vivado_manifest.get("vivado_error") or vivado_manifest.get("error") or "",
        "vivado_reports_present": bool(vivado_manifest.get("vivado_reports_present", False)),
        "bitstream_exists": bool(vivado_manifest.get("bitstream_exists", False)),
        "xsa_exists": bool(vivado_manifest.get("xsa_exists", False)),
    }
    row["paper_status"] = _paper_status(row)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "design",
        "board",
        "mode",
        "group",
        "prediction_available",
        "hls_status",
        "vivado_ok",
        "vivado_failure_class",
        "bitstream_exists",
        "xsa_exists",
        "paper_status",
    ]
    lines = []
    lines.append("# Paper artifact index")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="paper_experiments/full_pipeline_gate/sprint26_paper_matrix",
        help="Paper matrix experiment directory containing runs/",
    )
    ap.add_argument(
        "--out",
        default="paper_results/index",
        help="Output directory for artifact index files",
    )
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)

    if not (base / "runs").exists():
        raise SystemExit(f"Missing runs directory: {base / 'runs'}")

    rows = [_row_for_design(base, design) for design in PAPER_DESIGN_ORDER]

    # Include unexpected extra run dirs at the end, but keep the canonical paper
    # matrix ordering stable.
    known = set(PAPER_DESIGN_ORDER)
    for run_dir in sorted((base / "runs").iterdir()):
        if run_dir.is_dir() and run_dir.name not in known:
            rows.append(_row_for_design(base, run_dir.name))

    _write_csv(out / "paper_artifact_index.csv", rows)
    _write_json(out / "paper_artifact_index.json", rows)
    _write_md(out / "paper_artifact_index.md", rows)

    print(f"[OK] wrote {out / 'paper_artifact_index.csv'}")
    print(f"[OK] wrote {out / 'paper_artifact_index.json'}")
    print(f"[OK] wrote {out / 'paper_artifact_index.md'}")

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["paper_status"])] = counts.get(str(row["paper_status"]), 0) + 1

    print("[SUMMARY]")
    print(f"designs={len(rows)}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
