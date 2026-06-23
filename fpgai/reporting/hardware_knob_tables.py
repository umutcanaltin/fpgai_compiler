#!/usr/bin/env python3
"""
Build paper-facing hardware knob tables from Vivado implementation reports.

Inputs:
  reports/vivado_impl_summary/vivado_impl_summary.csv

Outputs:
  reports/hardware_knobs/precision_table.csv/.md
  reports/hardware_knobs/pipeline_table.csv/.md
  reports/hardware_knobs/parallel_envelope.csv/.md
  reports/hardware_knobs/hardware_knobs.md
  reports/hardware_knobs/hardware_knobs_summary.json

The collector is conservative: it only uses rows already present in the
CSV and does not infer missing Vivado metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


NUMERIC_COLUMNS = {
    "wns_ns",
    "fmax_mhz",
    "safe_clock_mhz",
    "power_w",
    "lut",
    "ff",
    "bram",
    "dsp",
    "LUT",
    "FF",
    "BRAM",
    "DSP",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], intro: str = "") -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if intro:
            f.write(intro.rstrip() + "\n\n")
        f.write(f"Rows: {len(rows)}\n\n")
        if not rows:
            f.write("No matching rows found.\n")
            return
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join("---:" if c.lower() in NUMERIC_COLUMNS else "---" for c in columns) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(clean_cell(row.get(c, "")) for c in columns) + " |\n")


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def normalized_metric_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "design": row.get("design", ""),
        "status": row.get("status", ""),
        "bitstream": row.get("bitstream", ""),
        "xsa": row.get("xsa", ""),
        "wns_ns": row.get("wns_ns", ""),
        "fmax_mhz": row.get("fmax_mhz", ""),
        "safe_clock_mhz": row.get("safe_clock_mhz", ""),
        "power_w": row.get("power_w", ""),
        "lut": row.get("lut", row.get("LUT", "")),
        "ff": row.get("ff", row.get("FF", "")),
        "bram": row.get("bram", row.get("BRAM", "")),
        "dsp": row.get("dsp", row.get("DSP", "")),
        "notes": row.get("notes", ""),
        "artifact_dir": row.get("artifact_dir", ""),
    }


def sort_design_key(row: dict[str, Any]) -> tuple[int, str]:
    design = str(row.get("design", ""))
    nums = re.findall(r"\d+", design)
    first = int(nums[0]) if nums else 999
    return first, design


def add_precision_mode(row: dict[str, Any]) -> dict[str, Any]:
    design = str(row.get("design", ""))
    out = dict(row)
    if "fx8" in design:
        out["precision_mode"] = "fx8_like"
    elif "fx12" in design:
        out["precision_mode"] = "fx12_like"
    elif "fx16" in design:
        out["precision_mode"] = "fx16_like"
    else:
        out["precision_mode"] = "UNKNOWN"
    return out


def add_pipeline_policy(row: dict[str, Any]) -> dict[str, Any]:
    design = str(row.get("design", ""))
    out = dict(row)
    if "conservative" in design:
        out["pipeline_policy"] = "conservative"
    elif "balanced" in design:
        out["pipeline_policy"] = "balanced"
    elif "aggressive" in design:
        out["pipeline_policy"] = "aggressive"
    else:
        out["pipeline_policy"] = "UNKNOWN"
    return out


def add_parallel_factor(row: dict[str, Any]) -> dict[str, Any]:
    design = str(row.get("design", ""))
    out = dict(row)
    m = re.search(r"parallel_(\d+)", design)
    out["parallel_factor"] = int(m.group(1)) if m else "UNKNOWN"
    return out


def summarize_table(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "rows": len(rows),
        "pass_rows": sum(1 for r in rows if r.get("status") == "pass"),
        "timing_fail_rows": sum(1 for r in rows if r.get("status") == "timing_fail"),
        "bitstream_rows": sum(1 for r in rows if str(r.get("bitstream", "")).lower() == "true"),
        "complete_resource_rows": sum(
            1
            for r in rows
            if all(str(r.get(k, "")).strip() for k in ["lut", "ff", "bram", "dsp"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vivado-summary",
        default="reports/vivado_impl_summary/vivado_impl_summary.csv",
        help="Vivado implementation summary CSV",
    )
    parser.add_argument(
        "--out",
        default="reports/hardware_knobs",
        help="Output report directory",
    )
    args = parser.parse_args()

    summary_path = Path(args.vivado_summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Vivado summary CSV: {summary_path}")

    all_rows = [normalized_metric_row(r) for r in read_csv(summary_path)]

    precision_rows = [add_precision_mode(r) for r in all_rows if str(r.get("design", "")).startswith("hw_prec_")]
    precision_order = {"fx8_like": 0, "fx12_like": 1, "fx16_like": 2}
    precision_rows.sort(key=lambda r: precision_order.get(str(r.get("precision_mode")), 99))

    pipeline_rows = [add_pipeline_policy(r) for r in all_rows if str(r.get("design", "")).startswith("hw_pipeline_")]
    pipeline_order = {"conservative": 0, "balanced": 1, "aggressive": 2}
    pipeline_rows.sort(key=lambda r: pipeline_order.get(str(r.get("pipeline_policy")), 99))

    parallel_rows = [add_parallel_factor(r) for r in all_rows if "parallel" in str(r.get("design", ""))]
    parallel_rows.sort(key=sort_design_key)

    base_cols = [
        "design",
        "status",
        "bitstream",
        "xsa",
        "wns_ns",
        "fmax_mhz",
        "safe_clock_mhz",
        "power_w",
        "lut",
        "ff",
        "bram",
        "dsp",
        "notes",
    ]

    precision_cols = ["precision_mode"] + base_cols
    pipeline_cols = ["pipeline_policy"] + base_cols
    parallel_cols = ["parallel_factor"] + base_cols

    write_csv(out / "precision_table.csv", precision_rows, precision_cols)
    write_csv(out / "pipeline_table.csv", pipeline_rows, pipeline_cols)
    write_csv(out / "parallel_envelope.csv", parallel_rows, parallel_cols)

    write_md_table(
        out / "precision_table.md",
        "Precision Hardware Table",
        precision_rows,
        precision_cols,
        "Precision rows are selected from Vivado implementation reports.",
    )
    write_md_table(
        out / "pipeline_table.md",
        "Pipeline Policy Hardware Table",
        pipeline_rows,
        pipeline_cols,
        "Pipeline rows are selected from Vivado implementation reports.",
    )
    write_md_table(
        out / "parallel_envelope.md",
        "Parallel Feasibility Envelope",
        parallel_rows,
        parallel_cols,
        "Parallel rows are selected from Vivado implementation reports.",
    )

    summary = {
        "source": str(summary_path),
        "precision": summarize_table(precision_rows, "precision"),
        "pipeline": summarize_table(pipeline_rows, "pipeline"),
        "parallel": summarize_table(parallel_rows, "parallel"),
        "safe_claim": (
            "FPGAI hardware knobs produce measurable implementation-level differences "
            "in resources, timing, and power for evaluated designs."
        ),
        "limitations": [
            "Tables are generated from existing Vivado reports and do not rerun Vivado.",
            "Claims are limited to evaluated design points and target flow.",
            "FPGAI does not claim global design-space optimality.",
        ],
    }
    (out / "hardware_knobs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (out / "hardware_knobs.md").open("w", encoding="utf-8") as f:
        f.write("# Hardware Knob Report\n\n")
        f.write("This file summarizes paper-facing hardware knob tables generated from Vivado implementation reports.\n\n")
        f.write("## Summary\n\n")
        for key in ["precision", "pipeline", "parallel"]:
            item = summary[key]
            f.write(
                f"- {key}: rows={item['rows']}, pass_rows={item['pass_rows']}, "
                f"timing_fail_rows={item['timing_fail_rows']}, bitstream_rows={item['bitstream_rows']}, "
                f"complete_resource_rows={item['complete_resource_rows']}\n"
            )
        f.write("\n## Precision table\n\n")
        f.write((out / "precision_table.md").read_text(encoding="utf-8"))
        f.write("\n\n## Pipeline table\n\n")
        f.write((out / "pipeline_table.md").read_text(encoding="utf-8"))
        f.write("\n\n## Parallel feasibility envelope\n\n")
        f.write((out / "parallel_envelope.md").read_text(encoding="utf-8"))
        f.write("\n\n## Safe claim\n\n")
        f.write(summary["safe_claim"] + "\n")
        f.write("\n## Limitations\n\n")
        for limitation in summary["limitations"]:
            f.write(f"- {limitation}\n")

    print(f"Wrote {out / 'precision_table.csv'}")
    print(f"Wrote {out / 'pipeline_table.csv'}")
    print(f"Wrote {out / 'parallel_envelope.csv'}")
    print(f"Wrote {out / 'hardware_knobs.md'}")
    print(
        "precision_rows={precision} pipeline_rows={pipeline} parallel_rows={parallel}".format(
            precision=len(precision_rows), pipeline=len(pipeline_rows), parallel=len(parallel_rows)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
