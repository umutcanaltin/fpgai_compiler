"""Paper-facing tables from an already source-attributed precision comparison.

The native experiment itself is never repeated, files are never edited in
place, and missing or unqualified numbers are not described as accepted
design choices. JSON keeps complete evidence, including invalid designs.
"""
from __future__ import annotations

import csv
from pathlib import Path

SCHEMA = "fpgai-research/cross-suite-precision/v1"
COLUMNS = (
    "model", "format", "case", "evidence", "numerical_pass",
    "comparison_qualified", "csim_failures", "max_abs_error", "rmse",
    "compiled_top1_accuracy", "sample_count",
    "routed_lut", "routed_ff", "routed_dsp", "routed_bram_tiles",
    "routed_uram", "routed_wns_ns",
    "hls_latency_cycles", "hls_ii_cycles",
    "source_sha256", "route_report_sha256", "drc_report_sha256", "blockers",
)


def publication_rows(result: dict) -> list[dict]:
    if result.get("schema") != SCHEMA or not isinstance(result.get("cases"), dict):
        raise ValueError("A source-attributed precision comparison is required")
    cases = result["cases"]
    if not cases:
        raise ValueError("Cannot publish empty precision comparison")
    output = []
    for label, case in cases.items():
        if not isinstance(case, dict) or not isinstance(case.get("blockers"), list):
            raise ValueError("Incomplete precision case evidence")
        e4 = case.get("evidence") == "E4-Vivado-implementation"
        qualified = case.get("qualified") is True
        if qualified and (not e4 or case.get("csim_passed") is not True
                          or case.get("blockers")):
            raise ValueError("Qualified paper row lacks numerical/E4 evidence")
        resources = case.get("routed_utilization") or {} if e4 else {}
        reports = case.get("report_sha256") or {} if e4 else {}
        classification = case.get("classification") or {}
        row = dict.fromkeys(COLUMNS)
        row.update({
            "model": result.get("model"),
            "format": f'{case["width"]}/{case["integer_bits"]}',
            "case": case.get("case"), "evidence": case.get("evidence"),
            "numerical_pass": case.get("csim_passed"),
            "comparison_qualified": qualified,
            "csim_failures": case.get("csim_failures"),
            "max_abs_error": case.get("csim_max_absolute_error"),
            "rmse": case.get("csim_rmse"),
            "compiled_top1_accuracy": classification.get("compiled_top1_accuracy"),
            "sample_count": classification.get("samples"),
            "routed_lut": resources.get("lut"),
            "routed_ff": resources.get("ff"),
            "routed_dsp": resources.get("dsp"),
            "routed_bram_tiles": resources.get("bram"),
            "routed_uram": resources.get("uram"),
            "routed_wns_ns": case.get("wns_ns") if e4 else None,
            "hls_latency_cycles": case.get("hls_latency_cycles"),
            "hls_ii_cycles": case.get("hls_ii_cycles"),
            "source_sha256": case.get("case_source_sha256"),
            "route_report_sha256": reports.get("route_status.rpt"),
            "drc_report_sha256": reports.get("drc.rpt"),
            "blockers": "; ".join(case["blockers"]) if case["blockers"] else None,
        })
        output.append(row)
    return output


def _text(value) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def render_precision_markdown(result: dict, rows: list[dict]) -> str:
    """Short paper-facing summary with a visible qualification boundary."""
    selected = ("format", "comparison_qualified", "max_abs_error",
                "routed_lut", "routed_ff", "routed_dsp",
                "routed_bram_tiles", "routed_wns_ns", "hls_latency_cycles")
    headers = ("Format", "Qualified", "Max abs error", "LUT", "FF", "DSP",
               "BRAM tiles", "WNS (ns)", "HLS latency (cycles)")
    lines = [
        "# Fixed-point precision study — " + result["model"],
        "",
        "Source-matched, out-of-context routed accelerator results.",
        "A qualified result requires passing native fixed-point C simulation,",
        "attested E4 Vivado routing, timing closure, and accepted DRC evidence.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_text(row[k]) for k in selected) + " |")
    lines.extend(["", "## Comparison qualification", ""])
    for label, comparison in result.get("comparisons", {}).items():
        if comparison.get("qualified") is True:
            deltas = comparison["deltas"]
            summary = deltas.get("lut")
            if summary is not None:
                lines.append(f"- {label}: qualified; LUT {summary['baseline']} → "
                             f"{summary['observed']} ({summary['reduction_percent']}% fewer)")
            else:
                lines.append(f"- {label}: qualified; no routed LUT delta available")
        else:
            lines.append(f"- {label}: unqualified; "
                         + _text(comparison.get("reason")))
    lines.extend([
        "", "## Evidence limitations", "",
        result.get("limitations", "No limitations statement available."),
        "",
        "All counts concern out-of-context FPGA modules, not a deployed board.",
        "HLS latency is an estimate, not measured end-to-end execution time.",
        "Fixed-point accuracy is limited to the original testbench samples.",
        "The companion JSON retains model, inputs, labels and report SHA-256 digests.",
        "",
    ])
    return "\n".join(lines)


def write_precision_publication(result: dict, destination: Path) -> list[dict]:
    """Write two derived artifacts into a staging directory only."""
    destination = Path(destination)
    rows = publication_rows(result)
    csv_path = destination / "precision_comparison.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({
            key: "TBD" if item.get(key) is None else item[key]
            for key in COLUMNS
        } for item in rows)
    (destination/"precision_comparison.md").write_text(
        render_precision_markdown(result, rows))
    return rows
