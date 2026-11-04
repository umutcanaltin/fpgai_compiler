"""Read-only survey of Vitis HLS text report *format*, not scheduling results.

Legacy human-readable Vitis reports may be inspected to decide which
version-aware parser to implement next, but an unattributed old report
MUST NOT become E3 evidence or yield claimed loop II/latency.

The survey exports only whitelisted report heading/column vocabulary,
occurrence counts and fingerprints. No arbitrary raw report lines,
operator identifiers, timing numbers, trained weights or tool runs.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_schedule_reports import (
    REPORT_REL, REPORT_LIMIT, inspect_native_schedule_case,
)

SCHEMA = "fpgai-research/hls-report-format-survey/v1"
INVENTORY_SCHEMA = "fpgai-research/native-hls-schedule-report-inventory/v1"

# Whitelist rather than raw text excerpts: reports can include private IP paths.
_FIELDS = (
    ("loop", re.compile(r"\bloops?\b", re.IGNORECASE)),
    ("latency", re.compile(r"\blatenc(?:y|ies)\b", re.IGNORECASE)),
    ("interval", re.compile(r"\binterval\b", re.IGNORECASE)),
    ("pipeline", re.compile(r"\bpipelin(?:e|ed|ing)\b", re.IGNORECASE)),
    ("tripcount", re.compile(r"\btrip[\s_-]*counts?\b", re.IGNORECASE)),
    ("iteration", re.compile(r"\biterations?\b", re.IGNORECASE)),
    ("ii", re.compile(r"(?<![A-Za-z0-9])II(?![A-Za-z0-9])")),
    ("depth", re.compile(r"\bdepth\b", re.IGNORECASE)),
    ("performance", re.compile(r"\bperformance\b", re.IGNORECASE)),
)
MAX_LINES = 100_000
MAX_SIGNATURES = 24
MAX_PROFILED_ROWS = 256


def _cells(line: str) -> list[str] | None:
    if "|" not in line:
        return None
    parts=[item.strip() for item in line.strip().strip("|").split("|")]
    return parts if 2<=len(parts)<=32 else None


def _layout(cells: list[str]) -> tuple[str,...]:
    return tuple("+".join(_tokens(cell)) if _tokens(cell) else "other"
                 for cell in cells)


def _separator(cells: list[str]) -> bool:
    return all(bool(re.fullmatch(r"[-=:+\s]*", cell)) for cell in cells)


def _profile_adjacent_rows(lines: list[str], start: int, columns: int) -> dict:
    """Profile only the shape of adjacent pipe-table rows.

    This is intentionally a layout heuristic; it exports no loop names,
    numerical values, ordering of operations, or native schedule claims.
    """
    numeric_rows=0
    other_rows=0
    separators=0
    truncated=False
    number=re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?")
    for line in lines[start+1:start+MAX_PROFILED_ROWS+2]:
        cells=_cells(line)
        if cells is None or len(cells)!=columns:
            break
        if _separator(cells):
            separators+=1
            continue
        # Another header is not a data row; stop at the new table.
        candidate=_layout(cells)
        if sum(v!="other" for v in candidate)>=2 and (
            "latency" in _tokens(line) or "interval" in _tokens(line)
            or "ii" in _tokens(line) or "pipeline" in _tokens(line)):
            break
        numeric_cells=sum(number.fullmatch(cell) is not None for cell in cells)
        if numeric_cells>=2:
            numeric_rows+=1
        else:
            other_rows+=1
        if numeric_rows+other_rows>=MAX_PROFILED_ROWS:
            truncated=True
            break
    return {
        "possible_numeric_data_rows":numeric_rows,
        "possible_other_data_rows":other_rows,
        "separator_rows":separators,
        "truncated_at_row_limit":truncated,
    }




def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _tokens(text: str) -> list[str]:
    return [name for name, pattern in _FIELDS if pattern.search(text)]


def _survey_text(data: bytes) -> dict:
    if len(data) == 0 or len(data) > REPORT_LIMIT:
        raise ValueError("Text report outside safe survey size")
    decoded = data.decode("utf-8", errors="replace")
    lines = decoded.splitlines()
    if len(lines) > MAX_LINES:
        raise ValueError("Unbounded Vitis report line count")

    line_presence: Counter[str] = Counter()
    column_signature_counts: Counter[tuple[str, ...]] = Counter()
    profile_counts: dict[tuple[str,...],Counter[str]] = {}
    loop_heading_candidates = 0
    table_header_candidates = 0
    for line_index,line in enumerate(lines):
        # Never archive full original report text or values. Only capture
        # structural tokens and order of recognized column cells.
        found = _tokens(line)
        line_presence.update(found)
        stripped = line.strip()
        if re.match(r"^(?:Loop\s*(?::|Summary|Details|Performance|Latency)|\[\s*Loop\b)", stripped, re.I):
            loop_heading_candidates += 1
        cells=_cells(line)
        if cells is not None:
            # Only recognized vocabulary shapes; never original column text.
            named_tokens=_layout(cells)
            matches=sum(token!="other" for token in named_tokens)
            if matches >= 2 and (
                "latency" in found or "interval" in found or
                "ii" in found or "pipeline" in found
            ):
                table_header_candidates += 1
                column_signature_counts[named_tokens] += 1
                row_profile=_profile_adjacent_rows(lines,line_index,len(cells))
                current=profile_counts.setdefault(named_tokens,Counter())
                current.update({
                    "numeric_rows":row_profile["possible_numeric_data_rows"],
                    "other_rows":row_profile["possible_other_data_rows"],
                    "separators":row_profile["separator_rows"],
                    "truncated_tables":int(row_profile["truncated_at_row_limit"]),
                })
    ordered = sorted(
        column_signature_counts.items(),
        key=lambda entry: (-entry[1], entry[0]),
    )
    signature_samples = [
        {"columns": list(signature), "occurrences": n}
        for signature, n in ordered[:MAX_SIGNATURES]
    ]
    structural_profiles=[
        {"columns":list(signature),
         "candidate_role":("POSSIBLY_LOOP_OR_ITERATION_TABLE"
                           if any("loop" in v or "iteration" in v
                                  for v in signature)
                           else "NON_LOOP_OR_AMBIGUOUS_SUMMARY_TABLE"),
         "occurrences":n,
         "possible_numeric_data_rows":profile_counts[signature]["numeric_rows"],
         "possible_other_data_rows":profile_counts[signature]["other_rows"],
         "separator_rows":profile_counts[signature]["separators"],
         "truncated_tables":profile_counts[signature]["truncated_tables"]}
        for signature,n in ordered[:MAX_SIGNATURES]
    ]
    return {
        "line_count": len(lines),
        "recognized_line_counts": dict(sorted(line_presence.items())),
        "possible_loop_heading_lines": loop_heading_candidates,
        "possible_schedule_table_headers": table_header_candidates,
        "possible_column_layouts": signature_samples,
        "candidate_table_row_profiles":structural_profiles,
        "possible_loop_or_iteration_table_headers":sum(
            n for signature,n in ordered
            if any("loop" in v or "iteration" in v for v in signature)),
        "column_layout_variants": len(column_signature_counts),
        "report_format_digest": _digest(json.dumps(
            {"tokens": dict(sorted(line_presence.items())),
             "layouts": [[list(sig), n] for sig, n in ordered],
             "headings": loop_heading_candidates,
             "line_count": len(lines)},
            sort_keys=True, separators=(",", ":")
        ).encode()),
        "claims_about_achieved_loop_ii": False,
        "extracted_numerical_schedule_values": False,
    }


def survey_schedule_report_formats(suite_directory, inventory_document) -> dict:
    """Survey report headers only, with cross-checks against the saved inventory.

    A legacy report is eligible for *format inspection*, not evidence
    qualification. Verify source hash, report bytes and prior status
    have not changed since the independently generated inventory.
    """
    suite = Path(suite_directory)
    if suite.is_symlink() or not suite.is_dir():
        raise ValueError("A regular research suite directory is required")
    if not isinstance(inventory_document, dict) or inventory_document.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("Expected source-verified native schedule inventory")
    path = suite / "suite_results.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Pinned suite manifest missing or linked")
    pinned_sha = _digest(path.read_bytes())
    if inventory_document.get("source_suite_sha256") != pinned_sha:
        raise ValueError("Native report inventory is not from this pinned suite")
    cases = inventory_document.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) != inventory_document.get("case_count"):
        raise ValueError("Unusable or incomplete native report inventory")
    records = []
    seen = set()
    status_counts: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Invalid report inventory record")
        model, label = case.get("model"), case.get("case")
        for value in (model, label):
            if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,80}", value)
                    or value in (".", "..")):
                raise ValueError("Unsafe model/case identifier")
        key = (model, label)
        if key in seen:
            raise ValueError("Duplicate report case")
        seen.add(key)
        model_dir = suite / model
        if model_dir.is_symlink() or not model_dir.is_dir():
            raise ValueError("Unsafe or missing model study directory")
        folder = model_dir / label
        current = inspect_native_schedule_case(
            folder, case.get("source_sha256"), case.get("case_evidence"))
        if (current["status"] != case.get("status") or
                current["report_sha256"] != case.get("report_sha256") or
                current["source_sha256"] != case.get("source_sha256")):
            raise ValueError("Report/source bytes or evidence status changed after inventory")
        status = current["status"]
        status_counts[status] += 1
        record = {
            "model": model, "case": label, "axis": case.get("axis"),
            "provenance_status": status,
            "report_sha256": current["report_sha256"],
            "surveyed": False, "format_only": True,
            "native_schedule_verified": False,
            "per_loop_ii_cycles": None,
        }
        if status in ("PRESENT_NOT_FRESHLY_ATTESTED",
                      "PRESENT_NOT_ATTESTED", "NATIVE_TEXT_WITHOUT_E3_QUALIFICATION",
                      "ATTESTED_RAW_REPORT_AVAILABLE"):
            # inspect_native_schedule_case validated path components and
            # permitted report length. Recheck bytes after opening.
            report = folder / REPORT_REL
            data = report.read_bytes()
            if len(data) != current["report_size_bytes"] or _digest(data) != current["report_sha256"]:
                raise ValueError("Vitis report changed while surveying")
            analysis = _survey_text(data)
            record.update(surveyed=True, **analysis)
            profiles[analysis["report_format_digest"]] += 1
        records.append(record)
    contrasts=[]
    by_model={}
    for record in records:
        by_model.setdefault(record["model"], []).append(record)
    for model,group in by_model.items():
        baselines=[r for r in group if r["case"]=="case_0000"
                   and r["axis"]=="baseline"]
        if len(baselines)!=1:
            raise ValueError("Cannot compare report structures without unique model baseline")
        baseline=baselines[0]
        for variant in group:
            if variant is baseline or variant.get("axis") not in (
                    "simd","pes","parallelism"):
                continue
            if not baseline["surveyed"] or not variant["surveyed"]:
                status="REPORT_LAYOUT_UNAVAILABLE"
                matching=None
            else:
                matching=baseline["report_format_digest"]==variant["report_format_digest"]
                status="SAME_SURVEY_FORMAT_FINGERPRINT" if matching else "DIFFERENT_SURVEY_FORMAT_FINGERPRINT"
            contrasts.append({
                "model":model,"axis":variant["axis"],
                "baseline_case":baseline["case"],"variant_case":variant["case"],
                "format_comparison_status":status,
                "matching_format_fingerprint":matching,
                "baseline_possible_loop_table_headers":baseline.get(
                    "possible_loop_or_iteration_table_headers"),
                "variant_possible_loop_table_headers":variant.get(
                    "possible_loop_or_iteration_table_headers"),
                "scheduling_equivalence_or_cause_proven":False,
            })
    return {
        "schema": SCHEMA,
        "source_suite_sha256": pinned_sha,
        "case_count": len(records),
        "within_model_layout_contrasts":contrasts,
        "surveyed_report_count": sum(r["surveyed"] for r in records),
        "status_counts": dict(sorted(status_counts.items())),
        "distinct_format_count": len(profiles),
        "format_fingerprint_counts": dict(sorted(profiles.items())),
        "native_tools_executed": False,
        "weights_exported": False,
        "per_loop_schedule_evidence_available": False,
        "native_qor_or_evidence_upgraded": False,
        "cases": records,
        "limitations": (
            "Only whitelisted heading/column vocabulary is exported. "
            "Header recognition is heuristic and does not establish a "
            "version-aware loop parser. Existing unauthenticated reports "
            "remain unauthenticated; a format survey does not prove source "
            "execution, operator identity, achieved loop II, synthesis "
            "causality, numerical acceptance or RTL behavior."
        ),
    }


def export_schedule_report_formats(suite_directory, inventory_file, output):
    destination = Path(output)
    source = Path(inventory_file)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Native report inventory must be an existing regular JSON")
    if source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Native inventory size exceeds bound")
    inventory = json.loads(source.read_text())
    report = survey_schedule_report_formats(suite_directory, inventory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        raise FileExistsError("Do not overwrite immutable scheduling format surveys")
    with TemporaryDirectory(prefix=".fpgai-hls-layout-", dir=destination.parent) as folder:
        staged = Path(folder)
        (staged / "hls_report_format_survey.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
        lines = [
            "# Vitis HLS report format survey (exploratory only)", "",
            f"Cases: {report['case_count']}; reports surveyed: {report['surveyed_report_count']}.",
            "No old report was re-attested. No loop II or latency was extracted.", "",
            "## Provenance classifications", "",
        ]
        for status, count in report["status_counts"].items():
            lines.append(f"- {status}: {count}")
        lines.extend(["", "## Format fingerprints and header clues", ""])
        for case in report["cases"]:
            if not case["surveyed"]:
                continue
            lines.append(
                f"- {case['model']} / {case['case']} ({case['axis']}): "
                f"{case['possible_loop_heading_lines']} possible loop headings, "
                f"{case['possible_schedule_table_headers']} possible table headers; "
                f"format hash {case['report_format_digest']}"
            )
            for layout in case["possible_column_layouts"][:5]:
                lines.append(
                    f"  - columns: {', '.join(layout['columns'])}; "
                    f"occurrences: {layout['occurrences']}"
                )
        lines.extend(["", "## Within-model report-format contrasts", "",
            "These compare header/layout fingerprints only, **not** achieved "
            "loop II, numerical outcomes, scheduling equivalence or causality.",""])
        for item in report["within_model_layout_contrasts"]:
            lines.append(
                f"- {item['model']} {item['axis']}: "
                f"{item['format_comparison_status']}; "
                f"possible loop/iteration headers "
                f"{item['baseline_possible_loop_table_headers']} → "
                f"{item['variant_possible_loop_table_headers']}")
        lines.extend(["", "## Interpretation limit", "", report["limitations"], ""])
        (staged / "hls_report_format_survey.md").write_text("\n".join(lines))
        staged.rename(destination)
    return report
