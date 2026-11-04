"""Read-only publication triage for unresolved operator-local scheduling evidence.

Join source-verified IR/HLS questions, immutable raw-report provenance and
redacted Vitis-2023.2 loop-table classifications. Report *gaps*, not II.
No report bytes, loop/operator identifiers or numerical HLS results exported.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.ir_research_questions import (
    analyze_ir_research_questions,
)
from fpgai_research.experiments.hls_loop_table_diagnostics import SCHEMA as DIAG_SCHEMA
from fpgai_research.experiments.native_schedule_reports import SCHEMA as INVENTORY_SCHEMA

SCHEMA = "fpgai-research/loop-attribution-gap-triage/v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9_]{1,80}\Z")
_REPORT_STATUS = frozenset({
    "PRESENT_NOT_FRESHLY_ATTESTED", "PRESENT_NOT_ATTESTED",
    "NATIVE_TEXT_WITHOUT_E3_QUALIFICATION", "ATTESTED_RAW_REPORT_AVAILABLE",
    "REPORT_NOT_PRESENT", "REPORT_EMPTY_OR_TOO_LARGE", "HASH_MISMATCH",
    "UNSAFE_SYMLINKED_REPORT",
})
_CSV_COLUMNS = (
    "model", "case", "axis", "observation",
    "baseline_table_status", "variant_table_status",
    "baseline_provenance", "variant_provenance",
    "operator_ii_established", "causal_explanation_verified",
    "next_evidence_step",
)


def _key(row: dict) -> tuple[str, str]:
    if not isinstance(row, dict):
        raise ValueError("Invalid case record")
    model, case = row.get("model"), row.get("case")
    if (not isinstance(model, str) or not isinstance(case, str)
            or not _SAFE_ID.fullmatch(model) or not _SAFE_ID.fullmatch(case)):
        raise ValueError("Unsafe or missing model/case identifier")
    return model, case


def _index(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected nonempty case inventory")
    out = {}
    for item in rows:
        key = _key(item)
        if key in out:
            raise ValueError("Duplicate model/case in evidence")
        out[key] = item
    return out


def _structure(row: dict | None) -> str:
    if row is None:
        return "NOT_IN_SELECTED_DIAGNOSTIC"
    if not row.get("classified"):
        return "REPORT_NOT_CLASSIFIED"
    if row.get("table_limit_reached") or row.get("row_limit_reached"):
        return "TRUNCATED_OR_INCOMPLETE_TABLE_SCAN"
    tables, scanned = row.get("paired_loop_table_count"), row.get("candidate_rows_scanned")
    counts = row.get("row_status_counts")
    if (type(tables) is not int or tables < 0 or type(scanned) is not int
            or scanned < 0 or not isinstance(counts, dict)
            or any(type(value) is not int or value < 0 for value in counts.values())):
        raise ValueError("Invalid diagnostic table counts")
    valid = counts.get("SHAPE_VALID_ROWS", 0)
    malformed = counts.get("MALFORMED_OR_AMBIGUOUS_ROW", 0)
    if valid + malformed != scanned:
        raise ValueError("Diagnostic row partition is inconsistent")
    achieved = sum(counts.get(field, 0) for field in (
        "ACHIEVED_UNAVAILABLE", "ACHIEVED_UNKNOWN",
        "NUMERIC_ACHIEVED_FIELD_UNATTRIBUTED"))
    if valid != achieved:
        raise ValueError("Diagnostic achieved-field partition is inconsistent")
    if tables == 0:
        if scanned:
            raise ValueError("Rows cannot exist without paired loop tables")
        return "NO_PAIRED_LOOP_TABLE"
    if scanned == 0:
        return "NO_RECOGNIZED_LOOP_DATA_ROWS"
    if malformed:
        return "AMBIGUOUS_OR_MALFORMED_LOOP_ROW"
    if counts.get("NUMERIC_ACHIEVED_FIELD_UNATTRIBUTED", 0):
        return "NUMERIC_ACHIEVED_CANDIDATE_NOT_OPERATOR_BOUND"
    if counts.get("ACHIEVED_UNKNOWN", 0):
        return "ACHIEVED_FIELD_UNKNOWN"
    if counts.get("ACHIEVED_UNAVAILABLE", 0):
        return "ACHIEVED_FIELD_UNAVAILABLE"
    return "NO_VERIFIED_LOOP_II"


def _evidence_step(left: dict, right: dict, left_shape: str, right_shape: str) -> str:
    statuses = {left["status"], right["status"]}
    if statuses & {"HASH_MISMATCH", "UNSAFE_SYMLINKED_REPORT"}:
        return "REPAIR_SOURCE_OR_REPORT_PROVENANCE"
    if statuses & {"REPORT_NOT_PRESENT", "REPORT_EMPTY_OR_TOO_LARGE"}:
        return "GENERATE_MISSING_NATIVE_HLS_REPORT"
    if statuses != {"ATTESTED_RAW_REPORT_AVAILABLE"}:
        return "REEXECUTE_VITIS_WITH_FRESH_REPORT_DIGEST"
    shapes = {left_shape, right_shape}
    if "NOT_IN_SELECTED_DIAGNOSTIC" in shapes or "REPORT_NOT_CLASSIFIED" in shapes:
        return "CLASSIFY_BOTH_NATIVE_REPORTS"
    if shapes & {"NO_PAIRED_LOOP_TABLE", "NO_RECOGNIZED_LOOP_DATA_ROWS",
                 "ACHIEVED_FIELD_UNAVAILABLE", "ACHIEVED_FIELD_UNKNOWN",
                 "TRUNCATED_OR_INCOMPLETE_TABLE_SCAN",
                 "AMBIGUOUS_OR_MALFORMED_LOOP_ROW"}:
        return "OBTAIN_REVIEWED_NATIVE_LOOP_SCHEDULING_SOURCE"
    return "VALIDATE_LOOP_TO_GENERATED_SOURCE_AND_OPERATOR_BINDING"


def analyze_loop_attribution_gaps(schedule: dict, inventory: dict,
                                  diagnostics: dict) -> dict:
    """Recheck source provenance and diagnose comparisons without claiming II."""
    questions = analyze_ir_research_questions(schedule, inventory)
    if diagnostics.get("schema") != DIAG_SCHEMA or inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("Unexpected diagnostics or inventory schema")
    suite_sha = questions["source_suite_sha256"]
    if diagnostics.get("source_suite_sha256") != suite_sha:
        raise ValueError("Diagnostic data belong to a different pinned suite")
    native = _index(inventory.get("cases"))
    diagnostic = _index(diagnostics.get("cases"))
    if (len(native) != inventory.get("case_count")
            or len(diagnostic) != diagnostics.get("case_count")
            or not diagnostic.keys() <= native.keys()
            or sum(bool(x.get("classified")) for x in diagnostic.values())
               != diagnostics.get("classified_report_count")):
        raise ValueError("Inconsistent native or selected diagnostic inventory")
    if diagnostics.get("existing_evidence_upgraded") is not False or (
            diagnostics.get("per_loop_schedule_evidence_available") is not False):
        raise ValueError("Exploratory diagnostics must not assert schedule qualification")
    for key, row in diagnostic.items():
        saved = native[key]
        if (row.get("axis") != saved.get("axis")
                or row.get("provenance_status") != saved.get("status")
                or row.get("report_sha256") != saved.get("report_sha256")
                or saved.get("status") not in _REPORT_STATUS):
            raise ValueError("Diagnostic and original native report identity differ")
        _structure(row)
    rows = []
    counts = Counter()
    for question in questions["questions"]:
        key = _key(question)
        baseline = (key[0], question.get("baseline_case"))
        if baseline not in native:
            raise ValueError("Missing verified baseline")
        a, b = native[baseline], native[key]
        if (a.get("status") != question.get("baseline_raw_report_status")
                or b.get("status") != question.get("variant_raw_report_status")):
            raise ValueError("Question and native evidence status disagree")
        base_shape = _structure(diagnostic.get(baseline))
        variant_shape = _structure(diagnostic.get(key))
        step = _evidence_step(a, b, base_shape, variant_shape)
        row = {
            "model":key[0], "case":key[1], "axis":question["axis"],
            "observation":question["observation"],
            "baseline_table_status":base_shape,
            "variant_table_status":variant_shape,
            "baseline_provenance":a["status"],
            "variant_provenance":b["status"],
            "operator_ii_established":False,
            "causal_explanation_verified":False,
            "next_evidence_step":step,
        }
        rows.append(row)
        counts[step] += 1
    return {
        "schema":SCHEMA, "source_suite_sha256":suite_sha,
        "verified_question_count":len(rows),
        "diagnostic_case_count":len(diagnostic),
        "diagnostic_case_statuses":{
            f"{model}/{case}":_structure(row)
            for (model, case), row in sorted(diagnostic.items())
        },
        "evidence_step_counts":dict(sorted(counts.items())),
        "comparisons":rows,
        "native_tools_executed":False,
        "existing_evidence_upgraded":False,
        "numeric_loop_ii_exported":False,
        "operator_names_exported":False,
        "operator_schedule_attribution_verified":False,
        "limitations":(
            "This is a source- and receipt-consistent E0 diagnostic of gaps "
            "between IR/HLS questions and legacy/native loop report grammar. "
            "No numeric operator II, private loop name, causal bottleneck or "
            "new E2/E3/E4 measurement is inferred. Recommended reruns are "
            "planning suggestions, not automatic tool execution."
        ),
    }


def _load(path: str | Path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Expected a regular, bounded input JSON file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid evidence JSON")
    return data


def export_loop_attribution_gaps(schedule_path, inventory_path,
                                  diagnostic_path, output) -> dict:
    result = analyze_loop_attribution_gaps(
        _load(schedule_path), _load(inventory_path), _load(diagnostic_path))
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Never overwrite an existing immutable evidence export")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-attribution-gap-", dir=target.parent) as folder:
        stage = Path(folder)
        (stage / "loop_attribution_gaps.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        with (stage / "loop_attribution_gaps.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            for row in result["comparisons"]:
                writer.writerow({k:row[k] for k in _CSV_COLUMNS})
        lines = [
            "# MLP/CNN IR-to-native loop attribution: evidence gaps", "",
            f"Compared variants: {result['verified_question_count']}; "
            f"diagnostic cases examined: {result['diagnostic_case_count']}.",
            "**No numeric achieved loop II or causal explanation is established.**", "",
            "| Model | Case | Axis | Baseline loop table | Variant loop table | Next evidence step |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in result["comparisons"]:
            lines.append("| " + " | ".join(str(row[k]) for k in (
                "model", "case", "axis", "baseline_table_status",
                "variant_table_status", "next_evidence_step")) + " |")
        lines.extend(["", "## Limits", "", result["limitations"], ""])
        (stage / "loop_attribution_gaps.md").write_text(
            "\n".join(lines), encoding="utf-8")
        stage.rename(target)
    return result
