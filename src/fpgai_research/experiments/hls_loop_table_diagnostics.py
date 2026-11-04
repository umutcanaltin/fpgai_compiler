"""Conservative Vitis HLS 2023.2 loop-table shape classifier.

Exploratory only: detects the paired multi-line report grammar, classifies
anonymous row cells, and NEVER exports numeric scheduling values or loop names.
Report provenance is inherited from the independent source/digest inventory;
no parse can upgrade legacy E3/E4 evidence or bind a loop to an IR operator.

The grammar follows AMD XD099 Vitis 2023.2 loop-table examples:
https://docs.amd.com/r/2023.2-English/Vitis-Tutorials-Hardware-Acceleration/Vitis-HLS-for-Kernel-Optimizations
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls_report_format_survey import _cells
from fpgai_research.experiments.hls_report_schema_probe import (
    _role, _type, probe_hls_report_grammar,
)
from fpgai_research.experiments.native_schedule_reports import REPORT_LIMIT, REPORT_REL

SCHEMA = "fpgai-research/vitis-2023.2-loop-table-diagnostics/v1"
HEADER = (
    "LOOP_HEADER", "MIN_HEADER", "MAX_HEADER", "LATENCY_HEADER",
    "ACHIEVED_HEADER", "TARGET_HEADER", "COUNT_HEADER", "PIPELINED_HEADER",
)
MAX_TABLES = 16
MAX_ROWS_PER_TABLE = 512
MAX_LINES = 100_000
_NONNEGATIVE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_ALLOWED_GENERAL = frozenset(("NUMBER", "NUMBER_RANGE", "UNKNOWN", "UNAVAILABLE"))


def _is_separator(cells: list[str]) -> bool:
    return all(_type(cell) == "SEPARATOR" for cell in cells)


def _is_ascii_divider(line: str) -> bool:
    """Vitis uses +---+ table borders between headers and data rows."""
    return bool(line.strip()) and all(
        char in "+-|=:" or char.isspace() for char in line
    )


def _has_grouped_header(lines: list[str], index: int) -> bool:
    """Require the preceding Vitis two-line group header, not any 8-col table."""
    if index <= 0:
        return False
    preceding = _cells(lines[index - 1])
    if preceding is None:
        return False
    roles = [_role(cell) for cell in preceding]
    return (len(roles) >= 4 and "LATENCY_HEADER" in roles and
            "INTERVAL_HEADER" in roles and
            roles.index("LATENCY_HEADER") < roles.index("INTERVAL_HEADER"))


def _interval_kind(value: str) -> str:
    value = value.strip()
    if value == "-":
        return "UNAVAILABLE"
    if value == "?":
        return "UNKNOWN"
    if _NONNEGATIVE.fullmatch(value) and int(value) >= 1:
        return "POSITIVE_INTEGER_CANDIDATE"
    return "INVALID_OR_AMBIGUOUS"


def _row_shape(cells: list[str]) -> dict:
    """Never return a data value, even for freshly attested report bytes."""
    kinds = [_type(cell) for cell in cells]
    achieved = _interval_kind(cells[4])
    target = _interval_kind(cells[5])
    general_valid = (
        bool(cells[0].strip())
        and all(kinds[i] in _ALLOWED_GENERAL for i in (1, 2, 3, 6))
        and kinds[7] == "BOOLEAN"
    )
    return {
        "valid_shape":bool(general_valid and achieved != "INVALID_OR_AMBIGUOUS"
                           and target != "INVALID_OR_AMBIGUOUS"),
        "achieved_field":achieved,
        "target_field":target,
        "numeric_fields_exported":False,
    }


def inspect_vitis_2023_loop_tables(data: bytes) -> dict:
    """Read-only structural classification of *one* human-readable report."""
    if not isinstance(data, bytes) or not data or len(data) > REPORT_LIMIT:
        raise ValueError("HLS report is empty or exceeds the report size limit")
    try:
        lines = data.decode("utf-8", "strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("HLS report is not valid UTF-8") from exc
    if len(lines) > MAX_LINES:
        raise ValueError("HLS report exceeds the line-count bound")
    counts: Counter[str] = Counter()
    table_count = 0
    grouped_count = 0
    rows_scanned = 0
    table_limit_reached = False
    row_limit_reached = False
    for index, line in enumerate(lines):
        cells = _cells(line)
        if cells is None or tuple(_role(cell) for cell in cells) != HEADER:
            continue
        table_count += 1
        if table_count > MAX_TABLES:
            table_limit_reached = True
            break
        if not _has_grouped_header(lines, index):
            counts["HEADER_WITHOUT_GROUPING"] += 1
            continue
        grouped_count += 1
        examined = 0
        for later in lines[index + 1:]:
            if _is_ascii_divider(later):
                continue
            row = _cells(later)
            if row is None or len(row) != len(HEADER):
                break
            if _is_separator(row):
                continue
            if tuple(_role(cell) for cell in row) == HEADER:
                break
            if examined >= MAX_ROWS_PER_TABLE:
                row_limit_reached = True
                break
            examined += 1
            rows_scanned += 1
            shape = _row_shape(row)
            if not shape["valid_shape"]:
                counts["MALFORMED_OR_AMBIGUOUS_ROW"] += 1
            else:
                achieved, target = shape["achieved_field"], shape["target_field"]
                if achieved == "POSITIVE_INTEGER_CANDIDATE":
                    counts["NUMERIC_ACHIEVED_FIELD_UNATTRIBUTED"] += 1
                elif achieved == "UNAVAILABLE":
                    counts["ACHIEVED_UNAVAILABLE"] += 1
                else:
                    counts["ACHIEVED_UNKNOWN"] += 1
                if target == "POSITIVE_INTEGER_CANDIDATE":
                    counts["NUMERIC_TARGET_FIELD_UNATTRIBUTED"] += 1
                elif target == "UNAVAILABLE":
                    counts["TARGET_UNAVAILABLE"] += 1
                else:
                    counts["TARGET_UNKNOWN"] += 1
                counts["SHAPE_VALID_ROWS"] += 1
        if row_limit_reached:
            break
    return {
        "table_header_count":min(table_count, MAX_TABLES),
        "paired_loop_table_count":grouped_count,
        "candidate_rows_scanned":rows_scanned,
        "row_status_counts":dict(sorted(counts.items())),
        "table_limit_reached":table_limit_reached,
        "row_limit_reached":row_limit_reached,
        "format_assumption":"Vitis HLS 2023.2 paired loop table (provisional)",
        "numeric_schedule_values_exported":False,
        "loop_identifiers_exported":False,
        "operator_mapping_verified":False,
        "per_loop_schedule_evidence_available":False,
    }


def classify_vitis_loop_tables(suite_directory, inventory_document, *,
                               selected_models: tuple[str, ...] = (),
                               selected_cases: tuple[str, ...] = ()) -> dict:
    """Reuse strict survey inventory checks; never upgrade any report status."""
    survey = probe_hls_report_grammar(
        suite_directory, inventory_document,
        selected_models=selected_models, selected_cases=selected_cases)
    root = Path(suite_directory)
    records = []
    counts: Counter[str] = Counter()
    for item in survey["cases"]:
        entry = {
            "model":item["model"], "case":item["case"], "axis":item["axis"],
            "provenance_status":item["provenance_status"],
            "report_sha256":item["report_sha256"],
            "classified":False,
            "per_loop_schedule_evidence_available":False,
        }
        if item["grammars_probed"]:
            path = root / item["model"] / item["case"] / REPORT_REL
            # Recheck all path components and digest after the earlier survey.
            for parent in (root / item["model"], root / item["model"] / item["case"]):
                if parent.is_symlink():
                    raise ValueError("Linked model/case path is unsafe")
            node = root / item["model"] / item["case"]
            for part in REPORT_REL.parts:
                node = node / part
                if node.is_symlink():
                    raise ValueError("Linked report path is unsafe")
            data = path.read_bytes()
            if sha256(data).hexdigest() != entry["report_sha256"]:
                raise ValueError("HLS report changed after inventory validation")
            entry.update(inspect_vitis_2023_loop_tables(data))
            entry["classified"] = True
            for key, value in entry["row_status_counts"].items():
                counts[key] += value
        records.append(entry)
    return {
        "schema":SCHEMA,
        "source_suite_sha256":survey["source_suite_sha256"],
        "case_count":len(records),
        "classified_report_count":sum(bool(e["classified"]) for e in records),
        "status_counts":dict(sorted(counts.items())),
        "cases":records,
        "native_tools_executed":False,
        "existing_evidence_upgraded":False,
        "numeric_schedule_values_exported":False,
        "operator_names_exported":False,
        "per_loop_schedule_evidence_available":False,
        "limitations":(
            "Vitis 2023.2 loop-table shape classification is provisional. "
            "Even a numeric ACHIEVED cell is not published as loop II: source "
            "freshness, exact backend version, row-to-generated-loop mapping, "
            "operator identity and independent field validation are required. "
            "A dash or question mark is not an achieved II. "
            "No original native report, metric or loop name is exported."
        ),
    }


def export_vitis_loop_table_diagnostics(suite_directory, inventory_path, output, *,
                                        selected_models: tuple[str, ...] = (),
                                        selected_cases: tuple[str, ...] = ()) -> dict:
    path = Path(inventory_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Inventory must be a normal bounded JSON file")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    result = classify_vitis_loop_tables(
        suite_directory, inventory,
        selected_models=selected_models, selected_cases=selected_cases)
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Do not overwrite immutable loop-table diagnostics")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-loop-grammar-", dir=destination.parent) as folder:
        stage = Path(folder)
        (stage / "loop_table_diagnostics.json").write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        lines = [
            "# Provisional Vitis 2023.2 loop-table diagnostic", "",
            f"Reports: {result['classified_report_count']} / {result['case_count']}.",
            "**No achieved II, loop names, operator identifiers or latency values exported.**", "",
        ]
        for item in result["cases"]:
            lines.extend([f"## {item['model']} / {item['case']}",
                          f"Provenance: {item['provenance_status']}"])
            if item["classified"]:
                lines.extend([
                    f"Paired tables: {item['paired_loop_table_count']}; "
                    f"candidate rows: {item['candidate_rows_scanned']}",
                    "Field-shape counts: " + json.dumps(item["row_status_counts"], sort_keys=True),
                ])
                if item["table_limit_reached"] or item["row_limit_reached"]:
                    lines.append("**Warning: parsing bound reached; results incomplete.**")
            else:
                lines.append("No source-verified report available to classify.")
            lines.append("")
        lines.extend(["## Evidence boundary", "", result["limitations"], ""])
        (stage / "loop_table_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
        stage.rename(destination)
    return result
