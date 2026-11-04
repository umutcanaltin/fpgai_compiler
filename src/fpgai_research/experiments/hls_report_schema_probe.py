"""Redacted Vitis 2023.2 HLS table grammar probe (E0 exploratory ONLY).

The report text can contain private operator paths, code filenames,
weight values or confidential IP. Export only whitelisted header roles,
generic data-cell *types*, and bounded layout/line-position information.

Every input is source/hash/provenance checked against the existing native
schedule inventory. Legacy files are NEVER re-attested and the output
NEVER contains achieved loop II, latency, or inferred schedule causality.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls_report_format_survey import (
    _cells, _tokens, survey_schedule_report_formats,
)
from fpgai_research.experiments.native_schedule_reports import REPORT_REL, REPORT_LIMIT

SCHEMA = "fpgai-research/hls-redacted-table-grammar/v1"
MAX_TABLES = 12
MAX_CONTEXT = 3
MAX_ROWS = 4
_ALLOWED_STATUSES = frozenset((
    "ATTESTED_RAW_REPORT_AVAILABLE", "PRESENT_NOT_FRESHLY_ATTESTED",
    "PRESENT_NOT_ATTESTED", "NATIVE_TEXT_WITHOUT_E3_QUALIFICATION",
))
_NUM = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?")
_RANGE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*[~\-]\s*"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
)
_ASCII_BAR = re.compile(r"[-+=: ]+")


def _role(cell: str) -> str:
    """Only permitted header vocabulary; never return caller's text."""
    lower = " ".join(cell.lower().strip().split())
    categories = _tokens(lower)
    if "iteration" in categories and "latency" in categories:
        return "ITERATION_LATENCY_HEADER"
    if "iteration" in categories and "interval" in categories:
        return "ITERATION_INTERVAL_HEADER"
    if "tripcount" in categories:
        return "TRIP_COUNT_HEADER"
    if "latency" in categories:
        return "LATENCY_HEADER"
    if "interval" in categories:
        return "INTERVAL_HEADER"
    if "pipeline" in categories:
        return "PIPELINED_HEADER"
    if "loop" in categories:
        return "LOOP_HEADER"
    # _tokens() intentionally recognizes uppercase standalone "II" in raw
    # survey text, but _role() normalizes headers to lowercase first.
    # Accept the normalized isolated column explicitly, never arbitrary text.
    if lower == "ii" or "ii" in categories:
        return "II_HEADER"
    if "performance" in categories:
        return "PERFORMANCE_HEADER"
    if lower in ("min", "minimum", "best", "best case", "best-case"):
        return "MIN_HEADER"
    if lower in ("max", "maximum", "worst", "worst case", "worst-case"):
        return "MAX_HEADER"
    if lower == "target":
        return "TARGET_HEADER"
    if lower == "achieved":
        return "ACHIEVED_HEADER"
    if lower == "count":
        return "COUNT_HEADER"
    if lower == "name":
        return "NAME_HEADER"
    if lower in ("pipelined", "yes/no"):
        return "PIPELINED_HEADER"
    return "OTHER_HEADER"


def _type(cell: str) -> str:
    """Return a coarse structural category, never literal cell content."""
    cell = cell.strip()
    if not cell:
        return "EMPTY"
    if cell == "?":
        return "UNKNOWN"
    if cell == "-":
        return "UNAVAILABLE"
    if _ASCII_BAR.fullmatch(cell):
        return "SEPARATOR"
    if _NUM.fullmatch(cell):
        return "NUMBER"
    if _RANGE.fullmatch(cell):
        return "NUMBER_RANGE"
    if cell.lower() in ("yes", "no", "true", "false"):
        return "BOOLEAN"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-:. /+]*", cell):
        return "TEXT"
    return "MIXED"


def _is_candidate(cells: list[str]) -> bool:
    roles = [_role(x) for x in cells]
    schedule = sum(x in {
        "LATENCY_HEADER", "ITERATION_LATENCY_HEADER",
        "INTERVAL_HEADER", "ITERATION_INTERVAL_HEADER",
        "PIPELINED_HEADER", "II_HEADER"
    } for x in roles)
    return schedule >= 1 and sum(x != "OTHER_HEADER" for x in roles) >= 2


def _header_context(lines: list[str], index: int) -> list[dict]:
    """Capture normalized nearby header lines; their text is discarded."""
    out=[]
    for position in range(max(0, index-MAX_CONTEXT), min(len(lines),index+MAX_CONTEXT+1)):
        cells=_cells(lines[position])
        if cells is None or not _is_candidate(cells):
            continue
        out.append({
            "relative_line":position-index,
            "column_count":len(cells),
            "column_roles":[_role(c) for c in cells],
        })
    return out


def _row_shapes(lines: list[str], index: int, expected_cells: int) -> list[dict]:
    """Capture only the first four adjacent table-row grammars."""
    rows=[]
    for line in lines[index+1:index+MAX_ROWS+MAX_CONTEXT+3]:
        cells=_cells(line)
        if cells is None:
            if rows:
                break
            continue
        if len(cells)!=expected_cells:
            if rows:
                break
            continue
        roles=[_role(x) for x in cells]
        if _is_candidate(cells):
            break
        kinds=[_type(x) for x in cells]
        if all(x=="SEPARATOR" for x in kinds):
            continue
        rows.append({
            "cell_types":kinds,
            "numeric_like_columns":[i for i,k in enumerate(kinds)
                                    if k in ("NUMBER","NUMBER_RANGE")],
            "unavailable_columns":[i for i,k in enumerate(kinds)
                                   if k in ("UNKNOWN","UNAVAILABLE")],
        })
        if len(rows)>=MAX_ROWS:
            break
    return rows


def _examine(data: bytes) -> dict:
    if not data or len(data)>REPORT_LIMIT:
        raise ValueError("Legacy report is empty or unbounded")
    text=data.decode("utf-8",errors="replace")
    lines=text.splitlines()
    if len(lines)>100_000:
        raise ValueError("Too many legacy HLS report lines")
    sites=[]
    for number,line in enumerate(lines):
        cells=_cells(line)
        if cells is None or not _is_candidate(cells):
            continue
        if len(sites)>=MAX_TABLES:
            break
        sites.append({
            "source_line_1based":number+1,
            "column_count":len(cells),
            "sanitized_header_roles":[_role(x) for x in cells],
            "adjacent_header_shapes":_header_context(lines,number),
            "sample_row_grammars":_row_shapes(lines,number,len(cells)),
        })
    return {
        "table_probe_count":len(sites),
        "truncated_at_table_limit":len(sites)==MAX_TABLES,
        "table_grammars":sites,
        "numeric_schedule_values_exported":False,
        "operator_names_exported":False,
        "source_text_exported":False,
    }


def probe_hls_report_grammar(suite_directory,inventory_document, *,
                             selected_models:tuple[str,...]=(),
                             selected_cases:tuple[str,...]=()) -> dict:
    """Read-only, digest-checked schema probe over previously surveyed reports."""
    survey=survey_schedule_report_formats(suite_directory,inventory_document)
    suite=Path(suite_directory)
    models=set(selected_models)
    cases=set(selected_cases)
    if any(not re.fullmatch(r"[A-Za-z0-9_]{1,80}",v) for v in models|cases):
        raise ValueError("Unsafe model/case filter")
    existing_models={row["model"] for row in survey["cases"]}
    existing_cases={row["case"] for row in survey["cases"]}
    if not models.issubset(existing_models) or not cases.issubset(existing_cases):
        raise ValueError("Requested model/case is not in the pinned inventory")

    records=[]
    layout_counter:Counter[str]=Counter()
    for row in survey["cases"]:
        if models and row["model"] not in models:
            continue
        if cases and row["case"] not in cases:
            continue
        record={
            "model":row["model"],"case":row["case"],"axis":row["axis"],
            "report_sha256":row.get("report_sha256"),
            "provenance_status":row["provenance_status"],
            "grammars_probed":False,
            "per_loop_schedule_attested":False,
        }
        if row["surveyed"] and row["provenance_status"] in _ALLOWED_STATUSES:
            path=suite/row["model"]/row["case"]/REPORT_REL
            # The previous survey independently checked all components
            # for symlink escapes, pinned original source and report digest.
            # Recheck the actual bytes to fail if they changed mid-run.
            data=path.read_bytes()
            if sha256(data).hexdigest()!=row["report_sha256"]:
                raise ValueError("HLS source report changed after layout survey")
            record.update(_examine(data))
            record["grammars_probed"]=True
            # Store only a hash of the sanitized grammar for grouping.
            digest=sha256(json.dumps(
                record["table_grammars"],sort_keys=True,
                separators=(",",":")).encode()).hexdigest()
            record["redacted_grammar_sha256"]=digest
            layout_counter[digest]+=1
        records.append(record)
    if not records:
        raise ValueError("Model/case selection produced no inventory rows")
    return {
        "schema":SCHEMA,
        "source_suite_sha256":survey["source_suite_sha256"],
        "case_count":len(records),
        "probed_report_count":sum(bool(row["grammars_probed"]) for row in records),
        "distinct_redacted_grammars":len(layout_counter),
        "redacted_grammar_counts":dict(sorted(layout_counter.items())),
        "cases":records,
        "native_tools_executed":False,"weight_values_exported":False,
        "existing_evidence_upgraded":False,
        "per_loop_schedule_evidence_available":False,
        "limitations":(
            "Exploratory table-grammar probe of immutable native HLS text "
            "only. Header roles and row data types are heuristics. "
            "Legacy reports retain their original provenance status. "
            "Numbers, operator names and raw text are not exported. "
            "No per-loop achieved II, causality, numerical or RTL claims."
        ),
    }


def export_hls_report_grammar(suite_directory,inventory_path,output, *,
                              selected_models:tuple[str,...]=(),
                              selected_cases:tuple[str,...]=()) -> dict:
    inventory_file=Path(inventory_path)
    if inventory_file.is_symlink() or not inventory_file.is_file():
        raise ValueError("Inventory must be a normal JSON file")
    if inventory_file.stat().st_size>16*1024*1024:
        raise ValueError("Inventory is too large")
    data=json.loads(inventory_file.read_text())
    result=probe_hls_report_grammar(suite_directory,data,
                                    selected_models=selected_models,
                                    selected_cases=selected_cases)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Cannot overwrite immutable report grammar probe")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls-grammar-",dir=target.parent) as folder:
        stage=Path(folder)
        (stage/"hls_table_grammar.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=[
            "# Vitis HLS table grammar: redacted and exploratory", "",
            f"Cases selected: {result['case_count']}; reports examined: "
            f"{result['probed_report_count']}.", "",
            "**No report was re-attested. No numerical II/latency or private "
            "loop name was exported.**", "",
        ]
        for row in result["cases"]:
            lines.extend([
                "## "+row["model"]+" / "+row["case"]+" ("+str(row["axis"])+")",
                "Provenance: "+row["provenance_status"],"",
            ])
            if not row["grammars_probed"]:
                lines.extend(["Report grammar unavailable.",""])
                continue
            for site in row["table_grammars"]:
                lines.append(
                    "Candidate header at source line "
                    +str(site["source_line_1based"])+": "
                    +" | ".join(site["sanitized_header_roles"]))
                for neighbor in site["adjacent_header_shapes"]:
                    if neighbor["relative_line"]==0:
                        continue
                    lines.append(
                        "  adjacent header offset "
                        +str(neighbor["relative_line"])+": "
                        +" | ".join(neighbor["column_roles"]))
                for sample in site["sample_row_grammars"]:
                    lines.append("  redacted row: "
                                 +" | ".join(sample["cell_types"]))
            lines.append("")
        lines.extend(["## Evidence boundary","",result["limitations"],""])
        (stage/"hls_table_grammar.md").write_text("\n".join(lines))
        stage.rename(target)
    return result
