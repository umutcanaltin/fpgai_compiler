"""Safely stage isolated source-identical Vitis HLS replays and audit them.

Never re-run HLS inside a historically routed E4 campaign. The generated
replay campaigns contain only verified E1 inputs and independent new native
output directories, preserving original Vitis and Vivado receipts untouched.
They contain PRIVATE trained model weights and testbench data: do not publish.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_hls_rerun_plan import (
    _CASE, _read_json, plan_hls_reruns,
)
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.native_schedule_reports import (
    REPORT_LIMIT, REPORT_REL, inspect_native_schedule_case,
)
from fpgai_research.experiments.hls_loop_table_diagnostics import (
    inspect_vitis_2023_loop_tables,
)
from fpgai_research.experiments.hls_report_schema_probe import _examine
from fpgai_research.toolchains.vitis_hls.package import make_vitis_tcl

SCHEMA = "fpgai-research/isolated-native-hls-replay/v1"
AUDIT_SCHEMA = "fpgai-research/isolated-native-hls-replay-audit/v1"
_COPY_FILES = {
    "model.cpp": 64*1024*1024,
    "tb.cpp": 32*1024*1024,
    "run_hls.tcl": 128*1024,
    "manifest.json": 4*1024*1024,
    "progressive_ir.json": 64*1024*1024,
}


def _regular_bytes(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing, linked or oversized replay input: " + path.name)
    return path.read_bytes()


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _emit_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")


def stage_isolated_hls_replay(suite_directory, inventory_path,
                              triage_path, destination) -> dict:
    """Create only the 8 focused E1 inputs as private independent campaigns."""
    root=Path(suite_directory)
    plan=plan_hls_reruns(
        root, _read_json(Path(inventory_path)), _read_json(Path(triage_path)))
    target=Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Replay workspace exists; never overwrite native data")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls-replay-",dir=target.parent) as temp:
        stage=Path(temp)
        studies=[]
        for group in plan["models"]:
            model=group["model"]
            old_dir=root/model
            if old_dir.is_symlink() or not old_dir.is_dir():
                raise ValueError("Symlinked or missing original model campaign")
            campaign_path=old_dir/"campaign.json"
            original=_read_json(campaign_path)
            if _digest(campaign_path.read_bytes())!=group["campaign_sha256"]:
                raise ValueError("Pinned campaign changed since replay planning")
            origin={row["case"]:row for row in original["cases"]}
            dest_dir=stage/model
            dest_dir.mkdir()
            cases=[]
            records=[]
            for selected in group["selected_cases"]:
                case=selected["case"]
                prepared=origin.get(case)
                if (not isinstance(prepared,dict)
                        or prepared.get("status")!="READY"
                        or prepared.get("dimension")!=selected["axis"]
                        or prepared.get("cpp_sha256")!=selected["source_sha256"]):
                    raise ValueError("Replay case differs from prepared source/axis")
                src=old_dir/case
                if src.is_symlink() or not src.is_dir():
                    raise ValueError("Missing or linked original case")
                dst=dest_dir/case
                dst.mkdir()
                digests={}
                for name,limit in _COPY_FILES.items():
                    source=src/name
                    if name=="progressive_ir.json" and not source.exists() and not source.is_symlink():
                        continue
                    data=_regular_bytes(source,limit)
                    (dst/name).write_bytes(data)
                    digests[name]=_digest(data)
                manifest=json.loads((dst/"manifest.json").read_text(encoding="utf-8"))
                if (not isinstance(manifest,dict)
                        or manifest.get("cpp_sha256")!=selected["source_sha256"]
                        or manifest.get("testbench_sha256")!=digests["tb.cpp"]
                        or digests["model.cpp"]!=selected["source_sha256"]):
                    raise ValueError("HLS source/testbench manifest differs from frozen inputs")
                # Use only generator-produced Tcl; no arbitrary extra Tcl commands
                # and no absolute paths referring back to the original study.
                if (dst/"run_hls.tcl").read_text(encoding="utf-8")!=make_vitis_tcl(
                        manifest.get("part"),manifest.get("clock_ns")):
                    raise ValueError("HLS script differs from canonical generated Tcl")
                from fpgai_research.experiments.results import summarize_case
                summary=summarize_case(str(dst))
                if summary["source_sha256"]!=selected["source_sha256"] or (
                        summary["evidence"]!="E1-HLS-source"):
                    raise ValueError("Fresh isolated replay must begin from verified E1")
                cases.append(prepared)
                records.append({
                    "case":case,"axis":selected["axis"],
                    "original_ready_index":selected["ready_index"],
                    "source_sha256":selected["source_sha256"],
                    "copied_file_sha256":digests,
                    "original_report_status":selected["report_status_before_rerun"],
                    "replay_evidence_at_creation":"E1-HLS-source",
                })
            _emit_json(dest_dir/"campaign.json",{
                "schema":"fpgai-research/paper-campaign/v1","cases":cases,
            })
            studies.append({
                "model":model,"original_campaign_sha256":group["campaign_sha256"],
                "selected_cases":records,
            })
        provenance={
            "schema":SCHEMA,
            "source_suite_sha256":plan["source_suite_sha256"],
            "selected_case_count":sum(len(x["selected_cases"]) for x in studies),
            "studies":studies,
            "tools_executed":False,
            "inherited_native_receipts":False,
            "historic_suite_untouched":True,
            "private_inputs":True,
            "limitations":(
                "Private replay contains trained weights and reference examples. "
                "Only E1 files are copied; historical native Vitis/Vivado output "
                "and receipts are NOT migrated. Tool completion does not prove "
                "fixed-point numerical acceptance, E3, achieved per-loop II or E4."
            ),
        }
        _emit_json(stage/"replay_provenance.json",provenance)
        # Recheck each staged SHA after copying before atomically publishing.
        for study in studies:
            for item in study["selected_cases"]:
                for name,digest in item["copied_file_sha256"].items():
                    checked=_regular_bytes(stage/study["model"]/item["case"]/name,
                                           _COPY_FILES[name])
                    if _digest(checked)!=digest:
                        raise ValueError("Replay input modified during staging")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Replay workspace destination appeared")
        stage.rename(target)
    return provenance


def audit_isolated_hls_replay(directory) -> dict:
    """Read-only source-hash and real vendor receipt audit; no native launch."""
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Replay workspace missing or linked")
    receipt=_read_json(root/"replay_provenance.json")
    if receipt.get("schema")!=SCHEMA or receipt.get("private_inputs") is not True:
        raise ValueError("Unrecognized isolated replay workspace")
    studies=receipt.get("studies")
    if not isinstance(studies,list) or not studies:
        raise ValueError("No staged studies")
    checked=[]
    seen=set()
    for study in studies:
        model=study.get("model")
        if not isinstance(model,str) or model in seen or model not in ("mnist_mlp","mnist_cnn"):
            raise ValueError("Invalid or duplicated replay study")
        seen.add(model)
        model_dir=root/model
        if model_dir.is_symlink() or not model_dir.is_dir():
            raise ValueError("Linked or missing replay model directory")
        campaign=_read_json(model_dir/"campaign.json")
        declared=campaign.get("cases")
        cases=study.get("selected_cases")
        if (campaign.get("schema")!="fpgai-research/paper-campaign/v1"
                or not isinstance(declared,list) or not isinstance(cases,list)
                or len(declared)!=len(cases)):
            raise ValueError("Campaign and source snapshot mismatch")
        for entry,record in zip(declared,cases):
            if not isinstance(entry,dict) or not isinstance(record,dict):
                raise ValueError("Invalid replay case entry")
            case=record.get("case")
            if not isinstance(case,str) or not _CASE.fullmatch(case):
                raise ValueError("Unsafe replay case identifier")
            if (entry.get("case")!=case or entry.get("dimension")!=record.get("axis")
                    or entry.get("status")!="READY" or
                    entry.get("cpp_sha256")!=record.get("source_sha256")):
                raise ValueError("Replay campaign no longer matches its source snapshot")
            source=root/model/case
            if source.is_symlink() or not source.is_dir():
                raise ValueError("Linked or missing replay case")
            digests=record.get("copied_file_sha256")
            if (not isinstance(digests,dict) or
                    not {"model.cpp","tb.cpp","run_hls.tcl","manifest.json"}.issubset(digests)):
                raise ValueError("Incomplete pinned replay sources")
            for name,digest in digests.items():
                if name not in _COPY_FILES or _digest(_regular_bytes(
                        source/name,_COPY_FILES[name]))!=digest:
                    raise ValueError("Source input modified after isolated preparation")
            summary=summarize_case(str(source))
            if summary["source_sha256"]!=record["source_sha256"]:
                raise ValueError("Native source is not the frozen replay source")
            probe=inspect_native_schedule_case(
                source,record["source_sha256"],summary["evidence"])
            # Independently verify exact text report freshness before
            # inspecting anonymous loop-row grammar. Old reports, incomplete
            # receipts and even a source-matched E1 are intentionally excluded.
            shape_status="NOT_SCANNED_UNATTESTED_OR_MISSING_REPORT"
            paired_tables=None
            candidate_rows=None
            field_shape_counts=None
            alternate_table_candidates=None
            alternate_table_count=None
            if probe["status"]=="ATTESTED_RAW_REPORT_AVAILABLE":
                raw=_regular_bytes(source/REPORT_REL,REPORT_LIMIT)
                if _digest(raw)!=probe["report_sha256"]:
                    raise ValueError("Attested report bytes changed during inspection")
                # Separately inspect *candidate* alternative HLS table grammar.
                # This reuses the privacy-redacted, bounded exploratory probe.
                # Header roles alone never bind a native loop to an IR operator.
                alternative=_examine(raw)
                alternate_table_candidates=[{
                    "header_roles":site["sanitized_header_roles"],
                    "column_count":site["column_count"],
                    "sample_row_types":[sample["cell_types"]
                                        for sample in site["sample_row_grammars"]],
                } for site in alternative["table_grammars"]]
                alternate_table_count=alternative["table_probe_count"]
                try:
                    shape=inspect_vitis_2023_loop_tables(raw)
                except ValueError:
                    shape_status="FRESH_REPORT_UNSUPPORTED_GRAMMAR"
                else:
                    if shape["table_limit_reached"] or shape["row_limit_reached"]:
                        shape_status="FRESH_REPORT_INCOMPLETE_TABLE_SCAN"
                    elif shape["paired_loop_table_count"]==0:
                        shape_status="FRESH_REPORT_NO_PAIRED_LOOP_TABLE"
                    elif shape["candidate_rows_scanned"]==0:
                        shape_status="FRESH_REPORT_NO_RECOGNIZED_ROWS"
                    elif shape["row_status_counts"].get(
                            "NUMERIC_ACHIEVED_FIELD_UNATTRIBUTED",0):
                        shape_status="FRESH_REPORT_UNATTRIBUTED_NUMERIC_CANDIDATE"
                    elif shape["row_status_counts"].get("MALFORMED_OR_AMBIGUOUS_ROW",0):
                        shape_status="FRESH_REPORT_MALFORMED_OR_AMBIGUOUS_ROW"
                    elif shape["row_status_counts"].get("ACHIEVED_UNAVAILABLE",0):
                        shape_status="FRESH_REPORT_ACHIEVED_UNAVAILABLE"
                    elif shape["row_status_counts"].get("ACHIEVED_UNKNOWN",0):
                        shape_status="FRESH_REPORT_ACHIEVED_UNKNOWN"
                    else:
                        shape_status="FRESH_REPORT_UNRESOLVED_LOOP_GRAMMAR"
                    paired_tables=shape["paired_loop_table_count"]
                    candidate_rows=shape["candidate_rows_scanned"]
                    field_shape_counts=shape["row_status_counts"]
            checked.append({
                "model":model,"case":case,"axis":record["axis"],
                "source_sha256":record["source_sha256"],
                "evidence":summary["evidence"],
                "fixed_point_csim_passed":summary.get("csim_passed"),
                "human_readable_hls_report_status":probe["status"],
                "fresh_hls_report_attested":probe["status"]=="ATTESTED_RAW_REPORT_AVAILABLE",
                "loop_table_diagnostic_status":shape_status,
                "paired_loop_table_count":paired_tables,
                "candidate_loop_rows_scanned":candidate_rows,
                "anonymous_loop_field_shape_counts":field_shape_counts,
                "alternative_header_candidate_count":alternate_table_count,
                "alternative_redacted_table_grammars":alternate_table_candidates,
                "operator_loop_ii_verified":False,
            })
    if len(checked)!=receipt.get("selected_case_count"):
        raise ValueError("Replay source inventory count changed")
    return {
        "schema":AUDIT_SCHEMA,
        "source_suite_sha256":receipt["source_suite_sha256"],
        "case_count":len(checked),
        "e3_source_attested_case_count":sum(x["evidence"] in (
            "E3-HLS-synthesis-report","E4-Vivado-implementation") for x in checked),
        "csim_passed_case_count":sum(x["fixed_point_csim_passed"] is True for x in checked),
        "fresh_raw_hls_report_count":sum(x["fresh_hls_report_attested"] for x in checked),
        "fresh_loop_table_scanned_case_count":sum(
            x["anonymous_loop_field_shape_counts"] is not None for x in checked),
        "fresh_alternative_grammar_probed_case_count":sum(
            x["alternative_redacted_table_grammars"] is not None for x in checked),
        "fresh_loop_table_diagnostic_status_counts":{
            status:sum(row["loop_table_diagnostic_status"]==status for row in checked)
            for status in sorted({x["loop_table_diagnostic_status"] for x in checked})
        },
        "fully_e3_and_csim_qualified_count":sum(
            x["evidence"] in ("E3-HLS-synthesis-report","E4-Vivado-implementation")
            and x["fixed_point_csim_passed"] is True for x in checked),
        "cases":checked,
        "native_tools_executed":False,
        "per_loop_schedule_evidence_available":False,
        "historic_native_receipts_inherited":False,
        "limitations":(
            "Audited only source-bound, fresh-invocation E3 and passing CSim, "
            "not E4 timing or operator loop scheduling. Fresh loop-table scans "
            "and alternative candidate header roles are anonymous structural "
            "observations only: no operator binding, numeric achieved II or "
            "causal explanation is established. "
            "Native tools were not run by this audit; prior execution may "
            "have produced artifacts."
        ),
    }


def export_isolated_hls_replay_audit(directory, output) -> dict:
    result=audit_isolated_hls_replay(directory)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Never replace a saved replay audit")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-replay-audit-",dir=target.parent) as tmp:
        stage=Path(tmp)
        _emit_json(stage/"native_hls_replay_audit.json",result)
        lines=["# Isolated Vitis replay: read-only native attestation audit","",
               f"Cases: {result['case_count']}; source-attested E3: "
               f"{result['e3_source_attested_case_count']}; "
               f"fixed-point CSim passed: {result['csim_passed_case_count']}; "
               f"fresh raw HLS text: {result['fresh_raw_hls_report_count']}; "
               f"redacted fresh table scans: "
               f"{result['fresh_loop_table_scanned_case_count']}.","",
               "| Model | Case | Evidence | Fixed-point CSim | Raw HLS report | Redacted loop-table diagnostic |",
               "| --- | --- | --- | --- | --- | --- |"]
        for case in result["cases"]:
            lines.append("| "+" | ".join(str(case[field]) for field in (
                "model","case","evidence","fixed_point_csim_passed",
                "human_readable_hls_report_status",
                "loop_table_diagnostic_status"))+" |")
        lines.extend(["", "## Alternative pipe-table header grammar (exploratory only)", "",
                      "Only whitelisted header roles and generic row cell types "
                      "from freshly attested report bytes appear here. These "
                      "may be summary tables and are not verified loop schedules.", ""])
        for case in result["cases"]:
            if case["alternative_redacted_table_grammars"] is None:
                continue
            lines.append("### "+case["model"]+" / "+case["case"])
            lines.append("Candidate headers: "+str(case["alternative_header_candidate_count"]))
            for site in case["alternative_redacted_table_grammars"]:
                lines.append("Header roles: "+", ".join(site["header_roles"]))
                for row_types in site["sample_row_types"]:
                    lines.append("Anonymous row types: "+", ".join(row_types))
            lines.append("")
        lines.extend(["","**Operator-local achieved II remains unresolved.**","",
                      result["limitations"],""])
        (stage/"native_hls_replay_audit.md").write_text("\n".join(lines),
                                                          encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Replay audit destination appeared")
        stage.rename(target)
    return result
