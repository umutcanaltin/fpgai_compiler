"""Bounded, read-only E4 qualification of *selected* source-attested suite cases.

Unlike audit-suite --required-evidence E4, this gate does not demand that
all other architecture cases in a progressive paper study have been run.
It never invokes Vitis/Vivado, changes evidence, or infers E4 from exit 0.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.results import summarize_case

SCHEMA = "fpgai-research/bounded-suite-native-audit/v1"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _positive_wns(metrics: dict) -> bool:
    timing = metrics.get("timing") or {}
    wns = timing.get("wns_ns")
    return (type(wns) in (int, float) and isfinite(wns)
            and wns >= 0 and timing.get("timing_met") is True)


def _qualify(summary: dict) -> list[str]:
    blockers = []
    physical = summary.get("vivado_metrics") or {}
    drc = physical.get("drc") or {}
    if summary.get("evidence") != "E4-Vivado-implementation":
        blockers.append("Missing source/RTL-attested E4 Vivado implementation")
    if summary.get("csim_passed") is not True:
        blockers.append("Fixed-point CSim not numerically accepted")
    if summary.get("vivado_implementation_mode") != "out_of_context":
        blockers.append("Missing matched out-of-context implementation")
    if physical.get("routed_nets_complete") is not True:
        blockers.append("Fully routed net count not verified")
    if not _positive_wns(physical):
        blockers.append("Post-route timing closure not verified")
    if not (drc.get("summary_verified") is True
            and drc.get("violations") == 0
            and drc.get("clean") is True
            and drc.get("critical_warnings") == 0
            and drc.get("unconstrained_io") is False):
        blockers.append("Clean DRC gate not met: zero violations, verified summary, no critical/error or unconstrained I/O")
    utilization = physical.get("utilization") or {}
    for resource in ("lut", "ff", "dsp", "bram", "uram"):
        value = utilization.get(resource)
        if not (type(value) in (int, float) and isfinite(value) and value >= 0):
            blockers.append(f"Missing/invalid post-route {resource}")
    return blockers


def _qualify_hls(summary: dict) -> list[str]:
    """E3 synthesis plus passing fixed-point CSim, independent of E4 timing.

    The attestation is established by summarize_case: it cross-checks
    Vitis vendor version, invocation, fresh XML and exact C++ source.
    """
    blockers = []
    if summary.get("evidence") not in ("E3-HLS-synthesis-report",
                                       "E4-Vivado-implementation"):
        blockers.append("Missing source-attested E3 HLS synthesis report")
    if summary.get("csim_passed") is not True:
        blockers.append("Fixed-point CSim not numerically accepted")
    receipt_hash=summary.get("hls_report_sha256")
    if (not isinstance(receipt_hash,str) or
        re.fullmatch(r"[a-f0-9]{64}",receipt_hash) is None):
        blockers.append("Missing source-attested HLS report SHA-256")
    metrics=summary.get("metrics") or {}
    for name in ("latency_best","ii_min"):
        value=metrics.get(name)
        if (type(value) is not str or not value.isdigit() or int(value)<0):
            blockers.append("Missing/invalid HLS estimate: "+name)
    return blockers


def audit_suite_batch(directory, *, first=0, limit=2,
                      required_evidence="E4") -> dict:
    """Reaudit scientific IR before qualifying the requested ready-case slice."""
    if required_evidence not in ("E3", "E4"):
        raise ValueError("Required native evidence must be E3 or E4")
    if type(first) is not int or first < 0:
        raise ValueError("first must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("limit must be 1..256")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("A real prepared suite directory is required")
    source_audit = audit_suite(root, require_host=True, require_all_axes=False)
    if (not source_audit.get("all_prepared_models_have_legal_executable_subsets",
                             source_audit["all_models_source_ready"])
            or source_audit.get("blocked_models",0)):
        raise ValueError("Every model needs an independently audited legal-case subset")
    suite_file = root / "suite_results.json"
    manifest = json.loads(suite_file.read_text())
    studies = manifest["studies"]
    results = []
    for study in studies:
        identity = study["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", identity):
            raise ValueError("Invalid prepared study ID")
        folder = root / identity
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Study directory must be real")
        campaign_file = folder / "campaign.json"
        if campaign_file.is_symlink() or not campaign_file.is_file():
            raise ValueError("Missing or linked campaign")
        campaign = json.loads(campaign_file.read_text())
        if campaign.get("schema") != "fpgai-research/paper-campaign/v1":
            raise ValueError("Unexpected campaign schema")
        ready = [entry for entry in campaign.get("cases", [])
                 if entry.get("status") == "READY"]
        chosen = ready[first:first + limit]
        if not chosen:
            raise ValueError(f"No READY cases selected for {identity}")
        selected = []
        for entry in chosen:
            case_name = entry.get("case")
            if not isinstance(case_name, str) or not re.fullmatch(r"case_[0-9]{4,8}", case_name):
                raise ValueError("Unsafe case ID")
            case = folder / case_name
            if case.is_symlink() or not case.is_dir():
                raise ValueError("Prepared case must be a real directory")
            summary = summarize_case(str(case))
            if summary["source_sha256"] != entry.get("cpp_sha256"):
                raise ValueError("Case source does not match frozen campaign")
            hls_blockers = _qualify_hls(summary)
            e4_blockers = _qualify(summary)
            blockers = hls_blockers if required_evidence=="E3" else e4_blockers
            physical = summary.get("vivado_metrics") or {}
            measured = summary.get("evidence") == "E4-Vivado-implementation"
            csim = summary.get("csim_receipt") or {}
            selected.append({
                "case": case_name,
                "axis": entry.get("dimension"),
                "requested": entry.get("requested"),
                "evidence": summary["evidence"],
                "source_sha256": summary["source_sha256"],
                "numerical_pass": summary.get("csim_passed"),
                "csim_failed_outputs": csim.get("failures"),
                "csim_max_absolute_error": csim.get("max_absolute_error"),
                "hls_metrics": summary.get("metrics"),
                "implementation_mode": summary.get("vivado_implementation_mode"),
                "postroute_timing": physical.get("timing") if measured else None,
                "postroute_utilization": physical.get("utilization") if measured else None,
                "postroute_drc": physical.get("drc") if measured else None,
                "report_sha256": physical.get("raw_report_sha256") if measured else None,
                "attested_e3": summary.get("evidence") in
                    ("E3-HLS-synthesis-report","E4-Vivado-implementation"),
                "attested_e4": measured,
                "hls_report_sha256": summary.get("hls_report_sha256"),
                "hls_qualified": not hls_blockers,
                "e4_qualified": not e4_blockers,
                "qualified": not blockers,
                "blockers": blockers,
                "warnings": summary.get("warnings", []),
            })
        results.append({
            "id": identity,
            "campaign_sha256": _sha(campaign_file),
            "requested_cases": len(selected),
            "qualified_cases": sum(case["qualified"] for case in selected),
            "cases": selected,
        })
    all_cases = [case for study in results for case in study["cases"]]
    return {
        "schema": SCHEMA,
        "source_suite_sha256": _sha(suite_file),
        "source_audit_schema": source_audit["schema"],
        "first_ready_index": first,
        "limit_per_model": limit,
        "selected_case_count": len(all_cases),
        "required_evidence": required_evidence,
        "attested_e3_count": sum(case["attested_e3"] for case in all_cases),
        "attested_e4_count": sum(case["attested_e4"] for case in all_cases),
        "hls_qualified_count": sum(case["hls_qualified"] for case in all_cases),
        "e4_qualified_count": sum(case["e4_qualified"] for case in all_cases),
        "qualified_case_count": sum(case["qualified"] for case in all_cases),
        "all_selected_qualified": all(case["qualified"] for case in all_cases),
        "studies": results,
        "vendor_tools_executed": False,
        "limitations": (
            "A bounded native-E3 gate requires source-attributed Vitis HLS "
            "synthesis and passing fixed-point CSim; it does not require "
            "Vivado timing closure. The E4 gate separately demands routed "
            "OOC timing and verified DRC. This is not full campaign qualification, "
            "board deployment, a bitstream, or full-dataset accuracy. "
            "HLS cycles are synthesis estimates, not end-to-end runtime."
        ),
    }
