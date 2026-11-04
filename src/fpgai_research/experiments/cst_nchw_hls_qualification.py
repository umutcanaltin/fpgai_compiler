"""Strict source-attested native Vitis CSim/CSynth qualification for CST NCHW.

Evidence is NOT created here. Only read original HLS results already
attributed by the canonical vendor runner and source report auditor.
Reject contradictory clock/uncertainty, missing estimated resources,
infeasible HLS schedule and incomplete numerical tests. All no-evidence
cases remain explicit UNKNOWN; E3 estimates are never physical QoR.
"""
from __future__ import annotations

from hashlib import sha256
from math import isfinite
from pathlib import Path
import json

from fpgai_research.experiments.results import summarize_case
from rl_environment.timing import audited_hls_timing

SCHEMA="fpgai-research/cst-NCHW-qualified-native-HLS-comparison/v1"
_REQUIRED_RESOURCES=("bram_18k","dsp","lut","ff")


def _metric(value,*,positive:bool=False):
    """Accept source-parser decimal strings; never infer missing as zero."""
    if type(value) is int and value>=0:
        parsed=value
    elif isinstance(value,str) and value.isdecimal():
        parsed=int(value)
    else:
        return None
    return parsed if not positive or parsed>=1 else None


def qualify_case(case_path: str|Path, expected_case:dict, *,
                 summary:dict|None=None):
    """Read-only actual source/timing/precision qualified HLS receipt.

    A supplied summary is for deterministic CPU unit tests only; no
    fabricated summary may be written to a study's evidence directory.
    """
    case=Path(case_path)
    if case.is_symlink() or not case.is_dir():
        raise ValueError("Original vendor case directory absent or symlinked")
    for filename in ("model.cpp","tb.cpp","run_hls.tcl",
                     "manifest.json"):
        path=case/filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("Original native input missing or symlinked")
    source_sha=sha256((case/"model.cpp").read_bytes()).hexdigest()
    if source_sha!=expected_case["source_cpp_sha256"]:
        raise ValueError("Native source hash differs from prepared source")
    original=summary if summary is not None else summarize_case(str(case))
    if original.get("source_sha256")!=source_sha:
        raise ValueError("Source-attested report summary differs from case")
    prepared=json.loads((case/"manifest.json").read_text())
    clock=prepared.get("clock_ns")
    if type(clock) not in (float,int) or not isfinite(clock) or clock<=0:
        raise ValueError("Original native clock must be finite")
    base={
        "case":expected_case["case"],
        "source_cpp_sha256":source_sha,
        "source_storage":expected_case["source_storage"],
        "graph":expected_case["graph"],
        "mapping":expected_case["mapping"],
        "requested_clock_ns":clock,
        "requested_uncertainty_ns":prepared.get("explicit_clock_uncertainty_ns"),
        "evidence_level":original.get("evidence"),
        "csim_passed":original.get("csim_passed"),
        "status":"UNKNOWN_NO_ORIGINAL_VITIS",
        "native_e2_accepted":False,
        "native_e3_qualified":False,
        "hls_timing_feasible":None,
        "hls_estimated_clock_ns":None,
        "hls_effective_scheduling_budget_ns":None,
        "hls_scheduling_margin_ns":None,
        "hls_worst_cycles":None,
        "estimated_resources":{key:None for key in (*_REQUIRED_RESOURCES,"uram")},
        "hls_report_sha256":None,
        "physical_qualified":False,
        "physical_clock_ns":None,
        "warnings":list(original.get("warnings") or ()),
    }
    if original.get("csim_passed") is False:
        base["status"]="FAILED_ORIGINAL_CSIM"
        return base
    if original.get("csim_passed") is not True:
        return base
    if original.get("evidence") not in (
            "E2-Vitis-C-simulation","E3-HLS-synthesis-report",
            "E4-Vivado-implementation"):
        base["status"]="UNKNOWN_UNATTESTED_CSIM"
        return base
    numerical=original.get("csim_receipt") or {}
    if (numerical.get("failures")!=0 or
            type(numerical.get("samples")) is not int or
            numerical["samples"]!=prepared.get("csim_sample_count") or
            numerical.get("outputs_per_sample")!=prepared.get("csim_outputs_per_sample") or
            type(numerical.get("max_absolute_error")) not in (int,float) or
            not isfinite(numerical["max_absolute_error"]) or
            numerical["max_absolute_error"]>prepared.get("numerical_tolerance",0)):
        base["status"]="REJECTED_NUMERIC_COVERAGE"
        return base
    base["native_e2_accepted"]=True
    base["status"]="QUALIFIED_CSIM_ONLY"
    if original.get("evidence") not in (
            "E3-HLS-synthesis-report","E4-Vivado-implementation"):
        return base
    receipt_file=case/"vitis_invocation.json"
    if receipt_file.is_symlink() or not receipt_file.is_file():
        base["status"]="UNKNOWN_NO_ATTESTED_INVOCATION"
        return base
    receipt=json.loads(receipt_file.read_text())
    # source+testbench+Tcl+vendor identity and report digest were checked
    # already by summarize_case; check exact stable native source here too.
    if (receipt.get("source_sha256")!=source_sha or
            receipt.get("completed_successfully") is not True or
            receipt.get("inputs_stable_during_execution") is not True or
            receipt.get("testbench_sha256")!=sha256((case/"tb.cpp").read_bytes()).hexdigest() or
            receipt.get("script_sha256")!=sha256((case/"run_hls.tcl").read_bytes()).hexdigest()):
        base["status"]="REJECTED_NATIVE_PROVENANCE"
        return base
    try:
        timings=audited_hls_timing(case,receipt,original,clock)
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        base["status"]="REJECTED_ORIGINAL_HLS_TIMING"
        base["warnings"].append(f"Native HLS timing qualification refused: {exc}")
        return base
    base["hls_timing_feasible"]=timings["hls_timing_feasible"]
    base["hls_estimated_clock_ns"]=timings["hls_estimated_clock_ns"]
    base["hls_effective_scheduling_budget_ns"]=timings["hls_effective_scheduling_budget_ns"]
    base["hls_scheduling_margin_ns"]=timings["hls_scheduling_margin_ns"]
    base["hls_report_sha256"]=timings["hls_report_sha256"]
    if not timings["hls_timing_feasible"]:
        base["status"]="FAILED_HLS_EFFECTIVE_CLOCK_BUDGET"
        return base
    metrics=original.get("metrics") or {}
    worst=_metric(metrics.get("latency_worst"),positive=True)
    resources={key:_metric(metrics.get(key)) for key in (*_REQUIRED_RESOURCES,"uram")}
    base["hls_worst_cycles"]=worst
    base["estimated_resources"]=resources
    if worst is None or any(resources[key] is None for key in _REQUIRED_RESOURCES):
        base["status"]="UNKNOWN_INCOMPLETE_HLS_METRICS"
        return base
    base["native_e3_qualified"]=True
    base["status"]="QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL"
    return base


def compare_qualified_pairs(cases: list[dict]):
    """No post-hoc exclusion: return each pair, unqualified stays UNKNOWN."""
    if len(cases)%2:
        raise ValueError("Complete adjacent disjoint/reused pairs required")
    result=[]
    for i in range(0,len(cases),2):
        a,b=cases[i:i+2]
        if ((a["source_storage"],b["source_storage"])!=("real_disjoint","real_shared")
                or a["graph"]!=b["graph"] or a["mapping"]!=b["mapping"]
                or a["source_cpp_sha256"]==b["source_cpp_sha256"]
                or a["requested_clock_ns"]!=b["requested_clock_ns"]
                or a["requested_uncertainty_ns"]!=b["requested_uncertainty_ns"]):
            raise ValueError("Unmatched native source pair or clock constraints")
        accepted=a["native_e3_qualified"] and b["native_e3_qualified"]
        has_resource=accepted and all(
            a["estimated_resources"][k] is not None and
            b["estimated_resources"][k] is not None
            for k in _REQUIRED_RESOURCES)
        before=a["estimated_resources"]["bram_18k"] if has_resource else None
        after=b["estimated_resources"]["bram_18k"] if has_resource else None
        result.append({
            "group":i//2,
            "cases":[a["case"],b["case"]],
            "csim_both_pass":a["native_e2_accepted"] and b["native_e2_accepted"],
            "hls_both_qualified":bool(accepted),
            "estimated_bram_before":before,
            "estimated_bram_after":after,
            "hls_estimated_bram_difference":before-after if has_resource else None,
            "hls_latency_worst_cycles_before":
                a["hls_worst_cycles"] if accepted else None,
            "hls_latency_worst_cycles_after":
                b["hls_worst_cycles"] if accepted else None,
            "difference_is_hls_estimate_only":True,
            "physical_bram_savings":None,
            "achieved_fmax":None,
            "status":"PAIRED_HLS_ESTIMATE_ONLY" if accepted else "UNKNOWN_OR_REJECTED_HLS_PAIR",
        })
    return result
