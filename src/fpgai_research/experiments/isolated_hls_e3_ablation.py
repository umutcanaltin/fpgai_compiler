"""Within-model E3-only comparisons from a source-attested isolated HLS replay.

This is a separate scientific output from E4 routed/qualified comparisons:
Vitis XML HLS estimated cycles and resources are not FPGA runtime or routed
utilization. The csynth summary top-level interval is not operator loop II.
The private prepared input data are never included in the export.
"""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.isolated_hls_replay import (
    AUDIT_SCHEMA, audit_isolated_hls_replay,
)
from fpgai_research.experiments.native_hls_rerun_plan import _read_json
from fpgai_research.experiments.results import summarize_case

SCHEMA = "fpgai-research/isolated-native-hls-e3-ablation/v1"
E3 = "E3-HLS-synthesis-report"
_METRICS = ("latency_best", "latency_worst", "ii_min", "ii_max",
            "lut", "ff", "dsp", "bram_18k", "uram")
_RESOURCES = ("lut", "ff", "dsp", "bram_18k", "uram")
_COLS = ("model", "case", "axis", "comparison_status", "blocker",
         "evidence", "numerical_csim_passed", "fresh_report_attested",
         "baseline_latency_cycles", "estimated_latency_cycles",
         "estimated_latency_delta_cycles", "estimated_latency_reduction_pct",
         "estimated_top_call_interval_min", "operator_loop_ii_verified",
         "measured_postroute_qor",
         *(f"{key}_{name}" for name in _RESOURCES
           for key in ("baseline_estimated","case_estimated","delta_estimated")))
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z][a-z0-9_]{0,40}\Z")
_CASE = re.compile(r"case_[0-9]{4,8}\Z")


def _metric(value, *, positive=False):
    """Only nonnegative finite integral XML measurements may be compared."""
    if type(value) is int:
        n = value
    elif isinstance(value,str) and value.isdecimal():
        n = int(value)
    else:
        return None
    return n if n >= (1 if positive else 0) else None


def _group(records):
    grouped = {}
    keys = set()
    for row in records:
        if not isinstance(row,dict):
            raise ValueError("Invalid E3 case record")
        model,case=row.get("model"),row.get("case")
        if (not isinstance(model,str) or not _ID.fullmatch(model)
                or not isinstance(case,str) or not _CASE.fullmatch(case)
                or (model,case) in keys):
            raise ValueError("Missing, unsafe or duplicate case identity")
        keys.add((model,case))
        grouped.setdefault(model,[]).append(row)
    if not grouped:
        raise ValueError("Empty E3 comparison inventory")
    return grouped


def _quality(row):
    if row.get("evidence")!=E3:
        return "NOT_SOURCE_ATTESTED_E3"
    if row.get("numerical_csim_passed") is not True:
        return "FIXED_POINT_CSIM_NOT_ACCEPTED"
    if row.get("fresh_report_attested") is not True:
        return "FRESH_HLS_TEXT_REPORT_NOT_ATTESTED"
    if _metric((row.get("hls_estimates") or {}).get("latency_best"),positive=True) is None:
        return "HLS_LATENCY_ESTIMATE_NOT_AVAILABLE"
    if not isinstance(row.get("functional_sha256"),str) or not _DIGEST.fullmatch(row["functional_sha256"]):
        return "FUNCTIONAL_MODEL_FINGERPRINT_NOT_AVAILABLE"
    if (not isinstance(row.get("target_part"),str) or not row["target_part"]
            or type(row.get("clock_ns")) not in (int,float)
            or not isfinite(row["clock_ns"]) or row["clock_ns"]<=0
            or type(row.get("precision_bits")) is not int
            or type(row.get("integer_bits")) is not int
            or row["precision_bits"]<=0
            or not 0<row["integer_bits"]<=row["precision_bits"]
            or type(row.get("numerical_tolerance")) not in (float,int)
            or not isfinite(row["numerical_tolerance"])
            or row["numerical_tolerance"]<=0
            or type(row.get("csim_sample_count")) is not int
            or row["csim_sample_count"]<1
            or type(row.get("csim_outputs_per_sample")) is not int
            or row["csim_outputs_per_sample"]<1
            or not isinstance(row.get("tool_version_sha256"),str)
            or not _DIGEST.fullmatch(row["tool_version_sha256"])):
        return "NATIVE_CONFIGURATION_OR_VERSION_INCOMPLETE"
    return None


def analyze_isolated_hls_e3_ablation(audit:dict, records:list[dict]) -> dict:
    """Compute accepted per-model HLS-estimate contrasts, never causal II."""
    if (audit.get("schema")!=AUDIT_SCHEMA
            or audit.get("operator_schedule_attribution_verified") is True
            or audit.get("per_loop_schedule_evidence_available") is not False):
        raise ValueError("Expected a non-attributing isolated replay audit")
    audited=audit.get("cases")
    if (not isinstance(audited,list) or not isinstance(records,list)
            or len(audited)!=audit.get("case_count") or len(records)!=len(audited)):
        raise ValueError("Case inventory differs from trusted isolated audit")
    trusted = {(r["model"],r["case"]):r for r in audited
               if isinstance(r,dict) and isinstance(r.get("model"),str)
               and isinstance(r.get("case"),str)}
    if len(trusted)!=len(audited):
        raise ValueError("Ambiguous audit case identities")
    models=_group(records)
    if set(trusted)!=set((r["model"],r["case"]) for cases in models.values() for r in cases):
        raise ValueError("Evidence summary does not cover the audited cases")
    rows=[]
    for model,cases in models.items():
        baseline=[r for r in cases if r["case"]=="case_0000" and r.get("axis")=="baseline"]
        if len(baseline)!=1 or sum(r.get("axis")=="baseline" for r in cases)!=1:
            raise ValueError("Missing or ambiguous original model baseline")
        base=baseline[0]
        base_block=_quality(base)
        for case in cases:
            audit_row=trusted[(model,case["case"])]
            if (audit_row.get("axis")!=case.get("axis")
                    or audit_row.get("source_sha256")!=case.get("source_sha256")
                    or audit_row.get("evidence")!=case.get("evidence")
                    or audit_row.get("fixed_point_csim_passed")!=case.get("numerical_csim_passed")
                    or audit_row.get("fresh_hls_report_attested")!=case.get("fresh_report_attested")
                    or audit_row.get("operator_loop_ii_verified") is not False):
                raise ValueError("Audited report and source-verified HLS metrics disagree")
            blocker=_quality(case)
            if base_block is not None:
                blocker="BASELINE_"+base_block
            elif blocker is None and any(case.get(field)!=base.get(field) for field in (
                    "functional_sha256","target_part","clock_ns",
                    "precision_bits","integer_bits","tool_version_sha256",
                    "numerical_tolerance","csim_sample_count",
                    "csim_outputs_per_sample")):
                blocker="MODEL_NUMERICAL_TARGET_CLOCK_OR_TOOL_MISMATCH"
            eligible=blocker is None
            base_metrics=base.get("hls_estimates") or {}
            case_metrics=case.get("hls_estimates") or {}
            base_latency=_metric(base_metrics.get("latency_best"),positive=True) if eligible else None
            current_latency=_metric(case_metrics.get("latency_best"),positive=True) if eligible else None
            delta=current_latency-base_latency if eligible else None
            pct=round(100*(base_latency-current_latency)/base_latency,6) if eligible else None
            resource={}
            for name in _RESOURCES:
                b=_metric(base_metrics.get(name)) if eligible else None
                c=_metric(case_metrics.get(name)) if eligible else None
                resource[name]={
                    "baseline_hls_estimated":b,
                    "case_hls_estimated":c if eligible else None,
                    "estimated_delta":c-b if b is not None and c is not None else None,
                }
            rows.append({
                "model":model,"case":case["case"],"axis":case["axis"],
                "evidence":case["evidence"],
                "comparison_status":"QUALIFIED_E3_HLS_ESTIMATE" if eligible else "BLOCKED",
                "blocker":blocker,
                "numerical_csim_passed":case["numerical_csim_passed"],
                "fresh_report_attested":case["fresh_report_attested"],
                "source_sha256":case["source_sha256"],
                "functional_sha256":case.get("functional_sha256"),
                "baseline_latency_cycles":base_latency,
                "estimated_latency_cycles":current_latency,
                "estimated_latency_delta_cycles":delta,
                "estimated_latency_reduction_pct":pct,
                "estimated_top_call_interval_min":_metric(
                    case_metrics.get("ii_min"),positive=True) if eligible else None,
                "estimated_hls_resource_counts":resource,
                "operator_loop_ii_verified":False,
                "measured_postroute_qor":False,
            })
    counts=Counter(r["comparison_status"] for r in rows)
    return {
        "schema":SCHEMA,
        "source_suite_sha256":audit.get("source_suite_sha256"),
        "case_count":len(rows),
        "variant_count":sum(r["axis"]!="baseline" for r in rows),
        "qualified_e3_hls_estimate_case_count":counts["QUALIFIED_E3_HLS_ESTIMATE"],
        "blocked_case_count":counts["BLOCKED"],
        "comparison_rows":rows,
        "native_tools_executed":False,"numerical_loop_ii_exported":False,
        "operator_schedule_attribution_verified":False,
        "e4_physical_comparison_performed":False,"private_tensor_values_exported":False,
        "limitations":(
            "Only matched, within-model, source-attested E3 HLS SYNTHESIS "
            "ESTIMATES are compared. No E4 routed LUT/BRAM, WNS, energy, "
            "FPGA throughput or full MNIST test-set accuracy is measured. "
            "The HLS top-level call interval is not operator achieved loop II; "
            "anonymous loop-table grammar does not explain changes in cycles. "
            "Numerical acceptance is bounded by the prepared CSim samples."
        )
    }


def collect_isolated_hls_e3_ablation(directory) -> dict:
    """Revalidate original replay provenance and build a weight-free summary."""
    root=Path(directory)
    audit=audit_isolated_hls_replay(root)
    provenance=_read_json(root/"replay_provenance.json")
    records=[]
    for study in provenance["studies"]:
        model=study["model"]
        for item in study["selected_cases"]:
            case=item["case"]
            folder=root/model/case
            meta=_read_json(folder/"manifest.json")
            summary=summarize_case(str(folder))
            metrics=summary.get("metrics") or {}
            receipt_path=folder/"vitis_invocation.json"
            receipt=_read_json(receipt_path) if receipt_path.is_file() and not receipt_path.is_symlink() else {}
            version=receipt.get("tool_version_output")
            records.append({
                "model":model,"case":case,"axis":item["axis"],
                "source_sha256":summary["source_sha256"],
                "evidence":summary["evidence"],
                "numerical_csim_passed":summary.get("csim_passed"),
                "fresh_report_attested":next(
                    r["fresh_hls_report_attested"] for r in audit["cases"]
                    if r["model"]==model and r["case"]==case),
                "functional_sha256":meta.get("functional_sha256"),
                "target_part":summary.get("target_part"),
                "clock_ns":summary.get("clock_ns"),
                "precision_bits":summary.get("precision_bits"),
                "integer_bits":summary.get("integer_bits"),
                "numerical_tolerance":summary.get("numerical_tolerance"),
                "csim_sample_count":meta.get("csim_sample_count"),
                "csim_outputs_per_sample":meta.get("csim_outputs_per_sample"),
                "tool_version_sha256":sha256(version.encode("utf-8")).hexdigest()
                    if isinstance(version,str) and version else None,
                "hls_estimates":{k:metrics.get(k) for k in _METRICS},
            })
    return analyze_isolated_hls_e3_ablation(audit,records)


def export_isolated_hls_e3_ablation(directory,output) -> dict:
    result=collect_isolated_hls_e3_ablation(directory)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable E3 comparison destination already exists")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-isolated-e3-",dir=target.parent) as tmp:
        stage=Path(tmp)
        (stage/"isolated_hls_e3_ablation.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        with (stage/"isolated_hls_e3_ablation.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=_COLS)
            writer.writeheader()
            for row in result["comparison_rows"]:
                flat={key:row[key] for key in _COLS if key in row}
                for name in _RESOURCES:
                    measure=row["estimated_hls_resource_counts"][name]
                    flat[f"baseline_estimated_{name}"]=measure["baseline_hls_estimated"]
                    flat[f"case_estimated_{name}"]=measure["case_hls_estimated"]
                    flat[f"delta_estimated_{name}"]=measure["estimated_delta"]
                writer.writerow(flat)
        lines=[
            "# Source-attested isolated MLP/CNN HLS E3 architecture ablations","",
            "**HLS estimates only. No E4 physical QoR or achieved operator loop II.**","",
            f"Cases: {result['case_count']}; eligible: "
            f"{result['qualified_e3_hls_estimate_case_count']}; "
            f"blocked: {result['blocked_case_count']}.","",
            "| Model | Case | Axis | E3 eligibility | HLS latency (cycles) | "
            "Delta vs same-model baseline (cycles) | Est. reduction | HLS top-call min interval |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in result["comparison_rows"]:
            values=(row["model"],row["case"],row["axis"],row["comparison_status"],
                    row["estimated_latency_cycles"],
                    row["estimated_latency_delta_cycles"],
                    row["estimated_latency_reduction_pct"],
                    row["estimated_top_call_interval_min"])
            lines.append("| "+" | ".join("TBD" if x is None else str(x) for x in values)+" |")
            if row["blocker"] is not None:
                lines.append("Blocked: "+row["blocker"])
        lines+=["","## Estimated HLS resource changes (NOT post-route utilization)","",
                "| Model | Case | Axis | LUT Δ | FF Δ | DSP Δ | BRAM_18K Δ | URAM Δ |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in result["comparison_rows"]:
            measured=row["estimated_hls_resource_counts"]
            values=(row["model"],row["case"],row["axis"],
                    *(measured[k]["estimated_delta"] for k in _RESOURCES))
            lines.append("| "+" | ".join("TBD" if x is None else str(x) for x in values)+" |")
        lines+=["","## Evidence boundary","",result["limitations"],"",
                "The JSON and CSV include separate estimated HLS "
                "LUT/FF/DSP/BRAM_18K/URAM counts and baseline deltas, "
                "without native loop names, training weights, or post-route claims.",""]
        (stage/"isolated_hls_e3_ablation.md").write_text("\n".join(lines),encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("E3 comparison destination appeared")
        stage.rename(target)
    return result
