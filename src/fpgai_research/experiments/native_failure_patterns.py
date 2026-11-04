"""Diagnose unavailable native evidence without treating blockers as root causes.

Read ONLY a previously source-reaudited seven-axis effects.json export.
No vendor calls, requalification, source mutation or claim of independent
global-precision experiments. All input evidence remains immutable.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.native_seven_axis_effects import (
    SCHEMA as EFFECTS_SCHEMA, AXES,
)

SCHEMA="fpgai-research/native-seven-axis-failure-pattern-triage/v1"
NUMERIC="Fixed-point CSim not numerically accepted"
NO_E4="Missing source/RTL-attested E4 Vivado implementation"
TIMING="Post-route timing closure not verified"
UNDEF_PREFIX="HLS XML measurement rejected"
_ID=re.compile(r"(mlp|cnn)_operator_[0-9]{2}\Z")


def _flags(case):
    blockers=case.get("blockers",[])
    warnings=case.get("source_warnings",[])
    if not isinstance(blockers,list) or not all(isinstance(s,str) for s in blockers):
        raise ValueError("Invalid source-audited blocker inventory")
    if not isinstance(warnings,list) or not all(isinstance(s,str) for s in warnings):
        raise ValueError("Invalid source-audited warning inventory")
    return {
        "numerical_acceptance_unverified":NUMERIC in blockers,
        "invalid_hls_xml":any(UNDEF_PREFIX in s for s in warnings),
        "source_attested_e4_missing":NO_E4 in blockers,
        "postroute_timing_unverified":TIMING in blockers,
    }


def _counts():
    return {"eligible":0,"routed_qualified":0,
            "unqualified":0,"numerical_acceptance_unverified":0,
            "invalid_hls_xml":0,"source_attested_e4_missing":0,
            "postroute_timing_unverified":0}


def _add(counts,flags,qualified):
    counts["eligible"]+=1
    counts["routed_qualified"]+=int(qualified)
    counts["unqualified"]+=int(not qualified)
    for label,present in flags.items():
        counts[label]+=int(present)


def analyze_failure_patterns(effects):
    """Case/axis breakdown, blockers, and duplicate *nominal* global controls."""
    if (not isinstance(effects,dict)
            or effects.get("schema")!=EFFECTS_SCHEMA
            or effects.get("vendor_tools_executed_by_analysis") is not False
            or not isinstance(effects.get("studies"),list)
            or not effects["studies"]):
        raise ValueError("Expected prior source-reaudited native effects export")
    per_study=[]
    total=_counts()
    global_controls={}
    all_case_keys=set()
    for study in effects["studies"]:
        name=study.get("study")
        if not isinstance(name,str) or not _ID.fullmatch(name):
            raise ValueError("Expected operator-local original MLP/CNN study identity")
        if study.get("status")=="BLOCKED_SOURCE_MODEL":
            per_study.append({"study":name,"source_study_blocked":True})
            continue
        cases=study.get("cases")
        if not isinstance(cases,list) or not cases:
            raise ValueError("Study lacks complete case inventory")
        baselines=[c for c in cases if c.get("axis")=="baseline"]
        if len(baselines)!=1 or not isinstance(baselines[0].get("requested"),dict):
            raise ValueError("Exactly one architecture baseline required")
        defaults=baselines[0]["requested"]
        keys=set(defaults)
        if keys!={"precision_bits","integer_bits","parallelism","pes","simd",
                 "serialization","tile","pipeline_ii"}:
            raise ValueError("Baseline architecture lacks complete knob inventory")
        stats=_counts()
        by_axis={}
        by_changed_dimensions={}
        diagnostics=[]
        for case in cases:
            ident=case.get("case")
            if not isinstance(ident,str) or re.fullmatch(r"case_[0-9]{4,8}",ident) is None:
                raise ValueError("Unsafe or missing case identity")
            if (name,ident) in all_case_keys:
                raise ValueError("Duplicate study/case identity")
            all_case_keys.add((name,ident))
            request=case.get("requested")
            if not isinstance(request,dict) or set(request)!=keys:
                raise ValueError("Unverifiable architecture request")
            status=case.get("qualification")
            if status not in ("QUALIFIED_ROUTED","UNQUALIFIED"):
                raise ValueError("Unsupported scientific qualification status")
            qualified=(status=="QUALIFIED_ROUTED")
            if qualified != isinstance(case.get("qualified_metrics"),dict):
                raise ValueError("Inconsistent routed qualification and numerical costs")
            changed=[a for a in AXES if request[a]!=defaults[a]]
            axis=case.get("axis")
            if not isinstance(axis,str) or axis not in (*AXES,"baseline","interaction"):
                raise ValueError("Unrecognized architecture intervention dimension")
            if axis=="baseline" and changed:
                raise ValueError("Baseline is not the declared model default")
            if axis in AXES and changed!=[axis]:
                raise ValueError("Single-axis case is not a single-axis intervention")
            if axis=="interaction" and len(changed)<2:
                raise ValueError("Interaction changed fewer than two axes")
            flags=_flags(case)
            _add(stats,flags,qualified)
            _add(total,flags,qualified)
            _add(by_axis.setdefault(axis,_counts()),flags,qualified)
            signature="+".join(changed) if changed else "baseline"
            _add(by_changed_dimensions.setdefault(signature,_counts()),flags,qualified)
            if not qualified:
                diagnostics.append({
                    "case":ident,"axis":axis,
                    "changed_axes":changed,
                    "requested":request,
                    "native_evidence":case.get("native_evidence"),
                    "flags":flags,
                    "warnings":case.get("source_warnings",[]),
                    "blockers":case.get("blockers",[]),
                })
            if axis in ("baseline","precision_bits"):
                family=name.split("_operator_")[0]
                key=(family,json.dumps(request,sort_keys=True,
                                        separators=(",",":"),allow_nan=False))
                global_controls.setdefault(key,[]).append({
                    "study":name,"case":ident,"axis":axis,
                    "source_sha256":case.get("source_sha256"),
                    "qualification":status,
                    "qualified_metrics":case.get("qualified_metrics"),
                })
        if (stats["eligible"]!=study.get("eligible_case_count")
                or stats["routed_qualified"]!=study.get("routed_qualified_case_count")):
            raise ValueError("Case qualification totals disagree with original export")
        per_study.append({
            "study":name,"source_study_blocked":False,
            "counts":stats,"by_declared_axis":by_axis,
            "by_changed_dimensions":by_changed_dimensions,
            "unqualified_case_diagnostics":diagnostics,
        })
    repeated=[]
    for (family,request),members in sorted(global_controls.items()):
        if len(members)<2:
            continue
        measured=[m["qualified_metrics"] for m in members
                  if m["qualified_metrics"] is not None]
        repeated.append({
            "model_family":family,
            "homogeneous_precision_request":json.loads(request),
            "members":members,
            "matched_source_sha256":len({m["source_sha256"] for m in members})==1,
            "qualified_costs_equal_if_multiple":(
                len({json.dumps(v,sort_keys=True,allow_nan=False)
                     for v in measured})==1 if len(measured)>=2 else None),
            "nominal_control_repeated_across_studies":True,
            "independent_operator_local_precision_replication":False,
        })
    return {
        "schema":SCHEMA,
        "totals":total,
        "studies":per_study,
        "repeated_nominal_model_global_controls":repeated,
        "tool_execution":False,
        "original_vendor_evidence_reaudited_by_triage":False,
        "warning":(
            "Numerical acceptance unverified does not necessarily prove "
            "numerical error; missing/invalid XML is not an architectural "
            "infeasibility proof. Flags overlap. Every reported case remains "
            "source-attested only to the level established in effects.json. "
            "Homogeneous precision conditions appearing in different "
            "operator-named studies are not independent per-operator "
            "experiments, even if source comments change." ),
    }


def export_failure_patterns(effects_json,output):
    source=Path(effects_json)
    dest=Path(output)
    if source.is_symlink() or not source.is_file() or not 0<source.stat().st_size<=32*1024*1024:
        raise ValueError("Expected bounded regular source-audited effects.json")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Native triage destination must be new")
    source_bytes=source.read_bytes()
    result=analyze_failure_patterns(json.loads(source_bytes))
    result["source_effects_sha256"]=sha256(source_bytes).hexdigest()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".native-triage-",dir=dest.parent) as temp:
        stage=Path(temp)/"triage"
        stage.mkdir()
        (stage/"triage.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        with (stage/"failed_cases.csv").open("w",newline="",encoding="utf-8") as f:
            columns=("study","case","axis","changed_axes","precision_bits",
                     "parallelism","pes","simd","serialization","tile",
                     "pipeline_ii","native_evidence",
                     "numerical_acceptance_unverified","invalid_hls_xml",
                     "source_attested_e4_missing","postroute_timing_unverified")
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for study in result["studies"]:
                for c in study.get("unqualified_case_diagnostics",[]):
                    req=c["requested"]
                    writer.writerow({
                        "study":study["study"],"case":c["case"],"axis":c["axis"],
                        "changed_axes":"+".join(c["changed_axes"]),
                        "native_evidence":c["native_evidence"],
                        **{k:json.dumps(req.get(k)) for k in (
                            "precision_bits","parallelism","pes","simd",
                            "serialization","tile","pipeline_ii")},
                        **c["flags"],
                    })
        report=["# Native architecture failure and replication triage","",
                "All categories below may overlap. They are diagnostics, not "
                "proven root causes. No vendor tools executed.","",
                "| Study | Eligible | Qualified | Numerical acceptance unverified | Invalid HLS XML | Missing E4 | Unverified timing |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for study in result["studies"]:
            if study["source_study_blocked"]:
                continue
            c=study["counts"]
            report.append(f"| {study['study']} | {c['eligible']} | "
                          f"{c['routed_qualified']} | "
                          f"{c['numerical_acceptance_unverified']} | "
                          f"{c['invalid_hls_xml']} | "
                          f"{c['source_attested_e4_missing']} | "
                          f"{c['postroute_timing_unverified']} |")
        report.extend(["","## Dimension combinations involved",""])
        for study in result["studies"]:
            if study["source_study_blocked"]:
                continue
            report.append("### "+study["study"])
            for signature,counts in sorted(study["by_changed_dimensions"].items()):
                if not counts["unqualified"]:
                    continue
                report.append(
                    f"- {signature}: {counts['unqualified']}/{counts['eligible']} "
                    f"unqualified; acceptance unverified="
                    f"{counts['numerical_acceptance_unverified']}, "
                    f"invalid HLS XML={counts['invalid_hls_xml']}, "
                    f"timing unverified={counts['postroute_timing_unverified']}")
            report.append("")
        report.extend([
            "## Repeated homogeneous-precision conditions","",
            f"{len(result['repeated_nominal_model_global_controls'])} "
            "candidate repeated source settings across operator-named "
            "studies. These are *not independent operator-local precision "
            "replicates*. See triage.json for their case identities and "
            "whether attested source digests or observed metrics tie.","",
            "Read failed_cases.csv for each failed case's exact requested "
            "knobs and flags. Zero/missing estimates are never imputed.",
        ])
        (stage/"summary.md").write_text("\n".join(report)+"\n")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Native triage destination appeared during export")
        stage.rename(dest)
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--effects",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args(argv)
    result=export_failure_patterns(args.effects,args.output)
    print(json.dumps({
        "schema":result["schema"],"totals":result["totals"],
        "repeated_nominal_model_global_control_count":
            len(result["repeated_nominal_model_global_controls"]),
        "vendor_tools_executed":False,
    },sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
