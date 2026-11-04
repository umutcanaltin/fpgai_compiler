"""Read-only source-audited diagnosis of one unqualified native geometry case.

Inspect the original immutable campaign inventory, generated source,
Vitis CSim/HLS and Vivado implementation evidence. Does NOT rerun or
promote tools, relax scientific gates, or mutate any prepared source.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.suite_native_batch import _qualify, _qualify_hls

SCHEMA="fpgai-research/geometry-native-failure-forensics/v1"
STUDIES={
    "probe_matmul_16","probe_conv_12","probe_matmul_32","probe_conv_16",
}


def inspect_failure(directory,*,study,case):
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected original prepared geometry suite directory")
    if study not in STUDIES:
        raise ValueError("Unsupported registered geometry study")
    if not isinstance(case,str) or not case.startswith("case_") or (
        len(case)!=9 or not case[5:].isdigit() or int(case[5:])>=42):
        raise ValueError("Expected registered case_0000 through case_0041")
    suite=root/"suite_results.json"
    if suite.is_symlink() or not suite.is_file():
        raise ValueError("Missing prepared source manifest")
    source_audit=audit_suite(root,require_host=True,require_all_axes=False)
    if not source_audit.get("all_prepared_models_have_legal_executable_subsets"):
        raise ValueError("Prepared source suite failed independent audit")
    manifest=json.loads(suite.read_text())
    declaration=[x for x in manifest["studies"]
                 if x.get("id")==study and x.get("status")=="E1-PREPARED"]
    if len(declaration)!=1:
        raise ValueError("Missing exactly one prepared registered study")
    campaign_file=root/study/"campaign.json"
    if campaign_file.is_symlink() or not campaign_file.is_file():
        raise ValueError("Missing original architecture campaign")
    campaign=json.loads(campaign_file.read_text())
    matches=[x for x in campaign["cases"] if x.get("case")==case]
    if len(matches)!=1 or matches[0].get("status")!="READY":
        raise ValueError("Requested case not uniquely prepared and READY")
    expected=matches[0]
    result=summarize_case(str(root/study/case))
    if result.get("source_sha256")!=expected.get("cpp_sha256"):
        raise ValueError("Source hash does not match frozen architecture request")
    e3=_qualify_hls(result)
    e4=_qualify(result)
    physical=result.get("vivado_metrics") or {}
    m=physical.get("utilization") or {}
    t=physical.get("timing") or {}
    drc=physical.get("drc") or {}
    report={
        "schema":SCHEMA,
        "study":study,
        "case":case,
        "source_sha256":result["source_sha256"],
        "campaign_sha256":sha256(campaign_file.read_bytes()).hexdigest(),
        "source_suite_sha256":sha256(suite.read_bytes()).hexdigest(),
        "requested_architecture":expected.get("requested"),
        "native_evidence":result.get("evidence"),
        "csim_passed":result.get("csim_passed"),
        "csim_receipt":result.get("csim_receipt"),
        "hls_synthesis_qualified":not e3,
        "e4_routed_qualified":not e4,
        "hls_blockers":e3,
        "physical_blockers":e4,
        "source_warnings":result.get("warnings",[]),
        "hls_metrics":result.get("metrics"),
        "postroute_utilization":m if physical else None,
        "postroute_timing":t if physical else None,
        "postroute_drc":drc if physical else None,
        "routed_nets_complete":physical.get("routed_nets_complete"),
        "implementation_mode":result.get("vivado_implementation_mode"),
        "vendor_tools_executed":False,
        "report_is_original_hardware_evidence":False,
        "scientific_limitations":(
            "Exact case observations are independently audited by summarize_case "
            "and existing E3/E4 gates, but the diagnosis is read-only. "
            "Vitis/Vivado exit code does not imply E4 qualification; "
            "source-linked warnings and native reports must be interpreted "
            "before any new execution. Never turn a failed timing/DRC "
            "observation into a passing circuit by changing acceptance criteria."),
    }
    return report


def export_failure_report(directory,output,*,study,case):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Use a new immutable native diagnosis directory")
    report=inspect_failure(directory,study=study,case=case)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".native-failure-") as tmp:
        stage=Path(tmp)/"result"
        stage.mkdir()
        (stage/"diagnostic.json").write_text(json.dumps(
            report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        lines=[
            "# Source-linked native case diagnosis","",
            f"Study: {study}, case: {case}",
            f"Evidence: {report['native_evidence']}",
            f"CSim numerically passed: {report['csim_passed']}",
            f"E3 qualified: {report['hls_synthesis_qualified']}",
            f"E4 routed-qualified: {report['e4_routed_qualified']}","",
            f"Source SHA-256: {report['source_sha256']}","",
            "## HLS blockers","",
            *("- "+x for x in report["hls_blockers"]),
            *([] if report["hls_blockers"] else ["None"]),
            "","## Physical blockers","",
            *("- "+x for x in report["physical_blockers"]),
            *([] if report["physical_blockers"] else ["None"]),
            "","## Source-linked warnings","",
            *("- "+str(x) for x in report["source_warnings"]),
            *([] if report["source_warnings"] else ["None"]),
            "","## Key native measurements","",
            f"Vitis HLS metrics: {json.dumps(report['hls_metrics'],sort_keys=True)}",
            f"Postroute timing: {json.dumps(report['postroute_timing'],sort_keys=True)}",
            f"Postroute utilization: {json.dumps(report['postroute_utilization'],sort_keys=True)}",
            f"Postroute DRC: {json.dumps(report['postroute_drc'],sort_keys=True)}",
            "","Exit-zero is never proof of routed qualification. "
            "This reporter does not invoke native tools or repair the failed case.",
            ""
        ]
        (stage/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Output appeared during immutable export")
        stage.rename(dest)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True)
    p.add_argument("--study",required=True,choices=sorted(STUDIES))
    p.add_argument("--case",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    r=export_failure_report(args.directory,args.output,
                            study=args.study,case=args.case)
    print(json.dumps({
        "schema":SCHEMA,"study":r["study"],"case":r["case"],
        "native_evidence":r["native_evidence"],
        "e3_qualified":r["hls_synthesis_qualified"],
        "e4_qualified":r["e4_routed_qualified"],
        "physical_blockers":r["physical_blockers"],
        "vendor_tools_executed":False,
    },sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
