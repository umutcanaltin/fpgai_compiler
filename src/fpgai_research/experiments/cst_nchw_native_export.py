"""Create-only, zero-tool, denominator-preserving native NCHW HLS paper report.

Reads original source/Vitis receipts through independent strict audits.
Always exports ALL 12 prespecified cases and 6 matched pairs, with
UNKNOWN/FAILED cases retained. HLS estimates are not physical QoR.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.experiments.cst_nchw_native_execute import audit_native
from fpgai_research.experiments.cst_nchw_native_pairs import audit_prepared

SCHEMA="fpgai-research/cst-NCHW-native-paired-paper-report/v1"
CASE_FIELDS=(
    "case","graph","mapping","source_storage","source_cpp_sha256",
    "native_invocation_present","attested_evidence_level",
    "csim_passed","original_csim_qualified","native_qualification_status",
    "original_hls_effective_clock_feasible",
    "original_hls_estimated_clock_ns","original_hls_effective_budget_ns",
    "original_hls_timing_margin_ns","reported_hls_worst_cycles",
    "reported_hls_bram_18k","observed_numerical_and_hls_report",
    "achieved_fpga_clock_ns","qualified_physical_BRAM_savings",
)
PAIR_FIELDS=(
    "group","cases","both_original_CSim_qualified",
    "both_original_CSim_CSynth_attested","original_matched_worst_cycles",
    "original_matched_bram_18k","estimated_HLS_bram_difference",
    "qualified_HLS_estimate_only","qualification_status",
    "physical_QoR_qualified",
)


def export_native_evidence(*,study:str|Path,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original audited native paper export must be create-only")
    prepared=audit_prepared(study)
    received=audit_native(study)
    if (received["cases"]!=12 or received["matched_pairs"]!=6
            or received["source_revision"]!=prepared["source_revision"]
            or len(received["cases_audited"])!=12
            or len(received["pairs_audited"])!=6):
        raise ValueError("Native paper report lacks complete original case denominator")
    source_file=Path(study)/"native_source_manifest.json"
    receipt={
        "schema":SCHEMA,
        "original_source_revision":prepared["source_revision"],
        "original_package_manifest_sha256":sha256(source_file.read_bytes()).hexdigest(),
        "original_cases":12,"original_pairs":6,
        "original_invocations_with_receipts":received["native_invocation_receipts_present"],
        "source_attested_csim_csynth_cases":received["original_CSim_CSynth_attested_cases"],
        "source_attested_csim_csynth_pairs":received["original_matched_report_pairs"],
        "numerical_CSim_qualified_cases":sum(
            x["original_csim_qualified"] for x in received["cases_audited"]),
        "hls_estimate_pair_comparisons":sum(
            x["qualified_HLS_estimate_only"] for x in received["pairs_audited"]),
        "failed_or_unknown_cases":sum(
            not x["observed_numerical_and_hls_report"]
            for x in received["cases_audited"]),
        "achieved_fpga_clock_ns":None,
        "physical_bram_savings":None,
        "physical_QoR_qualified_cases":0,
        "new_vendor_invocations_for_report":0,
        "caveats":[
            "Only qualified source-attributed, CSim-pass, timing-feasible HLS pairs compared",
            "HLS resource numbers are estimates, not board measurements or physical RAM proof",
            "All missing, infeasible, unsupported, failed and unattempted rows retained",
        ],
    }
    root.mkdir(parents=True)
    for filename,fields,rows in (
        ("cases.csv",CASE_FIELDS,received["cases_audited"]),
        ("pairs.csv",PAIR_FIELDS,received["pairs_audited"]),
    ):
        with (root/filename).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                formatted={name:(json.dumps(row.get(name),sort_keys=True)
                                 if isinstance(row.get(name),(list,dict))
                                 else row.get(name))
                           for name in fields}
                writer.writerow(formatted)
    (root/"summary.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    (root/"full_native_audit.json").write_text(
        json.dumps(received,sort_keys=True,indent=2)+"\n")
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(export_native_evidence(study=args.study,output=args.output),
                     indent=2,sort_keys=True))


if __name__=="__main__":
    main()
