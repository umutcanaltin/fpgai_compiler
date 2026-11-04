"""Analyze original source-attested NCHW Vitis HLS pair exports.

Secondary analysis of the create-only full_native_audit.json report;
does NOT revalidate XML or invoke vendor tools. Preserve all six pairs
including failures and unknowns. Pareto comparisons are only within
identically mapped source pairs. HLS estimates are not physical QoR.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

SCHEMA="fpgai-research/cst-NCHW-native-paired-HLS-effects/v1"
FIELDS=("worst_cycles","bram_18k","dsp","lut","ff")


def _values(case):
    resources=case["estimated_hls_resources"]
    values={
        "worst_cycles":case["reported_hls_worst_cycles"],
        "bram_18k":case["reported_hls_bram_18k"],
        "dsp":resources["dsp"],"lut":resources["lut"],"ff":resources["ff"],
    }
    if any(type(v) is not int or v<0 or (k=="worst_cycles" and v==0)
           for k,v in values.items()):
        raise ValueError("Missing/invalid qualified HLS numeric metric")
    if resources["bram_18k"]!=values["bram_18k"]:
        raise ValueError("Inconsistent BRAM resource report")
    return values


def _dominates(a,b,fields):
    return all(a[k]<=b[k] for k in fields) and any(a[k]<b[k] for k in fields)


def _classification(before,after,fields):
    if _dominates(after,before,fields):
        return "SHARED_DOMINATES"
    if _dominates(before,after,fields):
        return "DISJOINT_DOMINATES"
    if all(before[k]==after[k] for k in fields):
        return "TIE"
    return "TRADEOFF"


def analyze(audit):
    if (audit.get("schema")!="fpgai-research/cst-NCHW-source-paired-native-Vitis-audit/v1"
            or audit.get("cases")!=12 or audit.get("matched_pairs")!=6
            or audit.get("physical_qualified_cases")!=0
            or not isinstance(audit.get("source_revision"),str)
            or len(audit["source_revision"])!=40):
        raise ValueError("Require complete source-attributed NCHW native audit")
    cases=audit.get("cases_audited")
    pairs=audit.get("pairs_audited")
    if not isinstance(cases,list) or len(cases)!=12 or not isinstance(pairs,list) or len(pairs)!=6:
        raise ValueError("Incomplete prespecified source pair denominator")
    result=[]
    for i in range(6):
        a,b=cases[2*i:2*i+2]
        p=pairs[i]
        if (a.get("case")!=f"case_{2*i:03d}" or
                b.get("case")!=f"case_{2*i+1:03d}" or
                (a.get("source_storage"),b.get("source_storage"))!=(
                    "real_disjoint","real_shared") or
                a.get("graph")!=b.get("graph") or a.get("mapping")!=b.get("mapping") or
                not isinstance(a.get("source_cpp_sha256"),str) or
                not isinstance(b.get("source_cpp_sha256"),str) or
                a["source_cpp_sha256"]==b["source_cpp_sha256"] or
                p.get("group")!=i or p.get("cases")!=[a["case"],b["case"]]):
            raise ValueError("Unmatched, reordered, or duplicate source variants")
        qualified=all((
            a.get("observed_numerical_and_hls_report") is True,
            b.get("observed_numerical_and_hls_report") is True,
            p.get("both_original_CSim_CSynth_attested") is True,
            p.get("qualified_HLS_estimate_only") is True,
            a.get("original_hls_effective_clock_feasible") is True,
            b.get("original_hls_effective_clock_feasible") is True,
            not a.get("warnings"),not b.get("warnings"),
        ))
        entry={
            "pair":i,"cases":[a["case"],b["case"]],
            "graph":a["graph"],"mapping":a["mapping"],
            "evidence_status":"QUALIFIED_HLS_ESTIMATES_ONLY" if qualified
                              else "UNKNOWN_OR_FAILED_HLS",
            "original_cpp_sha256":[a["source_cpp_sha256"],b["source_cpp_sha256"]],
            "estimated_bram_saved":None,"worst_cycles_saved":None,
            "worst_cycles_pct_improvement":None,
            "pareto_cycles_bram":"UNKNOWN",
            "pareto_cycles_bram_dsp_lut_ff":"UNKNOWN",
            "metrics":None,"physical_bram_saved":None,
            "achieved_fmax":None,
        }
        if qualified:
            budget=a.get("original_hls_effective_budget_ns")
            if (type(budget) not in (int,float) or budget<=0 or
                    budget!=b.get("original_hls_effective_budget_ns")):
                raise ValueError("Mismatched effective clock budgets")
            before,after=_values(a),_values(b)
            difference={k:before[k]-after[k] for k in FIELDS}
            entry.update({
                "estimated_bram_saved":difference["bram_18k"],
                "worst_cycles_saved":difference["worst_cycles"],
                "worst_cycles_pct_improvement":(
                    100*difference["worst_cycles"]/before["worst_cycles"]),
                "pareto_cycles_bram":_classification(
                    before,after,("worst_cycles","bram_18k")),
                "pareto_cycles_bram_dsp_lut_ff":_classification(before,after,FIELDS),
                "metrics":{"disjoint":before,"shared":after,
                           "disjoint_minus_shared":difference},
            })
        result.append(entry)
    qualified_rows=[x for x in result if x["evidence_status"]=="QUALIFIED_HLS_ESTIMATES_ONLY"]
    return {
        "schema":SCHEMA,"source_revision":audit["source_revision"],
        "prespecified_cases":12,"prespecified_pairs":6,
        "qualified_pairs":len(qualified_rows),"unqualified_pairs":6-len(qualified_rows),
        "estimated_bram_decreases":sum(x["estimated_bram_saved"]>0 for x in qualified_rows),
        "estimated_bram_ties":sum(x["estimated_bram_saved"]==0 for x in qualified_rows),
        "estimated_bram_increases":sum(x["estimated_bram_saved"]<0 for x in qualified_rows),
        "shared_faster":sum(x["worst_cycles_saved"]>0 for x in qualified_rows),
        "shared_cycle_ties":sum(x["worst_cycles_saved"]==0 for x in qualified_rows),
        "shared_slower":sum(x["worst_cycles_saved"]<0 for x in qualified_rows),
        "physical_qualified_pairs":0,"new_vendor_calls":0,
        "evidence_ceiling":"SOURCE_ATTESTED_VITIS_HLS_ESTIMATES_NOT_PHYSICAL",
        "cases_and_pairs":result,
        "limitations":[
            "Secondary analysis; primary source and vendor reports must be archived separately",
            "Do not compare hardware cost across different graph or Phi tasks",
            "Source scratch array savings do not imply inferred FPGA BRAM savings",
            "HLS latency/clock and resources are estimates, not placed-and-routed",
            "Six pairs do not establish statistical generalization or scientific firstness",
        ],
    }


def export(*,input_audit:str|Path,output:str|Path):
    src,dest=Path(input_audit),Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Effects report output must be create-only")
    if src.is_symlink() or not src.is_file():
        raise ValueError("Missing ordinary audited original report")
    report=analyze(json.loads(src.read_text()))
    report["input_audit_sha256"]=sha256(src.read_bytes()).hexdigest()
    dest.mkdir(parents=True)
    (dest/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    columns=("pair","graph","mapping","evidence_status",
             "estimated_bram_saved","worst_cycles_saved",
             "worst_cycles_pct_improvement","pareto_cycles_bram",
             "pareto_cycles_bram_dsp_lut_ff")
    with (dest/"pairs.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(report["cases_and_pairs"])
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audit",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args(argv)
    print(json.dumps(export(input_audit=a.input_audit,output=a.output),
                     indent=2,sort_keys=True))


if __name__=="__main__":
    main()
