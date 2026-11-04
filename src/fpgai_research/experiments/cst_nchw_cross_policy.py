"""Cross-policy original Vitis NCHW study: strict 2x2 source and HLS evidence.

Read-only source + native audits of two independent, create-only 12-case
studies: automatic inference and explicit AMD BRAM_2P binding. Different
compiler Git revisions are permitted ONLY when the actual C++ sources
are byte-identical after removing the sole BRAM pragma; all other inputs
must have identical hashes. A partly run policy is not filled with zero
BRAM/latency observations.

No new vendor calls, no post-route evidence and no claim of general
optimality. The four policies per graph are (auto/bram) x
(disjoint/shared scratch). All six graph/mapping groups remain included.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.codegen.hls.nchw_vendor_directives import PRAGMA
from fpgai_research.experiments.cst_nchw_native_execute import audit_native
from fpgai_research.experiments.cst_nchw_native_pairs import audit_prepared

SCHEMA="fpgai-research/CST-NCHW-two-by-two-original-Vitis-policy-audit/v1"
FIELDS=("worst_cycles","bram_18k","dsp","lut","ff")
POLICIES=("auto","bram_2p")
STORAGE=("real_disjoint","real_shared")


def _original_source_without_pragma(source:bytes,*,explicit:bool)->bytes:
    line=("  "+PRAGMA+"\n").encode()
    if source.count(line)!=(1 if explicit else 0):
        raise ValueError("Source does not have exactly the expected AMD BRAM pragma")
    if not explicit:
        return source
    return source.replace(line,b"",1)


def _metric(cell):
    if cell.get("observed_numerical_and_hls_report") is not True:
        return None
    resources=cell.get("estimated_hls_resources")
    if not isinstance(resources,dict):
        raise ValueError("Claimed qualified native HLS case has missing resources")
    result={"worst_cycles":cell.get("reported_hls_worst_cycles"),
            "bram_18k":cell.get("reported_hls_bram_18k"),
            "dsp":resources.get("dsp"),"lut":resources.get("lut"),
            "ff":resources.get("ff")}
    if any(type(v) is not int or v<0 or (k=="worst_cycles" and v==0)
           for k,v in result.items()):
        raise ValueError("Claimed qualified Vitis HLS source has invalid metrics")
    if result["bram_18k"]!=resources.get("bram_18k"):
        raise ValueError("Original HLS BRAM receipt mismatch")
    return result


def _dominates(a,b,keys):
    return all(a[k]<=b[k] for k in keys) and any(a[k]<b[k] for k in keys)


def _report(original_auto,original_bram,root_auto:Path,root_bram:Path):
    am,bm=original_auto["manifest"],original_bram["manifest"]
    common=("part","requested_clock_ns","explicit_clock_uncertainty_ns",
            "numerical_tolerance","case_count","matched_source_pairs")
    if any(am.get(k)!=bm.get(k) for k in common):
        raise ValueError("Unmatched FPGA part, numeric tolerance or native clock constraints")
    if (am.get("requested_native_memory_binding")!="auto"
            or bm.get("requested_native_memory_binding")!="bram_2p"
            or am.get("case_count")!=12 or bm.get("case_count")!=12):
        raise ValueError("Require one auto study and one bram_2p study with all 12 cases")
    if (len(original_auto["audit"]["cases_audited"])!=12 or
            len(original_bram["audit"]["cases_audited"])!=12):
        raise ValueError("Missing exact native audit denominator")
    outcomes=[]
    for group in range(6):
        cells=[]
        for j,layout in enumerate(STORAGE):
            index=2*group+j
            row_a,row_b=am["cases"][index],bm["cases"][index]
            audited_a=original_auto["audit"]["cases_audited"][index]
            audited_b=original_bram["audit"]["cases_audited"][index]
            comparable=("case","graph","mapping","group","model_sha256",
                        "architecture_sha256","input_sha256",
                        "testbench_sha256","tcl_sha256","real_scratch_elements",
                        "source_storage")
            if (any(row_a.get(k)!=row_b.get(k) for k in comparable)
                    or row_a["source_storage"]!=layout or
                    row_a["case"]!=f"case_{index:03d}" or
                    audited_a.get("case")!=row_a["case"] or
                    audited_b.get("case")!=row_b["case"]):
                raise ValueError("Native source policy pair changes model/Phi/inputs/storage")
            before=(root_auto/row_a["case"]/"model.cpp").read_bytes()
            after=(root_bram/row_b["case"]/"model.cpp").read_bytes()
            canonical_auto=_original_source_without_pragma(before,explicit=False)
            canonical_bram=_original_source_without_pragma(after,explicit=True)
            if canonical_auto!=canonical_bram:
                raise ValueError(
                    "UNMATCHED_SOURCE_BEYOND_PRAGMA: cross-revision C++ has other changes")
            if (sha256(before).hexdigest()!=row_a["source_cpp_sha256"] or
                    sha256(after).hexdigest()!=row_b["source_cpp_sha256"]):
                raise ValueError("Original source SHA disagrees with matched prepared receipt")
            cell_policy=[]
            for policy,reported in (("auto",audited_a),("bram_2p",audited_b)):
                qualified=(reported.get("observed_numerical_and_hls_report") is True and
                           reported.get("original_hls_effective_clock_feasible") is True and
                           reported.get("original_csim_qualified") is True and
                           not reported.get("warnings"))
                value=_metric(reported) if qualified else None
                if qualified and value is None:
                    raise ValueError("Claimed qualified native HLS case has no metrics")
                cell_policy.append({
                    "directive":policy,"storage":layout,
                    "case":row_a["case"],
                    "native_evidence_status":reported["native_qualification_status"],
                    "qualified_hls_estimates":bool(qualified),
                    "metrics":value,
                    "canonical_cpp_sha256":sha256(canonical_auto).hexdigest(),
                    "source_cpp_sha256":row_a["source_cpp_sha256"]
                        if policy=="auto" else row_b["source_cpp_sha256"],
                    "physical_qualified":False,
                })
            a,b=cell_policy
            both=a["qualified_hls_estimates"] and b["qualified_hls_estimates"]
            cells.extend(cell_policy)
            outcomes.append({
                "group":group,"graph":row_a["graph"],"mapping":row_a["mapping"],
                "storage":layout,"case":row_a["case"],
                "matched_source_except_directive":True,
                "different_source_revision":
                    am["source_revision"]!=bm["source_revision"],
                "auto_qualified":a["qualified_hls_estimates"],
                "bram_2p_qualified":b["qualified_hls_estimates"],
                "both_qualified":bool(both),
                "auto_minus_bram_hls_cycles":(
                    a["metrics"]["worst_cycles"]-b["metrics"]["worst_cycles"]
                    if both else None),
                "auto_minus_bram_hls_bram18k":(
                    a["metrics"]["bram_18k"]-b["metrics"]["bram_18k"]
                    if both else None),
                "cells":cell_policy,
                "physical_comparison_qualified":False,
            })
    four_way=[]
    for i in range(6):
        p=outcomes[2*i:2*i+2]
        candidates=[c for row in p for c in row["cells"]]
        fully=all(c["qualified_hls_estimates"] for c in candidates)
        pareto=[]
        if fully:
            for c in candidates:
                dims=("worst_cycles","bram_18k","dsp","lut","ff")
                dominating=[
                    f"{other['directive']}:{other['storage']}"
                    for other in candidates if other is not c and
                    _dominates(other["metrics"],c["metrics"],dims)]
                pareto.append({
                    "option":f"{c['directive']}:{c['storage']}",
                    "estimated_hls_pareto_nondominated":not dominating,
                    "dominated_by":dominating,
                })
        four_way.append({
            "group":i,"graph":p[0]["graph"],"mapping":p[0]["mapping"],
            "all_four_e3_qualified":fully,
            "available_options":sum(c["qualified_hls_estimates"] for c in candidates),
            "nondominated_full_hls_resource_objective":pareto if fully else None,
            "physical_pareto":None,
        })
    qualified=[p for p in outcomes if p["both_qualified"]]
    return {
        "schema":SCHEMA,
        "auto_source_revision":am["source_revision"],
        "bram_source_revision":bm["source_revision"],
        "different_revisions_canonical_source_checked":
            am["source_revision"]!=bm["source_revision"],
        "original_source_pairs_checked":12,
        "four_way_group_count":6,
        "native_policy_cells":24,
        "qualified_cross_policy_storage_pairs":len(qualified),
        "pending_or_failed_cross_policy_pairs":12-len(qualified),
        "auto_less_bram_estimate_count":sum(
            p["auto_minus_bram_hls_bram18k"]<0 for p in qualified),
        "bram_less_bram_estimate_count":sum(
            p["auto_minus_bram_hls_bram18k"]>0 for p in qualified),
        "equal_bram_estimate_count":sum(
            p["auto_minus_bram_hls_bram18k"]==0 for p in qualified),
        "source_comparisons":outcomes,
        "four_way_pareto":four_way,
        "physical_qualified_comparisons":0,
        "new_vendor_calls":0,
        "evidence_ceiling":"SOURCE_ATTESTED_ORIGINAL_VITIS_HLS_ESTIMATE_NOT_PLACED_ROUTE",
        "limitations":[
            "Independent studies must share all source bytes except the single requested BRAM pragma",
            "All twelve source comparisons preserved even when only one pair has Vitis evidence",
            "No implied default storage or directive choice without HLS/physical qualification",
            "Vendor HLS estimates are NOT achieved clock or physical FPGA resource counts",
        ],
    }


def cross_audit(auto_study:str|Path,bram_study:str|Path):
    auto,bram=Path(auto_study),Path(bram_study)
    if auto.resolve()==bram.resolve():
        raise ValueError("Require two genuinely distinct original source-only studies")
    states=[]
    for root,expected in ((auto,"auto"),(bram,"bram_2p")):
        attested=audit_prepared(root,require_current_source=False)
        manifest=json.loads((root/"native_source_manifest.json").read_text())
        if manifest.get("requested_native_memory_binding")!=expected:
            raise ValueError("Incorrect native policy study supplied")
        audit=audit_native(root,require_current_source=False)
        if (attested["source_revision"]!=manifest.get("source_revision")
                or audit["source_revision"]!=manifest.get("source_revision")):
            raise ValueError("Native study revision/source attestation mismatch")
        states.append({"manifest":manifest,"audit":audit})
    return _report(states[0],states[1],auto,bram)


def export_cross_audit(*,auto_study:str|Path,bram_study:str|Path,
                       output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Cross-policy Vitis evidence export is create-only")
    report=cross_audit(auto_study,bram_study)
    report["auto_prepared_manifest_sha256"]=sha256(
        (Path(auto_study)/"native_source_manifest.json").read_bytes()).hexdigest()
    report["bram_prepared_manifest_sha256"]=sha256(
        (Path(bram_study)/"native_source_manifest.json").read_bytes()).hexdigest()
    root.mkdir(parents=True)
    (root/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    with (root/"comparisons.csv").open("w",newline="") as stream:
        keys=("group","graph","mapping","storage","case",
              "matched_source_except_directive","auto_qualified",
              "bram_2p_qualified","both_qualified",
              "auto_minus_bram_hls_cycles","auto_minus_bram_hls_bram18k",
              "physical_comparison_qualified")
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction="ignore")
        writer.writeheader();writer.writerows(report["source_comparisons"])
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-study",required=True)
    parser.add_argument("--bram-study",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    result=export_cross_audit(auto_study=args.auto_study,
                              bram_study=args.bram_study,output=args.output)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
