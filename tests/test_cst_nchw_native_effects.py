"""Paired HLS analysis must not assume memory reuse improves physical FPGA QoR.

Numbers are regression fixtures matching the user-reported 2026-11-02
12-case HLS audit. Test fixtures are NOT original vendor evidence.
"""
from copy import deepcopy
import csv
import json
import pytest

from fpgai_research.experiments.cst_nchw_native_effects import analyze,export


def fixture_audit():
    source="8814bd0005e8a852348b49f19131b2410117216b"
    # Original audit values transcribed as test fixtures only.
    groups=(
        ("branch_concat","scalar",1005,975,5,5,29,29,14683,14642,10857,10868),
        ("branch_concat","independent_conv_phi",588,586,64,64,112,112,43581,43423,7266,7138),
        ("pool_fork","scalar",757,758,9,9,33,33,14696,14605,9935,10053),
        ("pool_fork","independent_conv_phi",537,537,1,1,64,64,15380,15380,2203,2205),
        ("broadcast_fork","scalar",425,426,32,32,32,32,15437,15412,2878,3384),
        ("broadcast_fork","independent_conv_phi",1659,1633,7,12,55,55,24752,26120,20874,19969),
    )
    cases=[]
    pairs=[]
    for i,group in enumerate(groups):
        graph,mapping,bc,sc,bb,sb,bd,sd,bl,sl,bf,sf=group
        pair_names=[]
        for j in range(2):
            n=2*i+j
            case=f"case_{n:03d}"
            pair_names.append(case)
            cases.append({
                "case":case,"graph":graph,"mapping":mapping,
                "source_storage":"real_disjoint" if j==0 else "real_shared",
                "source_cpp_sha256":f"{n+1:064x}",
                "observed_numerical_and_hls_report":True,
                "original_hls_effective_clock_feasible":True,
                "original_hls_effective_budget_ns":23.36,
                "reported_hls_worst_cycles":bc if j==0 else sc,
                "reported_hls_bram_18k":bb if j==0 else sb,
                "estimated_hls_resources":{
                    "bram_18k":bb if j==0 else sb,
                    "dsp":bd if j==0 else sd,
                    "lut":bl if j==0 else sl,
                    "ff":bf if j==0 else sf,
                },"warnings":[],
            })
        pairs.append({
            "group":i,"cases":pair_names,
            "both_original_CSim_CSynth_attested":True,
            "qualified_HLS_estimate_only":True,
        })
    return {
        "schema":"fpgai-research/cst-NCHW-source-paired-native-Vitis-audit/v1",
        "source_revision":source,"cases":12,"matched_pairs":6,
        "physical_qualified_cases":0,
        "cases_audited":cases,"pairs_audited":pairs,
    }


def test_original_pair_regression_fixtures_do_not_imply_bram_reduction():
    result=analyze(fixture_audit())
    assert result["prespecified_cases"]==12
    assert result["qualified_pairs"]==6
    assert result["unqualified_pairs"]==0
    assert (result["estimated_bram_decreases"],result["estimated_bram_ties"],
            result["estimated_bram_increases"])==(0,5,1)
    assert (result["shared_faster"],result["shared_cycle_ties"],
            result["shared_slower"])==(3,1,2)
    assert [r["worst_cycles_saved"] for r in result["cases_and_pairs"]]==[
        30,2,-1,0,-1,26]
    assert [r["estimated_bram_saved"] for r in result["cases_and_pairs"]]==[
        0,0,0,0,0,-5]
    assert [r["pareto_cycles_bram"] for r in result["cases_and_pairs"]]==[
        "SHARED_DOMINATES","SHARED_DOMINATES","DISJOINT_DOMINATES",
        "TIE","DISJOINT_DOMINATES","TRADEOFF",
    ]
    assert result["cases_and_pairs"][0]["pareto_cycles_bram_dsp_lut_ff"]=="TRADEOFF"
    assert result["cases_and_pairs"][1]["pareto_cycles_bram_dsp_lut_ff"]=="SHARED_DOMINATES"
    assert result["physical_qualified_pairs"]==result["new_vendor_calls"]==0
    assert all(r["physical_bram_saved"] is None
               for r in result["cases_and_pairs"])


def test_missing_HLS_side_retained_and_not_treated_as_zero_difference():
    audit=fixture_audit()
    audit["cases_audited"][5]["observed_numerical_and_hls_report"]=False
    audit["pairs_audited"][2]["both_original_CSim_CSynth_attested"]=False
    audit["pairs_audited"][2]["qualified_HLS_estimate_only"]=False
    report=analyze(audit)
    assert report["qualified_pairs"]==5
    assert report["unqualified_pairs"]==1
    assert report["cases_and_pairs"][2]["estimated_bram_saved"] is None
    assert report["cases_and_pairs"][2]["worst_cycles_saved"] is None
    assert report["cases_and_pairs"][2]["pareto_cycles_bram"]=="UNKNOWN"


def test_invalid_pair_geometry_or_clock_never_promoted():
    audit=fixture_audit()
    audit["cases_audited"][1]["mapping"]="wrong_graph"
    with pytest.raises(ValueError,match="Unmatched"):
        analyze(audit)
    audit=fixture_audit()
    audit["cases_audited"][1]["original_hls_effective_budget_ns"]=21.0
    with pytest.raises(ValueError,match="clock"):
        analyze(audit)
    audit=fixture_audit()
    audit["cases_audited"][0]["estimated_hls_resources"]["bram_18k"]=0
    with pytest.raises(ValueError,match="Inconsistent"):
        analyze(audit)


def test_effects_export_is_create_only_and_honest(tmp_path):
    inp=tmp_path/"full_native_audit.json"
    inp.write_text(json.dumps(fixture_audit()))
    folder=tmp_path/"effects"
    report=export(input_audit=inp,output=folder)
    assert (folder/"summary.json").is_file()
    assert report["input_audit_sha256"]
    with (folder/"pairs.csv").open(newline="") as fd:
        rows=list(csv.DictReader(fd))
    assert len(rows)==6
    assert rows[5]["estimated_bram_saved"]=="-5"
    assert rows[3]["worst_cycles_saved"]=="0"
    assert report["evidence_ceiling"]=="SOURCE_ATTESTED_VITIS_HLS_ESTIMATES_NOT_PHYSICAL"
    with pytest.raises(FileExistsError):
        export(input_audit=inp,output=folder)
