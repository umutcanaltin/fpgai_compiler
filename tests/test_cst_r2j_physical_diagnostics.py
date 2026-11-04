"""R2j saved routed results: negative E4 controls; zero vendor invocations."""
from __future__ import annotations
import json
import pytest

from fpgai_research.experiments.cst_r2j_physical_diagnostics import (
    diagnostic,main,SCHEMA,
)

def _study(tmp_path):
    root=tmp_path/"original_read_only"
    root.mkdir()
    cases=[]
    pairs=[]
    for pair in range(36):
        pnames=[f"case_{2*pair:03d}",f"case_{2*pair+1:03d}"]
        for j,storage in enumerate(("disjoint","per_element_optimal")):
            observed=pair==20
            cases.append({
                "case":pnames[j],"group":pair,
                "source_storage":storage,
                "graph":f"fanout_{pair}",
                "mapping":"all_pes2",
                "source_cpp_sha256":("a" if j==0 else "b")*64,
                "original_hls_qualified":True,
                "original_vivado_invocation_present":observed,
                "routed_observation_valid":observed,
                "routed_qualified":False,
                "routed_warning_observation":observed,
                "routed_timing_met":True if observed else None,
                "routed_wns_ns":(26.356 if j==0 else 27.304) if observed else None,
                "physical_resources":(
                    {"lut":3000-j*2,"ff":600-j,"dsp":4,
                     "bram":3-j,"uram":0} if observed else None),
                "drc_warning_summary":({
                    "summary_verified":True,"clean":False,"critical_warnings":0,
                    "unconstrained_io":False,"violations":1,
                    "rules":{"RTSTAT-10":{"severity":"Warning","violations":1}},
                } if observed else None),
                "drc_verified_clean":False if observed else None,
                "rom_input_evidence_mode":(
                    "ORIGINAL_SOURCE_AND_WORKING_ROM_STABLE" if observed else None),
                "requested_clock_ns":32.0,
            })
        pairs.append({
            "pair":pair,"cases":pnames,
            "both_routed_observation_valid":pair==20,
            "both_source_attributed_routed_timing_drc_qualified":False,
            "descriptive_bram_tile_difference":1 if pair==20 else None,
            "descriptive_lut_difference":2 if pair==20 else None,
            "descriptive_ff_difference":1 if pair==20 else None,
            "descriptive_dsp_difference":0 if pair==20 else None,
            "status":("OBSERVED_MATCHED_OOC_ROUTE_WITH_DISCLOSED_WARNINGS"
                      if pair==20 else "UNKNOWN_OR_REJECTED_PHYSICAL_PAIR"),
        })
    summary={
        "schema":"fpgai-research/cst-R2j-original-routed-fanout-pair-export/v1",
        "source_revision":"c"*40,
        "prespecified_cases":72,"prespecified_pairs":36,
        "original_vivado_invocation_receipts":2,
    }
    audit={"prespecified_cases":72,"cases":cases}
    (root/"summary.json").write_text(json.dumps(summary))
    (root/"original_physical_audit.json").write_text(json.dumps(audit))
    (root/"matched_pairs.json").write_text(json.dumps(pairs))
    return root

def test_physical_warned_observation_produces_only_descriptive_differences(tmp_path):
    root=_study(tmp_path)
    report=diagnostic(root,pair=20)
    assert report["schema"]==SCHEMA
    assert report["original_cases"]==["case_040","case_041"]
    assert report["both_source_attributed_routed_observations_valid"]
    assert report["both_strict_clean_DRC_E4_qualified"] is False
    assert report["descriptive_physical_resource_delta_disjoint_minus_optimized"][
        "bram"]==1
    assert report["descriptive_physical_resource_delta_disjoint_minus_optimized"][
        "lut"]==2
    assert report["descriptive_WNS_improvement_optimized_minus_disjoint_ns"]==pytest.approx(.948)
    assert report["qualified_physical_BRAM_savings"] is None
    assert report["qualified_physical_timing_advantage_ns"] is None
    assert report["new_original_GCC_Vitis_Vivado_calls"]==0
    assert report["cases"][0]["DRC_rule_details"]["RTSTAT-10"]["violations"]==1
    target=tmp_path/"new_prospective_readonly_diagnostic.json"
    main(["--export-study",str(root),"--pair","20","--output",str(target)])
    assert json.loads(target.read_text())==report
    with pytest.raises(FileExistsError,match="create-only"):
        main(["--export-study",str(root),"--pair","20","--output",str(target)])


def test_unexecuted_physical_pairs_remain_null_and_not_false_regression(tmp_path):
    report=diagnostic(_study(tmp_path),pair=19)
    assert report["both_source_attributed_routed_observations_valid"] is False
    assert report["descriptive_physical_resource_delta_disjoint_minus_optimized"][
        "bram"] is None
    assert report["descriptive_WNS_improvement_optimized_minus_disjoint_ns"] is None


def test_tampered_physical_source_or_false_E4_promotion_rejected(tmp_path):
    root=_study(tmp_path)
    p=root/"matched_pairs.json"
    rows=json.loads(p.read_text())
    rows[20]["descriptive_bram_tile_difference"]=3
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="differs from original matched"):
        diagnostic(root,pair=20)
    rows[20]["descriptive_bram_tile_difference"]=1
    rows[20]["both_source_attributed_routed_timing_drc_qualified"]=True
    rows[20]["status"]="MATCHED_OOC_ROUTED_RESULTS"
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="claimed pair qualification contradicts"):
        diagnostic(root,pair=20)


def test_incomplete_denominator_or_wrong_pair_is_rejected(tmp_path):
    root=_study(tmp_path)
    with pytest.raises(ValueError,match="prespecified pairs"):
        diagnostic(root,pair=36)
    path=root/"original_physical_audit.json"
    original=json.loads(path.read_text())
    original["cases"].pop()
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match="incomplete original cohort"):
        diagnostic(root,pair=20)


def test_vivado_fractional_BRAM_counts_are_legitimate_routed_observations(tmp_path):
    # The real Vivado utilization parser returns floats for half BRAM
    # tiles (e.g. 2.5), while LUT/FF may remain integer counts.
    root=_study(tmp_path)
    audit_path=root/"original_physical_audit.json"
    audit=json.loads(audit_path.read_text())
    audit["cases"][40]["physical_resources"]["bram"]=2.5
    audit["cases"][41]["physical_resources"]["bram"]=1.5
    # Optional URAM is not guaranteed to appear in every Vivado report.
    audit["cases"][40]["physical_resources"]["uram"]=None
    audit["cases"][41]["physical_resources"]["uram"]=None
    audit_path.write_text(json.dumps(audit))
    report=diagnostic(root,pair=20)
    delta=report["descriptive_physical_resource_delta_disjoint_minus_optimized"]
    assert report["both_source_attributed_routed_observations_valid"] is True
    assert report["both_strict_clean_DRC_E4_qualified"] is False
    assert report["cases"][0]["source_attributed_physical_resources"]["bram"]==2.5
    assert report["cases"][1]["source_attributed_physical_resources"]["bram"]==1.5
    assert delta["bram"]==1.0
    assert delta["uram"] is None
    assert report["qualified_physical_BRAM_savings"] is None
    assert report["qualified_physical_timing_advantage_ns"] is None
    assert report["new_original_GCC_Vitis_Vivado_calls"]==0


@pytest.mark.parametrize("bad_value",[None,-0.5,float("nan"),float("inf"),
                                      float("-inf"),True,"3.5"])
def test_invalid_or_missing_physical_resources_fail_closed(tmp_path,bad_value):
    root=_study(tmp_path)
    path=root/"original_physical_audit.json"
    audit=json.loads(path.read_text())
    audit["cases"][40]["physical_resources"]["bram"]=bad_value
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="lacks valid timing/resources"):
        diagnostic(root,pair=20)


@pytest.mark.parametrize("bad_wns",[None,float("nan"),float("inf"),True])
def test_invalid_routed_WNS_fails_closed(tmp_path,bad_wns):
    root=_study(tmp_path)
    path=root/"original_physical_audit.json"
    audit=json.loads(path.read_text())
    audit["cases"][41]["routed_wns_ns"]=bad_wns
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match="lacks valid timing/resources"):
        diagnostic(root,pair=20)
