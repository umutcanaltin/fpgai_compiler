"""Cross-revision original Vitis source-equivalence and 2x2 policy audit.

All CPU fixtures are source-only preparations. No Vitis is called.
Incomplete auto policy cannot be promoted to universal superiority.
"""
from copy import deepcopy
from hashlib import sha256
import csv
import json
import pytest

from fpgai_research.experiments.cst_nchw_native_pairs import prepare
from fpgai_research.experiments.cst_nchw_cross_policy import (
    cross_audit,export_cross_audit,_original_source_without_pragma,_report,
)


@pytest.fixture
def two_studies(tmp_path):
    auto=tmp_path/"native_auto"
    bram=tmp_path/"native_bram"
    prepare(output=auto,scratch_binding="auto")
    prepare(output=bram,scratch_binding="bram_2p")
    return auto,bram


def test_unrun_studies_full_2x2_match_and_no_invented_hardware(two_studies,tmp_path):
    auto,bram=two_studies
    result=cross_audit(auto,bram)
    assert result["four_way_group_count"]==6
    assert result["native_policy_cells"]==24
    assert result["original_source_pairs_checked"]==12
    assert result["qualified_cross_policy_storage_pairs"]==0
    assert result["pending_or_failed_cross_policy_pairs"]==12
    assert result["physical_qualified_comparisons"]==0
    assert result["new_vendor_calls"]==0
    assert all(x["matched_source_except_directive"] for x in result["source_comparisons"])
    assert all(x["auto_minus_bram_hls_bram18k"] is None
               for x in result["source_comparisons"])
    assert all(not x["all_four_e3_qualified"] for x in result["four_way_pareto"])
    report=export_cross_audit(auto_study=auto,bram_study=bram,
                              output=tmp_path/"audit")
    assert report["native_policy_cells"]==24
    with (tmp_path/"audit"/"comparisons.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream)))==12
    with pytest.raises(FileExistsError):
        export_cross_audit(auto_study=auto,bram_study=bram,
                           output=tmp_path/"audit")


def test_cross_policy_requires_only_pragma_change_not_another_generated_source(two_studies):
    auto,bram=two_studies
    a=(auto/"case_000"/"model.cpp").read_bytes()
    b=(bram/"case_000"/"model.cpp").read_bytes()
    assert a!=b
    assert _original_source_without_pragma(a,explicit=False)==(
        _original_source_without_pragma(b,explicit=True))
    with pytest.raises(ValueError,match="pragma"):
        _original_source_without_pragma(a,explicit=True)
    with pytest.raises(ValueError,match="pragma"):
        _original_source_without_pragma(b,explicit=False)
    # Read-only native source hash validation is independent of effect policy.
    bpath=bram/"case_000"/"model.cpp"
    bpath.write_bytes(bpath.read_bytes().replace(b"float",b"double",1))
    with pytest.raises(ValueError,match="Tampered"):
        cross_audit(auto,bram)


def test_matched_2x2_partial_vendor_results_never_fills_unrun_with_zeros(two_studies):
    auto,bram=two_studies
    a=json.loads((auto/"native_source_manifest.json").read_text())
    b=json.loads((bram/"native_source_manifest.json").read_text())
    empty={"cases_audited":[]}
    for manifest,scope in ((a,"auto"),(b,"bram_2p")):
        for row in manifest["cases"]:
            empty["cases_audited"].append(row["case"])
    def record(row,*,qualified,bram_value=0,cycles=975):
        return {
            "case":row["case"],"native_qualification_status":
                "QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL" if qualified
                else "UNKNOWN_NO_ORIGINAL_VITIS",
            "observed_numerical_and_hls_report":qualified,
            "original_csim_qualified":qualified,
            "original_hls_effective_clock_feasible":True if qualified else None,
            "estimated_hls_resources":{
                "bram_18k":bram_value,"dsp":29,"lut":14683,"ff":10857
            } if qualified else None,
            "reported_hls_bram_18k":bram_value if qualified else None,
            "reported_hls_worst_cycles":cycles if qualified else None,
            "warnings":[],
        }
    original_auto={"manifest":a,"audit":{"cases_audited":[
        record(row,qualified=i<2,bram_value=0)
        for i,row in enumerate(a["cases"])]}}
    original_bram={"manifest":b,"audit":{"cases_audited":[
        record(row,qualified=True,bram_value=5,cycles=1005 if i%2==0 else 975)
        for i,row in enumerate(b["cases"])]}}
    result=_report(original_auto,original_bram,auto,bram)
    assert result["qualified_cross_policy_storage_pairs"]==2
    assert result["pending_or_failed_cross_policy_pairs"]==10
    assert result["auto_less_bram_estimate_count"]==2
    assert result["four_way_pareto"][0]["all_four_e3_qualified"] is True
    assert all(not x["all_four_e3_qualified"]
               for x in result["four_way_pareto"][1:])
    assert all(x["auto_minus_bram_hls_bram18k"] is None
               for x in result["source_comparisons"][2:])
    # Previous source revision is allowed if canonical emitted C++ is equal.
    original_bram["manifest"]["source_revision"]="8814bd0005e8a852348b49f19131b2410117216b"
    different=_report(original_auto,original_bram,auto,bram)
    assert different["different_revisions_canonical_source_checked"] is True
