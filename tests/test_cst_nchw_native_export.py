"""Native paper export cannot invent HLS metrics or omit failed cases."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_nchw_native_pairs import prepare
from fpgai_research.experiments.cst_nchw_native_export import export_native_evidence


@pytest.mark.parametrize("binding",["auto","bram_2p"])
def test_original_paired_hls_paper_export_keeps_all_unknowns_and_no_native_calls(tmp_path,binding):
    root=tmp_path/"study"
    output=tmp_path/"original_report"
    prep=prepare(output=root,scratch_binding=binding)
    result=export_native_evidence(study=root,output=output)
    assert prep["case_count"]==12
    assert result["original_cases"]==12
    assert result["original_pairs"]==6
    assert result["original_invocations_with_receipts"]==0
    assert result["source_attested_csim_csynth_cases"]==0
    assert result["source_attested_csim_csynth_pairs"]==0
    assert result["numerical_CSim_qualified_cases"]==0
    assert result["hls_estimate_pair_comparisons"]==0
    assert result["failed_or_unknown_cases"]==12
    assert result["achieved_fpga_clock_ns"] is None
    assert result["physical_bram_savings"] is None
    assert result["new_vendor_invocations_for_report"]==0
    assert result==json.loads((output/"summary.json").read_text())
    with (output/"cases.csv").open(newline="") as stream:
        cases=list(csv.DictReader(stream))
    with (output/"pairs.csv").open(newline="") as stream:
        pairs=list(csv.DictReader(stream))
    assert len(cases)==12
    assert len(pairs)==6
    assert all(r["reported_hls_bram_18k"]=="" for r in cases)
    assert all(r["observed_numerical_and_hls_report"]=="False" for r in cases)
    assert all(r["qualification_status"]=="UNKNOWN_OR_REJECTED_HLS_PAIR"
               for r in pairs)
    with pytest.raises(FileExistsError):
        export_native_evidence(study=root,output=output)


def test_native_export_requires_original_unmodified_prepared_packages(tmp_path):
    study=tmp_path/"study"
    prepare(output=study)
    (study/"case_004"/"run_hls.tcl").write_text("open_project fake\n")
    with pytest.raises(ValueError,match="Tampered"):
        export_native_evidence(study=study,output=tmp_path/"report")
    assert not (tmp_path/"report").exists()
