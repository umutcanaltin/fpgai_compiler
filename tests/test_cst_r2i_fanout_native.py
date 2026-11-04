"""R2i: 72 source-only Vitis packages and opt-in native safety controls.

No pytest test invokes a real vendor process. Native audit must classify
unexecuted cases UNKNOWN and preserve exact 36-pair study denominators.
"""
from __future__ import annotations

from hashlib import sha256
import json
import shutil

import pytest

from fpgai_research.experiments.cst_r2i_fanout_native_prepare import (
    TOTAL_PAIRS,prepare,audit_prepared,
)
from fpgai_research.experiments.cst_r2i_fanout_native_execute import (
    execute,audit_native,
)


@pytest.fixture(scope="module")
def source_only(tmp_path_factory):
    root=tmp_path_factory.mktemp("r2i_native_preparation")/"original_72_packages"
    meta=prepare(output=root)
    assert meta["source_cases"]==72
    assert meta["matched_pairs"]==36
    return root


def test_native_packages_preserve_matched_graph_Phi_TB_Tcl_and_source_sha(source_only):
    root=source_only
    report=audit_prepared(root)
    assert report["status"]=="ORIGINAL_NATIVE_SOURCE_ONLY_PREPARED"
    assert report["matched_pairs"]==TOTAL_PAIRS==36
    assert report["cases"]==72
    assert report["original_Vitis_HLS_calls_during_audit"]==0
    meta=json.loads((root/"native_source_manifest.json").read_text())
    assert meta["original_vitis_calls"]==0
    assert meta["physical_BRAM_or_Fmax_qualified_cases"]==0
    cases=meta["cases"]
    assert len(cases)==72
    for i in range(0,72,2):
        a,b=cases[i:i+2]
        assert (a["source_storage"],b["source_storage"])==(
            "disjoint","per_element_optimal")
        assert a["group"]==b["group"]==i//2
        assert a["graph"]==b["graph"]
        assert a["mapping"]==b["mapping"]
        assert a["functional_sha256"]==b["functional_sha256"]
        assert a["architecture_sha256"]==b["architecture_sha256"]
        assert a["numeric_input_sha256"]==b["numeric_input_sha256"]
        assert a["testbench_sha256"]==b["testbench_sha256"]
        assert a["tcl_sha256"]==b["tcl_sha256"]
        assert a["source_cpp_sha256"]!=b["source_cpp_sha256"]
        assert a["original_source_scratch_unit_slots"]>(
            b["original_source_scratch_unit_slots"])
        for entry in (a,b):
            case=root/entry["case"]
            assert (case/"model.cpp").is_file()
            assert (case/"tb.cpp").is_file()
            assert (case/"run_hls.tcl").is_file()
            assert (case/"manifest.json").is_file()
            assert sha256((case/"model.cpp").read_bytes()).hexdigest()==(
                entry["source_cpp_sha256"])
            assert "csim_design" in (case/"run_hls.tcl").read_text()
            assert "csynth_design" in (case/"run_hls.tcl").read_text()
            assert "xck26-sfvc784-2LV-c" in (case/"run_hls.tcl").read_text()
            assert not (case/"vitis_invocation.json").exists()
            assert not (case/"logs").exists()
            assert not (case/"fpgai_hls_project").exists()


def test_readonly_native_audit_reports_every_unexecuted_pair_unknown_not_zero(
        source_only):
    evidence=audit_native(source_only)
    assert evidence["prespecified_source_cases"]==72
    assert evidence["prespecified_matched_pairs"]==36
    assert evidence["original_vendor_invocation_receipts_present"]==0
    assert evidence["original_CSim_qualified_cases"]==0
    assert evidence["original_CSynth_timing_resource_qualified_cases"]==0
    assert evidence["original_CSynth_both_qualified_pairs"]==0
    assert evidence["original_HLS_estimated_BRAM_UNKNOWN"]==36
    assert evidence["original_HLS_estimated_BRAM_reductions"]==0
    assert evidence["original_HLS_estimated_BRAM_ties"]==0
    assert evidence["original_HLS_estimated_BRAM_regressions"]==0
    assert evidence["new_vendor_tool_calls_during_readonly_audit"]==0
    assert evidence["physical_BRAM_or_Fmax_established"] is False
    assert all(item["hls_estimated_resources"] is None for item
               in evidence["per_case_status"])
    assert all(pair["HLS_estimated_resource_delta_disjoint_minus_optimized"][
        "bram_18k"] is None for pair in evidence["all_original_matched_pairs"])


def test_native_foreground_executor_requires_explicit_double_confirmation(
        source_only):
    dry=execute(source_only,start_pair=35,max_pairs=1)
    assert dry["status"]=="DRY_RUN_ZERO_NATIVE_CALLS"
    assert dry["original_Vitis_calls"]==0
    assert dry["planned_exact_original_cases"]==["case_070","case_071"]
    with pytest.raises(ValueError,match="confirm-isolated-native"):
        execute(source_only,execute_native=True,
                confirm_isolated_native=False,start_pair=0,max_pairs=1)
    with pytest.raises(ValueError,match="pair range"):
        execute(source_only,start_pair=36,max_pairs=1)
    with pytest.raises(ValueError,match="pair range"):
        execute(source_only,start_pair=35,max_pairs=2)
    assert not (source_only/"case_000"/"vitis_invocation.json").exists()


def test_original_native_preparation_rejects_overwrite_and_invalid_clock(source_only,tmp_path):
    with pytest.raises(FileExistsError,match="unused"):
        prepare(output=source_only)
    with pytest.raises(ValueError,match="clock"):
        prepare(output=tmp_path/"invalid_clock",
                clock_ns=10.0,uncertainty_ns=12.0)
    assert not (tmp_path/"invalid_clock").exists()


def test_source_only_native_sha_tamper_is_rejected_without_any_vendor_calls(
        source_only,tmp_path):
    clone=tmp_path/"tampered_package"
    # Copy only one original case directory + study manifest via hard-links
    # would let tampering modify originals: always copy real independent bytes.
    shutil.copytree(source_only,clone)
    path=clone/"case_003"/"model.cpp"
    path.write_text(path.read_text()+"\n// mutation\n")
    with pytest.raises(ValueError,match="SHA differs"):
        audit_prepared(clone)
    assert not (clone/"case_003"/"vitis_invocation.json").exists()


def test_original_native_rejects_reordered_or_forged_graph_Phi_even_when_files_untouched(
        source_only,tmp_path):
    clone=tmp_path/"forged_graph_Phi_manifest"
    shutil.copytree(source_only,clone)
    doc=clone/"native_source_manifest.json"
    source=json.loads(doc.read_text())
    # File SHAs are all original, but the claimed graph policy
    # population is wrong. Do not accept internally consistent
    # source archives with a post-hoc altered cohort label.
    source["cases"][3]["mapping"]="all_pes2"
    doc.write_text(json.dumps(source,indent=2,sort_keys=True)+"\n")
    with pytest.raises(ValueError,match="fixed model/Phi"):
        audit_prepared(clone)
