"""Original Vivado OOC bridge is create-only, source-attested and zero-native by default.

Synthetic RTL files and HLS audits are test fixtures ONLY. Never call
Vitis/Vivado in CPU unit tests or treat fixtures as paper evidence.
"""
import json
from hashlib import sha256

import pytest

from fpgai_research.experiments.cst_nchw_native_pairs import prepare
from fpgai_research.experiments.cst_nchw_physical import (
    prepare_routed_study,audit_routed_study,run_routed_study,export_routed_evidence,
)


@pytest.fixture
def mock_native_with_original_rtl(tmp_path,monkeypatch):
    src=tmp_path/"hls_study"
    prepare(output=src,scratch_binding="auto")
    for i in range(2):
        case=src/f"case_{i:03d}"
        source=(case/"model.cpp").read_bytes()
        folder=case/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
        folder.mkdir(parents=True)
        rtl=b"module fpgai_model(input ap_clk); endmodule\n"
        (folder/"fpgai_model.v").write_bytes(rtl)
        receipt={
            "completed_successfully":True,
            "source_sha256":sha256(source).hexdigest(),
            "rtl_output_sha256":{"fpgai_model.v":sha256(rtl).hexdigest()},
            "inputs_stable_during_execution":True,
        }
        (case/"vitis_invocation.json").write_text(json.dumps(receipt))
    def fake_qualified(study,*,require_current_source=False):
        return {
            "cases":12,"matched_pairs":6,
            "cases_audited":[
                {"case":f"case_{i:03d}",
                 "observed_numerical_and_hls_report":i<2}
                for i in range(12)],
        }
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_physical.audit_native",
        fake_qualified)
    return src


def test_prepare_only_qualified_rtl_and_original_source_untouched(
    mock_native_with_original_rtl,tmp_path,monkeypatch
):
    src=mock_native_with_original_rtl
    dest=tmp_path/"physical"
    original=(src/"case_000"/"model.cpp").read_bytes()
    study=prepare_routed_study(native_study=src,output=dest)
    assert study["prespecified_cases"]==12
    assert study["rtl_packaged_cases"]==2
    assert study["not_yet_qualified_hls_cases"]==10
    assert study["original_vivado_calls"]==0
    assert study["routed_e4_qualified_cases"]==0
    assert (src/"case_000"/"model.cpp").read_bytes()==original
    assert (dest/"case_000"/"model.cpp").read_bytes()==original
    assert (dest/"case_000"/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/"fpgai_model.v").is_file()
    assert not (dest/"case_002").exists()
    assert "route_design" in (dest/"case_000"/"vivado"/"run_vivado.tcl").read_text()
    assert "out_of_context" in (dest/"case_000"/"vivado"/"run_vivado.tcl").read_text()
    audited=audit_routed_study(dest)
    assert audited["prespecified_cases"]==12
    assert audited["routed_qualified_cases"]==0
    assert audited["vivado_invocation_receipts"]==0
    assert audited["cases"][0]["status"]=="PREPARED_NOT_RUN"
    assert audited["cases"][2]["status"]=="BLOCKED_NO_QUALIFIED_ORIGINAL_HLS"
    assert run_routed_study(dest)["status"]=="DRY_RUN_NO_VIVADO"
    assert run_routed_study(dest)["new_vivado_calls"]==0
    with pytest.raises(FileExistsError):
        prepare_routed_study(native_study=src,output=dest)
    with pytest.raises(ValueError,match="confirm-isolated-native"):
        run_routed_study(dest,execute_native=True)


def test_rtl_hash_and_vivado_script_tamper_rejected(
    mock_native_with_original_rtl,tmp_path
):
    dest=tmp_path/"routed"
    prepare_routed_study(native_study=mock_native_with_original_rtl,output=dest)
    rtl=dest/"case_001"/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/"fpgai_model.v"
    rtl.write_text(rtl.read_text()+"// forgery")
    with pytest.raises(ValueError,match="tampered"):
        audit_routed_study(dest)


def test_native_receipt_forgery_is_not_physical_proof(
    mock_native_with_original_rtl,tmp_path
):
    dest=tmp_path/"routed"
    prepare_routed_study(native_study=mock_native_with_original_rtl,output=dest)
    receipt=dest/"case_000"/"vivado"/"vivado_invocation.json"
    receipt.write_text(json.dumps({
        "tool":"vivado","completed_successfully":True,
        "inputs_stable_during_execution":True,
        "source_sha256":"0"*64,"script_sha256":"0"*64,
        "rtl_input_sha256":{},
    }))
    audit=audit_routed_study(dest)
    assert audit["cases"][0]["status"]=="REJECTED_VENDOR_PROVENANCE"
    assert audit["cases"][0]["routed_timing_met"] is None
    assert audit["cases"][0]["routed_qualified"] is False
    assert audit["routed_qualified_cases"]==0
    with pytest.raises(FileExistsError,match="Refusing"):
        run_routed_study(dest,execute_native=True,confirm_isolated_native=True,
                         vivado_executable="sh")


def test_original_hls_rtl_forgery_rejected_during_preparation(
    mock_native_with_original_rtl,tmp_path
):
    source=mock_native_with_original_rtl/"case_000"/"vitis_invocation.json"
    receipt=json.loads(source.read_text())
    receipt["rtl_output_sha256"]={"fpgai_model.v":"f"*64}
    source.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="source attribution"):
        prepare_routed_study(native_study=mock_native_with_original_rtl,
                             output=tmp_path/"routed")
    assert not (tmp_path/"routed").exists()


def test_original_route_requires_independently_clean_drc(
    mock_native_with_original_rtl,tmp_path,monkeypatch
):
    dest=tmp_path/"physical"
    manifest=prepare_routed_study(
        native_study=mock_native_with_original_rtl,output=dest)
    first=manifest["cases"][0]
    case=dest/"case_000"
    receipt={
        "tool":"vivado","completed_successfully":True,
        "inputs_stable_during_execution":True,
        "source_sha256":first["source_cpp_sha256"],
        "script_sha256":first["vivado_tcl_sha256"],
        "rtl_input_sha256":first["copied_rtl_sha256"],
        "rom_input_sha256":{},
        "rom_inputs_stable_during_execution":True,
        "staged_rom_inputs_stable_during_execution":True,
        "staged_rom_input_integrity_before":{"matches_original_roms":True},
        "staged_rom_input_integrity_after":{"matches_original_roms":True},
        "fresh_report_sha256":{
            "route_status.rpt":"1"*64,
            "timing_summary.rpt":"2"*64,
            "utilization.rpt":"3"*64,
            "drc.rpt":"4"*64,
        },
    }
    (case/"vivado"/"vivado_invocation.json").write_text(
        json.dumps(receipt))
    physical={
        "routed_nets_complete":True,
        "timing":{"timing_met":True,"wns_ns":0.25},
        "utilization":{"lut":10,"ff":20,"dsp":0,"bram":0,"uram":0},
        "raw_report_sha256":{
            "route_status.rpt":"1"*64,
            "timing_summary.rpt":"2"*64,
            "utilization.rpt":"3"*64,
            "drc.rpt":"4"*64,
        },
        "drc":{"summary_verified":False,"clean":None,
               "critical_warnings":None},
    }
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_physical.parse_vivado_reports",
        lambda *_:physical)
    unqualified=audit_routed_study(dest)["cases"][0]
    assert unqualified["status"]=="UNKNOWN_OR_FAILED_VIVADO_DRC"
    assert unqualified["routed_wns_ns"]==0.25
    assert unqualified["routed_qualified"] is False
    physical["drc"]={"summary_verified":True,"clean":True,
                     "critical_warnings":0}
    qualified=audit_routed_study(dest)["cases"][0]
    assert qualified["status"]=="QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4"
    assert qualified["routed_qualified"] is True
    assert qualified["board_bitstream_qualified"] is False
    physical["timing"]={"timing_met":False,"wns_ns":-0.1}
    missed=audit_routed_study(dest)["cases"][0]
    assert missed["status"]=="FAILED_POST_ROUTE_TIMING"
    assert missed["routed_qualified"] is False
    physical["timing"]={"timing_met":True,"wns_ns":0.25}
    receipt["fresh_report_sha256"]["drc.rpt"]="9"*64
    (case/"vivado"/"vivado_invocation.json").write_text(json.dumps(receipt))
    stale=audit_routed_study(dest)["cases"][0]
    assert stale["status"]=="REJECTED_STALE_OR_UNATTRIBUTED_PHYSICAL_REPORTS"
    assert stale["routed_qualified"] is False


def test_original_physical_pair_export_preserves_full_denominator(
    mock_native_with_original_rtl,tmp_path
):
    study=tmp_path/"physical"
    prepare_routed_study(native_study=mock_native_with_original_rtl,
                         output=study)
    out=tmp_path/"paper"
    result=export_routed_evidence(study=study,output=out)
    assert result["prespecified_cases"]==12
    assert result["prespecified_pairs"]==6
    assert result["routed_qualified_cases"]==0
    assert result["routed_qualified_pairs"]==0
    assert result["unqualified_or_unattempted_cases"]==12
    assert result["new_native_calls_during_export"]==0
    assert json.loads((out/"summary.json").read_text())==result
    pairs=json.loads((out/"matched_pairs.json").read_text())
    assert len(pairs)==6
    assert all(x["physical_bram_tile_difference"] is None for x in pairs)
    assert all(x["status"]=="UNKNOWN_OR_REJECTED_PHYSICAL_PAIR"
               for x in pairs)
    with pytest.raises(FileExistsError):
        export_routed_evidence(study=study,output=out)


def _add_original_rom(native,*,filename="fpgai_weights_ROM_AUTO_1R.dat",
                      contents=b"0123abcd\n"):
    """Make a synthetic original HLS ROM and revise its fixture RTL receipt."""
    case=native/"case_000"
    rtl=case/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    original=rtl/"fpgai_model.v"
    original.write_text(
        "module fpgai_model(input ap_clk);\n"
        "  reg [31:0] weights [0:0];\n"
        f'  initial $readmemh("{filename}", weights);\n'
        "endmodule\n")
    (rtl/filename).write_bytes(contents)
    receipt_file=case/"vitis_invocation.json"
    receipt=json.loads(receipt_file.read_text())
    receipt["rtl_output_sha256"]["fpgai_model.v"]=sha256(
        original.read_bytes()).hexdigest()
    receipt_file.write_text(json.dumps(receipt))
    return case,rtl,filename


def test_original_hls_rom_dat_is_copied_in_both_locations_and_source_hashed(
    mock_native_with_original_rtl,tmp_path
):
    native=mock_native_with_original_rtl
    _,rtl,name=_add_original_rom(native)
    dest=tmp_path/"ooc"
    prepared=prepare_routed_study(native_study=native,output=dest)
    record=prepared["cases"][0]
    digest=sha256((rtl/name).read_bytes()).hexdigest()
    assert record["copied_rom_sha256"]=={name:digest}
    assert record["literal_readmem_references"]==[name]
    assert record["rom_attestation"]=="LEGACY_ROM_HASH_CAPTURED_DURING_PHYSICAL_PREPARATION"
    original=rtl/name
    staged_rtl=dest/"case_000"/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/name
    staged_cwd=dest/"case_000"/"vivado"/name
    assert staged_rtl.read_bytes()==staged_cwd.read_bytes()==original.read_bytes()
    tcl=(dest/"case_000"/"vivado"/"run_vivado.tcl").read_text()
    assert "add_files -fileset sources_1 -norecurse" in tcl
    assert "set_property file_type" not in tcl
    assert "Memory Initialization Files" not in tcl
    audit=audit_routed_study(dest)
    assert audit["cases"][0]["status"]=="PREPARED_NOT_RUN"
    assert audit["cases"][0]["original_rom_sha256"]=={name:digest}
    assert audit["routed_qualified_cases"]==0
    assert original.read_bytes()==b"0123abcd\n"


def test_rom_tamper_in_working_dir_fails_closed(
    mock_native_with_original_rtl,tmp_path
):
    native=mock_native_with_original_rtl
    _,_,name=_add_original_rom(native)
    dest=tmp_path/"ooc"
    prepare_routed_study(native_study=native,output=dest)
    (dest/"case_000"/"vivado"/name).write_bytes(b"ffffffff\n")
    result=audit_routed_study(dest)
    bad=result["cases"][0]
    assert bad["status"]=="REJECTED_ORIGINAL_STAGED_ROM_MODIFIED_OR_MISSING"
    assert bad["routed_qualified"] is False
    assert bad["staged_rom_integrity"]["modified_original_roms"]==[name]


def test_missing_and_path_escaped_rom_fail_before_package_creation(
    mock_native_with_original_rtl,tmp_path
):
    native=mock_native_with_original_rtl
    _,rtl,name=_add_original_rom(native)
    (rtl/name).unlink()
    with pytest.raises(ValueError,match="MISSING_ORIGINAL_HLS_ROM_INIT"):
        prepare_routed_study(native_study=native,output=tmp_path/"missing")
    assert not (tmp_path/"missing").exists()
    source=rtl/"fpgai_model.v"
    source.write_text(source.read_text().replace(name,"../bad.dat"))
    receipt=native/"case_000"/"vitis_invocation.json"
    manifest=json.loads(receipt.read_text())
    manifest["rtl_output_sha256"]["fpgai_model.v"]=sha256(
        source.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="UNSAFE_OR_UNSUPPORTED_RTL_READMEM_PATH"):
        prepare_routed_study(native_study=native,output=tmp_path/"escaped")
    assert not (tmp_path/"escaped").exists()


def test_unsupported_hls_ip_script_fails_before_study_creation(
    mock_native_with_original_rtl,tmp_path
):
    native=mock_native_with_original_rtl
    rtl=native/"case_000"/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    (rtl/"generate_floating_ip.tcl").write_text("puts unsupported\n")
    with pytest.raises(ValueError,match="UNSUPPORTED_HLS_RTL_IP_OR_MEMORY_ASSETS"):
        prepare_routed_study(native_study=native,output=tmp_path/"unsupported")
    assert not (tmp_path/"unsupported").exists()


def test_vendor_vivado_receipt_must_attest_original_rom_inputs(
    mock_native_with_original_rtl,tmp_path
):
    native=mock_native_with_original_rtl
    _,_,name=_add_original_rom(native)
    dest=tmp_path/"ooc"
    result=prepare_routed_study(native_study=native,output=dest)
    first=result["cases"][0]
    receipt_file=dest/"case_000"/"vivado"/"vivado_invocation.json"
    forged={
        "tool":"vivado","completed_successfully":True,
        "inputs_stable_during_execution":True,
        "source_sha256":first["source_cpp_sha256"],
        "script_sha256":first["vivado_tcl_sha256"],
        "rtl_input_sha256":first["copied_rtl_sha256"],
        # Historical vendor receipt formats omit ROM source hashing.
    }
    receipt_file.write_text(json.dumps(forged))
    audit=audit_routed_study(dest)
    assert audit["cases"][0]["status"]=="REJECTED_VENDOR_PROVENANCE"
    forged["rom_input_sha256"]=first["copied_rom_sha256"]
    forged["rom_inputs_stable_during_execution"]=True
    # The post-2026-11-02 source requires an independent hash attestation
    # of the actual Vivado working directory before AND after the tool.
    receipt_file.write_text(json.dumps(forged))
    audit=audit_routed_study(dest)
    assert audit["cases"][0]["status"]=="REJECTED_VENDOR_PROVENANCE"
    forged["staged_rom_inputs_stable_during_execution"]=True
    forged["staged_rom_input_integrity_before"]={"matches_original_roms":True}
    forged["staged_rom_input_integrity_after"]={"matches_original_roms":True}
    receipt_file.write_text(json.dumps(forged))
    audit=audit_routed_study(dest)
    assert audit["cases"][0]["status"]=="REJECTED_ROUTING_OR_REPORTS"
    assert audit["routed_qualified_cases"]==0


def test_vitis_single_dot_slash_readmemh_is_safe_and_source_attested(
    mock_native_with_original_rtl,tmp_path
):
    # Actual Vitis HLS 2023.2 can emit "./ROM_AUTO_1R.dat".
    # This is NOT a path escape; the original ROM bytes remain immutable.
    native=mock_native_with_original_rtl
    raw_name="./fpgai_weights_ROM_AUTO_1R.dat"
    _,rtl,_=_add_original_rom(native,filename=raw_name)
    name="fpgai_weights_ROM_AUTO_1R.dat"
    source=rtl/"fpgai_model.v"
    assert f'$readmemh("{raw_name}"' in source.read_text()
    physical=tmp_path/"physical"
    prepared=prepare_routed_study(native_study=native,output=physical)
    record=prepared["cases"][0]
    digest=sha256((rtl/name).read_bytes()).hexdigest()
    assert record["copied_rom_sha256"]=={name:digest}
    assert record["literal_readmem_references"]==[name]
    assert (physical/"case_000"/"vivado"/name).read_bytes()==(rtl/name).read_bytes()
    assert (physical/"case_000"/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/name).read_bytes()==(rtl/name).read_bytes()
    audited=audit_routed_study(physical)
    assert audited["cases"][0]["status"]=="PREPARED_NOT_RUN"
    assert audited["cases"][0]["original_rom_sha256"]=={name:digest}
    assert audited["routed_qualified_cases"]==0
    assert run_routed_study(physical)["new_vivado_calls"]==0


@pytest.mark.parametrize("unsafe",[
    "../weights.dat",
    "./../weights.dat",
    "././weights.dat",
    ".//weights.dat",
    "subdir/weights.dat",
    "/tmp/weights.dat",
    ".\\\\weights.dat",
    "weights.v",
])
def test_vitis_readmem_path_normalization_rejects_unsafe_inputs(
    mock_native_with_original_rtl,tmp_path,unsafe
):
    native=mock_native_with_original_rtl
    _,rtl,name=_add_original_rom(native)
    original=rtl/"fpgai_model.v"
    original.write_text(original.read_text().replace(name,unsafe))
    receipt_path=native/"case_000"/"vitis_invocation.json"
    receipt=json.loads(receipt_path.read_text())
    receipt["rtl_output_sha256"]["fpgai_model.v"]=sha256(
        original.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    destination=tmp_path/"unsafe_physical"
    with pytest.raises(ValueError,match="UNSAFE_OR_UNSUPPORTED_RTL_READMEM_PATH"):
        prepare_routed_study(native_study=native,output=destination)
    assert not destination.exists()


def _write_synthetic_failed_vivado_receipt(case):
    """No native tool execution: verify diagnostic with hash-attested fixture logs."""
    folder=case/"vivado"/"logs"
    folder.mkdir()
    stdout=folder/"vivado.stdout.log"
    stderr=folder/"vivado.stderr.log"
    stdout.write_text(
        "INFO: Opening project\n"
        "ERROR: [Common 17-53] User Tcl failed to read memory file\n"
        "INFO: Exiting Vivado\n")
    stderr.write_text("Vivado returned nonzero status\n")
    receipt={
        "tool":"vivado","returncode":1,"timed_out":False,
        "completed_successfully":False,
        "stdout_sha256":sha256(stdout.read_bytes()).hexdigest(),
        "stderr_sha256":sha256(stderr.read_bytes()).hexdigest(),
    }
    (case/"vivado"/"vivado_invocation.json").write_text(
        json.dumps(receipt))
    return stdout,stderr


def test_read_only_original_vivado_failure_diagnostics_are_hash_attested(
    mock_native_with_original_rtl,tmp_path
):
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import diagnose_failed_case
    study=tmp_path/"physical"
    prepare_routed_study(native_study=mock_native_with_original_rtl,output=study)
    case=study/"case_000"
    stdout,stderr=_write_synthetic_failed_vivado_receipt(case)
    receipt=case/"vivado"/"vivado_invocation.json"
    originals={path:path.read_bytes() for path in (stdout,stderr,receipt)}
    result=diagnose_failed_case(study,case_index=0)
    assert result["audit_status"]=="REJECTED_VENDOR_PROVENANCE"
    assert result["original_receipt_present"] is True
    assert result["returncode"]==1
    assert result["completed_successfully"] is False
    assert result["new_vendor_calls"]==0
    assert result["new_files_written"]==0
    assert result["logs"]["stdout"]["matches_original_receipt"] is True
    assert any("Common 17-53" in line
               for line in result["logs"]["stdout"]["error_lines"])
    assert result["logs"]["stderr"]["matches_original_receipt"] is True
    assert all(path.read_bytes()==data for path,data in originals.items())
    assert diagnose_failed_case(study,case_index=1)["original_receipt_present"] is False


def test_diagnostic_rejects_modified_logs_and_symlinks(
    mock_native_with_original_rtl,tmp_path
):
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import diagnose_failed_case
    study=tmp_path/"physical"
    prepare_routed_study(native_study=mock_native_with_original_rtl,output=study)
    case=study/"case_000"
    stdout,stderr=_write_synthetic_failed_vivado_receipt(case)
    stdout.write_text("ERROR: [Common 17-53] FORGED LOG\n")
    checked=diagnose_failed_case(study,case_index=0)
    assert checked["logs"]["stdout"]["matches_original_receipt"] is False
    assert checked["logs"]["stdout"]["error_lines"]==[]
    assert checked["logs"]["stdout"]["last_lines"]==[]
    assert checked["logs"]["stderr"]["matches_original_receipt"] is True
    stderr.unlink()
    stderr.symlink_to(stdout)
    with pytest.raises(ValueError,match="Symlinked Vivado evidence"):
        diagnose_failed_case(study,case_index=0)
    with pytest.raises(ValueError,match="case index"):
        diagnose_failed_case(study,case_index=12)


def test_vivado_dat_file_type_is_not_mutated(
    mock_native_with_original_rtl,tmp_path
):
    """Prevent original Vivado 2023.2 filemgmt 20-1707 regression.

    A source-only test verifies the exact Tcl that will be staged for the
    original Vitis ROM; this is not evidence of a successful native route.
    """
    native=mock_native_with_original_rtl
    _,rtl,rom=_add_original_rom(native)
    physical=tmp_path/"new_ooc"
    original_dat=(rtl/rom).read_bytes()
    manifest=prepare_routed_study(native_study=native,output=physical)
    tcl=(physical/"case_000"/"vivado"/"run_vivado.tcl").read_text()
    assert "add_files -fileset sources_1 -norecurse $rom" in tcl
    assert "get_files -quiet $rom" in tcl
    assert "add_files -fileset sources_1 -norecurse $local_rom" not in tcl
    assert "if {![file isfile $local_rom]}" in tcl
    assert "set_property file_type" not in tcl
    assert "Memory Initialization Files" not in tcl
    assert "synth_design -mode out_of_context" in tcl
    assert "ROM_READ_ATTESTATION_PROTOCOL_V1" in tcl
    assert manifest["cases"][0]["original_hls_qualified"] is True
    assert (physical/"case_000"/"vivado"/rom).read_bytes()==original_dat
    assert audit_routed_study(physical)["cases"][0]["status"]=="PREPARED_NOT_RUN"


def test_extra_vendor_rom_file_is_reported_without_erasing_real_receipt(
    mock_native_with_original_rtl,tmp_path
):
    """Regression for original Vivado v4: exit=0 then audit threw on ROM set."""
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import diagnose_failed_case
    native=mock_native_with_original_rtl
    _,rtl,name=_add_original_rom(native)
    dest=tmp_path/"v4_fixture"
    prepare_routed_study(native_study=native,output=dest)
    vivado=dest/"case_000"/"vivado"
    (vivado/"generated_by_vendor.mem").write_text("1010\n")
    (vivado/"vivado_invocation.json").write_text(json.dumps({
        "tool":"vivado","returncode":0,"completed_successfully":True,
    }))
    report=audit_routed_study(dest)
    first=report["cases"][0]
    assert first["original_vivado_invocation_present"] is True
    assert first["staged_rom_integrity"]["missing_original_roms"]==[]
    assert first["staged_rom_integrity"]["modified_original_roms"]==[]
    assert first["staged_rom_integrity"]["unattributed_rom_files"]==[
        "generated_by_vendor.mem"]
    assert first["status"]=="REJECTED_UNATTRIBUTED_VIVADO_ROM_ASSETS"
    assert first["routed_qualified"] is False
    assert report["vivado_invocation_receipts"]==1
    dry=run_routed_study(dest,execute_native=False,start_case=0,max_cases=1)
    assert dry["new_vivado_calls"]==0
    assert dry["cases"][0]["status"]==first["status"]
    diagnostic=diagnose_failed_case(dest,case_index=0)
    assert diagnostic["returncode"]==0
    assert diagnostic["staged_rom_integrity"]["unattributed_rom_files"]==[
        "generated_by_vendor.mem"]
    assert (rtl/name).is_file()


def test_missing_and_symlinked_staged_roms_fail_closed(
    mock_native_with_original_rtl,tmp_path
):
    native=mock_native_with_original_rtl
    _,_,name=_add_original_rom(native)
    dest=tmp_path/"physical"
    prepare_routed_study(native_study=native,output=dest)
    local=dest/"case_000"/"vivado"/name
    local.unlink()
    missing=audit_routed_study(dest)["cases"][0]
    assert missing["status"]=="REJECTED_ORIGINAL_STAGED_ROM_MODIFIED_OR_MISSING"
    assert missing["staged_rom_integrity"]["missing_original_roms"]==[name]
    local.symlink_to(dest/"case_000"/"model.cpp")
    invalid=audit_routed_study(dest)["cases"][0]
    assert invalid["routed_qualified"] is False
    assert invalid["staged_rom_integrity"]["invalid_rom_files"]==[name]


def test_staged_rom_audit_rejects_unattributed_asset_even_on_qualified_fixture(
    mock_native_with_original_rtl,tmp_path,monkeypatch
):
    native=mock_native_with_original_rtl
    _,_,name=_add_original_rom(native)
    dest=tmp_path/"physical"
    prepare_routed_study(native_study=native,output=dest)
    case=dest/"case_000"
    (case/"vivado"/"extra.dat").write_text("abcdef\n")
    # No fake route reports or mock E4 qualification may be reached here.
    def reject_parser(*args,**kwargs):
        raise AssertionError("Unattributed ROM must block E4 report parser")
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_physical.parse_vivado_reports",
        reject_parser)
    assert audit_routed_study(dest)["cases"][0]["routed_qualified"] is False


def test_mock_original_vivado_exit_zero_with_extra_data_keeps_auditable_result(
    mock_native_with_original_rtl,tmp_path
):
    """End-to-end regression for v4: never lose an exit-zero receipt on audit."""
    import stat
    native=mock_native_with_original_rtl
    _add_original_rom(native)
    study=tmp_path/"physical"
    prepare_routed_study(native_study=native,output=study)
    executable=tmp_path/"mock_vivado"
    executable.write_text(
        "#!/bin/sh\n"
        "echo 1010 > extra_vendor_output.mem\n"
        "exit 0\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    result=run_routed_study(
        study,execute_native=True,confirm_isolated_native=True,
        start_case=0,max_cases=1,vivado_executable=str(executable),
        timeout_seconds=20)
    assert result["new_vivado_calls"]==1
    assert result["cases"][0]["status"]=="REJECTED_UNATTRIBUTED_VIVADO_ROM_ASSETS"
    assert result["cases"][0]["routed_qualified"] is False
    audit=audit_routed_study(study)
    assert audit["vivado_invocation_receipts"]==1
    assert audit["routed_qualified_cases"]==0
    receipt=json.loads((study/"case_000"/"vivado"/"vivado_invocation.json").read_text())
    assert receipt["returncode"]==0
    assert receipt["completed_successfully"] is False
    assert receipt["staged_rom_inputs_stable_during_execution"] is False


def test_missing_working_rom_cannot_mask_tampered_source_provenance(
    mock_native_with_original_rtl,tmp_path
):
    """The original HLS case hash gate precedes vendor-side file rejection."""
    native=mock_native_with_original_rtl
    _,_,rom=_add_original_rom(native)
    study=tmp_path/"audit_source_before_working"
    prepare_routed_study(native_study=native,output=study)
    case=study/"case_000"
    (case/"vivado"/rom).unlink()
    (case/"model.cpp").write_text("void forged_source() {}\n")
    with pytest.raises(ValueError,match="immutable RTL/C\\+\\+/Vivado source tampered"):
        audit_routed_study(study)


def test_v4_missing_rom_diagnostic_retains_original_receipt(
    mock_native_with_original_rtl,tmp_path
):
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import diagnose_failed_case
    native=mock_native_with_original_rtl
    _,_,rom=_add_original_rom(native)
    study=tmp_path/"v4_original_fixture"
    prepare_routed_study(native_study=native,output=study)
    case=study/"case_000"
    (case/"vivado"/rom).unlink()
    receipt=case/"vivado"/"vivado_invocation.json"
    receipt.write_text(json.dumps({
        "tool":"vivado","returncode":0,"timed_out":False,
        "completed_successfully":True,
    }))
    original_bytes=receipt.read_bytes()
    diagnostic=diagnose_failed_case(study,case_index=0)
    assert diagnostic["returncode"]==0
    assert diagnostic["original_receipt_present"] is True
    assert diagnostic["audit_status"]=="REJECTED_ORIGINAL_STAGED_ROM_MODIFIED_OR_MISSING"
    assert diagnostic["staged_rom_integrity"]["missing_original_roms"]==[rom]
    assert diagnostic["new_vendor_calls"]==0
    assert diagnostic["new_files_written"]==0
    assert receipt.read_bytes()==original_bytes


def test_read_only_diagnostic_locates_original_rom_in_project_without_promotion(
    mock_native_with_original_rtl,tmp_path
):
    """A migrated ROM copy is diagnostic only, never a v5 E4 pass."""
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import diagnose_failed_case
    native=mock_native_with_original_rtl
    _,_,rom=_add_original_rom(native)
    study=tmp_path/"physical"
    prepare_routed_study(native_study=native,output=study)
    vendor=study/"case_000"/"vivado"
    project=vendor/"research_implementation"/"research_implementation.srcs"/"sources_1"
    project.mkdir(parents=True)
    original=(vendor/rom).read_bytes()
    (project/rom).write_bytes(original)
    (vendor/rom).unlink()
    (vendor/"vivado_invocation.json").write_text(json.dumps({
        "tool":"vivado","returncode":0,"timed_out":False,
        "completed_successfully":False,
    }))
    result=diagnose_failed_case(study,case_index=0)
    assert result["audit_status"]=="REJECTED_ORIGINAL_STAGED_ROM_MODIFIED_OR_MISSING"
    locations=result["project_rom_locations"]
    assert locations["project_present"] is True
    assert locations["scan_truncated"] is False
    assert len(locations["project_rom_candidates"])==1
    assert locations["project_rom_candidates"][0]["matches_original_sha256"]
    assert result["new_vendor_calls"]==0
    assert result["new_files_written"]==0
    assert not (vendor/rom).exists()


def test_read_only_diagnostic_reports_hash_matched_vendor_rom_activity(
    mock_native_with_original_rtl,tmp_path
):
    """Only original hash-matched stdout can reveal ROM activity lines."""
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import diagnose_failed_case
    study=tmp_path/"physical"
    prepare_routed_study(native_study=mock_native_with_original_rtl,output=study)
    case=study/"case_000"
    stdout,stderr=_write_synthetic_failed_vivado_receipt(case)
    stdout.write_text(
        "INFO: source imported weights.dat\n"
        "ERROR: [Common 17-53] source rejected\n")
    receipt=case/"vivado"/"vivado_invocation.json"
    original=json.loads(receipt.read_text())
    original["stdout_sha256"]=sha256(stdout.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(original))
    result=diagnose_failed_case(study,case_index=0)
    assert len(result["logs"]["stdout"]["rom_activity_lines"])==1
    assert "weights.dat" in result["logs"]["stdout"]["rom_activity_lines"][0]
    stdout.write_text("INFO: FORGED weights_ROM_AUTO_1R.dat\n")
    changed=diagnose_failed_case(study,case_index=0)
    assert changed["logs"]["stdout"]["matches_original_receipt"] is False
    assert changed["logs"]["stdout"]["rom_activity_lines"]==[]


def test_new_study_verified_original_readmem_postrun_absence_is_not_an_e4_pass(
    mock_native_with_original_rtl,tmp_path
):
    """Prospective vendor receipt can survive readmem cleanup; no fake E4."""
    import stat
    native=mock_native_with_original_rtl
    _,_,rom=_add_original_rom(native)
    study=tmp_path/"new_physical_study"
    manifest=prepare_routed_study(native_study=native,output=study)
    assert manifest["rom_postrun_policy"]=="ATTESTED_VENDOR_READ_POST_RUN_ABSENCE_V1"
    executable=tmp_path/"mock_vivado"
    executable.write_text(
        "#!/bin/sh\n"
        "cat <<'VENDOR_READ_SUCCESS'\n"
        f"INFO: [Synth 8-3876] $readmem data file './{rom}' is read successfully\n"
        "VENDOR_READ_SUCCESS\n"
        f"rm -- '{rom}'\n"
        "exit 0\n")
    executable.chmod(executable.stat().st_mode|stat.S_IXUSR)
    outcome=run_routed_study(
        study,execute_native=True,confirm_isolated_native=True,
        start_case=0,max_cases=1,vivado_executable=str(executable),
        timeout_seconds=30)
    assert outcome["new_vivado_calls"]==1
    assert outcome["cases"][0]["original_vendor_process_success"] is True
    assert outcome["cases"][0]["routed_qualified"] is False
    assert outcome["cases"][0]["status"]=="REJECTED_ROUTING_OR_REPORTS"
    case=study/"case_000"
    assert not (case/"vivado"/rom).exists()
    original=json.loads((case/"vivado"/"vivado_invocation.json").read_text())
    assert original["staged_rom_inputs_stable_during_execution"] is False
    assert original["verified_readmem_post_absence_witness"][
        "qualified_postrun_absence"] is True
    audit=audit_routed_study(study)
    assert audit["cases"][0]["verified_vendor_read_post_absence"] is True
    assert audit["routed_qualified_cases"]==0


def test_legacy_study_never_reinterprets_missing_rom_as_qualified(
    mock_native_with_original_rtl,tmp_path
):
    """Missing post-run ROMs in original v4/v5 stay strictly rejected."""
    import stat
    native=mock_native_with_original_rtl
    _,_,rom=_add_original_rom(native)
    study=tmp_path/"legacy_fixture"
    prepare_routed_study(native_study=native,output=study)
    manifest_path=study/"physical_study_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest.pop("rom_postrun_policy")
    manifest_path.write_text(json.dumps(manifest))
    executable=tmp_path/"mock_vivado"
    executable.write_text(
        "#!/bin/sh\n"
        "cat <<'VENDOR_READ_SUCCESS'\n"
        f"INFO: [Synth 8-3876] $readmem data file './{rom}' is read successfully\n"
        "VENDOR_READ_SUCCESS\n"
        f"rm -- '{rom}'\n"
        "exit 0\n")
    executable.chmod(executable.stat().st_mode|stat.S_IXUSR)
    result=run_routed_study(
        study,execute_native=True,confirm_isolated_native=True,
        start_case=0,max_cases=1,vivado_executable=str(executable),
        timeout_seconds=30)
    assert result["cases"][0]["status"]=="REJECTED_ORIGINAL_STAGED_ROM_MODIFIED_OR_MISSING"
    receipt=json.loads((study/"case_000"/"vivado"/"vivado_invocation.json").read_text())
    assert receipt["completed_successfully"] is False
    assert receipt["verified_readmem_post_absence_witness"] is None
    assert audit_routed_study(study)["routed_qualified_cases"]==0


def test_manifest_cannot_enable_new_readmem_policy_on_legacy_tcl(
    mock_native_with_original_rtl,tmp_path
):
    """The new policy must be source-marked; manifest-only retcon fails."""
    study=tmp_path/"old_source_fixture"
    prepare_routed_study(native_study=mock_native_with_original_rtl,output=study)
    case=study/"case_000"
    tcl=case/"vivado"/"run_vivado.tcl"
    old=tcl.read_text().replace("ROM_READ_ATTESTATION_PROTOCOL_V1",
                                "OLDER_TCL_NO_POST_ABSENCE_POLICY")
    tcl.write_text(old)
    manifest_path=study/"physical_study_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    entry=manifest["cases"][0]
    entry["vivado_tcl_sha256"]=sha256(old.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="source-Tcl protocol marker"):
        audit_routed_study(study)


def test_readonly_vivado_report_preview_requires_original_report_hashes(
    mock_native_with_original_rtl,tmp_path,monkeypatch
):
    """CPU fixture preview never rewrites or qualifies original vendor data."""
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import diagnose_failed_case
    study=tmp_path/"preview_fixture"
    prepare_routed_study(native_study=mock_native_with_original_rtl,output=study)
    vendor=study/"case_000"/"vivado"
    _write_synthetic_failed_vivado_receipt(study/"case_000")
    receipt_path=vendor/"vivado_invocation.json"
    receipt=json.loads(receipt_path.read_text())
    hashes={
        "route_status.rpt":"1"*64,"timing_summary.rpt":"2"*64,
        "utilization.rpt":"3"*64,"drc.rpt":"4"*64,
    }
    receipt["fresh_report_sha256"]=hashes
    receipt_path.write_text(json.dumps(receipt))
    original_bytes=receipt_path.read_bytes()
    fake_reports={
        "raw_report_sha256":dict(hashes),
        "routed_nets_complete":True,
        "timing":{"wns_ns":0.25,"timing_met":True},
        "utilization":{"lut":10,"ff":20,"dsp":1,"bram":0},
        "drc":{"summary_verified":True,"clean":True,"critical_warnings":0},
    }
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_vivado_diagnose.parse_vivado_reports",
        lambda *_:fake_reports)
    first=diagnose_failed_case(study,case_index=0)
    assert first["audit_status"]=="REJECTED_VENDOR_PROVENANCE"
    assert first["physical_report_preview"]["e4_qualified"] is False
    assert first["physical_report_preview"]["status"]==(
        "UNQUALIFIED_BUT_ORIGINAL_REPORT_HASHES_MATCH")
    assert first["physical_report_preview"]["timing"]["wns_ns"]==0.25
    fake_reports["raw_report_sha256"]["route_status.rpt"]="f"*64
    bad=diagnose_failed_case(study,case_index=0)
    assert bad["physical_report_preview"]=={
        "status":"UNQUALIFIED_REPORT_HASH_MISMATCH","e4_qualified":False}
    assert receipt_path.read_bytes()==original_bytes


def test_drc_rule_details_use_only_immutable_hash_matched_report(tmp_path):
    """Read-only diagnosis must show specific warnings, never elevate E4."""
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import (
        _inspect_original_drc_details,
    )
    root=tmp_path/"original"
    report=root/"case_000"/"vivado"/"reports"/"drc.rpt"
    report.parent.mkdir(parents=True)
    report.write_text(
        "Report DRC\n"
        "2. REPORT DETAILS\n"
        "DPIP-2#1 Warning\n"
        "Input pipelining\n"
        "DSP primitive input A is not pipelined\n"
        "DPOP-3#1 Warning\n"
        "PREG Output pipelining\n"
        "DSP primitive PREG=0\n"
        "RTSTAT-10#1 Warning\n"
        "No routable loads\n"
        "An original net has no routable loads\n")
    digest=sha256(report.read_bytes()).hexdigest()
    rules={"DPIP-2":{"severity":"Warning","violations":1},
           "DPOP-3":{"severity":"Warning","violations":1},
           "RTSTAT-10":{"severity":"Warning","violations":1}}
    good=_inspect_original_drc_details(
        report,root=root,expected_sha=digest,selected_rules=rules)
    assert good["matches_original_receipt"] is True
    assert "Input pipelining" in good["rule_examples"]["DPIP-2"][0]
    assert "PREG Output pipelining" in good["rule_examples"]["DPOP-3"][0]
    assert "No routable loads" in good["rule_examples"]["RTSTAT-10"][0]
    report.write_text(report.read_text()+"\nAFTER-RECEIPT-TAMPER\n")
    bad=_inspect_original_drc_details(
        report,root=root,expected_sha=digest,selected_rules=rules)
    assert bad["matches_original_receipt"] is False
    assert bad["rule_examples"]=={}


def test_drc_detailed_report_symlink_refused(tmp_path):
    from fpgai_research.experiments.cst_nchw_vivado_diagnose import (
        _inspect_original_drc_details,
    )
    source=tmp_path/"source.drc"
    source.write_text("DRC outside study")
    root=tmp_path/"case"
    root.mkdir()
    fake=root/"drc.rpt"
    fake.symlink_to(source)
    with pytest.raises(ValueError,match="Symlinked Vivado evidence"):
        _inspect_original_drc_details(fake,root=root,
                                      expected_sha="0"*64,selected_rules={})


def test_prospective_drc_warning_observations_never_become_strict_e4(
    mock_native_with_original_rtl,tmp_path,monkeypatch
):
    """Two source-paired synthetic cases: warning-only versus clean."""
    native=mock_native_with_original_rtl
    study=tmp_path/"prospective_warning_study"
    manifest=prepare_routed_study(native_study=native,output=study)
    assert manifest["drc_warning_observation_policy"]==(
        "DISCLOSED_OOC_NONCRITICAL_WARNING_OBSERVATION_V1")
    assert manifest["drc_warning_allowlist"]==[
        "DPIP-2","DPOP-3","DPOP-4","RTSTAT-10"]
    reports={
        "route_status.rpt":"1"*64,
        "timing_summary.rpt":"2"*64,
        "utilization.rpt":"3"*64,
        "drc.rpt":"4"*64,
    }
    for i in (0,1):
        e=manifest["cases"][i]
        receipt={
            "tool":"vivado","completed_successfully":True,
            "inputs_stable_during_execution":True,
            "source_sha256":e["source_cpp_sha256"],
            "script_sha256":e["vivado_tcl_sha256"],
            "rtl_input_sha256":e["copied_rtl_sha256"],
            "rom_input_sha256":e["copied_rom_sha256"],
            "rom_inputs_stable_during_execution":True,
            "staged_rom_inputs_stable_during_execution":True,
            "staged_rom_input_integrity_before":{"matches_original_roms":True},
            "staged_rom_input_integrity_after":{"matches_original_roms":True},
            "fresh_report_sha256":dict(reports),
        }
        (study/f"case_{i:03d}"/"vivado"/"vivado_invocation.json").write_text(
            json.dumps(receipt))
    def fake_vivado_report(path):
        case=str(path)
        shared="case_001" in case
        return {
            "raw_report_sha256":dict(reports),
            "routed_nets_complete":True,
            "timing":{"timing_met":True,"wns_ns":0.25 if shared else 0.50},
            "utilization":{"lut":110 if shared else 120,
                           "ff":180 if shared else 200,
                           "dsp":2,"bram":2.0 if shared else 2.5,"uram":0},
            "drc":(
                {"summary_verified":True,"clean":True,"critical_warnings":0,
                 "violations":0,"unconstrained_io":False,"rules":{}}
                if shared else
                {"summary_verified":True,"clean":False,"critical_warnings":0,
                 "violations":55,"unconstrained_io":False,
                 "rules":{
                     "DPIP-2":{"severity":"Warning","violations":16},
                     "DPOP-3":{"severity":"Warning","violations":19},
                     "DPOP-4":{"severity":"Warning","violations":19},
                     "RTSTAT-10":{"severity":"Warning","violations":1},
                 }}
            ),
        }
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_physical.parse_vivado_reports",
        fake_vivado_report)
    audit=audit_routed_study(study)
    warning,clean=audit["cases"][:2]
    assert warning["status"]=="OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS"
    assert warning["routed_warning_observation"] is True
    assert warning["routed_qualified"] is False
    assert warning["routed_observation_valid"] is True
    assert warning["drc_warning_summary"]["violations"]==55
    assert clean["status"]=="QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4"
    assert clean["routed_qualified"] is True
    assert clean["routed_warning_observation"] is False
    assert audit["routed_qualified_cases"]==1
    assert audit["routed_warning_observation_cases"]==1
    assert audit["routed_observation_valid_cases"]==2
    result=export_routed_evidence(study=study,output=tmp_path/"paper")
    assert result["routed_qualified_pairs"]==0
    assert result["routed_observation_valid_pairs"]==1
    assert result["routed_observation_pairs_containing_warnings"]==1
    assert result["unqualified_or_unattempted_cases"]==11
    pair=json.loads((tmp_path/"paper"/"matched_pairs.json").read_text())[0]
    assert pair["physical_bram_tile_difference"] is None
    assert pair["descriptive_bram_tile_difference"]==0.5
    assert pair["descriptive_lut_difference"]==10
    assert pair["status"]=="OBSERVED_MATCHED_OOC_ROUTE_WITH_DISCLOSED_WARNINGS"
    assert pair["case_drc_warning_rules"][0]==[
        "DPIP-2","DPOP-3","DPOP-4","RTSTAT-10"]
    assert (tmp_path/"paper"/"cases.csv").is_file()


@pytest.mark.parametrize("warning_change",[
    "unknown_rule","critical_warning","error_severity","constrained_io",
    "unverified_summary","mismatched_count","unsupported_severity",
])
def test_warning_observation_rejects_unknown_or_unsafe_drc(
    mock_native_with_original_rtl,tmp_path,monkeypatch,warning_change
):
    """A vendor warning count alone is never an E4 or observation pass."""
    native=mock_native_with_original_rtl
    study=tmp_path/"prospective_drc_negative"
    original=prepare_routed_study(native_study=native,output=study)
    e=original["cases"][0]
    hashes={k:c*64 for k,c in (
        ("route_status.rpt","1"),("timing_summary.rpt","2"),
        ("utilization.rpt","3"),("drc.rpt","4"))}
    (study/"case_000"/"vivado"/"vivado_invocation.json").write_text(
        json.dumps({
            "tool":"vivado","completed_successfully":True,
            "inputs_stable_during_execution":True,
            "source_sha256":e["source_cpp_sha256"],
            "script_sha256":e["vivado_tcl_sha256"],
            "rtl_input_sha256":e["copied_rtl_sha256"],
            "rom_input_sha256":e["copied_rom_sha256"],
            "rom_inputs_stable_during_execution":True,
            "staged_rom_inputs_stable_during_execution":True,
            "staged_rom_input_integrity_before":{"matches_original_roms":True},
            "staged_rom_input_integrity_after":{"matches_original_roms":True},
            "fresh_report_sha256":dict(hashes),
        }))
    drc={"summary_verified":True,"clean":False,
         "critical_warnings":0,"violations":1,
         "unconstrained_io":False,
         "rules":{"DPIP-2":{"severity":"Warning","violations":1}}}
    if warning_change=="unknown_rule":
        drc["rules"]={"UNKNOWN-1":{"severity":"Warning","violations":1}}
    elif warning_change=="critical_warning":
        drc["critical_warnings"]=1
    elif warning_change=="error_severity":
        drc["rules"]["DPIP-2"]["severity"]="Error"
    elif warning_change=="constrained_io":
        drc["unconstrained_io"]=True
    elif warning_change=="unverified_summary":
        drc["summary_verified"]=False
    elif warning_change=="mismatched_count":
        drc["violations"]=2
    elif warning_change=="unsupported_severity":
        drc["rules"]["DPIP-2"]["severity"]="Advisory"
    physical={
        "raw_report_sha256":dict(hashes),
        "timing":{"timing_met":True,"wns_ns":0.2},
        "utilization":{"lut":10,"ff":10,"dsp":1,"bram":1,"uram":0},
        "drc":drc,"routed_nets_complete":True,
    }
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_physical.parse_vivado_reports",
        lambda *_:physical)
    row=audit_routed_study(study)["cases"][0]
    assert row["routed_warning_observation"] is False
    assert row["routed_observation_valid"] is False
    assert row["routed_qualified"] is False
    assert row["status"]=="UNKNOWN_OR_FAILED_VIVADO_DRC"


def test_manifest_warning_allowlist_must_be_source_pinned(
    mock_native_with_original_rtl,tmp_path
):
    """Refuse even dry-run audit if the prespecified DRC policy changes."""
    study=tmp_path/"source_pinned_warning_policy"
    prepare_routed_study(native_study=mock_native_with_original_rtl,output=study)
    path=study/"physical_study_manifest.json"
    manifest=json.loads(path.read_text())
    manifest["drc_warning_allowlist"].append("UCIO-1")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="warning allowlist"):
        audit_routed_study(study)


def test_stale_drc_report_blocks_both_physical_evidence_tiers(
    mock_native_with_original_rtl,tmp_path,monkeypatch
):
    study=tmp_path/"bad_original_report"
    manifest=prepare_routed_study(native_study=mock_native_with_original_rtl,
                                  output=study)
    e=manifest["cases"][0]
    hashes={k:c*64 for k,c in (
        ("route_status.rpt","1"),("timing_summary.rpt","2"),
        ("utilization.rpt","3"),("drc.rpt","4"))}
    (study/"case_000"/"vivado"/"vivado_invocation.json").write_text(
        json.dumps({
            "tool":"vivado","completed_successfully":True,
            "inputs_stable_during_execution":True,
            "source_sha256":e["source_cpp_sha256"],
            "script_sha256":e["vivado_tcl_sha256"],
            "rtl_input_sha256":e["copied_rtl_sha256"],
            "rom_input_sha256":e["copied_rom_sha256"],
            "rom_inputs_stable_during_execution":True,
            "staged_rom_inputs_stable_during_execution":True,
            "staged_rom_input_integrity_before":{"matches_original_roms":True},
            "staged_rom_input_integrity_after":{"matches_original_roms":True},
            "fresh_report_sha256":dict(hashes),
        }))
    fake={
        "raw_report_sha256":dict(hashes),
        "timing":{"timing_met":True,"wns_ns":0.2},
        "utilization":{"lut":10,"ff":10,"dsp":1,"bram":1,"uram":0},
        "drc":{"summary_verified":True,"clean":False,
               "critical_warnings":0,"violations":1,"unconstrained_io":False,
               "rules":{"DPIP-2":{"severity":"Warning","violations":1}}},
        "routed_nets_complete":True,
    }
    fake["raw_report_sha256"]["drc.rpt"]="f"*64
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_physical.parse_vivado_reports",
        lambda *_:fake)
    row=audit_routed_study(study)["cases"][0]
    assert row["status"]=="REJECTED_STALE_OR_UNATTRIBUTED_PHYSICAL_REPORTS"
    assert row["routed_warning_observation"] is False
    assert row["routed_observation_valid"] is False


def test_unverified_route_never_counts_as_warning_observation(
    mock_native_with_original_rtl,tmp_path,monkeypatch
):
    study=tmp_path/"route_incomplete"
    manifest=prepare_routed_study(
        native_study=mock_native_with_original_rtl,output=study)
    e=manifest["cases"][0]
    hashes={k:c*64 for k,c in (
        ("route_status.rpt","1"),("timing_summary.rpt","2"),
        ("utilization.rpt","3"),("drc.rpt","4"))}
    (study/"case_000"/"vivado"/"vivado_invocation.json").write_text(
        json.dumps({
            "tool":"vivado","completed_successfully":True,
            "inputs_stable_during_execution":True,
            "source_sha256":e["source_cpp_sha256"],
            "script_sha256":e["vivado_tcl_sha256"],
            "rtl_input_sha256":e["copied_rtl_sha256"],
            "rom_input_sha256":e["copied_rom_sha256"],
            "rom_inputs_stable_during_execution":True,
            "staged_rom_inputs_stable_during_execution":True,
            "staged_rom_input_integrity_before":{"matches_original_roms":True},
            "staged_rom_input_integrity_after":{"matches_original_roms":True},
            "fresh_report_sha256":dict(hashes),
        }))
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_nchw_physical.parse_vivado_reports",
        lambda *_:{
            "raw_report_sha256":dict(hashes),
            "routed_nets_complete":False,
            "timing":{"timing_met":True,"wns_ns":0.9},
            "utilization":{"lut":10,"ff":10,"dsp":1,"bram":1,"uram":0},
            "drc":{"summary_verified":True,"clean":False,
                   "critical_warnings":0,"violations":1,
                   "unconstrained_io":False,
                   "rules":{"DPIP-2":{"severity":"Warning","violations":1}}},
        })
    row=audit_routed_study(study)["cases"][0]
    assert row["status"]=="REJECTED_INCOMPLETE_OR_UNVERIFIED_ROUTE"
    assert row["routed_qualified"] is False
    assert row["routed_warning_observation"] is False
