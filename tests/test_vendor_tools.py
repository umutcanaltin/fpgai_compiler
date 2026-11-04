"""Synthetic vendor fixtures verify failure modes; never paper physical measurements."""
from hashlib import sha256
from pathlib import Path
import json
import stat
import pytest

from fpgai_research.toolchains.vitis_hls.runner import run_vitis
from fpgai_research.toolchains.vivado.project import make_vivado_tcl, write_vivado_package
from fpgai_research.toolchains.vivado.report import parse_vivado_reports
from fpgai_research.toolchains.vivado.runner import run_vivado
from fpgai_research.experiments.results import summarize_case

def _case(tmp_path):
    path=tmp_path/"model.cpp"
    path.write_text("void placeholder() {}")
    (tmp_path/"manifest.json").write_text(json.dumps(
        {"cpp_sha256":sha256(path.read_bytes()).hexdigest()}))
    (tmp_path/"run_hls.tcl").write_text("exit\n")
    return tmp_path

def _fake(path, script):
    path.write_text("#!/bin/sh\n"+script+"\n")
    path.chmod(path.stat().st_mode|stat.S_IXUSR)
    return str(path)

def test_vivado_tcl_is_model_only():
    script=make_vivado_tcl()
    assert "route_design" in script
    assert "report_timing_summary" in script
    assert "read_verilog" in script
    assert "add_files -fileset sources_1 -norecurse $rom" in script
    assert "add_files -fileset sources_1 -norecurse $local_rom" not in script
    assert "if {![file isfile $local_rom]}" in script
    assert "set_property file_type" not in script
    assert "write_bitstream" not in script
    with pytest.raises(ValueError):
        make_vivado_tcl(part="invalid;evil")
    with pytest.raises(ValueError):
        make_vivado_tcl(clock_ns=float("nan"))

def test_missing_vendor_executable_cannot_claim_evidence(tmp_path):
    case=_case(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_vitis(case,executable="missing_vitis_hls_executable_830118")
    assert summarize_case(str(case))["evidence"]=="E1-HLS-source"

def test_opt_in_mock_vitis_execution(tmp_path):
    case=_case(tmp_path)
    fake=_fake(tmp_path/"mock_vitis",
        'echo "CSim done with 0 errors"; echo "FPGAI_METRICS samples=1 outputs=1 cosine=1 rmse=0 maxabs=0 failures=0"')
    receipt=run_vitis(case,executable=fake)
    assert receipt["completed_successfully"]
    assert receipt["evidence_level"] is None
    result=summarize_case(str(case))
    # A mock vendor invocation without an attested testbench cannot establish E2.
    assert result["evidence"]=="E1-HLS-source"
    assert result["metrics"]["lut"] is None
    (case/"run_hls.tcl").write_text("exit\n# changed")
    assert summarize_case(str(case))["evidence"]=="E1-HLS-source"

def test_missing_simulation_metric_marker_does_not_claim_e2(tmp_path):
    case=_case(tmp_path)
    fake=_fake(tmp_path/"mock_vitis",'echo "CSim done with 0 errors"')
    run_vitis(case,executable=fake)
    assert summarize_case(str(case))["evidence"]=="E1-HLS-source"

def test_vivado_report_parser_checks_route_and_timing(tmp_path):
    # A lone "unrouted=0" is not enough to establish E4. Include the
    # complete Vivado route inventory and Design Timing Summary instead.
    report=tmp_path/"reports"
    report.mkdir()
    complete_route = """Design Route Status
    # of routable nets...................... :      1,300 :
    # of fully routed nets.................. :      1,300 :
    # of nets with routing errors.......... :          0 :
"""
    (report/"route_status.rpt").write_text(complete_route)
    (report/"timing_summary.rpt").write_text("""Design Timing Summary
WNS(ns)  TNS(ns)  TNS Failing Endpoints
-------  -------  ---------------------
 0.125     0.000                      0
Clock Summary
""")
    (report/"utilization.rpt").write_text(
      "| CLB LUTs | 1,300 |  |\\n| CLB Registers | 220 |  |\\n| DSPs | 3 |  |\\n".replace("\\n", "\n"))
    summary=parse_vivado_reports(tmp_path)
    assert summary["routed_nets_complete"]
    assert summary["utilization"]["lut"]==1300
    assert summary["timing"]["wns_ns"]==pytest.approx(.125)
    assert summary["timing"]["timing_met"] is True

    # Failed routing is never promoted, even when the tool writes reports.
    (report/"route_status.rpt").write_text(complete_route.replace(
        "# of nets with routing errors.......... :          0 :",
        "# of nets with routing errors.......... :          2 :"))
    with pytest.raises(ValueError,match="routing errors"):
        parse_vivado_reports(tmp_path)

    # The old summary-only fixture must be rejected as unqualified evidence.
    (report/"route_status.rpt").write_text("Number of unrouted nets: 0\\n".replace("\\n", "\n"))
    with pytest.raises(ValueError,match="Cannot confirm complete routing"):
        parse_vivado_reports(tmp_path)

def test_fake_vivado_does_not_create_missing_synthesis_proof(tmp_path):
    case=_case(tmp_path)
    write_vivado_package(case)
    fake=_fake(tmp_path/"mock_vivado",'echo "Vivado finished"')
    run_vivado(case,executable=fake)
    assert summarize_case(str(case))["evidence"]=="E1-HLS-source"

def test_testbench_tampering_blocks_evidence(tmp_path):
    case=_case(tmp_path)
    (case/"tb.cpp").write_text("original")
    manifest=json.loads((case/"manifest.json").read_text())
    manifest["testbench_sha256"]=sha256((case/"tb.cpp").read_bytes()).hexdigest()
    (case/"manifest.json").write_text(json.dumps(manifest))
    (case/"tb.cpp").write_text("changed")
    with pytest.raises(ValueError,match="hash"):
        summarize_case(str(case))


def test_old_synthesis_xml_cannot_gain_new_e3_from_noop_command(tmp_path):
    case=_case(tmp_path)
    (case/"tb.cpp").write_text("// testbench\n")
    xml=case/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text("<Profile><LUT>1234</LUT><FF>1</FF></Profile>")
    fake=_fake(tmp_path/"mock_vitis","echo nothing_synthesized")
    run=run_vitis(case,executable=fake)
    assert "csynth_xml" not in run["fresh_report_sha256"]
    outcome=summarize_case(str(case))
    assert outcome["evidence"]=="E1-HLS-source"
    assert outcome["metrics"]["lut"] is None
    assert any("fresh" in warning for warning in outcome["warnings"])


def test_changing_bench_after_run_invalidates_invocation(tmp_path):
    case=_case(tmp_path)
    (case/"tb.cpp").write_text("original")
    fake=_fake(tmp_path/"mock_vitis",
        'echo "CSim done with 0 errors"; echo "FPGAI_METRICS samples=1 outputs=1 cosine=1 rmse=0 maxabs=0 failures=0"')
    run_vitis(case,executable=fake)
    (case/"tb.cpp").write_text("modified")
    outcome=summarize_case(str(case))
    assert outcome["evidence"]=="E1-HLS-source"


def test_mock_vivado_extra_staged_rom_does_not_count_as_vendor_success(tmp_path):
    # A clean process exit is insufficient if a new unaccounted .mem file
    # appears in Vivado cwd. This is a mock shell, never actual FPGA evidence.
    case=_case(tmp_path)
    write_vivado_package(case)
    fake=_fake(tmp_path/"mock_rom_vivado",
        'printf "beef\\\\n" > unexplained_generated.mem; exit 0')
    receipt=run_vivado(case,executable=fake)
    assert receipt["returncode"]==0
    assert receipt["completed_successfully"] is False
    assert receipt["staged_rom_inputs_stable_during_execution"] is False
    assert receipt["staged_rom_input_integrity_before"]["matches_original_roms"]
    assert receipt["staged_rom_input_integrity_after"]["unattributed_rom_files"]==[
        "unexplained_generated.mem"]


def test_mock_vivado_matching_local_roms_stay_source_attested(tmp_path):
    from fpgai_research.toolchains.vivado.rtl_inputs import compare_staged_roms
    case=_case(tmp_path)
    root=case/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    root.mkdir(parents=True)
    source=root/"weights.dat"
    source.write_text("feedcafe\\n")
    write_vivado_package(case)
    local=case/"vivado"/"weights.dat"
    local.write_bytes(source.read_bytes())
    fake=_fake(tmp_path/"mock_stable_rom",'echo initialized; exit 0')
    receipt=run_vivado(case,executable=fake)
    assert receipt["returncode"]==0
    assert receipt["completed_successfully"] is True
    assert receipt["rom_input_sha256"]=={
        "weights.dat":sha256(source.read_bytes()).hexdigest()}
    assert receipt["staged_rom_inputs_stable_during_execution"] is True
    assert compare_staged_roms(case/"vivado",receipt["rom_input_sha256"])[
        "matches_original_roms"] is True


def test_original_vivado_reports_reject_symlinked_drc(tmp_path):
    """Even a readable DRC symlink cannot create original E4 evidence."""
    study=tmp_path/"original_physical"
    report=study/"reports"
    report.mkdir(parents=True)
    (report/"route_status.rpt").write_text(
        "Design Route Status\n"
        "# of routable nets...................... : 12 :\n"
        "# of fully routed nets.................. : 12 :\n"
        "# of nets with routing errors.......... : 0 :\n")
    (report/"timing_summary.rpt").write_text(
        "Design Timing Summary\n"
        "WNS(ns)  TNS(ns)\n-------  -------\n0.5 0.0\n")
    (report/"utilization.rpt").write_text("| CLB LUTs | 10 |\n")
    outside=tmp_path/"unattributed_drc.rpt"
    outside.write_text("Violations found: 0\n")
    (report/"drc.rpt").symlink_to(outside)
    with pytest.raises(ValueError,match="Symlinked original Vivado physical report"):
        parse_vivado_reports(study)
