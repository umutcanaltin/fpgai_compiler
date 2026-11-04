"""Source-only deployment inventory must never certify a board run."""
from __future__ import annotations

import json

import pytest

from scripts import check_fpga_deployment as deploy


def _case(tmp_path):
    root=tmp_path/"hls_case"
    root.mkdir()
    (root/"model.cpp").write_text("dummy")
    (root/"tb.cpp").write_text("dummy")
    (root/"run_hls.tcl").write_text("dummy")
    (root/"manifest.json").write_text(
        json.dumps({"part":"xck26-sfvc784-2LV-c","clock_ns":10.0}))
    (root/"vivado").mkdir()
    (root/"vivado"/"run_vivado.tcl").write_text("dummy")
    return root


@pytest.mark.parametrize("target", deploy.TARGETS)
def test_prepared_hls_source_does_not_claim_running_board(tmp_path,target):
    root=_case(tmp_path)
    report=deploy.inspect_case(root,target)
    assert report["target"]==target
    assert report["artifacts"]["generated_hls_cpp"]
    assert report["artifacts"]["numeric_testbench"]
    assert report["artifacts"]["vitis_hls_script"]
    assert report["artifacts"]["source_manifest"]
    assert not report["artifacts"]["vitis_synthesis_verilog"]
    assert not report["artifacts"]["vivado_ooc_timing_report"]
    assert report["source_manifest_part"]=="xck26-sfvc784-2LV-c"
    assert report["source_manifest_clock_ns"]==10.0
    assert report["complete_board_runtime_verified"] is False
    assert report["vendor_commands_executed"] is False
    assert report["hardware_programmed"] is False
    assert report["board_bitstream_generated_by_this_script"] is False
    assert report["out_of_context_evidence_is_not_board_qualification"] is True
    assert len(report["next_steps"])>=5


def test_even_fake_complete_ooc_reports_do_not_prove_board_runtime(tmp_path):
    root=_case(tmp_path)
    rtl=root/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    rtl.mkdir(parents=True)
    (rtl/"fpgai_model.v").write_text("module fpgai_model; endmodule")
    reports=root/"vivado"/"reports"
    reports.mkdir()
    for name in ("timing_summary.rpt","utilization.rpt","drc.rpt"):
        (reports/name).write_text("not qualified fake example")
    report=deploy.inspect_case(root,"pynq")
    assert report["missing_artifacts"]==[]
    assert report["complete_board_runtime_verified"] is False
    assert report["out_of_context_evidence_is_not_board_qualification"] is True


def test_invalid_target_and_manifest_fail_closed(tmp_path):
    root=_case(tmp_path)
    with pytest.raises(ValueError,match="Target"):
        deploy.inspect_case(root,"unknown")
    (root/"manifest.json").write_text("[]")
    with pytest.raises(ValueError,match="object"):
        deploy.inspect_case(root,"kv260")
