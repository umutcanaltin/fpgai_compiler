from pathlib import Path
from types import SimpleNamespace

import fpgai.implementations.vhdl_integration.integration as integration
from fpgai.implementations.vhdl_integration import ExternalVHDLProjectResult


def test_vhdl_runner_accepts_sim_pass_marker_from_xsim_log(tmp_path, monkeypatch):
    vhdl=tmp_path/'vhdl'; reports=tmp_path/'reports'; rtl=vhdl/'rtl'; vhdl.mkdir(); reports.mkdir(); rtl.mkdir()
    tcl=vhdl/'run_vivado.tcl'; tcl.write_text('exit\n')
    simlog=vhdl/'vivado_proj'/'x.sim'/'sim_1'/'behav'/'xsim'/'simulate.log'; simlog.parent.mkdir(parents=True); simlog.write_text('FPGAI_VHDL_SIM_PASS\n')
    (reports/'utilization_synth.rpt').write_text('ok')
    (reports/'timing_synth.rpt').write_text('ok')
    monkeypatch.setattr(integration.subprocess,'run',lambda *a,**k: SimpleNamespace(returncode=0,stdout='',stderr=''))
    result=integration.run_external_vhdl_project(ExternalVHDLProjectResult(True,rtl,None,tcl,None))
    assert result['status']=='passed'
    assert result['rtl_simulation_passed'] is True
    assert result['validation_level']=='vivado_synthesized'
    assert result['simulation_log'].endswith('simulate.log')
