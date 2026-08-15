from pathlib import Path
from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.vhdl_integration import ExternalVHDLProjectRequest, emit_external_vhdl_operator_project


def test_vhdl_project_generates_behavioral_testbench_and_synthesis_flow(tmp_path):
    contract=implementation_contract_from_manifest(Path('examples/packages/scale_bias_vhdl'))
    result=emit_external_vhdl_operator_project(ExternalVHDLProjectRequest(out_dir=tmp_path,contract=contract))
    assert result.ok
    assert result.testbench and 'FPGAI_VHDL_SIM_PASS' in result.testbench.read_text()
    tcl=result.run_tcl.read_text()
    assert 'launch_simulation' in tcl
    assert 'synth_design' in tcl


def test_vhdl_simulation_uses_project_fileset_command_supported_by_vivado(tmp_path):
    contract=implementation_contract_from_manifest(Path('examples/packages/scale_bias_vhdl'))
    result=emit_external_vhdl_operator_project(ExternalVHDLProjectRequest(out_dir=tmp_path,contract=contract))
    assert result.ok
    tcl=result.run_tcl.read_text()
    assert 'read_vhdl -fileset' not in tcl
    assert 'add_files -fileset sim_1 -norecurse' in tcl
    assert 'update_compile_order -fileset sim_1' in tcl
