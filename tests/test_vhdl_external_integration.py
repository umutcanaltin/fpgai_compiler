from pathlib import Path
from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.vhdl_integration import ExternalVHDLProjectRequest, emit_external_vhdl_operator_project, parse_vhdl_scalar_stream_abi


def test_vhdl_scalar_stream_contract_and_project_generation(tmp_path):
    contract=implementation_contract_from_manifest(Path("examples/packages/scale_bias_vhdl"))
    abi=parse_vhdl_scalar_stream_abi(contract)
    assert abi.data_width == 16
    result=emit_external_vhdl_operator_project(ExternalVHDLProjectRequest(out_dir=tmp_path,contract=contract))
    assert result.ok
    assert result.wrapper and result.wrapper.exists()
    assert result.run_tcl and result.run_tcl.exists()
    assert "scale_bias_vhdl" in result.wrapper.read_text()
    assert "synth_design" in result.run_tcl.read_text()
