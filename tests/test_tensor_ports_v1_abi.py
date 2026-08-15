from pathlib import Path
from types import SimpleNamespace

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.hls_integration import parse_tensor_ports_abi, validate_hls_integration_contract
from fpgai.implementations.hls_composition import build_hls_composition_plan, package_declarations


def test_tensor_ports_v1_manifest_parses_two_inputs_one_output():
    contract = implementation_contract_from_manifest(Path("examples/packages/add_tensor_ports_hls"))
    abi = parse_tensor_ports_abi(contract)
    assert [p.name for p in abi.inputs] == ["lhs", "rhs"]
    assert [p.name for p in abi.outputs] == ["output"]
    assert validate_hls_integration_contract(contract) == ()


def test_tensor_ports_v1_composition_binds_all_runtime_tensors():
    contract = implementation_contract_from_manifest(Path("examples/packages/add_tensor_ports_hls"))
    class G:
        name="g"; inputs=["lhs","rhs"]; outputs=["out"]; constants={}
        ops=[SimpleNamespace(name="add_ext",op_type="Add",inputs=["lhs","rhs"],outputs=["out"],attrs={"_fpgai_external_operator":{"operator_id":"fpgai.operator.add","package_id":"community.add_operator","package_version":"1.0.0"}})]
        def get_tensor(self,n): return SimpleNamespace(shape=(1,4))
    plan=build_hls_composition_plan(G(),selected_contracts={"add_ext":contract})
    b=plan.bindings[0]
    assert b.input_tensors == ("lhs","rhs")
    assert b.output_tensors == ("out",)
    assert b.port_words == 4
    decl="\n".join(package_declarations(plan))
    assert "const float* lhs" in decl and "const float* rhs" in decl and "float* output" in decl
